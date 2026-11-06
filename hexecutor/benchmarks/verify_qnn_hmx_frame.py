"""Gate and time the aligned SDK leaf on Frame against saved ISA fixtures.

Does not load cdsp_mem or change installed executors. Upper placements require
the preservation probe and restore every byte of their 512 KB working range.
Fixture outputs are simulator-native, which may differ from the exact reference.
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport
from verify_resident_decoder import snapshot
MAGIC=0x314e5151

def fixtures(path):
    raw=path.read_bytes();at=0;rows=[]
    while at<len(raw):
        h,w,ci,co,random=struct.unpack_from('<5I',raw,at);at+=20
        if h not in (8,16) or w not in (8,24) or ci not in (32,64,96,128) or co not in (32,64,96,128) or random not in (0,1):
            raise ValueError('invalid fixture header')
        sizes=(h*w*ci,ci*co,co*8,h*w*co)
        parts=[]
        for n in sizes:
            if at+n>len(raw):raise ValueError('incomplete fixture')
            parts.append(raw[at:at+n]);at+=n
        rows.append(((h,w,ci,co,random),parts))
    if len(rows)!=128 or len({r[0] for r in rows})!=128:raise ValueError('expected complete 128-case ISA gate')
    return rows

def dense(raw,h,w,c):
    return np.frombuffer(raw,'u1').reshape(h//8,w//8,c//32,8,8,32).transpose(0,3,1,4,2,5).reshape(h,w,c).astype(np.int64)

def check_integer_reference(header,parts):
    h,w,ci,co,random=header
    a=dense(parts[0],h,w,ci)
    weights=np.frombuffer(parts[1],'i1').reshape(co//32,ci//32,8,32,4).transpose(1,2,4,0,3).reshape(ci,co).astype(np.int64)
    fi=np.frombuffer(parts[2],'<i4').reshape(co//32,64).astype(np.int64)
    acc=a@weights
    if random:
        np.testing.assert_array_equal(fi[:,:32],0x404200)
        acc+=fi[:,32:].reshape(co)
        want=np.clip((((acc>>4)*1536+8192)>>14),0,255)
    else:
        np.testing.assert_array_equal(fi[:,:32],0x406000)
        want=acc
    delta=dense(parts[3],h,w,co)-want
    if delta.min()<0 or delta.max()>(1 if random else 0):raise ValueError('fixture fails independent integer reference')
    return int(np.count_nonzero(delta))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixtures',type=Path,default=(PATHS.native / 'build/qnn-hmx-1x1-output.bin'))
    ap.add_argument('--limit',type=int)
    ap.add_argument('--iterations',type=int,action='append',help='repeat leaf calls inside each DSP request; default 1 and 100')
    ap.add_argument('--drain-each',action='store_true',help='poison output, drain and verify every call; exclude poison/check time from kernel timer')
    ap.add_argument('--vtcm-offset',type=lambda s:int(s,0),choices=(0,0x300000,0x700000),default=0)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();repeats=args.iterations or [1,100]
    if any(not 1<=n<=10000 for n in repeats):ap.error('iterations must be 1..10000')
    todo=fixtures(args.fixtures)
    if args.limit is not None:
        if not 1<=args.limit<=len(todo):ap.error('invalid limit')
        todo=todo[:args.limit]
    report={'before':snapshot(),'fixtures_sha256':hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
            'runs':[],'iteration_counts':repeats,'drain_each':args.drain_each,'vtcm_offset':args.vtcm_offset}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        with FrameTransport((PATHS.native / 'build/qd_loader.elf'),(PATHS.native / 'build/qnn_hmx_1x1_frame.elf'),10) as t:
            report['payload_sha256']=t.payload_sha256
            for index,(header,parts) in enumerate(todo):
                h,w,ci,co,random=header
                different_exact=check_integer_reference(header,parts)
                packet=b''.join(parts[:3]);expected=parts[3]
                for reps in repeats:
                    def prepare(seq):
                        t.arena[:4096]=bytes(4096)
                        struct.pack_into('<10I',t.arena,0,MAGIC,1,seq,h,w,ci,co,reps,int(args.drain_each),args.vtcm_offset)
                        t.arena[0x10000:0x10000+len(packet)]=packet
                        if args.drain_each:t.arena[0x10000+len(packet):0x10000+len(packet)+len(expected)]=expected
                        t.fence()
                    def collect():
                        words=struct.unpack_from('<32I',t.arena,2048)
                        if args.vtcm_offset and words[18] and words[17]!=1:
                            t._stuck=True
                            raise RuntimeError('upper VTCM not restored; retain callback/backup and paused eye client')
                        return (words,bytes(t.arena[0x100000:0x100000+len(expected)]),
                                bytes(t.arena[0x200000:0x200000+len(packet)]))
                    started=time.monotonic()
                    words,got,echo=t.invoke(prepare,lambda:t.workers_done() and t.workers_exited(),collect)
                    row={'fixture':index,'h':h,'w':w,'ci':ci,'co':co,'random':bool(random),'iterations':reps,
                         'seconds':time.monotonic()-started,'words':list(words),
                         'equal_simulator_native':got==expected,'operands_unchanged':echo==packet,
                         'different_exact_reference':different_exact,
                         'sha256':hashlib.sha256(got).hexdigest(),
                         'kernel_us_per_call':(words[5]+(words[6]<<32))/1843.2/reps,
                         'setup_us':words[4]/1843.2,'collect_us':words[7]/1843.2,
                         'vtcm_saved_us':words[15]/1843.2,'vtcm_restored_us':words[16]/1843.2,'vtcm_restored':bool(words[17])}
                    report['runs'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k!='words'}),flush=True)
                    if words[0]!=MAGIC or words[1] or (args.drain_each and words[12]!=reps) or not row['equal_simulator_native'] or not row['operands_unchanged']:
                        raise RuntimeError(f'hardware leaf gate failed: {row}')
        report['after']=snapshot();report['passed']=True
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:save()
    print('ALL PASS',flush=True)

if __name__=='__main__':main()
