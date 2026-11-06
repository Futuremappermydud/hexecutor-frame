"""Inventory/preserve upper VTCM under the existing exclusive callback lease.

Default is read-only. --write-restore requires a validated read-only snapshot
in the same exclusive session. Runtime contents can differ from the static
firmware blob; accepting that baseline requires --runtime-snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import struct
import sys
FRAME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(FRAME))
from frame_transport import FrameTransport,FIRMWARE,FIRMWARE_SHA256
from verify_resident_decoder import snapshot
LOW=0x2d0000;SIZE=0x800000-LOW;BACKUP=0x10000;MAGIC=0x31545551

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write-restore',action='store_true')
    ap.add_argument('--runtime-snapshot',action='store_true',help='accept stable live bytes as the baseline even when they differ from the static firmware blob')
    ap.add_argument('--passes',type=int,default=8)
    ap.add_argument('--sessions',type=int,default=2)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--dump',type=Path,help='save the last read-only snapshot for offline ownership analysis')
    args=ap.parse_args()
    if not 1<=args.passes<=32 or not 1<=args.sessions<=8:ap.error('invalid pass/session count')
    firmware=FIRMWARE.read_bytes()
    if hashlib.sha256(firmware).hexdigest()!=FIRMWARE_SHA256:raise ValueError('firmware ABI differs')
    # Decoded stock copy: VA 0x870052e0 copies [0x8707c400,+0x50ea80)
    # to 0x311e0000. ELF's single PT_LOAD file offset is 0x54.
    source=0x7c400+0x54+(LOW-0x1e0000);params_end=0x1e0000+0x50ea80
    expected=firmware[source:source+params_end-LOW]
    report={'before':snapshot(),'firmware_sha256':FIRMWARE_SHA256,'write_restore':args.write_restore,'runtime_snapshot':args.runtime_snapshot,
            'low':LOW,'bytes':SIZE,'parameter_end':params_end,'sessions':[]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for session in range(args.sessions):
            # Even write mode first inventories the range without changing it.
            with FrameTransport(FRAME/'build/qd_loader.elf',FRAME/'build/vtcm_upper_probe.elf',10) as t:
                report['payload_sha256']=t.payload_sha256
                for write in ([0,1] if args.write_restore else [0]):
                    nonce=secrets.randbits(32)
                    def prepare(seq):
                        t.arena[:4096]=bytes(4096)
                        t.arena[BACKUP+SIZE:BACKUP+SIZE+128]=b'\xa5'*128
                        struct.pack_into('<6I',t.arena,0,MAGIC,1,seq,write,args.passes,nonce)
                        t.fence()
                    def collect():
                        words=struct.unpack_from('<32I',t.arena,2048)
                        if write and words[1]==5:
                            t._stuck=True
                            raise RuntimeError('VTCM restore mismatch; retain callback/backup and paused eye client')
                        return words,bytes(t.arena[BACKUP:BACKUP+SIZE]),bytes(t.arena[BACKUP+SIZE:BACKUP+SIZE+128])
                    words,data,guard=t.invoke(prepare,lambda:t.workers_done() and t.workers_exited(),collect)
                    if args.dump is not None and not write:
                        args.dump.parent.mkdir(parents=True,exist_ok=True);args.dump.write_bytes(data)
                    row={'session':session,'write':bool(write),'nonce':nonce,'words':list(words),
                         'sha256':hashlib.sha256(data).hexdigest(),'upper_parameters_match_firmware':data[:len(expected)]==expected,
                         'arena_guard_ok':guard==b'\xa5'*128,'stable_passes':words[6],
                         'copy_ms':words[4]/1843200,'restore_ms':words[10]/1843200}
                    report['sessions'].append(row);save();print(json.dumps(row),flush=True)
                    if words[0]!=MAGIC or words[1] or not row['arena_guard_ok'] or words[6]!=args.passes or (write and words[11]!=1):
                        raise RuntimeError(f'VTCM probe failed: {row}')
                    if not row['upper_parameters_match_firmware'] and not args.runtime_snapshot:
                        raise RuntimeError('upper parameters differ from decoded firmware; stop before allowing writes')
            report['sessions'][-1]['after']=snapshot();save()
        report['passed']=True
    except BaseException as exc:
        report['error']=str(exc);raise
    finally:save()
    print('ALL PASS',flush=True)

if __name__=='__main__':main()
