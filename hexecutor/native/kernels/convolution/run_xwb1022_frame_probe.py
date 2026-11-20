#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""One-shot Frame runner; device invocation is intentionally separate from host validation."""

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

import argparse, hashlib, json, struct, sys, time
from pathlib import Path

FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))

HERE = (PATHS.native / 'kernels/convolution')
MAGIC, FIXTURE_MAGIC = 0x334b4851, 0x334b4851
FIXTURE_OFFSET, OUTPUT_OFFSET, ECHO_OFFSET = 0x10000, 0x120000, 0x130000
EXPECTED_TLB = [0xF7062020, 0xC0031000, 0xF7062820, 0xC0031400]
HEADER = 204
SIZES = (208896, 9216, 256, 65536, 65536)
IN_BYTES = SIZES[0]

def parse_fixture(data):
    if len(data) < HEADER: raise ValueError('fixture truncated')
    h = struct.unpack_from('<11I', data)
    magic, version, seq, index, ib, ob = h[:6]
    sizes = h[6:]
    if (magic, version, seq, index, ib, ob, tuple(sizes)) != (FIXTURE_MAGIC,1,1022,836,102,32,SIZES):
        raise ValueError('fixture header does not match reviewed command 1022 geometry')
    pos=HEADER; pieces=[]
    for n in SIZES:
        pieces.append(data[pos:pos+n]); pos+=n
    if pos != len(data): raise ValueError('fixture has truncated or trailing bytes')
    hashes=[data[44+i*32:44+(i+1)*32].hex() for i in range(5)]
    if any(hashlib.sha256(x).hexdigest()!=want for x,want in zip(pieces,hashes)):
        raise ValueError('fixture operand/output SHA mismatch')
    return pieces

def physical_to_logical(raw):
    logical=bytearray(8*256*32)
    for y in range(8):
        for x in range(256):
            page=(y//8)*32+x//8
            src=page*2048+(y%8)*256+(x%8)*32
            dst=(y*256+x)*32
            logical[dst:dst+32]=raw[src:src+32]
    return logical

def validate_outputs(expected_sim, cpu, got, strict=True):
    if len(got) != 65536 or len(cpu) != 65536 or len(expected_sim) != 65536:
        raise ValueError('output buffers must contain the full physical output')
    if strict and got != expected_sim: raise RuntimeError('Frame HMX bytes differ from the simulator physical output')
    logical=physical_to_logical(got); counts={}; first=None; lo=255; hi=-255; valid=True
    for i,(a,b) in enumerate(zip(logical,cpu)):
        d=int(a)-int(b); counts[str(d)]=counts.get(str(d),0)+1; lo=min(lo,d); hi=max(hi,d)
        if d and first is None: first=[i//(256*32),(i//32)%256,i%32]
        if d not in (0,1): valid=False
    return {'simulator_physical_exact':got==expected_sim,'logical_delta_counts':counts,
            'logical_delta_min':lo,'logical_delta_max':hi,
            'cpu_0_or_plus1':valid,'first_mismatch':first}

def restore_problem(words):
    before, after = list(words[19:23]), list(words[23:27])
    return bool(words[10] or words[11] or (words[18] and
                (not words[17] or before != EXPECTED_TLB or after != EXPECTED_TLB)))

def fnv32(data):
    h=2166136261
    for b in data: h=((h^b)*16777619)&0xffffffff
    return h

def make_packet(sequence):
    return struct.pack('<6I',MAGIC,1,sequence,FIXTURE_OFFSET,0x700000,1)

def main():
    from hexecutor.offline.frame_transport import FrameTransport
    from bench.verify_resident_decoder import snapshot
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',type=Path,default=(PATHS.native / 'kernels/convolution/xwb1022_frame_fixture.bin'))
    ap.add_argument('--loader',type=Path,default=(PATHS.native / 'build/qd_loader.elf'))
    ap.add_argument('--payload',type=Path,default=(PATHS.native / 'kernels/convolution/xwb1022_frame_probe.elf'))
    ap.add_argument('--iterations',type=int,default=1)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if args.iterations!=1: raise ValueError('this reviewed probe allows exactly one kernel invocation')
    fixture=args.fixture.read_bytes(); parts=parse_fixture(fixture)
    inp,weights,fi,simout,cpu=parts
    if FIXTURE_OFFSET+len(fixture)>0x80000 or OUTPUT_OFFSET+len(simout)>ECHO_OFFSET or ECHO_OFFSET+len(inp)>0x170000:
        raise ValueError('arena range overlap or out-of-bounds')
    report={'offline_only':False,'purpose':'single simulator-correlated probe; leaf timing excludes staging, transfer, and graph overhead',
            'fixture_sha256':hashlib.sha256(fixture).hexdigest(),'before':snapshot(),
            'vtcm_offset':0x700000,'iterations':1,'run':None}
    with FrameTransport(args.loader,args.payload,10) as transport:
        report['payload_sha256']=transport.payload_sha256
        def prepare(sequence):
            transport.arena[:4096]=bytes(4096)
            packet=make_packet(sequence)
            transport.arena[:len(packet)]=packet
            transport.arena[FIXTURE_OFFSET:FIXTURE_OFFSET+len(fixture)]=fixture
            transport.fence()
        def collect():
            words=struct.unpack_from('<32I',transport.arena,2048)
            before=list(words[19:23]); restore=list(words[23:27])
            if restore_problem(words):
                transport.retain_session('3x3 probe could not verify HMX shutdown, TLB mapping, or full VTCM restore; retain session and backup')
            return words,bytes(transport.arena[OUTPUT_OFFSET:OUTPUT_OFFSET+65536]),bytes(transport.arena[ECHO_OFFSET:ECHO_OFFSET+IN_BYTES])
        started=time.monotonic()
        words,got,echo=transport.invoke(prepare,lambda:transport.workers_done() and transport.workers_exited(),collect)
    row={'words':list(words),'tlb_before_backup':list(words[19:23]),'tlb_before_restore':list(words[23:27]),
         'tlb_expected':EXPECTED_TLB,'input_unchanged':echo==inp,'input_echo_sha256':hashlib.sha256(echo).hexdigest(),
         'staged_input_fnv32':words[28],'expected_input_fnv32':fnv32(inp),
         'staged_weights_fnv32':words[29],'expected_weights_fnv32':fnv32(weights),
         'staged_fi_fnv32':words[30],'expected_fi_fnv32':fnv32(fi),
         'output_sha256':hashlib.sha256(got).hexdigest(),'simulator_output_sha256':hashlib.sha256(simout).hexdigest(),
         'elapsed_seconds_includes_transport':time.monotonic()-started,'vtcm_restored':bool(words[17]),
         'leaf_drained_us':(words[4]+(words[5]<<32))/1843.2,
         'prepare_us_probe_validation_and_staging':(words[6]+(words[7]<<32))/1843.2,
         'stage_us':words[12]/1843.2,'debug_operand_hash_us':words[13]/1843.2,
         'publish_config_us':words[14]/1843.2}
    row.update(validate_outputs(simout,cpu,got,strict=False));report['run']=row
    report['after']=snapshot();report['passed']=False
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.with_suffix('.output.bin').write_bytes(got)
    args.report.with_suffix('.input-echo.bin').write_bytes(echo)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row,indent=2))
    if restore_problem(words):
        raise RuntimeError(f'HMX/power/TLB/VTCM restore verification failed; session retained: {row}')
    if words[0]!=MAGIC or words[1] or not row['input_unchanged'] or not words[17] or not row['simulator_physical_exact'] or not row['cpu_0_or_plus1'] or any(row[k]!=row[v] for k,v in [('staged_input_fnv32','expected_input_fnv32'),('staged_weights_fnv32','expected_weights_fnv32'),('staged_fi_fnv32','expected_fi_fnv32')]):
        raise RuntimeError(f'3x3 probe failed validation: {row}')
    report['after']=snapshot();report['passed']=True
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
