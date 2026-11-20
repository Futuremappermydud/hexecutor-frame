#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate and measure whole VTCM GEMMs through the serialized HMX callback."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
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

HERE=(PATHS.benchmarks)
sys.path.insert(0,str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport

def pack_tiles(x):
    r,c=x.shape
    return x.reshape(r//32,16,2,c//32,32).transpose(0,3,1,4,2).astype(np.float16)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',nargs='+',default=['32,32,32','32,32,1024','128,128,128','256,256,256','512,512,512','1024,1024,1024'])
    p.add_argument('--flags',type=int,nargs='+',default=[3,1,0])
    p.add_argument('--reps',type=int,default=1)
    p.add_argument('--hmx-mhz',type=float,choices=(0,364.8,499.2,672,806.4,979.2),default=0)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    sizes=[tuple(map(int,s.split(','))) for s in args.sizes]
    if any(len(s)!=3 or any(d<32 or d>1024 or d%32 for d in s) for s in sizes):p.error('M,N,K must be multiples of 32, <=1024')
    if not 1<=args.reps<=100000 or any(f not in (0,1,2,3,4,8) for f in args.flags):p.error('invalid flags/repetitions')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    report={'rows':[], 'timer_frequency_hz':19200000,
            'timer_basis':'SM8650 architectural timer; kernel reports 19.20 MHz; counters measure DSP core ~1843.2 MHz',
            'timing_scope':'VTCM GEMM; excludes power, host dispatch, packing and DDR copies'}
    rng=np.random.default_rng(args.seed)
    build=(PATHS.native / 'build')
    try:
        with FrameTransport(build/'qd_loader.elf',build/'hmx_tune.elf',5) as t:
            report['payload_sha256']=t.payload_sha256
            for m,n,k in sizes:
                a=rng.integers(-1,2,(m,k),dtype=np.int32)
                b=rng.integers(-1,2,(k,n),dtype=np.int32)
                reference=a.astype(np.int64)@b.astype(np.int64)
                ap=pack_tiles(a).tobytes()
                bp=pack_tiles(b).transpose(1,0,2,3,4).copy().tobytes()
                co=0x20000+len(ap)+len(bp)
                cb=m*n*2
                for flags in args.flags:
                    row={'m':m,'n':n,'k':k,'reps':args.reps,'flags':flags,'hmx_mhz_requested':args.hmx_mhz}
                    def prepare(seq):
                        t.arena[:256]=bytes(256)
                        t.arena[2048:2304]=bytes(256)
                        struct.pack_into('<10I',t.arena,0,0x4e555448,1,seq,0,m,n,k,args.reps,flags,round(args.hmx_mhz*1e6))
                        t.arena[0x20000:0x20000+len(ap)]=ap
                        t.arena[0x20000+len(ap):co]=bp
                        t.arena[co:co+cb]=bytes(cb)
                        t.fence()
                    def collect():
                        w=list(struct.unpack_from('<32I',t.arena,2048))
                        raw=bytes(t.arena[co:co+cb])
                        c=np.frombuffer(raw,np.float16).reshape(m//32,n//32,16,32,2).transpose(0,2,4,1,3).reshape(m,n)
                        row.update(progress=w[0],error=w[1],on_error=w[2],on_pc=w[3],off_error=w[8],off_pc=w[9],
                                   cycles=w[4]+(w[5]<<32),timer_ticks=w[6]+(w[7]<<32),
                                   exact=bool(np.array_equal(c,reference)),max_abs_error=float(np.max(np.abs(c.astype(float)-reference))),
                                   output_sha256=hashlib.sha256(raw).hexdigest())
                        row.update(clock_error=w[11],restored_clock_cfg=w[12],restored_pll_mode=w[13])
                        row.update(completion_word=w[14],clock_restore_error=w[15])
                        report['clock_registers']=list(struct.unpack_from('<320I',t.arena,0x3000))
                    start=time.monotonic()
                    try:
                        t.invoke(prepare,lambda:t.workers_done() and t.workers_exited(),collect)
                    except BaseException as exc:
                        row['exception']=str(exc);row['progress_words']=list(struct.unpack_from('<32I',t.arena,2048))
                        report['rows'].append(row);raise
                    row['invoke_wall_seconds']=time.monotonic()-start
                    ops=2*m*n*k*args.reps
                    if row['timer_ticks']:
                        row['kernel_seconds']=row['timer_ticks']/19200000
                        row['gflops']=ops/row['kernel_seconds']/1e9
                        row['core_hz_from_counters']=row['cycles']/row['kernel_seconds']
                    report['rows'].append(row)
                    args.report.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(row),flush=True)
                    if row['error'] or not row['exact'] or row['on_pc']!=32 or row['off_pc']!=68 or row['completion_word'] or row['clock_restore_error']:
                        raise RuntimeError('HMX tuning validation failed')
    finally:
        args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
