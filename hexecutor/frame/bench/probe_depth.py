#!/usr/bin/env python3
"""Exact direct-conv checks with persistent command gaps and clock cleanup."""
import argparse
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_transport import FrameTransport


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--no-clock',action='store_true')
    args=ap.parse_args()
    build=Path(__file__).resolve().parents[1]/'build'
    rng=np.random.default_rng(20261029)
    cases=[((10,34,128),(3,3,128,128)),((10,258,32),(3,3,32,32)),
           ((9,33,512),(2,2,512,128)),((3,5,3),(2,2,3,8))]
    report={'raise_clock':not args.no_clock,'cases':[]}
    with FrameTransport(build/'qd_loader.elf',build/'qd_conv.elf',10,raise_clock=not args.no_clock) as t:
        for xs,ws in cases*2 + [cases[-1]]*30:
            # Reproduce host work gaps that previously triggered worker exit.
            time.sleep(.02)
            x=rng.integers(0,256,xs,dtype=np.uint8)
            w=rng.integers(-128,128,ws,dtype=np.int32)
            begin=time.monotonic()
            got=t.conv(x,w)
            seconds=time.monotonic()-begin
            kh,kw=ws[:2]; oh,ow=xs[0]-kh+1,xs[1]-kw+1
            expected=sum(x[ky:ky+oh,kx:kx+ow].astype(np.int64) @ w[ky,kx].astype(np.int64)
                         for ky in range(kh) for kx in range(kw))
            equal=np.array_equal(got,expected)
            item={'x':xs,'w':ws,'equal':equal,'seconds':seconds,
                  'worker_cycles':[struct.unpack_from('<4I',t.arena,2048+16*i)[:3] for i in range(3)]}
            report['cases'].append(item)
            print(json.dumps(item),flush=True)
            if not equal: raise RuntimeError('DSP convolution differs from CPU integer reference')
        t.stop_persistent()
        clock=struct.unpack_from('<23I',t.arena,256)
        report['clock']={'active_after_exit':clock[0], 'cfg_before':hex(clock[1]),
                         'cfg_raised':hex(clock[18]),'cfg_restored':hex(clock[20]),
                         'error':clock[21],'command_error':clock[22]}
        if clock[0] or clock[21] or clock[22]: raise RuntimeError(report['clock'])
        report['dispatch_seconds']=t.dispatch_seconds
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['clock']),flush=True)


if __name__=='__main__': main()
