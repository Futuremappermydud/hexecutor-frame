#!/usr/bin/env python3
"""Verify the new packet callback on small signed/boundary dot products."""
import argparse
import json
from pathlib import Path
import signal

import numpy as np

from frame_transport import FrameTransport


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--callback',type=Path,default=Path(__file__).parent/'build/qd_hvx.elf')
    ap.add_argument('--report',required=True,type=Path)
    args=ap.parse_args()
    def interrupt(_signum,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    rng=np.random.default_rng(2026)
    cases=[]
    with FrameTransport(args.callback,timeout=15) as transport:
        for m,n,k in ((32,32,32),(19,64,257),(3,32,512)):
            a=rng.integers(-32768,32768,(m,k),dtype=np.int16)
            b=rng.integers(-128,128,(k,n),dtype=np.int32)
            if k==512:a[:]=-32768;b[:]=-128
            actual=transport.matmul(a,b)
            expected=a.astype(np.int64)@b.astype(np.int64)
            equal=bool(np.array_equal(actual,expected))
            cases.append({'shape':[m,n,k],'equal':equal,'differing':int(np.count_nonzero(actual!=expected))})
            print(cases[-1],flush=True)
            if not equal:raise RuntimeError('hardware dot product differs from CPU reference')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'hardware_executed':True,'callback_sha256':transport.callback_sha256,
                                      'cases':cases,'dispatch_seconds':transport.dispatch_seconds},indent=2)+'\n')


if __name__=='__main__':main()
