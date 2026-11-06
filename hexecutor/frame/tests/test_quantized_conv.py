#!/usr/bin/env python3
"""Compare fused u8 convolution with independent int64 accumulation/rounding."""
import argparse
import ctypes
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conv_protocol import pack_conv,conv_result
from fast_depth import native_library
from local_transport import LocalTransport


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--hardware',action='store_true');ap.add_argument('--hmx',action='store_true');args=ap.parse_args()
    build=Path(__file__).resolve().parents[1]/'build';native=native_library()
    if args.hardware:
        from frame_transport import FrameTransport
        transport=FrameTransport(build/'qd_loader.elf',build/'qd_conv.elf',10,use_hmx=args.hmx)
    else:transport=LocalTransport(conv_library=build/'libqd_conv_host.so')
    rng=np.random.default_rng(2027)
    with transport as t:
        for xs,ws in [((3,5,3),(2,2,3,8)),((10,34,128),(3,3,128,128)),((1,129,20),(1,1,20,32))]:
            for boundary in (False,True):
                x=rng.integers(0,256,xs,dtype=np.uint8);w=rng.integers(-128,128,ws,dtype=np.int8)
                if boundary:x[:]=255;w[:]=-128
                q=np.empty((ws[3],3),np.int32);q[:,0]=rng.integers(-100000,100000,ws[3],dtype=np.int32)
                q[:,1]=rng.integers(1024,2048,ws[3],dtype=np.int32);q[:,2]=rng.integers(0,13,ws[3],dtype=np.int32)
                if args.hardware:got=t.conv_quantized(x,w,q)
                else:
                    L=pack_conv(t.arena,x,w,1,native=native,quantization=q)
                    ptrs=[(ctypes.c_uint8*len(b)).from_buffer(b) for b in (t.arena,t.vtcm,t.l2)]
                    for phase in (0,1):
                        for tid in range(3):t.conv_library.qdc_host_run(*ptrs,tid,phase)
                    got=conv_result(t.arena,L,1,native)
                kh,kw=ws[:2];oh,ow=xs[0]-kh+1,xs[1]-kw+1
                acc=sum(x[ky:ky+oh,kx:kx+ow].astype(np.int64) @ w[ky,kx].astype(np.int64) for ky in range(kh) for kx in range(kw))
                expected=np.clip(((((acc+q[:,0])>>q[:,2])*q[:,1])+8192)>>14,0,255).astype(np.uint8)
                np.testing.assert_array_equal(got,expected)
                print(xs,ws,'boundary' if boundary else 'random','PASS',flush=True)


if __name__=='__main__':main()
