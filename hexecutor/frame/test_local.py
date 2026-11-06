#!/usr/bin/env python3
"""Check the production packet dispatcher with its native scalar vector substitute."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from frame_hvx import HvxConvolution
from hvx_protocol import (ARENA_BYTES, HEADER, STATUS, STATUS_OFFSET, MAGIC, VERSION,
                          packet_layout)
from local_transport import LocalTransport


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library', type=Path, default=Path(__file__).parent / 'build/libqd_host.so')
    ap.add_argument('--report', required=True, type=Path)
    args = ap.parse_args()
    transport = LocalTransport(args.library)
    rng = np.random.default_rng(1829)
    checks = []
    for m, n, k in [(1,32,1), (2,64,12), (3,32,127), (4,128,128),
                    (19,32,255), (19,64,256), (19,64,257), (64,128,1152), (64,256,2048)]:
        for boundary in (False, True):
            a = rng.integers(-32768, 32768, (m, k), dtype=np.int16)
            b = rng.integers(-128, 128, (k, n), dtype=np.int32)
            if boundary:
                a[:] = -32768; b[:] = -128
            actual = transport.matmul(a, b)
            expected = a.astype(np.int64) @ b.astype(np.int64)
            assert np.array_equal(actual, expected), (m,n,k,boundary)
            checks.append({'shape': [m,n,k], 'boundary': boundary, 'equal': True})
    # Verify u16 centering and the int64 correction, including values >=32768.
    conv = HvxConvolution(transport, verify=True)
    for dtype, maxval in ((np.uint8, 255), (np.uint16, 65535)):
        x = rng.integers(0, maxval+1, (5,7,32), dtype=dtype)
        x.reshape(-1)[:4] = [0, maxval//2, maxval//2+1, maxval]
        w = rng.integers(-128,128,(3,3,32,64), dtype=np.int32)
        for zp in (0, maxval//2, maxval):
            actual = conv.accumulate(x,w,zp)
            expected = np.zeros((3,5,64),np.int64)
            for ky in range(3):
                for kx in range(3):
                    expected += (x[ky:ky+3,kx:kx+5].astype(np.int64)-zp) @ w[ky,kx].astype(np.int64)
            assert np.array_equal(actual,expected)
            checks.append({'convolution_dtype': str(np.dtype(dtype)), 'zero_point': zp, 'equal': True})
    # Forged commands must fail before touching any matrix storage.
    invalid = [('magic',0,0), ('version',1,99), ('shape-m',4,8193), ('shape-n',5,31),
               ('shape-k',6,8193), ('a-alignment',7,4097), ('b-range',8,ARENA_BYTES-128),
               ('c-alias',9,4096)]
    for name, field, value in invalid:
        m,n,k=3,32,256; ao,bo,co,_,_=packet_layout(m,n,k)
        values=[MAGIC,VERSION,1,ARENA_BYTES,m,n,k,ao,bo,co,*([0]*6)]
        values[field]=value
        arena=bytearray(ARENA_BYTES); HEADER.pack_into(arena,0,*values)
        before=hashlib.sha256(arena[4096:]).digest()
        buf=(ctypes.c_uint8*len(arena)).from_buffer(arena)
        for tid in range(3):transport.library.qd_host_run(buf,tid)
        for tid in range(3):
            done,error,rows,_=STATUS.unpack_from(arena,STATUS_OFFSET+tid*STATUS.size)
            assert done==1 and error!=0 and rows==0,(name,tid,done,error,rows)
        assert hashlib.sha256(arena[4096:]).digest()==before,name
        checks.append({'invalid_command':name,'rejected_without_matrix_writes':True})
    report={'hardware_executed':False,'native_dispatcher':str(args.library),'checks':checks,'passed':len(checks)}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} local dispatcher/arithmetic checks passed')


if __name__=='__main__':main()
