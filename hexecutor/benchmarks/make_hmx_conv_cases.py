# SPDX-License-Identifier: LGPL-3.0-or-later
"""Create independent exact convolution cases for the HMX ISA simulation."""

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

from pathlib import Path
import sys,struct
import numpy as np
ROOT=(PATHS.repo);sys.path[:0]=[str((PATHS.native))]
from hexecutor.offline.conv_protocol import pack_conv
from hexecutor.offline.hvx_protocol import ARENA_BYTES
rng=np.random.default_rng(751);cases=[];expected=[]
shapes=[((3,5,3),(2,2,3,32)),((10,34,128),(3,3,128,128)),((1,129,20),(1,1,20,32)),((9,33,512),(2,2,512,128)),((10,34,256),(3,3,256,128)),((10,98,64),(3,3,64,64)),((1,129,64),(1,1,64,64)),((3,65,32),(1,1,32,32))]
for xs,ws in shapes:
 x=rng.integers(0,256,xs,dtype=np.uint8);w=rng.integers(-128,128,ws,dtype=np.int8)
 n=ws[-1];q=np.empty((n,3),np.int32);q[:,0]=rng.integers(-100000,100000,n,dtype=np.int32);q[:,1]=rng.integers(1024,2048,n,dtype=np.int32);q[:,2]=rng.integers(0,13,n,dtype=np.int32)
 arena=bytearray(ARENA_BYTES);L=pack_conv(arena,x,w,1,quantization=q)
 kb=(ws[0]*ws[1]*L['g']+7)//8;D=kb*32
 padded=np.pad(w,((0,0),(0,0),(0,4*L['g']-xs[-1]),(0,0))).reshape(-1,n)
 padded=np.pad(padded,((0,D-len(padded)),(0,0)))
 packed=padded.reshape(kb,8,4,n//32,32).transpose(3,0,1,4,2).copy()
 cases.append(bytes(arena[:80])+bytes(arena[L['xo']:L['xo']+L['xb']+128]).ljust((L['xb']+128+2047)&~2047,b'\0')+packed.tobytes()+q.tobytes())
 kh,kw=ws[:2];oh,ow=xs[0]-kh+1,xs[1]-kw+1
 acc=sum(x[ky:ky+oh,kx:kx+ow].astype(np.int64)@w[ky,kx].astype(np.int64) for ky in range(kh) for kx in range(kw))
 pre=np.clip((acc+q[:,0])>>q[:,2],0,65535).astype(np.uint16)
 expected.append({'x':xs,'w':ws,'q':q,'pre':pre,'owp':L['owp']})
np.savez((PATHS.native / 'build/hmx-conv-expected.npz'),**{f'{i}_{key}':v for i,case in enumerate(expected) for key,v in case.items()})
((PATHS.native / 'build/hmx-conv-input.bin')).write_bytes(struct.pack('<I',len(cases))+b''.join(cases))
