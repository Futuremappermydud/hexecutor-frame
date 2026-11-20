# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check the original native leaf against independent int64 matrix arithmetic."""

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
import hashlib
import json
import struct
import numpy as np

build=(PATHS.native / 'build')
raw=((PATHS.native / 'build/qnn-hmx-1x1-output.bin')).read_bytes()
at=0
rows=[]
def read(dtype,count):
    global at
    n=np.dtype(dtype).itemsize*count
    if at+n>len(raw):raise ValueError('incomplete simulator output')
    result=np.frombuffer(raw,dtype,count,at)
    at+=n
    return result
def dense(v,h,w,c):
    return v.reshape(h//8,w//8,c//32,8,8,32).transpose(0,3,1,4,2,5).reshape(h,w,c)
while at<len(raw):
    h,w,ci,co,random=map(int,read('<u4',5))
    assert h in (8,16) and w in (8,24) and ci in (32,64,96,128) and co in (32,64,96,128) and random in (0,1)
    a=dense(read('u1',h*w*ci),h,w,ci).astype(np.int64)
    weights=read('i1',ci*co).reshape(co//32,ci//32,8,32,4).transpose(1,2,4,0,3).reshape(ci,co).astype(np.int64)
    fi=read('<i4',co*2).reshape(co//32,64).astype(np.int64)
    got=dense(read('u1',h*w*co),h,w,co).astype(np.int64)
    if random:
        np.testing.assert_array_equal(fi[:,:32],0x404200)
        acc=a@weights+fi[:,32:].reshape(co)
        want=np.clip((((acc>>4)*1536+8192)>>14),0,255)
    else:
        np.testing.assert_array_equal(fi[:,:32],0x406000)
        want=a@weights
    delta=got-want
    assert delta.min()>=0 and delta.max()<=(1 if random else 0),(h,w,ci,co,random,int(delta.min()),int(delta.max()))
    rows.append({'h':h,'w':w,'ci':ci,'co':co,'random':bool(random),'values':got.size,'different':int(np.count_nonzero(delta)),'min':int(delta.min()),'max':int(delta.max())})
assert len(rows)==128 and len({(r['h'],r['w'],r['ci'],r['co'],r['random']) for r in rows})==128
report={'passed':True,'fixtures':rows,'values':sum(r['values'] for r in rows),'sha256':hashlib.sha256(raw).hexdigest(),'native_minus_exact_bound':[0,1]}
((PATHS.native / 'build/qnn-hmx-1x1-sim.json')).write_text(json.dumps(report,indent=2)+'\n')
print(f"PASS {len(rows)} fixtures, {report['values']} outputs, native conversion differences bounded to 0/+1")
