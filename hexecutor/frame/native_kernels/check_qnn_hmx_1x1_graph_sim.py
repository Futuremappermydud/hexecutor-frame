"""Check the original native leaf against independent int64 matrix arithmetic."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

build=Path(__file__).resolve().parent
raw=(build/'qnn-hmx-1x1-graph-output.bin').read_bytes()
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
    assert (h,w,ci,co) in {(8,8,128,64),(8,8,128,128),(8,16,128,128),(8,32,128,128),(8,64,128,128),(4,16,128,128),(2,8,128,128)} and random in (0,1)
    hp=(h+7)//8*8;wp=(w+7)//8*8
    a=dense(read('u1',hp*wp*ci),hp,wp,ci).astype(np.int64)
    weights=read('i1',ci*co).reshape(co//32,ci//32,8,32,4).transpose(1,2,4,0,3).reshape(ci,co).astype(np.int64)
    fi=read('<i4',co*2).reshape(co//32,64).astype(np.int64)
    got=dense(read('u1',hp*wp*co),hp,wp,co).astype(np.int64)
    if random:
        np.testing.assert_array_equal(fi[:,:32],0x404200)
        acc=a@weights+fi[:,32:].reshape(co)
        want=np.clip((((acc>>4)*1536+8192)>>14),0,255)
    else:
        np.testing.assert_array_equal(fi[:,:32],0x406000)
        want=a@weights
    delta=got-want
    assert delta.min()>=0 and delta.max()<=(1 if random else 0),(h,w,ci,co,random,int(delta.min()),int(delta.max()))
    rows.append({'h':h,'w':w,'ci':ci,'co':co,'random':bool(random),'values':got.size,'logical_values':h*w*co,'padded_values':got.size-h*w*co,'different':int(np.count_nonzero(delta)),'min':int(delta.min()),'max':int(delta.max())})
assert len(rows)==14 and len({(r['h'],r['w'],r['ci'],r['co'],r['random']) for r in rows})==14
report={'passed':True,'fixtures':rows,'values':sum(r['values'] for r in rows),'sha256':hashlib.sha256(raw).hexdigest(),'native_minus_exact_bound':[0,1]}
(build/'qnn-hmx-1x1-graph-sim.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"PASS {len(rows)} graph geometry fixtures, {report['values']} outputs, native conversion differences bounded to 0/+1")
