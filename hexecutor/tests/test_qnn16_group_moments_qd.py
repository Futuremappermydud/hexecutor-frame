"""Compare grouped moments against a captured graph, VTCM writes and tensor dump.

Usage: python test_qnn16_group_moments_qd.py DECODED_DIR TRACE_DIR
"""
import sys,struct,ctypes as C,tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'hexecutor/offline'), str(ROOT / 'hexecutor/tests')]
from qnn16_program import Lowering,read_dumps,align
from qnn16_oracle import Oracle
from test_qnn16_program import build
if len(sys.argv) != 3:
 raise SystemExit(__doc__)
o=Oracle(sys.argv[1],sys.argv[2]);d=read_dumps(Path(sys.argv[2]) / 'tensors.bin')
with tempfile.TemporaryDirectory() as td:
 lib=build(td); worst=0;ncheck=0
 for r,n,b,a in o.run(lambda n:n['type_name'].startswith('va_parallel_welford_reduce_grouped_moments_sf')):
  L=Lowering(o.graph,d);L.lower_node(n); ts=L.tensor_rows; ops=L.ops; word=L.words
  tso=128; opo=tso+80*len(ts); wo=opo+64*len(ops); io=wo+4*len(word); total=io+64; scratch=align(L.arena_bytes+128,128)
  hdr=struct.pack('<16I',0x50363151,2,total,len(ts),len(ops),len(word),1,1,tso,opo,wo,io,len(L.constants),scratch,256,8<<20)+bytes(32)+struct.pack('<8I',L.arena_bytes,0,0,0,0,0,0,0)
  image=bytearray(hdr+b''.join(struct.pack('<20I',*t) for t in ts)+b''.join(struct.pack('<16I',*op) for op in ops)+struct.pack(f'<{len(word)}I',*word)+struct.pack('<8I',1,1,1,1,0,4,0,0)+struct.pack('<8I',1,1,1,1,128,4,0,0))
  img=(C.c_ubyte*len(image)).from_buffer(image);err=C.c_uint32();rc=lib.q16_host_open(img,len(image),len(L.constants),C.byref(err))
  if rc:raise RuntimeError((n['sequence'],rc,err.value))
  mem=b.copy(); ex=np.zeros(256,np.uint8); tmp=np.zeros(scratch,np.uint8); constants=np.frombuffer(bytes(L.constants),np.uint8).copy()
  row=ts[L.tensor_index[n['inputs'][6]]];tmp[row[2]:row[2]+row[3]]=np.frombuffer(d[n['inputs'][6]],np.uint8)
  lib.q16_host_bind(mem.ctypes.data,ex.ctypes.data,tmp.ctypes.data,constants.ctypes.data)
  for i in range(len(ops)):
   rc=lib.q16_host_step(i)
   if rc:raise RuntimeError((n['sequence'],i,rc))
  row=ts[L.tensor_index[n['outputs'][0]]]; got=tmp[row[2]:row[2]+row[3]].view(np.float32);want=np.frombuffer(d[n['outputs'][0]],np.float32)
  delta=np.max(abs(got-want));tol=1e-4+np.max(abs(want))*3e-6
  if delta>tol:raise RuntimeError((n['sequence'],delta,tol,got[:8],want[:8]))
  worst=max(worst,float(delta));ncheck+=1
 print('PASS',ncheck,'group moments; max abs error',worst)
