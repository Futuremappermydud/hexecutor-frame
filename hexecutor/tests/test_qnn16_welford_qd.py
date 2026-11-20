# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare Q16 Welford, instance normalization and Swish against a captured QNN graph and VTCM trace.

Usage: python test_qnn16_welford_qd.py DECODED_DIR TRACE_DIR
"""
import sys,struct,ctypes as C,tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'hexecutor/offline'), str(ROOT / 'hexecutor/tests')]
from qnn16_program import Lowering,KIND,align
from qnn16_oracle import Oracle
from test_qnn16_program import build
if len(sys.argv) != 3:
 raise SystemExit(__doc__)
o=Oracle(sys.argv[1],sys.argv[2])
with tempfile.TemporaryDirectory() as td:
 lib=build(td); worst=0; ncheck=0
 for r,n,b,a in o.run(lambda n:n['type_name'].startswith(('InstanceNorm.fp16.compute_welford_partial_sums', 'va_qf32_parallel_welford_no_moments', 'InstanceNorm_sf.fp16.tcm', 'Swish_fp16', 'reduce_sum.fp16.tcm.crouton'))):
  L=Lowering(o.graph,{}); L.lower_node(n); word=L.words; ts=L.tensor_rows; ops=L.ops
  ts_off=128; op_off=ts_off+80*len(ts); word_off=op_off+64; io_off=word_off+4*len(word); total=io_off+64
  scratch=max(128,sum(align(np.prod(t[6:10])*4,128) for t in ts))
  hdr=struct.pack('<16I',0x50363151,1,total,len(ts),1,len(word),1,1,ts_off,op_off,word_off,io_off,0,scratch,256,8<<20)+bytes(64)
  image=bytearray(hdr+b''.join(struct.pack('<20I',*t) for t in ts)+struct.pack('<16I',*ops[0])+struct.pack(f'<{len(word)}I',*word)+struct.pack('<8I',1,1,1,1,0,4,0,0)+struct.pack('<8I',1,1,1,1,128,4,0,0))
  img=(C.c_ubyte*len(image)).from_buffer(image); err=C.c_uint32(); rc=lib.q16_host_open(img,len(image),0,C.byref(err))
  if rc: raise RuntimeError((n['sequence'],rc,err.value))
  mem=b.copy(); ex=np.zeros(256,np.uint8); tmp=np.zeros(scratch,np.uint8); constants=np.zeros(1,np.uint8)
  lib.q16_host_bind(mem.ctypes.data,ex.ctypes.data,tmp.ctypes.data,constants.ctypes.data)
  rc=lib.q16_host_step(0)
  if rc: raise RuntimeError((n['sequence'],rc))
  actual=o.tensor(n['outputs'][0],mem); expected=o.tensor(n['outputs'][0],a)
  diff=np.max(np.abs(actual-expected)); tol=(0.004 + np.max(np.abs(expected))*0.001 if n["type_name"].startswith(("InstanceNorm_sf", "Swish_fp16", "reduce_sum.fp16.tcm.crouton"))
       else 1e-4+np.max(np.abs(expected))*3e-6)
  if diff>tol:
   idx=np.unravel_index(np.argmax(abs(actual-expected)),actual.shape)
   raise RuntimeError((n['sequence'],diff,tol,idx,actual[idx],expected[idx]))
  worst=max(worst,float(diff)); ncheck+=1
 print('PASS',ncheck,'Welford/normalization/Swish operations; max abs error',worst)
