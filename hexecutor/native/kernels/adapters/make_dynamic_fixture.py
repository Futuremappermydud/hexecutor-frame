#!/usr/bin/env python3
"""Make command 2380 fixture from a real NativeGraph CPU prefix snapshot."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse,struct,sys
from pathlib import Path
FRAME=(PATHS.native);ROOT=(PATHS.repo)
sys.path[:0]=[str(FRAME),str(ROOT/'tools')]
from depth_model.adapters import NativeGraph
from depth_model.lowering.resident_image import unpack
from check_kernels import load_inputs
MAGIC=0x31585151

def blocks_for(t, pools):
 rows=[]
 for entry in t['storage']['blocks']:
  pool,offset=entry['pool'],entry['offset']
  rows.append(bytes(pools[pool][offset:offset+2048]))
 return b''.join(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--image',type=Path,default=(PATHS.native / 'build/resident-primary.qdrx'));ap.add_argument('--inputs',type=Path,default=(PATHS.native / 'build/depth-inputs/random'));ap.add_argument('--sequence',type=int,default=2380);ap.add_argument('--output',type=Path);a=ap.parse_args()
 g=unpack(a.image.read_bytes());cmd=next(c for c in g['commands'] if c['sequence']==a.sequence);idx=g['commands'].index(cmd);ts={t['id']:t for t in g['tensors']};x,w,fi=(ts[i] for i in cmd['inputs'][:3]);y=ts[cmd['outputs'][0]]
 host=NativeGraph(a.image,ROOT/'recovered/primary/constants/constant_pool_5.bin')
 try:
  host.use_cpu();host.run(load_inputs(a.inputs),last=idx)
  xb=blocks_for(x,host.pools);yb=blocks_for(y,host.pools)
  def direct(t):
   p,o=t['storage']['direct']['pool'],t['storage']['direct']['offset'];n=t['element_bytes']
   for d in t['shape']['dimensions']:n*=d
   return bytes(host.pools[p][o:o+n])
  wb,fb=direct(w),direct(fi)
  if len(xb)!=len(x['storage']['blocks'])*2048 or len(yb)!=len(y['storage']['blocks'])*2048:raise ValueError('bad block table payload')
  # 32-word header keeps all data sections naturally 128-byte aligned.
  vals=[MAGIC,a.sequence,x['shape']['dimensions'][1],x['shape']['dimensions'][2],x['shape']['max_dimensions'][1],x['shape']['max_dimensions'][2],x['storage']['slice_offset'],len(x['storage']['blocks']),y['shape']['dimensions'][1],y['shape']['dimensions'][2],y['shape']['max_dimensions'][1],y['shape']['max_dimensions'][2],y['storage']['slice_offset'],len(y['storage']['blocks']),w['shape']['dimensions'][2],w['shape']['dimensions'][3],1 if w['class_name']=='Xpb' else 2,len(wb),len(fb),x['shape']['padding'][1],x['shape']['padding'][2],y['shape']['padding'][1],y['shape']['padding'][2],x.get('quantization',{}).get('zero_point',0)]
  vals+= [0]*(32-len(vals));raw=struct.pack('<32I',*vals)+xb+yb+wb+fb;out=a.output or (PATHS.native / 'kernels/adapters/fixture-qdrx-2380.bin');out.write_bytes(raw)
  print(f'wrote actual CPU-prefix command {a.sequence} fixture: X{len(xb)} Y{len(yb)} W{len(wb)} Fi{len(fb)} bytes, slices {vals[6]}/{vals[12]} to {out}')
 finally:host.close()
if __name__=='__main__':main()
