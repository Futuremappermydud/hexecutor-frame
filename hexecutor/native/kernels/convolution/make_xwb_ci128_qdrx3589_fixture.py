#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture actual CPU-prefix sequence 3589 (CI128/CO128 stride-1 Xwb)."""

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

import hashlib,struct,sys
from pathlib import Path
HERE=(PATHS.native / 'kernels/convolution');FRAME=(PATHS.native);ROOT=FRAME.parent
sys.path[:0]=[str(FRAME),str(ROOT/'tools')]
from depth_model.adapters import NativeGraph
from depth_model.lowering.resident_image import unpack
from check_kernels import load_inputs
MAGIC=0x394B5748;SEQUENCE=3589
def blocks(t,p):return b''.join(bytes(p[e['pool']][e['offset']:e['offset']+2048]) for e in t['storage']['blocks'])
def direct(t,p):
 q=t['storage']['direct'];n=t['element_bytes']
 for d in t['shape']['dimensions']:n*=d
 return bytes(p[q['pool']][q['offset']:q['offset']+n])
def main():
 image=FRAME/'build/resident-primary.qdrx';g=unpack(image.read_bytes());cmd=next(c for c in g['commands'] if c['sequence']==SEQUENCE);idx=g['commands'].index(cmd)
 ts={t['id']:t for t in g['tensors']};x,w,fi=(ts[i] for i in cmd['inputs'][:3]);y=ts[cmd['outputs'][0]]
 host=NativeGraph(image,ROOT/'recovered/primary/constants/constant_pool_5.bin')
 try:
  host.use_cpu();inputs=load_inputs(FRAME/'build/depth-inputs/random');host.run(inputs,last=idx)
  xb=blocks(x,host.pools);wb=direct(w,host.pools);fb=direct(fi,host.pools)
  host.run(inputs,first=idx,last=idx+1);yb=blocks(y,host.pools)
  v=[MAGIC,SEQUENCE,x['shape']['dimensions'][1],x['shape']['dimensions'][2],x['shape']['max_dimensions'][1],x['shape']['max_dimensions'][2],x['storage']['slice_offset'],len(x['storage']['blocks']),y['shape']['dimensions'][1],y['shape']['dimensions'][2],y['shape']['max_dimensions'][1],y['shape']['max_dimensions'][2],y['storage']['slice_offset'],len(y['storage']['blocks']),w['shape']['dimensions'][2],w['shape']['dimensions'][3],2 if w['class_name']=='Xwb' else 1,len(wb),len(fb),x['shape']['padding'][1],x['shape']['padding'][2],y['shape']['padding'][1],y['shape']['padding'][2],x.get('quantization',{}).get('zero_point',0)]
  v += [0]*(32-len(v));raw=struct.pack('<32I',*v)+xb+yb+wb+fb;out=(PATHS.native / 'kernels/convolution/xwb_ci128_qdrx3589_cpu_prefix_fixture.bin');out.write_bytes(raw)
  print(f'fixture={out} bytes={len(raw)} X={len(xb)} Y={len(yb)} W={len(wb)} FI={len(fb)} output_slice={v[12]}')
  print(f'fixture_sha256={hashlib.sha256(raw).hexdigest()} image_sha256={hashlib.sha256(image.read_bytes()).hexdigest()}')
 finally:host.close()
if __name__=='__main__':main()
