#!/usr/bin/env python3
"""Capture actual runtime packed weights/FI at QDRX command 2380 using CPU graph."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
FRAME=Path(__file__).resolve().parents[1];ROOT=FRAME.parent
sys.path[:0]=[str(FRAME),str(ROOT/'tools')]
from native_graph import NativeGraph
from resident_image import unpack
from check_kernels import load_inputs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--image',type=Path,default=FRAME/'build/resident-primary.qdrx')
 ap.add_argument('--inputs',type=Path,default=FRAME/'build/depth-inputs/random');ap.add_argument('--output',type=Path,default=Path(__file__).with_name('dynamic-fi-2380.json'));a=ap.parse_args()
 raw=a.image.read_bytes();g=unpack(raw);c=next(x for x in g['commands'] if x['sequence']==2380);idx=g['commands'].index(c)
 t={x['id']:x for x in g['tensors']};w,fi=(t[z] for z in c['inputs'][1:3]);
 host=NativeGraph(a.image,ROOT/'recovered/primary/constants/constant_pool_5.bin')
 try:
  host.use_cpu();host.run(load_inputs(a.inputs),last=idx+1)
  def bytes_for(z):
   ref=z['storage']['direct'];pool,off=ref['pool'],ref['offset'];n=z['element_bytes']
   for d in z['shape']['dimensions']:n*=d
   return host.pools[pool][off:off+n].tobytes(),ref
  wb,wr=bytes_for(w);fb,fr=bytes_for(fi)
  report={'sequence':2380,'command_index':idx,'executed_cpu_prefix_through_index':idx,
    'variant':c['variant'],'input_bytes_sha256':hashlib.sha256(a.inputs.joinpath('x.raw').read_bytes()).hexdigest(),
    'weights_tensor':w['id'],'weights_ref':wr,'weights_bytes':len(wb),'weights_sha256':hashlib.sha256(wb).hexdigest(),
    'fi_tensor':fi['id'],'fi_ref':fr,'fi_bytes':len(fb),'fi_sha256':hashlib.sha256(fb).hexdigest(),
    'execution':'NativeGraph CPU QDRX prefix, no device or DSP'}
  a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 finally:host.close()
if __name__=='__main__':main()
