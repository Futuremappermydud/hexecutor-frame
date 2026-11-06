#!/usr/bin/env python3
"""Capture real NativeGraph CPU state immediately around two blocked ops."""
from __future__ import annotations

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


import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT=(PATHS.repo)
sys.path.insert(0,str((PATHS.native)))
sys.path.insert(0,str((PATHS.model_tools)))
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

TARGETS=((10,'Add.tcm@CB*3','add'),(39,'linearclip@CH.CB.fi','linearclip'),(8,'linearclip_simple_rq@CB*2.fi','requant'))

def output_pages(tensor,width:int,pool_expected:int):
    sh=tensor['shape'];dims=sh['dimensions'];maximum=sh['max_dimensions'];origin=sh['padding']
    storage=tensor['storage'];blocks=storage['blocks'];slice_origin=storage.get('slice_offset',0)
    nbh=(maximum[1]+7)//8;nbw=(maximum[2]+width-1)//width;nbd=(maximum[3]+31)//32
    pages=set()
    for b in range(dims[0]):
      for h in range(dims[1]):
       H=h+origin[1]
       for w in range(dims[2]):
        W=w+origin[2]
        for d in range(dims[3]):
         D=d+origin[3]
         ix=(((b+origin[0])*nbh+H//8)*nbw+W//width)*nbd+slice_origin+D//32
         if ix>=len(blocks):raise ValueError('CPU fixture output page table does not cover logical shape')
         entry=blocks[ix]
         if entry['pool']!=pool_expected:raise ValueError('elementwise CPU fixture output is not entirely in pool1')
         pages.add(entry['offset'])
    return sorted(pages)

def capture(model:str,case:str,outdir:Path):
    image=(PATHS.native / f'build/resident-{model}.qdrx')
    graph_json=resident_image.unpack(image.read_bytes())
    tensors={t['id']:t for t in graph_json['tensors']}
    matches=[]
    for i,c in enumerate(graph_json['commands']):
        for opcode,variant_name,label in TARGETS:
            if c['opcode']==opcode and c['variant']==variant_name:matches.append((label,i,c))
    targets={}
    for label,i,c in matches:targets.setdefault(label,(i,c))
    constants=(PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    case_dir=next(row[1] for row in cases(model,[case])[1] if row[0]==case)
    inputs=load_inputs(case_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    graph=NativeGraph(image,constants)
    try:
        graph.use_cpu();graph.set_hoist(False);graph.set_block_output(False)
        for opcode,variant_name,label in TARGETS:
            if label not in targets:raise ValueError(f'missing {label} fixture command in {model}')
            index,c=targets[label]
            # First argument 0 triggers the same zero-pool initialization as
            # NativeGraph's normal complete run. Stop immediately before the
            # target, snapshot real producer results/current Fi, then run only
            # the target command against those exact buffers.
            graph.run(inputs,0,index)
            before={p:(graph.pools[p].copy() if p in graph.pools else np.zeros(0,dtype=np.uint8)) for p in range(7)}
            fi_bytes=b'';fi_pool=0xffffffff;fi_offset=0
            if label in ('linearclip', 'requant'):
                fi=tensors[c['inputs'][1]];st=fi['storage']['direct']
                fi_pool=st['pool'];fi_offset=st['offset']
                fi_bytes=before[fi_pool][fi_offset:fi_offset+48].tobytes()
                if len(fi_bytes)!=48:raise ValueError(f'{label} Fi does not fit direct pool range')
            graph.run(inputs,index,index+1)
            after=graph.pools[1]
            output=tensors[c['outputs'][0]]
            pages=output_pages(output,4 if label=='linearclip' else 8,1)
            fixture=outdir/f'{model}-{case}-{label}.epcpu'
            with fixture.open('wb') as f:
                variant=resident_image.VARIANTS[variant_name]
                pool_sizes=[before[p].nbytes for p in range(7)]
                f.write(struct.pack('<4s16I',b'EPCP',1,opcode,variant,c['sequence'],index,
                                    fi_pool,fi_offset,len(fi_bytes),len(pages),*pool_sizes))
                for p in range(7):f.write(before[p].tobytes())
                for off in pages:
                    f.write(struct.pack('<I',off));f.write(after[off:off+2048].tobytes())
            print(f'{model}/{case}/{label}: command={index} sequence={c["sequence"]} '
                  f'pages={len(pages)} fixture={fixture} bytes={fixture.stat().st_size}',flush=True)
    finally:
        graph.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('smoke','random'),default='smoke')
    ap.add_argument('--model',choices=('primary','alternate'),action='append')
    ap.add_argument('--out',type=Path,default=(PATHS.native / 'program/fixtures'))
    args=ap.parse_args()
    for model in args.model or ('primary','alternate'):capture(model,args.case,args.out)

if __name__=='__main__':main()
