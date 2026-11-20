#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture exact NativeGraph CPU-prefix pool state around prepared convs.

Fixtures are independent of the prepared native handler. They contain every
pool immediately before the target call and complete physical output pages
after that call, so a host/V75 runner can prove both outputs and untouched
pool bytes against the Qualcomm-compatible CPU implementation.
"""
from __future__ import annotations

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

import argparse
import struct
import sys
from pathlib import Path

ROOT=(PATHS.repo)
sys.path.insert(0,str((PATHS.native)));sys.path.insert(0,str((PATHS.model_tools)))
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

# first 1x1 Xpb and the later padded-CI20 Xpb profile, in schedule order.
TARGET_WEIGHT_DIMS={
    'first12':(1,1,12,32),
    'ci20':(1,1,20,128),
}

def output_pages(t):
    st=t['storage'];sh=t['shape'];
    if 'blocks' not in st:raise ValueError('expected blocked convolution output')
    md=sh['max_dimensions'];n=md[0]*((md[1]+7)//8)*((md[2]+7)//8)*((md[3]+31)//32)
    origin=st.get('slice_offset',0);blocks=st['blocks'];out=set()
    if origin+n>len(blocks):raise ValueError('output max-grid exceeds block table')
    for b in blocks[origin:origin+n]:
        if b['pool']!=1:raise ValueError('native output block is outside pool1')
        out.add(b['offset'])
    return sorted(out)

def find_target(g,ts,key):
    wanted=TARGET_WEIGHT_DIMS[key]
    for i,c in enumerate(g['commands']):
        if c['opcode']!=6 or c['variant'] not in (
            'ConvLayer_s1.opt@CB*2.Xpb.Fi.t','ConvLayer_s1.opt@CB*2.Xwb.Fi.t'):
            continue
        w=ts[c['inputs'][1]]
        if tuple(w.get('shape',{}).get('dimensions',()))==wanted:
            return i,c
    raise ValueError(f'missing {key} profile with W dims {wanted}')

def capture(model,case,key,outdir):
    image=(PATHS.native / f'build/resident-{model}.qdrx');gj=resident_image.unpack(image.read_bytes())
    ts={t['id']:t for t in gj['tensors']};index,c=find_target(gj,ts,key)
    constants=(PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    case_dir=next(row[1] for row in cases(model,[case])[1] if row[0]==case)
    inputs=load_inputs(case_dir);outdir.mkdir(parents=True,exist_ok=True)
    graph=NativeGraph(image,constants)
    try:
        graph.use_cpu();graph.set_hoist(False);graph.set_block_output(False)
        # Match the CPU graph's ordinary zero-pool initialization, then stop
        # immediately before the chosen schedule command.
        graph.run(inputs,0,index)
        before={p:(graph.pools[p].copy() if p in graph.pools else b'') for p in range(7)}
        graph.run(inputs,index,index+1)
        pages=output_pages(ts[c['outputs'][0]])
        path=outdir/f'{model}-{case}-{key}.ncpfx'
        sizes=[len(before[p]) for p in range(7)]
        with path.open('wb') as f:
            # Header: version/op/variant/schedule and tensor ids, page count,
            # reserved word, then seven pool byte sizes (76 bytes total).
            f.write(struct.pack('<4s18I',b'NCPF',1,c['opcode'],resident_image.VARIANTS[c['variant']],
                                c['sequence'],index,c['inputs'][0],c['inputs'][1],
                                c['inputs'][2],c['outputs'][0],len(pages),0,*sizes))
            for p in range(7):f.write(before[p].tobytes() if hasattr(before[p],'tobytes') else before[p])
            after=graph.pools[1]
            for off in pages:
                f.write(struct.pack('<I',off));f.write(after[off:off+2048].tobytes())
        print(f'{model}/{case}/{key}: command={index} sequence={c["sequence"]} '
              f'W={tuple(ts[c["inputs"][1]]["shape"]["dimensions"])} '
              f'output_pages={len(pages)} fixture_bytes={path.stat().st_size}',flush=True)
    finally:graph.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('smoke','random'),default='smoke')
    ap.add_argument('--model',choices=('primary','alternate'),action='append')
    ap.add_argument('--profile',choices=tuple(TARGET_WEIGHT_DIMS),action='append')
    ap.add_argument('--out',type=Path,default=(PATHS.native / 'program/fixtures'))
    a=ap.parse_args()
    for m in a.model or ('primary','alternate'):
        for key in a.profile or tuple(TARGET_WEIGHT_DIMS):capture(m,a.case,key,a.out)

if __name__=='__main__':main()
