#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture CPU-prefix pool state for the two exact CB activation-DMA fixtures."""
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

ROOT=(PATHS.repo)
sys.path.insert(0,str((PATHS.native)));sys.path.insert(0,str((PATHS.model_tools)))
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

TARGET='ConvLayer.opt.activations_to_vtcm_dma@CB.cB.'

def capture(model,case,outdir):
    image=(PATHS.native / f'build/resident-{model}.qdrx')
    graph_json=resident_image.unpack(image.read_bytes())
    expected=(5,67) if model=='primary' else (7,58)
    targets=[(i,c) for i,c in enumerate(graph_json['commands']) if i==expected[0] and c['opcode']==17 and c['variant']==TARGET]
    if len(targets)!=1:raise ValueError(f'{model}: expected one early {TARGET} at command {expected[0]}, found {len(targets)}')
    index,c=targets[0]
    if (index,c['sequence'])!=expected:raise ValueError(f'{model}: unexpected target command/index {index}/{c["sequence"]}')
    tensors={t['id']:t for t in graph_json['tensors']}
    x=tensors[c['inputs'][0]];y=tensors[c['outputs'][0]]
    if x['class_name']!='cB' or y['class_name']!='CB' or x['shape']['max_dimensions']!=[1,8,8,32] or y['shape']['max_dimensions']!=[1,8,8,32]:
        raise ValueError(f'{model}: actual activation tensor profile changed')
    out_storage=y['storage']
    if 'blocks' not in out_storage:raise ValueError('target output is not blocked')
    slice_origin=out_storage.get('slice_offset',0)
    encoded=out_storage['blocks'][slice_origin]
    pool=encoded['pool'];offset=encoded['offset']
    if pool!=1:raise ValueError(f'{model}: output page is in pool {pool}, expected VTCM pool 1')
    constants=(PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    case_dir=next(row[1] for row in cases(model,[case])[1] if row[0]==case)
    inputs=load_inputs(case_dir);graph=NativeGraph(image,constants);outdir.mkdir(parents=True,exist_ok=True)
    try:
        graph.use_cpu();graph.set_hoist(False);graph.set_block_output(False)
        graph.run(inputs,0,index)
        before={p:(graph.pools[p].copy() if p in graph.pools else bytes()) for p in range(7)}
        graph.run(inputs,index,index+1)
        sizes=[len(before[p]) for p in range(7)]
        path=outdir/f'{model}-{case}-activation-dma.ppcpu'
        with path.open('wb') as f:
            # PPFX header: version/op/variant/sequence/index/io-index/io bytes/range count/pool sizes.
            f.write(struct.pack('<4s15I',b'PPFX',1,c['opcode'],resident_image.VARIANTS[TARGET],c['sequence'],index,0xffffffff,0,1,*sizes))
            for p in range(7):f.write(before[p].tobytes() if hasattr(before[p],'tobytes') else before[p])
            f.write(struct.pack('<II',offset,2048));f.write(graph.pools[1][offset:offset+2048].tobytes())
        print(f'{model}/{case}: opcode={c["opcode"]} variant={resident_image.VARIANTS[TARGET]} command={index} seq={c["sequence"]} src_pool={x["storage"]["blocks"][0]["pool"]} output_page={offset} bytes={path.stat().st_size}',flush=True)
    finally:graph.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('smoke','random'),default='smoke')
    ap.add_argument('--out',type=Path,default=(PATHS.native / 'program/fixtures'))
    a=ap.parse_args()
    for model in ('primary','alternate'):capture(model,a.case,a.out)
if __name__=='__main__':main()
