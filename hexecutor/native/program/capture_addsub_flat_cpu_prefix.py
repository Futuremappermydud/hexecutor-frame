#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture CPU before/after pool images for each AddSubFlat profile."""
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

import argparse, struct, sys
from pathlib import Path
import numpy as np
ROOT=(PATHS.repo)
sys.path[:0]=[str((PATHS.native)),str((PATHS.model_tools))]
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',default='smoke');ap.add_argument('--out',type=Path,default=(PATHS.native / 'program/fixtures'))
    ap.add_argument('--model',choices=('primary','alternate','both'),default='both');a=ap.parse_args()
    models=('primary','alternate') if a.model=='both' else (a.model,)
    a.out.mkdir(parents=True,exist_ok=True)
    for model in models:
        case_dir=next(r[1] for r in cases(model,[a.case])[1] if r[0]==a.case)
        inputs=load_inputs(case_dir)
        image=(PATHS.native / f'build/resident-{model}.qdrx');decoded=resident_image.unpack(image.read_bytes())
        sites=[(i,c) for i,c in enumerate(decoded['commands']) if c.get('opcode')==42 and c.get('variant')=='addsub_op_flat@FH*3.fi']
        if len(sites)!=3:raise ValueError(f'{model}: expected three AddSubFlat profiles, got {len(sites)}')
        graph=NativeGraph(image,(PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin'))
        try:
            graph.use_cpu();graph.set_hoist(False);graph.set_block_output(False)
            for cmd,c in sites:
                graph.run(inputs,0,cmd)
                before=[graph.pools.get(i,np.zeros(0,dtype=np.uint8)).copy() for i in range(7)]
                graph.run(inputs,cmd,cmd+1)
                after=[graph.pools.get(i,np.zeros(0,dtype=np.uint8)).copy() for i in range(7)]
                changed=[]
                for pi,(x,y) in enumerate(zip(before,after)):
                    changed.extend((pi,int(q)) for q in np.flatnonzero(x!=y))
                out_id=c['outputs'][0];t={z['id']:z for z in decoded['tensors']}[out_id]
                storage=t['storage'].get('direct')
                if not storage or storage['pool']!=1:raise ValueError(f'{model} command {cmd}: expected direct pool1 output')
                off=storage['offset'];expected=[(1,off),(1,off+1)]
                if not changed or any(item not in expected for item in changed):
                    raise ValueError(f'{model} command {cmd}: unexpected changed bytes {changed[:16]}')
                path=a.out/f'{model}-smoke-addsub-flat-{cmd}.epcpu';sizes=[len(x) for x in before]
                header=struct.pack('<4sIIIIIIIII7I',b'EPCP',1,42,62,c['sequence'],cmd,0xffffffff,0,0,1,*sizes)
                page=off//2048*2048
                with path.open('wb') as f:
                    f.write(header)
                    for x in before:f.write(x.tobytes())
                    f.write(struct.pack('<I',page));f.write(after[1][page:page+2048].tobytes())
                print(f'captured {path} ({path.stat().st_size} bytes), output pool1[{off}:{off+2}]')
        finally:graph.close()
if __name__=='__main__':main()
