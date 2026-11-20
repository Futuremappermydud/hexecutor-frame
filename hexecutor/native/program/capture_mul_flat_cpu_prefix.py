#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture CPU-prefix pools and command output for alternate flat multiply."""
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

import argparse, sys
from pathlib import Path
ROOT=(PATHS.repo)
sys.path.insert(0,str((PATHS.native))); sys.path.insert(0,str((PATHS.model_tools)))
import depth_model.lowering.resident_image as resident_image
from depth_model.adapters import NativeGraph
from check_kernels import load_inputs
from bench.verify_native_graph import cases

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',default='smoke'); ap.add_argument('--model',choices=['primary','alternate'],default='alternate'); ap.add_argument('--out',type=Path,default=(PATHS.native / 'program/fixtures')); a=ap.parse_args()
    model=a.model; image=(PATHS.native / f'build/resident-{model}.qdrx'); gjson=resident_image.unpack(image.read_bytes())
    hits=[(i,c) for i,c in enumerate(gjson['commands']) if c.get('variant')=='mul_op_flat@FH*2.fH.fi']
    if not hits: raise SystemExit(f'no mul_op_flat command in {model} resident image')
    case_dir=next(row[1] for row in cases(model,[a.case])[1] if row[0]==a.case); inputs=load_inputs(case_dir)
    tensors={t['id']:t for t in gjson['tensors']}
    for i,c in hits:
        graph=NativeGraph(image,(PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')); graph.use_cpu(); graph.set_hoist(False); graph.set_block_output(False)
        graph.run(inputs,0,i); before=[graph.pools[p].copy() if p in graph.pools else __import__('numpy').zeros(0,dtype='uint8') for p in range(7)]
        graph.run(inputs,i,i+1); after=[graph.pools[p].copy() if p in graph.pools else __import__('numpy').zeros(0,dtype='uint8') for p in range(7)]
        out=tensors[c['outputs'][0]]; storage=out['storage']['direct']; off=storage['offset']
        n=out['element_bytes']
        for d in out['shape']['dimensions']: n*=d
        import numpy as np, struct
        for pi,(old,new) in enumerate(zip(before,after)):
            dif=np.where(old!=new)[0]
            if pi!=1 and len(dif): raise SystemExit(f'CPU graph unexpectedly changed pool {pi}')
            if pi==1 and len(dif) and (dif.min()<off or dif.max()>=off+n): raise SystemExit('CPU graph wrote outside expected output')
        page0=(off//2048)*2048; page1=((off+n-1)//2048)*2048
        pages=list(range(page0,page1+1,2048))
        header=struct.pack('<4sIIIIIIIII7I',b'EPCP',1,41,75,c['sequence'],i,0xffffffff,0,0,len(pages),*[len(x) for x in before])
        body=bytearray(header)
        for x in before: body.extend(x.tobytes())
        for page in pages:
            body.extend(struct.pack('<I',page)); body.extend(after[1][page:page+2048].tobytes())
        a.out.mkdir(parents=True,exist_ok=True); path=a.out/f'{model}-{a.case}-mul-flat-{i}.epcpu'; path.write_bytes(body)
        print(f'command={i} sequence={c["sequence"]} fixture={path}')
        graph.close()
if __name__=='__main__': main()
