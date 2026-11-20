#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture sparse CPU-prefix fixtures for every prepared stride-2 S2D profile."""
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

import argparse, struct, sys
from pathlib import Path

ROOT = (PATHS.repo)
sys.path[:0] = [str((PATHS.native)), str((PATHS.model_tools))]
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

VARIANT = 'ConvLayer_SpaceToDepth_stride2.tcm@CB*2.s4'
OPCODE, VARIANT_ID = 7, 31

def encoded(block):
    return (int(block['offset']) & ~127) | int(block['pool'])

def capture(model: str, case: str, outdir: Path):
    image = (PATHS.native / f'build/resident-{model}.qdrx')
    graph_data = resident_image.unpack(image.read_bytes())
    tensors = {t['id']: t for t in graph_data['tensors']}
    selected = {}
    for index, c in enumerate(graph_data['commands']):
        if c['opcode'] != OPCODE or c['variant'] != VARIANT:
            continue
        x, y = (tensors[t] for t in (c['inputs'][0], c['outputs'][0]))
        shape = (tuple(x['shape']['max_dimensions']), tuple(y['shape']['dimensions']),
                 tuple(y['shape']['max_dimensions']))
        selected.setdefault(shape, (index, c, x, y))
    if len(selected) != 10:
        raise ValueError(f'{model}: expected 10 layout profiles, found {len(selected)}')
    case_dir = next(row[1] for row in cases(model, [case])[1] if row[0] == case)
    inputs = load_inputs(case_dir)
    constants = (PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    outdir.mkdir(parents=True, exist_ok=True)
    graph = NativeGraph(image, constants)
    try:
        graph.use_cpu(); graph.set_hoist(False); graph.set_block_output(False)
        for n, (shape, (index, c, x, y)) in enumerate(selected.items()):
            graph.run(inputs, 0, index)
            before = [graph.pools[p].copy() if p in graph.pools else None for p in range(7)]
            graph.run(inputs, index, index + 1)
            after = graph.pools
            in_entries = x['storage']['blocks']
            out_entries = y['storage']['blocks']
            in_pages = sorted({encoded(b) for b in in_entries})
            out_pages = sorted({encoded(b) for b in out_entries})
            path = outdir / f'{model}-s2d-profile-{n}.s2pf'
            with path.open('wb') as f:
                f.write(struct.pack('<4s7I', b'S2PF', 1, OPCODE, VARIANT_ID,
                                    c['sequence'], index, len(in_pages), len(out_pages)))
                for block in in_pages:
                    pool, off = block & 127, block & ~127
                    f.write(struct.pack('<I', block))
                    f.write(before[pool][off:off+2048].tobytes())
                for block in out_pages:
                    pool, off = block & 127, block & ~127
                    f.write(struct.pack('<I', block))
                    f.write(before[pool][off:off+2048].tobytes())
                    f.write(after[pool][off:off+2048].tobytes())
            print(f'{model} profile {n}: cmd={index} seq={c["sequence"]} '
                  f'in={len(in_pages)} out={len(out_pages)} fixture={path.name}')
    finally:
        graph.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=('smoke', 'random'), default='smoke')
    ap.add_argument('--model', choices=('primary', 'alternate'), action='append')
    ap.add_argument('--out', type=Path, default=(PATHS.native / 'program/fixtures'))
    args = ap.parse_args()
    for model in args.model or ('primary',): capture(model, args.case, args.out)

if __name__ == '__main__': main()
