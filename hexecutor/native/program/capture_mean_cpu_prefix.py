#!/usr/bin/env python3
"""Capture real NativeGraph CPU state immediately around Mean ops."""
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

ROOT = (PATHS.repo)
sys.path.insert(0, str((PATHS.native)))
sys.path.insert(0, str((PATHS.model_tools)))
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

TARGETS = (
    (31, 'Mean@FH*2.fi.t', 'mean'),
)

def capture(model: str, case: str, outdir: Path):
    image = (PATHS.native / f'build/resident-{model}.qdrx')
    graph_json = resident_image.unpack(image.read_bytes())
    tensors = {t['id']: t for t in graph_json['tensors']}
    targets = {}
    for i, c in enumerate(graph_json['commands']):
        for opcode, variant_name, label in TARGETS:
            if c['opcode'] == opcode and c['variant'] == variant_name:
                targets.setdefault(label, (i, c))

    constants = (PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    case_dir = next(row[1] for row in cases(model, [case])[1] if row[0] == case)
    inputs = load_inputs(case_dir)

    for label, (cmd_index, c) in targets.items():
        graph = NativeGraph(image, constants)
        graph.use_cpu()
        graph.set_hoist(False)
        graph.set_block_output(False)
        out = tensors[c['outputs'][0]]
        out_offset = out['storage']['direct']['offset']
        out_bytes = 2  # 1 uint16

        # Run up to cmd_index
        graph.run(inputs, 0, cmd_index)
        pools_before = [(graph.pools[p].copy() if p in graph.pools else np.zeros(0, dtype=np.uint8)) for p in range(7)]
        pool_sizes = [len(p) for p in pools_before]

        # Run target command
        graph.run(inputs, cmd_index, cmd_index + 1)
        pools_after = [(graph.pools[p].copy() if p in graph.pools else np.zeros(0, dtype=np.uint8)) for p in range(7)]

        for p_idx in range(7):
            if p_idx == 1:
                diff = np.where(pools_before[1] != pools_after[1])[0]
                if len(diff) > 0:
                    if diff.min() < out_offset or diff.max() >= out_offset + out_bytes:
                        raise ValueError(f"Pool 1 mutated outside output slice: {diff}")
            else:
                if not np.array_equal(pools_before[p_idx], pools_after[p_idx]):
                    raise ValueError(f"Pool {p_idx} unexpectedly modified")

        # 2048-aligned page offset containing out_offset
        page_off = (out_offset // 2048) * 2048
        target_page_after = pools_after[1][page_off : page_off + 2048].tobytes()

        # EPCP header: 68 bytes
        # magic=EPCP(4), version=1(4), opcode(4), variant(4), sequence(4), command_index(4),
        # fi_pool(4), fi_offset(4), fi_bytes(4), page_count(4)=1, pool_bytes[7](28)
        header = struct.pack(
            '<4sIIIIIIIII7I',
            b'EPCP', 1, c['opcode'], 47, c['sequence'], cmd_index,
            0xffffffff, 0, 0, 1,
            *pool_sizes
        )
        body = bytearray()
        body.extend(header)
        for p in pools_before:
            body.extend(p.tobytes())
        body.extend(struct.pack('<I', page_off))
        body.extend(target_page_after)

        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f'{model}-{case}-{label}.epcpu'
        path.write_bytes(body)
        print(f'{model}/{case}/{label}: command={cmd_index} sequence={c["sequence"]} pages=1 fixture={path} bytes={len(body)}')
        graph.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='smoke')
    ap.add_argument('--model', default='both')
    ap.add_argument('--out', type=Path, default=(PATHS.native / 'program/fixtures'))
    args = ap.parse_args()
    models = ['primary', 'alternate'] if args.model == 'both' else [args.model]
    for m in models:
        capture(m, args.case, args.out)

if __name__ == '__main__':
    main()
