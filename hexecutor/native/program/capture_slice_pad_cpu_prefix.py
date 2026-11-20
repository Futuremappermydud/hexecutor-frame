#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture CPU-prefix pool state for the Alternate InputSlicePad ops (cmd 68 & 69)."""
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

ROOT = (PATHS.repo)
sys.path.insert(0, str((PATHS.native)))
sys.path.insert(0, str((PATHS.model_tools)))
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

TARGETS = {
    'alternate': (
        (68, 118, 'cmd68'),
        (69, 120, 'cmd69'),
    ),
    'primary': (
        (19, 7, 'cmd19'),
        (20, 15, 'cmd20'),
    ),
}

def capture(model: str, case: str, outdir: Path):
    image = (PATHS.native / f'build/resident-{model}.qdrx')
    graph_json = resident_image.unpack(image.read_bytes())
    tensors = {t['id']: t for t in graph_json['tensors']}
    targets = TARGETS[model]

    constants = (PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    case_dir = next(row[1] for row in cases(model, [case])[1] if row[0] == case)
    inputs = load_inputs(case_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for cmd_idx, expected_seq, label in targets:
        c = graph_json['commands'][cmd_idx]
        if c['opcode'] != 2 or not c['variant'].startswith('*InputSlicePad'):
            raise ValueError(f'{model}: command {cmd_idx} is not InputSlicePad ({c["variant"]})')
        if c['sequence'] != expected_seq:
            raise ValueError(f'{model}: command {cmd_idx} sequence is {c["sequence"]}, expected {expected_seq}')

        out_tensor = tensors[c['outputs'][0]]
        if out_tensor['class_name'] != 'FB' or out_tensor['dtype'] != 'uint8':
            raise ValueError(f'{model}: output tensor {c["outputs"][0]} is not u8 FB')
        if 'direct' not in out_tensor['storage']:
            raise ValueError(f'{model}: output tensor is not direct storage')

        st = out_tensor['storage']['direct']
        if st['pool'] != 1:
            raise ValueError(f'{model}: output pool is {st["pool"]}, expected 1')
        offset = st['offset']
        shape = out_tensor['shape']['dimensions']
        nbytes = 1
        for d in shape:
            nbytes *= d

        graph = NativeGraph(image, constants)
        try:
            graph.use_cpu()
            graph.set_hoist(False)
            graph.set_block_output(False)

            # Run up to target command
            graph.run(inputs, 0, cmd_idx)
            before = {p: (graph.pools[p].copy() if p in graph.pools else bytes()) for p in range(7)}

            io_idx = c['io_index']
            io_data = inputs[io_idx].tobytes()

            # Run target command
            graph.run(inputs, cmd_idx, cmd_idx + 1)
            after_pool1 = graph.pools[1]

            sizes = [len(before[p]) for p in range(7)]
            path = outdir / f'{model}-{case}-slice-pad-{label}.ppcpu'
            with path.open('wb') as f:
                variant_id = resident_image.VARIANTS[c['variant']]
                # PPFX header: version/op/variant/sequence/index/io-index/io bytes/range count/pool sizes
                f.write(struct.pack('<4s15I', b'PPFX', 1, c['opcode'], variant_id, c['sequence'],
                                    cmd_idx, io_idx, len(io_data), 1, *sizes))
                for p in range(7):
                    f.write(before[p].tobytes() if hasattr(before[p], 'tobytes') else before[p])
                f.write(io_data)
                f.write(struct.pack('<II', offset, nbytes))
                f.write(after_pool1[offset:offset + nbytes].tobytes())

            print(f'{model}/{case}: opcode={c["opcode"]} variant={variant_id} command={cmd_idx} '
                  f'seq={c["sequence"]} out_offset={offset} out_bytes={nbytes} fixture={path.name} ({path.stat().st_size} bytes)', flush=True)
        finally:
            graph.close()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=('smoke', 'random'), default='smoke')
    ap.add_argument('--model', choices=('alternate', 'primary'), default='alternate')
    ap.add_argument('--out', type=Path, default=(PATHS.native / 'program/fixtures'))
    args = ap.parse_args()
    capture(args.model, args.case, args.out)

if __name__ == '__main__':
    main()
