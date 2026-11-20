#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-check HTP graph Xwb shape/padding with recovered Xwb element offsets."""

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

import importlib.util
import json
from pathlib import Path

ROOT = (PATHS.repo)
GRAPH = (PATHS.model_data / 'recovered/primary/graph.json')
EXPORT = (PATHS.model_tools / 'export_constants.py')
spec = importlib.util.spec_from_file_location('export_constants', EXPORT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
graph = json.loads(GRAPH.read_text())
tensors = {t['id']: t for t in graph['tensors']}
node = next(n for n in graph['nodes']
            if n['type_name'].startswith('ConvLayer_s1.opt@CB*2.Xwb')
            and tensors[n['inputs'][0]]['shape']['dimensions'] == [1, 10, 258, 32]
            and tensors[n['inputs'][1]]['shape']['dimensions'] == [3, 3, 32, 32])
in_t = tensors[node['inputs'][0]]
w_t = tensors[node['inputs'][1]]
out_t = tensors[node['outputs'][0]]
chunks = (1, 8, 8, 32)  # R4CroutonLayout from QAIRT memory_layout.h
padded = tuple(((d + c - 1) // c) * c
               for d, c in zip(in_t['shape']['max_dimensions'], chunks))
assert padded == (1, 24, 272, 32), padded
assert in_t['shape']['padding'] == [0, 7, 7, 0]
assert out_t['shape']['max_dimensions'] == [1, 8, 256, 32]
assert in_t['class_name'] == out_t['class_name'] == 'CB'
assert in_t['dtype'] == 'uint8' and in_t['element_bytes'] == 1
assert in_t['quantization']['zero_point'] == 0
assert w_t['class_name'] == 'Xwb'
assert w_t['shape']['max_dimensions'] == [3, 3, 32, 32]

print(f"node sequence={node['sequence']} type={node['type_name']}")
print(f"input dims={in_t['shape']['dimensions']} max={in_t['shape']['max_dimensions']} padding={in_t['shape']['padding']} padded={padded}")
print(f"output dims={out_t['shape']['dimensions']} max={out_t['shape']['max_dimensions']}")
rowstride = (padded[2] // 8) * (padded[3] // 32)
blocks = (padded[1] // 8) * rowstride
print(f"input H/W/depth crouton blocks={padded[1]//8}x{padded[2]//8}x{padded[3]//32}; table row stride={rowstride}; blocks={blocks}; bytes={blocks*2048}")
print(f"derived setter Hblocks={padded[1]//8} x-origin-r5={(1-3-in_t['shape']['padding'][2]) & 7} y-trim={(-in_t['shape']['padding'][1]) & 7} input_zp={in_t['quantization']['zero_point']}")
print(f"weights class={w_t['class_name']} dims={w_t['shape']['dimensions']} max={w_t['shape']['max_dimensions']}")
for y in range(3):
    for x in range(3):
        # tools/export_constants.py's SDK-derived Xwb element_addr formula
        offset = mod.weight_offset((y, x, 0, 0), (3, 3, 32, 32), packed=False)
        # Existing simulator's conventional one-hot byte for K=(tap*CI+ci), O=0.
        k = (y * 3 + x) * 32
        fixture_offset = (((k // 32) * 8 + (k % 32) // 4) * 32) * 4 + k % 4
        assert offset == fixture_offset == (y * 3 + x) * 1024, (y, x, offset, fixture_offset)
        print(f"Xwb weight[y={y},x={x},i=0,o=0] byte={offset} fixture_byte={fixture_offset}")
print('PASS Xwb one-hot byte offsets and activation padding match recovered graph shape')
