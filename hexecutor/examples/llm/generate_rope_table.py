#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate Qwen3-0.6B's static NeoX RoPE coefficients for 257 positions."""

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

import math
from pathlib import Path
import struct

target=(PATHS.repo / 'hexecutor/examples/llm/rope_table.f32')
with target.open('wb') as output:
    for position in range(257):
        for index in range(64):
            angle=position/(1000000.0 ** (index/64.0))
            output.write(struct.pack('<ff',math.cos(angle),math.sin(angle)))
print(target,target.stat().st_size)
