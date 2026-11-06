#!/usr/bin/env python3
"""Extract the physical crouton output byte dump from a v75 simulator log."""

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

import hashlib
import json
from pathlib import Path
here = (PATHS.native / 'kernels/convolution')
log = ((PATHS.native / 'kernels/convolution/xwb1022-actual-prefix-hmx-output-dump.log')).read_text()
start, end = 'RAW_OUTPUT_PHYSICAL_HEX_BEGIN', 'RAW_OUTPUT_PHYSICAL_HEX_END'
if log.count(start) != 1 or log.count(end) != 1:
    raise SystemExit('missing or duplicate output dump markers')
hexdata = ''.join(log.split(start,1)[1].split(end,1)[0].split())
raw = bytes.fromhex(hexdata)
if len(raw) != 32 * 2048:
    raise SystemExit(f'wrong output size: {len(raw)}')
out = (PATHS.native / 'kernels/convolution/xwb1022_hmx_output.bin')
out.write_bytes(raw)
report = {'qdrx_sequence':1022,'command_index':836,'bytes':len(raw),
          'layout':'physical 32-page crouton output, row-major page table',
          'sha256':hashlib.sha256(raw).hexdigest(),
          'sim_log_sha256':hashlib.sha256(((PATHS.native / 'kernels/convolution/xwb1022-actual-prefix-hmx-output-dump.log')).read_bytes()).hexdigest()}
((PATHS.native / 'kernels/convolution/xwb1022_hmx_output.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
