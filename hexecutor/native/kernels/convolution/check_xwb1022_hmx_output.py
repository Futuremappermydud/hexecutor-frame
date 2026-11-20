#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare raw physical simulator output with the CPU-prefix logical tensor."""

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

import collections,hashlib,json
from pathlib import Path
here=(PATHS.native / 'kernels/convolution')
physical=((PATHS.native / 'kernels/convolution/xwb1022_hmx_output.bin')).read_bytes()
reference=((PATHS.native / 'kernels/convolution/xwb1022_cpu_output.bin')).read_bytes()
if len(physical)!=65536 or len(reference)!=65536:raise SystemExit('bad byte count')
counts=collections.Counter();first=None
for y in range(8):
 for x in range(256):
  page=(y//8)*32+x//8
  within=(y%8)*256+(x%8)*32
  for c in range(32):
   got=physical[page*2048+within+c]
   ref=reference[(y*256+x)*32+c]
   d=got-ref
   counts[d]+=1
   if d and first is None:first={'y':y,'x':x,'co':c,'got':got,'cpu':ref,'delta':d}
other=sum(v for d,v in counts.items() if d not in (0,1))
report={'qdrx_sequence':1022,'input_tensor':998,'weights_tensor':960,'fi_tensor':933,
 'physical_output_bytes':len(physical),'physical_output_sha256':hashlib.sha256(physical).hexdigest(),
 'cpu_logical_output_sha256':hashlib.sha256(reference).hexdigest(),
 'logical_compare':{'exact':counts[0],'plus1':counts[1],'minus1':counts[-1],
                    'other':other,'first_mismatch':first,
                    'criterion':'all deltas are 0 or +1'},
 'result':'PASS' if other==0 else 'FAIL'}
((PATHS.native / 'kernels/convolution/xwb1022_hmx_output_validation.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if other:raise SystemExit(1)
