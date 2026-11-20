# SPDX-License-Identifier: LGPL-3.0-or-later
"""Report exact differences; this calibration is not a passing backend gate."""

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

from pathlib import Path
import json
import numpy as np
BUILD=(PATHS.native / 'build')
r=np.load((PATHS.native / 'build/native-hmx-expected.npz'));meta=r['meta'];want=r['output']
got=np.fromfile((PATHS.native / 'build/native-hmx-output.bin'),np.uint8).reshape(-1,16,32,4).transpose(0,1,3,2).reshape(-1,64,32)
if got.shape!=want.shape:raise ValueError('incomplete simulator output')
rows=[]
for mode in np.unique(meta[:,2]):
    p=meta[:,2]==mode;d=got[p].astype(np.int16)-want[p]
    row={'mode':int(mode),'different':int(np.count_nonzero(d)),'total':d.size,'min':int(d.min()),'max':int(d.max())}
    row['exact']=row['different']==0;rows.append(row);print(json.dumps(row))
((PATHS.native / 'build/native-hmx-conversion-validation.json')).write_text(json.dumps({'purpose':'direct native conversion calibration; no hardware/backend integration','modes':rows},indent=2)+'\n')
