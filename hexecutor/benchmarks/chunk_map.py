# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read-only: which host 2 MB chunk does each DSP IOVA chunk resolve to?"""

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

import json
from pathlib import Path
import sys
sys.path.insert(0,str((PATHS.native)))
from hexecutor.offline.cdsp_mem_check import chunk_map

bad=chunk_map()
print(json.dumps({'bad_chunks':{str(k):v for k,v in bad.items()},'n_bad':len(bad)}))
sys.exit(1 if bad else 0)
