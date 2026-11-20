# SPDX-License-Identifier: LGPL-3.0-or-later
"""Locate the first graph command whose pool bytes differ from native C."""

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

import json,sys,os
from pathlib import Path
import numpy as np
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from depth_model.adapters import ResidentDSP
from verify_native_graph import cases,ensure_image,load_inputs,NativeGraph
model=sys.argv[1] if len(sys.argv)>1 else 'primary'
image=ensure_image(model);constants=(PATHS.model_data / 'recovered')/model/'constants/constant_pool_5.bin'
case=sys.argv[2] if len(sys.argv)>2 else 'smoke'
inputs=load_inputs(cases(model,[case])[1][0][1]);rows=[]
with ResidentDSP(image,constants) as dsp:
    host=NativeGraph(image,constants)
    try:
        lo,hi=0,dsp.commands
        while lo+1<hi:
            mid=(lo+hi)//2;dsp.run(inputs,last=mid);host.run(inputs,last=mid)
            diffs={}
            for p in (1,3,4,5,6):
                a=dsp.read_pool(p);b=host.pools[p][:dsp.sizes[p]];idx=np.flatnonzero(a!=b)
                if idx.size:diffs[p]={'count':int(idx.size),'first':int(idx[0]),'dsp':int(a[idx[0]]),'host':int(b[idx[0]])}
            row={'prefix':mid,'diffs':diffs,**dsp.last};rows.append(row);print(json.dumps(row),flush=True)
            if diffs:hi=mid
            else:lo=mid
        print('FIRST DIFFERING COMMAND',hi-1,flush=True)
    finally:host.close()
((PATHS.native / 'build/resident-isolate.json')).write_text(json.dumps(rows,indent=2)+'\n')
