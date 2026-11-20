# SPDX-License-Identifier: LGPL-3.0-or-later
"""Same-session A/B/A comparison of legacy and fused HMX activation staging."""

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

import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from depth_model.adapters import ResidentDSP
from verify_native_graph import cases,ensure_image,load_inputs
from verify_resident_decoder import snapshot
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--repeat',type=int,default=2);ap.add_argument('--model',choices=('primary','alternate'),action='append');args=ap.parse_args()
report={'before':snapshot(),'models':{}}
def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
for model in args.model or ('primary','alternate'):
    names,todo=cases(model,['random']);inputs=load_inputs(todo[0][1]);expected=todo[0][2];rows=[]
    constants=(PATHS.model_data / 'recovered')/model/'constants/constant_pool_5.bin'
    with ResidentDSP(ensure_image(model),constants) as dsp:
        report['models'][model]={'runs':rows,'engine_sha256':dsp.engine_sha256}
        for label,direct,reps in [('cold',False,1),('A',False,args.repeat),('B',True,args.repeat),('A2',False,args.repeat)]:
            for _ in range(reps):
                outputs=dsp.run(inputs,direct_hmx=direct)
                hashes={n+'.raw':hashlib.sha256(outputs[i].tobytes()).hexdigest() for i,n in enumerate(names)}
                row={'phase':label,**dsp.last,'equal':hashes==expected,'hashes':hashes,'expected':expected};rows.append(row);print(model,json.dumps(row),flush=True);save()

    report['models'][model]['after']=snapshot();save()
report['all_exact']=all(r['equal'] for m in report['models'].values() for r in m['runs']);save()
print('ALL EXACT' if report['all_exact'] else 'EXPERIMENT FAILED EXACTNESS',flush=True)
raise SystemExit(0 if report['all_exact'] else 1)
