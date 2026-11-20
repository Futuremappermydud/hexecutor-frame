# SPDX-License-Identifier: LGPL-3.0-or-later
"""Measure the experimental native graph; report numerical differences explicitly.

Finite, repeatable output and clean memory recovery do not establish numerical
equivalence to Qualcomm or the integer graph. This records those separately.
"""

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
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from depth_model.adapters import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs


def difference(actual,expected):
    if actual.shape!=expected.shape:raise ValueError('reference output shape mismatch')
    delta=actual.astype(np.float64)-expected.astype(np.float64)
    finite=bool(np.isfinite(actual).all())
    return {'finite':finite,'equal':bool(np.array_equal(actual,expected)),
            'different':int(np.count_nonzero(delta)),
            'mae':float(np.mean(np.abs(delta))) if finite else None,
            'max_abs':float(np.max(np.abs(delta))) if finite else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),action='append')
    ap.add_argument('--case',action='append')
    ap.add_argument('--repeat',type=int,default=2)
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if args.repeat<2:raise ValueError('at least two calls per case are required for determinism')
    report={'purpose':__doc__,'numerical_equivalence_claim':False,'models':{},'completed':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
    for model in args.model or ('primary','alternate'):
        image=ensure_image(model);constants=(PATHS.model_data / 'recovered')/model/'constants/constant_pool_5.bin'
        references=[]
        # Compute references before installing or starting any DSP callback.
        for label,directory,_ in cases(model,args.case or ['smoke','random'])[1]:
            inputs=load_inputs(directory);host=NativeGraph(image,constants)
            try:
                host.use_cpu();expected=[a.copy() for a in host.run(inputs)]
            finally:host.close()
            references.append((label,inputs,expected))
        row={'runs':[],'bootstrap_sha256':hashlib.sha256(args.bootstrap.read_bytes()).hexdigest()}
        report['models'][model]=row;dsp=ResidentDSP(image,constants,engine=args.engine,bootstrap=args.bootstrap)
        try:
            with dsp:
                row.update(engine_sha256=dsp.engine_sha256,preserve_vtcm=dsp.preserve_vtcm)
                for label,inputs,expected in references:
                    first_hash=None
                    for iteration in range(args.repeat):
                        outputs=dsp.run(inputs)
                        hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in outputs]
                        if first_hash is None:first_hash=hashes
                        run={'case':label,'iteration':iteration,**dsp.last,'output_sha256':hashes,
                             'repeat_equal':hashes==first_hash,
                             'integer_reference':[difference(a,b) for a,b in zip(outputs,expected)]}
                        row['runs'].append(run);print(model,json.dumps(run),flush=True);save()
                    row['profile']=sorted(dsp.profile,key=lambda x:-x['ms'])
        finally:
            row['exit_errors']=getattr(dsp,'exit_errors',None)
            row['vtcm_lease']=getattr(dsp,'vtcm_lease',None);save()
    report['completed']=all(m['exit_errors']==[0,0,0] and m['vtcm_lease'] and
        m['vtcm_lease']['phase']==2 and m['vtcm_lease']['error']==0 and
        all(r['repeat_equal'] and all(h['finite'] for h in r['integer_reference'])
            for r in m['runs']) for m in report['models'].values())
    save()
    if not report['completed']:raise SystemExit('native graph characterization did not complete cleanly')


if __name__=='__main__':main()
