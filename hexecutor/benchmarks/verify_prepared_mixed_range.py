# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare a bounded prepared command interval with the CPU reference on Frame."""

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

import argparse
import hashlib
import json
from pathlib import Path
import sys

FRAME=(PATHS.native)
sys.path.insert(0,str(FRAME))
import depth_model.lowering.resident_image as resident_image
from depth_model.adapters import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),required=True)
    ap.add_argument('--first',type=int,required=True)
    ap.add_argument('--last',type=int,required=True)
    ap.add_argument('--case',action='append')
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--prepared-program',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args()
    image=ensure_image(a.model)
    graph=resident_image.unpack(image.read_bytes())
    if not 0<=a.first<a.last<=len(graph['commands']):raise ValueError('invalid command range')
    constants=(PATHS.model_data / 'recovered')/a.model/'constants/constant_pool_5.bin'
    report={'purpose':__doc__,'model':a.model,'first':a.first,'last':a.last,'runs':[],
            'passed':False,'engine_sha256':hashlib.sha256(a.engine.read_bytes()).hexdigest(),
            'bootstrap_sha256':hashlib.sha256(a.bootstrap.read_bytes()).hexdigest(),
            'prepared_sha256':hashlib.sha256(a.prepared_program.read_bytes()).hexdigest()}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    def save():a.report.write_text(json.dumps(report,indent=2)+'\n')
    fixtures=[]
    for label,directory,_ in cases(a.model,a.case or ['smoke','random'])[1]:
        inputs=load_inputs(directory)
        host=NativeGraph(image,constants)
        try:
            host.use_cpu();host.run(inputs,last=a.first)
            before={p:host.pools[p].copy() for p in (1,3,4,5,6)}
            host.run(inputs,first=a.first,last=a.last)
            after={p:host.pools[p].copy() for p in before}
        finally:host.close()
        fixtures.append((label,inputs,before,after))
    with ResidentDSP(image,constants,engine=a.engine,bootstrap=a.bootstrap,
                     prepared_program=a.prepared_program) as dsp:
        if not dsp.preserve_vtcm:raise ValueError('preserving bootstrap required')
        for label,inputs,before,after in fixtures:
            for p,values in before.items():
                off=dsp.pools[p];n=dsp.sizes[p]
                dsp.mem[off:off+n]=values[:n].tobytes()
            dsp.run(inputs,first=a.first,last=a.last)
            mismatch={str(p):int((dsp.read_pool(p)!=after[p][:dsp.sizes[p]]).sum()) for p in before}
            trace=dsp.last.get('native_direct_trace',[])
            row={'case':label,**dsp.last,'pool_mismatches':mismatch}
            row['passed']=(dsp.last.get('status')==0 and
                           dsp.last.get('native_prepared_program') is True and
                           not any(mismatch.values()) and len(trace)==16 and
                           trace[8]==a.last-a.first and trace[9]==a.last-a.first and
                           all(dsp.last.get('native_'+f,{}).get('calls')==0 for f in ('1x1','3x3')))
            report['runs'].append(row);save()
            print(json.dumps({'case':label,'passed':row['passed'],'pool_mismatches':mismatch,
                              'dsp_ms':row.get('dsp_ms')}),flush=True)
    report['exit_errors']=dsp.exit_errors;report['vtcm_lease']=dsp.vtcm_lease
    report['passed']=(bool(report['runs']) and all(r['passed'] for r in report['runs'])
                      and not any(dsp.exit_errors) and dsp.vtcm_lease['phase']==2
                      and not dsp.vtcm_lease['error'] and dsp.vtcm_lease['host_fingerprint_equal'])
    save()
    if not report['passed']:raise SystemExit('Prepared mixed range failed; see report')


if __name__=='__main__':main()
