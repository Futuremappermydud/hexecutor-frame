"""Validate prepared elementwise execution using real CPU-prefix state."""

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
import numpy as np

FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
import depth_model.lowering.resident_image as resident_image
from depth_model.adapters import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),default='primary')
    ap.add_argument('--case',action='append')
    ap.add_argument('--sequence',type=int,action='append',help='Select an audited ADD or LinearClip sequence')
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--prepared-program',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();image=ensure_image(a.model)
    graph=resident_image.unpack(image.read_bytes())
    # These are the actual first ADD_U8 v13 and LinearClip v68 records also
    # captured by the independent specialist's SDK fixture generator.
    targets={'primary':(3424,4987),'alternate':(3374,4940)}[a.model]
    if a.sequence:
        if not set(a.sequence)<=set(targets):raise SystemExit('Sequence is outside audited elementwise targets')
        targets=tuple(a.sequence)
    constants=(PATHS.model_data / 'recovered')/a.model/'constants/constant_pool_5.bin'
    report={'purpose':__doc__,'model':a.model,'runs':[],'passed':False,
            'prepared_sha256':hashlib.sha256(a.prepared_program.read_bytes()).hexdigest()}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    def save():a.report.write_text(json.dumps(report,indent=2)+'\n')
    fixtures=[]
    for sequence in targets:
        index,c=next((i,c) for i,c in enumerate(graph['commands']) if c['sequence']==sequence)
        for label,directory,_ in cases(a.model,a.case or ['smoke','random'])[1]:
            inputs=load_inputs(directory);host=NativeGraph(image,constants)
            try:
                host.use_cpu();host.run(inputs,last=index)
                before={p:host.pools[p].copy() for p in (1,3,4,5,6)}
                host.run(inputs,first=index,last=index+1)
                after={p:host.pools[p].copy() for p in before}
            finally:host.close()
            fixtures.append((sequence,index,c['variant'],label,inputs,before,after))
    with ResidentDSP(image,constants,engine=a.engine,bootstrap=a.bootstrap,
                     prepared_program=a.prepared_program) as dsp:
        report['engine_sha256']=dsp.engine_sha256
        for sequence,index,variant,label,inputs,before,after in fixtures:
            for p,values in before.items():
                off=dsp.pools[p];n=dsp.sizes[p];dsp.mem[off:off+n]=values[:n].tobytes()
            dsp.run(inputs,first=index,last=index+1)
            mismatch={str(p):int(np.count_nonzero(dsp.read_pool(p)!=values[:dsp.sizes[p]])) for p,values in after.items()}
            trace=dsp.last.get('native_direct_trace',[])
            row={'sequence':sequence,'index':index,'variant':variant,'case':label,**dsp.last,
                 'pool_mismatches':mismatch}
            row['passed']=(dsp.last.get('native_prepared_program') is True and not any(mismatch.values())
                and len(trace)==16 and trace[8]==1 and trace[9]==1
                and all(dsp.last['native_'+f]['calls']==0 for f in ('1x1','3x3')))
            report['runs'].append(row);save();print(json.dumps(row),flush=True)
    report.update(exit_errors=dsp.exit_errors,vtcm_lease=dsp.vtcm_lease)
    report['passed']=(all(r['passed'] for r in report['runs']) and not any(dsp.exit_errors)
        and dsp.vtcm_lease['phase']==2 and not dsp.vtcm_lease['error'] and dsp.vtcm_lease['host_fingerprint_equal'])
    save()
    if not report['passed']:raise SystemExit('Prepared elementwise Frame validation failed')


if __name__=='__main__':main()
