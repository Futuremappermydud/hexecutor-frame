# SPDX-License-Identifier: LGPL-3.0-or-later
"""Frame bench for prepared InputSlice/Im2col against NativeGraph CPU prefixes."""

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

import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=(PATHS.repo);sys.path.insert(0,str(ROOT));sys.path.insert(0,str((PATHS.native)));sys.path.insert(0,str((PATHS.benchmarks)))
import depth_model.lowering.resident_image as resident_image
from depth_model.adapters import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs

def targets_for(model,graph):
    out=[]
    for wanted,name in ((3,'*InputSlice@Ff.s4*6.'),(5,'ConvLayer.opt.inconv_im2col.tcm@CB.FB.s4.t')):
        matches=[(i,c) for i,c in enumerate(graph['commands']) if c['opcode']==wanted and c['variant']==name]
        if not matches:raise ValueError(f'missing preprocess command {name} in {model}')
        out.append(matches[0])
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',choices=('primary','alternate'),default='primary');ap.add_argument('--case',action='append');ap.add_argument('--engine',type=Path,required=True);ap.add_argument('--bootstrap',type=Path,required=True);ap.add_argument('--prepared-program',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    image=ensure_image(a.model);graph=resident_image.unpack(image.read_bytes());targets=targets_for(a.model,graph);constants=(PATHS.model_data / 'recovered')/a.model/'constants/constant_pool_5.bin';report={'purpose':__doc__,'model':a.model,'runs':[],'passed':False,'prepared_sha256':hashlib.sha256(a.prepared_program.read_bytes()).hexdigest()};a.report.parent.mkdir(parents=True,exist_ok=True)
    def save():a.report.write_text(json.dumps(report,indent=2)+'\n')
    cases_to_run=[]
    for index,c in targets:
        for label,directory,_ in cases(a.model,a.case or ['smoke','random'])[1]:
            inputs=load_inputs(directory);host=NativeGraph(image,constants)
            try:
                host.use_cpu();host.set_hoist(False);host.set_block_output(False);host.run(inputs,last=index)
                before={p:host.pools[p].copy() for p in (1,3,4,5,6)};host.run(inputs,first=index,last=index+1);after={p:host.pools[p].copy() for p in before}
            finally:host.close()
            cases_to_run.append((index,c['sequence'],c['variant'],label,inputs,before,after))
    with ResidentDSP(image,constants,engine=a.engine,bootstrap=a.bootstrap,prepared_program=a.prepared_program) as dsp:
        report['engine_sha256']=dsp.engine_sha256
        for index,seq,variant,label,inputs,before,after in cases_to_run:
            for p,values in before.items():off=dsp.pools[p];dsp.mem[off:off+dsp.sizes[p]]=values[:dsp.sizes[p]].tobytes()
            dsp.run(inputs,first=index,last=index+1)
            mismatch={str(p):int(np.count_nonzero(dsp.read_pool(p)!=values[:dsp.sizes[p]])) for p,values in after.items()}
            row={'index':index,'sequence':seq,'variant':variant,'case':label,**dsp.last,'pool_mismatches':mismatch}
            trace=dsp.last.get('native_direct_trace',[])
            row['passed']=(len(trace)==16 and trace[8:10]==[1,1] and all(dsp.last.get('native_'+f,{}).get('calls')==0 for f in ('1x1','3x3')) and dsp.last.get('native_prepared_program') is True and not any(mismatch.values()) and dsp.last.get('hmx_calls',0)==0 and dsp.last.get('direct_hmx_calls',0)==0)
            report['runs'].append(row);save();print(json.dumps(row),flush=True)
    report.update(exit_errors=dsp.exit_errors,vtcm_lease=dsp.vtcm_lease);report['passed']=(all(r['passed'] for r in report['runs']) and not any(dsp.exit_errors) and dsp.vtcm_lease['phase']==2 and not dsp.vtcm_lease['error'] and dsp.vtcm_lease['host_fingerprint_equal']);save()
    if not report['passed']:raise SystemExit('Prepared preprocess Frame validation failed')
if __name__=='__main__':main()
