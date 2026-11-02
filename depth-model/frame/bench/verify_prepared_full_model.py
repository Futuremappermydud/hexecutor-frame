"""Run every prepared command of one model on Frame and check both outputs.

Each case runs once from zeroed mutable pools and then again without any pool
reset, the way a persistent depth worker calls it. Both runs must reproduce the
emulator output sha256 values; lease restoration and guards are checked on exit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

FRAME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FRAME))
import resident_image
from resident_dsp import ResidentDSP
from verify_native_graph import cases,ensure_image,load_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),required=True)
    ap.add_argument('--case',action='append')
    ap.add_argument('--repeat',type=int,default=2)
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--prepared-program',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args()
    image=ensure_image(a.model)
    commands=len(resident_image.unpack(image.read_bytes())['commands'])
    constants=FRAME.parent/'recovered'/a.model/'constants/constant_pool_5.bin'
    names,todo=cases(a.model,a.case or ['smoke','random'])
    report={'purpose':__doc__,'model':a.model,'commands':commands,'runs':[],'passed':False,
            'engine_sha256':hashlib.sha256(a.engine.read_bytes()).hexdigest(),
            'bootstrap_sha256':hashlib.sha256(a.bootstrap.read_bytes()).hexdigest(),
            'prepared_sha256':hashlib.sha256(a.prepared_program.read_bytes()).hexdigest()}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    def save():a.report.write_text(json.dumps(report,indent=2)+'\n')
    with ResidentDSP(image,constants,engine=a.engine,bootstrap=a.bootstrap,
                     prepared_program=a.prepared_program) as dsp:
        if not dsp.preserve_vtcm:raise ValueError('preserving bootstrap required')
        for label,directory,expected in todo:
            inputs=load_inputs(directory)
            for p in (0,1,2,3,4,6):
                if dsp.sizes[p]:dsp.mem[dsp.pools[p]:dsp.pools[p]+dsp.sizes[p]]=bytes(dsp.sizes[p])
            for run in range(a.repeat):
                outs=dsp.run(inputs)
                hashes={n+'.raw':hashlib.sha256(outs[i].tobytes()).hexdigest() for i,n in enumerate(names)}
                trace=dsp.last.get('native_direct_trace',[])
                row={'case':label,'run':run,'pools_zeroed':run==0,**dsp.last,'output_sha256':hashes,
                     'outputs_equal':hashes==expected,
                     'variant_profile':sorted(dsp.profile,key=lambda v:-v['ms']),
                     'handler_profile':dsp.handler_profile,'executor_profile':dsp.executor_profile}
                # The prepared reference engine reuses the first four conv phase slots.
                phases=list(dsp.last.get('conv_phases_ms',{}).values())[:4]
                if len(phases)==4:
                    row['near_conv_phases_ms']=dict(zip(('staging','mac','requant','output'),phases))
                row['passed']=(dsp.last.get('status')==0 and row['outputs_equal'] and
                               dsp.last.get('native_prepared_program') is True and len(trace)==16 and
                               trace[8]==commands and trace[9]==commands and
                               all(dsp.last.get('native_'+f,{}).get('calls')==0 for f in ('1x1','3x3')))
                report['runs'].append(row);save()
                print(json.dumps({'case':label,'run':run,'passed':row['passed'],'outputs_equal':row['outputs_equal'],
                                  'dsp_ms':row.get('dsp_ms'),'total_seconds':row.get('total_seconds'),
                                  'near_conv_phases_ms':row.get('near_conv_phases_ms'),
                                  'top_variants_ms':{v['variant']:round(v['ms'],1) for v in row['variant_profile'][:6]},
                                  'top_handlers_ms':{f"{h['opcode']}/{h['variant']}":[h['count'],round(h['run_ms'],1),round(h['validate_ms'],1)]
                                                     for h in row['handler_profile'][:8]},
                                  'executor_profile_ms':{k:(round(v,2) if isinstance(v,float) else v) for k,v in row['executor_profile'].items()}}),
                      flush=True)
    report['exit_errors']=dsp.exit_errors;report['vtcm_lease']=dsp.vtcm_lease
    report['passed']=(bool(report['runs']) and all(r['passed'] for r in report['runs'])
                      and not any(dsp.exit_errors) and dsp.vtcm_lease['phase']==2
                      and not dsp.vtcm_lease['error'] and dsp.vtcm_lease['host_fingerprint_equal'])
    save()
    if not report['passed']:raise SystemExit('Prepared full model failed; see report')


if __name__=='__main__':main()
