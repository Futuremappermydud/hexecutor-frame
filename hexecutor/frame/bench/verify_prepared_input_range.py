"""Check a prepared InputSlicePad range against captured CPU-prefix pools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

FRAME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FRAME))
import resident_image
from resident_dsp import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),default='primary')
    ap.add_argument('--case',action='append')
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--prepared-program',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();image=ensure_image(a.model)
    graph=resident_image.unpack(image.read_bytes())
    first=next(i for i,c in enumerate(graph['commands']) if c['variant'].startswith('*InputSlicePad'))
    last=first
    while last<len(graph['commands']) and graph['commands'][last]['variant'].startswith('*InputSlicePad'):last+=1
    constants=FRAME.parent/'recovered'/a.model/'constants/constant_pool_5.bin'
    report={'purpose':__doc__,'model':a.model,'first':first,'last':last,'runs':[],'passed':False,
            'prepared_sha256':hashlib.sha256(a.prepared_program.read_bytes()).hexdigest()}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    def save():a.report.write_text(json.dumps(report,indent=2)+'\n')
    fixtures=[]
    for label,directory,_ in cases(a.model,a.case or ['smoke','random'])[1]:
        inputs=load_inputs(directory);host=NativeGraph(image,constants)
        try:
            host.use_cpu();host.run(inputs,last=first)
            before={p:host.pools[p].copy() for p in (1,3,4,5,6)}
            host.run(inputs,first=first,last=last)
            after={p:host.pools[p].copy() for p in before}
        finally:host.close()
        fixtures.append((label,inputs,before,after))
    with ResidentDSP(image,constants,engine=a.engine,bootstrap=a.bootstrap,
                     prepared_program=a.prepared_program) as dsp:
        report['engine_sha256']=dsp.engine_sha256
        for label,inputs,before,after in fixtures:
            for repeat in range(2):
                for p,values in before.items():
                    off=dsp.pools[p];n=dsp.sizes[p];dsp.mem[off:off+n]=values[:n].tobytes()
                if 'rejection' not in report:
                    try:dsp.run(inputs,first=first,last=dsp.commands)
                    except RuntimeError:
                        if dsp.last.get('status')!=6:raise
                    else:raise RuntimeError('unsupported suffix unexpectedly executed')
                    unchanged=all((dsp.read_pool(p)==values[:dsp.sizes[p]]).all() for p,values in before.items())
                    report['rejection']={**dsp.last,'all_pools_unchanged':bool(unchanged)}
                    if not unchanged:raise RuntimeError('preflight rejection wrote graph pools')
                dsp.run(inputs,first=first,last=last)
                mismatch={str(p):int((dsp.read_pool(p)!=values[:dsp.sizes[p]]).sum()) for p,values in after.items()}
                trace=dsp.last.get('native_direct_trace',[])
                row={'case':label,'repeat':repeat,**dsp.last,'pool_mismatches':mismatch}
                row['passed']=(dsp.last.get('native_prepared_program') is True and not any(mismatch.values())
                    and len(trace)==16 and trace[8]==last-first and trace[9]==last-first
                    and all(dsp.last.get('native_'+f,{}).get('calls')==0 for f in ('1x1','3x3')))
                report['runs'].append(row);save();print(json.dumps(row),flush=True)
    report['exit_errors']=dsp.exit_errors;report['vtcm_lease']=dsp.vtcm_lease
    report['passed']=(all(r['passed'] for r in report['runs']) and not any(dsp.exit_errors)
        and dsp.vtcm_lease['phase']==2 and not dsp.vtcm_lease['error']
        and dsp.vtcm_lease['host_fingerprint_equal'])
    save()
    if not report['passed']:raise SystemExit('Prepared input range validation failed')


if __name__=='__main__':main()
