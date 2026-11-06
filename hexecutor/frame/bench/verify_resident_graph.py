"""Run the actual graph on DSP; compare hashes or a prefix's pools with native C."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
FRAME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(FRAME))
from resident_dsp import ResidentDSP
from verify_native_graph import cases,ensure_image,load_inputs,NativeGraph

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',choices=('primary','alternate'),action='append')
ap.add_argument('--case',action='append')
ap.add_argument('--prefix',type=int)
ap.add_argument('--engine',type=Path)
ap.add_argument('--bootstrap',type=Path,help='explicit bootstrap paired with the experimental engine/cache mode')
ap.add_argument('--direct-hmx',action='store_true')
ap.add_argument('--repeat',type=int,default=2)
ap.add_argument('--report',type=Path,required=True)
args=ap.parse_args();report={}
for model in args.model or ('primary','alternate'):
    constants=FRAME.parent/'recovered'/model/'constants/constant_pool_5.bin'
    names,todo=cases(model,args.case)
    rows=[];report[model]={'runs':rows}
    with ResidentDSP(ensure_image(model),constants,engine=args.engine,direct_hmx=args.direct_hmx,bootstrap=args.bootstrap) as dsp:
        report[model].update(engine=str(dsp.engine),engine_sha256=dsp.engine_sha256,
                            bootstrap=str(dsp.bootstrap),bootstrap_sha256=hashlib.sha256(dsp.bootstrap.read_bytes()).hexdigest(),
                            preserve_vtcm=dsp.preserve_vtcm)
        for case,directory,expected in todo:
            inputs=load_inputs(directory)
            outs=dsp.run(inputs,last=args.prefix)
            row={'case':case,**dsp.last}
            if args.prefix is not None:
                host=NativeGraph(ensure_image(model),constants)
                try:
                    host.run(inputs,last=args.prefix)
                    row['pools']={p:bool((dsp.read_pool(p)==host.pools[p][:dsp.sizes[p]]).all()) for p in (1,3,4,5,6)}
                    row['equal']=all(row['pools'].values())
                finally:host.close()
            else:
                hashes={n+'.raw':hashlib.sha256(outs[i].tobytes()).hexdigest() for i,n in enumerate(names)}
                row['equal']=hashes==expected;row['hashes']=hashes
            rows.append(row);print(model,json.dumps(row),flush=True)
            args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
            if not row['equal']:raise RuntimeError('resident output mismatch')
        for _ in range(args.repeat):
            smoke=cases(model,['smoke'])[1][0];inputs=load_inputs(smoke[1]);outs=dsp.run(inputs,last=args.prefix)
            hashes={n+'.raw':hashlib.sha256(outs[i].tobytes()).hexdigest() for i,n in enumerate(names)}
            row={'case':'warm',**dsp.last,'equal':hashes==smoke[2] if args.prefix is None else True}
            rows.append(row);print(model,json.dumps(row),flush=True)
            args.report.write_text(json.dumps(report,indent=2)+'\n')
            if not row['equal']:raise RuntimeError('resident warm mismatch')
        report[model]['profile']=sorted(dsp.profile,key=lambda r:-r['ms'])
    report[model]['exit_errors']=dsp.exit_errors
    if hasattr(dsp,'vtcm_lease'):report[model]['vtcm_lease']=dsp.vtcm_lease
    args.report.write_text(json.dumps(report,indent=2)+'\n')
print('ALL PASS',flush=True)
