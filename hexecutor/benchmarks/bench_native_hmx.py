# SPDX-License-Identifier: LGPL-3.0-or-later
"""A/B/A native integer HMX timing with final-head differences against exact v6."""

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

import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport
from verify_native_graph import cases,ensure_image,load_inputs,NativeGraph
from verify_resident_decoder import snapshot
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',choices=('primary','alternate'),action='append');ap.add_argument('--case',action='append')
ap.add_argument('--all-cases',action='store_true')
ap.add_argument('--native-cm',action='store_true',help='also compare hardware crouton load/store')
ap.add_argument('--repeat',type=int,default=3);ap.add_argument('--block-output',action='store_true')
ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{7})
payload=(PATHS.native / 'build/qd_conv_native.elf')
report={'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),'block_output':args.block_output,'native_cm':args.native_cm,'host_source_sha256':hashlib.sha256(((PATHS.native / 'qd_graph.c')).read_bytes()).hexdigest(),'before':snapshot(),'models':{}}
def save():args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
def stats(graph):
    v=graph.stats()
    return {'dsp_calls':v.calls,'hmx_calls':v.hmx_calls,'prepare_ms':v.prepare_ns/1e6,'wait_ms':v.wait_ns/1e6,
            'collect_ms':v.collect_ns/1e6,'conv_ms':v.conv_ns/1e6,'hmx_phases_ms':[n/1843200 for n in v.hmx_phase_cycles],
            'bytes_uploaded':v.bytes_uploaded,'weights_reused':v.weights_reused,
            'profile':sorted([{'variant':i,'ms':v.variant_ns[i]/1e6,'count':v.variant_count[i]} for i in range(80) if v.variant_count[i]],key=lambda r:-r['ms'])}
def differences(got,want):
    result=[]
    for a,b in zip(got,want):
        d=a.astype(np.float64)-b.astype(np.float64);finite=np.isfinite(d);v=np.abs(d[finite]);r=b.astype(np.float64)
        result.append({'dtype':str(a.dtype),'values':a.size,'different':int(np.count_nonzero(a!=b)),
                       'nonfinite':int(np.count_nonzero(~finite)),'mean_abs':float(v.mean()) if v.size else None,
                       'p95_abs':float(np.quantile(v,.95)) if v.size else None,'max_abs':float(v.max()) if v.size else None,
                       'rmse':float(np.sqrt(np.mean(d[finite]**2))) if v.size else None})
    return result
for model in args.model or ('primary','alternate'):
    names,todo=cases(model,None if args.all_cases else args.case or ['smoke','random']);rows=[]
    graph=NativeGraph(ensure_image(model),(PATHS.model_data / 'recovered')/model/'constants/constant_pool_5.bin')
    try:
        with FrameTransport((PATHS.native / 'build/qd_loader.elf'),payload,15,raise_clock=True,use_hmx=True) as transport:
            graph.set_helpers([6,5,4]);graph.use_dsp(transport,timeout=15);graph.set_profile(True);graph.set_block_output(args.block_output)
            report['models'][model]={'runs':rows}
            for case,directory,expected in todo:
                inputs=load_inputs(directory);graph.set_native_hmx(False);graph.reset_stats()
                begin=time.monotonic();reference=[v.copy() for v in graph.run(inputs)];elapsed=time.monotonic()-begin
                hashes={n+'.raw':hashlib.sha256(reference[i].tobytes()).hexdigest() for i,n in enumerate(names)}
                if hashes!=expected:raise RuntimeError(f'exact backend mismatch {model}/{case}')
                rows.append({'case':case,'mode':'exact-reference','seconds':elapsed,'equal':True,'stats':stats(graph)});save()
                native_reference=None
                phases=[('A',0),('B-native',1)]+([('C-native-cm',2)] if args.native_cm else [])+[('A2',0)]
                for phase,enabled in phases:
                    graph.set_native_hmx(enabled)
                    for iteration in range(args.repeat):
                        graph.reset_stats();begin=time.monotonic();got=graph.run(inputs);elapsed=time.monotonic()-begin
                        metrics=differences(got,reference);equal=all(m['different']==0 for m in metrics)
                        native_equal=None
                        if enabled==1:native_reference=[v.copy() for v in got]
                        if enabled==2:
                            native_equal=all(np.array_equal(a,b) for a,b in zip(got,native_reference))
                            if not native_equal:raise RuntimeError('crouton mode changed native mathematical outputs')
                        row={'case':case,'phase':phase,'iteration':iteration,'native_hmx':enabled,'seconds':elapsed,
                             'equal':equal,'equal_native_plain':native_equal,'hashes':[hashlib.sha256(v.tobytes()).hexdigest() for v in got],'difference':metrics,'stats':stats(graph)};rows.append(row)
                        print(model,json.dumps(row),flush=True);save()
                        if not enabled and not equal:raise RuntimeError('exact mode did not recover after native mode')
        report['models'][model]['after']=snapshot();save()
    finally:graph.close()
report['completed']=True;save();print('BENCHMARK COMPLETE',flush=True)
