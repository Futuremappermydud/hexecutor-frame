"""Exercise cancellation, invalid allocation contracts, and fresh sessions."""
import argparse,hashlib,json,os,struct,sys,threading
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
FRAME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(FRAME))
from resident_dsp import ResidentDSP
from verify_native_graph import cases,ensure_image,load_inputs
from verify_resident_decoder import snapshot
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
report={'before':snapshot(),'sessions':[]}
def save():args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
def expect(dsp,model,case):
    names,todo=cases(model,[case]);_,directory,want=todo[0]
    outputs=dsp.run(load_inputs(directory))
    got={n+'.raw':hashlib.sha256(outputs[i].tobytes()).hexdigest() for i,n in enumerate(names)}
    if got!=want:raise RuntimeError(f'{model}/{case} mismatch')
    return {'case':case,**dsp.last,'equal':True}
for model in ('primary','alternate'):
    rows=[];constants=FRAME.parent/'recovered'/model/'constants/constant_pool_5.bin'
    with ResidentDSP(ensure_image(model),constants) as dsp:
        rows.append(expect(dsp,model,'smoke'));heap=dsp.last['heap_used']
        _,todo=cases(model,['random']);inputs=load_inputs(todo[0][1])
        timer=threading.Timer(.05,dsp.cancel);timer.start()
        try:
            try:dsp.run(inputs)
            except RuntimeError:
                if dsp.last['status']!=12:raise
            else:raise RuntimeError('cancellation was not observed')
        finally:timer.cancel();timer.join()
        rows.append({'case':'cancel','observed':True,**dsp.last})
        dsp.check_guards()
        rows.append(expect(dsp,model,'random'));rows.append(expect(dsp,model,'smoke'))
        if dsp.last['heap_used']!=heap:raise RuntimeError('heap grew after cancellation')
        sha=dsp.engine_sha256
    report['sessions'].append({'model':model,'engine_sha256':sha,'runs':rows,'after':snapshot()});save()
# Each invalid contract is rejected before heap/graph writes, then tears down.
constants=FRAME.parent/'recovered/primary/constants/constant_pool_5.bin'
for label,index,value in (('heap-overlap',30,0x1000000),('input-output-overlap',18,None),('output-overflow',28,0xffffffff)):
    with ResidentDSP(ensure_image('primary'),constants) as dsp:
        if value is None:value=dsp.outputs[0]
        struct.pack_into('<I',dsp.transport.arena,4096+index*4,value)
        _,todo=cases('primary',['smoke'])
        try:dsp.run(load_inputs(todo[0][1]))
        except RuntimeError:
            if dsp.last['status']!=1:raise
        else:raise RuntimeError('invalid contract was accepted')
        dsp.check_guards();row={'case':label,'rejected':True,**dsp.last}
    report['sessions'].append({'runs':[row],'after':snapshot()});save()
with ResidentDSP(ensure_image('primary'),constants) as dsp:row=expect(dsp,'primary','random')
report['sessions'].append({'runs':[row],'after':snapshot()});report['passed']=True;save();print('ALL PASS',flush=True)
