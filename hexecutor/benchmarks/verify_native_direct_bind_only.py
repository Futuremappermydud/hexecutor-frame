"""Inspect direct bindings and operands on Frame without issuing a convolution."""

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
import math
from pathlib import Path
import struct
import sys
import numpy as np

FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
import depth_model.lowering.resident_image as resident_image
from depth_model.adapters import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',type=int,default=247)
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args()
    if b'QD_BIND_ONLY_1\0' not in a.engine.read_bytes():
        raise ValueError('requires the marked bind-only engine; convolution engines are prohibited')
    image=ensure_image('primary');graph=resident_image.unpack(image.read_bytes());ts={t['id']:t for t in graph['tensors']}
    index,c=next((i,c) for i,c in enumerate(graph['commands']) if c['sequence']==a.sequence)
    x,w,f=[ts[i] for i in c['inputs'][:3]];y=ts[c['outputs'][0]]
    constants=(PATHS.model_data / 'recovered/primary/constants/constant_pool_5.bin')
    _,directory,_=cases('primary',['random'])[1][0];inputs=load_inputs(directory)
    host=NativeGraph(image,constants)
    try:
        host.use_cpu();host.run(inputs,last=index)
        before={p:host.pools[p].copy() for p in (1,3,4,5,6)}
    finally:host.close()
    def selected(t):
        m=t['shape']['max_dimensions'];n=(m[1]//8)*(m[2]//8)*(m[3]//32)
        start=t['storage']['slice_offset'];return t['storage']['blocks'][start:start+n]
    xb,yb=selected(x),selected(y)
    expected_in=b''.join(before[b['pool']][b['offset']:b['offset']+2048].tobytes() for b in xb)
    def direct_bytes(t,esize):
        d=t['storage']['direct'];n=math.prod(t['shape']['max_dimensions'])*esize
        return before[d['pool']][d['offset']:d['offset']+n].tobytes()
    ew,ef=direct_bytes(w,1),direct_bytes(f,4)
    it=[0x31000000+b['offset'] for b in xb]+[0x31780800]*(256-len(xb))
    ot=[0x31000000+b['offset'] for b in yb]+[0x31780000]*(256-len(yb))
    report={'purpose':__doc__,'sequence':a.sequence,'passed':False}
    with ResidentDSP(image,constants,engine=a.engine,bootstrap=a.bootstrap) as dsp:
        report['engine_sha256']=dsp.engine_sha256
        for p,v in before.items():
            off=dsp.pools[p];n=dsp.sizes[p];dsp.mem[off:off+n]=v[:n].tobytes()
        dsp.run(inputs,first=index,last=index+1)
        echo=struct.unpack_from('<1102I',dsp.transport.arena,0x5000)
        if echo[:7]!=(0x31424451,3,index,len(xb),len(yb),len(ew),len(ef)):
            raise RuntimeError('unexpected binding echo header')
        tables_before=list(echo[16:528]);tables_after=list(echo[560:1072])
        abi_before=list(echo[528:558]);abi_after=list(echo[1072:1102])
        gotw=bytes(dsp.transport.arena[0x8000:0x8000+len(ew)])
        gotf=bytes(dsp.transport.arena[0x8000+len(ew):0x8000+len(ew)+len(ef)])
        gotx=bytes(dsp.transport.arena[0x20000:0x20000+len(expected_in)])
        xm=x['shape']['max_dimensions'];xp=x['shape']['padding'];xd=x['shape']['dimensions']
        ym=y['shape']['max_dimensions'];yd=y['shape']['dimensions']
        desc=[0x30000000,xm[3]//32,(xm[2]//8)*(xm[3]//32),xd[2]+xp[2],xd[1]+xp[1],xm[3],
              0x30000400,ym[3]//32,(ym[2]//8)*(ym[3]//32),yd[2],yd[1],ym[3]]
        checks={'tables_before_match':tables_before==it+ot,'tables_after_match':tables_after==it+ot,
            'abi_unchanged':abi_before==abi_after,'weights_exact':gotw==ew,'fi_exact':gotf==ef,
            'descriptors_match':abi_before[:12]==desc,'generic_mode':bool(abi_before[24]&8) and abi_before[28:]==[1,0],
            'input_pages_exact':gotx==expected_in,
            'all_graph_pools_unchanged':all(np.array_equal(dsp.read_pool(p),v[:dsp.sizes[p]]) for p,v in before.items()),
            'no_native_calls':all(dsp.last['native_'+k]['calls']==0 for k in ('1x1','3x3'))}
        report.update(run=dsp.last,checks=checks,abi_before=abi_before,abi_after=abi_after,
            table_changes=[i for i,(b,c) in enumerate(zip(tables_before,tables_after)) if b!=c],
            data_sha256={'input':hashlib.sha256(gotx).hexdigest(),'weights':hashlib.sha256(gotw).hexdigest(),'fi':hashlib.sha256(gotf).hexdigest()})
    report.update(exit_errors=dsp.exit_errors,vtcm_lease=dsp.vtcm_lease)
    report['passed']=(all(report['checks'].values()) and not any(dsp.exit_errors)
        and dsp.vtcm_lease['phase']==2 and not dsp.vtcm_lease['error'] and dsp.vtcm_lease['host_fingerprint_equal'])
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit('Binding-only operand verification failed')


if __name__=='__main__':main()
