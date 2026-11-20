# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate one native 1x1 command in its live resident graph prefix.

Only the target command may dispatch natively. Its active u8 output permits
the independently proven native-minus-integer bound [0,+1]; every other pool
byte must match the integer graph exactly. This is not full-graph validation.
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

FRAME=(PATHS.native)
sys.path.insert(0,str(FRAME))
import depth_model.lowering.resident_image as resident_image
from depth_model.adapters import ResidentDSP
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs


def output_footprint(tensor,weight,input_tensor,pool_sizes):
    shape=tensor['shape'];dims=shape['dimensions'];maximum=shape['max_dimensions']
    padding=shape['padding'];xm=input_tensor['shape']['max_dimensions'];xp=input_tensor['shape']['padding']
    ci,co=weight['shape']['max_dimensions'][2:]
    padded_ci=(weight.get('class_name')=='Xpb' and weight['shape']['max_dimensions'][:2]==[1,1]
               and xm[3]==32 and ((ci==12 and input_tensor['shape']['dimensions'][3]==9)
                                  or (ci==20 and input_tensor['shape']['dimensions'][3]==18)))
    if tensor['class_name']!='CB' or tensor['dtype']!='uint8' or co!=maximum[3] or (ci%32 and not padded_ci) or co%32:
        raise ValueError('target output is outside the validated u8 1x1 feature set')
    kh,kw=weight['shape']['max_dimensions'][:2]
    oh=min((dims[1]+7)//8*8,maximum[1],xm[1]-xp[1]-kh+1)
    ow=min((dims[2]+7)//8*8,maximum[2],xm[2]-xp[2]-kw+1)
    masks={p:np.zeros(size,dtype=bool) for p,size in pool_sizes.items()}
    storage=tensor['storage'];blocks=storage['blocks'];nbw=(maximum[2]+7)//8;nbd=(maximum[3]+31)//32
    for y in range(oh):
        for x in range(ow):
            for cb in range(co//32):
                ay=y+padding[1];ax=x+padding[2]
                index=((ay//8)*nbw+ax//8)*nbd+storage['slice_offset']+cb
                block=blocks[index];p=block['pool'];offset=block['offset']+(ay%8)*256+(ax%8)*32
                if p not in masks or offset<0 or offset+32>len(masks[p]):
                    raise ValueError('target output footprint exceeds a checked pool')
                masks[p][offset:offset+32]=True
    return masks


def compare_pool(actual,expected,mask):
    actual=np.asarray(actual,dtype=np.uint8);expected=np.asarray(expected,dtype=np.uint8)
    if actual.shape!=expected.shape or actual.shape!=mask.shape:
        raise ValueError('pool comparison lengths differ')
    delta=actual.astype(np.int16)-expected.astype(np.int16)
    bad_outside=int(np.count_nonzero(delta[~mask]))
    values=delta[mask];bad_inside=int(np.count_nonzero((values<0)|(values>1)))
    unique,counts=np.unique(values,return_counts=True)
    return {'equal_outside_target':bad_outside==0,'outside_mismatches':bad_outside,
            'target_bound_0_plus1':bad_inside==0,'target_bad_values':bad_inside,
            'target_bytes':int(mask.sum()),'target_delta_counts':{str(int(k)):int(v) for k,v in zip(unique,counts)},
            'passed':bad_outside==0 and bad_inside==0}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),default='primary')
    ap.add_argument('--sequence',type=int,default=1195)
    ap.add_argument('--native-family',choices=('1x1','3x3'),default='1x1')
    ap.add_argument('--expected-native-calls',type=int,choices=(0,1),default=1,
                    help='0 validates the exact legacy control; 1 validates the first native command')
    ap.add_argument('--case',action='append')
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();image=ensure_image(args.model)
    graph=resident_image.unpack(image.read_bytes());tensors={t['id']:t for t in graph['tensors']}
    matches=[(i,c) for i,c in enumerate(graph['commands']) if c['sequence']==args.sequence]
    if len(matches)!=1:raise ValueError('target sequence must identify one compiled command')
    index,command=matches[0];xt,wt,fi=[tensors[i] for i in command['inputs'][:3]];out=tensors[command['outputs'][0]]
    kernel=1 if args.native_family=='1x1' else 3
    if wt['shape']['max_dimensions'][:2]!=[kernel,kernel]:raise ValueError('target does not match the requested kernel family')
    constants=(PATHS.model_data / 'recovered')/args.model/'constants/constant_pool_5.bin'
    report={'purpose':__doc__,'model':args.model,'sequence':args.sequence,'index':index,
            'native_family':args.native_family,'expected_native_calls':args.expected_native_calls,'runs':[],'passed':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with ResidentDSP(image,constants,engine=args.engine,bootstrap=args.bootstrap) as dsp:
        report.update(engine_sha256=dsp.engine_sha256,preserve_vtcm=dsp.preserve_vtcm,
                      bootstrap_sha256=hashlib.sha256(dsp.bootstrap.read_bytes()).hexdigest())
        if not dsp.preserve_vtcm:raise ValueError('native dispatch requires the preserving bootstrap')
        masks=output_footprint(out,wt,xt,{p:dsp.sizes[p] for p in (1,3,4,5,6)})
        if args.expected_native_calls==0:
            for mask in masks.values():mask[:]=False
        for name,directory,_ in cases(args.model,args.case or ['random'])[1]:
            inputs=load_inputs(directory)
            host=NativeGraph(image,constants)
            try:
                host.use_cpu();host.run(inputs,last=index+1)
                # Prefixes leave some pools untouched. Establish the same
                # zero initial state as NativeGraph before comparing them.
                for p in (1,3,4,6):
                    start=dsp.pools[p];size=dsp.sizes[p]
                    dsp.mem[start:start+size]=bytes(size)
                dsp.run(inputs,last=index+1)
                pools={str(p):compare_pool(dsp.read_pool(p),host.pools[p][:dsp.sizes[p]],masks[p]) for p in masks}
            finally:host.close()
            native=dsp.last.get('native_'+args.native_family,{})
            other=dsp.last.get('native_'+('3x3' if args.native_family=='1x1' else '1x1'),{})
            row={'case':name,**dsp.last,'pools':pools,
                 'passed':native.get('calls',0)==args.expected_native_calls and not other.get('calls',0) and all(r['passed'] for r in pools.values())}
            report['runs'].append(row);args.report.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    report['exit_errors']=dsp.exit_errors;report['vtcm_lease']=dsp.vtcm_lease
    report['passed']=bool(report['runs']) and all(r['passed'] for r in report['runs']) and not any(dsp.exit_errors)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit('Native prefix validation failed; see report')


if __name__=='__main__':main()
