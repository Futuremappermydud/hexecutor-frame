"""Validate a CPU-seeded, native-only VTCM command range on Frame.

This proves one direct convolution at a time. Boundary pool transfers are
reported separately; this is not whole-model memory-resident inference.
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
from verify_native_graph import NativeGraph,cases,ensure_image,load_inputs
from verify_native_resident_prefix import compare_pool,output_footprint


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('primary','alternate'),default='primary')
    ap.add_argument('--sequence',type=int,action='append',required=True)
    ap.add_argument('--case',action='append')
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--prepared-program',type=Path,
                    help='execute the range using the prepared NPRG executor instead of QDRX dispatch')
    ap.add_argument('--staged-control',action='store_true',
                    help='validate a frozen staged engine using the same CPU-seeded command, without a direct-memory claim')
    ap.add_argument('--compact-input-control',action='store_true',help='input-only staging diagnostic; requires marked opt-in engine')
    ap.add_argument('--prepared-input-staging',action='store_true',help='prepared A-only staging3x3 backend; requires marked engine')
    ap.add_argument('--prepared-padded-weights',action='store_true',help='prepared CI12/20 weight expansion; requires marked engine')
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--check-rejection',action='store_true',
                    help='prove a range beginning at a supported convolution rejects a later unsupported command before writes')
    args=ap.parse_args()
    if args.staged_control and (args.prepared_program or args.check_rejection):
        raise ValueError('staged control does not support prepared programs or direct-only rejection checks')
    if args.compact_input_control:
        if args.prepared_program or args.check_rejection or args.staged_control:raise ValueError('compact control requires one direct range only')
        if not any(m in args.engine.read_bytes() for m in (b'QD_COMPACT_INPUT_1',b'QD_COMPACT_INPUT_2')):raise ValueError('compact control engine marker absent')
    if args.prepared_input_staging:
        if not args.prepared_program or args.staged_control or args.compact_input_control:raise ValueError('prepared staging requires NPRG and excludes other controls')
        if b'QD_PREPARED_A3_1' not in args.engine.read_bytes():raise ValueError('prepared A3 engine marker absent')
    if args.prepared_padded_weights:
        if not args.prepared_program or args.staged_control or args.compact_input_control:raise ValueError('prepared padded weights requires NPRG only')
        if b'QD_PREPARED_W1_1' not in args.engine.read_bytes():raise ValueError('prepared padded weight engine marker absent')
    image=ensure_image(args.model);graph=resident_image.unpack(image.read_bytes())
    tensors={t['id']:t for t in graph['tensors']}
    constants=FRAME.parent/'recovered'/args.model/'constants/constant_pool_5.bin'
    report={'purpose':__doc__,'model':args.model,'runs':[],'passed':False,
            'staged_control':args.staged_control,'compact_input_control':args.compact_input_control,
            'prepared_input_staging':args.prepared_input_staging,'prepared_padded_weights':args.prepared_padded_weights,
            'compact_input_preserves_aliases':bool(args.compact_input_control and b'QD_COMPACT_INPUT_2' in args.engine.read_bytes()),
            'bootstrap_sha256':hashlib.sha256(args.bootstrap.read_bytes()).hexdigest()}
    if args.prepared_program:
        report['prepared_program_sha256']=hashlib.sha256(args.prepared_program.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
    # Finish every CPU calculation before registering a DSP callback.
    fixtures=[]
    for sequence in args.sequence:
        matches=[(i,c) for i,c in enumerate(graph['commands']) if c['sequence']==sequence]
        if len(matches)!=1:raise ValueError('sequence must identify one command')
        index,command=matches[0]
        x,w=[tensors[i] for i in command['inputs'][:2]];y=tensors[command['outputs'][0]]
        kernel=w['shape']['max_dimensions'][:2]
        if kernel not in ([1,1],[3,3]):raise ValueError('only proven native families are permitted')
        family='1x1' if kernel==[1,1] else '3x3'
        for label,directory,_ in cases(args.model,args.case or ['random'])[1]:
            inputs=load_inputs(directory);host=NativeGraph(image,constants)
            try:
                host.use_cpu();host.run(inputs,last=index)
                before={p:host.pools[p].copy() for p in (1,3,4,5,6)}
                host.run(inputs,first=index,last=index+1)
                after={p:host.pools[p].copy() for p in before}
            finally:host.close()
            fixtures.append((sequence,index,label,inputs,family,x,w,y,before,after))
    with ResidentDSP(image,constants,engine=args.engine,bootstrap=args.bootstrap,
                     prepared_program=args.prepared_program) as dsp:
        report['engine_sha256']=dsp.engine_sha256
        if not dsp.preserve_vtcm:raise ValueError('preserving bootstrap is required')
        for sequence,index,label,inputs,family,x,w,y,before,after in fixtures:
            for p,values in before.items():
                off=dsp.pools[p];n=dsp.sizes[p]
                dsp.mem[off:off+n]=values[:n].tobytes()
            if args.check_rejection and 'rejection' not in report:
                try:
                    dsp.run(inputs,first=index,last=dsp.commands)
                except RuntimeError:
                    if dsp.last.get('status')!=6:raise
                else:raise RuntimeError('unsupported suffix unexpectedly executed')
                unchanged=all((dsp.read_pool(p)==values[:dsp.sizes[p]]).all() for p,values in before.items())
                dsp.check_guards()
                report['rejection']={**dsp.last,'all_pools_unchanged':bool(unchanged)}
                report['rejection']['passed']=(unchanged and dsp.last.get('native_direct_range') is True
                    and all(dsp.last.get('native_'+f,{}).get('calls')==0 for f in ('1x1','3x3'))
                    and dsp.last.get('read_ms')==0 and dsp.last.get('write_ms')==0)
                save()
                if not report['rejection']['passed']:raise RuntimeError('unsupported range wrote graph memory')
            dsp.run(inputs,first=index,last=index+1)
            masks=output_footprint(y,w,x,{p:dsp.sizes[p] for p in before})
            pools={str(p):compare_pool(dsp.read_pool(p),after[p][:dsp.sizes[p]],masks[p]) for p in before}
            native=dsp.last.get('native_'+family,{})
            other=dsp.last.get('native_'+('3x3' if family=='1x1' else '1x1'),{})
            row={'sequence':sequence,'index':index,'case':label,**dsp.last,'pools':pools}
            mode_ok=(native.get('workspace_bytes_staged',0)>0 if args.staged_control or args.compact_input_control else
                     dsp.last.get('native_direct_range') is True and native.get('workspace_bytes_staged')==0)
            if args.prepared_input_staging and family=='3x3':mode_ok=dsp.last.get('native_prepared_program') is True and native.get('workspace_bytes_staged',0)>0
            if args.prepared_padded_weights and family=='1x1' and w['shape']['max_dimensions'][2] in (12,20):mode_ok=dsp.last.get('native_prepared_program') is True and native.get('workspace_bytes_staged',0)>0
            if args.compact_input_control:mode_ok &= dsp.last.get('native_direct_range') is True and dsp.last.get('native_prepared_program') is not True
            row['passed']=(mode_ok and native.get('calls')==1 and not native.get('failures')
                           and not other.get('calls',0) and all(p['passed'] for p in pools.values()))
            if args.prepared_program:row['passed'] &= dsp.last.get('native_prepared_program') is True
            report['runs'].append(row);save();print(json.dumps(row),flush=True)
    report['exit_errors']=dsp.exit_errors;report['vtcm_lease']=dsp.vtcm_lease
    report['passed']=(bool(report['runs']) and all(r['passed'] for r in report['runs'])
                      and (not args.check_rejection or report['rejection']['passed'])
                      and not any(dsp.exit_errors) and dsp.vtcm_lease['phase']==2
                      and not dsp.vtcm_lease['error'] and dsp.vtcm_lease['host_fingerprint_equal'])
    save()
    if not report['passed']:raise SystemExit('Direct VTCM range validation failed; see report')


if __name__=='__main__':main()
