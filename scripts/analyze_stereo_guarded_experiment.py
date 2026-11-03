#!/usr/bin/env python3
"""Verify, pool and gate the complete guarded-support development screen."""
import argparse
from array import array
from pathlib import Path
import sys
from stereo_composite_common import ARMS,pooled,write_json
from stereo_guarded_common import load
from analyze_stereo_composite_experiment import summarize,compare_common,gate


def diagnostics(folder,arm):
    output={}
    for side in ['left','right']:
        output[side]={}
        for label in ['anchors','information','support-bits','anchor-stops']:
            path=folder/f'guarded-adaptive-{arm}-{side}-{label}.f32'
            values=array('f');values.frombytes(path.read_bytes())
            # Stage exports use native little-endian on both supported machines.
            if sys.byteorder!='little':values.byteswap()
            output[side][label]=dict(pixels=len(values),nonzero=sum(v>0 for v in values),
                                    mean=sum(values)/len(values),maximum=max(values))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    root=p.parse_args().root
    plan,cases=load(root)
    grouped={}
    for case in cases:grouped.setdefault(case['job']['name'].split('/')[0],[]).append(case)
    report={};passing=[]
    for key,group in grouped.items():
        metrics=[r for case in group for r in case['metrics']]
        comparisons=[r for case in group for r in case['comparison']]
        entry=dict(arms={},seed_regressions={},recorded_mean_area_pct={},diagnostics={})
        for arm in ARMS:
            arm_result={};reasons=[]
            for suffix in ['', '-speckle']:
                method='guarded-adaptive-'+arm+suffix
                candidate=summarize([r for r in metrics if r['method']==method])
                references={}
                for reference in ['default','adaptive']:
                    name=reference+'-'+arm+suffix
                    baseline=summarize([r for r in metrics if r['method']==name])
                    common=compare_common(comparisons,name,method)
                    failures=gate(candidate,baseline,common)
                    references[reference]=dict(scenes=baseline,common=common,failures=failures)
                    if reference=='default':reasons.extend([suffix+'/'+reason for reason in failures])
                arm_result['cleanup' if suffix else 'support']=dict(candidate=candidate,references=references)
            arm_result.update(passed=not reasons,failure_reasons=reasons)
            entry['arms'][arm]=arm_result
            if not reasons:passing.append([key,arm])
        for case in group:
            job=case['job']
            if job['case'].startswith('synthetic-'):
                regressions={}
                for arm in ARMS:
                    for suffix in ['', '-speckle']:
                        method,reference='guarded-adaptive-'+arm+suffix,'default-'+arm+suffix
                        regressions[arm+suffix]=gate(
                            summarize([r for r in case['metrics'] if r['method']==method]),
                            summarize([r for r in case['metrics'] if r['method']==reference]),
                            compare_common(case['comparison'],reference,method))
                entry['seed_regressions'][job['case']]=regressions
            if job['case'] in ['synthetic-1709','frame-70']:
                base=root/job['name']
                folders=[base/'recorded'] if job['case']=='frame-70' else sorted(f for f in base.iterdir() if f.is_dir())
                entry['diagnostics'][job['case']]={f.name:{arm:diagnostics(f,arm) for arm in ARMS} for f in folders}
        recorded=[r for r in metrics if r['scene']=='recorded']
        for method in ['default','adaptive','guarded-adaptive']:
            for arm in ARMS:
                for suffix in ['', '-speckle']:
                    name=method+'-'+arm+suffix
                    entry['recorded_mean_area_pct'][name]=pooled([r for r in recorded if r['method']==name],'supported_pct','pixels')
        report[key]=entry
    write_json(root/'summary.json',dict(status='screen-pass-needs-calibration' if passing else 'failed-quality-gates',
               passing_arms=passing,verified_jobs=len(cases),
               verified_outputs=sum(len(c['complete']['output_sha256']) for c in cases),configurations=report))
    print(f'Verified {len(cases)} jobs. Passing arms: {passing}')


if __name__=='__main__':main()
