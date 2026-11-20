"""Frozen guarded-support development screen and verified evidence loading."""
from pathlib import Path
from stereo_composite_common import (ROOT, DEVELOPMENT_SEEDS, FRAMES, ARMS, SCENES,
                                    digest, read_json, rows)
SOURCES = ['include/stereo_matcher_experiment.hpp', 'include/depth_speckle_filter.hpp',
           'tools/compare_stereo_matchers.cpp', 'tools/test_stereo_matcher_experiment.cpp',
           'tools/test_stereo_adaptive.cpp', 'tools/test_stereo_guarded.cpp',
           'tools/test_stereo_guarded_workflow.py', 'scripts/run_stereo_guarded_experiment.py',
           'scripts/analyze_stereo_guarded_experiment.py', 'scripts/stereo_guarded_common.py',
           'scripts/stereo_composite_common.py', 'scripts/analyze_stereo_composite_experiment.py',
           'docs/stereo_guarded_support_plan.md', 'Makefile']
METHODS = ['default', 'calibrated', 'adaptive', 'guarded-adaptive']
CASES = [f'synthetic-{seed}' for seed in DEVELOPMENT_SEEDS] + [f'frame-{f}' for f in FRAMES]


def jobs(binary, inputs, output):
    result = []
    for guard in range(4):
        for exposure in [False, True]:
            key = f'g{guard}-e{int(exposure)}'
            for case in CASES:
                name = f'{key}/{case}'
                command = [str(binary), '--output', str(output/name), '--methods', 'adaptive,guarded-adaptive',
                           '--support-radius', '8', '--edge-threshold', '32', '--support-guard', str(guard),
                           '--refinement', 'both', '--baseline-controls', '--reuse-selection',
                           '--extended-controls', '--cost-cutoff', '120', '--uniqueness-margin', '3',
                           '--runs', '1', '--warmup', '0']
                if exposure:
                    command += ['--support-exposure']
                if case not in ['synthetic-1709', 'frame-70']:
                    command += ['--metrics-only']
                if case.startswith('synthetic-'):
                    command += ['--synthetic', '--synthetic-seed', case.split('-')[1]]
                else:
                    f = case.split('-')[1]
                    command += ['--left', str(inputs/f'frame-{f}-left.pgm'), '--right', str(inputs/f'frame-{f}-right.pgm')]
                result.append(dict(name=name, guard=guard, exposure=exposure, case=case, command=command))
    return result


def make_plan(binary, inputs, output):
    return dict(version=1, binary=str(binary), binary_sha256=digest(binary), inputs=str(inputs), output=str(output),
                source_sha256={s:digest(ROOT/s) for s in SOURCES},
                input_sha256={str(inputs/f'frame-{f}-{side}.pgm'):digest(inputs/f'frame-{f}-{side}.pgm')
                              for f in FRAMES for side in ['left','right']}, jobs=jobs(binary,inputs,output))


def verify_complete(folder, job):
    complete = read_json(folder/'complete.json')
    if complete['job'] != job or not complete['output_sha256']:
        raise ValueError(f'Completion mismatch: {folder}')
    expected_files = {str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file() and f.name != 'complete.json'}
    if expected_files != set(complete['output_sha256']):
        raise ValueError(f'Output inventory changed: {folder}')
    for name, checksum in complete['output_sha256'].items():
        if digest(folder/name) != checksum:
            raise ValueError(f'Output changed: {folder/name}')
    return complete


def load(root):
    plan = read_json(root/'plan.json')
    if plan['jobs'] != jobs(Path(plan['binary']),Path(plan['inputs']),Path(plan['output'])):
        raise ValueError('Incomplete or modified frozen screen')
    if set(plan['source_sha256']) != set(SOURCES):
        raise ValueError('Incomplete source inventory')
    for source, checksum in plan['source_sha256'].items():
        if digest(root/'sources'/source) != checksum:
            raise ValueError(f'Source snapshot changed: {source}')
    if digest(root/'matcher-binary') != plan['binary_sha256']:
        raise ValueError('Archived binary changed')
    loaded=[]
    for job in plan['jobs']:
        folder=root/job['name']
        complete=verify_complete(folder,job)
        metrics, common, fixed = (rows(folder/(name+'.csv')) for name in ['metrics','comparison','fixed-mask'])
        scenes=set(SCENES if job['case'].startswith('synthetic-') else ['recorded'])
        for arm in ARMS:
            for suffix in ['', '-speckle']:
                for method in METHODS:
                    found=[r['scene'] for r in metrics if r['method']==method+'-'+arm+suffix]
                    if len(found)!=len(scenes) or set(found)!=scenes:
                        raise ValueError(f'Missing/duplicate metric scene in {folder}')
                for reference in ['default','adaptive']:
                    found=[r['scene'] for r in common if r['reference']==reference+'-'+arm+suffix and
                           r['method']=='guarded-adaptive-'+arm+suffix]
                    if len(found)!=len(scenes) or set(found)!=scenes:
                        raise ValueError(f'Missing/duplicate common-mask scene in {folder}')
        loaded.append(dict(job=job,metrics=metrics,comparison=common,fixed=fixed,complete=complete))
    return plan,loaded
