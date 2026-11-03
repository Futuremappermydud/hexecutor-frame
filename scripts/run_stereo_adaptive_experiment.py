#!/usr/bin/env python3
"""Run the frozen bounded-support development screen; no evaluation inputs."""
import argparse
import os
from pathlib import Path
import subprocess
from stereo_composite_common import DEVELOPMENT_SEEDS, FRAMES, ARMS, digest, rows, write_json, ROOT
from analyze_stereo_composite_experiment import summarize, compare_common, gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    binary, inputs, output = args.binary.resolve(), args.inputs.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite existing output')
    sources = ['include/stereo_matcher_experiment.hpp', 'include/depth_speckle_filter.hpp',
               'tools/compare_stereo_matchers.cpp', 'tools/test_stereo_matcher_experiment.cpp',
               'tools/test_stereo_adaptive.cpp', 'scripts/run_stereo_adaptive_experiment.py',
               'scripts/stereo_composite_common.py', 'scripts/analyze_stereo_composite_experiment.py',
               'docs/stereo_adaptive_support_plan.md', 'Makefile']
    commands = []
    for radius in [2, 4, 8]:
        for threshold in [16, 32]:
            for case in [f'synthetic-{seed}' for seed in DEVELOPMENT_SEEDS] + [f'frame-{f}' for f in FRAMES]:
                name = f'r{radius}-t{threshold}/{case}'
                command = [str(binary), '--output', str(output / name), '--methods', 'adaptive',
                           '--support-radius', str(radius), '--edge-threshold', str(threshold),
                           '--refinement', 'both', '--baseline-controls', '--reuse-selection',
                           '--extended-controls', '--runs', '1', '--warmup', '0']
                # Save a representative synthetic and recorded case per configuration.
                if case not in ['synthetic-1709', 'frame-70']:
                    command += ['--metrics-only']
                if case.startswith('synthetic-'):
                    command += ['--synthetic', '--synthetic-seed', case.split('-')[1]]
                else:
                    f = case.split('-')[1]
                    command += ['--left', str(inputs / f'frame-{f}-left.pgm'),
                                '--right', str(inputs / f'frame-{f}-right.pgm')]
                commands.append(dict(name=name, radius=radius, threshold=threshold, case=case, command=command))
    input_hashes = {str(inputs / f'frame-{f}-{side}.pgm'): digest(inputs / f'frame-{f}-{side}.pgm')
                    for f in FRAMES for side in ['left', 'right']}
    output.mkdir(parents=True)
    write_json(output / 'plan.json', dict(binary_sha256=digest(binary),
               source_sha256={s: digest(ROOT / s) for s in sources}, input_sha256=input_hashes, jobs=commands))
    for s in sources:
        target = output / 'sources' / s
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / s).read_bytes())
    all_metrics, all_common = {}, {}
    for index, job in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {job['name']}", flush=True)
        folder = output / job['name']
        folder.parent.mkdir(parents=True, exist_ok=True)
        with folder.with_suffix('.log').open('x') as log:
            subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS='8'), check=True)
        write_json(folder / 'complete.json', dict(job=job, output_sha256={str(f.relative_to(folder)): digest(f)
                   for f in sorted(folder.rglob('*')) if f.is_file()}))
        key = f"r{job['radius']}-t{job['threshold']}"
        all_metrics.setdefault(key, []).extend(rows(folder / 'metrics.csv'))
        all_common.setdefault(key, []).extend(rows(folder / 'comparison.csv'))
    report = {}
    for key, metrics in all_metrics.items():
        report[key] = {}
        for arm in ARMS:
            reasons = []
            for suffix in ['', '-speckle']:
                method, reference = 'adaptive-' + arm + suffix, 'default-' + arm + suffix
                candidate = summarize([r for r in metrics if r['method'] == method])
                baseline = summarize([r for r in metrics if r['method'] == reference])
                common = compare_common(all_common[key], reference, method)
                failures = gate(candidate, baseline, common)
                reasons += [suffix + '/' + reason for reason in failures]
                report[key][arm + suffix] = dict(candidate=candidate, baseline=baseline, common=common, failures=failures)
            report[key][arm + '-passed'] = not reasons
    passed = [(key, arm) for key in report for arm in ARMS if report[key][arm + '-passed']]
    write_json(output / 'summary.json', dict(status='screen-pass-needs-calibration' if passed else 'failed-quality-gates',
               passing_arms=passed, configurations=report))
    print(f'Screen complete. Passing arms: {passed}', flush=True)


if __name__ == '__main__':
    main()
