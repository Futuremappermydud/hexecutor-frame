#!/usr/bin/env python3
"""Audit a completed adaptive screen and report per-seed regressions."""
import argparse
from pathlib import Path
from stereo_composite_common import (ARMS, SCENES, FRAMES, DEVELOPMENT_SEEDS, digest,
                                     read_json, rows, write_json, pooled)
from analyze_stereo_composite_experiment import summarize, gate, compare_common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    root = p.parse_args().root
    plan = read_json(root / 'plan.json')
    expected = {f'r{r}-t{t}/{case}' for r in [2,4,8] for t in [16,32]
                for case in [f'synthetic-{s}' for s in DEVELOPMENT_SEEDS] + [f'frame-{f}' for f in FRAMES]}
    if len(plan['jobs']) != len(expected) or {j['name'] for j in plan['jobs']} != expected:
        raise ValueError('Incomplete or unexpected screen plan')
    for source, checksum in plan['source_sha256'].items():
        if digest(root / 'sources' / source) != checksum:
            raise ValueError(f'Source snapshot changed: {source}')
    report, comparisons = {}, 0
    for job in plan['jobs']:
        folder = root / job['name']
        complete = read_json(folder / 'complete.json')
        if complete['job'] != job:
            raise ValueError(f'Completion job mismatch: {folder}')
        for name, checksum in complete['output_sha256'].items():
            if digest(folder / name) != checksum:
                raise ValueError(f'Changed result: {folder / name}')
            comparisons += 1
        metrics, common = rows(folder / 'metrics.csv'), rows(folder / 'comparison.csv')
        scenes = set(SCENES if job['case'].startswith('synthetic-') else ['recorded'])
        for arm in ARMS:
            for suffix in ['', '-speckle']:
                for method in ['adaptive', 'default', 'calibrated']:
                    selected = [r['scene'] for r in metrics if r['method'] == method + '-' + arm + suffix]
                    if set(selected) != scenes or len(selected) != len(scenes):
                        raise ValueError(f'Missing or duplicate metric scene: {folder}')
        key = job['name'].split('/')[0]
        entry = report.setdefault(key, dict(seed_regressions={}, recorded=[]))
        if scenes == {'recorded'}:
            entry['recorded'].extend(metrics)
        else:
            entry['seed_regressions'][job['case']] = {}
            for arm in ARMS:
                for suffix in ['', '-speckle']:
                    method, baseline = 'adaptive-' + arm + suffix, 'default-' + arm + suffix
                    entry['seed_regressions'][job['case']][arm + suffix] = gate(
                        summarize([r for r in metrics if r['method'] == method]),
                        summarize([r for r in metrics if r['method'] == baseline]),
                        compare_common(common, baseline, method))
    for entry in report.values():
        records = entry.pop('recorded')
        entry['recorded_mean_area_pct'] = {m: pooled([r for r in records if r['method'] == m], 'supported_pct', 'pixels')
                                        for m in ['adaptive-parabola', 'adaptive-parabola-speckle',
                                                  'default-parabola', 'default-parabola-speckle']}
    write_json(root / 'audit.json', dict(verified_jobs=len(expected), verified_outputs=comparisons, configurations=report))
    print(f'Verified {len(expected)} jobs and {comparisons} outputs')


if __name__ == '__main__':
    main()
