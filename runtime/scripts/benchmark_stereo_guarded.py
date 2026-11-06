#!/usr/bin/env python3
"""Run on Frame: alternate saved-pair variants, enforce exact output, save hashes/tails.
This is matcher replay, not a history or human-motion benchmark.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def percentile(values, q):
    return sorted(values)[int(q * (len(values) - 1))] if values else None


def summarize(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    pairs = {r['pair']: r for r in rows}
    result = {'pairs': len(pairs), 'failed': sum(r['success'] != '1' for r in pairs.values())}
    for name in ('pair_ms', 'histogram_ms', 'record_submit_ms', 'fence_ms', 'yield_ms', 'readback_ms'):
        values = [float(r[name]) for r in pairs.values()]
        result[name] = {f'p{int(q*100)}': percentile(values, q) for q in (.5, .95, .99)}
    for stage in sorted({r['stage'] for r in rows if r['gpu_ms']}):
        values = [float(r['gpu_ms']) for r in rows if r['stage'] == stage]
        result['gpu_' + stage + '_ms'] = {f'p{int(q*100)}': percentile(values, q) for q in (.5, .95, .99)}
    result['allocation_high_water_bytes'] = max(int(r['allocated_bytes']) for r in rows)
    result['valid_pixels'] = sorted({int(r['valid_pixels']) for r in rows})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--pairs', type=int, default=10)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--scheduling', action='store_true')
    a = p.parse_args()
    if a.pairs < 1 or a.rounds < 1:
        p.error('pairs and rounds must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    variants = [('control', 0, 0, 8, 3000, 8, 8), ('arms', 1, 0, 8, 3000, 8, 8),
                ('cache', 0, 1, 8, 3000, 8, 8), ('arms_cache', 1, 1, 8, 3000, 8, 8)]
    if a.scheduling:
        variants += [(f'rows{rows}_yield{pause}_local{x}x{y}', 1, 1, rows, pause, x, y)
                     for rows, pause, x, y in [(8, 1000, 8, 8), (16, 3000, 8, 8), (8, 3000, 16, 4)]]
    files = [a.left, a.right, Path('bin/benchmark_stereo_guarded'),
             Path('src/vulkan_guarded_stereo_pipeline.cpp'), Path('include/calibration.hpp')]
    files += sorted(Path('shaders').glob('stereo*.spv')) + sorted(Path('shaders').glob('stereo*.comp'))
    manifest = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    mode='saved_rectified_pair_no_history', omp_threads=8, variants=variants,
                    pairs=a.pairs, rounds=a.rounds, warmup_pairs=1,
                    hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summaries = {}
    for round_index in range(a.rounds):
        for name, arms, cache, rows, pause, x, y in (variants if round_index % 2 == 0 else variants[::-1]):
            stem = f'{round_index:02d}_{name}'
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='8', FRAME_GUARDED_ARMS=str(arms), FRAME_GUARDED_CACHE=str(cache),
                       FRAME_GUARDED_BATCH_ROWS=str(rows), FRAME_GUARDED_YIELD_US=str(pause),
                       FRAME_GUARDED_LOCAL_X=str(x), FRAME_GUARDED_LOCAL_Y=str(y))
            path = a.output / (stem + '.csv')
            with (a.output / (stem + '.log')).open('w') as log:
                run = subprocess.run(['./bin/benchmark_stereo_guarded', 'shaders', str(a.left), str(a.right),
                                      str(a.pairs), str(path)], env=env, stdout=log, stderr=subprocess.STDOUT)
            summaries[stem] = summarize(path) if path.exists() else {}
            summaries[stem]['exit_code'] = run.returncode
            (a.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
            if run.returncode:
                raise RuntimeError(f'{stem} failed; inspect {a.output}')
            print(stem, summaries[stem]['pair_ms'], flush=True)


if __name__ == '__main__':
    main()
