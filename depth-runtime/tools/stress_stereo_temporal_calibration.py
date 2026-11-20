#!/usr/bin/env python3
"""Frozen offline temporal BA stress sweep using independent 3-D projections.

Requires a frozen directory containing stereo_temporal_calibration_v4.py and
stereo_self_calibration.py. The 29 cases and two seeds below are declared before
running. Prediction errors use the known physical baseline, 27 independent
left-frame probes, and the maximum absolute horizontal original-right error.
Accepted is a training gate, never proof of metric accuracy. Synthetic results
and frozen source copies belong in an ignored directory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

K = np.array([[600., 0., 640.], [0., 610., 480.], [0., 0., 1.]])
BASELINE = .065
R_TRUE = np.array([.004, -.008, .003])
FOCAL = 1.008
SEEDS = (517, 991)


def cases():
    out = []
    def add(group, motion=.03, depth='varied', coverage='full', noise=.08,
            outlier_fraction=0., shear_axis='none', shear_px=0.):
        out.append(dict(group=group, motion_m_per_frame=motion, depth=depth,
                        coverage=coverage, noise_px=noise, outlier_fraction=outlier_fraction,
                        shear_axis=shear_axis, shear_px=shear_px))
    for m in (0., .003, .01, .03, .10):
        add('motion', motion=m)
        add('planar_motion', motion=m, depth='planar')
        add('central_motion', motion=m, coverage='central')
    for noise in (.3, .7):
        for coverage in ('full', 'central'):
            add('noise', noise=noise, coverage=coverage)
    for frac in (.05, .15):
        for coverage in ('full', 'central'):
            add('random_outliers', outlier_fraction=frac, coverage=coverage)
    for axis in ('horizontal', 'vertical'):
        for amp in (.1, .3, .5):
            add('coherent_shear', shear_axis=axis, shear_px=amp)
    assert len(out) == 29
    return out


def rotation(v):
    angle = np.linalg.norm(v)
    if not angle:
        return np.eye(3)
    x, y, z = v / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1. - np.cos(angle)) * skew @ skew


def image(p, focal):
    return np.column_stack((K[0, 0] * focal * p[:, 0] / p[:, 2] + K[0, 2],
                            K[1, 1] * focal * p[:, 1] / p[:, 2] + K[1, 2]))


def make_scene(cfg, seed):
    rng = np.random.default_rng(seed)
    n = 45
    z = rng.uniform(.8, 2.8, n) if cfg['depth'] == 'varied' else np.full(n, 1.5)
    spread = 1. if cfg['coverage'] == 'full' else .25
    world = np.column_stack((rng.uniform(-.5, .5, n) * spread * z,
                              rng.uniform(-.35, .35, n) * spread * z, z))
    rows = []
    for frame in range(5):
        frame_rotation = rotation(np.array([.004, -.006, .002]) * frame)
        center = np.array([1., .15, .05]) * cfg['motion_m_per_frame'] * frame
        left = (world - center) @ frame_rotation
        right = (left - [BASELINE, 0., 0.]) @ rotation(R_TRUE)
        lp, rp = image(left, 1.), image(right, FOCAL)
        if cfg['shear_axis'] == 'horizontal':
            rp[:, 0] += cfg['shear_px'] * (lp[:, 1] - K[1, 2]) / (K[1, 1] * .35)
        elif cfg['shear_axis'] == 'vertical':
            rp[:, 1] += cfg['shear_px'] * (lp[:, 0] - K[0, 2]) / (K[0, 0] * .5)
        rows.extend([[frame, track, *lp[track], *rp[track]] for track in range(n)])
    data = np.asarray(rows)
    data[:, 2:] += rng.normal(0., cfg['noise_px'], data[:, 2:].shape)
    count = int(len(data) * cfg['outlier_fraction'])
    if count:
        bad = rng.choice(len(data), count, replace=False)
        data[bad, 4:6] += rng.normal(0., 8., (count, 2))
    return data


def prediction_error(result):
    probe = np.array([[x*z, y*z, z] for z in (.7, 1.5, 2.5)
                       for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
    def right(v, scale):
        return image((probe - [BASELINE, 0., 0.]) @ rotation(np.asarray(v)), scale)
    params = result['parameters']
    delta = right(params['rotation_vector_rad'], params['focal_scale']) - right(R_TRUE, FOCAL)
    return {'max_abs_horizontal_px': float(np.max(np.abs(delta[:, 0]))),
            'median_abs_horizontal_px': float(np.median(np.abs(delta[:, 0]))),
            'max_abs_vertical_px': float(np.max(np.abs(delta[:, 1])))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('frozen_dir'); ap.add_argument('output')
    args = ap.parse_args()
    directory = Path(args.frozen_dir).resolve()
    temporal = directory / 'stereo_temporal_calibration_v4.py'
    core = directory / 'stereo_self_calibration.py'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('frozen_temporal', temporal)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    output = Path(args.output)
    results = {'protocol': {'case_count': 29, 'seeds': list(SEEDS), 'cases': cases(),
               'baseline_m': BASELINE, 'baseline_assumed_known': True,
               'truth_threshold_px': .3, 'truth_metric': 'max abs original-right horizontal error on 27 probes',
               'probe_x_y': [-.5, 0., .5], 'probe_depth_m': [.7, 1.5, 2.5],
               'shear': 'right x += amplitude * left normalized y / .35, or right y += amplitude * left normalized x / .5',
               'outliers': 'independent random right x/y corruption Gaussian sigma 8px on selected observations',
               'noise': 'independent Gaussian px added to every left/right coordinate'},
               'provenance': {'temporal_sha256': sha(temporal), 'core_sha256': sha(core),
                              'script_sha256': sha(__file__), 'numpy': np.__version__}, 'runs': []}
    start = time.monotonic(); cpu_start = time.process_time()
    for case_id, case in enumerate(cases()):
        for seed in SEEDS:
            started = time.monotonic(); started_cpu = time.process_time()
            try:
                result = module.fit_temporal_calibration(make_scene(case, seed), K, K, BASELINE)
                err = prediction_error(result)
                diag = result['diagnostics']
                row = {'case_id': case_id, 'seed': seed, 'accepted': result['accepted'],
                       'reasons': diag.get('reasons', []), 'converged': diag.get('converged'),
                       'inlier_fraction': diag.get('inlier_fraction'),
                       'per_view_inlier_fraction': diag.get('per_view_inlier_fraction'),
                       'noise_mad_px': diag.get('scalar_residual_mad_sigma_px'),
                       'nominal_probe_std_px': diag.get('nominal_independent_noise_max_probe_std_px'),
                       'spatial_support': diag.get('spatial_support'),
                       'truth_error': err, 'accepted_wrong_gt_0_3px': bool(result['accepted'] and err['max_abs_horizontal_px'] > .3),
                       'parameters': result['parameters']}
            except Exception as exc:
                row = {'case_id': case_id, 'seed': seed, 'error': str(exc), 'exception': type(exc).__name__}
            row['wall_seconds'] = time.monotonic() - started
            row['cpu_seconds'] = time.process_time() - started_cpu
            results['runs'].append(row)
            results['elapsed_seconds'] = time.monotonic() - start
            results['cpu_seconds'] = time.process_time() - cpu_start
            output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print('runs', len(results['runs']), 'elapsed_seconds', round(results['elapsed_seconds'], 2),
          'cpu_seconds', round(results['cpu_seconds'], 2),
          'accepted_wrong', sum(r.get('accepted_wrong_gt_0_3px', False) for r in results['runs']))


if __name__ == '__main__':
    main()
