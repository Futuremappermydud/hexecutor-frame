#!/usr/bin/env python3
"""Audit frozen v2 LK support and candidate stability without touching saved outputs.

The v2 evaluator is instrumented in memory at a checked source anchor to collect
its exact per-frame match arrays; it is neither rewritten nor replaced on disk.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, Delaunay, cKDTree, QhullError

import stereo_self_calibration as estimator

K = np.array([[380., 0, 308], [0, 380., 308], [0, 0, 1]])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def distribution(x):
    x = np.asarray(x)
    if not len(x):
        return None
    return {'median': float(np.median(x)), 'p90_abs': float(np.percentile(abs(x), 90))}


def cat(arrays):
    return np.concatenate(arrays) if arrays else np.empty((0, 4))


def support(frames):
    bins, counts, frame_counts = {}, [], []
    for frame in frames:
        p = frame['points']
        frame_counts.append(len(p))
        unique = set(map(tuple, np.floor(p[:, :2] / 2).astype(int)))
        counts.append(len(unique))
        for b in unique:
            bins[b] = bins.get(b, 0) + 1
    visits = np.asarray(list(bins.values()))
    return {'matches': sum(frame_counts), 'frames': len(frames),
            'unique_2px_bins': len(bins),
            'frame_unique_bin_counts': counts,
            'frame_match_counts': frame_counts,
            'bin_frame_visits_median': float(np.median(visits)) if len(visits) else 0,
            'bin_frame_visits_p90': float(np.percentile(visits, 90)) if len(visits) else 0,
            'bin_frame_visits_max': int(np.max(visits)) if len(visits) else 0,
            'fraction_bins_in_multiple_frames': float(np.mean(visits > 1)) if len(visits) else 0}


def evaluate(points, result):
    if not len(points):
        return None
    corrected = estimator.rectify_points(points, K, K, result)
    dy = corrected[:, 1] - corrected[:, 3]
    change = corrected[:, 0] - corrected[:, 2] - points[:, 0] + points[:, 2]
    return {'count': len(points), 'vertical_px': distribution(dy),
            'horizontal_disparity_change_px': distribution(change)}


def fit(points, held, model):
    try:
        r = estimator.fit_calibration(points, K, K, model=model)
        return {'result': r, 'heldout': evaluate(held, r)}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        return {'error': str(e)}


def profile(points, held, model, fitted):
    """Fixed yaw offsets; robustly refit other parameters, no acceptance implied."""
    p0 = fitted['parameters']
    base = np.r_[p0['rotation_vector_rad'], np.log(p0['focal_scale'])]
    if model == 'full6':
        base = np.r_[base, p0['baseline_slopes_yz']]
    bins = np.floor(points[:, :2] / 2).astype(int)
    _, inverse, count = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(count[inverse])
    residual = estimator._full_residual if model == 'full6' else estimator._residual
    indices = [i for i in range(len(base)) if i != 1]
    out = []
    for offset in (-.005, -.0025, 0., .0025, .005):
        fixed_yaw = base[1] + offset
        def unpack(x):
            p = base.copy(); p[1] = fixed_yaw; p[indices] = x
            return p
        def objective(x):
            return residual(points, K, K, unpack(x)) * weights
        try:
            sol = least_squares(objective, base[indices], loss='soft_l1', f_scale=.5,
                                max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            p = unpack(sol.x)
            pars = {'rotation_vector_rad': p[:3].tolist(), 'focal_scale': float(np.exp(p[3]))}
            if model == 'full6':
                direction, qvec, _ = estimator._baseline_frame(p[4:6])
                pars.update(baseline_direction_left=direction.tolist(),
                            baseline_slopes_yz=p[4:6].tolist(),
                            rectification_rotation_vector_rad=qvec.tolist())
            r = {'model': model, 'parameters': pars}
            out.append({'yaw_offset_rad': offset, 'yaw_rad': float(p[1]), 'success': bool(sol.success),
                        'soft_l1_weighted_cost': float(sol.cost), 'parameters': pars,
                        'heldout': evaluate(held, r)})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            out.append({'yaw_offset_rad': offset, 'error': str(e)})
    return out


def tag_extrapolation(train, tags, model, fitted):
    """Compare tag locations to natural training support; never fit to tags."""
    if not len(train) or not len(tags):
        return None
    points = train[:, :2]
    queries = tags[:, :2]
    nearest = cKDTree(points).query(queries)[0]
    try:
        hull = ConvexHull(points)
        outside = Delaunay(points[hull.vertices]).find_simplex(queries) < 0
        fraction = float(np.mean(outside))
        area = float(hull.volume)
    except QhullError:
        fraction, area = None, None
    pars = fitted['parameters']
    p = np.r_[pars['rotation_vector_rad'], np.log(pars['focal_scale'])]
    if model == 'full6':
        p = np.r_[p, pars['baseline_slopes_yz']]
    residual = estimator._full_residual if model == 'full6' else estimator._residual
    def sensitivity(matches):
        plus, minus = p.copy(), p.copy(); plus[1] += 1e-5; minus[1] -= 1e-5
        vertical = (residual(matches, K, K, plus) - residual(matches, K, K, minus)) / 2e-5 * .001
        def disparity(value):
            candidate = {'model': model, 'parameters': dict(pars, rotation_vector_rad=value[:3].tolist())}
            z = estimator.rectify_points(matches, K, K, candidate)
            return z[:, 0] - z[:, 2]
        horizontal = (disparity(plus) - disparity(minus)) / 2e-5 * .001
        return {'vertical_px_per_0_001rad': distribution(vertical),
                'disparity_px_per_0_001rad': distribution(horizontal)}
    def span(x):
        return {'x_min_max_px': [float(x[:, 0].min()), float(x[:, 0].max())],
                'y_min_max_px': [float(x[:, 1].min()), float(x[:, 1].max())],
                'normalized_x_span': float(np.ptp(x[:, 0]) / 616),
                'normalized_y_span': float(np.ptp(x[:, 1]) / 616)}
    return {'training_span': span(points), 'heldout_tag_span': span(queries),
            'train_convex_hull_area_px2': area, 'tag_fraction_outside_train_convex_hull': fraction,
            'tag_nearest_training_point_distance_px': {'median': float(np.median(nearest)),
                'p90': float(np.percentile(nearest, 90)), 'max': float(np.max(nearest))},
            'training_yaw_sensitivity': sensitivity(train), 'tag_yaw_sensitivity': sensitivity(tags)}


def collect(manifest, evaluator_path, expected_hash):
    source = Path(evaluator_path).read_text()
    if sha(evaluator_path) != expected_hash:
        raise ValueError('v2 evaluator hash differs from frozen result')
    marker = '        parts[split].append(pts)'
    model_loop = "    for model in ('rotation3', 'relative4', 'full6'):"
    if source.count(marker) != 1 or source.count(model_loop) != 1:
        raise ValueError('v2 evaluator instrumentation anchors differ')
    source = source.replace(marker, "        audit_callback(ordinal, split, pts.copy())\n" + marker)
    source = source.replace(model_loop, '    for model in ():')
    frames = []
    namespace = {'__name__': 'selfcal_support_instrumented', '__file__': str(evaluator_path),
                 'audit_callback': lambda ordinal, split, pts: frames.append(
                     {'ordinal': ordinal, 'split': split, 'points': pts})}
    exec(compile(source, str(evaluator_path), 'exec'), namespace)
    observed = namespace['run'](str(manifest), matching='lk')
    return frames, observed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest'); ap.add_argument('v2_result'); ap.add_argument('output')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    v2 = json.loads(Path(args.v2_result).read_text())
    if sha(here / 'stereo_self_calibration.py') != v2['estimator_sha256']:
        raise ValueError('estimator hash differs from frozen v2 result')
    if sha(args.manifest) != v2['manifest_sha256']:
        raise ValueError('manifest hash differs from frozen v2 result')
    frames, observed = collect(args.manifest, here / 'evaluate_stereo_self_calibration.py', v2['source_sha256'])
    for key in ('manifest_sha256', 'source_sha256', 'estimator_sha256', 'rectifier_sha256',
                'calibration_sha256', 'observations_sha256', 'pairs_sha256', 'input_sha256'):
        if observed[key] != v2[key]:
            raise ValueError('regenerated provenance differs from frozen v2: ' + key)
    actual_frames = [(f['ordinal'], f['split'], f['matches'], f['orb_seed_count']) for f in observed['frames']]
    frozen_frames = [(f['ordinal'], f['split'], f['matches'], f['orb_seed_count']) for f in v2['frames']]
    if actual_frames != frozen_frames:
        raise ValueError('regenerated per-frame counts differ from frozen v2')
    train_frames = [f for f in frames if f['split'] == 'train']
    held_frames = [f for f in frames if f['split'] == 'heldout']
    train, held = cat([f['points'] for f in train_frames]), cat([f['points'] for f in held_frames])
    if len(train) != v2['train']['count'] or len(held) != v2['heldout']['uncorrected']['count']:
        raise ValueError('regenerated match counts differ from frozen v2')
    quarter_size = len(frames) // 4
    manifest = json.loads(Path(args.manifest).read_text())
    observations = json.loads(Path(manifest['observations']).read_text())
    held_ids = {f['ordinal'] for f in held_frames}
    tag_rows = [np.column_stack((o['left'], o['right'])) for o in observations if o['ordinal'] in held_ids]
    tags = cat(tag_rows)
    fits = {}
    for model in ('relative4', 'full6'):
        entire = fit(train, held, model)
        if 'result' not in entire:
            raise ValueError('full-training fit failed during regeneration: ' + model)
        expected = v2['models'][model]['fit']
        actual = entire['result']
        if (actual['accepted'] != expected['accepted'] or
                not np.allclose(actual['parameters']['rotation_vector_rad'], expected['parameters']['rotation_vector_rad'], rtol=0, atol=1e-9) or
                not np.isclose(actual['parameters']['focal_scale'], expected['parameters']['focal_scale'], rtol=0, atol=1e-9)):
            raise ValueError('regenerated full-training fit differs from frozen v2: ' + model)
        if model == 'full6' and not np.allclose(actual['parameters']['baseline_slopes_yz'], expected['parameters']['baseline_slopes_yz'], rtol=0, atol=1e-9):
            raise ValueError('regenerated full6 baseline differs from frozen v2')
        quarters = {str(q): fit(cat([f['points'] for f in train_frames if f['ordinal']//quarter_size == q]), held, model)
                    for q in (0, 2)}
        regions = {}
        for y in range(2):
            for x in range(2):
                remove = ((train[:, 0] >= x*308) & (train[:, 0] < (x+1)*308) &
                          (train[:, 1] >= y*308) & (train[:, 1] < (y+1)*308))
                regions[f'{x},{y}'] = {'excluded_count': int(remove.sum()),
                                      **fit(train[~remove], held, model)}
        frame_scores = [{'ordinal': f['ordinal'], 'uncorrected_vertical_px': distribution(f['points'][:, 1] - f['points'][:, 3]),
                         'corrected': evaluate(f['points'], entire['result'])}
                        for f in held_frames] if 'result' in entire else []
        fits[model] = {'all_training': entire, 'training_quarter_only': quarters,
                       'leave_spatial_quadrant_out': regions, 'heldout_frames': frame_scores,
                       'yaw_profile': profile(train, held, model, entire['result']) if 'result' in entire else [],
                       'tag_extrapolation': tag_extrapolation(train, tags, model, entire['result']) if 'result' in entire else None}
    output = {'audit_source_sha256': sha(__file__), 'evaluator_sha256': v2['source_sha256'],
              'estimator_sha256': v2['estimator_sha256'], 'manifest_sha256': sha(args.manifest),
              'v2_result_sha256': sha(args.v2_result),
              'observed_provenance': {key: observed[key] for key in ('source_sha256', 'estimator_sha256', 'rectifier_sha256', 'calibration_sha256', 'observations_sha256', 'pairs_sha256', 'input_sha256')},
              'settings': {'quarter_size': quarter_size, 'train_quarters': [0, 2],
                           'quadrants': '2x2 over 616x616 left grid',
                           'profile': 'yaw offsets [-.005,-.0025,0,.0025,.005] rad around full fit; nuisance parameters soft_l1 refit, f_scale .5, per-2px-bin weighted; exploratory, not a confidence interval or acceptance test'},
              'train_support': support(train_frames), 'heldout_support': support(held_frames),
              'models': fits,
              'caveats': ['Rejected fit parameters shown only as diagnostics.', 'Pixel bins and frames are correlated; not independent sample counts.', 'Yaw profiles are local model-conditional sensitivity, not calibrated uncertainty.', 'Estimated disparity change is not metric truth; full6 also reorients both grids.', 'Undetected tags can remain in natural-feature support.']}
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
