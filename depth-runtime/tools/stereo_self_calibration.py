#!/usr/bin/env python3
"""Offline relative stereo rectification from undistorted pinhole matches.

Inspired by Wang et al., CVPR 2023, section 4, equation (4):
https://openaccess.thecvf.com/content/CVPR2023/html/Wang_A_Practical_Stereo_Depth_System_for_Smart_Glasses_CVPR_2023_paper.html

The rotation3/relative4 models hold the left camera and horizontal baseline
fixed and optimize an exact rotation of right rays instead of the paper's
linearized symmetric parameterization. Their relative4 constraint corresponds
to equation (4) to first order; it omits the two disparity-dependent terms.
The experimental full6 model estimates the relative rotation, focal scale, and
two baseline-direction slopes from the exact two-view epipolar constraint. Its
left camera frame is the gauge, and its common rectifying rotation is a change
of coordinates, not a physical absolute-angle measurement. Observability gates
are conservative diagnostics, not guarantees. There is no baseline-length
estimate, absolute yaw estimate, or metric-depth acceptance. Small residuals cannot establish metric accuracy. Inputs must already
be undistorted pinhole coordinates, including any sensor parity correction.

CLI input JSON: {"matches": [[ul,vl,ur,vr], ...], "K_left": [[...],...],
"K_right": [[...],...], "model": "relative4", "config": {...}}. Output is
JSON containing the fit result. No image extraction or live camera access.
"""
import argparse
import json
import sys

import numpy as np

_DEFAULTS = {
    "min_matches": 100, "min_inlier_fraction": 0.60,
    "inlier_threshold_px": 1.0, "min_axis_span": 0.15,
    "min_occupied_cells": 6, "min_singular_ratio": 1e-5,
    "max_rotation_rad": 0.15, "max_focal_log_scale": 0.15,
    "max_iterations": 60,
}


def _inputs(matches, kl, kr):
    m = np.asarray(matches, dtype=float)
    if m.ndim != 2 or m.shape[1] != 4 or not np.isfinite(m).all():
        raise ValueError("matches must be a finite N-by-4 array")
    matrices = []
    for name, k in (("K_left", kl), ("K_right", kr)):
        k = np.asarray(k, dtype=float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or
                k[0, 0] <= 0 or k[1, 1] <= 0 or
                not np.allclose(k[2], [0, 0, 1]) or
                abs(k[1, 0]) > 1e-12):
            raise ValueError(name + " must be a finite pinhole intrinsic matrix")
        matrices.append(k)
    return m, *matrices


def _rays(points, k, scale=1.0):
    y = (points[:, 1] - k[1, 2]) / (k[1, 1] * scale)
    x = (points[:, 0] - k[0, 2] - k[0, 1] * scale * y) / (k[0, 0] * scale)
    return np.column_stack((x, y, np.ones(len(points))))


def _rotation(v):
    a = np.linalg.norm(v)
    x, y, z = v
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    if a < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + np.sin(a) / a * cross + (1 - np.cos(a)) / (a * a) * (cross @ cross)


def _project(rays, k):
    q = rays @ k.T
    return q[:, :2] / q[:, 2, None]


def _residual(m, kl, kr, p):
    left = _rays(m[:, :2], kl)
    right = _rays(m[:, 2:], kr, np.exp(p[3])) @ _rotation(p[:3]).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return (right[:, 1] / right[:, 2] - left[:, 1]) * kl[1, 1]


def _jacobian(m, kl, kr, p, n):
    cols = []
    for j in range(n):
        d = np.zeros(4)
        d[j] = 1e-6
        cols.append((_residual(m, kl, kr, p + d) - _residual(m, kl, kr, p - d)) / 2e-6)
    return np.column_stack(cols)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    index = np.searchsorted(np.cumsum(weights), q * weights.sum(), side='left')
    return float(values[min(index, len(values) - 1)])


def fit_calibration(matches, K_left, K_right, model="relative4", config=None):
    """Fit deterministic relative rotation/focal scale; reject unsupported fits.

    Rotation vector is radians, right ray to left orientation. focal_scale
    multiplies both right focal axes (and skew), with principal point fixed.
    Rejection reasons and conditioning are returned even if a candidate exists.
    """
    if model == "full6":
        return _fit_full6(matches, K_left, K_right, config)
    m, kl, kr = _inputs(matches, K_left, K_right)
    if model not in ("relative4", "rotation3"):
        raise ValueError("model must be relative4 or rotation3")
    cfg = dict(_DEFAULTS)
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(cfg):
            raise ValueError("unknown configuration option")
        cfg.update(config)
    if (cfg['min_matches'] < 4 or not 0 < cfg['min_inlier_fraction'] <= 1 or
            cfg['inlier_threshold_px'] <= 0 or cfg['min_axis_span'] < 0 or
            cfg['min_occupied_cells'] < 1 or cfg['min_singular_ratio'] <= 0 or
            cfg['max_rotation_rad'] <= 0 or cfg['max_focal_log_scale'] <= 0 or
            cfg['max_iterations'] < 1):
        raise ValueError("invalid configuration value")
    n = 4 if model == 'relative4' else 3
    p = np.zeros(4)
    # Count/weight pixel locations, not repeated observations at one location.
    # This is not a claim of temporal or scene independence.
    bins = np.floor(m[:, :2] / 2.0).astype(np.int64)
    _, inverse, multiplicities = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    support_weights = 1.0 / multiplicities[inverse] if len(m) else np.empty(0)
    unique_support = len(multiplicities)
    reasons = []
    converged = False
    if len(m) >= max(n, cfg['min_matches']):
        # Continuation from Huber to a bounded redescending weight. The robust
        # scale is floored to avoid giving exact synthetic data zero weight.
        for iteration in range(int(cfg['max_iterations'])):
            r = _residual(m, kl, kr, p)
            if not np.isfinite(r).all():
                reasons.append('nonfinite_projection')
                break
            med = _weighted_quantile(r, support_weights, 0.5)
            sigma = max(0.05, 1.4826 * _weighted_quantile(np.abs(r - med), support_weights, 0.5))
            if iteration < 5:
                cutoff = max(1.5, 2.5 * sigma)
                w = np.minimum(1., cutoff / np.maximum(np.abs(r), 1e-12))
            else:
                cutoff = max(1.5 * cfg['inlier_threshold_px'], 4.685 * sigma)
                t = r / cutoff
                w = np.where(np.abs(t) < 1, (1 - t*t)**2, 0.)
            w *= support_weights
            j = _jacobian(m, kl, kr, p, n)
            try:
                step = np.linalg.lstsq(j * np.sqrt(w[:, None]), -r * np.sqrt(w), rcond=None)[0]
            except np.linalg.LinAlgError:
                reasons.append('solver_failure')
                break
            candidate = p.copy()
            candidate[:n] += step
            if (not np.isfinite(candidate).all() or
                    np.linalg.norm(candidate[:3]) > cfg['max_rotation_rad'] or
                    abs(candidate[3]) > cfg['max_focal_log_scale']):
                reasons.append('parameter_bound_exceeded')
                break
            p = candidate
            if np.linalg.norm(step) < 1e-10 and iteration >= 5:
                converged = True
                break
    else:
        reasons.append('insufficient_matches')
    residual = _residual(m, kl, kr, p)
    inliers = np.isfinite(residual) & (np.abs(residual) <= cfg['inlier_threshold_px'])
    count = int(inliers.sum())
    unique_inlier_support = float(support_weights[inliers].sum())
    fraction = unique_inlier_support / unique_support if unique_support else 0.0
    if unique_support < cfg['min_matches'] or unique_inlier_support < cfg['min_matches']:
        reasons.append('insufficient_unique_support')
    if fraction < cfg['min_inlier_fraction']:
        reasons.append('insufficient_inlier_fraction')
    if count < cfg['min_matches']:
        reasons.append('insufficient_inliers')
    if not converged and 'insufficient_matches' not in reasons:
        reasons.append('not_converged')
    span = [0., 0.]
    cells = 0
    ratio = 0.0
    singular = []
    if count:
        rays = _rays(m[inliers, :2], kl)[:, :2]
        iw = support_weights[inliers]
        span = [_weighted_quantile(rays[:, axis], iw, .95) -
                _weighted_quantile(rays[:, axis], iw, .05) for axis in range(2)]
        # Fixed angular bins, so a tiny cluster cannot fill the occupancy grid.
        bins = np.floor(rays / 0.1).astype(int)
        cells = int(len(np.unique(bins, axis=0)))
        j = _jacobian(m[inliers], kl, kr, p, n)
        singular = np.linalg.svd(j * np.sqrt(iw[:, None]), compute_uv=False).tolist()
        if len(singular) == n and singular[0] > 0:
            ratio = singular[-1] / singular[0]
    if min(span) < cfg['min_axis_span'] or cells < cfg['min_occupied_cells']:
        reasons.append('insufficient_spatial_support')
    if ratio < cfg['min_singular_ratio']:
        reasons.append('rank_or_conditioning_failure')
    return {
        'model': model, 'method': 'fixed_baseline_exact_relative_rays_irls',
        'convention': 'right_to_left_rotation_vector_radians; right focal scale multiplies fx, fy and skew; left fixed',
        'parameters': {'rotation_vector_rad': p[:3].tolist(), 'focal_scale': float(np.exp(p[3]))},
        'accepted': not reasons, 'metric_acceptance': 'unevaluated',
        'diagnostics': {'reasons': reasons, 'match_count': len(m), 'inlier_count': count,
                        'inlier_fraction': fraction, 'unique_support_count': unique_support,
                        'unique_inlier_support': unique_inlier_support, 'support_bin_size_px': 2.0, 'inlier_threshold_px': cfg['inlier_threshold_px'],
                        'median_abs_vertical_px': float(np.median(np.abs(residual))) if len(m) and np.isfinite(residual).all() else None,
                        'spatial_span_normalized_5_95': span, 'occupied_angular_cells': cells,
                        'singular_values': singular, 'singular_ratio': float(ratio), 'converged': converged},
    }


def rectify_points(matches, K_left, K_right, result):
    """Apply fitted right-ray map; caller must check result['accepted']."""
    m, kl, kr = _inputs(matches, K_left, K_right)
    params = result['parameters']
    rotation = np.asarray(params['rotation_vector_rad'], dtype=float)
    scale = float(params['focal_scale'])
    if rotation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid fitted parameters')
    rays = _rays(m[:, 2:], kr, scale) @ _rotation(rotation).T
    if result.get('model') == 'full6':
        q = _rotation(np.asarray(params['rectification_rotation_vector_rad'], dtype=float))
        left_rays = _rays(m[:, :2], kl) @ q.T
        rays = rays @ q.T
        if np.any(left_rays[:, 2] <= 0):
            raise ValueError('corrected left ray is not forward-facing')
        left = _project(left_rays, kl)
    else:
        left = m[:, :2]
    if np.any(rays[:, 2] <= 0):
        raise ValueError('corrected right ray is not forward-facing')
    return np.column_stack((left, _project(rays, kr)))



def _baseline_frame(slopes):
    """Minimal rotation Q taking the left-frame baseline direction to +x."""
    t = np.array([1., slopes[0], slopes[1]])
    t /= np.linalg.norm(t)
    axis = np.cross(t, [1., 0., 0.])
    length = np.linalg.norm(axis)
    v = axis * (np.arctan2(length, t[0]) / length) if length > 1e-15 else np.zeros(3)
    return t, v, _rotation(v)


def _full_residual(m, kl, kr, p):
    # P = lambda_left * l = baseline * t + lambda_right * R r.
    # Rotating both rays by Q with Q t = +x makes their normalized y equal.
    _, _, q = _baseline_frame(p[4:6])
    left = _rays(m[:, :2], kl) @ q.T
    right = _rays(m[:, 2:], kr, np.exp(p[3])) @ _rotation(p[:3]).T @ q.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return (right[:, 1] / right[:, 2] - left[:, 1] / left[:, 2]) * kl[1, 1]


def _full_jacobian(m, kl, kr, p):
    columns = []
    for j in range(6):
        d = np.zeros(6)
        d[j] = 1e-6
        columns.append((_full_residual(m, kl, kr, p + d) - _full_residual(m, kl, kr, p - d)) / 2e-6)
    return np.column_stack(columns)


def _fit_full6(matches, K_left, K_right, config):
    """Experimental exact two-view geometry; baseline length is unobservable.

    Left camera defines the coordinate gauge. R rotates right rays into it;
    t is the left-to-right baseline direction. Q is only a common rectification
    coordinate change, not an independently inferred physical camera rotation.
    The additional baseline-direction terms correspond to the disparity-dependent
    degrees in Wang et al. section 4, but no small-angle approximation is used.
    A scene plane admits an ambiguous projective explanation: require disparity
    variation beyond its best affine image-plane fit as a conservative proxy for
    nonplanarity, with a noise floor estimated from vertical residuals. This
    assumes vertical noise is representative of horizontal matching noise.
    This test and local Jacobian conditioning do not prove metric
    accuracy or detect all critical configurations.
    """
    m, kl, kr = _inputs(matches, K_left, K_right)
    cfg = dict(_DEFAULTS, max_baseline_slope=0.15,
               min_nonplanar_disparity_span=0.003, full6_min_singular_ratio=1e-4,
               nonplanar_noise_multiplier=6.0)
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(cfg):
            raise ValueError('unknown configuration option')
        cfg.update(config)
    if (cfg['min_matches'] < 6 or not 0 < cfg['min_inlier_fraction'] <= 1 or
            cfg['inlier_threshold_px'] <= 0 or cfg['min_axis_span'] < 0 or
            cfg['min_occupied_cells'] < 1 or cfg['min_singular_ratio'] <= 0 or
            cfg['max_rotation_rad'] <= 0 or cfg['max_focal_log_scale'] <= 0 or
            cfg['max_baseline_slope'] <= 0 or cfg['min_nonplanar_disparity_span'] <= 0 or
            cfg['full6_min_singular_ratio'] <= 0 or cfg['nonplanar_noise_multiplier'] <= 0 or
            cfg['max_iterations'] < 1):
        raise ValueError('invalid configuration value')
    bins = np.floor(m[:, :2] / 2.).astype(np.int64)
    _, inverse, multiplicities = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    support_weights = 1. / multiplicities[inverse] if len(m) else np.empty(0)
    unique_support = len(multiplicities)
    p = np.zeros(6)
    reasons = []
    converged = False
    if len(m) < cfg['min_matches']:
        reasons.append('insufficient_matches')
    else:
        for iteration in range(int(cfg['max_iterations'])):
            r = _full_residual(m, kl, kr, p)
            if not np.isfinite(r).all():
                reasons.append('nonfinite_projection')
                break
            med = _weighted_quantile(r, support_weights, .5)
            sigma = max(.05, 1.4826 * _weighted_quantile(np.abs(r - med), support_weights, .5))
            if iteration < 5:
                cutoff = max(1.5, 2.5 * sigma)
                w = np.minimum(1., cutoff / np.maximum(np.abs(r), 1e-12))
            else:
                cutoff = max(1.5 * cfg['inlier_threshold_px'], 4.685 * sigma)
                u = r / cutoff
                w = np.where(np.abs(u) < 1, (1 - u*u)**2, 0.)
            w *= support_weights
            j = _full_jacobian(m, kl, kr, p)
            try:
                step = np.linalg.lstsq(j * np.sqrt(w[:, None]), -r * np.sqrt(w), rcond=None)[0]
            except np.linalg.LinAlgError:
                reasons.append('solver_failure')
                break
            candidate = p + step
            if (not np.isfinite(candidate).all() or
                    np.linalg.norm(candidate[:3]) > cfg['max_rotation_rad'] or
                    abs(candidate[3]) > cfg['max_focal_log_scale'] or
                    np.linalg.norm(candidate[4:]) > cfg['max_baseline_slope']):
                reasons.append('parameter_bound_exceeded')
                break
            p = candidate
            if np.linalg.norm(step) < 1e-10 and iteration >= 5:
                converged = True
                break
    residual = _full_residual(m, kl, kr, p)
    inliers = np.isfinite(residual) & (np.abs(residual) <= cfg['inlier_threshold_px'])
    count = int(inliers.sum())
    support = float(support_weights[inliers].sum())
    fraction = support / unique_support if unique_support else 0.
    if unique_support < cfg['min_matches'] or support < cfg['min_matches']:
        reasons.append('insufficient_unique_support')
    if count < cfg['min_matches']:
        reasons.append('insufficient_inliers')
    if fraction < cfg['min_inlier_fraction']:
        reasons.append('insufficient_inlier_fraction')
    if not converged and 'insufficient_matches' not in reasons:
        reasons.append('not_converged')
    t, qvec, q = _baseline_frame(p[4:6])
    span = [0., 0.]
    cells = 0
    singular = []
    ratio = 0.
    nonplanar_span = 0.
    estimated_vertical_noise = 0.
    nonplanar_gate = cfg['min_nonplanar_disparity_span']
    if count:
        iw = support_weights[inliers]
        left = _rays(m[inliers, :2], kl) @ q.T
        right = _rays(m[inliers, 2:], kr, np.exp(p[3])) @ _rotation(p[:3]).T @ q.T
        xy = left[:, :2] / left[:, 2, None]
        xy_right = right[:, :2] / right[:, 2, None]
        span = [_weighted_quantile(xy[:, a], iw, .95) - _weighted_quantile(xy[:, a], iw, .05) for a in range(2)]
        cells = int(len(np.unique(np.floor(xy / .1).astype(int), axis=0)))
        j = _full_jacobian(m[inliers], kl, kr, p)
        singular = np.linalg.svd(j * np.sqrt(iw[:, None]), compute_uv=False).tolist()
        if len(singular) == 6 and singular[0] > 0:
            ratio = singular[-1] / singular[0]
        # An affine inverse-depth field covers every 3-D plane, including fronto-parallel.
        design = np.column_stack((np.ones(count), xy))
        disparity = xy[:, 0] - xy_right[:, 0]
        fit = np.linalg.lstsq(design * np.sqrt(iw[:, None]), disparity * np.sqrt(iw), rcond=None)[0]
        difference = disparity - design @ fit
        nonplanar_span = _weighted_quantile(difference, iw, .95) - _weighted_quantile(difference, iw, .05)
        vertical = residual[inliers] / kl[1, 1]
        center = _weighted_quantile(vertical, iw, .5)
        estimated_vertical_noise = 1.4826 * _weighted_quantile(np.abs(vertical - center), iw, .5)
        nonplanar_gate = max(nonplanar_gate, cfg['nonplanar_noise_multiplier'] * estimated_vertical_noise)
    if min(span) < cfg['min_axis_span'] or cells < cfg['min_occupied_cells']:
        reasons.append('insufficient_spatial_support')
    if ratio < cfg['full6_min_singular_ratio']:
        reasons.append('rank_or_conditioning_failure')
    if nonplanar_span < nonplanar_gate:
        reasons.append('insufficient_nonplanar_depth_support')
    return {'model': 'full6', 'method': 'experimental_exact_full_geometry_irls',
        'convention': 'R maps right rays into left frame; baseline direction points left to right; Q maps baseline to +x',
        'parameters': {'rotation_vector_rad': p[:3].tolist(), 'focal_scale': float(np.exp(p[3])),
                       'baseline_direction_left': t.tolist(), 'baseline_slopes_yz': p[4:6].tolist(),
                       'rectification_rotation_vector_rad': qvec.tolist()},
        'accepted': not reasons, 'metric_acceptance': 'unevaluated',
        'diagnostics': {'reasons': reasons, 'match_count': len(m), 'inlier_count': count,
            'inlier_fraction': fraction, 'unique_support_count': unique_support,
            'unique_inlier_support': support, 'support_bin_size_px': 2.,
            'inlier_threshold_px': cfg['inlier_threshold_px'],
            'median_abs_vertical_px': float(np.median(np.abs(residual))) if len(m) and np.isfinite(residual).all() else None,
            'spatial_span_normalized_5_95': span, 'occupied_angular_cells': cells,
            'singular_values': singular, 'singular_ratio': float(ratio), 'converged': converged,
            'nonplanar_disparity_span_normalized': float(nonplanar_span),
            'min_nonplanar_disparity_span': cfg['min_nonplanar_disparity_span'],
            'estimated_vertical_noise_normalized': float(estimated_vertical_noise),
            'nonplanar_noise_multiplier': cfg['nonplanar_noise_multiplier'],
            'effective_nonplanar_disparity_threshold': float(nonplanar_gate),
            'min_singular_ratio': cfg['full6_min_singular_ratio'],
            'bounds': {'max_rotation_rad': cfg['max_rotation_rad'],
                       'max_focal_log_scale': cfg['max_focal_log_scale'],
                       'max_baseline_slope': cfg['max_baseline_slope']}}}


def project_right_points(points_left, K_right, result, baseline):
    """Project left-frame metric points to original right pixels with known baseline.

    Baseline length is supplied externally, in the same units as points_left;
    it is never inferred from stereo matches. No result acceptance is implied.
    """
    points = np.asarray(points_left, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('points_left must be a finite N-by-3 array')
    _, _, kr = _inputs(np.empty((0, 4)), np.eye(3), K_right)
    baseline = float(baseline)
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('baseline must be positive and finite')
    params = result['parameters']
    r = np.asarray(params['rotation_vector_rad'], dtype=float)
    t = np.asarray(params.get('baseline_direction_left', [1., 0., 0.]), dtype=float)
    scale = float(params['focal_scale'])
    if (r.shape != (3,) or t.shape != (3,) or not np.isfinite(r).all() or
            not np.isfinite(t).all() or not np.isfinite(scale) or scale <= 0 or
            not np.isclose(np.linalg.norm(t), 1.)):
        raise ValueError('invalid fitted parameters')
    right = (points - baseline * t) @ _rotation(r)
    if np.any(right[:, 2] <= 0):
        raise ValueError('point is not in front of right camera')
    intrinsic = kr.copy()
    intrinsic[:2, :2] *= scale
    return _project(right, intrinsic)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON correspondence input path, or - for stdin')
    parser.add_argument('-o', '--output', help='result JSON path; defaults to stdout')
    args = parser.parse_args()
    try:
        if args.input == '-':
            data = json.load(sys.stdin)
        else:
            with open(args.input, encoding='utf-8') as f:
                data = json.load(f)
        result = fit_calibration(data['matches'], data['K_left'], data['K_right'], data.get('model', 'relative4'), data.get('config'))
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(encoded)
        else:
            sys.stdout.write(encoded)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
