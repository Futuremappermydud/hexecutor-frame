#!/usr/bin/env python3
"""Offline consistency guard for stereo calibration candidate updates.

Compares ORIGINAL right-camera projections of a fixed left-camera probe grid
across the candidate and separately fitted subsets. This is a consistency test,
not metric acceptance or proof of the cause of instability. Real rig changes
within a capture can also fail it. The default 0.3 threshold is on the supplied
pinhole pixel grid, not a claim about the historical angular-grid target.
"""
import argparse
import json
import sys

import numpy as np

from stereo_self_calibration import project_right_points


_XY = [-0.5, 0.0, 0.5]
_DEPTHS_M = [0.7, 1.5, 2.5]


def _intrinsics(k, name):
    k = np.asarray(k, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or
            k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1]) or abs(k[1, 0]) > 1e-12):
        raise ValueError(name + ' must be a finite pinhole intrinsic matrix')
    return k


def assess_update_stability(candidate, subset_fits, K_left, K_right, baseline_m,
                            max_horizontal_spread_px=0.3):
    """Check fitted models agree across a fixed grid, without assessing truth.

    The baseline is supplied in metres and is never estimated. Rejected fits
    invalidate the guard even if numerical projections can still be reported.
    Configuration errors raise ValueError; bad fits return rejection reasons.
    """
    _intrinsics(K_left, 'K_left')
    kr = _intrinsics(K_right, 'K_right')
    try:
        baseline = float(baseline_m)
        threshold = float(max_horizontal_spread_px)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('baseline and threshold must be positive finite numbers') from exc
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('baseline and threshold must be positive finite numbers')
    if not isinstance(subset_fits, list):
        raise ValueError('subset_fits must be a list')
    grid = {'normalized_x': list(_XY), 'normalized_y': list(_XY),
            'depth_m': list(_DEPTHS_M), 'point_count': 27}
    probes = np.array([[x * z, y * z, z] for z in _DEPTHS_M for y in _XY for x in _XY])
    reasons = []
    if len(subset_fits) < 2:
        reasons.append('insufficient_subset_fits')
    fits = [candidate, *subset_fits]
    projections = []
    for fit in fits:
        if not isinstance(fit, dict) or fit.get('accepted') is not True:
            if 'unaccepted_fit' not in reasons:
                reasons.append('unaccepted_fit')
        try:
            if not isinstance(fit, dict) or not isinstance(fit.get('parameters'), dict):
                raise ValueError('invalid fit parameters')
            projection = project_right_points(probes, kr, fit, baseline)
            if projection.shape != (27, 2) or not np.isfinite(projection).all():
                raise ValueError('nonfinite projection')
            projections.append(projection[:, 0])
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, np.linalg.LinAlgError):
            if 'invalid_fit_projection' not in reasons:
                reasons.append('invalid_fit_projection')
    spread = None
    if len(projections) == len(fits) and len(fits) >= 2:
        spread = float(np.max(np.ptp(np.stack(projections), axis=0)))
        if not np.isfinite(spread):
            spread = None
            reasons.append('invalid_fit_projection')
        elif spread > threshold:
            reasons.append('horizontal_spread_exceeded')
    return {'stability_passed': not reasons, 'reasons': reasons,
            'max_horizontal_spread_px': spread, 'threshold_px': threshold,
            'grid': grid, 'baseline_m': baseline,
            'fit_count': len(fits), 'metric_acceptance': 'unevaluated',
            'interpretation': 'Pinhole original-right projection consistency only; not metric accuracy or evidence of a cause.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON path or - for stdin')
    parser.add_argument('-o', '--output', help='output JSON path; defaults to stdout')
    args = parser.parse_args()
    try:
        if args.input == '-':
            data = json.load(sys.stdin)
        else:
            with open(args.input, encoding='utf-8') as f:
                data = json.load(f)
        result = assess_update_stability(data['candidate'], data['subset_fits'], data['K_left'],
                                         data['K_right'], data['baseline_m'],
                                         data.get('max_horizontal_spread_px', .3))
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
