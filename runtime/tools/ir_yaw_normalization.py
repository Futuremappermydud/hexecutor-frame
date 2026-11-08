#!/usr/bin/env python3
"""Score saved IR motion-stereo matches with per-ray spherical yaw geometry.

Offline only: this tool opens no cameras and does not run depth inference.
Input is an Nx10 .npy array from the motion-stereo replay, with columns:
keyframe, time_s, left_u_px, left_v_px, distance_m, expected_disparity_px,
disparity_at_zero_yaw_px, disparity_at_perturbed_yaw_px, stereo_ncc, motion_ncc.
The perturbed match is retained for provenance, not used as a geometric scale.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def spherical_rays(u, v, width, height, step_deg):
    k = np.radians(step_deg)
    theta = (np.asarray(u) - (width - 1) / 2) * k
    phi = (np.asarray(v) - (height - 1) / 2) * k
    return np.stack([np.tan(theta), np.sin(phi), np.cos(phi)], axis=-1)


def yaw_sensitivity(u_right, v_right, *, width=512, height=640,
                    step_deg=0.145, roll_deg=0.4, delta_deg=-0.3):
    """Return disparity pixels/degree at each right-image ray.

    The right rectifier has R(yaw) = A Rx(pitch) Ry(yaw) Rz(roll).
    Holding a physical raw-camera ray fixed gives
    ray(delta) = Rz(-roll) Ry(-delta) Rz(roll) ray(0).
    A and pitch cancel, so no rig translation or fitted motion scale is needed.
    A common finite angular step preserves the previous experiment's convention.
    """
    if (not np.isfinite([step_deg, roll_deg, delta_deg]).all()
            or step_deg <= 0 or delta_deg == 0 or width <= 0 or height <= 0):
        raise ValueError('Invalid spherical image or angular perturbation')
    rays = spherical_rays(u_right, v_right, width, height, step_deg)
    roll, delta = np.radians([roll_deg, -delta_deg])
    cr, sr, cy, sy = np.cos(roll), np.sin(roll), np.cos(delta), np.sin(delta)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    shifted = rays @ (rz.T @ ry @ rz).T
    u_shifted = (np.arctan2(shifted[..., 0], np.hypot(shifted[..., 1], shifted[..., 2]))
                 / np.radians(step_deg) + (width - 1) / 2)
    sensitivity = (np.asarray(u_right) - u_shifted) / delta_deg
    if not np.all(np.isfinite(sensitivity) & (sensitivity > 1e-6)):
        raise ValueError('Nonfinite or nonpositive per-ray yaw sensitivity')
    return sensitivity


def normalize_matches(samples, **geometry):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 10 or len(samples) == 0:
        raise ValueError('Expected a nonempty Nx10 motion-stereo sample array')
    if not np.isfinite(samples).all():
        raise ValueError('Input matches contain nonfinite values')
    scale = yaw_sensitivity(samples[:, 2] - samples[:, 6], samples[:, 3], **geometry)
    return (samples[:, 5] - samples[:, 6]) / scale, scale


def frame_interval(samples, yaw, mask, rng, repeats=500):
    frame_ids = samples[mask, 0]
    frames = np.unique(frame_ids)
    values = yaw[mask]
    groups = {f: values[frame_ids == f] for f in frames}
    boots = [np.median(np.concatenate([groups[f] for f in rng.choice(frames, len(frames))]))
             for _ in range(repeats)]
    return np.percentile(boots, [2.5, 97.5]).tolist()


def summarize(samples, yaw, scale, correction_deg):
    rng = np.random.default_rng(3)
    interval = frame_interval(samples, yaw, np.ones(len(samples), dtype=bool), rng)
    bins = {}
    for lo, hi in ((.3, .5), (.5, .75), (.75, 1), (1, 1.5), (1.5, 2.5), (2.5, 5)):
        mask = (samples[:, 4] >= lo) & (samples[:, 4] < hi)
        if mask.sum() >= 15:
            value = float(np.median(yaw[mask]))
            bins[f'{lo}-{hi}'] = {'samples': int(mask.sum()),
                'keyframes': len(np.unique(samples[mask, 0])), 'yaw_deg': value,
                'corrected_yaw_deg': value + correction_deg}
    selected = [bins[k]['yaw_deg'] for k in ['0.5-0.75', '0.75-1', '1-1.5', '1.5-2.5'] if k in bins]
    by_time = {}
    span = samples[:, 1].max()
    for part in range(4):
        lo, hi = span * part / 4, span * (part + 1) / 4
        mask = (samples[:, 1] >= lo) & (samples[:, 1] <= hi)
        if mask.sum() >= 15:
            by_time[f'{lo:.0f}-{hi:.0f}s'] = {'samples': int(mask.sum()),
                'yaw_deg': float(np.median(yaw[mask]))}
    return {'samples': len(samples), 'keyframes': len(np.unique(samples[:, 0])),
            'yaw_deg': float(np.median(yaw)), 'correction_deg': correction_deg,
            'corrected_yaw_deg': float(np.median(yaw) + correction_deg),
            'frame_bootstrap_95pct_deg': interval,
            'corrected_frame_bootstrap_95pct_deg': [v + correction_deg for v in interval],
            'interval_width_deg': float(np.ptp(interval)),
            'sensitivity_px_per_degree_p05_p50_p95': np.percentile(scale, [5, 50, 95]).tolist(),
            'by_distance': bins, 'by_time': by_time,
            'depth_bin_spread_deg': float(np.ptp(selected)) if len(selected) >= 2 else None,
            'depth_bins_covered': len(selected), 'required_depth_bins': 4,
            'uncertainty_scope': 'Frame resampling only; excludes calibration, pose scale, and scene/session variability.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--step-deg', type=float, default=.145)
    parser.add_argument('--roll-deg', type=float, default=.4)
    parser.add_argument('--delta-deg', type=float, default=-.3)
    parser.add_argument('--correction-deg', type=float, default=0,
                        help='Previously selected additive correction; never fitted by this tool')
    args = parser.parse_args()
    if not np.isfinite(args.correction_deg):
        parser.error('--correction-deg must be finite')
    samples = np.load(args.samples, allow_pickle=False)
    geometry = dict(width=args.width, height=args.height, step_deg=args.step_deg,
                    roll_deg=args.roll_deg, delta_deg=args.delta_deg)
    yaw, scale = normalize_matches(samples, **geometry)
    report = summarize(samples, yaw, scale, args.correction_deg)
    report.update(geometry=geometry, input=str(args.samples.resolve()),
                  input_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  acceptance_status='not_evaluated; this tool does not declare held-out acceptance')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'per-point-yaw.json').write_text(json.dumps(report, indent=2) + '\n')
    np.save(args.out/'per-point-yaw.npy', np.column_stack([yaw, scale]))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
