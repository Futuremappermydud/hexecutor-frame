#!/usr/bin/env python3
"""Offline exposure-offset control using LEFT natural tracks and recorded motion.

No right measurements or tags enter point fitting or offset selection. Stereo
matching still selects the track population. Head translation scale defaults to
one for this diagnostic, not as metric truth. Timing, lens error, extrinsics and
tracking error can confound these scores; no calibration update is approved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from evaluate_stereo_temporal_calibration import extract, sha, K
from stereo_direct_tracks import track_window
from stereo_pose_constraints import recorded_motion


def score_left_tracks(observations, rotations, centers, intrinsic=K):
    """Fit one point per >=3-view track to fixed poses; ignore right columns."""
    rows = np.asarray(observations, float)
    rotations, centers = np.asarray(rotations, float), np.asarray(centers, float)
    tracks = [t for t in np.unique(rows[:, 1]) if np.sum(rows[:, 1] == t) >= 3]
    entries = []
    for track in tracks:
        subset = rows[rows[:, 1] == track]
        frames = subset[:, 0].astype(int)
        pixels = subset[:, 2:4]
        rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsic).T
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        directions = np.einsum('nij,nj->ni', rotations[frames], rays)
        cosines = np.clip(directions @ directions.T, -1., 1.)
        parallax = float(np.degrees(np.arccos(np.min(cosines))))
        entry = {'track_id': int(track), 'observations': len(subset), 'parallax_deg': parallax}
        if parallax < .5:
            entry['error'] = 'weak_parallax'; entries.append(entry); continue
        projectors = np.eye(3)[None] - directions[:, :, None]*directions[:, None, :]
        a = projectors.sum(axis=0)
        b = np.einsum('nij,nj->ni', projectors, centers[frames]).sum(axis=0)
        if np.linalg.cond(a) > 1e10:
            entry['error'] = 'singular'; entries.append(entry); continue
        point = np.linalg.solve(a, b)
        def residual(p):
            local = np.einsum('nji,nj->ni', rotations[frames], p-centers[frames])
            z = np.where(np.abs(local[:, 2]) > 1e-8, local[:, 2], 1e-8)
            projected = local @ intrinsic.T
            return (projected[:, :2]/z[:, None] - pixels).ravel()
        fit = least_squares(residual, point, loss='soft_l1', f_scale=1., max_nfev=100)
        local = np.einsum('nji,nj->ni', rotations[frames], fit.x-centers[frames])
        if not fit.success or np.any(local[:, 2] <= 0):
            entry['error'] = 'not_converged_or_behind_camera'; entries.append(entry); continue
        errors = residual(fit.x).reshape(-1, 2)
        entry.update(errors_px=errors.tolist())
        entries.append(entry)
    def summarize(heldout):
        selected = [e for e in entries if (e['track_id'] % 5 == 0) == heldout]
        valid = [e for e in selected if 'errors_px' in e]
        scores = {'requested_tracks': len(selected), 'valid_tracks': len(valid),
                  'track_ids': [e['track_id'] for e in valid]}
        if valid:
            errors = np.concatenate([e['errors_px'] for e in valid]).ravel()
            scores.update(scalar_count=len(errors), rms_px=float(np.sqrt(np.mean(errors**2))),
                          p90_abs_px=float(np.quantile(np.abs(errors), .9)),
                          mean_soft_l1=float(np.mean(2*(np.sqrt(1+errors**2)-1))))
        return scores
    return {'training': summarize(False), 'heldout': summarize(True), 'tracks': entries}


def audit(manifest_path, frozen_path, poses_path, config_path, offsets):
    if len(offsets) < 2 or len(set(offsets)) != len(offsets) or not np.isfinite(offsets).all():
        raise ValueError('at least two distinct finite offsets required')
    manifest = json.loads(Path(manifest_path).read_text())
    frames, observed = extract(manifest_path, json.loads(Path(frozen_path).read_text()))
    ordinals = sorted(frames)
    rows, diagnostics = track_window([frames[i] for i in ordinals])
    results = []
    for offset in offsets:
        rotations, translations, levers, provenance = recorded_motion(manifest_path, poses_path, config_path, offset)
        world_r = np.array([rotations[i] for i in ordinals])
        r = world_r[0].T @ world_r
        world_c = np.array([translations[i]+levers[i] for i in ordinals])
        centers = (world_c-world_c[0]) @ world_r[0]
        result = score_left_tracks(rows, r, centers)
        result.update(offset_ms=offset, pose_provenance=provenance)
        results.append(result)
    # No selection when candidate support changes or any eligible track fails.
    populations = [result['training']['track_ids'] for result in results]
    eligible = (bool(populations[0]) and all(p == populations[0] for p in populations) and
                all(r['training']['valid_tracks'] == r['training']['requested_tracks'] for r in results))
    best = min(results, key=lambda r: r['training']['mean_soft_l1'])['offset_ms'] if eligible else None
    return {'source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path), 'frozen_sha256': sha(frozen_path),
            'extractor_sha256': sha(Path(__file__).with_name('evaluate_stereo_temporal_calibration.py')),
            'tracking_sha256': sha(Path(__file__).with_name('stereo_direct_tracks.py')),
            'observed_provenance': observed['input_sha256'], 'tracking_diagnostics': diagnostics,
            'ordinals': ordinals, 'head_translation_scale_assumed_for_control': 1.,
            'offsets': results, 'training_choice_offset_ms': best,
            'selection': 'Lowest training-only mean soft-L1, only on identical fully valid >=3-view track support. Held-out tracks do not select offset.',
            'metric_acceptance': 'unevaluated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'frozen', 'poses', 'config', 'output'):
        parser.add_argument(name)
    parser.add_argument('--offset-ms', action='append', type=float, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.frozen, args.poses, args.config, args.offset_ms)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
