#!/usr/bin/env python3
"""Audit fitted camera motion against existing capture-time headset pose logs.

Relative rotation ANGLES are invariant to unknown constant camera-to-head and
world-frame rotations. This audit does not need their axes, tag geometry, or a
metric SLAM translation assumption. Translation magnitudes are descriptive and
include a camera/head lever-arm difference; they are not a metric certificate.
"""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angle(a, b):
    # atan2 is stable close to identity; logged matrices have rounded entries.
    q = a.T @ b
    sine = np.linalg.norm([q[2, 1]-q[1, 2], q[0, 2]-q[2, 0], q[1, 0]-q[0, 1]]) / 2
    return float(np.arctan2(sine, (np.trace(q)-1)/2))


def associate_poses(rows, targets, max_gap_ns, interpolate=False):
    if not isinstance(interpolate, bool):
        raise ValueError('interpolate must be boolean')
    if max_gap_ns < 0 or not rows:
        raise ValueError('pose rows required and max gap must be nonnegative')
    times = [int(r['mono_ns']) for r in rows]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('pose timestamps must be strictly increasing')
    selected, diagnostics = [], []
    for target in targets:
        idx = bisect.bisect_left(times, target)
        nearby = [j for j in (idx-1, idx) if 0 <= j < len(times)]
        j = min(nearby, key=lambda i: abs(times[i]-target))
        gap = abs(times[j]-target)
        row = rows[j]
        # Never jump across an invalid nearest pose to a more distant valid one.
        if row['valid'] != '1' or row['result'] != '200' or gap > max_gap_ns:
            raise ValueError('invalid, degraded, or distant nearest headset pose')
        pose = np.array([[float(row[f'm{r}{c}']) for c in range(4)] for r in range(3)])
        r = pose[:, :3]
        if (not np.isfinite(pose).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(r), 1., atol=1e-4)):
            raise ValueError('headset pose is not a finite proper rigid transform')
        diagnostic = {'target_monotonic_ns': int(target), 'pose_monotonic_ns': times[j], 'gap_ns': gap}
        if interpolate:
            diagnostic['association_mode'] = 'exact' if times[j] == target else 'interpolated'
            if times[j] != target:
                if idx == 0 or idx == len(times):
                    raise ValueError('interpolation requires a bracket; no extrapolation')
                endpoints = []
                for index in (idx-1, idx):
                    endpoint = rows[index]
                    if endpoint['valid'] != '1' or endpoint['result'] != '200' or abs(times[index]-target) > max_gap_ns:
                        raise ValueError('invalid, degraded, or distant interpolation bracket')
                    value = np.array([[float(endpoint[f'm{a}{b}']) for b in range(4)] for a in range(3)])
                    rotation = value[:, :3]
                    if not np.isfinite(value).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
                        raise ValueError('interpolation endpoint is not a rigid transform')
                    endpoints.append(value)
                from scipy.spatial.transform import Rotation, Slerp
                alpha = (target-times[idx-1])/(times[idx]-times[idx-1])
                rotation = Slerp([0., 1.], Rotation.from_matrix(np.array([v[:, :3] for v in endpoints])))([alpha]).as_matrix()[0]
                translation = (1.-alpha)*endpoints[0][:, 3]+alpha*endpoints[1][:, 3]
                pose = np.column_stack((rotation, translation))
                diagnostic.update(bracket_pose_times_ns=[times[idx-1], times[idx]], interpolation_fraction=alpha)
        selected.append(pose)
        diagnostics.append(diagnostic)
    return np.asarray(selected), diagnostics


def rotation_comparison(camera_rotations, head_rotations):
    if len(camera_rotations) != len(head_rotations) or len(camera_rotations) < 2:
        raise ValueError('matching multi-frame rotation arrays required')
    pairs = []
    for a in range(len(camera_rotations)):
        for b in range(a+1, len(camera_rotations)):
            ca, ha = angle(camera_rotations[a], camera_rotations[b]), angle(head_rotations[a], head_rotations[b])
            pairs.append({'frame_indices': [a, b], 'camera_angle_deg': float(np.degrees(ca)),
                          'head_angle_deg': float(np.degrees(ha)), 'angle_difference_deg': float(np.degrees(ca-ha))})
    errors = np.array([p['angle_difference_deg'] for p in pairs])
    return {'pairs': pairs, 'max_abs_angle_difference_deg': float(np.max(np.abs(errors))),
            'rms_angle_difference_deg': float(np.sqrt(np.mean(errors**2)))}


def audit(manifest_path, result_path, poses_path, exposure_offset_ms, max_gap_ms=25.):
    if not np.isfinite(exposure_offset_ms) or not np.isfinite(max_gap_ms) or max_gap_ms < 0:
        raise ValueError('finite exposure offset and nonnegative maximum gap required')
    manifest = json.loads(Path(manifest_path).read_text())
    result = json.loads(Path(result_path).read_text())
    if sha(manifest_path) != result['manifest_sha256']:
        raise ValueError('manifest does not match replay result')
    pairs = [json.loads(line) for line in Path(manifest['pairs']).read_text().splitlines() if line.strip()]
    by_left = {p['left_file']: p for p in pairs}
    if len(by_left) != len(pairs):
        raise ValueError('duplicate pair filename')
    frames = {f['ordinal']: f for f in manifest['frames']}
    with Path(poses_path).open(newline='') as stream:
        head_rows = list(csv.DictReader(stream))
    windows = []
    for window in result['windows']:
        entry = {'start': window['start'], 'models': {}}
        targets = [int(by_left[Path(frames[i]['left']).name]['left_monotonic_ns']) -
                   round(exposure_offset_ms*1e6) for i in window['ordinals']]
        try:
            head, association = associate_poses(head_rows, targets, round(max_gap_ms*1e6))
        except ValueError as exc:
            entry['error'] = str(exc)
            windows.append(entry)
            continue
        entry['association'] = association
        entry['head_max_translation_from_first_m'] = float(np.max(np.linalg.norm(head[:, :, 3]-head[0, :, 3], axis=1)))
        entry['head_max_rotation_from_first_deg'] = float(np.degrees(max(angle(head[0, :, :3], r) for r in head[:, :, :3])))
        for name, model in window.items():
            if not name.startswith('temporal_') or not isinstance(model, dict):
                continue
            fit = model.get('fit', {})
            poses = {p['frame_id']: p for p in fit.get('poses', [])}
            if not all(i in poses for i in range(len(targets))):
                entry['models'][name] = {'available': False}
                continue
            camera = np.array([cv2.Rodrigues(np.array(poses[i]['rotation_vector_rad'], float))[0] for i in range(len(targets))])
            comparison = rotation_comparison(camera, head[:, :, :3])
            comparison.update(available=True, training_accepted=fit.get('accepted'),
                              camera_max_translation_from_first_m=fit['diagnostics']['max_translation_m'])
            entry['models'][name] = comparison
        windows.append(entry)
    return {'source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path),
            'result_sha256': sha(result_path), 'poses_sha256': sha(poses_path),
            'pairs_sha256': sha(manifest['pairs']), 'exposure_offset_ms': exposure_offset_ms,
            'max_gap_ms': max_gap_ms, 'windows': windows,
            'scope': 'Relative rotation-angle consistency only. Head tracking is independent of this BA but may use the same IR cameras; no metric approval, axis/extrinsic validation, or scale truth.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'result', 'poses', 'output'):
        parser.add_argument(key)
    parser.add_argument('--exposure-offset-ms', type=float, required=True)
    parser.add_argument('--max-gap-ms', type=float, default=25.)
    args = parser.parse_args()
    result = audit(args.manifest, args.result, args.poses, args.exposure_offset_ms, args.max_gap_ms)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
