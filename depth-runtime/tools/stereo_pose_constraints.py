"""Offline recorded headset rotations for the frozen IR rectified camera basis.

Uses explicit factory geometry and device-config files. No head translations,
tags, or fitted calibration enter these rotations. Historical capture identity
of a device-config snapshot must be checked by the caller; a current copy alone
does not establish its historical identity.
"""
import csv
import json
from pathlib import Path
import numpy as np
from audit_stereo_temporal_pose import associate_poses, sha


def basis(record):
    x, z = np.array(record['plus_x'], float), np.array(record['plus_z'], float)
    r = np.column_stack((x, np.cross(z, x), z))
    if (not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(r), 1., atol=1e-5)):
        raise ValueError('invalid coordinate basis')
    return r


def rectified_in_head(calibration, device_config):
    # Factory rig is expressed in the first (slam_right/frontCam0) camera.
    if calibration['cameras'][0]['sourceCamera'] != 'slam_right':
        raise ValueError('expected slam_right as calibration coordinate origin')
    cameras = {c['sourceCamera']: c for c in calibration['cameras']}
    left, right = [cameras[name]['extrinsics'] for name in ('slam_left', 'slam_right')]
    rl, rr = basis(left), basis(right)
    if not np.allclose(rr, np.eye(3), atol=1e-6) or not np.allclose(right['position'], 0.):
        raise ValueError('expected identity slam_right origin')
    baseline = np.array(right['position'], float) - np.array(left['position'], float)
    if not np.isfinite(baseline).all() or np.linalg.norm(baseline) <= 0:
        raise ValueError('invalid baseline')
    x = baseline / np.linalg.norm(baseline)
    z = rl[:, 2] + rr[:, 2]
    z -= x * (x @ z)
    if np.linalg.norm(z) < 1e-8:
        raise ValueError('degenerate rectification basis')
    z /= np.linalg.norm(z)
    rectified = np.column_stack((x, np.cross(z, x), z))
    head_from_cal = basis(device_config['head']).T @ basis(device_config['cv']['cad_from_cal'])
    return head_from_cal @ rectified


def recorded_motion(manifest_path, poses_path, device_config_path, exposure_offset_ms, max_gap_ms=25., interpolate_poses=False):
    if not np.isfinite(exposure_offset_ms) or not np.isfinite(max_gap_ms) or max_gap_ms < 0:
        raise ValueError('finite offset and nonnegative max gap required')
    manifest = json.loads(Path(manifest_path).read_text())
    calibration_path = manifest['calibration']
    calibration = json.loads(Path(calibration_path).read_text())
    config = json.loads(Path(device_config_path).read_text())
    if any(c['extrinsics'].get('__units') != 'mm' for c in calibration['cameras']
           if c['sourceCamera'] in ('slam_left', 'slam_right')):
        raise ValueError('camera factory positions must explicitly use mm')
    extrinsic = rectified_in_head(calibration, config)
    left = next(c for c in calibration['cameras'] if c['sourceCamera'] == 'slam_left')
    cad_from_cal, head_config = config['cv']['cad_from_cal'], config['head']
    origin_in_head = basis(head_config).T @ (basis(cad_from_cal) @
        (np.array(left['extrinsics']['position'], float)/1000.) +
        np.array(cad_from_cal['position'], float) - np.array(head_config['position'], float))
    pairs = [json.loads(line) for line in Path(manifest['pairs']).read_text().splitlines() if line.strip()]
    by_left = {p['left_file']: p for p in pairs}
    if len(by_left) != len(pairs):
        raise ValueError('duplicate pair filename')
    frames = sorted(manifest['frames'], key=lambda f: f['ordinal'])
    if len({f['ordinal'] for f in frames}) != len(frames):
        raise ValueError('duplicate frame ordinal')
    targets = []
    for frame in frames:
        pair = by_left[Path(frame['left']).name]
        if pair['right_file'] != Path(frame['right']).name:
            raise ValueError('left/right pair mismatch')
        targets.append(int(pair['left_monotonic_ns']) - round(exposure_offset_ms*1e6))
    with Path(poses_path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    head, associations = associate_poses(rows, targets, round(max_gap_ms*1e6), interpolate=interpolate_poses)
    rotations = {f['ordinal']: h[:, :3] @ extrinsic for f, h in zip(frames, head)}
    # Logged matrices are rounded. Project to SO(3), rejecting malformed inputs
    # above rather than silently repairing nonrigid/invalid logs.
    for key, r in rotations.items():
        u, _, v = np.linalg.svd(r)
        rotations[key] = u @ v
    translations = {f['ordinal']: h[:, 3] for f, h in zip(frames, head)}
    levers = {f['ordinal']: h[:, :3] @ origin_in_head for f, h in zip(frames, head)}
    return rotations, translations, levers, {'poses_sha256': sha(poses_path), 'device_config_sha256': sha(device_config_path),
        'calibration_sha256': sha(calibration_path), 'pairs_sha256': sha(manifest['pairs']),
        'adapter_sha256': sha(__file__), 'association_source_sha256': sha(Path(__file__).with_name('audit_stereo_temporal_pose.py')), 'exposure_offset_ms': exposure_offset_ms, 'max_gap_ms': max_gap_ms, 'interpolate_poses': interpolate_poses,
        'association': associations, 'rectified_in_head_rotation': extrinsic.tolist(), 'camera_origin_in_head_m': origin_in_head.tolist(),
        'scope': 'Recorded pose trajectories; caller selects rotations only or also head translations up to a fitted scale with metric lever offsets. Stereo baseline supplies scale. Device-config capture identity and pose accuracy are assumptions to validate.'}


def recorded_rotations(*args, **kwargs):
    rotations, _, _, provenance = recorded_motion(*args, **kwargs)
    return rotations, provenance
