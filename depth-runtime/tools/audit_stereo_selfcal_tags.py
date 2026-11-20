#!/usr/bin/env python3
"""Audit both left-only IPPE square poses against saved stereo tag corners.

No right observation selects a pose, fit, baseline, tag size, or yaw. The +/-5%
baseline and tag size sweeps are illustrative sensitivity checks, not tolerance
claims or calibration estimates. Outputs may contain captured derived points;
write them only to ignored local paths.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

K = np.array([[380., 0., 308.], [0., 380., 308.], [0., 0., 1.]])
SENSITIVITY = (0.95, 1., 1.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(points):
    h = points @ K.T
    return h[:, :2] / h[:, 2, None]


def audit(manifest_path, result_path):
    manifest = json.loads(Path(manifest_path).read_text())
    result = json.loads(Path(result_path).read_text())
    for source, key in ((manifest_path, 'manifest_sha256'),
                        (manifest['observations'], 'observations_sha256'),
                        (manifest['calibration'], 'calibration_sha256'),
                        (manifest['rectifier_source'], 'rectifier_sha256')):
        actual = sha(source)
        if result.get(key) != actual:
            raise ValueError(f'{key} does not match frozen evaluation result')
    observations = json.loads(Path(manifest['observations']).read_text())
    heldout = {frame['ordinal'] for frame in manifest['frames'] if frame['split'] == 'heldout'}
    observations = [o for o in observations if o['ordinal'] in heldout]
    size, baseline = float(manifest['tag_size_m']), float(result['baseline_m'])
    h = size / 2
    # General IPPE accepts this order, which matches the frozen evaluator.
    obj = np.array([[-h, -h, 0.], [h, -h, 0.], [h, h, 0.], [-h, h, 0.]], dtype=float)
    candidates = {'identity': None}
    candidates.update({k: v['fit'] for k, v in result['models'].items()})
    records = []
    for obs in observations:
        left, right = np.asarray(obs['left'], float), np.asarray(obs['right'], float)
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, left, K, None, flags=cv2.SOLVEPNP_IPPE)
        poses = []
        for rvec, tvec in zip(rvecs, tvecs):
            rotation, _ = cv2.Rodrigues(rvec)
            q = obj @ rotation.T + tvec.ravel()
            if np.all(q[:, 2] > 0):
                err = project(q) - left
                poses.append((float(np.sqrt(np.mean(err * err))), q))
        poses.sort(key=lambda x: x[0])  # only left reprojection decides branch ordering
        for branch, (left_rms, q) in enumerate(poses):
            for name, fit in candidates.items():
                p = fit['parameters'] if fit is not None else {}
                rotation, _ = cv2.Rodrigues(np.asarray(p.get('rotation_vector_rad', [0., 0., 0.])))
                direction = np.asarray(p.get('baseline_direction_left', [1., 0., 0.]))
                scale = float(p.get('focal_scale', 1.))
                for bfactor in SENSITIVITY:
                    for sfactor in SENSITIVITY:
                        qr = (q * sfactor - baseline * bfactor * direction) @ rotation
                        if np.any(qr[:, 2] <= 0):
                            continue
                        out = project(qr)
                        out = (out - K[:2, 2]) * scale + K[:2, 2]
                        error = out - right
                        records.append({'ordinal': obs['ordinal'], 'tag': obs['tag'], 'branch': branch,
                                        'left_rms_px': left_rms, 'center_left_px': left.mean(axis=0).tolist(),
                                        'fit': name, 'fit_accepted': fit['accepted'] if fit else None,
                                        'baseline_factor': bfactor, 'size_factor': sfactor,
                                        'right_mean_error_px': error.mean(axis=0).tolist(),
                                        'right_rms_px': float(np.sqrt(np.mean(error * error)))})
    return {'provenance': {'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'result': str(result_path), 'result_sha256': sha(result_path),
            'observations': manifest['observations'], 'observations_sha256': sha(manifest['observations']),
            'calibration_sha256': sha(manifest['calibration']), 'script_sha256': sha(__file__),
            'opencv_version': cv2.__version__, 'K': K.tolist(), 'baseline_m': baseline, 'tag_size_m': size,
            'heldout_only': True, 'pose_method': 'IPPE on left corners; branches sorted by left RMS; positive depth required',
            'sensitivity': 'independent baseline and tag-size factors 0.95, 1, 1.05; illustrative only; no branch or parameter selection using right observations'},
            'heldout_observation_count': len(observations), 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('result'); parser.add_argument('output')
    args = parser.parse_args()
    data = audit(args.manifest, args.result)
    Path(args.output).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
