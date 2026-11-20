#!/usr/bin/env python3
"""Known stereo plane homographies with saved textures, explicit offline control.

The source photograph is used only as a planar texture. Synthetic scene depth,
slant, camera rotation and focal scale are known. This extends translation-only
controls; sensor distortion, occlusion and measured photometry are not simulated.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from evaluate_stereo_temporal_calibration import extract, sha, K
from evaluate_stereo_self_calibration import refine_matches as original_lk
from stereo_photometric_matching import refine_matches as roi_lk
from stereo_self_calibration import fit_calibration, project_right_points
from audit_stereo_photometric_control import transform

ROTATION = np.array([.004, -.008, .003])
FOCAL_SCALE = 1.008
BASELINE = .105
NORMALS = ([0., 0., 1.], [.5, 0., 1.], [0., -.5, 1.])
DEPTHS = (.7, 1.5, 2.5)


def plane_homography(kl, kr, baseline, rotation, normal, distance):
    normal = np.asarray(normal, float)
    normal = normal/np.linalg.norm(normal)
    relative = cv2.Rodrigues(np.asarray(rotation, float))[0]
    return kr @ relative.T @ (np.eye(3)-np.outer([baseline, 0., 0.], normal)/distance) @ np.linalg.inv(kl)


def apply_homography(points, homography):
    q = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return q[:, :2]/q[:, 2, None]


def orb_seeds(images, masks):
    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=4,
                         edgeThreshold=31, patchSize=31, fastThreshold=12)
    first, second = [orb.detectAndCompute(image, mask) for image, mask in zip(images, masks)]
    k0, d0 = first; k1, d1 = second
    if d0 is None or d1 is None or min(len(d0), len(d1)) < 2:
        return np.empty((0, 4))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward, reverse = matcher.knnMatch(d0, d1, k=2), matcher.knnMatch(d1, d0, k=2)
    good = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2
            for m, n in [pair] if m.distance < .75*n.distance}
    seeds = [[*k0[m.queryIdx].pt, *k1[m.trainIdx].pt] for pair in forward if len(pair) == 2
             for m, n in [pair] if m.distance < .75*n.distance and good.get(m.trainIdx) == m.queryIdx]
    return np.asarray(seeds).reshape(-1, 4)


def error_summary(matches, homography):
    errors = matches[:, 2:] - apply_homography(matches[:, :2], homography)
    result = {'retained': len(matches)}
    if len(errors):
        result.update(median_xy_px=np.median(errors, axis=0).tolist(),
                      p90_abs_xy_px=np.quantile(np.abs(errors), .9, axis=0).tolist(),
                      rms_xy_px=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                      count_over_one_px=int(np.sum(np.linalg.norm(errors, axis=1) > 1.)))
    return result


def run(manifest_path, frozen_path, ordinals, include_unique=False, include_affine=False):
    frames, observed = extract(manifest_path, json.loads(Path(frozen_path).read_text()))
    kr = K.copy(); kr[:2, :2] *= FOCAL_SCALE
    matchers = [('original_lk', original_lk), ('roi_lk', roi_lk)]
    if include_unique:
        from stereo_unique_matching import refine_matches as unique_lk
        matchers.append(('unique_roi_lk', unique_lk))
    if include_affine:
        from stereo_affine_matching import refine_matches as affine
        matchers.append(('affine_patch',affine))
    cases, pooled = [], {name: [] for name, _ in matchers}
    for ordinal in ordinals:
        frame = frames[ordinal]
        image, mask = frame['images'][0], frame['masks'][0]
        # Explicit additional footprint around the warped validity/tag masks.
        source_mask = cv2.erode(mask, np.ones((41, 41), np.uint8))
        for depth in DEPTHS:
            for normal in NORMALS:
                homography = plane_homography(K, kr, BASELINE, ROTATION, normal, depth)
                size = image.shape[::-1]
                warped = cv2.warpPerspective(image, homography, size, flags=cv2.INTER_LINEAR)
                target_mask = cv2.warpPerspective(mask, homography, size, flags=cv2.INTER_NEAREST)
                target_mask = cv2.erode(target_mask, np.ones((41, 41), np.uint8))
                for appearance in ('identity', 'gain_offset'):
                    images, masks = [image, transform(warped, appearance)], [source_mask, target_mask]
                    seeds = orb_seeds(images, masks)
                    entry = {'ordinal': ordinal, 'depth_m': depth, 'normal': normal,
                             'appearance': appearance, 'seed_count': len(seeds), 'matchers': {}}
                    for name, matcher in matchers:
                        points = matcher(images, masks, seeds)
                        entry['matchers'][name] = error_summary(points, homography)
                        pooled[name].append(points)
                    cases.append(entry)
        print('completed texture', ordinal, flush=True)
    probes = np.array([[x*z, y*z, z] for z in (.7, 1.5, 2.5) for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
    truth = {'parameters': {'rotation_vector_rad': ROTATION.tolist(), 'focal_scale': FOCAL_SCALE}}
    expected = project_right_points(probes, K, truth, BASELINE)
    fits = {}
    for name, arrays in pooled.items():
        points = np.concatenate(arrays) if arrays else np.empty((0, 4))
        fit = fit_calibration(points, K, K, model='relative4')
        prediction = project_right_points(probes, K, fit, BASELINE)
        fits[name] = {'fit': fit, 'truth_max_horizontal_probe_error_px': float(np.max(np.abs(prediction[:, 0]-expected[:, 0])))}
    return {'source_sha256': sha(__file__), 'extractor_sha256': sha(Path(__file__).with_name('evaluate_stereo_temporal_calibration.py')),
            'manifest_sha256': sha(manifest_path), 'frozen_sha256': sha(frozen_path),
            'matcher_sha256': {name: sha(Path(__file__).with_name(file)) for name, file in
                              [('original_lk', 'evaluate_stereo_self_calibration.py'), ('roi_lk', 'stereo_photometric_matching.py')]},
            'affine_matcher_sha256':sha(Path(__file__).with_name('stereo_affine_matching.py')) if include_affine else None,
            'unique_matcher_sha256': sha(Path(__file__).with_name('stereo_unique_matching.py')) if include_unique else None,
            'observed_provenance': observed['input_sha256'], 'opencv_version': cv2.__version__, 'numpy_version': np.__version__,
            'known_geometry': {'rotation_vector_rad': ROTATION.tolist(), 'right_focal_scale': FOCAL_SCALE, 'baseline_m': BASELINE,
                               'plane_distances_m': DEPTHS, 'normal_directions': NORMALS},
            'cases': cases, 'pooled_fits': fits,
            'caveats': ['Artificial planar textures with repeated source images; observations are correlated.',
                        'No sensor lens errors, measured illumination or physical occlusion is simulated.',
                        'Resampling can itself change local image appearance.', 'Known-geometry control, not real-capture metric approval.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'frozen', 'output'):
        parser.add_argument(name)
    parser.add_argument('--ordinal', type=int, action='append', required=True)
    parser.add_argument('--include-unique', action='store_true')
    parser.add_argument('--include-affine',action='store_true')
    args = parser.parse_args()
    result = run(args.manifest, args.frozen, args.ordinal, args.include_unique, args.include_affine)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
