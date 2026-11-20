#!/usr/bin/env python3
"""Explicit offline KB4 lens ablation; natural fitting and external tag scoring."""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from evaluate_stereo_temporal_calibration import extract, sha, K
from audit_stereo_projective_matching import orb_seeds
from stereo_photometric_matching import refine_matches
from stereo_lens_calibration import calibrated_left_camera, CameraRayModel, fit_lens_calibration, epipolar_residual, project_right


def camera_models(manifest, observed):
    if sha(manifest['rectifier_source']) != observed['rectifier_sha256']:
        raise ValueError('rectifier identity changed')
    spec = importlib.util.spec_from_file_location('lens_ablation_rectifier', manifest['rectifier_source'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rect = module.Rectifier(module.load_rig(manifest['calibration']))
    def model(rotation, lens):
        return CameraRayModel(K, rotation, [lens.fx, lens.fy, lens.cx, lens.cy, *lens.k])
    return model(rect.RL, rect.kl), model(rect.RR, rect.kr), rect.baseline


def tag_score(observations, left, right, parameters, baseline, tag_size):
    left = calibrated_left_camera(left,parameters)
    half = tag_size/2
    obj = np.array([[-half, -half, 0.], [half, -half, 0.], [half, half, 0.], [-half, half, 0.]])
    errors = []
    for observation in observations:
        rays = left.rays(observation['left'], parameters[4])
        corrected = rays @ K.T
        pixels = corrected[:, :2]/corrected[:, 2, None]
        ok, rv, tv = cv2.solvePnP(obj, pixels, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        points = obj @ cv2.Rodrigues(rv)[0].T + tv.ravel()
        predicted = project_right(points, right, parameters, baseline)
        errors.extend(predicted - np.asarray(observation['right']))
    errors = np.asarray(errors).reshape(-1, 2)
    return {'right_corner_count': len(errors), 'rms_xy_px': np.sqrt(np.mean(errors**2, axis=0)).tolist() if len(errors) else None,
            'scope': 'External development validation only; left-only PnP with candidate lens, supplied tag size/baseline, conditional lens and planar-pose truth.'}


def run(manifest_path, frozen_path):
    manifest = json.loads(Path(manifest_path).read_text())
    frames, observed = extract(manifest_path, json.loads(Path(frozen_path).read_text()))
    left, right, baseline = camera_models(manifest, observed)
    metadata = {f['ordinal']: f for f in manifest['frames']}
    pairs = {}
    for ordinal, frame in frames.items():
        seeds = orb_seeds(frame['images'], frame['masks'])
        pairs[ordinal] = refine_matches(frame['images'], frame['masks'], seeds)
    train_ids = sorted(i for i in pairs if metadata[i]['split'] == 'train')
    held_ids = sorted(i for i in pairs if metadata[i]['split'] == 'heldout')
    def collect(ids):
        return np.concatenate([pairs[i] for i in ids]) if len(ids) else np.empty((0, 4))
    train, held = collect(train_ids), collect(held_ids)
    observations = json.loads(Path(manifest['observations']).read_text())
    tags = [o for o in observations if o['ordinal'] in held_ids]
    def statistics(residual):
        return {'count': len(residual), 'median_px': float(np.median(residual)) if len(residual) else None,
                'p90_abs_px': float(np.quantile(np.abs(residual), .9)) if len(residual) else None,
                'rms_px': float(np.sqrt(np.mean(residual**2))) if len(residual) else None}
    null = np.zeros(6)
    result = {'source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path), 'frozen_sha256': sha(frozen_path),
              'estimator_sha256': sha(Path(__file__).with_name('stereo_lens_calibration.py')),
              'matcher_sha256': sha(Path(__file__).with_name('stereo_photometric_matching.py')),
              'seed_source_sha256': sha(Path(__file__).with_name('audit_stereo_projective_matching.py')),
              'extractor_sha256': sha(Path(__file__).with_name('evaluate_stereo_temporal_calibration.py')),
              'observed_provenance': observed, 'baseline_m': baseline,
              'train_count': len(train), 'heldout_count': len(held),
              'uncorrected_heldout_epipolar': statistics(epipolar_residual(held, left, right, null)),
              'uncorrected_external_tags': tag_score(tags, left, right, null, baseline, manifest['tag_size_m']), 'models': {}}
    probes = np.array([[x*z, y*z, z] for z in (.7, 1.5, 2.5) for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
    for estimate_lens in (False, True):
        fit = fit_lens_calibration(train, left, right, estimate_lens)
        entry = {'fit': fit, 'heldout_epipolar': statistics(epipolar_residual(held, left, right, fit['parameters_vector'])),
                 'external_tags': tag_score(tags, left, right, fit['parameters_vector'], baseline, manifest['tag_size_m'])}
        subsets = []
        for ids in np.array_split(train_ids, 4):
            subsets.append(fit_lens_calibration(collect(ids), left, right, estimate_lens))
        entry['temporal_quarter_fits'] = subsets
        projections = [project_right(probes, right, np.array(f['parameters_vector']), baseline) for f in [fit, *subsets]]
        left_projections = [calibrated_left_camera(left,f['parameters_vector']).project(probes, np.array(f['parameters_vector'])[4]) for f in [fit, *subsets]]
        entry['max_horizontal_probe_spread_px'] = float(np.max(np.ptp(np.array(projections)[:, :, 0], axis=0)))
        entry['max_left_horizontal_probe_spread_px'] = float(np.max(np.ptp(np.array(left_projections)[:, :, 0], axis=0)))
        disparities = np.array(left_projections)[:, :, 0] - np.array(projections)[:, :, 0]
        entry['max_disparity_probe_spread_px'] = float(np.max(np.ptp(disparities, axis=0)))
        entry['stability_passed'] = bool(all(f['accepted'] for f in [fit, *subsets]) and
            max(entry['max_horizontal_probe_spread_px'], entry['max_left_horizontal_probe_spread_px'],
                entry['max_disparity_probe_spread_px']) <= .3)
        result['models'][fit['model']] = entry
    result['metric_acceptance'] = 'unevaluated'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'frozen', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    result = run(args.manifest, args.frozen)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
