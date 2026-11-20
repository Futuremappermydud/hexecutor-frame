#!/usr/bin/env python3
"""Offline shared physical calibration across explicitly independent windows.

Each window has its own pose/point gauge. Only training-accepted windows enter
shared fitting; both physical ablations must accept for eligibility. External
held-out tags never select windows or parameters. Rejected inputs stay reported.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate_stereo_temporal_lens import prepare_window, heldout_prediction
from evaluate_stereo_temporal_calibration import sha, K, select_external_tags
from evaluate_stereo_lens_calibration import tag_score
from stereo_temporal_calibration import fit_temporal_calibration
from stereo_lens_calibration import calibrated_left_camera, project_right

BUDGET = {'max_frames': 60, 'max_tracks': 1000, 'max_observations': 9000}


def stereo_frame_identities(manifest, observed):
    hashes = observed['input_sha256']
    identities = [(hashes[str(Path(frame['left']).resolve())], hashes[str(Path(frame['right']).resolve())])
                  for frame in manifest['frames']]
    if len(set(identities)) != len(identities):
        raise ValueError('window repeats a raw stereo frame')
    return set(identities)


def probe_stability(fits, left, right, baseline, threshold=.3):
    report = {'stability_passed': False, 'threshold_px': threshold, 'fit_count': len(fits),
              'metric_acceptance': 'unevaluated'}
    if len(fits) < 3 or not all(f.get('accepted') for f in fits):
        report['reasons'] = ['insufficient_or_rejected_training_subset_fits']; return report
    probes = np.array([[x*z, y*z, z] for z in (.7, 1.5, 2.5) for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
    try:
        lp = np.array([calibrated_left_camera(left,f['parameters_vector']).project(probes, f['parameters_vector'][4])[:, 0] for f in fits])
        rp = np.array([project_right(probes, right, f['parameters_vector'], baseline)[:, 0] for f in fits])
        if not np.isfinite(lp).all() or not np.isfinite(rp).all():
            raise ValueError('nonfinite probe projection')
    except (ValueError, KeyError, IndexError):
        report['reasons'] = ['invalid_probe_projection']; return report
    spreads = {'left': float(np.max(np.ptp(lp, axis=0))), 'right': float(np.max(np.ptp(rp, axis=0))),
               'disparity': float(np.max(np.ptp(lp-rp, axis=0)))}
    report.update(max_horizontal_spread_px=spreads, stability_passed=max(spreads.values()) <= threshold,
                  reasons=[] if max(spreads.values()) <= threshold else ['horizontal_spread_exceeded'])
    return report


def combine_windows(prepared, indices):
    rows, rotations, groups = [], [], []
    for index in indices:
        _, train, _, rot, _, _, _ = prepared[index]
        if np.max(train[:, 1], initial=0.) >= 1000000:
            raise ValueError('track-ID allocation exceeded')
        obs = train.copy(); obs[:, 0] += index*100; obs[:, 1] += index*1000000
        rows.append(obs); rotations.extend(rot); groups.extend([index]*len(rot))
    return np.concatenate(rows), np.asarray(rotations), groups


def run(windows, poses_path, config_path, exposure_offset_ms, external_tag_manifest=None, refine_head_rotation=False,
        prune_support=False):
    if not 1 <= len(windows) <= 5:
        raise ValueError('one to five explicit windows required')
    prepared, entries, eligible, seen_frames = [], [], [], set()
    for index, (manifest, frozen) in enumerate(windows):
        item = prepare_window(manifest, frozen, poses_path, config_path, exposure_offset_ms, prune_support=prune_support)
        data, train, held, rotations, cameras, _, _ = item
        identities = stereo_frame_identities(json.loads(Path(manifest).read_text()), data['observed_provenance'])
        if identities & seen_frames:
            raise ValueError('independent windows reuse a raw stereo frame')
        seen_frames.update(identities)
        prepared.append(item)
        fits = {model: fit_temporal_calibration(train, K, K, data['baseline_m'], model=model,
                    camera_rotations=rotations, camera_ray_models=cameras,
                    refine_camera_rotation=refine_head_rotation) for model in ('physical4', 'physical6')}
        if all(f['accepted'] for f in fits.values()):
            eligible.append(index)
        entries.append({'manifest': str(Path(manifest).resolve()), 'frozen': str(Path(frozen).resolve()),
                        'prepared': data, 'standalone_training_fits': fits, 'eligible': index in eligible})
        print('prepared window', index, 'eligible', index in eligible, flush=True)
    left, right = prepared[0][4]; baseline = prepared[0][0]['baseline_m']
    for item in prepared[1:]:
        if not np.isclose(item[0]['baseline_m'], baseline, atol=1e-9, rtol=0):
            raise ValueError('window baseline differs')
        for a, b in zip(item[4], (left, right)):
            if not all(np.allclose(getattr(a, name), getattr(b, name), atol=1e-9, rtol=0)
                       for name in ('intrinsic', 'rotation', 'lens')):
                raise ValueError('windows do not share the same nominal physical cameras')
    result = {'source_sha256': sha(__file__), 'estimator_sha256': sha(Path(__file__).with_name('stereo_temporal_calibration.py')),
              'ray_model_sha256': sha(Path(__file__).with_name('stereo_lens_calibration.py')),
              'preparation_sha256': sha(Path(__file__).with_name('evaluate_stereo_temporal_lens.py')),
              'windows': entries, 'eligible_window_indices': eligible,
              'omitted_window_indices': [i for i in range(len(windows)) if i not in eligible],
              'budget': BUDGET, 'models': {}, 'metric_acceptance': 'unevaluated',
              'head_rotation_refinement_enabled': refine_head_rotation,
              'independent_raw_frame_identity_check': 'no repeated stereo hash pairs',
              'selection': 'Intersection of physical4/physical6 standalone training acceptance, before reading external tag observations. No support or motion threshold relaxed.'}
    if not eligible:
        result['reasons'] = ['no_training_accepted_windows']; return result
    excluded = set()
    for i in eligible:
        manifest = json.loads(Path(windows[i][0]).read_text())
        excluded.update(f['source_pair_id'] for f in manifest['frames'])
    tags, provenance, tag_size = [], None, 0.12
    if external_tag_manifest:
        tags, provenance = select_external_tags(external_tag_manifest, excluded)
        external = json.loads(Path(external_tag_manifest).read_text()); tag_size = external['tag_size_m']
        if sha(external['calibration']) != prepared[0][0]['observed_provenance']['calibration_sha256']:
            raise ValueError('external validation uses different physical camera snapshot')
    result['external_tag_provenance'] = provenance
    result['uncorrected_external_tags'] = tag_score(tags, left, right, np.zeros(6), baseline, tag_size)
    for model in ('physical4', 'physical6'):
        def fit_indices(indices):
            rows, rotations, groups = combine_windows(prepared, indices)
            return fit_temporal_calibration(rows, K, K, baseline, config=BUDGET, model=model,
                camera_rotations=rotations, camera_ray_models=(left, right), frame_groups=groups,
                refine_camera_rotation=refine_head_rotation)
        fit = fit_indices(eligible)
        omitted_fits = [{'omitted_index': i, 'fit': fit_indices([j for j in eligible if j != i])}
                        for i in eligible] if len(eligible) >= 3 else []
        entry = {'fit': fit, 'leave_one_window_out': omitted_fits,
                 'training_consistency': probe_stability([fit, *[f['fit'] for f in omitted_fits]], left, right, baseline)}
        if fit['accepted']:
            entry['external_tags'] = tag_score(tags, left, right, fit['parameters_vector'], baseline, tag_size)
            entry['heldout_natural_by_window'] = {}
            for index in eligible:
                local = dict(fit)
                local['poses'] = [dict(p, frame_id=p['frame_id']-index*100) for p in fit['poses'] if p['group_id'] == index]
                entry['heldout_natural_by_window'][str(index)] = heldout_prediction(prepared[index][2], local, left, right, baseline)
        result['models'][model] = entry
        print('shared model', model, 'accepted', fit['accepted'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('poses', 'config', 'output'):
        parser.add_argument(name)
    parser.add_argument('--window', nargs=2, action='append', required=True, metavar=('MANIFEST', 'FROZEN'))
    parser.add_argument('--exposure-offset-ms', type=float, required=True)
    parser.add_argument('--external-tag-manifest')
    parser.add_argument('--refine-head-rotation', action='store_true')
    parser.add_argument('--prune-support', action='store_true')
    args = parser.parse_args()
    result = run(args.window, args.poses, args.config, args.exposure_offset_ms, args.external_tag_manifest,
                 args.refine_head_rotation, args.prune_support)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
