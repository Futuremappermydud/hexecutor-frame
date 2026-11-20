#!/usr/bin/env python3
"""Offline bounded physical calibration over 30 consecutive existing references.

The 30-frame interval is predeclared by the capture planner. Tracks use unchanged
matching/parallax checks; support pruning sees training tracks only. This driver
expands explicit offline budgets, never supplies a metric tracking translation.
An explicit free-rotations ablation optimizes orientations rather than fixing
recorded values; input track/parallax selection is held unchanged.
"""
import argparse
import json
from pathlib import Path
from evaluate_stereo_temporal_lens import prepare_window, heldout_prediction
from evaluate_stereo_temporal_calibration import sha, K
from evaluate_stereo_lens_calibration import tag_score
from stereo_temporal_calibration import fit_temporal_calibration

BUDGET = {'max_frames': 30, 'max_tracks': 400, 'max_observations': 9000}


def run(manifest, frozen, poses, config, exposure_offset_ms, external_tag_manifest=None, prune_outlier_support=False, fixed_rotation_initialization=False, estimate_baseline_direction=False, free_rotations=False, anchor_refinement=False):
    if free_rotations and fixed_rotation_initialization:
        raise ValueError('free rotations cannot use fixed-rotation initialization')
    result, train, held, rotations, cameras, tags, tag_size = prepare_window(
        manifest, frozen, poses, config, exposure_offset_ms, external_tag_manifest,
        prune_support=True, maximum_frames=30, training_track_limit=400, anchor_refinement=anchor_refinement)
    if len(result['ordinals']) != 30:
        raise ValueError('exactly 30 predeclared reference frames required')
    result['preparation_sha256'] = result['source_sha256']
    result['source_sha256'] = sha(__file__)
    result['budget'] = BUDGET
    result['pose_rotation_mode'] = 'free' if free_rotations else 'recorded'
    left, right = cameras; baseline = result['baseline_m']
    for model in (('physical_baseline6', 'physical_baseline8') if estimate_baseline_direction else ('physical4', 'physical6')):
        fit = fit_temporal_calibration(train, K, K, baseline, config=BUDGET, model=model,
                                      camera_rotations=None if free_rotations else rotations, camera_ray_models=cameras,
                                      fixed_rotation_initialization=fixed_rotation_initialization)
        support_refit = None
        if prune_outlier_support:
            from stereo_support_refit import refit_physical_support
            fit, support_refit = refit_physical_support(train, rotations, cameras, baseline, fit, BUDGET, use_recorded_rotations=not free_rotations)
        entry = {'fit': fit, 'heldout_natural': heldout_prediction(held, fit, left, right, baseline)}
        if support_refit is not None:
            entry['support_refit'] = support_refit
            result['support_refit_sha256'] = sha(Path(__file__).with_name('stereo_support_refit.py'))
        if fit['accepted']:
            entry['external_tags'] = tag_score(tags, left, right, fit['parameters_vector'], baseline, tag_size)
        result['models'][model] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'frozen', 'poses', 'config', 'output'):
        parser.add_argument(name)
    parser.add_argument('--exposure-offset-ms', type=float, required=True)
    parser.add_argument('--external-tag-manifest')
    parser.add_argument('--prune-outlier-support', action='store_true')
    parser.add_argument('--fixed-rotation-initialization', action='store_true')
    parser.add_argument('--estimate-baseline-direction', action='store_true')
    parser.add_argument('--anchor-refinement', action='store_true', help='Refine temporal endpoints against their original birth patches.')
    parser.add_argument('--free-rotations', action='store_true', help='Fit camera orientations from natural tracks; recorded rotations still define the fixed input parallax/support selection.')
    args = parser.parse_args()
    result = run(args.manifest, args.frozen, args.poses, args.config, args.exposure_offset_ms, args.external_tag_manifest,
                 args.prune_outlier_support, args.fixed_rotation_initialization, args.estimate_baseline_direction, args.free_rotations, args.anchor_refinement)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
