#!/usr/bin/env python3
"""Offline temporal physical-KB4 ablation on fixed natural stereo tracks.

Recorded rotations fix the pose orientations; camera centers and points remain
free. No recorded translation scale or tags enter calibration fitting. Tags are
optional external development validation only. All acceptance is training-only.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from evaluate_stereo_temporal_calibration import extract, sha, K, cap_training_tracks, connectivity, select_external_tags
from evaluate_stereo_lens_calibration import camera_models, tag_score
from stereo_direct_tracks import track_window, filter_left_parallax
from stereo_pose_constraints import recorded_rotations
from stereo_temporal_calibration import fit_temporal_calibration
from stereo_lens_calibration import project_right, calibrated_left_camera


def validate_window_budget(maximum_frames, training_track_limit, continuous_window=False):
    """Explicit offline resource bounds; original defaults/ranges stay intact."""
    if not isinstance(continuous_window, bool):
        raise ValueError('continuous_window must be boolean')
    if (isinstance(maximum_frames, bool) or not isinstance(maximum_frames, int) or
            not 3 <= maximum_frames <= (120 if continuous_window else 60) or
            isinstance(training_track_limit, bool) or not isinstance(training_track_limit, int) or
            not 30 <= training_track_limit <= (1000 if continuous_window else 400)):
        raise ValueError('explicit bounded frame/track budgets required')


def heldout_prediction(rows, fit, left, right, baseline, refine_left_points=False):
    """Triangulate held-out left rays only with frozen training poses/lens.

    The null right projection is conditional on that SAME inferred left lens,
    motion and points, and is not an independent uncalibrated reconstruction.
    """
    if not isinstance(refine_left_points, bool):
        raise ValueError('refine_left_points must be boolean')
    refinements = []
    skipped = {'rejected_fit': 0, 'insufficient_views': 0, 'weak_parallax': 0,
               'singular': 0, 'behind_camera': 0, 'invalid_projection': 0}
    tracks = np.unique(rows[:, 1])
    if not fit.get('accepted'):
        skipped['rejected_fit'] = len(tracks)
        return {'available_tracks': 0, 'skipped_tracks': skipped, 'scores': {}}
    parameters = np.array(fit['parameters_vector'])
    left = calibrated_left_camera(left,parameters)
    poses = {p['frame_id']: (cv2.Rodrigues(np.array(p['rotation_vector_rad']))[0], np.array(p['translation_m']))
             for p in fit['poses']}
    predicted, null_errors, left_errors = [], [], []
    missing_pose_observations = int(np.sum(~np.isin(rows[:, 0], list(poses))))
    available = 0
    for track in tracks:
        subset = rows[rows[:, 1] == track]
        subset = subset[np.isin(subset[:, 0], list(poses))]
        if len(subset) < 2:
            skipped['insufficient_views'] += 1; continue
        rotations = np.array([poses[int(i)][0] for i in subset[:, 0]])
        centers = np.array([poses[int(i)][1] for i in subset[:, 0]])
        rays = left.rays(subset[:, 2:4], parameters[4])
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        directions = np.einsum('nij,nj->ni', rotations, rays)
        angle = np.arccos(np.clip(np.min(directions @ directions.T), -1., 1.))
        if angle < np.radians(.5):
            skipped['weak_parallax'] += 1; continue
        projectors = np.eye(3)[None] - directions[:, :, None]*directions[:, None, :]
        a = projectors.sum(axis=0)
        if np.linalg.cond(a) > 1e10:
            skipped['singular'] += 1; continue
        point = np.linalg.solve(a, np.einsum('nij,nj->ni', projectors, centers).sum(axis=0))
        if refine_left_points:
            from scipy.optimize import least_squares
            def left_residual(candidate):
                local_candidate = np.einsum('nji,nj->ni', rotations, candidate-centers)
                projected, valid = left.project_for_optimization(local_candidate, parameters[4])
                return (projected-subset[:, 2:4]).ravel()
            initial_error = left_residual(point)
            refined = least_squares(left_residual, point, max_nfev=100,
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
            refined_local = np.einsum('nji,nj->ni', rotations, refined.x-centers)
            final_error = left_residual(refined.x)
            _, final_valid = left.project_for_optimization(refined_local, parameters[4])
            used = bool(refined.success and np.isfinite(refined.x).all() and
                        np.all(refined_local[:, 2] > 0) and np.all(final_valid) and
                        np.sum(final_error**2) <= np.sum(initial_error**2))
            refinements.append({'track_id': int(track), 'used': used,
                                'initial_left_sse_px2': float(initial_error @ initial_error),
                                'final_left_sse_px2': float(final_error @ final_error)})
            if used:
                point = refined.x
        local = np.einsum('nji,nj->ni', rotations, point-centers)
        if np.any(local[:, 2] <= 0):
            skipped['behind_camera'] += 1; continue
        try:
            candidate = project_right(local, right, parameters, baseline)
            null = project_right(local, right, np.zeros(6), baseline)
            lp = left.project(local, parameters[4])
        except ValueError:
            skipped['invalid_projection'] += 1; continue
        available += 1
        predicted.extend(candidate-subset[:, 4:6]); null_errors.extend(null-subset[:, 4:6])
        left_errors.extend(lp-subset[:, 2:4])
    def summarize(errors):
        a = np.asarray(errors).reshape(-1, 2)
        return {'observations': len(a), 'rms_xy_px': np.sqrt(np.mean(a*a, axis=0)).tolist() if len(a) else None,
                'median_xy_px': np.median(a, axis=0).tolist() if len(a) else None,
                'p90_abs_xy_px': np.quantile(np.abs(a), .9, axis=0).tolist() if len(a) else None}
    return {'available_tracks': available, 'skipped_tracks': skipped,
            'left_point_refinement': {'enabled': refine_left_points, 'tracks': refinements,
                                      'scope': 'Left pixel least squares only; frozen training poses/lens; failed or non-improving refinement retains original linear point.'},
            'omitted_observations_missing_training_pose': missing_pose_observations,
            'scores': {'candidate_right': summarize(predicted), 'null_right_conditional_on_fitted_left': summarize(null_errors),
                       'left_reprojection': summarize(left_errors)},
            'scope': 'Held-out left-only point triangulation with frozen accepted training lens/poses; no held-out right pixels or tags enter geometry. Null right scores condition on the fitted left geometry.'}


def prepare_window(manifest_path, frozen_path, poses_path, config_path, exposure_offset_ms, external_tag_manifest=None,
                   prune_support=False, maximum_frames=12, training_track_limit=200, anchor_refinement=False, interpolate_poses=False, continuous_window=False):
    validate_window_budget(maximum_frames, training_track_limit, continuous_window)
    cv2.setNumThreads(1)
    manifest = json.loads(Path(manifest_path).read_text())
    frames, observed = extract(manifest_path, json.loads(Path(frozen_path).read_text()))
    ordinals = sorted(frames)
    if not 3 <= len(ordinals) <= maximum_frames:
        raise ValueError('manifest exceeds explicit frame budget')
    metadata = {f['ordinal']: f for f in manifest['frames']}
    if any(metadata[i]['split'] != 'train' for i in ordinals):
        raise ValueError('physical temporal training manifest must contain only training frames')
    left, right, baseline = camera_models(manifest, observed)
    world_rotations, pose_provenance = recorded_rotations(manifest_path, poses_path, config_path, exposure_offset_ms, interpolate_poses=interpolate_poses)
    rotations = np.array([world_rotations[i] for i in ordinals])
    relative_rotations = rotations[0].T @ rotations
    if anchor_refinement:
        rows, track_diagnostics = track_window([frames[i] for i in ordinals], anchor_refinement=True)
    else:
        rows, track_diagnostics = track_window([frames[i] for i in ordinals])
    rows, parallax_diagnostics = filter_left_parallax(rows, rotations, K, .5, 3)
    train_all, held = rows[rows[:, 1].astype(int) % 5 != 0], rows[rows[:, 1].astype(int) % 5 == 0]
    train = cap_training_tracks(train_all, training_track_limit)
    fitted_frames = list(range(len(ordinals)))
    support_diagnostics = {}
    if prune_support:
        from stereo_track_support import prune_training_support
        train, fitted_frames, support_diagnostics = prune_training_support(train, rotations, K)
        held = held[np.isin(held[:, 0], fitted_frames)]
        held, _ = filter_left_parallax(held, rotations, K, .5, 3)
        mapping = np.full(len(ordinals), -1, dtype=int)
        mapping[fitted_frames] = np.arange(len(fitted_frames))
        train = train.copy(); held = held.copy()
        train[:, 0] = mapping[train[:, 0].astype(int)]
        held[:, 0] = mapping[held[:, 0].astype(int)]
        selected_rotations = rotations[fitted_frames]
        relative_rotations = (selected_rotations[0].T @ selected_rotations if len(fitted_frames)
                              else np.empty((0, 3, 3)))
    if external_tag_manifest:
        tag_observations, tag_provenance = select_external_tags(external_tag_manifest, {metadata[i]['source_pair_id'] for i in ordinals})
        tag_manifest = json.loads(Path(external_tag_manifest).read_text())
        if sha(tag_manifest['calibration']) != observed['calibration_sha256']:
            raise ValueError('external tag lens snapshot differs')
        tag_size = tag_manifest['tag_size_m']
    else:
        tag_observations, tag_provenance, tag_size = [], None, manifest['tag_size_m']
    result = {'source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path), 'frozen_sha256': sha(frozen_path),
              'estimator_sha256': sha(Path(__file__).with_name('stereo_temporal_calibration.py')),
              'ray_model_sha256': sha(Path(__file__).with_name('stereo_lens_calibration.py')),
              'tracking_sha256': sha(Path(__file__).with_name('stereo_direct_tracks.py')),
              'extraction_sha256': sha(Path(__file__).with_name('evaluate_stereo_temporal_calibration.py')),
              'tag_evaluation_sha256': sha(Path(__file__).with_name('evaluate_stereo_lens_calibration.py')),
              'observed_provenance': observed, 'pose_provenance': pose_provenance,
              'opencv_threads': cv2.getNumThreads(), 'ordinals': ordinals, 'baseline_m': baseline,
              'maximum_input_frames': maximum_frames, 'training_track_limit': training_track_limit, 'continuous_window':continuous_window,
              'training_tracks': len(np.unique(train[:, 1])), 'training_observations': len(train),
              'heldout_tracks': len(np.unique(held[:, 1])), 'heldout_observations': len(held),
              'fitted_original_frame_indices': fitted_frames,
              'support_diagnostics': support_diagnostics,
              'support_pruning_sha256': sha(Path(__file__).with_name('stereo_track_support.py')) if prune_support else None,
              'training_connectivity': connectivity(train, len(fitted_frames)) if fitted_frames else {'connected': False},
              'tracking_diagnostics': track_diagnostics, 'parallax_diagnostics': parallax_diagnostics,
              'external_tag_provenance': tag_provenance,
              'uncorrected_external_tags': tag_score(tag_observations, left, right, np.zeros(6), baseline, tag_size),
              'models': {}, 'metric_acceptance': 'unevaluated'}
    return result, train, held, relative_rotations, (left, right), tag_observations, tag_size


def run(manifest_path, frozen_path, poses_path, config_path, exposure_offset_ms, external_tag_manifest=None, prune_support=False,
        prune_outlier_support=False):
    result, train, held, rotations, cameras, tags, tag_size = prepare_window(
        manifest_path, frozen_path, poses_path, config_path, exposure_offset_ms, external_tag_manifest, prune_support)
    left, right = cameras
    baseline = result['baseline_m']
    for model in ('physical4', 'physical6'):
        fit = fit_temporal_calibration(train, K, K, baseline, model=model,
                                      camera_rotations=rotations, camera_ray_models=(left, right))
        support_refit = None
        if prune_outlier_support:
            from stereo_support_refit import refit_physical_support
            fit, support_refit = refit_physical_support(train, rotations, cameras, baseline, fit)
        entry = {'fit': fit, 'heldout_natural': heldout_prediction(held, fit, left, right, baseline)}
        if support_refit is not None:
            entry['support_refit'] = support_refit
            result['support_refit_sha256'] = sha(Path(__file__).with_name('stereo_support_refit.py'))
        if fit.get('accepted'):
            entry['external_tags'] = tag_score(tags, left, right, np.array(fit['parameters_vector']), baseline, tag_size)
        result['models'][model] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'frozen', 'poses', 'config', 'output'):
        parser.add_argument(name)
    parser.add_argument('--exposure-offset-ms', type=float, required=True)
    parser.add_argument('--external-tag-manifest')
    parser.add_argument('--prune-support', action='store_true')
    parser.add_argument('--prune-outlier-support', action='store_true')
    args = parser.parse_args()
    result = run(args.manifest, args.frozen, args.poses, args.config, args.exposure_offset_ms, args.external_tag_manifest,
                 args.prune_support, args.prune_outlier_support)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
