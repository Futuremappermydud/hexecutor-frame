"""Offline one-shot recovery from orphaned support after robust outlier removal.

Only the existing support/connectivity rejection is recoverable. Other training
failures remain failures. Calibration residuals select training observations;
held-out pixels and tags are never inputs. All original minima remain enforced.
Explicit use_recorded_rotations=False keeps poses free in the refit; recorded
rotations still define the unchanged left-parallax support selection.
"""
import cv2
import numpy as np
from stereo_temporal_calibration import fit_temporal_calibration, _DEFAULTS
from stereo_track_support import prune_training_support
from stereo_lens_calibration import calibrated_left_camera, baseline_direction, shifted_principal_camera


def physical_inlier_mask(rows,cameras,baseline,initial,maximum_error_px=3.):
    poses = {p['frame_id']: p for p in initial['poses']}
    points = {p['track_id']: np.array(p['position_m']) for p in initial['points']}
    local = np.empty((len(rows), 3))
    for frame in np.unique(rows[:,0]).astype(int):
        mask = rows[:, 0] == frame; pose = poses[int(frame)]
        world = np.array([points[int(t)] for t in rows[mask, 1]])
        local[mask] = (world-np.array(pose['translation_m'])) @ cv2.Rodrigues(np.array(pose['rotation_vector_rad']))[0]
    p = np.array(initial['parameters_vector'])
    right_local = (local-baseline*baseline_direction(p)) @ cv2.Rodrigues(p[:3])[0]
    lp, lv = calibrated_left_camera(cameras[0],p).project_for_optimization(local, p[4])
    right=shifted_principal_camera(cameras[1],p[8:10]) if len(p) in (10,11) else cameras[1]
    rp, rv = right.project_for_optimization(right_local, p[5], np.exp(p[3]))
    errors = np.column_stack((np.linalg.norm(lp-rows[:, 2:4], axis=1), np.linalg.norm(rp-rows[:, 4:6], axis=1)))
    return lv & rv & np.all(errors <= maximum_error_px, axis=1)


def refit_physical_support(rows, rotations, cameras, baseline, initial, config=None, use_recorded_rotations=True, rotation_initialization_only=False):
    if not isinstance(use_recorded_rotations, bool) or not isinstance(rotation_initialization_only,bool):
        raise ValueError('rotation constraint and seed options must be boolean')
    reasons = set(initial.get('diagnostics', {}).get('reasons', []))
    allowed = {'insufficient_support_after_outlier_rejection', 'disconnected_after_outlier_rejection'}
    report = {'attempted': False, 'initial_reasons': sorted(reasons)}
    if (not reasons or not reasons <= allowed or not initial.get('diagnostics', {}).get('converged') or
            not initial.get('poses') or not initial.get('points')):
        report['reason'] = 'initial_fit_does_not_have_recoverable_support_rejection'
        return initial, report
    cfg = dict(_DEFAULTS); cfg.update(config or {})
    model = initial['model'].removeprefix('temporal_')
    if model not in ('physical4', 'physical6', 'physical_baseline6', 'physical_baseline8', 'physical_center8','physical_focal7'):
        raise ValueError('support refit requires a physical model')
    refine_axis=initial.get('camera_rotation_refinement_vector_rad') is not None
    if refine_axis and (not use_recorded_rotations or rotation_initialization_only):
        raise ValueError('shared axis refinement requires constrained recorded rotations')
    rows, rotations = np.asarray(rows, float), np.asarray(rotations, float)
    frame_ids = np.unique(rows[:, 0]).astype(int)
    if not np.array_equal(frame_ids, np.arange(len(rotations))):
        raise ValueError('support refit input frames must index the supplied rotation array')
    keep=physical_inlier_mask(rows,cameras,baseline,initial,cfg['refit_max_view_error_px'])
    clean, retained_frames, support = prune_training_support(rows[keep], rotations, cameras[0].intrinsic,
        minimum_frame_tracks=cfg['min_frame_tracks'], minimum_track_frames=max(3, cfg['min_track_frames']),
        minimum_frames=cfg['min_frames'], minimum_tracks=cfg['min_tracks'], minimum_observations=cfg['min_observations'])
    input_indices = {(int(r[0]), int(r[1])): i for i, r in enumerate(rows)}
    retained_indices = [input_indices[(int(r[0]), int(r[1]))] for r in clean]
    report.update(attempted=True, inlier_cut_px=cfg['refit_max_view_error_px'],
                  initial_fit=initial,
                  removed_stereo_observations=int(np.sum(~keep)), retained_stereo_observations=len(clean),
                  retained_original_frame_indices=retained_frames, retained_input_observation_indices=retained_indices, support=support)
    if not support['support_pruning']['eligible']:
        report['reason'] = 'insufficient_connected_support_after_cleanup'
        return initial, report
    selected = rotations[retained_frames]
    selected = selected[0].T @ selected
    # Keep original frame IDs. The estimator orders them and fixes its new first
    # pose; held-out scoring can then identify omitted training frames explicitly.
    fit = fit_temporal_calibration(clean, cameras[0].intrinsic, cameras[1].intrinsic, baseline,
                                  config=config, model=model, camera_rotations=selected if use_recorded_rotations or rotation_initialization_only else None,
                                  rotation_initialization_only=rotation_initialization_only, refine_camera_rotation=refine_axis, camera_ray_models=cameras,
                                  fixed_rotation_initialization=initial['diagnostics'].get('initialization_strategy') == 'fixed_rotation',
                                  global_rotation_initialization=initial['diagnostics'].get('initialization_strategy') == 'global_rotation')
    report['refit_training_accepted'] = fit['accepted']
    return fit, report


def refit_grouped_physical_support(rows,rotations,groups,cameras,baseline,initial,config=None,global_initialization=False,camera_motion=None,use_recorded_rotations=True,rotation_initialization_only=False):
    """One training-only support cleanup preserving every independent group."""
    if not isinstance(use_recorded_rotations, bool) or not isinstance(rotation_initialization_only,bool):
        raise ValueError('rotation constraint and seed options must be boolean')
    if rotation_initialization_only and (not global_initialization or camera_motion is not None):
        raise ValueError('grouped rotation seeds require global initialization and free motion')
    if not use_recorded_rotations and ((global_initialization and not rotation_initialization_only) or camera_motion is not None):
        raise ValueError('free rotations cannot use fixed-rotation initialization or motion')
    reasons=set(initial.get('diagnostics',{}).get('reasons',[]))
    allowed={'insufficient_support_after_outlier_rejection','disconnected_after_outlier_rejection'}
    report={'attempted':False,'initial_reasons':sorted(reasons)}
    if not reasons or not reasons<=allowed or not initial.get('diagnostics',{}).get('converged') or not initial.get('poses') or not initial.get('points'):
        return initial,report
    refine_axis=initial.get('camera_rotation_refinement_vector_rad') is not None
    if refine_axis and (not use_recorded_rotations or rotation_initialization_only or camera_motion is not None):
        raise ValueError('shared axis refinement requires constrained rotations and free centers')
    cfg=dict(_DEFAULTS);cfg.update(config or {})
    rows=np.asarray(rows,float);rotations=np.asarray(rotations,float);groups=np.asarray(groups)
    frame_ids=np.unique(rows[:,0]);keep=physical_inlier_mask(rows,cameras,baseline,initial,cfg['refit_max_view_error_px'])
    retained=[];selected_rotations=[];selected_groups=[];selected_frame_ids=[];selected_translation=[];selected_lever=[];supports=[]
    for group in np.unique(groups):
        selected_frames=np.flatnonzero(groups==group);original_ids=frame_ids[selected_frames]
        mapping={f:i for i,f in enumerate(original_ids)}
        subset=rows[keep & np.isin(rows[:,0],original_ids)].copy()
        subset[:,0]=[mapping[f] for f in subset[:,0]]
        clean,indices,support=prune_training_support(subset,rotations[selected_frames],cameras[0].intrinsic,
            minimum_frame_tracks=cfg['min_frame_tracks'],minimum_track_frames=max(3,cfg['min_track_frames']),
            minimum_frames=cfg['min_frames'],minimum_tracks=cfg['min_tracks'],minimum_observations=cfg['min_observations'])
        supports.append({'group_id':int(group),'support':support})
        if not support['support_pruning']['eligible']:
            report.update(attempted=True,initial_fit=initial,groups=supports,reason='insufficient_group_support_after_cleanup')
            return initial,report
        clean[:,0]=original_ids[clean[:,0].astype(int)]
        r=rotations[selected_frames[indices]];first_rotation=r[0].copy();r=r[0].T@r
        if camera_motion is not None:
            for name,destination in (('translation',selected_translation),('lever_arm',selected_lever)):
                values=np.asarray(camera_motion[name])[selected_frames[indices]]
                destination.extend((values-values[0])@first_rotation)
        retained.append(clean);selected_rotations.extend(r);selected_groups.extend([group]*len(r));selected_frame_ids.extend(original_ids[indices])
    order=np.argsort(selected_frame_ids)
    fit=fit_temporal_calibration(np.concatenate(retained),cameras[0].intrinsic,cameras[1].intrinsic,baseline,
        config=config,model=initial['model'].removeprefix('temporal_'),camera_rotations=np.array(selected_rotations)[order] if use_recorded_rotations or rotation_initialization_only else None,
        rotation_initialization_only=rotation_initialization_only,refine_camera_rotation=refine_axis,
        camera_ray_models=cameras,frame_groups=np.array(selected_groups)[order],global_rotation_initialization=global_initialization,
        camera_motion={'translation':np.array(selected_translation)[order],'lever_arm':np.array(selected_lever)[order]} if camera_motion is not None else None)
    report.update(attempted=True,initial_fit=initial,groups=supports,removed_stereo_observations=int(np.sum(~keep)),
        retained_stereo_observations=sum(len(r) for r in retained),refit_training_accepted=fit['accepted'])
    return fit,report
