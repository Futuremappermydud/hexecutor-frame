#!/usr/bin/env python3
"""Offline joint natural-track stereo bundle adjustment, experimental.

Requires NumPy, SciPy and OpenCV. Camera 0 defines the world gauge; fixed supplied
baseline is the sole metric scale input. Right-to-left orientation and optional relative
right focal scale are shared across frames. Optional recorded motion constraints
are explicit inputs; their translation scale is fitted. No tags enter fitting.
Results are training diagnostics only; accepted does not establish metric truth.
"""
import argparse
import json
import sys

import numpy as np

from stereo_self_calibration import _inputs, _rays, _weighted_quantile

_DEFAULTS = {'min_frames': 3, 'min_tracks': 30, 'min_observations': 90,
             'min_track_frames': 2, 'min_frame_tracks': 12,
             'max_frames': 12, 'max_tracks': 200, 'max_observations': 1800,
             'min_translation_m': .01, 'min_positive_depth_fraction': .95,
             'inlier_threshold_px': 1., 'min_inlier_fraction': .6,
             'min_calibration_singular_ratio': 1e-5,
             'max_rotation_rad': .15, 'max_focal_log_scale': .15,
             'max_physical_principal_shift_px':2.,
             'balance_track_weights':False, 'exact_small_problem':False, 'block_precondition':False, 'max_nfev': 100, 'robust_scale_px': 1., 'refit_max_view_error_px': 3.}


def _rejected(reason, diagnostics=None):
    info = {'reasons': [reason]}
    if diagnostics:
        info.update(diagnostics)
    return {'model': 'temporal_relative4', 'parameters': {'rotation_vector_rad': [0., 0., 0.],
            'focal_scale': 1.}, 'accepted': False, 'metric_acceptance': 'unevaluated',
            'diagnostics': info, 'poses': [], 'points': []}


def fit_temporal_calibration(observations, K_left, K_right, baseline_m, config=None, model="relative4",
                             camera_rotations=None, camera_motion=None, camera_ray_models=None, frame_groups=None, refine_camera_rotation=False, fixed_rotation_initialization=False, global_rotation_initialization=False, rotation_initialization_only=False):
    """Jointly fit shared stereo geometry, camera poses and natural 3-D tracks.

    Observations are [integer frame ID, integer track ID, left u,v, right u,v]
    in undistorted pinhole pixels. Pose rotation vectors map camera to world;
    first frame is fixed to identity. The right center defaults to [baseline_m,0,0] in
    each left camera frame. Explicit physical_baseline6/8 models instead fit
    direction normalize([1,slope_y,slope_z]) with the supplied norm unchanged.
    Calibration R maps right rays to left coordinates.
    Model rotation3 fixes right focal scale to one and removes that parameter
    from the solver and profiled Jacobian. relative4 estimates it. Fixed focal
    is a model assumption, not evidence that the supplied intrinsics are exact.
    Explicit config balance_track_weights=True equalizes total robust objective
    weight per surviving track on each solve, preserving pixel robust scale.
    It does not model within-track covariance or establish independence.
    Optional camera_rotations fix independently supplied camera-to-world
    rotations in sorted frame-ID order, with first identity. Camera centers and
    points remain free: no external translation scale is assumed. The caller
    owns exposure association and camera-to-head rotation provenance.
    Optional camera_motion contains per-group first-camera-gauge translation and lever_arm
    arrays; centers become exp(log_scale)*translation + lever_arm. Only one
    positive head-translation scale is fitted (bounds 0.1..10), with supplied
    metric camera/head lever offsets. Bound hits reject; none of these priors
    establish tracking accuracy or historical calibration identity.
    Model physical_focal7 additionally estimates physical left focal scale; its
    eleven-value export appends left log focal after zero baseline/principal slots.
    Models physical4/physical6 require two CameraRayModel instances. They use
    original virtual-grid residuals with physical right focal scale and optional
    physical k1 changes, retaining the same pose/point nuisance parameters.
    Their parameter schema differs from virtual pinhole models deliberately;
    use the physical camera models for projection and held-out triangulation.
    Out-of-domain trial projections are finite extensions; active final rays
    outside the validated domain reject the whole fit.
    Optional rotation_initialization_only uses supplied rotations for the chosen
    rotation initializer, then releases every non-gauge per-frame rotation.
    It adds no rotation prior or residual.
    Optional frame_groups labels independent windows sharing calibration.
    Each group's first pose fixes its own world gauge; tracks cannot cross
    groups. Support and motion minima apply to every group. Resource bounds
    still apply to the whole optimization and must be explicitly provisioned.
    Optional refine_camera_rotation fits one shared bounded rotation D such
    that fixed camera rotations become D.T @ prior_rotation @ D. This models
    camera/head axis error without introducing tracking translation scale.
    The correction is nuisance geometry, profiled out of calibration rank;
    bound hits reject and it is never silently applied to capture metadata.
    Input validation errors raise ValueError; inadequate geometry is rejected.
    """
    import cv2
    from scipy.sparse import lil_matrix

    if model not in ('relative4', 'rotation3', 'physical4', 'physical6', 'physical_baseline6', 'physical_baseline8', 'physical_center8', 'physical_focal7'):
        raise ValueError('unknown temporal calibration model')
    physical = model.startswith('physical')
    estimate_lens = model in ('physical6', 'physical_baseline8', 'physical_center8', 'physical_focal7')
    estimate_left_focal = model == 'physical_focal7'
    estimate_center = model == 'physical_center8'
    estimate_baseline = model in ('physical_baseline6', 'physical_baseline8')
    if physical != (camera_ray_models is not None):
        raise ValueError('physical models require two explicit physical camera ray models')
    if physical:
        from stereo_lens_calibration import CameraRayModel
        if (not isinstance(camera_ray_models, (tuple, list)) or len(camera_ray_models) != 2 or
                any(not isinstance(c, CameraRayModel) for c in camera_ray_models)):
            raise ValueError('two CameraRayModel instances required')
    if not isinstance(rotation_initialization_only,bool) or (rotation_initialization_only and
            (camera_rotations is None or camera_motion is not None or refine_camera_rotation or
             not (global_rotation_initialization or fixed_rotation_initialization))):
        raise ValueError('rotation initialization only requires supplied rotation seeds, an explicit rotation initializer, and free motion without shared-axis refinement')
    if not isinstance(global_rotation_initialization,bool) or (global_rotation_initialization and camera_rotations is None):
        raise ValueError('global rotation initialization requires supplied rotations')
    if global_rotation_initialization and fixed_rotation_initialization:
        raise ValueError('choose one rotation initialization strategy')
    if not isinstance(fixed_rotation_initialization, bool) or (fixed_rotation_initialization and camera_rotations is None):
        raise ValueError('fixed-rotation initialization requires supplied rotations')
    if not isinstance(refine_camera_rotation, bool) or (refine_camera_rotation and camera_rotations is None):
        raise ValueError('rotation refinement requires independently supplied camera rotations')
    if refine_camera_rotation and camera_motion is not None:
        raise ValueError('rotation refinement currently requires free camera centers')
    ncal = 7 if estimate_left_focal else 8 if model in ('physical_baseline8', 'physical_center8') else 6 if model in ('physical6', 'physical_baseline6') else 3 if model == 'rotation3' else 4
    baseline_index = 6 if model == 'physical_baseline8' else 4
    def lens_changes(p):
        return (p[4], p[5]) if estimate_lens else (0., 0.)
    def direction(p):
        d = np.r_[1., p[baseline_index:baseline_index+2]] if estimate_baseline else np.array([1., 0., 0.])
        return d/np.linalg.norm(d)
    def right_camera(p):
        if estimate_center:
            from stereo_lens_calibration import shifted_principal_camera
            return shifted_principal_camera(camera_ray_models[1],p[6:8])
        return camera_ray_models[1]
    def exported(p):
        vector = np.r_[p[:4], lens_changes(p)]
        if estimate_left_focal:return np.r_[vector,0.,0.,0.,0.,p[6]]
        if estimate_center:return np.r_[vector,0.,0.,p[6:8]]
        return np.r_[vector, p[baseline_index:baseline_index+2]] if estimate_baseline else vector
    def parameters(p):
        if physical:
            dl, dr = lens_changes(p)
            result = {'rotation_vector_rad': p[:3].tolist(), 'physical_right_focal_scale': float(np.exp(p[3])),
                      'left_delta_k1': float(dl), 'right_delta_k1': float(dr)}
            if estimate_left_focal:result['physical_left_focal_scale']=float(np.exp(p[6]))
            if estimate_center:result['physical_right_principal_shift_px']=p[6:8].tolist()
            if estimate_baseline:
                result['baseline_direction_left'] = direction(p).tolist()
                result['baseline_direction_slopes'] = p[baseline_index:baseline_index+2].tolist()
            return result
        return {'rotation_vector_rad': p[:3].tolist(), 'focal_scale': float(np.exp(p[3])) if ncal >= 4 else 1.}

    def rejected(reason, diagnostics=None):
        result = _rejected(reason, diagnostics)
        result['model'] = 'temporal_' + model
        result['diagnostics'].update(calibration_parameter_count=ncal, focal_estimated=(ncal >= 4))
        if physical:
            result['parameters'] = parameters(np.zeros(ncal))
            result['parameters_vector'] = exported(np.zeros(ncal)).tolist()
        return result

    obs = np.asarray(observations, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 6 or not np.isfinite(obs).all():
        raise ValueError('observations must be a finite N-by-6 array')
    _, kl, kr = _inputs(np.empty((0, 4)), K_left, K_right)
    if physical and (not np.allclose(camera_ray_models[0].intrinsic, kl) or
                     not np.allclose(camera_ray_models[1].intrinsic, kr)):
        raise ValueError('camera ray model intrinsics differ from observation grids')
    try:
        baseline = float(baseline_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('baseline must be positive and finite') from exc
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('baseline must be positive and finite')
    cfg = dict(_DEFAULTS)
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(cfg):
            raise ValueError('unknown configuration option')
        cfg.update(config)
    if any(not isinstance(cfg[k],bool) for k in ('balance_track_weights','exact_small_problem','block_precondition')):
        raise ValueError('track weighting and exact solver flags must be boolean')
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for k,v in cfg.items() if k not in ('balance_track_weights','exact_small_problem','block_precondition')):
        raise ValueError('configuration values must be positive and finite')
    count_keys = ('min_frames', 'min_tracks', 'min_observations', 'min_track_frames',
                  'min_frame_tracks', 'max_frames', 'max_tracks', 'max_observations', 'max_nfev')
    if any(isinstance(cfg[k], bool) or not isinstance(cfg[k], (int, np.integer)) for k in count_keys):
        raise ValueError('count configuration values must be integers')
    if (cfg['min_frames'] > cfg['max_frames'] or cfg['min_tracks'] > cfg['max_tracks'] or
            cfg['min_observations'] > cfg['max_observations'] or
            cfg['min_track_frames'] > cfg['max_frames'] or cfg['min_frame_tracks'] > cfg['max_tracks']):
        raise ValueError('inconsistent support and resource limits')
    if cfg['min_inlier_fraction'] > 1 or cfg['min_positive_depth_fraction'] > 1:
        raise ValueError('fractions cannot exceed one')
    if np.any(obs[:, :2] != np.floor(obs[:, :2])) or np.any(np.abs(obs[:, :2]) > 2**53):
        raise ValueError('frame and track IDs must be integers')
    ids = obs[:, :2].astype(np.int64)
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError('duplicate frame/track observations')
    frames, fi = np.unique(ids[:, 0], return_inverse=True)
    tracks, ti = np.unique(ids[:, 1], return_inverse=True)
    nf, nt = len(frames), len(tracks)
    if frame_groups is None:
        groups = np.zeros(nf, dtype=np.int64)
    else:
        labels = np.asarray(frame_groups, dtype=float)
        if (labels.shape != (nf,) or not np.isfinite(labels).all() or
                np.any(labels != np.floor(labels)) or np.any(np.abs(labels) > 2**53)):
            raise ValueError('one integer group ID per sorted frame required')
        groups = labels.astype(np.int64)
    group_ids, first_frames = np.unique(groups, return_index=True)
    gauges = np.sort(first_frames)
    free_frames = np.flatnonzero(~np.isin(np.arange(nf), gauges))
    pose_slots = np.full(nf, -1, dtype=int)
    pose_slots[free_frames] = np.arange(len(free_frames))
    if any(len(np.unique(groups[fi[ti == t]])) != 1 for t in range(nt)):
        raise ValueError('tracks cannot cross independent frame groups')
    group_counts = [{'group_id': int(g), 'frames': int(np.sum(groups == g)),
                     'tracks': int(len(np.unique(ti[groups[fi] == g]))),
                     'observations': int(np.sum(groups[fi] == g))} for g in group_ids]
    counts = {'frame_count': nf, 'track_count': nt, 'observation_count': len(obs)}
    if nf > cfg['max_frames'] or nt > cfg['max_tracks'] or len(obs) > cfg['max_observations']:
        return rejected('resource_limit_exceeded', counts)
    if nf < cfg['min_frames'] or nt < cfg['min_tracks'] or len(obs) < cfg['min_observations']:
        return rejected('insufficient_support', counts)
    if any(g['frames'] < cfg['min_frames'] or g['tracks'] < cfg['min_tracks'] or
           g['observations'] < cfg['min_observations'] for g in group_counts):
        return rejected('insufficient_group_support', dict(counts, groups=group_counts))
    if np.any(np.bincount(ti, minlength=nt) < cfg['min_track_frames']) or np.any(np.bincount(fi, minlength=nf) < cfg['min_frame_tracks']):
        return rejected('insufficient_track_support', counts)
    fixed_rotations = None
    if camera_rotations is not None:
        fixed_rotations = np.asarray(camera_rotations, dtype=float)
        if (fixed_rotations.shape != (nf, 3, 3) or not np.isfinite(fixed_rotations).all() or
                not np.allclose(np.swapaxes(fixed_rotations, 1, 2) @ fixed_rotations, np.eye(3), atol=1e-6) or
                not np.allclose(np.linalg.det(fixed_rotations), 1., atol=1e-6) or
                not np.allclose(fixed_rotations[gauges], np.eye(3), atol=1e-6)):
            raise ValueError('camera rotations must be proper rotations, first frame of each group identity')
    head_motion, lever_motion = None, None
    if camera_motion is not None:
        if fixed_rotations is None or not isinstance(camera_motion, dict) or set(camera_motion) != {'translation', 'lever_arm'}:
            raise ValueError('camera motion requires fixed rotations, translation and lever_arm')
        head_motion = np.asarray(camera_motion['translation'], dtype=float)
        lever_motion = np.asarray(camera_motion['lever_arm'], dtype=float)
        if any(a.shape != (nf, 3) or not np.isfinite(a).all() or not np.allclose(a[gauges], 0., atol=1e-8)
               for a in (head_motion, lever_motion)):
            raise ValueError('motion arrays must be finite N-by-3 in each group first-camera gauge with all gauges zero')
    # Bipartite graph reachability from first frame.
    seen_f = set(gauges.tolist())
    seen_t = set()
    while True:
        next_t = seen_t | set(ti[np.isin(fi, list(seen_f))].tolist())
        next_f = seen_f | set(fi[np.isin(ti, list(next_t))].tolist())
        if next_f == seen_f and next_t == seen_t:
            break
        seen_f, seen_t = next_f, next_t
    if len(seen_f) != nf or len(seen_t) != nt:
        return rejected('disconnected_track_graph', counts)

    # Nominal rectified triangulation seeds; PnP then registers connected views.
    lr = _rays(obs[:, 2:4], kl)
    rr = _rays(obs[:, 4:6], kr)
    disparity = lr[:, 0] - rr[:, 0]
    good = (disparity > baseline / 20.) & (disparity < baseline / .2)
    local = np.zeros((len(obs), 3))
    local[good] = lr[good] * (baseline / disparity[good])[:, None]
    world = np.full((nt, 3), np.nan)
    poses = np.zeros((nf, 6))  # camera-to-world rotation vector, world center
    registered = set(gauges.tolist())
    sel = np.isin(fi, gauges) & good
    world[ti[sel]] = local[sel]
    # OpenCV PnP ignores K[0,1]. Undo left intrinsic skew before passing
    # observations to its zero-skew model; BA retains original pixel K.
    pnp_pixels = obs[:, 2:4].copy()
    pnp_pixels[:, 0] -= kl[0, 1] * (pnp_pixels[:, 1] - kl[1, 2]) / kl[1, 1]
    pnp_k = kl.copy()
    pnp_k[0, 1] = 0.
    cv2.setRNGSeed(0)
    global_initialization_diagnostics=None
    if global_rotation_initialization:
        from stereo_global_initialization import initialize_rotation_graph
        initial=initialize_rotation_graph(obs[:,2:4],obs[:,4:6],fi,ti,fixed_rotations,nt,gauges,kl,kr,baseline)
        if initial is None:return rejected('global_pose_initialization_failed',counts)
        world,centers,global_initialization_diagnostics=initial
        poses[:,3:]=centers
        poses[:,:3]=np.array([cv2.Rodrigues(r)[0].ravel() for r in fixed_rotations])
        registered=set(range(nf))
    while len(registered) < nf:
        progress = False
        for frame in range(nf):
            if frame in registered:
                continue
            idx = np.flatnonzero((fi == frame) & np.isfinite(world[ti]).all(axis=1))
            if len(idx) < max(6, int(cfg['min_frame_tracks'])):
                continue
            if fixed_rotation_initialization:
                from stereo_rotation_initialization import initialize_center
                center = initialize_center(world[ti[idx]], obs[idx, 2:4], kl, fixed_rotations[frame],
                                           int(cfg['min_frame_tracks']))
                if center is None:
                    continue
                r_wc = fixed_rotations[frame]
                poses[frame, :3] = cv2.Rodrigues(r_wc)[0].ravel()
                poses[frame, 3:] = center
                new = np.flatnonzero((fi == frame) & good & ~np.isfinite(world[ti]).all(axis=1))
                world[ti[new]] = local[new] @ r_wc.T + center
                registered.add(frame); progress = True
                continue
            try:
                ok, rv, tv, inliers = cv2.solvePnPRansac(world[ti[idx]], pnp_pixels[idx], pnp_k, None,
                    iterationsCount=200, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
                if not ok or inliers is None or len(inliers) < int(cfg['min_frame_tracks']):
                    continue
                chosen = idx[inliers.ravel()]
                ok, rv, tv = cv2.solvePnP(world[ti[chosen]], pnp_pixels[chosen], pnp_k, None, rv, tv,
                                           True, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                r_wc = cv2.Rodrigues(rv)[0].T
                poses[frame, :3] = cv2.Rodrigues(r_wc)[0].ravel()
                poses[frame, 3:] = -(r_wc @ tv.ravel())
                new = np.flatnonzero((fi == frame) & good & ~np.isfinite(world[ti]).all(axis=1))
                world[ti[new]] = local[new] @ r_wc.T + poses[frame, 3:]
                registered.add(frame)
                progress = True
            except cv2.error:
                continue
        if not progress:
            return rejected('pose_initialization_failed', dict(counts, initialized_frames=len(registered)))
    if not np.isfinite(world).all():
        return rejected('point_initialization_failed', dict(counts, initialized_tracks=int(np.isfinite(world).all(axis=1).sum())))

    # Rotation seeds initialize poses but impose no rotation residual or constraint.
    if rotation_initialization_only:
        fixed_rotations = None

    # Shared calibration, optional head/camera rotation correction, poses, points.
    nhand = 3 if refine_camera_rotation else 0
    nshared = ncal+nhand
    pose_stride = 6 if fixed_rotations is None else 3
    npose = 1 if head_motion is not None else pose_stride * len(free_frames)
    if fixed_rotations is not None:
        poses[:, :3] = np.array([cv2.Rodrigues(r)[0].ravel() for r in fixed_rotations])
    pose_start = poses[free_frames] if fixed_rotations is None else poses[free_frames, 3:]
    p0 = np.r_[np.zeros(nshared), np.zeros(1) if head_motion is not None else pose_start.ravel(), world.ravel()]
    def unpack_poses(p):
        cp = np.zeros((nf, 6))
        if fixed_rotations is None:
            cp[free_frames] = p[nshared:nshared+npose].reshape(-1, 6)
        else:
            if nhand:
                correction = cv2.Rodrigues(p[ncal:nshared])[0]
                corrected = correction.T @ fixed_rotations @ correction
                cp[:, :3] = np.array([cv2.Rodrigues(r)[0].ravel() for r in corrected])
            else:
                cp[:, :3] = poses[:, :3]
            if head_motion is None:
                cp[free_frames, 3:] = p[nshared:nshared+npose].reshape(-1, 3)
            else:
                cp[:, 3:] = np.exp(p[nshared]) * head_motion + lever_motion
        return cp
    npar = len(p0)
    def residual(p, return_domain=False):
        cp = unpack_poses(p)
        pt = p[nshared+npose:].reshape(-1, 3)
        relative = cv2.Rodrigues(p[:3])[0]
        scale = np.exp(p[3]) if ncal >= 4 else 1.
        left = np.empty((len(obs), 3))
        for frame in range(nf):
            mask = fi == frame
            rot = cv2.Rodrigues(cp[frame, :3])[0]
            left[mask] = (pt[ti[mask]] - cp[frame, 3:]) @ rot
        right = (left - baseline*direction(p)) @ relative
        # Finite extension near z=0, then cheirality gate rejects bad solutions.
        zl = np.where(np.abs(left[:, 2]) > 1e-8, left[:, 2], 1e-8)
        zr = np.where(np.abs(right[:, 2]) > 1e-8, right[:, 2], 1e-8)
        domain = np.ones(len(obs), dtype=bool)
        if physical:
            dl, dr = lens_changes(p)
            lp, lv = camera_ray_models[0].project_for_optimization(left, dl, np.exp(p[6]) if estimate_left_focal else 1.)
            rp, rv = right_camera(p).project_for_optimization(right, dr, scale)
            domain = lv & rv
        else:
            lp = (left @ kl.T)[:, :2] / zl[:, None]
            kr_scaled = kr.copy()
            kr_scaled[:2, :2] *= scale
            rp = (right @ kr_scaled.T)[:, :2] / zr[:, None]
        value = np.column_stack((lp - obs[:, 2:4], rp - obs[:, 4:6])).ravel()
        return (value, domain) if return_domain else value
    sparsity = lil_matrix((4 * len(obs), npar), dtype=np.int8)
    for i, (f, t) in enumerate(zip(fi, ti)):
        sparsity[4*i+2:4*i+4, :min(ncal, 4)] = 1
        if estimate_lens:
            sparsity[4*i:4*i+2, 4] = 1
            sparsity[4*i+2:4*i+4, 5] = 1
        if estimate_left_focal:
            sparsity[4*i:4*i+2,6] = 1
        if estimate_center:
            sparsity[4*i+2:4*i+4,6:8] = 1
        if estimate_baseline:
            sparsity[4*i+2:4*i+4, baseline_index:baseline_index+2] = 1
        if nhand:
            sparsity[4*i:4*i+4, ncal:nshared] = 1
        if pose_slots[f] >= 0:
            if head_motion is None:
                sparsity[4*i:4*i+4, nshared+pose_stride*pose_slots[f]:nshared+pose_stride*(pose_slots[f]+1)] = 1
            else:
                sparsity[4*i:4*i+4, nshared:nshared+1] = 1
        sparsity[4*i:4*i+4, nshared+npose+3*t:nshared+npose+3*t+3] = 1
    boundlo = np.full(npar, -np.inf)
    boundhi = np.full(npar, np.inf)
    boundlo[:3], boundhi[:3] = -cfg['max_rotation_rad'], cfg['max_rotation_rad']
    focal_bound = min(cfg['max_focal_log_scale'], .05) if physical else cfg['max_focal_log_scale']
    if ncal >= 4:
        boundlo[3], boundhi[3] = -focal_bound, focal_bound
    if estimate_lens:
        boundlo[4:6], boundhi[4:6] = -.02, .02
    if estimate_left_focal:
        boundlo[6],boundhi[6] = -focal_bound,focal_bound
    if estimate_center:
        boundlo[6:8],boundhi[6:8] = -cfg['max_physical_principal_shift_px'],cfg['max_physical_principal_shift_px']
    if estimate_baseline:
        boundlo[baseline_index:baseline_index+2], boundhi[baseline_index:baseline_index+2] = -.05, .05
    if nhand:
        boundlo[ncal:nshared], boundhi[ncal:nshared] = -.1, .1
    if head_motion is not None:
        boundlo[nshared], boundhi[nshared] = np.log(.1), np.log(10.)
    nuisance_blocks=[]
    if head_motion is None:
        nuisance_blocks.extend((nshared+i*pose_stride,pose_stride) for i in range(len(free_frames)))
    nuisance_blocks.extend((nshared+npose+3*i,3) for i in range(nt))
    solver_attempts=[]
    def solve(start, keep):
        rows = np.repeat(keep, 4)
        loss = 'soft_l1'
        if cfg['balance_track_weights']:
            from stereo_track_weighting import equal_track_soft_l1
            loss = equal_track_soft_l1(ti[keep])
        from stereo_bundle_solver import solve_bundle
        result,diagnostic=solve_bundle(lambda x:residual(x)[rows],start,sparsity.tocsr()[rows],
            (boundlo,boundhi),loss,cfg['robust_scale_px'],int(cfg['max_nfev']),cfg['exact_small_problem'],nuisance_blocks if cfg['block_precondition'] else None)
        solver_attempts.append(diagnostic)
        return result

    keep = np.ones(len(obs), dtype=bool)
    refit_failure = None
    initial_nfev = 0
    refit_performed = False
    try:
        opt = solve(p0, keep)
        initial_nfev = int(opt.nfev)
        if opt.success:
            initial_error = np.linalg.norm(residual(opt.x).reshape(-1, 2, 2), axis=2)
            keep = np.all(initial_error <= cfg['refit_max_view_error_px'], axis=1)
            if not keep.all():
                # No points or poses may be orphaned by outlier rejection.
                if (int(keep.sum()) < cfg['min_observations'] or
                        any(np.sum(keep & (groups[fi] == g)) < cfg['min_observations'] for g in group_ids) or
                        np.any(np.bincount(ti[keep], minlength=nt) < cfg['min_track_frames']) or
                        np.any(np.bincount(fi[keep], minlength=nf) < cfg['min_frame_tracks'])):
                    refit_failure = 'insufficient_support_after_outlier_rejection'
                else:
                    sf, st = set(gauges.tolist()), set()
                    while True:
                        nt_seen = st | set(ti[keep & np.isin(fi, list(sf))].tolist())
                        nf_seen = sf | set(fi[keep & np.isin(ti, list(nt_seen))].tolist())
                        if nt_seen == st and nf_seen == sf:
                            break
                        sf, st = nf_seen, nt_seen
                    if len(sf) != nf or len(st) != nt:
                        refit_failure = 'disconnected_after_outlier_rejection'
                    else:
                        opt = solve(opt.x, keep)
                        refit_performed = True
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return rejected('solver_failure', dict(counts, detail=str(exc)))
    p = opt.x
    cp = unpack_poses(p)
    points = p[nshared+npose:].reshape(-1, 3)
    final_residual, projection_domain = residual(p, return_domain=True)
    residuals = final_residual.reshape(-1, 4)
    errors = np.linalg.norm(residuals.reshape(-1, 2, 2), axis=2)
    per_view_inlier = float(np.mean(errors <= cfg['inlier_threshold_px']))
    inlier = float(np.mean(np.all(errors <= cfg['inlier_threshold_px'], axis=1)))
    positive = 0
    rel = cv2.Rodrigues(p[:3])[0]
    for frame in range(nf):
        mask = fi == frame
        left = (points[ti[mask]] - cp[frame, 3:]) @ cv2.Rodrigues(cp[frame, :3])[0]
        right = (left - baseline*direction(p)) @ rel
        positive += int(np.sum((left[:, 2] > 0) & (right[:, 2] > 0)))
    cheirality = positive / len(obs)
    group_motion = {int(g): float(np.max(np.linalg.norm(cp[groups == g, 3:], axis=1))) for g in group_ids}
    motion = max(group_motion.values())
    # Profile out nuisance directions in the robust-weighted Jacobian. This is
    # a numerical rank diagnostic, not a covariance or accuracy estimate.
    from stereo_profile_projection import profile_calibration_columns
    profiled,profile_diagnostics=profile_calibration_columns(opt.jac,ncal)
    profile_stops=profile_diagnostics['stops']
    profile_iterations=profile_diagnostics['iterations']
    profile_residual_norms=profile_diagnostics['residual_norms']
    profile_condition_estimates=profile_diagnostics['scaled_condition_estimates']
    _, singular, right_vectors = np.linalg.svd(profiled, full_matrices=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.
    # Descriptive, local linearized noise amplification, never an acceptance
    # criterion. Independent scalar Gaussian residuals and this fitted model are
    # assumptions, not established facts. Robust weights and track correlation
    # mean this is not a valid covariance or confidence interval.
    active = keep if refit_performed else np.ones(len(obs), dtype=bool)
    scalar_errors = residuals[active].ravel()
    center = np.median(scalar_errors)
    noise_mad = float(1.4826 * np.median(np.abs(scalar_errors - center)))
    probes = np.array([[x*z, y*z, z] for z in (.7, 1.5, 2.5)
                       for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
    def probe_x(global_p):
        ray = (probes - baseline*direction(global_p)) @ cv2.Rodrigues(global_p[:3])[0]
        if physical:
            dk = lens_changes(global_p)[1]
            return right_camera(global_p).project(ray, dk, np.exp(global_p[3]))[:, 0]
        k = kr.copy()
        k[:2, :2] *= (np.exp(global_p[3]) if ncal >= 4 else 1.)
        q = ray @ k.T
        if np.any(ray[:, 2] <= 0) or not np.isfinite(q).all():
            raise ValueError('probe behind camera')
        return q[:, 0] / ray[:, 2]
    nominal_probe_std = None
    if (opt.success and not refit_failure and all(stop in (0, 1, 2, 4, 5) for stop in profile_stops)
            and ratio >= cfg['min_calibration_singular_ratio']
            and singular[-1] > np.finfo(float).eps * singular[0]):
        try:
            derivative = []
            for j in range(ncal):
                d = np.zeros(ncal)
                d[j] = 1e-6
                derivative.append((probe_x(p[:ncal] + d) - probe_x(p[:ncal] - d)) / 2e-6)
            g = np.column_stack(derivative)
            amplification = (g @ right_vectors.T) / singular[None, :]
            nominal_probe_std = float(noise_mad * np.max(np.linalg.norm(amplification, axis=1)))
            if not np.isfinite(nominal_probe_std):
                nominal_probe_std = None
        except (ValueError, FloatingPointError):
            pass
    # One earliest retained left observation per track; this intentionally does
    # not count repeated frames as independent spatial locations.
    active_rows = np.flatnonzero(active)
    ordered = active_rows[np.lexsort((fi[active_rows], ti[active_rows]))]
    _, first = np.unique(ti[ordered], return_index=True)
    support_pixels = obs[ordered[first], 2:4]
    support_xy = _rays(support_pixels, kl)[:, :2]
    support_bins = np.floor(support_pixels / 2.).astype(np.int64)
    _, support_inverse, support_counts = np.unique(support_bins, axis=0, return_inverse=True, return_counts=True)
    coverage_weights = 1. / support_counts[support_inverse]
    coverage_span = [_weighted_quantile(support_xy[:, axis], coverage_weights, .95) -
                     _weighted_quantile(support_xy[:, axis], coverage_weights, .05) for axis in range(2)]
    angular_cells = int(len(np.unique(np.floor(support_xy / .1).astype(np.int64), axis=0)))
    reasons = []
    if physical and not np.all(projection_domain[active]):
        reasons.append('projection_outside_validated_domain')
    if refit_failure:
        reasons.append(refit_failure)
    if not opt.success:
        reasons.append('not_converged')
    if head_motion is not None and (p[nshared] <= np.log(.1)+1e-5 or p[nshared] >= np.log(10.)-1e-5):
        reasons.append('motion_scale_bound_reached')
    if any(m < cfg['min_translation_m'] for m in group_motion.values()):
        reasons.append('insufficient_translation')
    if cheirality < cfg['min_positive_depth_fraction']:
        reasons.append('insufficient_positive_depth')
    if inlier < cfg['min_inlier_fraction']:
        reasons.append('insufficient_inlier_fraction')
    if any(stop not in (0, 1, 2, 4, 5) for stop in profile_stops):
        reasons.append('profile_solver_not_converged')
    if ratio < cfg['min_calibration_singular_ratio']:
        reasons.append('rank_or_conditioning_failure')
    if np.linalg.norm(p[:3]) >= cfg['max_rotation_rad'] or (ncal >= 4 and abs(p[3]) >= focal_bound * .999):
        reasons.append('parameter_bound_reached')
    if nhand and np.linalg.norm(p[ncal:nshared]) >= .1*.999:
        reasons.append('camera_rotation_refinement_bound_reached')
    if estimate_left_focal and abs(p[6]) >= focal_bound*.999:
        reasons.append('left_focal_bound_reached')
    if estimate_center and np.any(np.abs(p[6:8]) >= cfg['max_physical_principal_shift_px']*.999):
        reasons.append('principal_point_bound_reached')
    if estimate_baseline and np.any(np.abs(p[baseline_index:baseline_index+2]) >= .05*.999):
        reasons.append('baseline_direction_bound_reached')
    if estimate_lens and np.any(np.abs(p[4:6]) >= .02*.999):
        reasons.append('lens_parameter_bound_reached')
    result = {'model':'temporal_' + model, 'parameters':parameters(p),
            'accepted':not reasons, 'metric_acceptance':'unevaluated',
            'diagnostics':dict(counts, reasons=reasons, least_squares_solver_attempts=solver_attempts, balance_track_weights=cfg['balance_track_weights'], initialization_strategy='global_rotation' if global_rotation_initialization else 'fixed_rotation' if fixed_rotation_initialization else 'pnp', global_initialization=global_initialization_diagnostics, frame_group_count=len(group_ids), group_support=group_counts, group_max_translation_m=group_motion, projection_domain_fraction=float(np.mean(projection_domain[active])), camera_rotations_fixed=(fixed_rotations is not None), rotation_initialization_only=rotation_initialization_only,
                motion_trajectory_fixed_up_to_scale=(head_motion is not None),
                motion_translation_scale=float(np.exp(p[nshared])) if head_motion is not None else None,
                motion_translation_scale_bounds=[.1, 10.] if head_motion is not None else None, calibration_parameter_count=ncal, focal_estimated=(ncal >= 4), converged=bool(opt.success), solver_status=int(opt.status),
                solver_message=str(opt.message), evaluations=int(opt.nfev), initial_evaluations=initial_nfev,
                retained_observations=int(keep.sum()), rejected_observations=int((~keep).sum()),
                refit_performed=refit_performed,
                optimized_observations=int(keep.sum()) if refit_performed else len(obs),
                refit_max_view_error_px=cfg['refit_max_view_error_px'], robust_cost=float(opt.cost),
                inlier_fraction=inlier, inlier_fraction_scope='whole stereo observations, both view residual norms',
                per_view_inlier_fraction=per_view_inlier, inlier_threshold_px=cfg['inlier_threshold_px'], positive_depth_fraction=cheirality,
                max_translation_m=motion, min_translation_m=cfg['min_translation_m'],
                calibration_profile_singular_values=singular.tolist(), calibration_profile_singular_ratio=ratio,
                calibration_profile_weak_direction_rms_px_per_unit=float(singular[-1] / np.sqrt(opt.jac.shape[0])),
                scalar_residual_mad_sigma_px=noise_mad,
                nominal_independent_noise_max_probe_std_px=nominal_probe_std,
                nominal_noise_diagnostic_assumptions='local linearization; independent scalar residuals; robust-weighted Jacobian; not covariance, confidence, or a metric acceptance test',
                spatial_support={'one_earliest_left_ray_per_track':int(len(support_xy)),
                                 'unique_2px_left_bins':int(len(support_counts)),
                                 'normalized_xy_weighted_5_95_span':coverage_span,
                                 'angular_cell_width_normalized':.1, 'occupied_angular_cells':angular_cells},
                calibration_profile_projection=profile_diagnostics,
                calibration_profile_lsmr_stops=profile_stops,
                calibration_profile_lsmr_iterations=profile_iterations,
                calibration_profile_lsmr_residual_norms=profile_residual_norms,
                calibration_profile_lsmr_condition_estimates=profile_condition_estimates,
                min_calibration_singular_ratio=cfg['min_calibration_singular_ratio'],
                gate_scope='training only; no metric acceptance; profiled rank and unit sensitivity are local robust-weighted numerical diagnostics, not covariance or independent evidence'),
            'poses':[{'frame_id':int(f), 'rotation_vector_rad':cp[i,:3].tolist(), 'translation_m':cp[i,3:].tolist()} for i,f in enumerate(frames)],
            'points':[{'track_id':int(t), 'position_m':points[i].tolist()} for i,t in enumerate(tracks)]}

    if physical:
        result['parameters_vector'] = exported(p[:ncal]).tolist()
    if nhand:
        result['camera_rotation_refinement_vector_rad'] = p[ncal:nshared].tolist()
    if frame_groups is not None:
        for i, pose in enumerate(result['poses']):
            pose['group_id'] = int(groups[i])
        for t, point in enumerate(result['points']):
            point['group_id'] = int(groups[fi[np.flatnonzero(ti == t)[0]]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON path or - for stdin')
    parser.add_argument('-o','--output')
    args = parser.parse_args()
    try:
        if args.input == '-':
            data = json.load(sys.stdin)
        else:
            with open(args.input, encoding='utf-8') as f:
                data = json.load(f)
        ray_models = None
        if data.get('camera_ray_models') is not None:
            from stereo_lens_calibration import CameraRayModel
            ray_models = [CameraRayModel(c['intrinsic'], c['physical_from_virtual'], c['lens'])
                          for c in data['camera_ray_models']]
        result = fit_temporal_calibration(data['observations'],data['K_left'],data['K_right'],data['baseline_m'],
            data.get('config'),data.get('model', 'relative4'),data.get('camera_rotations'),data.get('camera_motion'),
            ray_models,data.get('frame_groups'),data.get('refine_camera_rotation', False),data.get('fixed_rotation_initialization', False),data.get('global_rotation_initialization',False))
        encoded = json.dumps(result,indent=2,allow_nan=False)+'\n'
        if args.output:
            with open(args.output,'w',encoding='utf-8') as f:
                f.write(encoded)
        else:
            sys.stdout.write(encoded)
    except (ValueError,KeyError,TypeError,OSError,ImportError) as exc:
        parser.exit(2,f'error: {exc}\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())
