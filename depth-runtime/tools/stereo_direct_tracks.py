"""Offline experimental propagation of stereo features without ORB redetection.

Pyramidal LK proposes temporal endpoints; ROI-normalized bidirectional refinement
and an independent stereo closure check retain them. No calibration, poses or
tags enter association. Caller masks include descriptor/tracking footprints.
"""
import cv2
import numpy as np
from stereo_photometric_matching import refine_matches


def propagate(a, b, mask_a, mask_b, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return np.empty((0, 2)), np.empty(0, dtype=int)
    target, status, _ = cv2.calcOpticalFlowPyrLK(
        a, b, points.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    if target is None or status is None:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    target = target.reshape(-1, 2)
    valid = np.flatnonzero(status.ravel().astype(bool) & np.isfinite(target).all(axis=1))
    refined, kept = refine_matches([a, b], [mask_a, mask_b],
        np.column_stack((points[valid], target[valid])), return_indices=True)
    return refined[:, 2:], valid[kept]


def refine_from_birth(frames, current_frame, ids, proposal, births):
    """Refine each eye against its frozen birth patch; never use geometry."""
    selected = []
    output = []
    for index, track in enumerate(ids):
        birth_frame, birth_pixels = births[int(track)]
        refined = []
        for eye in range(2):
            match = np.r_[birth_pixels[2*eye:2*eye+2], proposal[index, 2*eye:2*eye+2]][None]
            value = refine_matches([frames[birth_frame]['images'][eye], frames[current_frame]['images'][eye]],
                                   [frames[birth_frame]['masks'][eye], frames[current_frame]['masks'][eye]], match)
            if not len(value):
                break
            refined.extend(value[0, 2:])
        if len(refined) == 4:
            output.append(refined); selected.append(index)
    return np.asarray(output, float).reshape(-1, 4), np.asarray(selected, int)


def track_window(frames, anchor_refinement=False):
    """Return observations and diagnostics; spawn separated frozen stereo seeds.

    New seeds must be >=8 px from active tracks in BOTH eyes. Stereo closure
    must agree within .5 px with independently propagated right endpoints;
    retain those temporal endpoints to avoid feeding stereo refinement back
    into future right temporal observations. A failed track never reconnects.
    Optional anchor_refinement additionally refines each proposed eye endpoint
    against the frozen birth patch before closure; no calibration enters it.
    """
    records, active, ids = [], np.empty((0, 4)), np.empty(0, dtype=int)
    if not isinstance(anchor_refinement, bool):
        raise ValueError('anchor_refinement must be boolean')
    birth_patches = {}
    next_id = 0
    left_counts, right_counts, closure_counts, births = [], [], [], []
    for frame, current in enumerate(frames):
        if frame:
            previous = frames[frame-1]
            left, li = propagate(previous['images'][0], current['images'][0],
                                 previous['masks'][0], current['masks'][0], active[:, :2])
            right, ri = propagate(previous['images'][1], current['images'][1],
                                  previous['masks'][1], current['masks'][1], active[:, 2:])
            left_counts.append(len(li)); right_counts.append(len(ri))
            common, il, ir = np.intersect1d(li, ri, return_indices=True)
            proposal = np.column_stack((left[il], right[ir]))
            proposal_ids = ids[common]
            if anchor_refinement:
                proposal, anchored = refine_from_birth(frames, frame, proposal_ids, proposal, birth_patches)
                proposal_ids = proposal_ids[anchored]
            closure, kept = refine_matches(current['images'], current['masks'], proposal,
                                            return_indices=True)
            kept = kept[np.linalg.norm(closure[:, 2:] - proposal[kept, 2:], axis=1) <= .5]
            active, ids = proposal[kept], proposal_ids[kept]
            closure_counts.append(len(active))
        new_count = 0
        for seed in np.asarray(current['stereo']).reshape(-1, 4):
            if len(active) and (np.any(np.linalg.norm(active[:, :2]-seed[:2], axis=1) < 8.) or
                                np.any(np.linalg.norm(active[:, 2:]-seed[2:], axis=1) < 8.)):
                continue
            active = np.vstack((active, seed))
            if anchor_refinement:
                birth_patches[next_id] = (frame, np.array(seed, copy=True))
            ids = np.r_[ids, next_id]
            next_id += 1
            new_count += 1
        if anchor_refinement:
            birth_patches = {int(i): birth_patches[int(i)] for i in ids}
        births.append(new_count)
        records.extend((frame, int(track), *point) for track, point in zip(ids, active))
    rows = np.asarray(records, dtype=float).reshape(-1, 6)
    if len(rows):
        tracks, counts = np.unique(rows[:, 1], return_counts=True)
        rows = rows[np.isin(rows[:, 1], tracks[counts >= 2])]
    return rows, {'left_temporal_counts': left_counts, 'right_temporal_counts': right_counts,
                  'stereo_closure_counts': closure_counts, 'birth_counts': births,
                  'anchor_refinement': anchor_refinement}


def filter_left_parallax(rows, rotations, intrinsic, minimum_degrees=.5, minimum_views=3):
    """Select whole tracks using left rays and independently supplied rotations.

    Rotations map each left camera to a common world frame. Rotation-induced
    image movement is removed before measuring the maximum ray angle. Neither
    right pixels, fitted scene depth, translation scale nor tags select support.
    """
    rows = np.asarray(rows, float)
    rotations, intrinsic = np.asarray(rotations, float), np.asarray(intrinsic, float)
    if (rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all() or
            rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not np.isfinite(rotations).all() or
            intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()):
        raise ValueError('finite observations, rotations and intrinsic matrix required')
    if (not np.isfinite(minimum_degrees) or not 0 < minimum_degrees < 180 or
            isinstance(minimum_views, bool) or not isinstance(minimum_views, int) or minimum_views < 3):
        raise ValueError('positive parallax below 180 degrees and at least three views required')
    if (not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-6) or
            not np.allclose(np.linalg.det(rotations), 1., atol=1e-6)):
        raise ValueError('proper camera-to-world rotations required')
    if (np.any(rows[:, :2] != np.floor(rows[:, :2])) or np.any(rows[:, 0] < 0) or
            np.any(rows[:, 0] >= len(rotations)) or len(np.unique(rows[:, :2], axis=0)) != len(rows)):
        raise ValueError('unique integer frame/track IDs in rotation-array range required')
    inverse = np.linalg.inv(intrinsic)
    selected, tracks = [], []
    for track in np.unique(rows[:, 1]):
        subset = rows[rows[:, 1] == track]
        rays = np.column_stack((subset[:, 2:4], np.ones(len(subset)))) @ inverse.T
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        world = np.einsum('nij,nj->ni', rotations[subset[:, 0].astype(int)], rays)
        angle = float(np.degrees(np.arccos(np.clip(np.min(world @ world.T), -1., 1.))))
        accepted = len(subset) >= minimum_views and angle >= minimum_degrees
        tracks.append({'track_id': int(track), 'views': len(subset), 'parallax_degrees': angle, 'retained': accepted})
        if accepted:
            selected.append(track)
    return rows[np.isin(rows[:, 1], selected)], {'left_parallax_filter': {
        'minimum_degrees': minimum_degrees, 'minimum_views': minimum_views,
        'input_tracks': len(tracks), 'retained_tracks': len(selected), 'tracks': tracks,
        'scope': 'Support only; noisy rays, timing, tracking and extrinsic errors can mimic parallax. No metric acceptance.'}}
