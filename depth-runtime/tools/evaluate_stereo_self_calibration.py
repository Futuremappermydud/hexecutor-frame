#!/usr/bin/env python3
"""Offline saved-image evaluation; see --help for explicit manifest format.

Uses a saved KB4 rectifier snapshot to form 616 px pinhole images and ORB
full-image descriptor matching. This is not the matcher in Wang et al.
Tag detections only exclude pixels and provide held-out image-space scores.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
from stereo_self_calibration import fit_calibration, rectify_points


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(a):
    a = np.asarray(a)
    if not a.size:
        return None
    return {"median": float(np.median(a)), "p90_abs": float(np.percentile(abs(a), 90)),
            "rms": float(np.sqrt(np.mean(a * a)))}


def score(points, fitted, K):
    if not len(points):
        return None
    corrected = points if fitted is None else rectify_points(points, K, K, fitted)
    before_d = points[:, 0] - points[:, 2]
    return {"count": len(points), "vertical_px": summary(corrected[:, 1] - corrected[:, 3]),
            "disparity_px": summary(corrected[:, 0] - corrected[:, 2]),
            "disparity_change_px": summary(corrected[:, 0] - corrected[:, 2] - before_d)}


def refine_matches(images, masks, matches, return_indices=False):
    """Refine mutually unique ORB seeds with bounded 2D forward/backward LK.

    Masks are already eroded for the largest descriptor and tracking footprints.
    Segment validation covers each straight seed-to-end path, not internal LK iterates.
    """
    pts = np.asarray(matches, dtype=np.float32).reshape(-1, 4)
    if not len(pts):
        empty = np.empty((0, 4), dtype=float)
        return (empty, np.empty(0, dtype=int)) if return_indices else empty
    left = pts[:, :2].reshape(-1, 1, 2).copy()
    seed_right = pts[:, 2:].reshape(-1, 1, 2).copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)
    kw = dict(winSize=(15, 15), maxLevel=0, criteria=criteria,
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
    right, status_f, _ = cv2.calcOpticalFlowPyrLK(images[0], images[1], left, seed_right.copy(), **kw)
    back, status_b, _ = cv2.calcOpticalFlowPyrLK(images[1], images[0], right, left.copy(), **kw)
    right, back = right.reshape(-1, 2), back.reshape(-1, 2)
    def safe_segment(mask, start, end):
        # Half-pixel spacing prevents skipping a masked pixel between endpoints.
        steps = max(1, int(np.ceil(np.linalg.norm(end - start) * 2)))
        points = np.rint(np.linspace(start, end, steps + 1)).astype(int)
        valid = ((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) &
                 (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0]))
        return bool(np.all(valid) and np.all(mask[points[:, 1], points[:, 0]] != 0))
    out, indices = [], []
    for i, p in enumerate(pts):
        if not status_f[i, 0] or not status_b[i, 0] or not np.isfinite(right[i]).all() or not np.isfinite(back[i]).all():
            continue
        if np.linalg.norm(right[i] - p[2:]) > 2 or np.linalg.norm(back[i] - p[:2]) > .5:
            continue
        if not safe_segment(masks[0], p[:2], back[i]) or not safe_segment(masks[1], p[2:], right[i]):
            continue
        patch_l = cv2.getRectSubPix(images[0], (15, 15), tuple(map(float, p[:2]))).astype(float)
        patch_r = cv2.getRectSubPix(images[1], (15, 15), tuple(map(float, right[i]))).astype(float)
        patch_l -= patch_l.mean(); patch_r -= patch_r.mean()
        norm = np.linalg.norm(patch_l) * np.linalg.norm(patch_r)
        if np.std(patch_l) < 4 or np.std(patch_r) < 4 or np.sum(patch_l * patch_r) / norm < .8:
            continue
        out.append([*p[:2], *right[i]])
        indices.append(i)
    result = np.asarray(out, dtype=float).reshape(-1, 4)
    return (result, np.asarray(indices, dtype=int)) if return_indices else result


def tag_prediction(observations, fitted, K, baseline, tag_size):
    # Fit each pose from its left corners only. Square planar ambiguity remains.
    h = tag_size / 2
    obj = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], dtype=float)
    errors, left_errors = [], []
    for obs in observations:
        left, right = np.asarray(obs['left'], float), np.asarray(obs['right'], float)
        ok, rvec, tvec = cv2.solvePnP(obj, left, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        q = obj @ rotation.T + tvec.ravel()
        if np.any(q[:, 2] <= 0):
            continue
        left_proj = q @ K.T
        left_pred = left_proj[:, :2] / left_proj[:, 2, None]
        params = fitted['parameters'] if fitted is not None else {}
        rotation, _ = cv2.Rodrigues(np.asarray(params.get('rotation_vector_rad', [0., 0., 0.]), dtype=float))
        direction = np.asarray(params.get('baseline_direction_left', [1., 0., 0.]), dtype=float)
        q = (q - baseline * direction) @ rotation
        scale = float(params.get('focal_scale', 1.))
        expected = q[:, :2] / q[:, 2, None] * np.array([K[0, 0], K[1, 1]]) * scale + K[:2, 2]
        errors.extend(expected - right)
        left_errors.extend(left - left_pred)
    if not errors:
        return None
    err = np.asarray(errors)
    return {'tag_observation_count': len(err) // 4, 'right_corner_count': len(err),
            'left_plus_right_corner_count': len(err) * 2, 'right_error_x_px': summary(err[:, 0]),
            'right_error_y_px': summary(err[:, 1]), 'left_fit_rms_px': float(np.sqrt(np.mean(np.square(left_errors))))}


def coverage(points):
    if not len(points):
        return {"occupied_cells": 0, "grid": [4, 4]}
    xy = np.clip((points[:, :2] / 154).astype(int), 0, 3)
    return {"occupied_cells": len(set(map(tuple, xy))), "grid": [4, 4]}


def run(manifest_path, matching='orb'):
    manifest = json.loads(Path(manifest_path).read_text())
    snapshot = Path(manifest["rectifier_source"]).resolve()
    calib = Path(manifest["calibration"]).resolve()
    os.environ.update(IR_CALIB=str(calib), IR_PITCH="0.25", IR_ROLL_DEG="0.4", IR_YAW_DEG="0")
    spec = importlib.util.spec_from_file_location("saved_ir_dot_lib", snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rect = module.Rectifier(rig=module.load_rig(calib))
    xy = np.indices((616, 616), dtype=np.float64)
    rays = np.stack(((xy[1]-308)/380, (xy[0]-308)/380, np.ones((616, 616)))).reshape(3, -1)
    maps = []
    for rot, lens in ((rect.RL, rect.kl), (rect.RR, rect.kr)):
        x, y = lens.project(rot @ rays)
        maps.append((x.reshape(616, 616).astype('float32'), y.reshape(616, 616).astype('float32')))
    obs_path = Path(manifest['observations']).resolve()
    observations = json.loads(obs_path.read_text())
    by_frame = {}
    for o in observations:
        by_frame.setdefault(o['ordinal'], []).append(o)
    from pupil_apriltags import Detector
    det = Detector(families='tag16h5', nthreads=2, quad_decimate=1, quad_sigma=0,
                   refine_edges=1, decode_sharpening=.25)
    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=4, edgeThreshold=31,
                         patchSize=31, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    K = np.array([[380., 0, 308], [0, 380., 308], [0, 0, 1]])
    parts = {'train': [], 'heldout': []}
    seed_parts = {'train': [], 'heldout': []}
    tag_parts = {'train': [], 'heldout': []}
    frames, hashes, frame_points = [], {}, []
    for frame in manifest['frames']:
        ordinal, split = frame['ordinal'], frame['split']
        if split not in parts:
            raise ValueError('split must be train or heldout')
        images, masks, detections = [], [], []
        for side, mapping in zip(('left', 'right'), maps):
            p = Path(frame[side]).resolve()
            hashes[str(p)] = sha(p)
            raw = np.fromfile(p, np.uint8).reshape(1024, 1056)
            im = cv2.remap(raw, *mapping, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = np.where((mapping[0] >= 1) & (mapping[0] < 1054) &
                            (mapping[1] >= 1) & (mapping[1] < 1022), 255, 0).astype('uint8')
            ds = [d for d in det.detect(im) if d.decision_margin >= 7]
            polygons = [np.asarray(d.corners) for d in ds]
            polygons += [np.asarray(o[side]) for o in by_frame.get(ordinal, [])]
            for polygon in polygons:
                cv2.fillConvexPoly(mask, np.rint(polygon).astype('int32'), 0)
            # Include the ORB descriptor footprint and a margin around each polygon.
            mask = cv2.erode(mask, np.ones((97, 97), np.uint8))
            images.append(im); masks.append(mask); detections.append(len(ds))
        k0, d0 = orb.detectAndCompute(images[0], masks[0])
        k1, d1 = orb.detectAndCompute(images[1], masks[1])
        matches = []
        if d0 is not None and d1 is not None and len(d0) >= 2 and len(d1) >= 2:
            fwd = matcher.knnMatch(d0, d1, k=2)
            rev = matcher.knnMatch(d1, d0, k=2)
            good = {m.queryIdx: m.trainIdx for pair in rev if len(pair) == 2
                    for m, n in [pair] if m.distance < .75*n.distance}
            for pair in fwd:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < .75*n.distance and good.get(m.trainIdx) == m.queryIdx:
                    matches.append([*k0[m.queryIdx].pt, *k1[m.trainIdx].pt])
        pts = np.asarray(matches, dtype=float).reshape(-1, 4)
        seed_count = len(pts)
        if matching == 'lk':
            seeds = pts
            pts, selected = refine_matches(images, masks, pts, return_indices=True)
            seed_parts[split].append(seeds[selected])
        else:
            seed_parts[split].append(pts)
        parts[split].append(pts)
        frame_points.append(pts)
        tags = [np.column_stack((np.asarray(o['left']), np.asarray(o['right'])))
                for o in by_frame.get(ordinal, [])]
        tag_parts[split].extend(tags)
        frames.append({'ordinal': ordinal, 'split': split, 'matches': len(pts),
                       'coverage': coverage(pts), 'orb_seed_count': seed_count, 'independent_detections': detections,
                       'supplied_stereo_observations': len(tags)})
    def cat(items):
        return np.concatenate(items) if items else np.empty((0, 4))
    train, held = cat(parts['train']), cat(parts['heldout'])
    held_seeds = cat(seed_parts['heldout'])
    tags = cat(tag_parts['heldout'])
    held_ordinals = {f['ordinal'] for f in frames if f['split'] == 'heldout'}
    held_tag_obs = [o for o in observations if o['ordinal'] in held_ordinals]
    tag_size = float(manifest['tag_size_m'])
    fits = {}
    for model in ('rotation3', 'relative4', 'full6'):
        try:
            result = fit_calibration(train, K, K, model=model)
            frame_scores = [dict(ordinal=f['ordinal'], score=score(p, result, K))
                            for f, p in zip(frames, frame_points) if f['split'] == 'heldout']
            spatial_scores = {}
            for y in range(4):
                for x in range(4):
                    sel = ((held[:, 0] >= x*154) & (held[:, 0] < (x+1)*154) &
                           (held[:, 1] >= y*154) & (held[:, 1] < (y+1)*154))
                    if np.any(sel):
                        spatial_scores[f'{x},{y}'] = score(held[sel], result, K)
            fits[model] = {'fit': result, 'heldout': score(held, result, K),
                           'heldout_tags': score(tags, result, K),
                           'heldout_tag_prediction': tag_prediction(held_tag_obs, result, K, rect.baseline, tag_size),
                           'heldout_by_frame': frame_scores, 'heldout_by_spatial_cell': spatial_scores}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            fits[model] = {'error': str(exc)}
    return {'manifest_sha256': sha(manifest_path), 'source_sha256': sha(__file__),
            'estimator_sha256': sha(Path(__file__).with_name('stereo_self_calibration.py')),
            'rectifier_sha256': sha(snapshot), 'calibration_sha256': sha(calib),
            'observations_sha256': sha(obs_path), 'pairs_sha256': sha(manifest['pairs']), 'input_sha256': hashes,
            'matching_mode': matching,
            'refinement': ('LK 15x15 maxLevel=0, initial ORB flow, 30 iterations epsilon 0.01, minEig 1e-4; backward initialized at original left; both status, seed movement <=2 px, FB <=0.5 px; both bilinear patches std >=4 and ZNCC >=0.8; straight paths in pre-eroded masks' if matching == 'lk' else None),
            'matching': 'ORB 1800 features, four levels; full 2D BF Hamming; mutual nearest with 0.75 ratio in both directions; 48 px polygon margin; no disparity or row search restriction; ORB seed keypoints use integer/pyramid localization. This differs from the cited paper matcher.',
            'mask_provenance': 'Each view: supplied stereo-observation polygons plus independently detected tag16h5, detector-default hamming correction, margin >=7; undetected or undecoded tags can remain. No guarantee all tag footprints are excluded.',
            'caveats': ['Repeated image features are correlated, not independent observations.',
                        'Tag corner image agreement is not independent metric depth truth.',
                        'Fixed rectifier has historical pitch/roll corrections; capture-time calibration identity may differ.',
                        'Metric target 0.3 disparity pixels at 0.7–2.5 m remains unevaluated.'],
            'tag_prediction_convention': 'Left-only PnP on manifest-size square. Predicted minus observed original right pixels: R.T @ (left point - baseline * direction), focal scale applied; original 616 px pinhole grids. Planar ambiguity, lens, baseline and tag size conditional; no metric acceptance.',
            'baseline_m': rect.baseline, 'baseline_convention': 'positive norm of c_right minus c_left; expected right point in common rectified axes is left point minus [baseline, 0, 0]', 'tag_size_m': tag_size,
            'acceptance_scope': 'Estimator accepted means training rectification checks only; not metric accuracy or live deployment acceptance.',
            'score_convention': 'Model disparity change compares corrected coordinates to original coordinates; full6 rotates both views to align the estimated baseline so this difference includes that common reorientation.',
            'frames': frames, 'train': {'count': len(train), 'coverage': coverage(train)},
            'heldout': {'coverage': coverage(held), 'uncorrected': score(held, None, K),
                        'retained_orb_seeds_uncorrected': score(held_seeds, None, K),
                        'refinement_right_displacement_px': {'x': summary(held[:, 2] - held_seeds[:, 2]), 'y': summary(held[:, 3] - held_seeds[:, 3])},
                        'tags_uncorrected': score(tags, None, K), 'tag_prediction_uncorrected': tag_prediction(held_tag_obs, None, K, rect.baseline, tag_size)}, 'models': fits}


def main():
    parser = argparse.ArgumentParser(description=__doc__ + '\nManifest JSON: rectifier_source, calibration, observations, pairs, tag_size_m, frames [{ordinal, split, left, right}]; all paths explicit.')
    parser.add_argument('manifest'); parser.add_argument('output')
    parser.add_argument('--matching', choices=('orb', 'lk'), default='orb')
    args = parser.parse_args()
    result = run(args.manifest, args.matching)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
