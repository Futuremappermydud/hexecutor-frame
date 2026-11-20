#!/usr/bin/env python3
"""Offline temporal stereo calibration on explicitly recorded unlit frames.

Tracks use v2 masked ORB and bounded bidirectional LK with stereo/temporal cycle
agreement. Frozen v2 source is instrumented in memory; it is never modified.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from evaluate_stereo_self_calibration import refine_matches, score, tag_prediction
from stereo_self_calibration import fit_calibration, project_right_points
from stereo_selfcal_validation import assess_update_stability

K = np.array([[380., 0., 308.], [0., 380., 308.], [0., 0., 1.]])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def match_pair(a, b, mask_a, mask_b):
    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=4,
                         edgeThreshold=31, patchSize=31, fastThreshold=12)
    k0, d0 = orb.detectAndCompute(a, mask_a)
    k1, d1 = orb.detectAndCompute(b, mask_b)
    seeds = []
    if d0 is not None and d1 is not None and len(d0) >= 2 and len(d1) >= 2:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        fwd, rev = bf.knnMatch(d0, d1, k=2), bf.knnMatch(d1, d0, k=2)
        good = {m.queryIdx: m.trainIdx for pair in rev if len(pair) == 2
                for m, n in [pair] if m.distance < .75*n.distance}
        for pair in fwd:
            if len(pair) == 2:
                m, n = pair
                if m.distance < .75*n.distance and good.get(m.trainIdx) == m.queryIdx:
                    seeds.append([*k0[m.queryIdx].pt, *k1[m.trainIdx].pt])
    return refine_matches([a, b], [mask_a, mask_b], np.asarray(seeds).reshape(-1, 4))


def _mutual_indices(query, target, tolerance):
    if not len(query) or not len(target):
        return {}
    d, index = cKDTree(target).query(query)
    _, reverse = cKDTree(query).query(target)
    return {i: int(j) for i, (dist, j) in enumerate(zip(d, index))
            if dist <= tolerance and reverse[j] == i}


def build_tracks(stereo_points, left_temporal, right_temporal, tolerance_px=2.0):
    """Return [frame, track, left xy, right xy] for connected consecutive tracks."""
    if len(left_temporal) != len(stereo_points)-1 or len(right_temporal) != len(stereo_points)-1:
        raise ValueError('one temporal match array is required for each adjacent frame pair')
    stereo = [np.asarray(x, dtype=float).reshape(-1, 4) for x in stereo_points]
    ids = [np.arange(len(stereo[0]), dtype=int)]
    next_id = len(ids[0])
    for frame in range(len(stereo)-1):
        maps = []
        for temporal, offset in ((left_temporal[frame], 0), (right_temporal[frame], 2)):
            temporal = np.asarray(temporal, dtype=float).reshape(-1, 4)
            src = _mutual_indices(temporal[:, :2], stereo[frame][:, offset:offset+2], tolerance_px)
            dst = _mutual_indices(temporal[:, 2:], stereo[frame+1][:, offset:offset+2], tolerance_px)
            maps.append({(src[i], dst[i]) for i in src.keys() & dst.keys()})
        edges = sorted(maps[0] & maps[1])
        new = np.full(len(stereo[frame+1]), -1, dtype=int)
        for a, b in edges:
            new[b] = ids[frame][a]
        for i in range(len(new)):
            if new[i] < 0:
                new[i] = next_id
                next_id += 1
        ids.append(new)
    records = [(frame, int(track), *point) for frame, (row, points) in enumerate(zip(ids, stereo))
               for track, point in zip(row, points)]
    counts = {}
    for row in records:
        counts[row[1]] = counts.get(row[1], 0) + 1
    return np.array([r for r in records if counts[r[1]] >= 2], dtype=float).reshape(-1, 6)


def connectivity(rows, frame_count):
    graph = {i: set() for i in range(frame_count)}
    for track in np.unique(rows[:, 1]) if len(rows) else []:
        frames = sorted(set(rows[rows[:, 1] == track, 0].astype(int)))
        for a, b in zip(frames, frames[1:]):
            graph[a].add(b); graph[b].add(a)
    seen, stack = set(), [0]
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v); stack.extend(graph[v] - seen)
    return {'connected': len(seen) == frame_count, 'reachable_frames': [int(x) for x in sorted(seen)],
            'edge_counts': [len(graph[i]) for i in range(frame_count)]}


def cap_training_tracks(rows, limit=200):
    # Frozen, deterministic round-robin over the first left point's 4x4 cell.
    cells = [[] for _ in range(16)]
    for track in np.unique(rows[:, 1]).astype(int):
        point = rows[rows[:, 1] == track][0, 2:4]
        x, y = np.clip((point / 154).astype(int), 0, 3)
        cells[y*4+x].append(track)
    selected = []
    while len(selected) < limit and any(cells):
        for cell in cells:
            if cell and len(selected) < limit:
                selected.append(cell.pop(0))
    return rows[np.isin(rows[:, 1], selected)]


def heldout_left_prediction(rows, ba, candidates, baseline):
    """Triangulate left-only rays with frozen accepted BA camera-to-world poses."""
    skipped = {'rejected_or_missing_poses': 0, 'insufficient_views': 0,
               'weak_parallax': 0, 'singular': 0, 'behind_camera': 0}
    tracks = np.unique(rows[:, 1]).astype(int) if len(rows) else []
    if not ba.get('accepted') or len(ba.get('poses', [])) == 0:
        skipped['rejected_or_missing_poses'] = len(tracks)
        return {'available_tracks': 0, 'skipped_tracks': skipped, 'scores': {}}
    poses = {int(p['frame_id']): (cv2.Rodrigues(np.asarray(p['rotation_vector_rad']))[0],
                                   np.asarray(p['translation_m'])) for p in ba['poses']}
    errors = {name: [] for name in candidates}
    available = 0
    for track in tracks:
        obs = rows[rows[:, 1] == track]
        obs = [r for r in obs if int(r[0]) in poses]
        if len(obs) < 2:
            skipped['insufficient_views'] += 1; continue
        rays, centers = [], []
        for r in obs:
            rot, center = poses[int(r[0])]
            ray = np.linalg.solve(K, np.array([r[2], r[3], 1.]))
            ray = rot @ ray; ray /= np.linalg.norm(ray)
            rays.append(ray); centers.append(center)
        dots = np.clip(np.asarray(rays) @ np.asarray(rays).T, -1., 1.)
        if np.degrees(np.arccos(dots.min())) < .5:
            skipped['weak_parallax'] += 1; continue
        A, b = np.zeros((3, 3)), np.zeros(3)
        for ray, center in zip(rays, centers):
            projection = np.eye(3) - np.outer(ray, ray)
            A += projection; b += projection @ center
        try:
            world = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            skipped['singular'] += 1; continue
        local = [(world - poses[int(r[0])][1]) @ poses[int(r[0])][0] for r in obs]
        if any(q[2] <= 0 for q in local):
            skipped['behind_camera'] += 1; continue
        calculated = {}
        try:
            for name, candidate in candidates.items():
                calculated[name] = project_right_points(np.asarray(local), K, candidate, baseline) - np.asarray([r[4:6] for r in obs])
        except (ValueError, TypeError, KeyError):
            skipped['behind_camera'] += 1; continue
        available += 1
        for name, error in calculated.items():
            errors[name].extend(error)
    def summarize(error):
        a = np.asarray(error)
        if not len(a):
            return None
        return {'observations': len(a), 'x_median_px': float(np.median(a[:, 0])),
                'x_p90_abs_px': float(np.percentile(np.abs(a[:, 0]), 90)),
                'y_median_px': float(np.median(a[:, 1])), 'y_p90_abs_px': float(np.percentile(np.abs(a[:, 1]), 90))}
    return {'available_tracks': available, 'skipped_tracks': skipped,
            'scores': {name: summarize(value) for name, value in errors.items()},
            'method': 'Left-only multi-view linear ray triangulation with frozen accepted BA poses; >=0.5 degree maximum parallax and positive depth; predicted minus observed right pixels.'}


def extract(manifest_path, frozen):
    source_path = Path(__file__).with_name('evaluate_stereo_self_calibration.py')
    if sha(source_path) != frozen['source_sha256'] or sha(manifest_path) != frozen['manifest_sha256']:
        raise ValueError('frozen source or manifest differs')
    source = source_path.read_text()
    patch = {'        parts[split].append(pts)': '        collect_frame(ordinal, images, masks, pts)\n        parts[split].append(pts)',
             "    for model in ('rotation3', 'relative4', 'full6'):": '    for model in ():'}
    for a, b in patch.items():
        if source.count(a) != 1:
            raise ValueError('frozen evaluator instrumentation anchor changed')
        source = source.replace(a, b)
    collected = {}
    def capture(ordinal, images, masks, pts):
        collected[ordinal] = {'images': [x.copy() for x in images], 'masks': [x.copy() for x in masks], 'stereo': pts.copy()}
    namespace = {'__name__': 'temporal_evaluation_extraction', '__file__': str(source_path), 'collect_frame': capture}
    exec(compile(source, str(source_path), 'exec'), namespace)
    observed = namespace['run'](str(manifest_path), matching='lk')
    for key in ('source_sha256', 'estimator_sha256', 'rectifier_sha256', 'calibration_sha256', 'observations_sha256', 'pairs_sha256', 'input_sha256'):
        if observed[key] != frozen[key]:
            raise ValueError('v2 provenance differs: ' + key)
    if [(f['ordinal'],f['matches'],f['orb_seed_count']) for f in observed['frames']] != [(f['ordinal'],f['matches'],f['orb_seed_count']) for f in frozen['frames']]:
        raise ValueError('v2 per-frame counts differ')
    return collected, observed


def select_external_tags(manifest_path, excluded_pair_ids):
    """Read original held-out observations keyed by source pair, not dense ordinal."""
    manifest = json.loads(Path(manifest_path).read_text())
    pairs = [json.loads(line) for line in Path(manifest['pairs']).read_text().splitlines() if line.strip()]
    by_left, pair_ids = {}, set()
    for pair in pairs:
        if pair['left_file'] in by_left or pair['pair'] in pair_ids:
            raise ValueError('duplicate external pair filename or id')
        by_left[pair['left_file']] = (pair['pair'], pair['right_file'])
        pair_ids.add(pair['pair'])
    frames = {}
    for frame in manifest['frames']:
        if frame['ordinal'] in frames:
            raise ValueError('duplicate external ordinal')
        name = Path(frame['left']).name
        if name not in by_left:
            raise ValueError('external frame absent from pair metadata')
        pair_id, expected_right = by_left[name]
        if Path(frame['right']).name != expected_right:
            raise ValueError('external left/right files do not identify the same pair')
        frames[frame['ordinal']] = (frame['split'], pair_id)
    observations = json.loads(Path(manifest['observations']).read_text())
    if any(o['ordinal'] not in frames for o in observations):
        raise ValueError('tag observation has unknown external ordinal')
    selected = [o for o in observations if o['ordinal'] in frames and
                frames[o['ordinal']][0] == 'heldout' and frames[o['ordinal']][1] not in excluded_pair_ids]
    provenance = {'manifest': str(Path(manifest_path).resolve()), 'manifest_sha256': sha(manifest_path),
                  'observations_sha256': sha(manifest['observations']), 'pairs_sha256': sha(manifest['pairs']),
                  'selection': 'Original heldout ordinals mapped to source pair IDs; excluded only when source pair ID belongs to fitted window.'}
    return selected, provenance


def evaluate(manifest_path, frozen_path, starts, window_length=5, external_tag_manifest=None,
             track_builder=None, temporal_model="relative4", camera_rotations_by_ordinal=None,
             camera_motion_by_ordinal=None, track_filter=None):
    if temporal_model not in ('relative4', 'rotation3'):
        raise ValueError('temporal model must be relative4 or rotation3')
    temporal_key = 'temporal_' + temporal_model
    if not 2 <= window_length <= 12:
        raise ValueError('window length must be between 2 and 12')
    frozen = json.loads(Path(frozen_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    temporal = importlib.import_module('stereo_temporal_calibration')
    images, observed = extract(manifest_path, frozen)
    obs = json.loads(Path(manifest['observations']).read_text())
    baseline = observed['baseline_m']
    times = {e['left_file']: e['left_raw_ns'] for e in map(json.loads, Path(manifest['pairs']).read_text().splitlines())}
    frame_meta = {f['ordinal']: f for f in manifest['frames']}
    reference_ns = min(times[Path(f['left']).name] for f in manifest['frames'])
    windows = []
    for start in starts:
        ids = list(range(start, start + window_length))
        if not all(i in images for i in ids):
            available = [frame_meta[i] for i in ids if i in frame_meta]
            scope = ('heldout_diagnostic' if any(f.get('split') == 'heldout' for f in available)
                     else 'training' if available and all(f.get('split') == 'train' for f in available) else 'unknown')
            windows.append({'start': start, 'ordinals': ids, 'window_scope': scope,
                            'error': 'missing frame(s)'}); continue
        frames = [images[i] for i in ids]
        fixed_rotations = None
        camera_motion = None
        if camera_rotations_by_ordinal is not None:
            world_rotations = np.array([camera_rotations_by_ordinal[i] for i in ids])
            fixed_rotations = world_rotations[0].T @ world_rotations
            if camera_motion_by_ordinal is not None:
                camera_motion = {}
                for key in ('translation', 'lever_arm'):
                    values = np.array([camera_motion_by_ordinal[key][i] for i in ids])
                    camera_motion[key] = (values-values[0]) @ world_rotations[0]
        elif camera_motion_by_ordinal is not None:
            raise ValueError('camera motion requires independent rotations')
        stereo = [f['stereo'] for f in frames]
        left, right = [], []
        track_diagnostics = {}
        if track_builder is None:
            for a, b in zip(frames, frames[1:]):
                left.append(match_pair(a['images'][0], b['images'][0], a['masks'][0], b['masks'][0]))
                right.append(match_pair(a['images'][1], b['images'][1], a['masks'][1], b['masks'][1]))
            rows = build_tracks(stereo, left, right)
        else:
            rows, track_diagnostics = track_builder(frames)
        if track_filter is not None:
            rows, filter_diagnostics = track_filter(rows, ids)
            track_diagnostics.update(filter_diagnostics)
        train_all, held = rows[(rows[:, 1].astype(int) % 5) != 0], rows[(rows[:, 1].astype(int) % 5) == 0]
        train = cap_training_tracks(train_all)
        entry = {'start': start, 'ordinals': ids, 'seconds_from_first_reference': [(times[Path(frame_meta[i]['left']).name] - reference_ns) / 1e9 for i in ids], 'track_count': len(np.unique(rows[:, 1])),
                 'observation_count': len(rows), 'training_tracks_before_cap': len(np.unique(train_all[:, 1])), 'training_tracks': len(np.unique(train[:, 1])),
                 'heldout_tracks': len(np.unique(held[:, 1])), 'training_observations': len(train),
                 'heldout_observations': len(held), 'all_connectivity': connectivity(rows, window_length),
                 'training_connectivity': connectivity(train, window_length),
                 'stereo_counts': [len(a) for a in stereo], 'left_temporal_counts': [len(a) for a in left],
                 'right_temporal_counts': [len(a) for a in right],
                 'heldout_natural_uncorrected': score(held[:, 2:], None, K)}
        entry.update(track_diagnostics)
        entry['window_scope'] = 'training' if all(frame_meta[i]['split'] == 'train' for i in ids) else 'heldout_diagnostic'
        if not entry['training_connectivity']['connected']:
            entry['error'] = 'training_track_graph_disconnected_after_cap'
            windows.append(entry)
            continue
        if external_tag_manifest:
            source_ids = {frame_meta[i]['source_pair_id'] for i in ids}
            tag_obs, external_provenance = select_external_tags(external_tag_manifest, source_ids)
            entry['external_tag_provenance'] = external_provenance
        else:
            heldout_frame_ids = {f['ordinal'] for f in manifest['frames'] if f['split'] == 'heldout'} - set(ids)
            tag_obs = [o for o in obs if o['ordinal'] in heldout_frame_ids]
        for name in ('relative4', temporal_key):
            try:
                fit = (fit_calibration(train[:, 2:], K, K, model='relative4') if name == 'relative4'
                       else temporal.fit_temporal_calibration(train, K, K, baseline, model=temporal_model,
                                                              camera_rotations=fixed_rotations,
                                                              camera_motion=camera_motion))
                entry[name] = {'fit': fit, 'heldout_natural': score(held[:, 2:], fit, K)}
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                entry[name] = {'error': str(e)}
        for name in ('relative4', temporal_key):
            if 'fit' in entry[name]:
                entry[name]['heldout_tags'] = tag_prediction(tag_obs, entry[name]['fit'], K, baseline, manifest['tag_size_m'])
        entry['tag_prediction_uncorrected'] = tag_prediction(tag_obs, None, K, baseline, manifest['tag_size_m'])
        entry['tag_observations_scope'] = ('external manifest heldout observations excluding window source-pair IDs' if external_tag_manifest else 'manifest-heldout frames outside this window; exploratory tag confidence retained')
        ba = entry[temporal_key].get('fit', {})
        relative = entry['relative4'].get('fit')
        candidates = {'uncorrected': {'parameters': {'rotation_vector_rad': [0., 0., 0.], 'focal_scale': 1.}}}
        if relative is not None:
            candidates['relative4'] = relative
        if ba.get('accepted'):
            candidates[temporal_key] = ba
        entry['heldout_left_only_predictions'] = heldout_left_prediction(held, ba, candidates, baseline)
        windows.append(entry)
    training_windows = [w for w in windows if w.get('window_scope') != 'heldout_diagnostic']
    fits = [w[temporal_key]['fit'] for w in training_windows if temporal_key in w and 'fit' in w[temporal_key] and w[temporal_key]['fit'].get('accepted')]
    omitted = [w['start'] for w in training_windows if not w.get(temporal_key, {}).get('fit', {}).get('accepted')]

    guard = (assess_update_stability(fits[0], fits[1:], K, K, baseline) if len(fits) >= 2
             else {'stability_passed': False, 'reasons': ['insufficient_independent_accepted_window_fits']})
    guard['accepted_window_count'] = len(fits)
    guard['requested_training_window_count'] = len(training_windows)
    guard['omitted_rejected_or_failed_training_starts'] = omitted
    guard['all_training_windows_accepted'] = not omitted
    guard['session_stability_passed'] = bool(guard['stability_passed'] and not omitted)
    return {'source_sha256': sha(__file__), 'evaluator_sha256': observed['source_sha256'],
            'estimator_sha256': sha(Path(__file__).with_name('stereo_self_calibration.py')),
            'temporal_sha256': sha(temporal.__file__), 'validation_sha256': sha(Path(__file__).with_name('stereo_selfcal_validation.py')), 'manifest_sha256': sha(manifest_path), 'v2_result_sha256': sha(frozen_path),
            'observed_provenance': {k: observed[k] for k in ('rectifier_sha256','calibration_sha256','observations_sha256','pairs_sha256','input_sha256')},
            'baseline_m': baseline, 'temporal_model': temporal_model, 'windows': windows, 'independent_window_stability': guard,
            'window_length': window_length, 'method': 'Consecutive frames, v2 ORB mutual ratio and per-eye LK + FB + ZNCC. Stereo/adjacent temporal endpoint mutual nearest <=2 px in both eyes, agreeing stereo indices; tracks >=2 frames; track id %5 == 0 held out; up to 200 training tracks selected round-robin 4x4 grid by ascending ID. No row restriction.',
            'caveats': ['Tag masks are incomplete for undetected tags.', 'Natural tracks, overlapping windows and repeated frames are correlated.',
                        'Track-heldout feature correspondences are not metric truth; tag predictions depend on baseline, tag size and planar pose.',
                        'Estimator acceptance is not metric or deployment acceptance.']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest'); ap.add_argument('v2_result'); ap.add_argument('output')
    ap.add_argument('--start', action='append', type=int, required=True)
    ap.add_argument('--window-length', type=int, default=5)
    ap.add_argument('--external-tag-manifest')
    args = ap.parse_args()
    Path(args.output).write_text(json.dumps(evaluate(args.manifest, args.v2_result, args.start, args.window_length, args.external_tag_manifest), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
