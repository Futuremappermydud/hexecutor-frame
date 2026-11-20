#!/usr/bin/env python3
"""E09 plane detection on the fused TSDF: offline reference over periodic mesh snapshots.

Offline development only. Replaces per-frame detection (plane_patches.cpp), which failed
on real data because per-frame model depth ripples (1-2 cm at 5-10 cm scale) and walls
dome. The TSDF averages up to 32 frames from many views.

  snapshots  convert the arm's depth and replay the live TSDF (3 cm / 12 cm / 4 m), writing
             a mesh snapshot every N frames -> OUT/ARM/tsdf-planes/snap-FRAME.f32
  detect     planes per snapshot with persistent ids -> OUT/ARM/tsdf-planes/planes.json
  score-tags / score-scene   the E09 gate declared in plane_patch_test.py, judged per
             snapshot instead of per frame (coverage and identity over snapshots taken
             after a tag's first detection)

Detection (parameters fixed before the first real run): triangles are binned by centroid
into 5 cm voxels with area-weighted point scatter and normal tensor. A voxel is a seed
candidate when its normal tensor is concentrated (largest eigenvalue >= 0.85 of the area).
Regions grow over 26-neighbours while the voxel normal is within ANGLE_DEG of the region
normal and its centroid within DIST_M of the region plane; regions of at least MIN_AREA_M2
with RMS <= MAX_RMS_M and a fitted normal within ANGLE_DEG of their area-weighted surface
normal are planes. Persistent identity: a plane takes the id of the previous
snapshot's plane with the largest voxel overlap (>= 30% of the smaller, normals within
10 degrees, old centroid within 5 cm of the new plane); otherwise it gets a new id.
Merge pass (2026-11-08, after the first IR-dot run, before any rerun): regions whose voxels
touch, whose normals agree within 10 degrees, whose centroids lie within DIST_M of each
other's plane and whose union keeps RMS <= MAX_RMS_M are merged (growth can break between
coplanar parts of one wall). Revised the same day after a development-data look at tag 1:
the smaller region's centroid is tested against the larger region's plane only.

Relative wall check (d'), declared 2026-11-08 before any rerun, replacing (d) for planes
whose position is not the RGB mesh's (IR-dot planes): when tags 1 and 2 share a final
plane, the PnP tag centres' signed distances from it must agree within 15 mm. PnP uses no
stereo and a constant plane offset cancels, so this tests flatness and tilt across the
wall without D1. (d) is still reported.

Identity fix (2026-11-08, after the first run, before any gated run): matches are made
against planes seen in the last 3 snapshots, and a plane that merges several keeps the
earliest-born id.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import (EXPOSURE_OFFSET_MS, PATHS, capture_frames, frame_paths,  # noqa: E402
                                 segments_from_marks, write_json)

VOXEL_M = 0.05
ANGLE_DEG, DIST_M = 15.0, 0.02
MIN_AREA_M2, MAX_RMS_M = 0.04, 0.02
SEED_CONCENTRATION = 0.85
MATCH_OVERLAP, MATCH_ANGLE_DEG, MATCH_OFFSET_M = 0.3, 10.0, 0.05
MEMORY_SNAPSHOTS = 3  # a plane missing for up to 2 snapshots keeps its id when it returns
OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
UP = np.array([0.0, 1.0, 0.0])


def snapshots(args):
    from room_dataset import load_poses, write_depth_metadata
    from room_reconstruction_replay import replay_stream
    paths = frame_paths(args.session / 'rgb')
    depth = args.out / args.arm / 'depth'
    snaps = args.out / args.arm / 'tsdf-planes'
    depth.mkdir(parents=True, exist_ok=True)
    snaps.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.converter), str(args.session / 'rgb'), str(args.out / args.arm / 'model'),
                    str(depth), str(len(paths))], check=True)
    extrinsic = json.loads((args.session / 'rgb/rectified-camera-in-head.json').read_text())
    write_depth_metadata(paths, depth, load_poses(args.session / 'head_pose.csv'), extrinsic, EXPOSURE_OFFSET_MS)
    write_json(depth / 'capture.json', {'version': 2, 'model': 'primary', 'backend': 'prepared-qnn',
                                        'projection': 'spherical', 'temporal': args.arm, 'model_shape': [320, 256],
                                        'rectified_shape': [616, 616], 'right_disparity': True,
                                        'capture_clock': 'CLOCK_MONOTONIC', 'exposure_timestamp': True,
                                        'config_sha256': ''})
    try:
        manifest = replay_stream(depth, snaps / 'final', args.replay,
                                 tsdf_args=['--snapshot-every', str(args.every), str(snaps)])
        print(manifest['replay_stdout'])
    finally:
        shutil.rmtree(depth)


def symmetric_eigen(m):
    values, vectors = np.linalg.eigh(m)
    return values, vectors


def voxelize(triangles):
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 1e-8
    area, cross, centre = area[keep], cross[keep], ((v0 + v1 + v2) / 3)[keep]
    normal = cross / (2 * area[:, None])
    keys = np.floor(centre / VOXEL_M).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(unique)
    A = np.zeros(n); S = np.zeros((n, 3)); Q = np.zeros((n, 3, 3)); T = np.zeros((n, 3, 3))
    np.add.at(A, inverse, area)
    np.add.at(S, inverse, area[:, None] * centre)
    np.add.at(Q, inverse, area[:, None, None] * centre[:, :, None] * centre[:, None, :])
    np.add.at(T, inverse, area[:, None, None] * normal[:, :, None] * normal[:, None, :])
    values, vectors = np.linalg.eigh(T)
    voxel_normal = vectors[:, :, 2]
    concentration = values[:, 2] / np.maximum(A, 1e-12)
    return {tuple(k): i for i, k in enumerate(map(tuple, unique))}, unique, A, S, Q, voxel_normal, concentration


def plane_from(a, s, q):
    mean = s / a
    cov = q / a - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]
    return normal, float(normal @ mean), float(np.sqrt(max(values[0], 0))), mean


def detect_snapshot(triangles):
    index, keys, A, S, Q, N, conc = voxelize(triangles)
    cos_angle = np.cos(np.radians(ANGLE_DEG))
    region = np.full(len(keys), -1)
    planes = []
    for seed in np.argsort(-conc):
        if region[seed] != -1 or conc[seed] < SEED_CONCENTRATION:
            continue
        members = [seed]
        region[seed] = len(planes)
        a, s, q = A[seed], S[seed].copy(), Q[seed].copy()
        normal, offset = N[seed], float(N[seed] @ (S[seed] / A[seed]))
        queue = deque([seed])
        added = 0
        while queue:
            current = queue.popleft()
            x, y, z = keys[current]
            for dx, dy, dz in OFFSETS:
                j = index.get((x + dx, y + dy, z + dz))
                if j is None or region[j] != -1:
                    continue
                if abs(N[j] @ normal) < cos_angle:
                    continue
                if abs(normal @ (S[j] / A[j]) - offset) > DIST_M:
                    continue
                region[j] = len(planes)
                members.append(j)
                a, s, q = a + A[j], s + S[j], q + Q[j]
                added += 1
                if added % 8 == 0:
                    fitted, offset, _, _ = plane_from(a, s, q)
                    normal = fitted if fitted @ normal >= 0 else -fitted
                    offset = float(normal @ (s / a))
                queue.append(j)
        normal_fit, offset_fit, rms, mean = plane_from(a, s, q)
        member_normals = N[members]
        normal_tensor = np.einsum('i,ij,ik->jk', A[members], member_normals, member_normals)
        support_normal = np.linalg.eigh(normal_tensor)[1][:, 2]
        if (a < MIN_AREA_M2 or rms > MAX_RMS_M
                or angle(normal_fit, support_normal) > ANGLE_DEG):
            region[members] = -2
            continue
        planes.append({'normal': normal_fit, 'offset': offset_fit, 'rms': rms, 'area': float(a), 'centroid': mean,
                       'voxels': {tuple(keys[m]) for m in members}, 'sums': (a, s, q)})
    return merge_planes(planes)


def merge_planes(planes):
    """Merge touching coplanar regions whose union is still a plane."""
    def dilated(voxels):
        return {(x + dx, y + dy, z + dz) for x, y, z in voxels for dx, dy, dz in OFFSETS} | voxels
    halo = [dilated(p['voxels']) for p in planes]
    alive = [True] * len(planes)
    changed = True
    while changed:
        changed = False
        for i in range(len(planes)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(planes)):
                if not alive[j]:
                    continue
                a, b = planes[i], planes[j]
                if angle(a['normal'], b['normal']) > MATCH_ANGLE_DEG:
                    continue
                # Test the smaller region against the larger one's plane: a small patch's own
                # tilt is poorly determined and swings over a long lever arm.
                small, large = (a, b) if len(a['voxels']) < len(b['voxels']) else (b, a)
                if abs(large['normal'] @ small['centroid'] - large['offset']) > DIST_M:
                    continue
                if halo[i].isdisjoint(b['voxels']):
                    continue
                sa, ss, sq = a['sums'][0] + b['sums'][0], a['sums'][1] + b['sums'][1], a['sums'][2] + b['sums'][2]
                normal, offset, rms, mean = plane_from(sa, ss, sq)
                if rms > MAX_RMS_M:
                    continue
                planes[i] = {'normal': normal, 'offset': offset, 'rms': rms, 'area': float(sa), 'centroid': mean,
                             'voxels': a['voxels'] | b['voxels'], 'sums': (sa, ss, sq)}
                halo[i] = halo[i] | halo[j]
                alive[j] = False
                changed = True
    planes = [p for p, keep in zip(planes, alive) if keep]
    for p in planes:
        p.pop('sums', None)
    return planes


def angle(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b))) / np.linalg.norm(a) / np.linalg.norm(b)))))


def detect(args):
    snaps = args.out / args.arm / 'tsdf-planes'
    files = sorted(snaps.glob('snap-*.f32'), key=lambda p: int(p.stem.split('-')[1]))
    recent, next_id, history = {}, 1, []   # recent: id -> (plane, snapshot index last seen)
    for k, path in enumerate(files):
        frame = int(path.stem.split('-')[1])
        triangles = np.fromfile(path, '<f4').reshape(-1, 3, 3).astype(np.float64)
        planes = detect_snapshot(triangles)
        taken = set()
        for plane in sorted(planes, key=lambda p: -p['area']):
            matches = []
            for old_id, (old, seen) in recent.items():
                if old_id in taken or k - seen > MEMORY_SNAPSHOTS:
                    continue
                if angle(plane['normal'], old['normal']) > MATCH_ANGLE_DEG:
                    continue
                # Offset compared locally (old centroid to new plane): world-origin offsets move
                # by lever arm x tilt (2 degrees at 3 m is 10 cm) without the surface moving.
                if abs(plane['normal'] @ old['centroid'] - plane['offset']) > MATCH_OFFSET_M:
                    continue
                overlap = len(plane['voxels'] & old['voxels']) / min(len(plane['voxels']), len(old['voxels']))
                if overlap >= MATCH_OVERLAP:
                    matches.append(old_id)
            if not matches:
                plane['id'] = next_id
                next_id += 1
            else:
                # Several old planes merged into this one: keep the earliest-born id, so the
                # dominant surface's identity survives merges.
                plane['id'] = min(matches)
                taken.add(plane['id'])
        for plane in planes:
            recent[plane['id']] = (plane, k)
        history.append({'frame': frame, 'planes': [
            {'id': p['id'], 'normal': p['normal'].tolist(), 'offset': p['offset'], 'rms': p['rms'], 'area': p['area'],
             'centroid': p['centroid'].tolist(), 'voxels': sorted([int(c) for c in v] for v in p['voxels'])} for p in planes]})
        print(f'snapshot {frame}: {len(triangles)} triangles, {len(planes)} planes, '
              f'largest {sorted((round(p["area"], 2) for p in planes), reverse=True)[:5]} m2', flush=True)
    write_json(snaps / 'planes.json', {'voxel_m': VOXEL_M, 'angle_deg': ANGLE_DEG, 'dist_m': DIST_M,
                                       'min_area_m2': MIN_AREA_M2, 'max_rms_m': MAX_RMS_M, 'snapshots': history})


def load_history(args):
    data = json.loads((args.out / args.arm / 'tsdf-planes' / 'planes.json').read_text())
    for snap in data['snapshots']:
        for p in snap['planes']:
            p['voxels'] = set(map(tuple, p['voxels']))
            p['normal'] = np.array(p['normal'])
    return data['snapshots']


def plane_at(snapshot, point, normal_hint=None, max_angle=30.0):
    key = tuple(np.floor(np.asarray(point) / VOXEL_M).astype(int))
    near = {key} | {(key[0] + a, key[1] + b, key[2] + c) for a, b, c in OFFSETS}
    best = None
    for p in snapshot['planes']:
        if not (p['voxels'] & near):
            continue
        if normal_hint is not None and angle(p['normal'], normal_hint) > max_angle:
            continue
        if best is None or p['area'] > best['area']:
            best = p
    return best


def score_tags(args):
    from tsdf_variant_test import tag_planes
    frames = capture_frames(args.session)
    snaps = load_history(args)
    tags = tag_planes(args, frames)
    detections = json.loads(args.tags.read_text())['detections']
    mesh = json.loads(args.mesh_tags.read_text())['variants']['baseline']['tags']
    report = {'snapshots': len(snaps), 'tags': {}}
    final_ids = {}
    final = snaps[-1]
    for tag_id, tag in tags.items():
        first = min(d['frame'] for d in detections if d['id'] == tag_id)
        ids = []
        for snap in snaps:
            if snap['frame'] < first:
                continue
            p = plane_at(snap, tag['centre'], tag['normal'])
            ids.append(p['id'] if p else 0)
        labelled = [x for x in ids if x]
        modal, count = Counter(labelled).most_common(1)[0] if labelled else (0, 0)
        entry = {'snapshots': len(ids), 'coverage': len(labelled) / max(len(ids), 1),
                 'identity': count / max(len(labelled), 1), 'plane': modal}
        p = plane_at(final, tag['centre'], tag['normal'])
        if p:
            n = p['normal']
            along = float((p['offset'] - n @ tag['centre']) / (n @ tag['normal']) * 1000)
            entry.update(final_plane=p['id'], area_m2=p['area'], rms_mm=p['rms'] * 1000,
                         normal_error_deg=angle(n, tag['normal']), plane_offset_mm=along)
            ref = mesh.get(str(tag_id))
            if ref:
                entry['mesh_offset_mm'] = ref['offset_mm']
                entry['plane_minus_mesh_mm'] = along - ref['offset_mm']
        entry['checks'] = {'coverage': entry['coverage'] >= 0.8, 'identity': entry['identity'] >= 0.95,
                           'normal': 'normal_error_deg' in entry and entry['normal_error_deg'] <= 5,
                           'mesh_consistency': 'plane_minus_mesh_mm' in entry and abs(entry['plane_minus_mesh_mm']) <= 15}
        final_ids[tag_id] = entry.get('final_plane', 0)
        report['tags'][str(tag_id)] = entry
    structure = {'final_ids': final_ids, 'pnp_wall_desk_deg': angle(tags[1]['normal'], tags[3]['normal'])}
    ok = final_ids.get(1) and final_ids.get(1) == final_ids.get(2) and final_ids.get(3) and final_ids[3] != final_ids[1]
    if ok:
        planes = {p['id']: p for p in final['planes']}
        structure['plane_wall_desk_deg'] = angle(planes[final_ids[1]]['normal'], planes[final_ids[3]]['normal'])
        ok = abs(structure['plane_wall_desk_deg'] - structure['pnp_wall_desk_deg']) <= 3
    structure['check'] = bool(ok)
    if final_ids.get(1) and final_ids.get(1) == final_ids.get(2):
        plane = {p['id']: p for p in final['planes']}[final_ids[1]]
        s1 = float(plane['normal'] @ tags[1]['centre'] - plane['offset'])
        s2 = float(plane['normal'] @ tags[2]['centre'] - plane['offset'])
        structure['relative_tag_offset_mm'] = (s1 - s2) * 1000
        structure['relative_check'] = abs(s1 - s2) * 1000 <= 15
    else:
        structure['relative_check'] = False
    report['structure'] = structure
    report['planes_final'] = len(final['planes'])
    report['pass'] = bool(ok and all(all(t['checks'].values()) for t in report['tags'].values()))
    # (d') replaces (d) for planes not positioned by the RGB mesh.
    report['pass_relative'] = bool(ok and structure['relative_check'] and all(
        all(v for k, v in t['checks'].items() if k != 'mesh_consistency') for t in report['tags'].values()))
    write_json(args.out / args.arm / 'tsdf-planes' / 'score_tags.json', report)
    print(json.dumps(report, indent=1, default=float))


def score_scene(args):
    from temporal_event_test import load_left
    from temporal_tag_check import sphere_points
    from tsdf_variant_test import transient_cells
    frames = capture_frames(args.session)
    segments, _ = segments_from_marks(args.session, frames, args.segments)
    snaps = load_history(args)
    # Table height: world y of the still segment's image-centre points.
    first, last = segments['still']
    names = [frames[i]['name'] for i in range(first, last + 1)]
    heights = []
    for k, d in enumerate(load_left(args.out, args.arm, names)):
        R, t = frames[first + k]['pose']
        P = sphere_points(d)[150:170, 118:138].reshape(-1, 3)
        P = P[np.isfinite(P).all(1)]
        heights += list((P @ np.asarray(R).T + t)[:, 1])
    table_y = float(np.median(heights))
    table_point = None
    final = snaps[-1]
    horizontal = [p for p in final['planes'] if angle(p['normal'], UP) <= 5]
    table = min(horizontal, key=lambda p: abs(p['centroid'][1] - table_y), default=None)
    report = {'table_height_m': table_y, 'planes_final': len(final['planes']),
              'horizontal': [{'id': p['id'], 'height_m': p['centroid'][1], 'area_m2': p['area']} for p in horizontal]}
    table_ok = table is not None and abs(table['centroid'][1] - table_y) < 0.05
    floors = [p for p in horizontal if table_ok and 0.5 <= table['centroid'][1] - p['centroid'][1] <= 1.0]
    report['parallel'] = {'table': table['id'] if table_ok else None, 'floors': [p['id'] for p in floors],
                          'check': bool(table_ok and all(p['id'] != table['id'] for p in floors))}
    if table_ok:
        table_point = np.array(table['centroid'])
    cells = transient_cells(argparse.Namespace(out=args.out, arm=args.arm), frames, segments)
    bad = []
    for p in final['planes']:
        centres = (np.array(sorted(p['voxels']), float) + 0.5) * VOXEL_M
        inside = np.mean([tuple(c) in cells for c in np.floor(centres / 0.03).astype(np.int64)])
        if inside >= 0.5:
            bad.append({'id': p['id'], 'fraction': float(inside), 'area_m2': p['area']})
    report['transient_planes'] = {'planes': bad, 'check': not bad}
    end = segments['object'][1] + 1
    ids = []
    if table_point is not None:
        for snap in snaps:
            if snap['frame'] <= end:
                p = plane_at(snap, table_point, UP, 10)
                ids.append(p['id'] if p else 0)
    labelled = [x for x in ids if x]
    share = Counter(labelled).most_common(1)[0][1] / len(labelled) if labelled else 0
    report['identity'] = {'snapshots': len(ids), 'labelled': len(labelled), 'modal_share': share,
                          'check': bool(labelled) and share >= 0.95}
    report['pass'] = bool(report['parallel']['check'] and report['transient_planes']['check'] and report['identity']['check'])
    write_json(args.out / args.arm / 'tsdf-planes' / 'score_scene.json', report)
    print(json.dumps(report, indent=1, default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshots')
    s.add_argument('--every', type=int, default=25)
    s.add_argument('--converter', type=Path, default=PATHS.repo / 'depth-runtime/bin/model_batch_to_v2')
    s.add_argument('--replay', type=Path, default=PATHS.repo / 'depth-runtime/bin/replay_room_tsdf')
    d = sub.add_parser('detect')
    t = sub.add_parser('score-tags')
    t.add_argument('--tags', type=Path, required=True)
    t.add_argument('--mesh-tags', type=Path, required=True)
    e = sub.add_parser('score-scene')
    e.add_argument('--segments')
    for each in (s, d, t, e):
        each.add_argument('--session', type=Path, required=True)
        each.add_argument('--out', type=Path, required=True)
        each.add_argument('--arm', default='off')
    args = parser.parse_args()
    args.session, args.out = args.session.resolve(), args.out.resolve()
    {'snapshots': snapshots, 'detect': detect, 'score-tags': score_tags, 'score-scene': score_scene}[args.command](args)


if __name__ == '__main__':
    main()
