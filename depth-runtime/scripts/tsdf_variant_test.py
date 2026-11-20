#!/usr/bin/env python3
"""E07 real-capture comparison: TSDF integration variants on frozen model outputs.

Offline development tool. The model output (an arm in the temporal_event_test.py layout,
normally `off`, recurrence off, no filter) is frozen; only TsdfIntegrateOptions change,
through replay_room_tsdf flags. Voxel size, truncation and max weight stay at the
defaults (3 cm, 12 cm, 32).

  mesh         convert the arm's depth once and replay every variant -> ARM/mesh-VARIANT/
  score-events transient removal, near-range completeness and false surface (event capture)
  score-tags   tag-plane offset and normal of each mesh (tagged capture; tags.json from
               temporal_tag_check.py detect)

Variants, declared 2026-11-08 before the first real-capture run, each added alone and
then combined (E07: "add each component separately before evaluating the combination"):
  baseline   live defaults
  robust     w *= (1 - min(|sdf|/mu, 1))^2
  cap1       per-frame sample weight capped at 1 (one observation at 1 m)
  carve3     an established surface needs 3 full free-space updates before it is carved
  combined   robust + cap1 + carve3
Normal-incidence weighting is skipped: it needs per-pixel normals, and E04's normals were
not accepted (E06 must not use them, and neither does E07).

Acceptance, per variant against baseline, declared with the variants:
  transient  mesh vertices in transient cells (3 cm cells occupied by hand/cloth points
             from the event frames, minus cells within one cell of the still-segment
             scene): must not increase; a component is useful only if it lowers this by
             at least 10%, or lowers false surface by at least 10%, without failing the rest
  holes      baseline vertices within 2 m of the head path that have no variant vertex in
             the surrounding cells, excluding transient cells: at most 2% of the baseline's
             near-range vertices (the gate G completeness limit)
  false      vertices below the floor (y < -0.05 m) at most 1.05 x baseline; extra-surface
             clusters within 1.5 m of the head path need a visual check
  tags       per tag, |mesh-plane offset| increase at most 15 mm and normal increase at
             most 5 degrees (gate G, <= 1.25 m bin; the desk tags are about 1 m away)
A no-benefit variant is rejected even when it passes (gate G rule).

E08 set (--set e08), declared 2026-11-08 before its first run: voxel {2, 3, 4} cm x
truncation half-width {2, 4, 6} voxels (v3-t4 is the live baseline), plus far3, the
baseline with max depth 3.0 m instead of 4.0 m (carried from E07's false far surface).
Finer voxels make more vertices per surface, so E08 judges transient residue and false
surface by occupied 3 cm cells, not vertex counts; holes stay relative to the baseline's
vertices. A configuration is adopted only if it passes the E07 checks on those cell counts
and the tag check, keeps peak voxel memory within the live budget (300,000 leaves, about
116 MB at 388 bytes) with no eviction and the integrate p95 at most 2 x baseline, and
improves transient cells, false-surface cells or the mean tag normal error by at least 10%.
Otherwise the baseline is kept; the smallest voxel is not presumed best.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import (EXPOSURE_OFFSET_MS, FLOOR_Y_M, H, MIN_AREA, PATHS, W, capture_frames,  # noqa: E402
                                 components, extra_surface, frame_paths, load_left, plausible,
                                 read_ply_vertices, relative_motion, segments_from_marks, warped_previous,
                                 write_json)
from temporal_tag_check import sphere_points  # noqa: E402

VARIANT_SETS = {
    'e07': {'baseline': [], 'robust': ['--robust'], 'cap1': ['--frame-cap', '1'],
            'carve3': ['--carve-after', '3'],
            'combined': ['--robust', '--frame-cap', '1', '--carve-after', '3']},
    'e08': {'baseline': [], 'far3': ['--max-depth', '3.0'],
            **{f'v{v}-t{t}': ['--voxel', f'{v / 100:g}', '--truncation', f'{v * t / 100:g}']
               for v in (2, 3, 4) for t in (2, 4, 6) if (v, t) != (3, 4)}},
}
VARIANTS = {name: flags for group in VARIANT_SETS.values() for name, flags in group.items()}
LEAF_BYTES, LEAF_BUDGET = 388, 300_000
CELL_M = 0.03
NEAR_PATH_M, NEAR_HEAD_M = 2.0, 1.5
HOLE_LIMIT, FALSE_LIMIT, BENEFIT = 0.02, 1.05, 0.10
TAG_RADIUS_M, TAG_SLAB_M = 0.05, 0.10
TAG_OFFSET_LIMIT_MM, TAG_NORMAL_LIMIT_DEG = 15.0, 5.0
BIG_PX = 4.0


def mesh(args):
    from room_dataset import load_poses, write_depth_metadata
    from room_reconstruction_replay import replay_stream
    paths = frame_paths(args.session / 'rgb')
    depth = args.out / args.arm / 'depth'
    depth.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.converter), str(args.session / 'rgb'), str(args.out / args.arm / 'model'),
                    str(depth), str(len(paths))], check=True)
    extrinsic = json.loads((args.session / 'rgb/rectified-camera-in-head.json').read_text())
    write_depth_metadata(paths, depth, load_poses(args.session / 'head_pose.csv'), extrinsic, EXPOSURE_OFFSET_MS)
    write_json(depth / 'capture.json',
               {'version': 2, 'model': 'primary', 'backend': 'prepared-qnn', 'projection': 'spherical',
                'temporal': args.arm, 'model_shape': [320, 256], 'rectified_shape': [616, 616],
                'right_disparity': True, 'capture_clock': 'CLOCK_MONOTONIC',
                'exposure_timestamp': True, 'config_sha256': ''})
    try:
        for name in args.variants:
            manifest = replay_stream(depth, args.out / args.arm / f'mesh-{name}', args.replay,
                                     tsdf_args=VARIANTS[name])
            print(name, manifest['replay_stdout'], flush=True)
    finally:
        shutil.rmtree(depth)


def head_path(frames):
    return np.array([f['pose'][1] for f in frames if f['pose'] is not None])


def near_path(vertices, path, radius):
    """Boolean mask of vertices within radius of any head position (subsampled path)."""
    path = path[::5]
    mask = np.zeros(len(vertices), bool)
    for start in range(0, len(vertices), 20000):
        block = vertices[start:start + 20000]
        distance = np.min(np.linalg.norm(block[:, None, :] - path[None, :, :], axis=2), axis=1)
        mask[start:start + 20000] = distance < radius
    return mask


def cells(points):
    return set(map(tuple, np.unique(np.floor(points / CELL_M).astype(np.int64), axis=0)))


def dilate_cells(cell_set):
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    return {(x + a, y + b, z + c) for x, y, z in cell_set for a, b, c in offsets}


def in_cells(vertices, cell_set):
    keys = np.floor(vertices / CELL_M).astype(np.int64)
    return np.array([tuple(k) in cell_set for k in keys], bool)


def world_points(d, pose, mask):
    P = sphere_points(d)[mask]
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0.2) & (P[:, 2] < 5.0)]
    R, t = pose
    return P @ R.T + t


def transient_cells(args, frames, segments):
    """Cells of hand/cloth points (large nearer changes) minus the dilated still scene."""
    from model_temporal import ModelTemporalState
    from temporal_event_test import SPHERICAL_DEG
    names = [f['name'] for f in frames]
    off = load_left(args.out, args.arm, names)
    probe = ModelTemporalState(spherical_deg=SPHERICAL_DEG, pyramid=0)
    foreground = []
    for segment in ('hand', 'object'):
        first, last = segments[segment]
        for i in range(first + 1, last + 1):
            if frames[i]['pose'] is None or frames[i - 1]['pose'] is None:
                continue
            previous = warped_previous(probe, off[i - 1], relative_motion(frames[i - 1]['pose'], frames[i]['pose']))
            valid = (previous > 0) & plausible(previous) & plausible(off[i])
            change = np.where(valid, off[i] - previous, 0)
            mask = np.zeros((H, W), bool)
            for members in components(change > BIG_PX):
                if len(members) >= MIN_AREA:
                    mask.ravel()[members] = True
            if mask.any():
                foreground.append(world_points(off[i], frames[i]['pose'], mask))
    first, last = segments['still']
    scene = [world_points(off[i], frames[i]['pose'], plausible(off[i]))
             for i in range(first, last + 1, 4) if frames[i]['pose'] is not None]
    fg = cells(np.concatenate(foreground)) if foreground else set()
    return fg - dilate_cells(cells(np.concatenate(scene)))


def score_events(args):
    frames = capture_frames(args.session)
    segments, _ = segments_from_marks(args.session, frames, args.segments)
    path = head_path(frames)  # every frame: the meshes integrate the whole capture
    transient = transient_cells(args, frames, segments)
    meshes = {name: read_ply_vertices(args.out / args.arm / f'mesh-{name}' / 'mesh.ply') for name in args.variants}
    stdout = {name: json.loads((args.out / args.arm / f'mesh-{name}' / 'manifest.json').read_text())['replay_stdout']
              for name in args.variants}
    base = meshes['baseline']
    base_near = near_path(base, path, NEAR_PATH_M)
    base_transient = in_cells(base, transient)
    base_residue = int(base_transient.sum())
    base_below = int((base[:, 1] < FLOOR_Y_M).sum())
    report = {'transient_cells': len(transient), 'baseline_near_vertices': int(base_near.sum()), 'variants': {}}
    base_cells = None
    for name, vertices in meshes.items():
        residue = int(in_cells(vertices, transient).sum())
        below = int((vertices[:, 1] < FLOOR_Y_M).sum())
        replay = dict(item.split('=') for item in stdout[name].split())
        entry = {'vertices': len(vertices), 'replay': stdout[name], 'transient_residue': residue,
                 'below_floor': below, 'below_floor_ratio': below / max(base_below, 1),
                 'transient_cells_hit': len(cells(vertices) & transient) if len(vertices) else 0,
                 'below_floor_cells': len(cells(vertices[vertices[:, 1] < FLOOR_Y_M])) if (vertices[:, 1] < FLOOR_Y_M).any() else 0,
                 'peak_leaves': int(replay['peak_leaves']), 'evicted': int(replay['evicted']),
                 'peak_voxel_mb': int(replay['peak_leaves']) * LEAF_BYTES / 1e6,
                 'integrate_ms_p95': float(replay.get('integrate_ms_p95', 'nan')),
                 'triangle_cap_hit': int(replay['triangles']) >= 480000}
        if name != 'baseline':
            # Baseline vertices with no variant vertex nearby: holes, unless they were transient.
            variant_cells = dilate_cells(cells(vertices))
            missing = ~in_cells(base, variant_cells)
            holes = missing & base_near & ~base_transient
            _, clusters = extra_surface(vertices, base)
            near_clusters = int(sum(near_path(vertices[c].mean(0, keepdims=True), path, NEAR_HEAD_M)[0]
                                    for c in clusters))
            residue_change = (residue - base_residue) / max(base_residue, 1)
            false_change = (below - base_below) / max(base_below, 1)
            checks = {'transient_not_worse': bool(residue <= base_residue),
                      'holes': bool(holes.sum() <= HOLE_LIMIT * base_near.sum()),
                      'false_surface': bool(below <= FALSE_LIMIT * base_below)}
            entry.update(near_holes=int(holes.sum()), near_hole_fraction=float(holes.sum() / max(base_near.sum(), 1)),
                         extra_clusters=len(clusters), extra_clusters_near_head=near_clusters,
                         transient_change=residue_change, below_floor_change=false_change,
                         benefit=bool(residue_change <= -BENEFIT or false_change <= -BENEFIT), checks=checks)
        if name == 'baseline':
            base_cells = entry
        elif args.set == 'e08':
            cell_checks = {
                'transient_cells_not_worse': entry['transient_cells_hit'] <= base_cells['transient_cells_hit'],
                'holes': entry['checks']['holes'],
                'false_cells': entry['below_floor_cells'] <= FALSE_LIMIT * base_cells['below_floor_cells'],
                'memory': entry['peak_leaves'] <= LEAF_BUDGET and entry['evicted'] == 0,
                'runtime': entry['integrate_ms_p95'] <= 2 * base_cells['integrate_ms_p95'],
                'mesh_not_capped': not entry['triangle_cap_hit']}
            entry['e08_checks'] = {k: bool(v) for k, v in cell_checks.items()}
            entry['e08_cell_benefit'] = bool(
                entry['transient_cells_hit'] <= (1 - BENEFIT) * base_cells['transient_cells_hit'] or
                entry['below_floor_cells'] <= (1 - BENEFIT) * base_cells['below_floor_cells'])
        report['variants'][name] = entry
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()
                                if k not in ('replay',)}), flush=True)
    write_json(args.out / args.arm / 'tsdf_events.json', report)


def tag_planes(args, frames):
    detections = json.loads(args.tags.read_text())['detections']
    by_name = {f['name']: f for f in frames}
    per_tag = {}
    for tag in detections:
        frame = by_name.get(f'frame-{tag["frame"]}')
        if frame is None or frame['pose'] is None:
            continue
        R, t = frame['pose']
        position = np.array(tag['position'])
        normal = np.array(tag['rotation'])[:, 2]
        if normal @ position > 0:
            normal = -normal
        per_tag.setdefault(tag['id'], []).append((R @ position + t, R @ normal))
    planes = {}
    for tag_id, rows in per_tag.items():
        centres = np.array([r[0] for r in rows]); normals = np.array([r[1] for r in rows])
        normal = normals.mean(0); normal /= np.linalg.norm(normal)
        planes[tag_id] = {'centre': np.median(centres, 0), 'normal': normal, 'n': len(rows),
                          'centre_spread_mm': float(np.median(np.linalg.norm(centres - np.median(centres, 0), axis=1)) * 1000)}
    return planes


def mesh_plane(vertices, plane):
    offset = (vertices - plane['centre']) @ plane['normal']
    radial = np.linalg.norm((vertices - plane['centre']) - offset[:, None] * plane['normal'], axis=1)
    selected = (radial < TAG_RADIUS_M) & (np.abs(offset) < TAG_SLAB_M)
    if selected.sum() < 20:
        return None
    Q = vertices[selected]
    _, _, vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    angle = float(np.degrees(np.arccos(min(1.0, abs(vt[2] @ plane['normal'])))))
    return {'n': int(selected.sum()), 'offset_mm': float(np.median(offset[selected]) * 1000), 'normal_deg': angle}


def score_tags(args):
    frames = capture_frames(args.session)
    planes = tag_planes(args, frames)
    meshes = {name: read_ply_vertices(args.out / args.arm / f'mesh-{name}' / 'mesh.ply') for name in args.variants}
    report = {'tags': {str(k): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in p.items()}
                       for k, p in planes.items()}, 'variants': {}}
    base = {k: mesh_plane(meshes['baseline'], p) for k, p in planes.items()}
    for name, vertices in meshes.items():
        rows = {}
        ok = True
        for tag_id, plane in planes.items():
            row = mesh_plane(vertices, plane)
            if row and base[tag_id] and name != 'baseline':
                row['offset_increase_mm'] = abs(row['offset_mm']) - abs(base[tag_id]['offset_mm'])
                row['normal_increase_deg'] = row['normal_deg'] - base[tag_id]['normal_deg']
                ok &= row['offset_increase_mm'] <= TAG_OFFSET_LIMIT_MM and row['normal_increase_deg'] <= TAG_NORMAL_LIMIT_DEG
            elif name != 'baseline':
                ok = False
            rows[str(tag_id)] = row
        normals = [r['normal_deg'] for r in rows.values() if r]
        report['variants'][name] = {'tags': rows, 'pass': bool(ok) if name != 'baseline' else None,
                                    'mean_normal_deg': float(np.mean(normals)) if normals else None}
        print(name, json.dumps({k: (None if v is None else {kk: round(vv, 2) if isinstance(vv, float) else vv
                                                             for kk, vv in v.items()}) for k, v in rows.items()}),
              'pass' if ok else 'FAIL', flush=True)
    write_json(args.out / args.arm / 'tsdf_tags.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('mesh')
    m.add_argument('--converter', type=Path, default=PATHS.repo / 'depth-runtime/bin/model_batch_to_v2')
    m.add_argument('--replay', type=Path, default=PATHS.repo / 'depth-runtime/bin/replay_room_tsdf')
    e = sub.add_parser('score-events')
    e.add_argument('--segments', help='override marks: still:F-L,hand:F-L,... (1-based frames)')
    t = sub.add_parser('score-tags')
    t.add_argument('--tags', type=Path, required=True)
    for each in (m, e, t):
        each.add_argument('--session', type=Path, required=True)
        each.add_argument('--out', type=Path, required=True)
        each.add_argument('--arm', default='off')
        each.add_argument('--set', choices=list(VARIANT_SETS), default='e07')
        each.add_argument('--variants', nargs='+', choices=list(VARIANTS))
    args = parser.parse_args()
    args.variants = args.variants or list(VARIANT_SETS[args.set])
    args.session, args.out = args.session.resolve(), args.out.resolve()
    if args.command != 'mesh' and 'baseline' not in args.variants:
        parser.error('scoring needs the baseline variant')
    {'mesh': mesh, 'score-events': score_events, 'score-tags': score_tags}[args.command](args)


if __name__ == '__main__':
    main()
