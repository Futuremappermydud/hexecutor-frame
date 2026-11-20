#!/usr/bin/env python3
"""E09 walls from IR dots: plane shape from dot stereo, plane position from textured RGB.

Offline development only. Uses the existing per-session IR yaw (no yaw estimation): a
constant disparity offset from a yaw error keeps a plane planar and moves its offset (and
tilts oblique planes slightly), so each plane's offset is left free and solved against
textured RGB surface on the same footprint.

  points  dot points (world) for every lit, sham-validated IR pair -> OUT/irdots/points.f32
          (x, y, z, exposure_ns as float64 rows) and pairs.json
  planes  cumulative dot planes at each RGB snapshot frame of OUT/off/tsdf-planes, with
          tsdf_planes.py region growing and persistent ids; offsets shifted to the gated
          arm's final mesh -> OUT/irdots/tsdf-planes/planes.json, scored by
          `tsdf_planes.py score-tags --arm irdots` (the E09 gate, unchanged)

Declared before the first dot run (2026-11-08): dot voxels need at least 8 dots and
planarity (smallest / middle eigenvalue) at most 0.15 to seed; growth uses tsdf_planes.py's
15 degrees and 2 cm; the offset shift is the median signed distance of gated-mesh vertices
within one voxel of the plane's footprint and 10 cm of the plane, used only with at least
30 such vertices (otherwise the plane keeps its dot offset and is flagged).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import write_json  # noqa: E402
import tsdf_planes as T  # noqa: E402

MIN_DOTS, MAX_PLANARITY = 8, 0.15
SHIFT_MIN_VERTICES, SHIFT_SLAB_M = 30, 0.10


def points(args):
    os.environ['IR_YAW_DEG'] = str(args.ir_yaw_deg)
    os.environ.setdefault('IR_CALIB', '/persist/xrservice.json')
    from ir_dot_lib import (H, W, MAX_D, Rectifier, load_rgb_chain, rgb_disparity_to_ir, search, unwrapped_offset,
                            zncc_volume)
    from room_dataset import (ir_stereo_input, load_poses, nearest_pose, pose_from_row, select_lit_pairs)
    session = args.session
    rect = Rectifier()
    chain = load_rgb_chain(session / 'rgb/calibration-history-at-init.json')
    R_rl, t_rl, R_fr, t_fr, (Rrgb, _) = chain
    observations = [json.loads(l) for l in (session / 'ir/pairs.jsonl').read_text().splitlines()]
    cycles = [json.loads(l) for l in (session / 'pulses.jsonl').read_text().splitlines()]
    selected, selection = select_lit_pairs(observations, cycles)
    poses = load_poses(session / 'head_pose.csv')
    extrinsic = json.loads((session / 'rgb/rectified-camera-in-head.json').read_text())
    frames = sorted(session.glob('rgb/frame-*.json'), key=lambda p: int(p.stem.split('-')[1]))
    rgb_times = [(int(json.loads(p.read_text())['left_end_of_readout_ns']), p.stem) for p in frames]
    model = args.out / 'off' / 'model'
    dd = np.arange(MAX_D + 1)[:, None, None]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rows, meta = [], []
    for entry in selected:
        when = entry['left_monotonic_ns'] - int(entry['lead_ms'] * 1e6)
        rgb_time, name = min(rgb_times, key=lambda item: abs(item[0] - when))
        pose_row, _ = nearest_pose(poses, when)
        if abs(rgb_time - when) > 250_000_000 or pose_row is None:
            continue
        disp = np.fromfile(model / f'{name}-spherical-disparity.f32', '<f4').reshape(H, W)
        unc = np.fromfile(model / f'{name}-spherical-uncertainty.u8', np.uint8).reshape(H, W)
        left, right, _ = ir_stereo_input(session, entry, 'average')
        L, R, okl, okr = rect.rectify(left, right)
        prior = rgb_disparity_to_ir(disp, unc, rect, chain)
        vol = zncc_volume(L, R, okl, okr)
        anywhere = okl & np.isfinite(prior)
        ref, kept = search(vol, dd, prior, anywhere)
        prior = prior + unwrapped_offset(ref - prior, kept)
        ref, kept = search(vol, dd, prior, anywhere)
        Q = rect.triangulate(u[kept], v[kept], ref[kept])
        P = rect.Rrect @ Q + rect.cL[:, None]
        Pr = R_fr.T @ (P - t_fr[:, None])
        Pl = R_rl.T @ (Pr - t_rl[:, None])
        Prgb = (Rrgb.T @ Pl).T
        ok = (Prgb[:, 2] > 0.25) & (Prgb[:, 2] < 4.0)
        rotation, translation = pose_from_row(pose_row, extrinsic)
        world = Prgb[ok] @ np.asarray(rotation, float).reshape(3, 3).T + np.asarray(translation, float)
        rows.append(np.column_stack([world, np.full(len(world), float(when))]))
        meta.append({'pair': entry['pair'], 'exposure_ns': when, 'dots': int(len(world)), 'rgb_frame': name})
    out = args.out / 'irdots'
    out.mkdir(parents=True, exist_ok=True)
    np.concatenate(rows).astype('<f8').tofile(out / 'points.f64')
    write_json(out / 'pairs.json', {'ir_yaw_deg': args.ir_yaw_deg, 'selection': selection, 'pairs': meta})
    print(f'{len(meta)} lit pairs, {sum(m["dots"] for m in meta)} dots; selection {selection}')


def voxelize_points(P):
    keys = np.floor(P / T.VOXEL_M).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(unique)
    A = np.zeros(n); S = np.zeros((n, 3)); Q = np.zeros((n, 3, 3))
    np.add.at(A, inverse, 1.0)
    np.add.at(S, inverse, P)
    np.add.at(Q, inverse, P[:, :, None] * P[:, None, :])
    mean = S / A[:, None]
    cov = Q / A[:, None, None] - mean[:, :, None] * mean[:, None, :]
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, :, 0]
    planarity = values[:, 0] / np.maximum(values[:, 1], 1e-12)
    conc = np.where((A >= MIN_DOTS) & (planarity <= MAX_PLANARITY), 1.0, 0.0)
    return {tuple(k): i for i, k in enumerate(map(tuple, unique))}, unique, A, S, Q, normal, conc


def planes(args):
    snaps_rgb = json.loads((args.out / 'off' / 'tsdf-planes' / 'planes.json').read_text())['snapshots']
    frame_times = {}
    for p in sorted(args.session.glob('rgb/frame-*.json'), key=lambda p: int(p.stem.split('-')[1])):
        frame_times[int(p.stem.split('-')[1])] = int(json.loads(p.read_text())['left_end_of_readout_ns'])
    data = np.fromfile(args.out / 'irdots' / 'points.f64', '<f8').reshape(-1, 4)
    gated = np.fromfile(sorted((args.out / 'gated' / 'tsdf-planes').glob('snap-*.f32'),
                               key=lambda p: int(p.stem.split('-')[1]))[-1], '<f4').reshape(-1, 3).astype(float)
    original = T.voxelize
    T.voxelize = lambda P: voxelize_points(P)   # detect_snapshot on points instead of triangles
    recent, next_id, history = {}, 1, []
    try:
        for k, snap in enumerate(snaps_rgb):
            limit = frame_times[snap['frame']]
            P = data[data[:, 3] <= limit, :3]
            found = T.detect_snapshot(P) if len(P) > 1000 else []
            # detect_snapshot's area threshold counts dots here; require 0.04 m2 of voxels instead.
            found = [f for f in found if len(f['voxels']) * T.VOXEL_M ** 2 >= T.MIN_AREA_M2]
            taken = set()
            for plane in sorted(found, key=lambda p: -p['area']):
                matches = [old_id for old_id, (old, seen) in recent.items()
                           if old_id not in taken and k - seen <= T.MEMORY_SNAPSHOTS
                           and T.angle(plane['normal'], old['normal']) <= T.MATCH_ANGLE_DEG
                           and abs(plane['normal'] @ old['centroid'] - plane['offset']) <= T.MATCH_OFFSET_M
                           and len(plane['voxels'] & old['voxels']) / min(len(plane['voxels']), len(old['voxels']))
                           >= T.MATCH_OVERLAP]
                plane['id'] = min(matches) if matches else next_id
                if not matches:
                    next_id += 1
                taken.add(plane['id'])
            for plane in found:
                recent[plane['id']] = (plane, k)
            history.append({'frame': snap['frame'], 'dots': int(len(P)), 'planes': found})
            print(f'snapshot {snap["frame"]}: {len(P)} dots, {len(found)} planes', flush=True)
    finally:
        T.voxelize = original
    # Position from texture: shift each final plane to the final gated mesh (texture-supported
    # surface) on its footprint. Earlier snapshots only feed coverage and identity.
    gated_keys = [tuple(k) for k in np.floor(gated / T.VOXEL_M).astype(np.int64)]
    out_snaps = []
    for index, snap in enumerate(history):
        rows = []
        final_snapshot = index == len(history) - 1
        for p in snap['planes']:
            if not final_snapshot:
                rows.append({'id': p['id'], 'normal': p['normal'].tolist(), 'offset': p['offset'], 'rms': p['rms'],
                             'area': p['area'], 'centroid': p['centroid'].tolist(),
                             'voxels': sorted([int(c) for c in v] for v in p['voxels'])})
                continue
            near = set()
            for key in p['voxels']:
                near.add(key)
                near.update((key[0] + a, key[1] + b, key[2] + c) for a, b, c in T.OFFSETS)
            mask = np.array([k in near for k in gated_keys], bool) if len(gated) else np.zeros(0, bool)
            distance = gated[mask] @ p['normal'] - p['offset'] if mask.any() else np.zeros(0)
            distance = distance[np.abs(distance) < SHIFT_SLAB_M]
            shift = float(np.median(distance)) if len(distance) >= SHIFT_MIN_VERTICES else None
            offset = p['offset'] + (shift or 0.0)
            rows.append({'id': p['id'], 'normal': p['normal'].tolist(), 'offset': offset, 'dot_offset': p['offset'],
                         'shift_m': shift, 'shift_vertices': int(len(distance)), 'rms': p['rms'], 'area': p['area'],
                         'centroid': (p['centroid'] + (shift or 0.0) * p['normal']).tolist(),
                         'voxels': sorted([int(c) for c in v] for v in p['voxels'])})
        out_snaps.append({'frame': snap['frame'], 'dots': snap['dots'], 'planes': rows})
    target = args.out / 'irdots' / 'tsdf-planes'
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'planes.json', {'source': 'ir dots', 'min_dots': MIN_DOTS, 'max_planarity': MAX_PLANARITY,
                                        'snapshots': out_snaps})
    final = out_snaps[-1]['planes']
    print('final planes', len(final), 'shifts (m)', [None if r['shift_m'] is None else round(r['shift_m'], 3)
                                                     for r in sorted(final, key=lambda r: -r['area'])[:8]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('points')
    p.add_argument('--ir-yaw-deg', type=float, required=True, help='existing per-session value (E05: desk-tags -0.35)')
    q = sub.add_parser('planes')
    for each in (p, q):
        each.add_argument('--session', type=Path, required=True)
        each.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.session, args.out = args.session.resolve(), args.out.resolve()
    {'points': points, 'planes': planes}[args.command](args)


if __name__ == '__main__':
    main()
