#!/usr/bin/env python3
"""E06 external pose-aware temporal filter: offline CPU reference over saved model outputs.

Offline development only (not a live path). Reads an arm's model outputs
(OUT/SOURCE/model/, the model_offline_batch.py layout) and the capture's poses,
and writes a filtered arm in the same layout, so temporal_event_test.py mesh/score
apply unchanged. Only the left disparity plane is filtered; the right plane and
both uncertainty planes are copied, since depth conversion uses left disparity and
left uncertainty only.

State, per left-image pixel: fused disparity m, support count w and age (frames since
an observation was available). Single surface hypothesis with explicit conflicts:

  warp     each state pixel is triangulated with m, moved by the relative head pose,
           reprojected and splatted to the nearest pixel; on collisions the larger
           disparity (nearer surface) wins, which resolves visibility. Pixels nothing
           lands on have no history.
  edge     an observation whose largest 4-neighbour disparity step exceeds EDGE_PX
           is an edge: no averaging across depth discontinuities.
  agree    history, a plausible non-edge observation and |obs - m| <= gate:
           m <- (w m + obs) / (w + 1), w <- min(w + 1, w_max); output m.
  conflict history and |obs - m| > gate: output obs, state <- obs with w = 1.
           A contradicted history is never output, so a large change shows at once.
  missing  implausible observation: output obs; the warped history is kept for at
           most EXPIRE frames.
  reset    invalid pose, a gap over 250 ms or over 30 degrees per step (the worker's
           invalidation rules): state <- obs.

Declared tuning grid (frozen 2026-11-08 before the first filtered run): gate in
{0.75, 1.0, 1.5} px x w_max in {4, 8}; EDGE_PX 2.0 and EXPIRE 3 fixed. Selection rule:
among configurations that pass every frozen E06 check on the tuning capture, take the
lowest mean of the per-segment consecutive-frame jitter medians; ties go to the smaller
w_max, then the smaller gate.

Range-aware grid (declared 2026-11-08 after the first grid's mesh added false far-range
surface, before any range-aware run): gate = min(1.0 px, r x history disparity), a
constant relative depth tolerance, with r in {0.05, 0.08, 0.12}, and fusion only where
the history disparity is at least d_min in {0, 3.4} px (3.4 px is about 3 m); w_max 8.
A configuration must also meet a mesh criterion on the tuning capture: vertices below the
floor (world y < -0.05 m) at most 1.05 x the off arm's, and no extra-surface cluster
within 1.5 m of the head. Selection among passing configurations is unchanged (lowest
mean jitter median; ties go to the smaller r, then d_min 0).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import (H, W, RESET_GAP_NS, RESET_ROTATION_DEG, SPHERICAL_DEG,  # noqa: E402
                                 capture_frames, plausible, relative_motion, rotation_deg, write_json)
from model_spherical import SphericalModel  # noqa: E402

EDGE_PX = 2.0
EXPIRE = 3
GRID = [(gate, w_max) for gate in (0.75, 1.0, 1.5) for w_max in (4, 8)]
GATE_CAP_PX = 1.0
RANGE_GRID = [(r, d_min) for r in (0.05, 0.08, 0.12) for d_min in (0.0, 3.4)]
RANGE_W_MAX = 8
V, U = np.mgrid[0:H, 0:W]


def arm_name(source, gate, w_max, relative=None, d_min=0.0):
    if relative is not None:
        return f'filter-{source}-r{relative:g}-d{d_min:g}-w{w_max}'
    return f'filter-{source}-g{gate:g}-w{w_max}'


def edges(d):
    step = np.zeros_like(d)
    step[1:] = np.maximum(step[1:], np.abs(d[1:] - d[:-1]))
    step[:-1] = np.maximum(step[:-1], np.abs(d[1:] - d[:-1]))
    step[:, 1:] = np.maximum(step[:, 1:], np.abs(d[:, 1:] - d[:, :-1]))
    step[:, :-1] = np.maximum(step[:, :-1], np.abs(d[:, 1:] - d[:, :-1]))
    return step > EDGE_PX


def warp(sphere, m, w, age, motion):
    """Forward-splat the state into the current frame; nearest surface wins."""
    keep = w > 0
    R, t = motion
    P = sphere.triangulate(U[keep].astype(np.float64), V[keep].astype(np.float64), m[keep].astype(np.float64))
    P = R @ P + t[:, None]
    ok = np.isfinite(P).all(0) & (P[2] > 0.05)
    ul, _, vv, disp = sphere.project(P[:, ok])
    col, row = np.rint(ul).astype(np.int64), np.rint(vv).astype(np.int64)
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H) & np.isfinite(disp) & (disp > 0)
    index = (row * W + col)[inside]
    disp = disp[inside].astype(np.float32)
    w_src, age_src = w[keep][ok][inside], age[keep][ok][inside]
    order = np.argsort(disp, kind='stable')  # ascending: the last write (largest disparity) wins
    m_out = np.zeros(H * W, np.float32); w_out = np.zeros(H * W, np.int32); age_out = np.zeros(H * W, np.int32)
    m_out[index[order]] = disp[order]
    w_out[index[order]] = w_src[order]
    age_out[index[order]] = age_src[order]
    return m_out.reshape(H, W), w_out.reshape(H, W), age_out.reshape(H, W)


def write_png(path, image):
    import struct
    import zlib
    raw = b''.join(b'\x00' + image[y].tobytes() for y in range(image.shape[0]))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', image.shape[1], image.shape[0],
                                                                            8, 0, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def run_filter(session, out, source, gate, w_max, viz_frames=(), relative=None, d_min=0.0):
    frames = capture_frames(session)
    sphere = SphericalModel(SPHERICAL_DEG)
    src = out / source / 'model'
    name = arm_name(source, gate, w_max, relative, d_min)
    dst = out / name / 'model'
    dst.mkdir(parents=True, exist_ok=True)
    m = np.zeros((H, W), np.float32); w = np.zeros((H, W), np.int32); age = np.zeros((H, W), np.int32)
    stats, viz = [], []
    began = time.monotonic()
    for i, frame in enumerate(frames):
        prefix = f'{frame["name"]}-spherical'
        obs = np.fromfile(src / f'{prefix}-disparity.f32', '<f4').reshape(H, W)
        for suffix in ('right-disparity.f32', 'uncertainty.u8', 'right-uncertainty.u8'):
            target = dst / f'{prefix}-{suffix}'
            target.unlink(missing_ok=True)
            try:
                os.link(src / f'{prefix}-{suffix}', target)  # unchanged planes; saves about 0.4 GB per arm
            except OSError:
                shutil.copyfile(src / f'{prefix}-{suffix}', target)
        valid = plausible(obs)
        reset = (frame['pose'] is None or i == 0 or frames[i - 1]['pose'] is None or
                 frame['exposure_ns'] - frames[i - 1]['exposure_ns'] > RESET_GAP_NS)
        if not reset:
            motion = relative_motion(frames[i - 1]['pose'], frame['pose'])
            reset = rotation_deg(motion[0]) > RESET_ROTATION_DEG
        if reset:
            m, w, age = np.where(valid, obs, 0).astype(np.float32), valid.astype(np.int32), np.zeros((H, W), np.int32)
            output = obs.copy()
            stats.append({'frame': i + 1, 'reset': True})
        else:
            hm, hw, hage = warp(sphere, m, w, age, motion)
            history = hw > 0
            edge = edges(obs)
            usable = valid & ~edge
            limit = gate if relative is None else np.minimum(GATE_CAP_PX, relative * hm)
            agree = history & usable & (hm >= d_min) & (np.abs(obs - hm) <= limit)
            conflict = history & usable & ~agree
            fused = (hw * hm + obs) / (hw + 1)
            output = np.where(agree, fused, obs).astype(np.float32)
            m = np.where(valid, np.where(agree, fused, obs), np.where(history & (hage < EXPIRE), hm, 0)).astype(np.float32)
            w = np.where(valid, np.where(agree, np.minimum(hw + 1, w_max), 1),
                         np.where(history & (hage < EXPIRE), hw, 0)).astype(np.int32)
            age = np.where(valid, 0, hage + 1).astype(np.int32)
            n = float(valid.sum()) or 1.0
            stats.append({'frame': i + 1, 'reset': False, 'history': float((history & valid).sum() / n),
                          'agree': float(agree.sum() / n), 'conflict': float(conflict.sum() / n),
                          'edge': float((edge & valid).sum() / n), 'mean_support': float(hw[agree].mean()) if agree.any() else 0.0})
            if i + 1 in viz_frames:
                cls = np.full((H, W), 128, np.uint8)
                cls[~history] = 0; cls[agree] = 200; cls[conflict] = 255; cls[edge & valid] = 60
                scale = lambda d: np.clip(d / 30 * 255, 0, 255).astype(np.uint8)
                viz.append(np.hstack([scale(obs), scale(hm), scale(output), cls]))
        output.astype('<f4').tofile(dst / f'{prefix}-disparity.f32')
    summary = {'source': source, 'gate_px': gate, 'relative_gate': relative, 'gate_cap_px': GATE_CAP_PX,
               'd_min_px': d_min, 'w_max': w_max, 'edge_px': EDGE_PX, 'expire': EXPIRE,
               'frames': len(frames), 'resets': [s['frame'] for s in stats if s['reset']],
               'seconds': time.monotonic() - began,
               'mean_agree': float(np.mean([s['agree'] for s in stats if not s['reset']])),
               'mean_conflict': float(np.mean([s['conflict'] for s in stats if not s['reset']])),
               'mean_history': float(np.mean([s['history'] for s in stats if not s['reset']]))}
    write_json(out / name / 'filter.json', {'summary': summary, 'per_frame': stats})
    if viz:
        # Columns: observation, warped history, output, class (black none, dark edge, grey agree-free, light agree, white conflict).
        write_png(out / name / 'conflicts.png', np.vstack(viz))
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items() if k != 'resets'}),
          'resets', len(summary['resets']), flush=True)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', nargs='+', default=['off', 'recurrence'])
    parser.add_argument('--gate', type=float, help='one configuration instead of the declared grid')
    parser.add_argument('--w-max', type=int)
    parser.add_argument('--range-grid', action='store_true', help='run the declared range-aware grid')
    parser.add_argument('--viz', type=int, nargs='*', default=[], help='1-based frames for conflicts.png')
    args = parser.parse_args()
    for source in args.source:
        if args.range_grid:
            for relative, d_min in RANGE_GRID:
                run_filter(args.session.resolve(), args.out.resolve(), source, None, RANGE_W_MAX,
                           set(args.viz), relative, d_min)
            continue
        grid = [(args.gate, args.w_max)] if args.gate is not None else GRID
        for gate, w_max in grid:
            run_filter(args.session.resolve(), args.out.resolve(), source, gate, w_max, set(args.viz))


if __name__ == '__main__':
    main()
