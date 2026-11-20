#!/usr/bin/env python3
"""E06 ghost/lag event test: compare temporal arms with the recurrence-off arm.

Offline development tool (Frame for `run`; `mesh` and `score` anywhere with the data).
The capture comes from `room_dataset.py capture` at about 10 Hz with controller marks.
A trigger click starts each protocol segment, in this order (see PROTOCOL):

  still    10 s  head still, plain wall about 1.2 m away or a tabletop 0.8-1.2 m (noise floor)
  hand     20 s  head still on the same surface; sweep the free hand across the view at
                 about 40 cm, five times, with 2 s out of view between sweeps
  object   20 s  head still on a desk or floor; place a box, wait 3 s, remove it, twice
  approach 15 s  walk slowly from about 2.5 m to 1 m from it and back, surface centred
  turn     10 s  turn the head about 90 degrees quickly, hold 2 s, turn back, twice
  end            final click; hold grip and trigger 2 s to stop the recording

Commands:
  run   --arm off|recurrence   model outputs in short DSP sessions (resumable)
  mesh  --arm A                TSDF mesh of an arm's outputs on the capture's poses
  score --arms A [B ...]       event metrics and the frozen verdict for each arm

An arm is a directory OUT/ARM/model/ with frame-N-spherical-{disparity,right-disparity}.f32
and -{uncertainty,right-uncertainty}.u8, the layout of model_offline_batch.py. An external
filter writes the same layout from the off or recurrence arm, then reuses mesh and score.

Frozen tolerances (E06, approved 2026-11-08), each arm against the off arm of the same
capture, left disparity in model pixels, plausible depth 0.25-5 m:
  thr    a pixel disagrees when |arm - off| > max(1 px, 3 x the off arm's p95
         |off(t) - warp(off(t-1))| on the still segment)
  big    an off-arm change of at least max(4 px, 2 x thr) between warped consecutive
         frames; components of at least MIN_AREA pixels in the hand/object segments
         are events
  ghost  after a disocclusion event (off moves farther), the arm stays nearer than off
         by more than thr on over 10% of the event's pixels for at most 2 frames
         (lifetime counted from the event frame; a perfect arm has 0)
  lag    after a new-foreground event (off moves nearer), at most 1 frame
  slow   approach segment: the frame shift that best aligns the arm's central median
         disparity with the off arm's is at most 3 frames
  blob   anywhere, a connected disagreement region of at least MIN_AREA pixels
         persists for at most 2 consecutive frames
  mesh   no extra-surface cluster in the arm's mesh; candidates are written for a
         visual check, which decides (gate G critical failure)
Event masks stay in place, so hand and object segments assume a still head; score
reports head angular speed there and flags segments where it exceeds 5 deg/s.
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

SCRIPTS = PATHS.repo / 'depth-runtime/scripts'
sys.path.insert(0, str(SCRIPTS))
from room_dataset import frame_paths, load_poses, nearest_pose, pose_from_row, write_depth_metadata  # noqa: E402

H, W = 320, 256
SPHERICAL_DEG = 0.29
FB_MODEL = 24.3048 * 256 / 616
EXPOSURE_OFFSET_MS = 8.0
PROTOCOL = ('still', 'hand', 'object', 'approach', 'turn')
EVENT_SEGMENTS = ('hand', 'object')
MIN_AREA = 150
RESOLVED_FRACTION = 0.10
MAX_TRACK = 20
STILL_HEAD_DEG_S = 5.0
CENTRE = (slice(80, 240), slice(64, 192))
LIMITS = {'ghost_frames': 2, 'lag_frames': 1, 'slow_lag_frames': 3, 'blob_frames': 2, 'mesh_clusters': 0}
# Worker invalidation limits (cloud_worker.cpp, unmeasured starting values), applied to recurrence.
RESET_GAP_NS, RESET_ROTATION_DEG = 250_000_000, 30.0
MESH_CELL_M, MESH_CLUSTER_MIN = 0.03, 50
FLOOR_Y_M, BELOW_FLOOR_RATIO, NEAR_HEAD_M = -0.05, 1.05, 1.5  # standing universe: floor at y = 0
ZERO_H = np.zeros((1, 2, H, W), np.float32)
ZERO_T = np.zeros((1, 2, 4, 4), np.float32)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')


# Capture access --------------------------------------------------------------------------

def capture_frames(session):
    """Per frame: exposure time and camera-to-world pose (None when no valid pose)."""
    extrinsic = json.loads((session / 'rgb/rectified-camera-in-head.json').read_text())
    poses = load_poses(session / 'head_pose.csv')
    rows = []
    for path in frame_paths(session / 'rgb'):
        meta = json.loads(path.read_text())
        exposure = int(meta['left_end_of_readout_ns']) - int(EXPOSURE_OFFSET_MS * 1e6)
        row, _ = nearest_pose(poses, exposure)
        pose = None
        if row is not None:
            rotation, translation = pose_from_row(row, extrinsic)
            pose = (np.array(rotation, float).reshape(3, 3), np.array(translation, float))
        rows.append({'name': path.stem, 'exposure_ns': exposure, 'pose': pose})
    return rows


def relative_motion(previous, current):
    """Previous rectified-left camera coordinates to current ones."""
    (Rp, tp), (Rc, tc) = previous, current
    return Rc.T @ Rp, Rc.T @ (tp - tc)


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_images(session, name):
    return np.stack([np.fromfile(session / 'rgb' / f'{name}-spherical-{eye}.u8', np.uint8).reshape(H, W)
                     for eye in ('left', 'right')])


def arm_model(out, arm):
    return out / arm / 'model'


def load_left(out, arm, names):
    model = arm_model(out, arm)
    return np.stack([np.fromfile(model / f'{name}-spherical-disparity.f32', '<f4').reshape(H, W) for name in names])


def segments_from_marks(session, frames, override):
    """Segment frame ranges (0-based, inclusive) from trigger strokes or --segments."""
    if override:
        result = {}
        for item in override.split(','):
            name, span = item.split(':')
            first, last = (int(v) - 1 for v in span.split('-'))
            result[name] = (first, last)
        return result, 'override'
    starts = {}
    with (session / 'marks.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            key = (row['device'], row['stroke'])
            starts[key] = min(starts.get(key, 1 << 62), int(row['mono_ns']))
    clicks = sorted(starts.values())
    # Six clicks, plus one more stroke when grip+trigger was held to stop the recording.
    if len(clicks) not in (len(PROTOCOL) + 1, len(PROTOCOL) + 2):
        sys.exit(f'marks.csv has {len(clicks)} trigger strokes, expected {len(PROTOCOL) + 1} '
                 f'(or {len(PROTOCOL) + 2} with the stop gesture); '
                 'pass --segments still:F-L,hand:F-L,... (1-based frames)')
    clicks = clicks[:len(PROTOCOL) + 1]
    times = np.array([f['exposure_ns'] for f in frames])
    edges = [int(np.searchsorted(times, click)) for click in clicks]
    return {name: (edges[k], edges[k + 1] - 1) for k, name in enumerate(PROTOCOL)}, 'marks'


# run -------------------------------------------------------------------------------------

def run(args):
    root = args.package.resolve()
    sys.path[:0] = [str(root / 'frame'), str(root / 'tools')]
    if args.bridge_dir:
        # A package-era bridge with prebuilt DSP binaries (E05 ran QNN through it).
        sys.path.insert(0, str(args.bridge_dir.resolve()))
        import quest_depth_bridge as bridge
    else:
        from depth_model import offline_bridge as bridge
    from model_temporal import ModelTemporalState
    frames = capture_frames(args.session)
    out = arm_model(args.out, args.arm)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out.parent / 'progress.json'
    progress = (json.loads(progress_path.read_text()) if progress_path.is_file() else
                {'arm': args.arm, 'backend': 'prepared-qnn', 'package': str(args.package.resolve()),
                 'chunk': args.chunk, 'frames_done': 0,
                 'bridge': str(args.bridge_dir.resolve()) if args.bridge_dir else 'depth_model.offline_bridge', 'sessions': [], 'resets': [], 'infer_ms': []})
    state = ModelTemporalState(spherical_deg=SPHERICAL_DEG)
    done = progress['frames_done']
    if done and args.arm == 'recurrence':
        name = frames[done - 1]['name']
        state.save(np.stack([np.fromfile(out / f'{name}-spherical-{plane}disparity.f32', '<f4').reshape(H, W)
                             for plane in ('', 'right-')]))
    os.sched_setaffinity(0, {7})

    def temporal_inputs(i):
        if args.arm != 'recurrence':
            return ZERO_H, ZERO_T
        current, previous = frames[i], frames[i - 1] if i else None
        reason = None
        if current['pose'] is None:
            reason = 'pose_invalid'
        elif previous is None or previous['pose'] is None:
            reason = 'no_previous_pose'
        elif current['exposure_ns'] - previous['exposure_ns'] > RESET_GAP_NS:
            reason = 'gap'
        if reason is None:
            motion = relative_motion(previous['pose'], current['pose'])
            if rotation_deg(motion[0]) > RESET_ROTATION_DEG:
                reason = 'rotation'
        if reason:
            state.reset()
            if i:
                progress['resets'].append({'frame': i + 1, 'reason': reason})
            return ZERO_H, ZERO_T
        return state.inputs(motion)

    while progress['frames_done'] < len(frames):
        first = progress['frames_done']
        last = min(first + args.chunk, len(frames))
        record = {'first_frame': first + 1, 'exit_ok': False}
        began = time.monotonic()
        closing = None
        try:
            with bridge.prepared_session(args.package.resolve(), 'primary', bridge.QNN_ENGINE,
                                         bridge.QNN_BOOTSTRAP) as infer:
                infer(load_images(args.session, frames[first]['name']), ZERO_H, ZERO_T)  # warm-up
                for i in range(first, last):
                    hidden, t = temporal_inputs(i)
                    start = time.monotonic()
                    disparity, uncertainty = infer(load_images(args.session, frames[i]['name']), hidden, t)
                    progress['infer_ms'].append((time.monotonic() - start) * 1e3)
                    d = np.asarray(disparity, np.float32).reshape(2, H, W)
                    u = np.asarray(uncertainty).reshape(2, H, W).astype(np.uint8)
                    prefix = out / f'{frames[i]["name"]}-spherical'
                    d[0].tofile(f'{prefix}-disparity.f32'); d[1].tofile(f'{prefix}-right-disparity.f32')
                    u[0].tofile(f'{prefix}-uncertainty.u8'); u[1].tofile(f'{prefix}-right-uncertainty.u8')
                    if args.arm == 'recurrence':
                        state.save(d)
                    progress['frames_done'] = i + 1
                closing = time.monotonic()
            record['exit_ok'] = True
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            ended = time.monotonic()
            record.update(active_s=(closing or ended) - began, exit_s=ended - closing if closing else None,
                          frames_done=progress['frames_done'])
            progress['sessions'].append(record)
            write_json(progress_path, progress)
            print(json.dumps(record), flush=True)
    ms = np.percentile(progress['infer_ms'], [50, 95]).round(2).tolist()
    print(f'{args.arm}: {len(frames)} frames, infer ms p50/p95 {ms}, {len(progress["sessions"])} sessions, '
          f'{len(progress["resets"])} resets. Run chunk_map.py now and require n_bad 0.', flush=True)


# mesh ------------------------------------------------------------------------------------

def mesh(args):
    from room_reconstruction_replay import replay_stream
    paths = frame_paths(args.session / 'rgb')
    depth = args.out / args.arm / 'depth'
    depth.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.converter), str(args.session / 'rgb'), str(arm_model(args.out, args.arm)),
                    str(depth), str(len(paths))], check=True)
    extrinsic = json.loads((args.session / 'rgb/rectified-camera-in-head.json').read_text())
    skipped = write_depth_metadata(paths, depth, load_poses(args.session / 'head_pose.csv'),
                                   extrinsic, EXPOSURE_OFFSET_MS)
    write_json(depth / 'capture.json',
               {'version': 2, 'model': 'primary', 'backend': 'prepared-qnn', 'projection': 'spherical',
                'temporal': args.arm, 'model_shape': [320, 256], 'rectified_shape': [616, 616],
                'right_disparity': True, 'capture_clock': 'CLOCK_MONOTONIC',
                'exposure_timestamp': True, 'config_sha256': ''})
    manifest = replay_stream(depth, args.out / args.arm / 'mesh', args.replay)
    if not args.keep_depth:
        shutil.rmtree(depth)  # about 4.8 MB per frame; regenerated from the model outputs on demand
    print(f'{args.arm}: meshed {manifest["frames"]} frames, {len(skipped)} pose skips -> '
          f'{args.out / args.arm / "mesh/mesh.ply"}')


def read_ply_vertices(path):
    with path.open() as file:
        count = 0
        while (text := file.readline().strip()) != 'end_header':
            if not text and file.tell() == os.fstat(file.fileno()).st_size:
                raise ValueError(f'{path}: no end_header')
            if text.startswith('element vertex'):
                count = int(text.split()[2])
        rows = [file.readline().split()[:3] for _ in range(count)]
    return np.array(rows, np.float32).reshape(-1, 3)


def extra_surface(arm_vertices, off_vertices):
    """Arm vertices with no off vertex in the surrounding 3x3x3 cells, clustered by cell."""
    occupied = set(map(tuple, np.unique(np.floor(off_vertices / MESH_CELL_M).astype(np.int64), axis=0)))
    cells = np.floor(arm_vertices / MESH_CELL_M).astype(np.int64)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    lonely = np.array([not any((x + a, y + b, z + c) in occupied for a, b, c in offsets) for x, y, z in unique],
                      bool)
    extra = lonely[inverse.ravel()]
    by_cell = {}
    for index in np.nonzero(extra)[0]:
        by_cell.setdefault(tuple(cells[index]), []).append(index)
    seen, clusters = set(), []
    for start in by_cell:
        if start in seen:
            continue
        queue, members = deque([start]), []
        seen.add(start)
        while queue:
            cell = queue.popleft()
            members += by_cell[cell]
            for a, b, c in offsets:
                neighbour = (cell[0] + a, cell[1] + b, cell[2] + c)
                if neighbour in by_cell and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        clusters.append(members)
    return int(extra.sum()), [c for c in clusters if len(c) >= MESH_CLUSTER_MIN]


# score -----------------------------------------------------------------------------------

def plausible(d):
    return (d > FB_MODEL / 5.0) & (d < FB_MODEL / 0.25)


def components(mask):
    """4-connected components as flat index arrays (pure NumPy/Python; no SciPy needed)."""
    flat = mask.ravel()
    label = np.zeros(flat.size, np.int32)
    found = []
    for seed in np.flatnonzero(flat):
        if label[seed]:
            continue
        label[seed] = len(found) + 1
        queue, members = deque([seed]), []
        while queue:
            p = queue.popleft()
            members.append(p)
            r, c = divmod(p, W)
            for q, ok in ((p - W, r > 0), (p + W, r < H - 1), (p - 1, c > 0), (p + 1, c < W - 1)):
                if ok and flat[q] and not label[q]:
                    label[q] = label[seed]
                    queue.append(q)
        found.append(np.array(members))
    return found


def dilate(mask, radius=3):
    grown = mask.copy()
    for _ in range(radius):
        grown[1:] |= grown[:-1]; grown[:-1] |= grown[1:]
        grown[:, 1:] |= grown[:, :-1]; grown[:, :-1] |= grown[:, 1:]
    return grown


def warped_previous(probe, previous_left, motion):
    """Pose-warp the previous left disparity into the current frame (0 where nothing lands)."""
    probe.save(np.stack([previous_left, np.zeros_like(previous_left)]))
    return probe.inputs(motion)[0][0, 0]


def score(args):
    from model_temporal import ModelTemporalState
    frames = capture_frames(args.session)
    names = [f['name'] for f in frames]
    segments, segment_source = segments_from_marks(args.session, frames, args.segments)
    off = load_left(args.out, 'off', names)
    probe = ModelTemporalState(spherical_deg=SPHERICAL_DEG, pyramid=0)
    n = len(frames)

    # Off-arm change against the warped previous frame, used for the floor and the events.
    change = np.full((n, H, W), np.nan, np.float32)
    head_speed = np.full(n, np.nan)
    for i in range(1, n):
        if frames[i]['pose'] is None or frames[i - 1]['pose'] is None:
            continue
        motion = relative_motion(frames[i - 1]['pose'], frames[i]['pose'])
        dt = (frames[i]['exposure_ns'] - frames[i - 1]['exposure_ns']) / 1e9
        head_speed[i] = rotation_deg(motion[0]) / dt if dt > 0 else np.nan
        previous = warped_previous(probe, off[i - 1], motion)
        valid = (previous > 0) & plausible(previous) & plausible(off[i])
        change[i][valid] = off[i][valid] - previous[valid]

    first, last = segments['still']
    floor = np.abs(change[first + 1:last + 1])
    floor_p95 = float(np.nanpercentile(floor, 95))
    thr = max(1.0, 3 * floor_p95)
    big = max(4.0, 2 * thr)
    report = {'segments': {k: [a + 1, b + 1] for k, a, b in ((k, *v) for k, v in segments.items())},
              'segment_source': segment_source, 'frames': n,
              'rate_hz': float(1e9 / np.median(np.diff([f['exposure_ns'] for f in frames]))),
              'pose_invalid_frames': sum(f['pose'] is None for f in frames),
              'still_floor_p95_px': floor_p95, 'thr_px': thr, 'big_px': big, 'limits': LIMITS,
              'head_speed_deg_s': {}, 'events': {}, 'arms': {}}

    events = []
    for segment in EVENT_SEGMENTS:
        first, last = segments[segment]
        speeds = head_speed[first + 1:last + 1]
        median_speed = float(np.nanmedian(speeds)) if np.isfinite(speeds).any() else None
        report['head_speed_deg_s'][segment] = {'median': median_speed,
                                               'flag_not_still': bool(median_speed and median_speed > STILL_HEAD_DEG_S)}
        for i in range(first + 1, last + 1):
            for kind, mask in (('foreground', change[i] > big), ('disocclusion', change[i] < -big)):
                for members in components(mask):
                    if len(members) >= MIN_AREA:
                        events.append({'segment': segment, 'kind': kind, 'frame': i, 'end': last, 'pixels': members})
    for segment in EVENT_SEGMENTS:
        for kind in ('foreground', 'disocclusion'):
            report['events'][f'{segment}_{kind}'] = sum(e['segment'] == segment and e['kind'] == kind for e in events)

    for arm in args.arms:
        values = load_left(args.out, arm, names)
        result = {}
        # Ghost and lag lifetimes per event.
        lifetimes = {'foreground': [], 'disocclusion': []}
        for e in events:
            sign = 1.0 if e['kind'] == 'disocclusion' else -1.0
            life = MAX_TRACK + 1
            for k in range(MAX_TRACK + 1):
                t = e['frame'] + k
                if t > e['end']:
                    life = None  # segment ended unresolved; reported separately
                    break
                a, o = values[t].ravel()[e['pixels']], off[t].ravel()[e['pixels']]
                usable = plausible(o)
                if usable.sum() < MIN_AREA // 3:
                    life = k
                    break
                if (sign * (a[usable] - o[usable]) > thr).mean() <= RESOLVED_FRACTION:
                    life = k
                    break
            lifetimes[e['kind']].append(life)

        def summary(values_, limit):
            known = [v for v in values_ if v is not None]
            return {'events': len(values_), 'unresolved_at_segment_end': len(values_) - len(known),
                    'max_frames': max(known) if known else None,
                    'p95_frames': float(np.percentile(known, 95)) if known else None,
                    'over_limit': sum(v > limit for v in known), 'limit_frames': limit}
        result['ghost'] = summary(lifetimes['disocclusion'], LIMITS['ghost_frames'])
        result['lag'] = summary(lifetimes['foreground'], LIMITS['lag_frames'])

        # Slow-change lag on the approach segment.
        first, last = segments['approach']
        def centre_median(stack, i):
            region = stack[i][CENTRE]
            ok = plausible(region) & plausible(off[i][CENTRE])
            return float(np.median(region[ok])) if ok.sum() > 1000 else np.nan
        m_off = np.array([centre_median(off, i) for i in range(first, last + 1)])
        m_arm = np.array([centre_median(values, i) for i in range(first, last + 1)])
        costs = {}
        for k in range(0, 11):
            a, o = m_arm[k:], m_off[:len(m_off) - k]
            ok = np.isfinite(a) & np.isfinite(o)
            if ok.sum() >= 20:
                costs[k] = float(np.mean(np.abs(a[ok] - o[ok])))
        slow = min(costs, key=costs.get) if costs else None
        result['slow_lag'] = {'best_shift_frames': slow, 'cost_px_by_shift': costs,
                              'limit_frames': LIMITS['slow_lag_frames']}

        # Persistent disagreement blobs anywhere, and per-segment jitter.
        longest, longest_at, chain, previous_masks = 0, None, {}, []
        per_segment = {name: 0 for name in segments}
        for i in range(n):
            disagree = (np.abs(values[i] - off[i]) > thr) & plausible(off[i])
            blobs = [b for b in components(disagree) if len(b) >= MIN_AREA]
            current = []
            for b in blobs:
                mask = np.zeros(H * W, bool); mask[b] = True
                grown = dilate(mask.reshape(H, W)).ravel()
                length = 1 + max((length for m, length in previous_masks if (m & grown).any()), default=0)
                current.append((mask, length))
                if length > longest:
                    longest, longest_at = length, i + 1
                for name, (a, b_) in segments.items():
                    if a <= i <= b_:
                        per_segment[name] = max(per_segment[name], length)
            previous_masks = current
        result['blob'] = {'longest_frames': longest, 'ending_frame': longest_at, 'by_segment': per_segment,
                          'limit_frames': LIMITS['blob_frames']}

        jitter = {}
        for name, (first, last) in segments.items():
            meds, over = [], []
            for i in range(first + 1, last + 1):
                if frames[i]['pose'] is None or frames[i - 1]['pose'] is None:
                    continue
                previous = warped_previous(probe, values[i - 1],
                                           relative_motion(frames[i - 1]['pose'], frames[i]['pose']))
                ok = (previous > 0) & plausible(previous) & plausible(values[i])
                if ok.sum() > 1000:
                    diff = np.abs(values[i][ok] - previous[ok])
                    meds.append(float(np.median(diff))); over.append(float((diff > 1).mean()))
            jitter[name] = {'median_px': float(np.median(meds)) if meds else None,
                            'over_1px': float(np.mean(over)) if over else None}
        result['jitter_consecutive'] = jitter

        mesh_path = args.out / arm / 'mesh/mesh.ply'
        off_mesh = args.out / 'off/mesh/mesh.ply'
        if arm != 'off' and mesh_path.is_file() and off_mesh.is_file():
            arm_v, off_v = read_ply_vertices(mesh_path), read_ply_vertices(off_mesh)
            extra, clusters = extra_surface(arm_v, off_v)
            missing, _ = extra_surface(off_v, arm_v)
            candidates = args.out / arm / 'mesh/extra-candidates.ply'
            points = arm_v[np.concatenate(clusters)] if clusters else np.zeros((0, 3), np.float32)
            with candidates.open('w') as file:
                file.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                           'property float x\nproperty float y\nproperty float z\nend_header\n')
                np.savetxt(file, points, fmt='%.4f')
            head_path = np.array([f['pose'][1] for f in frames if f['pose'] is not None])
            near = [float(np.min(np.linalg.norm(head_path - arm_v[c].mean(0), axis=1))) for c in clusters]
            below_arm, below_off = int((arm_v[:, 1] < FLOOR_Y_M).sum()), int((off_v[:, 1] < FLOOR_Y_M).sum())
            result['mesh'] = {'arm_vertices': len(arm_v), 'off_vertices': len(off_v),
                              'extra_vertices': extra, 'missing_vertices': missing,
                              'extra_clusters': [len(c) for c in clusters],
                              'cluster_head_distance_m': near,
                              'clusters_near_head': sum(d < NEAR_HEAD_M for d in near),
                              'below_floor_vertices': below_arm, 'off_below_floor_vertices': below_off,
                              'below_floor_ratio': below_arm / max(below_off, 1),
                              'candidates_ply': str(candidates), 'decision': 'visual check required'}
        else:
            result['mesh'] = None

        checks = {'ghost': result['ghost']['over_limit'] == 0 and result['ghost']['unresolved_at_segment_end'] == 0,
                  'lag': result['lag']['over_limit'] == 0 and result['lag']['unresolved_at_segment_end'] == 0,
                  'slow_lag': slow is not None and slow <= LIMITS['slow_lag_frames'],
                  'blob': longest <= LIMITS['blob_frames'],
                  'mesh_candidates': bool(result['mesh']) and not result['mesh']['extra_clusters']}
        if result['mesh']:
            # Declared with the range-aware filter grid (2026-11-08).
            checks['mesh_below_floor'] = result['mesh']['below_floor_ratio'] <= BELOW_FLOOR_RATIO
            checks['mesh_near_head'] = result['mesh']['clusters_near_head'] == 0
        result['checks'] = checks
        if arm == 'off':
            result['verdict'] = 'reference'
        else:
            result['verdict'] = ('pass' if all(checks.values()) else
                                 'fail' if not all(v for k, v in checks.items() if k != 'mesh_candidates') else
                                 'pass; far clusters need a visual check')
        report['arms'][arm] = result
        print(arm, json.dumps({'checks': checks, 'ghost_max': result['ghost']['max_frames'],
                               'lag_max': result['lag']['max_frames'], 'slow_lag': slow,
                               'blob_longest': longest}), flush=True)
    write_json(args.out / 'score.json', report)
    print(f'thr {thr:.2f} px, big {big:.2f} px, events {report["events"]}, '
          f'head {report["head_speed_deg_s"]} -> {args.out / "score.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--arm', choices=('off', 'recurrence'), required=True)
    run_parser.add_argument('--package', type=Path, required=True)
    run_parser.add_argument('--bridge-dir', type=Path,
                            help='directory holding quest_depth_bridge.py, when this checkout has no '
                                 'hexecutor/native/build')
    run_parser.add_argument('--chunk', type=int, default=60,
                            help='frames per DSP session; E05 closed 60-frame (2.45 s) sessions cleanly, '
                                 'and a 17.9 s session stuck')
    mesh_parser = sub.add_parser('mesh')
    mesh_parser.add_argument('--arm', required=True)
    mesh_parser.add_argument('--converter', type=Path, default=PATHS.repo / 'depth-runtime/bin/model_batch_to_v2')
    mesh_parser.add_argument('--keep-depth', action='store_true', help='keep the converted per-frame depth')
    mesh_parser.add_argument('--replay', type=Path, default=PATHS.repo / 'depth-runtime/bin/replay_room_tsdf')
    score_parser = sub.add_parser('score')
    score_parser.add_argument('--arms', nargs='+', required=True)
    score_parser.add_argument('--segments', help='override marks: still:F-L,hand:F-L,... (1-based frames)')
    for each in (run_parser, mesh_parser, score_parser):
        each.add_argument('--session', type=Path, required=True)
        each.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.session, args.out = args.session.resolve(), args.out.resolve()
    {'run': run, 'mesh': mesh, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
