#!/usr/bin/env python3
"""E06 gate G tag check: compare temporal arms with the off arm on a tagged capture.

Offline development tool. Arms use the temporal_event_test.py layout (OUT/ARM/model/).

  detect  run frame_apriltag on every STEP-th rectified left image -> OUT/tags.json
  score   per-detection tag-plane metrics for each arm against off -> OUT/tag_score.json

Metrics follow E05's tag check (qnn_ab.py), left disparity on the spherical model grid:
  offset     median signed distance of model points inside the tag quad from the PnP tag
             plane. Gate G: p95 of (|arm offset| - |off offset|) at most 15 mm
             (<= 1.25 m PnP range), 30 mm (1.25-2 m), 36 mm (2-3 m).
  normal     angle between a plane fitted to a wall annulus around the tag (1.5-4 tag
             half-widths, within 0.3 m of the tag plane) and the PnP normal. Gate G:
             p95 increase at most 5 degrees (the tag reference limits it to 5).
  complete   fraction of plausible pixels with uncertainty < 89 over all frames;
             gate G: loss at most 2 points.
  jitter     per-tag standard deviation of the offset across frames: an independent
             temporal measure (no warp involved), though it also contains view changes.
D1 (stereo yaw drift) biases absolute offsets; only arm-minus-off changes are judged.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import H, W, SPHERICAL_DEG, frame_paths, plausible, write_json  # noqa: E402

TAG_TOOL = Path.home() / 'frame-spatial-components/build/frame_apriltag'
BASELINE_M = 24.3048 / 380
RANGE_LIMITS = ((0.0, 1.25, 15.0), (1.25, 2.0, 30.0), (2.0, 3.0, 36.0))
NORMAL_LIMIT_DEG = 5.0
COMPLETENESS_LIMIT_POINTS = 2.0
V, U = np.mgrid[0:H, 0:W].astype(float)


def sphere_points(d):
    k = np.radians(SPHERICAL_DEG)
    theta, phi = (U - 127.5) * k, (V - 159.5) * k
    with np.errstate(all='ignore'):
        rho = BASELINE_M / (np.tan(theta) - np.tan(theta - d * k))
    return np.stack([rho * np.tan(theta), rho * np.sin(phi), rho * np.cos(phi)], -1)


def inside_quad(P, quad):
    """Points whose pinhole projection on the 616 rectified image falls inside the tag quad."""
    focal, centre = 380.0, 308.0
    with np.errstate(all='ignore'):
        x, y = focal * P[..., 0] / P[..., 2] + centre, focal * P[..., 1] / P[..., 2] + centre
    ok, sign = np.ones(x.shape, bool), None
    for a in range(4):
        p0, p1 = quad[a], quad[(a + 1) % 4]
        current = np.sign((p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0]))
        sign = current if sign is None else sign
        ok &= current == sign
    return ok, x, y


def detect(args):
    paths = frame_paths(args.session / 'rgb')
    rows = []
    for number in range(1, len(paths) + 1, args.step):
        image = args.session / 'rgb' / f'frame-{number}-rect-left.u8'
        result = subprocess.run([str(args.tag_tool), '--raw', str(image), '--image-size', '616',
                                 '--tag-size-m', str(args.tag_size_m)], capture_output=True, text=True)
        try:
            detections = json.loads(result.stdout.splitlines()[0])['detections']
        except (IndexError, ValueError, KeyError):
            continue
        for tag in detections:
            if tag.get('accepted'):
                rows.append({'frame': number, 'id': tag['id'], 'corners': tag['corners'],
                             'position': tag['camera_from_tag_m'], 'rotation': tag['rotation_camera_from_tag']})
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'tags.json', {'session': str(args.session), 'step': args.step,
                                        'tag_size_m': args.tag_size_m, 'detections': rows})
    print(f'{len(rows)} accepted detections in {len({r["frame"] for r in rows})} frames, ids '
          f'{sorted({r["id"] for r in rows})}')


def measure(d, tag):
    position = np.array(tag['position'])
    normal = np.array(tag['rotation'])[:, 2]
    if normal @ position > 0:
        normal = -normal
    quad = np.array(tag['corners'])
    centre = quad.mean(0)
    half = np.abs(quad - centre).max()
    P = sphere_points(d)
    valid = np.isfinite(P).all(-1) & (P[..., 2] > 0.2)
    inside, x, y = inside_quad(np.where(valid[..., None], P, 1), quad)
    inside &= valid
    if inside.sum() < 20:
        return None
    result = {'offset': float(np.median((P[inside] - position) @ normal))}
    distance = np.hypot(x - centre[0], y - centre[1])
    annulus = valid & ~inside & (distance < 4 * half) & (distance > 1.5 * half)
    Q = P[annulus]
    Q = Q[np.abs((Q - position) @ normal) < 0.3]
    if len(Q) > 50:
        _, _, vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
        result['normal_deg'] = float(np.degrees(np.arccos(min(1.0, abs(vt[2] @ normal)))))
    return result


def load(out, arm, name, kind='disparity.f32'):
    path = out / arm / 'model' / f'{name}-spherical-{kind}'
    dtype = '<f4' if kind.endswith('f32') else np.uint8
    return np.fromfile(path, dtype).reshape(H, W)


def score(args):
    tags = json.loads((args.out / 'tags.json').read_text())['detections']
    names = [p.stem for p in frame_paths(args.session / 'rgb')]
    arms = ['off'] + [a for a in args.arms if a != 'off']
    per_arm = {arm: [] for arm in arms}
    for tag in tags:
        name = f'frame-{tag["frame"]}'
        row = {'frame': tag['frame'], 'id': tag['id'], 'range': float(np.linalg.norm(tag['position']))}
        measured = {arm: measure(load(args.out, arm, name), tag) for arm in arms}
        if any(m is None for m in measured.values()):
            continue
        for arm in arms:
            per_arm[arm].append({**row, **measured[arm]})
    complete = {}
    for arm in arms:
        values = [float((plausible(load(args.out, arm, n)) & (load(args.out, arm, n, 'uncertainty.u8') < 89)).mean())
                  for n in names[::args.complete_step]]
        complete[arm] = float(np.mean(values))
    report = {'detections_used': len(per_arm['off']), 'completeness': complete, 'arms': {}}
    for arm in arms:
        rows, base = per_arm[arm], per_arm['off']
        jitter = {}
        for tag_id in sorted({r['id'] for r in rows}):
            values = np.array([r['offset'] for r in rows if r['id'] == tag_id]) * 1000
            jitter[str(tag_id)] = {'n': len(values), 'offset_median_mm': float(np.median(values)),
                                   'offset_std_mm': float(values.std())}
        entry = {'per_tag': jitter, 'mean_offset_std_mm': float(np.mean([j['offset_std_mm'] for j in jitter.values()]))}
        if arm != 'off':
            bins, passed = {}, True
            for low, high, limit in RANGE_LIMITS:
                index = [k for k, r in enumerate(base) if low < r['range'] <= high]
                if not index:
                    continue
                increase = np.array([abs(rows[k]['offset']) - abs(base[k]['offset']) for k in index]) * 1000
                normals = [rows[k]['normal_deg'] - base[k]['normal_deg'] for k in index
                           if 'normal_deg' in rows[k] and 'normal_deg' in base[k]]
                p95 = float(np.percentile(increase, 95))
                normal_p95 = float(np.percentile(normals, 95)) if normals else None
                ok = p95 <= limit and (normal_p95 is None or normal_p95 <= NORMAL_LIMIT_DEG)
                passed &= ok
                bins[f'{low}-{high} m'] = {'n': len(index), 'limit_mm': limit, 'offset_increase_mean_mm': float(increase.mean()),
                                           'offset_increase_p95_mm': p95, 'normal_increase_p95_deg': normal_p95, 'pass': ok}
            loss = (complete['off'] - complete[arm]) * 100
            passed &= loss <= COMPLETENESS_LIMIT_POINTS
            entry.update(bins=bins, completeness_loss_points=loss, gate_g_pass=bool(passed))
        report['arms'][arm] = entry
        print(arm, json.dumps({k: entry[k] for k in ('mean_offset_std_mm', 'gate_g_pass') if k in entry}),
              {k: round(v['offset_increase_p95_mm'], 1) for k, v in entry.get('bins', {}).items()}, flush=True)
    write_json(args.out / 'tag_score.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    det = sub.add_parser('detect')
    det.add_argument('--step', type=int, default=2)
    det.add_argument('--tag-tool', type=Path, default=TAG_TOOL)
    det.add_argument('--tag-size-m', type=float, default=0.12)
    sc = sub.add_parser('score')
    sc.add_argument('--arms', nargs='+', required=True)
    sc.add_argument('--complete-step', type=int, default=5)
    for each in (det, sc):
        each.add_argument('--session', type=Path, required=True)
        each.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.session, args.out = args.session.resolve(), args.out.resolve()
    {'detect': detect, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
