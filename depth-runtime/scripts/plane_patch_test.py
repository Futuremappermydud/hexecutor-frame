#!/usr/bin/env python3
"""E09 real-capture check of PlanePatchMap (bin/replay_plane_patches) on frozen model outputs.

Offline development tool. Arms use the temporal_event_test.py layout (OUT/ARM/model/).

  run          stream the arm's left disparity, uncertainty and poses through the replay
               tool -> OUT/ARM/planes/{labels.u16,frames.jsonl,patches.json}
  score-tags   tagged capture (tags.json from temporal_tag_check.py detect)
  score-scene  event capture (segments as in temporal_event_test.py)

Gate, declared 2026-11-08 before the first real-capture run. Absolute offsets carry the
D1 stereo bias, so offsets are judged against the TSDF mesh plane at the same tag (same
bias), never against the tag itself.
  Tagged capture (desk-tags-full-001; tags 1 and 2 on the wall, 3 on the desk):
   (a) coverage: the tag centre (projected with the frame pose) lands on a patch label in
       at least 80% of its detection frames; identity: at least 95% of those labels resolve
       (following merges) to the same final patch
   (b) that final patch's normal is within 5 degrees of the median PnP tag normal (PnP is
       about +-2.5 degrees)
   (c) tags 1 and 2 resolve to one patch and tag 3 to another; the angle between those two
       patches is within 3 degrees of the tags' own wall-desk angle
   (d) the patch's plane passes within 15 mm of the TSDF mesh-plane offset at each tag
       (tsdf_tags_e08.json, baseline variant)
  Event capture (tabletop):
   (e) parallel offset planes: the table top and the floor, if a horizontal confirmed patch
       0.5-1.0 m below the table exists, are separate confirmed patches
   (f) no confirmed patch has at least half its footprint in hand/cloth transient cells
   (g) identity: the table-top patch under the still segment's image centre keeps one final
       id in at least 95% of labelled frames through the still, hand and object segments
  Reported, not gated: patch counts by state, merges, contests, replay time per frame.
Not gated here: curved surfaces (synthetic test shows ~40-50 degree facets on a 0.3 m pillar).
"""
import argparse
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import (H, PATHS, SPHERICAL_DEG, W, capture_frames, segments_from_marks,  # noqa: E402
                                 write_json)

K = np.radians(SPHERICAL_DEG)
VOXEL_M = 0.05
UP = np.array([0.0, 1.0, 0.0])


def run(args):
    frames = capture_frames(args.session)
    model = args.out / args.arm / 'model'
    planes = args.out / args.arm / 'planes'
    planes.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(args.replay), str(planes)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    process.stdin.write(struct.pack('<I', len(frames)))
    for frame in frames:
        d = np.fromfile(model / f'{frame["name"]}-spherical-disparity.f32', '<f4')
        u = np.fromfile(model / f'{frame["name"]}-spherical-uncertainty.u8', np.uint8)
        if frame['pose'] is None:
            process.stdin.write(struct.pack('<I12f', 0, *([0.0] * 12)))
        else:
            R, t = frame['pose']
            process.stdin.write(struct.pack('<I12f', 1, *np.asarray(R, float).ravel(), *np.asarray(t, float)))
        process.stdin.write(d.astype('<f4').tobytes())
        process.stdin.write(u.tobytes())
    process.stdin.close()
    out = process.stdout.read().decode().strip()
    if process.wait():
        sys.exit(f'replay failed: {out}')
    print(out)


def load(args):
    planes = args.out / args.arm / 'planes'
    patches = {p['id']: p for p in json.loads((planes / 'patches.json').read_text())['patches']}
    labels = np.memmap(planes / 'labels.u16', np.uint16, 'r').reshape(-1, H, W)
    frames = [json.loads(line) for line in (planes / 'frames.jsonl').read_text().splitlines()]
    return patches, labels, frames


def resolve(patches, pid):
    seen = set()
    while pid and patches[pid]['merged_into'] and pid not in seen:
        seen.add(pid)
        pid = patches[pid]['merged_into']
    return pid


def project(pose, point):
    """World point -> (u, v) on the spherical model grid."""
    R, t = pose
    c = np.asarray(R).T @ (np.asarray(point) - np.asarray(t))
    rho = np.hypot(c[1], c[2])
    return np.arctan2(c[0], rho) / K + 127.5, np.arctan2(c[1], c[2]) / K + 159.5


def label_at(labels, index, u, v):
    iu, iv = int(round(u)), int(round(v))
    if not (1 <= iu < W - 1 and 1 <= iv < H - 1):
        return None
    window = labels[index, iv - 1:iv + 2, iu - 1:iu + 2].ravel()
    window = window[window > 0]
    return int(Counter(window.tolist()).most_common(1)[0][0]) if len(window) else 0


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(min(1.0, abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)))))


def summary(patches, frames):
    states = Counter(p['state'] for p in patches.values() if not p['merged_into'])
    ms = [f['ms'] for f in frames if f['valid']]
    return {'patches_total': len(patches), 'merged': sum(1 for p in patches.values() if p['merged_into']),
            'final_by_state': {name: states.get(code, 0) for code, name in
                               ((0, 'tentative'), (1, 'confirmed'), (2, 'contested'), (3, 'retired'))},
            'contests': sum(f['contests'] for f in frames), 'ms_p50': float(np.median(ms)),
            'ms_p95': float(np.percentile(ms, 95))}


def score_tags(args):
    from tsdf_variant_test import tag_planes
    frames = capture_frames(args.session)
    patches, labels, stats = load(args)
    planes = tag_planes(args, frames)
    detections = json.loads(args.tags.read_text())['detections']
    mesh = json.loads(args.mesh_tags.read_text())['variants']['baseline']['tags']
    index = {f['name']: i for i, f in enumerate(frames)}
    report = {'summary': summary(patches, stats), 'tags': {}}
    final_ids = {}
    for tag_id, plane in planes.items():
        ids = []
        for det in detections:
            if det['id'] != tag_id:
                continue
            i = index.get(f'frame-{det["frame"]}')
            if i is None or frames[i]['pose'] is None:
                continue
            label = label_at(labels, i, *project(frames[i]['pose'], plane['centre']))
            if label is not None:
                ids.append(resolve(patches, label) if label else 0)
        labelled = [x for x in ids if x]
        coverage = len(labelled) / max(len(ids), 1)
        modal, count = Counter(labelled).most_common(1)[0] if labelled else (0, 0)
        entry = {'detections': len(ids), 'coverage': coverage,
                 'identity': count / max(len(labelled), 1), 'patch': modal}
        if modal:
            p = patches[modal]
            n = np.array(p['normal'])
            # Where the patch plane crosses the tag-normal line through the tag centre, measured
            # along the tag normal: the same quantity as the mesh-plane offset at that tag.
            along = float((p['offset'] - n @ plane['centre']) / (n @ plane['normal']) * 1000)
            entry.update(state=p['state'], views=p['views'], normal_error_deg=angle(n, plane['normal']),
                         patch_offset_mm=along, rms_mm=p['rms'] * 1000)
            ref = mesh.get(str(tag_id))
            if ref:
                entry['mesh_offset_mm'] = ref['offset_mm']
                entry['patch_minus_mesh_mm'] = along - ref['offset_mm']
        entry['checks'] = {'coverage': coverage >= 0.8, 'identity': entry['identity'] >= 0.95,
                           'normal': bool(modal) and entry['normal_error_deg'] <= 5,
                           'mesh_consistency': 'patch_minus_mesh_mm' in entry and abs(entry['patch_minus_mesh_mm']) <= 15}
        final_ids[tag_id] = modal
        report['tags'][str(tag_id)] = entry
    wall = {final_ids.get(1), final_ids.get(2)}
    desk = final_ids.get(3)
    pnp_angle = angle(planes[1]['normal'], planes[3]['normal']) if 1 in planes and 3 in planes else None
    structure = {'wall_ids': sorted(x for x in wall if x is not None), 'desk_id': desk, 'pnp_wall_desk_deg': pnp_angle}
    if len(wall) == 1 and desk and desk not in wall and None not in wall and 0 not in wall:
        patch_angle = angle(patches[final_ids[1]]['normal'], patches[desk]['normal'])
        structure.update(patch_wall_desk_deg=patch_angle, check=abs(patch_angle - pnp_angle) <= 3)
    else:
        structure['check'] = False
    report['structure'] = structure
    report['pass'] = bool(structure['check'] and all(all(t['checks'].values()) for t in report['tags'].values()))
    write_json(args.out / args.arm / 'planes' / 'score_tags.json', report)
    print(json.dumps(report, indent=1, default=float))


def score_scene(args):
    from tsdf_variant_test import transient_cells
    frames = capture_frames(args.session)
    segments, _ = segments_from_marks(args.session, frames, args.segments)
    patches, labels, stats = load(args)
    report = {'summary': summary(patches, stats)}
    final = {pid: p for pid, p in patches.items() if not p['merged_into'] and p['state'] == 1}
    horizontal = {pid: p for pid, p in final.items() if angle(p['normal'], UP) <= 5}
    # Table top: the patch under the image centre during the still segment.
    first, last = segments['still']
    centre_ids = [resolve(patches, label_at(labels, i, 127.5, 159.5) or 0) for i in range(first, last + 1)]
    centre_ids = [x for x in centre_ids if x]
    table = Counter(centre_ids).most_common(1)[0][0] if centre_ids else 0
    report['table_patch'] = table
    heights = {pid: float(p['centroid'][1]) for pid, p in horizontal.items()}
    floors = [pid for pid, h in heights.items() if table in heights and 0.5 <= heights[table] - h <= 1.0]
    report['horizontal_heights_m'] = heights
    report['parallel'] = {'table_confirmed': table in final, 'floor_patches': floors,
                          'check': table in final and (not floors or table not in floors)}
    # Transient patches: footprint mostly inside hand/cloth cells.
    cells = transient_cells(argparse.Namespace(out=args.out, arm=args.arm), frames, segments)
    bad = []
    for pid, p in final.items():
        keys = np.array(p['footprint'], np.int64)
        if not len(keys):
            continue
        field = lambda k: ((k & 0x1fffff) - (1 << 20)).astype(float)
        centres = np.stack([field(keys >> 42), field(keys >> 21), field(keys)], 1) * VOXEL_M + VOXEL_M / 2
        inside = np.mean([tuple(c) in cells for c in np.floor(centres / 0.03).astype(np.int64)])
        if inside >= 0.5:
            bad.append({'id': pid, 'fraction': float(inside), 'voxels': len(keys)})
    report['transient_patches'] = {'patches': bad, 'check': not bad}
    # Identity of the table patch through still, hand and object.
    ids = []
    for segment in ('still', 'hand', 'object'):
        a, b = segments[segment]
        for i in range(a, b + 1):
            label = label_at(labels, i, 127.5, 159.5)
            if label:
                ids.append(resolve(patches, label))
    share = Counter(ids).most_common(1)[0][1] / len(ids) if ids else 0
    report['identity'] = {'labelled_frames': len(ids), 'modal_share': share, 'check': share >= 0.95}
    report['pass'] = bool(report['parallel']['check'] and report['transient_patches']['check'] and report['identity']['check'])
    write_json(args.out / args.arm / 'planes' / 'score_scene.json', report)
    print(json.dumps(report, indent=1, default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--replay', type=Path, default=PATHS.repo / 'depth-runtime/bin/replay_plane_patches')
    t = sub.add_parser('score-tags')
    t.add_argument('--tags', type=Path, required=True)
    t.add_argument('--mesh-tags', type=Path, required=True, help='tsdf_tags_e08.json')
    s = sub.add_parser('score-scene')
    s.add_argument('--segments')
    for each in (r, t, s):
        each.add_argument('--session', type=Path, required=True)
        each.add_argument('--out', type=Path, required=True)
        each.add_argument('--arm', default='off')
    args = parser.parse_args()
    args.session, args.out = args.session.resolve(), args.out.resolve()
    {'run': run, 'score-tags': score_tags, 'score-scene': score_scene}[args.command](args)


if __name__ == '__main__':
    main()
