#!/usr/bin/env python3
"""Compare IR refinement with no subtraction, the earlier frame, and both neighbours, on the same pairs.

Sham cycles fired no light, so any match or plane they produce with a mode is a false positive.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from room_dataset import frame_paths, ir_stereo_input, rgb_prior, select_lit_pairs

MODES = ('none', 'before', 'average')


def sham_entries(observations, cycles):
    roles = {(entry['cycle'], entry['role']): entry for entry in observations if 'role' in entry}
    entries = []
    for cycle in cycles:
        lit = roles.get((cycle['cycle'], 'lit'))
        if cycle['status'] == 'sham' and lit:
            entries.append({**lit, 'references': [roles[(cycle['cycle'], role)] for role in ('reference', 'after')
                                                  if (cycle['cycle'], role) in roles]})
    return entries


def evenly_sampled(entries, count):
    if len(entries) <= count:
        return entries
    return [entries[int(i)] for i in np.linspace(0, len(entries) - 1, count)]


def summarize(rows):
    kept = [row['kept'] for row in rows]
    planes = [row for row in rows if row['plane_rms_mm'] is not None]
    return {'pairs': len(rows), 'median_kept': float(np.median(kept)) if kept else None,
            'plane_rate': len(planes) / len(rows) if rows else None,
            'median_inliers': float(np.median([row['inliers'] for row in planes])) if planes else None,
            'median_plane_rms_mm': float(np.median([row['plane_rms_mm'] for row in planes])) if planes else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('session', type=Path)
    parser.add_argument('--ir-yaw-deg', type=float, required=True)
    parser.add_argument('--ir-calib', type=Path, default=Path('/persist/xrservice.json'))
    parser.add_argument('--max-pairs', type=int, default=30, help='pairs per group, evenly spaced through the scan')
    args = parser.parse_args()
    out = args.session.resolve()
    os.environ['IR_YAW_DEG'] = str(args.ir_yaw_deg)
    os.environ['IR_CALIB'] = str(args.ir_calib)
    from ir_dot_lib import Rectifier, load_rgb_chain
    from ir_dot_refine_bridge import refine
    rect = Rectifier()
    chain = load_rgb_chain(out / 'rgb/calibration-history-at-init.json')
    observations = [json.loads(line) for line in (out / 'ir/pairs.jsonl').read_text().splitlines()]
    cycles = [json.loads(line) for line in (out / 'pulses.jsonl').read_text().splitlines()]
    lit, selection = select_lit_pairs(observations, cycles)
    print('selection:', json.dumps(selection))
    times = [(int(json.loads(path.read_text())['left_end_of_readout_ns']), path.stem) for path in frame_paths(out / 'rgb')]
    report = {'session': out.name, 'yaw_deg': args.ir_yaw_deg, 'selection': selection, 'groups': {}}
    for group, entries in (('lit', lit), ('sham', sham_entries(observations, cycles))):
        entries = [entry for entry in evenly_sampled(entries, args.max_pairs) if entry['references']]
        rows = {mode: [] for mode in MODES}
        for entry in entries:
            when = entry['left_monotonic_ns']
            rgb_time, name = min(times, key=lambda item: abs(item[0] - when))
            if abs(rgb_time - when) > 250_000_000:
                continue
            disp, unc = rgb_prior(out, name)
            for mode in MODES:
                left, right, _ = ir_stereo_input(out, entry, mode)
                kept, _, plane = refine(rect, chain, disp, unc, left, right, rng=np.random.default_rng(7))
                fit = plane[0] if plane else None
                rows[mode].append({'pair': entry['pair'], 'kept': kept, 'inliers': fit.count if fit else None,
                                   'plane_rms_mm': fit.rms * 1000 if fit else None})
        report['groups'][group] = {mode: {'summary': summarize(rows[mode]), 'rows': rows[mode]} for mode in MODES}
        for mode in MODES:
            print(f'{group:5s} {mode:8s}', json.dumps(report['groups'][group][mode]['summary']))
    path = out / 'ir/reference-comparison.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
