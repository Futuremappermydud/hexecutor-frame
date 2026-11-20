#!/usr/bin/env python3
"""IR refinement must use only pairs that a logged, correctly timed pulse lit."""

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

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str((PATHS.repo / 'depth-runtime/scripts')))
from ir_pulse_sweep import contrast_by_lead, fit_lead
from pull_room_dataset import is_raw_frame
from room_dataset import ir_stereo_input, select_lit_pairs

LEAD_NS = 8_000_000


def cycle(index, status, lit_score, on_offset_ns=-5_000_000, off_offset_ns=5_000_000):
    lit_ts = 1_000_000_000 + index * 200_000_000
    target = lit_ts - LEAD_NS
    pulse = {'cycle': index, 'status': status, 'lead_ms': 8.0, 'on_done_raw_ns': target + on_offset_ns,
             'off_command_raw_ns': target + off_offset_ns}
    pairs = [{'pair': 2 * index + 1, 'cycle': index, 'role': 'reference', 'dot_score': 5.0, 'right_dot_score': 5.0,
              'left_raw_ns': lit_ts - 33_333_333},
             {'pair': 2 * index + 2, 'cycle': index, 'role': 'lit', 'dot_score': lit_score, 'right_dot_score': lit_score,
              'left_raw_ns': lit_ts}]
    return pulse, pairs


def build(rows):
    pulses, pairs = [], []
    for index, row in enumerate(rows):
        pulse, entries = cycle(index, *row)
        pulses.append(pulse)
        pairs += entries
    return pairs, pulses


def main():
    shams = [('sham', 5.0 + 0.01 * (i % 5)) for i in range(20)]
    pairs, pulses = build(shams + [('fired', 6.0), ('fired', 5.0), ('fired', 6.0, 2_000_000), ('late', 6.0)])
    selected, report = select_lit_pairs(pairs, pulses)
    assert [entry['cycle'] for entry in selected] == [20], selected
    assert report['below_threshold'] == 1 and report['mistimed'] == 1 and report['fired'] == 3, report

    # A very strong sham raises the threshold, and motion cannot pass as dots.
    pairs, pulses = build([('sham', 7.0)] * 20 + [('fired', 6.0)])
    assert select_lit_pairs(pairs, pulses)[0] == []

    # A pulse that lit only one camera is not stereo evidence.
    pairs, pulses = build(shams + [('fired', 6.0)])
    pairs[-1]['right_dot_score'] = 5.0
    assert select_lit_pairs(pairs, pulses)[0] == []

    # Too few shams means no measured false-positive rate, so nothing is trusted.
    pairs, pulses = build(shams[:19] + [('fired', 9.0)])
    selected, report = select_lit_pairs(pairs, pulses)
    assert selected == [] and report['status'] == 'insufficient_sham_controls'

    # The sweep finds the centre of the lit leads and refuses a flat response.
    shams_only = [1.0, 1.01, 0.99, 1.0, 1.02, 0.98]
    by_lead = {0.0: 1.0, 2.0: 1.0, 4.0: 1.05, 6.0: 1.2, 8.0: 1.2, 10.0: 1.05, 12.0: 1.0}
    fit = fit_lead(by_lead, shams_only)
    assert fit and fit['lead_ms'] == 7.0 and fit['window_ms'] == [6.0, 8.0] and not fit['peak_at_sweep_edge'], fit
    assert fit_lead({0.0: 1.0, 2.0: 1.01, 4.0: 1.0}, shams_only) is None
    assert fit_lead({0.0: 1.3, 2.0: 1.1, 4.0: 1.0}, shams_only)['peak_at_sweep_edge']

    pairs, pulses = build([('sham', 5.0)] + [('fired', 6.0)])
    assert contrast_by_lead(pairs, pulses) == ({8.0: 1.2}, [1.0])
    # Accepted pairs carry both dark neighbours, and subtraction leaves only the pulse's light.
    pairs, pulses = build(shams + [('fired', 6.0)])
    pairs.append({'pair': 99, 'cycle': 20, 'role': 'after', 'dot_score': 5.0, 'right_dot_score': 5.0, 'left_raw_ns': 0})
    entry = select_lit_pairs(pairs, pulses)[0][0]
    assert [reference['role'] for reference in entry['references']] == ['reference', 'after']
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        (out / 'ir').mkdir()
        for name, value in (('lit', 60), ('before', 50), ('after', 40)):
            for eye in ('left', 'right'):
                np.full((1024, 1056), value, np.uint8).tofile(out / 'ir' / f'{name}-{eye}.u8')
        def files(name):
            return {'pair': 0, 'left_file': f'{name}-left.u8', 'right_file': f'{name}-right.u8'}
        lit = {**files('lit'), 'references': [files('before'), files('after')]}
        assert ir_stereo_input(out, lit, 'average')[0][0, 0] == 15 and ir_stereo_input(out, lit, 'average')[2] == 2
        assert ir_stereo_input(out, lit, 'before')[1][0, 0] == 10 and ir_stereo_input(out, lit, 'before')[2] == 1
        assert ir_stereo_input(out, lit, 'none')[0][0, 0] == 60 and ir_stereo_input(out, lit, 'none')[2] == 0
        lit['references'] = [files('before')]
        assert ir_stereo_input(out, lit, 'average')[0][0, 0] == 10
    # Only raw camera frames stay on the Frame by default.
    assert is_raw_frame('rgb/frame-12-raw-left.nv12') and is_raw_frame('ir/pair-3-right.u8')
    assert not any(map(is_raw_frame, ('rgb/frame-12-rect-left.u8', 'depth/frame-12-depth.f32', 'ir/pairs.jsonl')))
    print('IR pulse selection: PASS')


if __name__ == '__main__':
    main()
