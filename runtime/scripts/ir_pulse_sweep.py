#!/usr/bin/env python3
"""Find where the IR exposure sits by sweeping a short BELICE pulse across the frame cycle.

Hold the headset still, facing a matte wall about 1-2 m away with nobody in the beam. The
result is saved beside the sessions, and later captures take their pulse timing from it.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from room_dataset import (BELICE_FLASH, BELICE_PY, FLOODS_AUTO, FLOODS_DISABLED, REPO, set_floods)


def contrast_by_lead(pairs, cycles):
    """Median lit/reference dot contrast for each lead, plus the sham pulses' contrast."""
    roles = {(entry['cycle'], entry['role']): entry for entry in pairs}
    leads, shams = {}, []
    for cycle in cycles:
        lit, reference = roles.get((cycle['cycle'], 'lit')), roles.get((cycle['cycle'], 'reference'))
        if not lit or not reference:
            continue
        value = min(lit['dot_score'] / reference['dot_score'], lit['right_dot_score'] / reference['right_dot_score'])
        if cycle['status'] == 'sham':
            shams.append(value)
        elif cycle['status'] == 'fired':
            leads.setdefault(cycle['lead_ms'], []).append(value)
    return {lead: float(np.median(values)) for lead, values in sorted(leads.items())}, shams


def fit_lead(by_lead, shams):
    """The centre of the leads above half the peak, or None if no lead beats the shams."""
    if len(by_lead) < 3 or len(shams) < 5:
        return None
    floor = float(np.median(shams))
    leads = np.array(list(by_lead))
    excess = np.array(list(by_lead.values())) - floor
    if excess.max() <= float(np.percentile(shams, 95)) - floor:
        return None
    inside = excess >= excess.max() / 2
    lead = float(np.sum(leads[inside] * excess[inside]) / np.sum(excess[inside]))
    return {'lead_ms': round(lead, 1), 'window_ms': [float(leads[inside].min()), float(leads[inside].max())],
            'peak_contrast': float(excess.max() + floor), 'sham_median': floor,
            'peak_at_sweep_edge': bool(np.argmax(excess) in (0, len(excess) - 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, default=REPO / 'experiments/room-dataset')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--pulse-ms', type=float, default=4)
    args = parser.parse_args()
    work = args.root / f'sweep-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
    work.mkdir(parents=True)
    leads = ','.join(str(lead) for lead in range(0, 25, 2))
    command = [str(REPO / 'bin/record_ir_pairs'), str(work), str(int(args.seconds * 1000)), '80', '0',
               '--pulse-every', '6', '--pulse-ms', str(args.pulse_ms), '--lead-ms', leads, '--sham-percent', '20']
    set_floods(FLOODS_DISABLED)
    try:
        subprocess.run(command, check=True)
    finally:
        subprocess.run([str(BELICE_PY), str(BELICE_FLASH), 'emitter', '--seconds', '0'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        set_floods(FLOODS_AUTO)
    pairs = [json.loads(line) for line in (work / 'pairs.jsonl').read_text().splitlines()]
    cycles = [json.loads(line) for line in (work / 'pulses.jsonl').read_text().splitlines()]
    by_lead, shams = contrast_by_lead(pairs, cycles)
    for lead, value in by_lead.items():
        print(f'lead {lead:5.1f} ms  contrast {value:.3f}  {"#" * max(0, int((value - 1) * 200))}')
    if shams:
        print(f'shams: {len(shams)}, median {np.median(shams):.3f}, p95 {np.percentile(shams, 95):.3f}')
    fit = fit_lead(by_lead, shams)
    for path in work.glob('pair-*.u8'):
        path.unlink()
    if fit is None:
        raise SystemExit(f'No lead beat the sham pulses; calibration unchanged. Logs: {work}')
    if fit['peak_at_sweep_edge']:
        raise SystemExit(f'Peak is at the edge of the sweep, so its centre is unknown; calibration unchanged. Fit: {fit}')
    fit.update(pulse_ms=args.pulse_ms, measured_utc=time.strftime('%FT%TZ', time.gmtime()), logs=work.name)
    (args.root / 'ir-pulse-calibration.json').write_text(json.dumps(fit, indent=2) + '\n')
    print(f'Saved lead {fit["lead_ms"]} ms, exposure window about {fit["window_ms"]} ms')


if __name__ == '__main__':
    main()
