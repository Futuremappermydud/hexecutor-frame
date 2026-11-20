# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-family comparison table of prepared-reference A/B reports (bench/ab.sh).

    python3 frame/bench/ab_table.py DIR NAME...

Reads DIR/NAME_primary.json (and NAME_alternate.json when present). Steady
state is the mean dsp_ms of the primary smoke runs after the first (the first
run pays the one-time preflight); per-family numbers come from the last
primary smoke run.
"""
import json
import sys
from pathlib import Path

CONV = 6


def load(d, n, m):
    p = Path(d) / f'{n}_{m}.json'
    return json.loads(p.read_text()) if p.exists() else None


def steady(rep):
    runs = [r for r in rep['runs'] if r['case'] == 'smoke']
    later = [r['dsp_ms'] for r in runs if r['run'] > 0] or [r['dsp_ms'] for r in runs]
    return sum(later) / len(later), runs[-1]


def main():
    d, names = sys.argv[1], sys.argv[2:]
    cols = {}
    for n in names:
        p = load(d, n, 'primary')
        if not p:
            print(f'{n}: no primary report');
            continue
        a = load(d, n, 'alternate')
        ms, last = steady(p)
        cols[n] = (p, a, ms, last)
    if not cols:
        return
    w = max(10, *(len(n) + 2 for n in cols))
    hdr = lambda t: print(f'{t:<38}' + ''.join(f'{n:>{w}}' for n in cols))
    row = lambda t, f: print(f'{t:<38}' + ''.join(f'{f(*cols[n]):>{w}}' for n in cols))
    hdr('')
    row('passed (primary/alternate)', lambda p, a, ms, l: ('ok' if p['passed'] else 'FAIL') + '/' +
        ('-' if a is None else 'ok' if a['passed'] else 'FAIL'))
    row('steady ms (primary smoke)', lambda p, a, ms, l: f'{ms:.1f}')
    row('first run ms', lambda p, a, ms, l: f"{p['runs'][0]['dsp_ms']:.1f}")
    row('alternate steady ms', lambda p, a, ms, l: '-' if a is None else f'{steady(a)[0]:.1f}')
    row('handler run ms', lambda p, a, ms, l: f"{l['executor_profile'].get('run_ms', 0):.1f}")
    row('preflight / dispatch ms', lambda p, a, ms, l: f"{l['executor_profile'].get('preflight_ms', 0):.0f}/"
        f"{l['executor_profile'].get('dispatch_ms', 0):.1f}")
    row('vtcm in / out ms', lambda p, a, ms, l: f"{l.get('read_ms', 0):.1f}/{l.get('write_ms', 0):.1f}")
    for kind in ('u8', 'u16'):
        for ph in ('staging', 'mac', 'setup', 'output'):
            row(f'conv {kind} {ph} (cpu, all parts)',
                lambda p, a, ms, l, k=kind, h=ph: f"{l.get(f'near_conv_phases_{k}_ms', {}).get(h, float('nan')):.1f}")
    # Families.
    def hp(l):
        return {f"{h['opcode']}/{h['variant']}": h for h in l['handler_profile']}
    allh = {}
    for n, (p, a, ms, l) in cols.items():
        for k, h in hp(l).items():
            allh[k] = max(allh.get(k, 0), h['run_ms'])
    conv = sorted((k for k in allh if k.startswith(f'{CONV}/')), key=lambda k: -allh[k])
    other = sorted((k for k in allh if not k.startswith(f'{CONV}/')), key=lambda k: -allh[k])
    cell = lambda k: (lambda p, a, ms, l: (lambda h: '-' if not h else f"{h['run_ms']:.1f}")(hp(l).get(k)))
    tot = lambda keys: (lambda p, a, ms, l: f"{sum(h['run_ms'] for k, h in hp(l).items() if k in keys):.1f}")
    print()
    row('conv total', tot(set(conv)))
    for k in conv:
        cnt = next((hp(c[3])[k]['count'] for c in cols.values() if k in hp(c[3])), 0)
        row(f'  conv {k} (x{cnt})', cell(k))
    row('non-conv total', tot(set(other)))
    for k in other[:15]:
        cnt = next((hp(c[3])[k]['count'] for c in cols.values() if k in hp(c[3])), 0)
        row(f'  {k} (x{cnt})', cell(k))
    print()
    pp = lambda l: l.get('parallel_profile') or {}
    for kk in ('conv', 'ew', 'dm'):
        row(f'threads {kk} jobs par/serial/parts', lambda p, a, ms, l, k=kk: '-' if not pp(l) else
            f"{pp(l)['jobs_parallel'][k]}/{pp(l)['jobs_serial'][k]}/{pp(l)['parts_run'][k]}")
    row('threads fault/not_ready/declined', lambda p, a, ms, l: '-' if not pp(l) else
        f"{pp(l)['fault']}/{pp(l)['not_ready']}/{pp(l)['declined']}")


if __name__ == '__main__':
    main()
