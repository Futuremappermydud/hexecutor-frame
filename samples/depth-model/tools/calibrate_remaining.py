#!/usr/bin/env python3
"""Validate unary arithmetic against direct QAIRT 2.50 simulator helper calls.

Abs covers all 65,536 inputs and both HVX lane parities. Mean covers random
32-element reductions and boundary vectors. Needs sim/build/kcall built with
-rdynamic (see README). No model inference is involved.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import htp_emu
from kcall import Job, run
import sim_trace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recovered', type=Path)
    ap.add_argument('--sdk', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    args = ap.parse_args()
    graph = sim_trace.load_graph(args.recovered)
    emu = htp_emu.Emulator(graph, (args.recovered / 'constants/constant_pool_5.bin').read_bytes(), {})
    jobs, checks, seen = [], [], set()
    for node in graph['nodes']:
        if not node['type_name'].startswith('Abs@'):
            continue
        i, o = node['inputs'][0], node['outputs'][0]
        si, zi = htp_emu.q(emu.t[i]); so, zo = htp_emu.q(emu.t[o])
        ratio = np.float32(si) * np.float32(1 / so)
        mult = int(np.rint(np.float32(ratio * 262144)))
        off = int(np.rint(np.float32((np.float32(ratio * 32768) + zo) * 8)))
        if (mult, zi, off) in seen:
            continue
        seen.add((mult, zi, off))
        x = np.repeat(np.arange(65536, dtype=np.uint16), 2)
        j = Job(0x161c74e, rel=True).buf('x', x.tobytes()).buf('y', x.nbytes)
        j.args('@y', '@x', x.nbytes // 128, mult, zi, off).save('y')
        centered = np.abs(x.astype(np.int64) - zi) - 32768
        rounded = (np.arange(x.size) % 2 == 0) * 16384
        expected = np.clip((((centered * mult + rounded) >> 15) + off + 4) >> 3, 0, 65535)
        jobs.append(j); checks.append((f'Abs m={mult} zi={zi} off={off}', expected))
    rng = np.random.default_rng(4321)
    x = rng.integers(0, 65536, (4096, 32), dtype=np.uint16)
    x[:128] = np.arange(128, dtype=np.uint16)[:, None]
    x[128:256] = (65535 - np.arange(128, dtype=np.uint16))[:, None]
    # Include all residues of the aggregate sum near both range limits.
    x[256:288] = 0; x[256:288, 0] = np.arange(32)
    x[288:320] = 65535; x[288:320, 0] -= np.arange(32, dtype=np.uint16)
    j = Job(0xd4c210, rel=True).buf('x', x.tobytes()).buf('y', x.shape[0] * 128)
    # These internal reduce helpers only access data at tensor-object +24.
    j.ptrs('src', 'x', 0, 4).ptrs('dst', 'y', 0, 4)
    j.args('@dst', '@src', 1, 1, len(x), 32, stack=(1, 0, 0x44000)).save('y')
    jobs.append(j)
    checks.append(('Mean32', np.clip((x.astype(np.int64).sum(1) + 33) >> 5, 0, 65535)))
    results = {}
    for (name, expected), result in zip(checks, run(jobs, args.sdk)):
        actual = np.frombuffer(result['y'], np.uint16)[:expected.size]
        bad = int(np.count_nonzero(actual != expected.reshape(-1)))
        results[name] = {'samples': expected.size, 'differing': bad}
        print(f'{name}: {bad}/{expected.size} differ')
    args.report.write_text(json.dumps(results, indent=2) + '\n')
    if any(r['differing'] for r in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
