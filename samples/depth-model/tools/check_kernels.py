#!/usr/bin/env python3
"""Check htp_emu kernels one node at a time against a traced simulator run.

Before each executed node the emulator's VTCM is set to the simulator's VTCM
(replayed from writes.bin); the kernel runs; its output tensors are compared
with the simulator's VTCM after the node. DDR scratch, spill and outputs are
carried by the emulator itself, so nodes reading them depend on earlier
emulated nodes. Reports, per kernel type, how many nodes match exactly.

  python3.13 samples/depth-model/tools/check_kernels.py samples/depth-model/recovered/primary \
      TRACE_DIR --inputs INPUT_DIR [--types PREFIX ...] [--stop-on-fail]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import htp_emu
import sim_trace

GRAPH_INPUTS = (('x', np.uint8), ('y', np.uint8), ('hidden', np.float32), ('t', np.float32))


DMA = ('ConvLayer.opt.weights_to_vtcm', 'ConvLayer.opt.activations_to_vtcm_dma', '*InputSlice@',
       '@Fill')


def read_from(emu, vtcm, tid):
    saved, emu.mem.pools[1] = emu.mem.pools[1], vtcm
    try:
        return emu.mem.read(tid)
    finally:
        emu.mem.pools[1] = saved


def load_inputs(d):
    return {i: np.fromfile(Path(d) / f'{n}.raw', dt) for i, (n, dt) in enumerate(GRAPH_INPUTS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recovered', type=Path)
    ap.add_argument('trace', type=Path)
    ap.add_argument('--inputs', required=True, type=Path)
    ap.add_argument('--types', nargs='*', help='only report kernel types with these prefixes')
    ap.add_argument('--stop-on-fail', action='store_true')
    ap.add_argument('--strict', action='store_true',
                    help='also compare every VTCM block the simulator wrote, byte for byte')
    ap.add_argument('--show', type=int, default=1, help='failures to describe per type')
    ap.add_argument('--skip-conv', action='store_true', help='skip HMX computation during kernel development')
    ap.add_argument('--report', type=Path, help='write machine-readable per-type counts')
    args = ap.parse_args()

    graph = sim_trace.load_graph(args.recovered)
    pool = (args.recovered / 'constants' / 'constant_pool_5.bin').read_bytes()
    emu = htp_emu.Emulator(graph, pool, load_inputs(args.inputs))
    rows = sim_trace.read_trace(args.trace)
    nodes = sim_trace.map_entries(graph, rows)
    writes = sim_trace.read_writes(args.trace)
    tensor_writes = sim_trace.read_tensor_writes(args.trace)
    if (args.trace / 'tensors.bin').exists():
        for seq, lst, idx, *_ in rows:
            expected = [tid for tid in nodes[lst, idx]['outputs']
                        if (emu.t[tid].get('storage') or {}).get('direct', {}).get('pool') in (3, 4)]
            missing = set(expected) - tensor_writes.get(seq, {}).keys()
            if missing:
                raise ValueError(f'trace sequence {seq} is missing DDR snapshots {sorted(missing)}')
    mem = np.zeros(htp_emu.VTCM_BYTES, np.uint8)
    pos = 0
    stats, shown, pending = defaultdict(Counter), Counter(), {}
    for seq, lst, idx, _fn, _n in rows:
        node = nodes[(lst, idx)]
        kind = node['type_name']
        emu.mem.pools[1] = mem.copy()
        status = 'ok'
        try:
            if args.skip_conv and kind.startswith(('ConvLayer_s1.', 'ConvLayer_s2.')):
                status = 'skipped'
            else:
                emu.run_node(node)
        except NotImplementedError:
            status = 'missing'
        except Exception as e:  # noqa: BLE001 - report and continue
            status = f'error {type(e).__name__}: {e}'
        first = node['sequence'] not in emu.executed
        emu.executed.add(node['sequence'])
        written = []
        while pos < len(writes) and writes[pos]['seq'] == seq:
            b = int(writes[pos]['block'])
            mem[b * 2048:(b + 1) * 2048] = writes[pos]['data']
            written.append(b)
            pos += 1
        for tid, ref in tensor_writes.get(seq, {}).items():
            t = emu.t[tid]
            d = t['storage']['direct']
            mine = emu.mem.pools[d['pool']][d['offset']:d['offset'] + len(ref)]
            if status == 'ok' and not np.array_equal(mine, ref):
                status = f'mismatch DDR tid {tid}: {np.count_nonzero(mine != ref)}/{len(ref)} bytes differ'
                if shown[kind] < args.show:
                    shown[kind] += 1
                    print(f'seq {node["sequence"]} {kind}: {status}')
            # Teacher-force DDR too, so downstream errors remain local.
            mine[:] = ref
        # DMA results land at a later @DmaCheckpointWait: compare them once they appear.
        for key in list(pending):
            tid, mine, kind_p = pending[key]
            if np.array_equal(read_from(emu, mem, tid), mine):
                del pending[key]
                stats[kind_p]['ok'] += 1
                stats[kind_p]['dma-pending'] -= 1
        if status == 'ok' and first:
            for tid in node['outputs']:
                t = emu.t.get(tid)
                if not t or not t.get('shape') or not t.get('storage'):
                    continue
                pools, _ = emu.mem.addr(tid)
                if not (pools == 1).all():
                    continue
                mine = emu.mem.read(tid)
                ref = read_from(emu, mem, tid)
                if not np.array_equal(mine, ref) and kind.startswith(DMA):
                    pending[node['sequence']] = (tid, mine, kind)
                    status = 'dma-pending'
                    break
                if not np.array_equal(mine, ref):
                    diff = mine.astype(np.float64) - ref.astype(np.float64)
                    status = (f'mismatch tid {tid}: {np.count_nonzero(diff)}/{diff.size} differ, '
                              f'max |d| {np.abs(diff).max():g}')
                    if shown[kind] < args.show and (not args.types or kind.startswith(tuple(args.types))):
                        shown[kind] += 1
                        print(f'seq {node["sequence"]} {kind}: {status}')
                    break
        if status == 'ok' and first and args.strict and not kind.startswith(DMA):
            # Byte-level: every block the simulator wrote, padding lanes included.
            bad = [b for b in written
                   if not np.array_equal(emu.mem.pools[1][b * 2048:(b + 1) * 2048], mem[b * 2048:(b + 1) * 2048])]
            if bad:
                status = f'pad-mismatch {len(bad)}/{len(written)} blocks'
                if shown[kind] < args.show:
                    shown[kind] += 1
                    b = bad[0]
                    d = np.nonzero(emu.mem.pools[1][b * 2048:(b + 1) * 2048] != mem[b * 2048:(b + 1) * 2048])[0]
                    print(f'seq {node["sequence"]} {kind}: {status}; block {b} bytes {d[:8].tolist()}...')
        if not first:
            continue
        key = status.split()[0]
        stats[kind][key] += 1
        if status.startswith('error') and shown[kind] < args.show:
            shown[kind] += 1
            print(f'seq {node["sequence"]} {kind}: {status}')
        if args.stop_on_fail and key not in ('ok', 'missing'):
            break
    print()
    for kind, c in sorted(stats.items(), key=lambda kv: -sum(kv[1].values())):
        if args.types and not kind.startswith(tuple(args.types)):
            continue
        print(f'{kind:60s} ' + ' '.join(f'{k}={v}' for k, v in sorted(c.items())))
    if args.report:
        args.report.write_text(json.dumps(dict(stats), indent=2) + '\n')
    if any(count for counts in stats.values() for status, count in counts.items()
           if status not in ('ok', 'skipped')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
