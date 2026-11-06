#!/usr/bin/env python3
"""Read a traced simulator run (tools/trace_qnn_sim.py) back into tensors.

The trace hook logs, for every executed run-list entry, the VTCM blocks it
changed. This module maps run-list entries to the serialized graph nodes,
replays the writes in execution order, and decodes each node's output tensors
from their block tables, giving the exact simulator value of every VTCM tensor
right after its producer ran.

Run-list mapping (checked against kernel types for every entry):
  list 0 (main thread: HMX convs, DMA, sync): nodes with flags 0 or 2 in serial
          order, with each flags-6 node taking the next slot whose function is
          the InputSlicePad stub.
  list 1 (HVX worker): nodes with flags 4 or 6 in serial order.
Flags-1 nodes (Const, Shape, @DummyOp*) do not execute.

Layouts (QNN/HTP/core/memory_layout.h, tens/tensor_definitions.h):
  C?B Crouton_8:  2 KB chunks of h8 x w8 x d32, chunk offset (h, w, d)
  C?H Crouton_16: 2 KB chunks of h8 x w4 x d32 as (h, w//2, d, w%2)
  F?? flat:       row-major over the padded maximum dimensions
Needs NumPy (python3.13/3.14 here).
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import gzip
import json
import struct
from pathlib import Path

import numpy as np

BLOCK = 2048
VTCM_BYTES = 8 << 20
PAD_STUB = '0xebc680'  # InputSlicePad entry on the main list (2.50 x86 offset)
DTYPES = {'uint8': np.uint8, 'uint16': np.uint16, 'int16': np.int16, 'int8': np.int8,
          'int32': np.int32, 'float32': np.float32, 'int32_quantized': np.int32}


def load_graph(recovered):
    recovered = Path(recovered)
    path = recovered / 'graph.json'
    if path.exists():
        g = json.loads(path.read_text())
    else:
        g = json.loads(gzip.decompress((recovered / 'graph.json.gz').read_bytes()))
    g['_recovered'] = str(recovered)
    return g


def read_trace(trace_dir):
    """Return [(seq, list, idx, fn, nblocks)] in execution order."""
    rows = []
    for line in (Path(trace_dir) / 'trace.log').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        seq, lst, idx, fn, _tid, n = line.split()
        rows.append((int(seq), int(lst), int(idx), fn, int(n)))
    rows.sort()
    return rows


def map_entries(graph, rows):
    """Map (list, idx) -> serialized node, checking that every slot is filled."""
    nodes = graph['nodes']
    main = iter([n for n in nodes if n['flags'] in (0, 2)])
    pads = iter([n for n in nodes if n['flags'] == 6])
    worker = [n for n in nodes if n['flags'] in (4, 6)]
    fn_of = {(r[1], r[2]): r[3] for r in rows}
    out = {}
    for idx in range(sum(1 for k in fn_of if k[0] == 0)):
        if fn_of[(0, idx)] == PAD_STUB:
            # The alternate also has six executor-only no-op entries after its
            # flags-6 input-slice stubs. They consume no serialized operation.
            out[(0, idx)] = next(pads, {'sequence': -idx - 1, 'type_name': '@SyncOp',
                                      'flags': 2, 'inputs': [], 'outputs': [],
                                      'executor_only': True})
        else:
            out[(0, idx)] = next(main)
    for idx, n in enumerate(worker):
        out[(1, idx)] = n
    if len(out) != len(fn_of):
        raise ValueError(f'mapped {len(out)} entries, trace has {len(fn_of)}')
    if next(main, None) is not None or next(pads, None) is not None:
        raise ValueError('trace did not consume every main-list operation')
    known_path = (PATHS.model_data / 'sim/qairt_2_50_functions.json')
    if known_path.exists():
        known = json.loads(known_path.read_text())
        for key, node in out.items():
            fn = fn_of[key]
            if fn != PAD_STUB and fn in known and node['type_name'] not in known[fn]:
                raise ValueError(f'run-list {key}: {fn} is {known[fn]}, mapped to {node["type_name"]}')
    return out


def read_writes(trace_dir):
    rec = np.dtype([('seq', '<u4'), ('list', '<u4'), ('idx', '<u4'), ('block', '<u4'),
                    ('data', 'u1', BLOCK)])
    return np.fromfile(Path(trace_dir) / 'writes.bin', dtype=rec)


def read_tensor_writes(trace_dir):
    """Direct DDR output snapshots: {trace sequence: {tensor id: raw bytes}}."""
    path = Path(trace_dir) / 'tensors.bin'
    out = {}
    if not path.exists():
        return out
    with path.open('rb') as f:
        while header := f.read(12):
            if len(header) != 12:
                raise ValueError('truncated tensor snapshot header')
            seq, tid, size = struct.unpack('<III', header)
            data = f.read(size)
            if len(data) != size:
                raise ValueError(f'truncated tensor snapshot for {tid}')
            out.setdefault(seq, {})[tid] = np.frombuffer(data, np.uint8)
    return out


def _padded(shape):
    return [m + p for m, p in zip(shape['max_dimensions'], shape['padding'])]


def decode(tensor, mem):
    """Decode a VTCM tensor's logical [b, h, w, d] values from memory `mem`."""
    cls, shape, store = tensor['class_name'], tensor.get('shape'), tensor.get('storage') or {}
    dt = np.dtype(DTYPES[tensor['dtype']])
    dims = shape['dimensions']
    pad = shape['padding']
    if store.get('direct'):
        d = store['direct']
        if d['pool'] != 1:
            return None
        full = _padded(shape)
        n = int(np.prod(full)) * dt.itemsize
        arr = mem[d['offset']:d['offset'] + n].view(dt).reshape(full)
        idx = tuple(slice(p, p + s) for p, s in zip(pad, dims))
        return arr[idx].copy()
    blocks = store.get('blocks')
    if not blocks or cls[0] not in 'Cc':
        return None
    if dt.itemsize == 1:
        chunk = (8, 8, 32)
    elif dt.itemsize == 2:
        chunk = (8, 4, 32)
    else:
        raise ValueError(f'unsupported crouton element size for {cls}')
    full = _padded(shape)
    nb = [full[0], -(-full[1] // chunk[0]), -(-full[2] // chunk[1]), -(-full[3] // chunk[2])]
    start = store.get('slice_offset', 0)
    out = np.zeros([nb[0], nb[1] * chunk[0], nb[2] * chunk[1], nb[3] * chunk[2]], dt)
    k = start
    for b in range(nb[0]):
        for bh in range(nb[1]):
            for bw in range(nb[2]):
                for bd in range(nb[3]):
                    ent = blocks[k] if k < len(blocks) else None
                    k += 1
                    if not ent or ent.get('pool') != 1:
                        continue
                    raw = mem[ent['offset']:ent['offset'] + BLOCK].view(dt)
                    if dt.itemsize == 1:
                        tile = raw.reshape(8, 8, 32)
                    else:
                        tile = raw.reshape(8, 2, 32, 2).transpose(0, 1, 3, 2).reshape(8, 4, 32)
                    out[b, bh * 8:(bh + 1) * 8, bw * chunk[1]:(bw + 1) * chunk[1],
                        bd * 32:(bd + 1) * 32] = tile
    return out[:, pad[1]:pad[1] + dims[1], pad[2]:pad[2] + dims[2], pad[3]:pad[3] + dims[3]].copy()


def replay(graph, trace_dir, want=None):
    """Yield (node, {tensor_id: array}) after each executed node, in order.

    `want(node)` filters which nodes' outputs are decoded (default: all).
    """
    rows = read_trace(trace_dir)
    nodes = map_entries(graph, rows)
    tensors = {t['id']: t for t in graph['tensors']}
    writes = read_writes(trace_dir)
    mem = np.zeros(VTCM_BYTES, np.uint8)
    pos = 0
    for seq, lst, idx, _fn, _n in rows:
        while pos < len(writes) and writes[pos]['seq'] == seq:
            b = int(writes[pos]['block'])
            mem[b * BLOCK:(b + 1) * BLOCK] = writes[pos]['data']
            pos += 1
        node = nodes[(lst, idx)]
        if want is not None and not want(node):
            continue
        outs = {}
        for tid in node['outputs']:
            t = tensors.get(tid)
            if t and t.get('shape'):
                v = decode(t, mem)
                if v is not None:
                    outs[tid] = v
        yield node, outs


def read_calls(graph, trace_dir):
    """{node sequence: [(kernel, regs, stack)]} from calls.log, pointers as 'P'.

    Integer arguments are compile-time constants of the graph, except values an
    op reads from a tensor at run time (e.g. the broadcast scalar of
    elementwise_*1111_*); kernels re-read those from memory.
    """
    rows = read_trace(trace_dir)
    nodes = map_entries(graph, rows)
    by_seq = {r[0]: nodes[(r[1], r[2])]['sequence'] for r in rows}
    out = {}
    for line in (Path(trace_dir) / 'calls.log').read_text().splitlines():
        f = line.split()
        i = f.index('|')
        j = f.index('|', i + 1)
        conv = lambda v: 'P' if v.startswith('v+') or (len(v) >= 12 and v.startswith('7ff')) else int(v, 16)
        out.setdefault(by_seq[int(f[0])], []).append((f[1], [conv(v) for v in f[2:i]],
                                                      [conv(v) for v in f[i + 1:j]]))
    return out


def export_execution_plan(recovered, trace_dir):
    """Keep only node identities and compiled arguments, never memory snapshots."""
    recovered = Path(recovered)
    graph = load_graph(recovered)
    rows = read_trace(trace_dir)
    mapping = map_entries(graph, rows)
    plan = {'context_sha256': graph['context_sha256'],
            'nodes': [mapping[r[1:3]]['sequence'] for r in rows if mapping[r[1:3]]['sequence'] >= 0]}
    (recovered / 'execution_order.json').write_text(json.dumps(plan, indent=2) + '\n')
    calls = {'context_sha256': graph['context_sha256'], 'calls': read_calls(graph, trace_dir)}
    (recovered / 'kernel_calls.json').write_text(json.dumps(calls, indent=2) + '\n')
    return plan


def execution_nodes(graph, trace_dir=None):
    if trace_dir is not None:
        rows = read_trace(trace_dir)
        mapping = map_entries(graph, rows)
        return [mapping[r[1:3]] for r in rows]
    plan = json.loads((Path(graph['_recovered']) / 'execution_order.json').read_text())
    if plan['context_sha256'] != graph['context_sha256']:
        raise ValueError('execution order belongs to a different context')
    nodes = {n['sequence']: n for n in graph['nodes']}
    return [nodes[s] for s in plan['nodes']]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Export a trace schedule and compiled arguments for standalone inference.')
    ap.add_argument('recovered', type=Path)
    ap.add_argument('trace', type=Path)
    args = ap.parse_args()
    plan = export_execution_plan(args.recovered, args.trace)
    print(f'Exported {len(plan["nodes"])} entries to {args.recovered}')
