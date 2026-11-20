#!/usr/bin/env python3
"""Run the NumPy emulator in a saved simulator trace's execution order.

Only trace.log supplies the schedule: writes.bin and simulator tensors are
never loaded for inference. Kernel constants come from kernel_calls.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import htp_emu
import sim_trace
from check_kernels import load_inputs

def output_names(recovered):
    metadata = recovered.parent.parent / 'metadata' / f'{recovered.name}.qnn.json'
    info = json.loads(metadata.read_text())['info']['graphs'][0]['info']
    return [t['info']['name'] + ('_native' if t['info']['dataType'] != 'QNN_DATATYPE_FLOAT_32' else '')
            for t in info['graphOutputs']]


def execute(emu, nodes, inputs, progress=False):
    """Reset runtime memory for each inference; cached address maps are reusable."""
    for pool, data in emu.mem.pools.items():
        if pool != 5:
            data.fill(0)
    emu.inputs, emu.outputs, emu.executed = inputs, {}, set()
    for j, node in enumerate(nodes):
        emu.run_node(node)
        emu.executed.add(node['sequence'])
        if progress and j % 500 == 0:
            print(f'{j}/{len(nodes)} node {node["sequence"]}: {node["type_name"]}', flush=True)
    return emu.outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recovered', type=Path)
    ap.add_argument('trace', type=Path, nargs='?', help='optional trace.log schedule override')
    ap.add_argument('--inputs', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--compare', type=Path, help='simulator Result_0 directory')
    args = ap.parse_args()
    graph = sim_trace.load_graph(args.recovered)
    nodes = sim_trace.execution_nodes(graph, args.trace)
    emu = htp_emu.Emulator(graph, (args.recovered / 'constants/constant_pool_5.bin').read_bytes(),
                           load_inputs(args.inputs))
    missing = sorted({n['type_name'] for n in nodes if n['type_name'] not in htp_emu.KERNELS})
    if missing:
        raise SystemExit('Unimplemented kernels: ' + ', '.join(missing))
    execute(emu, nodes, emu.inputs, progress=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for idx, name in enumerate(output_names(args.recovered)):
        data = emu.outputs[idx].tobytes()
        (args.output / f'{name}.raw').write_bytes(data)
        result = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        if args.compare:
            ref = (args.compare / f'{name}.raw').read_bytes()
            expected = np.frombuffer(ref, emu.outputs[idx].dtype)
            actual = emu.outputs[idx].reshape(-1)
            result.update(equal=data == ref, differing=int(np.count_nonzero(actual != expected)),
                          max_abs_diff=float(np.max(np.abs(actual.astype(float) - expected))))
        report[name] = result
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if args.compare and not all(v['equal'] for v in report.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
