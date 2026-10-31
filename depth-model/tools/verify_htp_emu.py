#!/usr/bin/env python3
"""Verify all generated stereo probes against the recorded simulator hashes.

python3 tools/stereo_probes.py make PROBES
python3 tools/verify_htp_emu.py recovered/primary PROBES --report REPORT.json
Uses the exported execution_order.json and kernel_calls.json; no SDK or trace
memory is needed. An Emulator's address maps are reused between cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import htp_emu
import sim_trace
from check_kernels import load_inputs
from run_htp_emu import execute, output_names


def verify(recovered, probes, references, cases=None, report=None):
    graph = sim_trace.load_graph(recovered)
    nodes = sim_trace.execution_nodes(graph)
    emu = htp_emu.Emulator(graph, (recovered / 'constants/constant_pool_5.bin').read_bytes(), {})
    names = output_names(recovered)
    results = {'context_sha256': graph['context_sha256'], 'cases': {}}
    for case in cases or references:
        start = time.monotonic()
        outputs = execute(emu, nodes, load_inputs(probes / case))
        hashes = {name + '.raw': hashlib.sha256(outputs[i].tobytes()).hexdigest()
                  for i, name in enumerate(names)}
        equal = hashes == references[case]
        results['cases'][case] = {'equal': equal, 'hashes': hashes, 'seconds': round(time.monotonic() - start, 3)}
        print(f'{case}: {"PASS" if equal else "FAIL"} ({time.monotonic() - start:.1f}s)', flush=True)
        if report:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(results, indent=2) + '\n')
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recovered', type=Path)
    ap.add_argument('probes', type=Path)
    ap.add_argument('--hashes', type=Path, default=Path(__file__).resolve().parents[1] / 'reference/simulator_output_hashes.json')
    ap.add_argument('--cases', nargs='+')
    ap.add_argument('--report', required=True, type=Path)
    args = ap.parse_args()
    refs = json.loads(args.hashes.read_text())['models'][args.recovered.name]
    results = verify(args.recovered, args.probes, refs, args.cases, args.report)
    if not all(r['equal'] for r in results['cases'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
