#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact-output acceptance for the native whole-graph executor (qd_graph.c).

Runs the 25 recorded simulator probes, the smoke case and the seed-1234 random
case for each model, comparing both output heads with the reference hashes.
--backend cpu needs no hardware; --backend dsp drives Frame's persistent
payload. --step N compares every pool with the Python emulator after each
command of probe N to localize a mismatch (CPU backend only).
"""

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

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # idle BLAS workers spin on the cores we use
import numpy as np

ROOT = (PATHS.repo)
sys.path[:0] = [str((PATHS.model_tools)), str((PATHS.native))]
from check_kernels import load_inputs
from depth_model.adapters import NativeGraph
from run_htp_emu import output_names


def ensure_image(model):
    image = (PATHS.native / f'build/resident-{model}.qdrx')
    if not image.exists():
        import subprocess
        subprocess.run([sys.executable, str((PATHS.repo / 'samples/depth-model/python/depth_model/lowering/resident_image.py')), str((PATHS.model_data / 'recovered') / model),
                        '--output', str(image)], check=True, stdout=subprocess.DEVNULL)
    return image


def cases(model, only=None, expect='emulator'):
    """expect='hw': the silicon-rounding oracle (recovered/<model>/hw_*.json,
    tools/make_hw_oracle.py) instead of the emulator hashes; smoke and random only."""
    from stereo_probes import make
    probes = (PATHS.native / 'build/native-probes')
    if not (probes / 'same' / 'x.raw').exists():
        make(probes.resolve())
    refs = json.loads(((PATHS.model_data / 'reference/simulator_output_hashes.json')).read_text())['models'][model]
    recovered = (PATHS.model_data / 'recovered') / model
    names = output_names(recovered)
    smoke = (PATHS.native / 'build/native-smoke')
    if not (smoke / 'x.raw').exists():
        import subprocess
        subprocess.run([sys.executable, str((PATHS.model_tools / 'prepare_inputs.py')), str(smoke), '--smoke'],
                       check=True, stdout=subprocess.DEVNULL)
    smoke_ref = json.loads((recovered / f'{expect}_smoke.json').read_text())
    result = [('smoke', smoke, {n.removesuffix('.raw') + '.raw': v['sha256'] for n, v in smoke_ref.items()})]
    if expect == 'emulator':
        result += [(case, probes / case, expected) for case, expected in refs.items()]
    random = (PATHS.native / 'build/depth-inputs/random')
    if random.exists():
        expected = json.loads((recovered / f'{expect}_random.json').read_text())
        result.append(('random', random, {n + '.raw': expected[n]['sha256'] for n in names}))
    if only:
        result = [c for c in result if c[0] in only]
    return names, result


def step_compare(model, graph, case_dir):
    """Run command by command beside the Python emulator; report the first divergence."""
    import sim_trace
    from depth_model.reference.native_reference import NativeEmulator
    recovered = (PATHS.model_data / 'recovered') / model
    g = sim_trace.load_graph(recovered)
    nodes = sim_trace.execution_nodes(g)
    inputs = load_inputs(case_dir)
    emu = NativeEmulator(g, (recovered / 'constants/constant_pool_5.bin').read_bytes(), inputs)
    for p, data in emu.mem.pools.items():
        if p != 5: data.fill(0)
    emu.inputs, emu.outputs, emu.executed = inputs, {}, set()
    key = {1: 1, 3: 3, 4: 4, 5: 5, 'spill': 6}
    graph.run(inputs, 0, 0)
    assert len(nodes) == graph.commands
    for j, node in enumerate(nodes):
        emu.run_node(node); emu.executed.add(node['sequence'])
        graph.run(inputs, j, j + 1)
        for p, data in emu.mem.pools.items():
            mine = graph.pools[key[p]][:data.size]
            if not np.array_equal(mine, data):
                bad = np.flatnonzero(mine != data)
                print(f'command {j} seq {node["sequence"]} {node["type_name"]}: pool {p} differs at '
                      f'{bad.size} bytes, first {bad[:8]} mine={mine[bad[:8]]} ref={data[bad[:8]]}')
                return False
        if j % 500 == 0:
            print(f'{j}/{len(nodes)} identical', flush=True)
    print('all commands identical')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=('cpu', 'dsp'), default='cpu')
    ap.add_argument('--model', choices=('primary', 'alternate'), action='append')
    ap.add_argument('--case', action='append', help='only these cases')
    ap.add_argument('--repeat', type=int, default=1, help='extra timed smoke repetitions per model')
    ap.add_argument('--step', help='localize: compare pools after every command of this case')
    ap.add_argument('--no-hmx', action='store_true')
    ap.add_argument('--payload',type=Path,help='explicit experimental payload; default remains exact qd_conv.elf')
    ap.add_argument('--no-clock', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--no-hoist', action='store_true')
    ap.add_argument('--block-output', action='store_true', help='experimental direct HMX crouton output')
    ap.add_argument('--cpu', type=int, default=7)
    ap.add_argument('--helpers', default='', help='comma-separated cores for arena copy helper threads')
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    if args.cpu >= 0 and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {args.cpu})
    report = {'backend': args.backend, 'block_output': args.block_output, 'models': {}}
    for model in args.model or ['primary', 'alternate']:
        recovered = (PATHS.model_data / 'recovered') / model
        graph = NativeGraph(ensure_image(model), recovered / 'constants/constant_pool_5.bin')
        names, todo = cases(model, args.case)
        if args.step:
            case_dir = dict((c[0], c[1]) for c in cases(model)[1])[args.step]
            ok = step_compare(model, graph, case_dir)
            raise SystemExit(0 if ok else 1)
        transport = None
        if args.backend == 'dsp':
            from hexecutor.offline.frame_transport import FrameTransport
            import signal
            def interrupt(_s, _f): raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupt)
            transport = FrameTransport((PATHS.native / 'build/qd_loader.elf'), args.payload or (PATHS.native / 'build/qd_conv.elf'), 10,
                                       raise_clock=not args.no_clock, use_hmx=not args.no_hmx)
            transport.__enter__()
        try:
            if args.helpers:
                graph.set_helpers([int(c) for c in args.helpers.split(',')])
            if transport: graph.use_dsp(transport)
            else: graph.use_cpu()
            graph.set_profile(args.profile)
            graph.set_hoist(not args.no_hoist)
            graph.set_block_output(args.block_output)
            results = {}
            for case, directory, expected in todo:
                inputs = load_inputs(directory)
                graph.reset_stats()
                begin = time.monotonic()
                outs = graph.run(inputs)
                seconds = time.monotonic() - begin
                hashes = {n + '.raw': hashlib.sha256(outs[i].tobytes()).hexdigest() for i, n in enumerate(names)}
                equal = hashes == expected
                s = graph.stats()
                results[case] = {'equal': equal, 'seconds': seconds, 'dsp_calls': s.calls, 'hmx_calls': s.hmx_calls}
                print(model, case, 'PASS' if equal else 'FAIL', f'{seconds:.4f}s', flush=True)
                if not equal:
                    results[case]['hashes'] = hashes
            timings = []
            smoke = [c for c in cases(model, ['smoke'])[1]][0]
            inputs = load_inputs(smoke[1])
            for _ in range(args.repeat):
                graph.reset_stats()
                begin = time.monotonic()
                outs = graph.run(inputs)
                seconds = time.monotonic() - begin
                hashes = {n + '.raw': hashlib.sha256(outs[i].tobytes()).hexdigest() for i, n in enumerate(names)}
                s = graph.stats()
                run = {'seconds': seconds, 'equal': hashes == smoke[2], 'dsp_calls': s.calls, 'hmx_calls': s.hmx_calls,
                       'prepare_ms': s.prepare_ns / 1e6, 'wait_ms': s.wait_ns / 1e6, 'collect_ms': s.collect_ns / 1e6,
                       'conv_ms': s.conv_ns / 1e6, 'zero_ms': s.zero_ns / 1e6, 'bytes_uploaded': s.bytes_uploaded,
                       'hmx_phase_cycles': list(s.hmx_phase_cycles),
                       'bytes': {k: list(getattr(s, k)) for k in ('bytes_x', 'bytes_w', 'bytes_out')},
                       'weights_reused': s.weights_reused, 'hmx_block_calls': s.hmx_block_calls,
                       'hvx_phase_cycles': [list(r) for r in s.hvx_phase_cycles]}
                if args.profile:
                    import hexecutor.program.resident_schema as resident_schema
                    run['variants'] = sorted(({'variant': resident_schema.VARIANTS[v - 1], 'ms': s.variant_ns[v] / 1e6,
                                               'count': s.variant_count[v]} for v in range(1, 75) if s.variant_count[v]),
                                             key=lambda r: -r['ms'])
                    run['hazard_drains'] = {'count': s.variant_count[0], 'ms': s.variant_ns[0] / 1e6, 'hoisted': s.variant_count[78], 'no_conv': s.variant_count[75],
                                              'input_busy': s.variant_count[76], 'input_conflict': s.variant_count[77], 'output_conflict': s.variant_count[79],
                                              'conflict_distance': (s.variant_ns[77], s.variant_ns[79])}
                    run['variants'] = sorted(run['variants'],
                                             key=lambda r: -r['ms'])
                timings.append(run)
                print('TIMING', json.dumps({k: v for k, v in run.items() if k != 'variants'}), flush=True)
            if args.profile and timings:
                for row in timings[-1]['variants'][:30]:
                    print(f"{row['ms']:9.3f} ms {row['count']:6d} {row['variant']}")
            report['models'][model] = {'cases': results, 'timings': timings,
                                       'all_equal': all(r['equal'] for r in results.values()) and all(t['equal'] for t in timings)}
        finally:
            if transport:
                transport.stop_persistent()
                transport.__exit__(None, None, None)
            graph.close()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    ok = all(m['all_equal'] for m in report['models'].values())
    print('ALL PASS' if ok else 'FAILURES', flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
