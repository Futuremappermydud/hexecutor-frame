#!/usr/bin/env python3
"""Run the packaged stereo graph with Frame HVX convolution or local packet tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import signal
import os
import struct
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(Path(__file__).resolve().parent)]

import htp_emu
import sim_trace
from check_kernels import load_inputs
from run_htp_emu import execute, output_names
from frame_hvx import HvxConvolution
from local_transport import LocalTransport


class CompactMemory(htp_emu.Memory):
    """The recovered arenas fit 32-bit offsets and 8-bit pool identifiers."""
    def addr(self, tid, full=False, lanes=False):
        key = tid, full, lanes
        if key not in self._addr:
            pools, offsets = htp_emu.element_offsets(self.tensors[tid], full, lanes)
            if np.max(pools, initial=0) > 255 or np.min(offsets, initial=0) < 0 or np.max(offsets, initial=0) >= 2**32 - 4:
                raise ValueError('model exceeds the compact address representation')
            self._addr[key] = pools.astype(np.uint8), offsets.astype(np.uint32)
        return self._addr[key]


def verify_package():
    path = ROOT / 'depth-package.json'
    allowed = {'depth-model-direct-v3', 'depth-model-direct-v4', 'depth-model-direct-v5'}
    if not path.exists():
        path = ROOT / 'frame-package.json'
        allowed = {'depth-model-frame-v1'}
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    if manifest['format'] not in allowed:
        raise ValueError('unsupported Frame package format')
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'package checksum failed: {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('primary', 'alternate'), default='primary')
    ap.add_argument('--inputs', type=Path)
    ap.add_argument('--smoke', action='store_true', help='use deterministic smoke input and verify its recorded hashes')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--backend', choices=('hvx', 'packed-cpu', 'native-test', 'numpy'), default='hvx')
    ap.add_argument('--callback', type=Path, default=ROOT / 'frame/build/qd_loader.elf')
    ap.add_argument('--payload', type=Path, default=ROOT / 'frame/build/qd_conv.elf')
    ap.add_argument('--test-library', type=Path, default=ROOT / 'frame/build/libqd_conv_host.so')
    ap.add_argument('--verify-matmuls', action='store_true', help='compare each returned dot product with int64 CPU arithmetic')
    ap.add_argument('--reference-hashes', type=Path, help='compare both outputs with an emulator report containing sha256 fields')
    ap.add_argument('--compare', type=Path, help='directory of simulator/reference native outputs')
    ap.add_argument('--timeout', type=float, default=120, help='seconds per DSP request')
    ap.add_argument('--compute-engine', choices=('hvx','hmx'), default='hvx', help='exact vector convolution or integer matrix-engine convolution')
    ap.add_argument('--cpu', type=int, default=7, help='ARM core for the Frame host graph; -1 preserves current affinity')
    ap.add_argument('--graph-engine', choices=('native','reference'), default='native', help='native tensor access and integer requantization, or original NumPy reference')
    ap.add_argument('--repeat', type=int, default=1, help='repeat in one session to measure warm graph execution')
    ap.add_argument('--no-clock', action='store_true', help='leave the AXI-master clock at its original setting for comparison')
    args = ap.parse_args()
    if args.compute_engine == 'hmx' and (args.backend != 'hvx' or args.graph_engine != 'native'):
        ap.error('HMX requires the hardware backend and native graph engine')
    if args.repeat < 1:
        ap.error('--repeat must be positive')
    if args.smoke == bool(args.inputs):
        ap.error('choose --smoke or --inputs DIR')
    if args.backend == 'hvx' and args.cpu >= 0:
        os.sched_setaffinity(0, {args.cpu})
    verify_package()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        import subprocess
        d = args.output / 'smoke-inputs'
        subprocess.run([sys.executable, str(ROOT / 'tools/prepare_inputs.py'), str(d), '--smoke'], check=True)
        args.inputs = d
    recovered = ROOT / 'recovered' / args.model
    graph = sim_trace.load_graph(recovered)
    nodes = sim_trace.execution_nodes(graph)
    inputs = load_inputs(args.inputs)
    expected_sizes = {0: 81920, 1: 81920, 2: 163840, 3: 32}
    if any(inputs[i].size != size for i, size in expected_sizes.items()):
        raise ValueError('input files do not match the recovered native tensor contract')
    constants = (recovered / 'constants/constant_pool_5.bin').read_bytes()
    if args.graph_engine == 'native':
        from fast_depth import NativeEmulator
        emu = NativeEmulator(graph, constants, inputs)
    else:
        emu = htp_emu.Emulator(graph, constants, inputs)
        emu.mem = CompactMemory(graph, constants)
        emu.t = emu.mem.tensors
    transport = backend = None
    if args.backend == 'hvx':
        from frame_transport import FrameTransport
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        transport = FrameTransport(args.callback, args.payload, args.timeout, raise_clock=not args.no_clock, use_hmx=args.compute_engine == 'hmx')
    elif args.backend == 'native-test':
        transport = LocalTransport(conv_library=args.test_library)
    elif args.backend == 'packed-cpu':
        transport = LocalTransport()
    expected = (json.loads(args.reference_hashes.read_text()) if args.reference_hashes else
                json.loads((recovered / 'emulator_smoke.json').read_text()) if args.smoke else {})
    names = output_names(recovered)
    runs = []
    def run_once():
        begin = time.monotonic()
        counters = {k: getattr(transport, k, 0) for k in
                    ('calls', 'dispatch_seconds', 'prepare_seconds', 'wait_seconds', 'collect_seconds')}
        calls = backend.calls if backend else 0
        execute(emu, nodes, inputs, progress=True)
        hashes = {name + '.raw': hashlib.sha256(emu.outputs[i].tobytes()).hexdigest()
                  for i, name in enumerate(names)}
        run = {'seconds': time.monotonic() - begin, 'hashes': hashes,
               'convolutions': (backend.calls - calls) if backend else 0}
        if expected:
            run['equal'] = all(hashes[name + '.raw'] == expected.get(name, expected.get(name + '.raw'))['sha256']
                               for name in names)
        if args.compare:
            run['equal'] = run.get('equal', True) and all(emu.outputs[i].tobytes() == (args.compare / (name + '.raw')).read_bytes()
                               for i, name in enumerate(names))
        if transport:
            run.update({k: getattr(transport, k, 0) - v for k, v in counters.items()})
        runs.append(run)
        print('INFERENCE ' + json.dumps(run), flush=True)
        if not run.get('equal', True):
            raise RuntimeError('full-model output differs from reference')

    clock = hmx_power = None
    start = time.monotonic()
    if transport:
        with transport:
            backend = HvxConvolution(transport, verify=args.verify_matmuls, direct=args.backend != 'packed-cpu')
            emu.conv_backend = backend
            for _ in range(args.repeat):
                run_once()
            if args.backend == 'hvx':
                transport.stop_persistent()
                hw=struct.unpack_from('<43I',transport.arena,512)
                hmx_power={'active_after_exit':hw[0],'error':hw[41],'restored':hw[42]}
                if transport.hmx_calls and (hw[0] or hw[41] or not hw[42]):
                    raise RuntimeError(f'HMX power cleanup failed: {hmx_power}')
                words = struct.unpack_from('<23I', transport.arena, 256)
                clock = {'active_marker_after_exit': words[0], 'cfg_before': hex(words[1]),
                         'cfg_raised': hex(words[18]), 'cfg_restored': hex(words[20]),
                         'error': words[21], 'command_error': words[22]}
                if words[0] or words[21] or words[22]:
                    raise RuntimeError(f'AXI clock cleanup failed: {clock}')
    else:
        for _ in range(args.repeat):
            run_once()
    report = {'model': args.model, 'context_sha256': graph['context_sha256'], 'backend': args.backend,
              'hardware_executed': args.backend == 'hvx', 'seconds': time.monotonic() - start,
              'numpy_version': np.__version__, 'graph_engine': args.graph_engine, 'compute_engine': args.compute_engine,
              'host_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'runs': runs, 'clock': clock, 'hmx_power': hmx_power,
              'convolutions': backend.calls if backend else 0,
              'quantized_convolutions': backend.quantized_calls if backend else 0,
              'integer_macs': backend.macs if backend else 0, 'outputs': {}}
    report['address_cache_bytes'] = sum(p.nbytes + o.nbytes for p, o in emu.mem._addr.values())
    if backend and backend.calls != 1036 * args.repeat:
        raise RuntimeError(f'expected {1036 * args.repeat} HVX convolutions, executed {backend.calls}')
    if args.backend == 'hvx':
        report.update(callback_sha256=transport.callback_sha256, payload_sha256=transport.payload_sha256,
                      hmx_phase_cycles=transport.hmx_phase_cycles, hmx_convolutions=transport.hmx_calls, dispatch_seconds=transport.dispatch_seconds,
                      bytes_uploaded=transport.bytes_uploaded,
                      prepare_seconds=transport.prepare_seconds, wait_seconds=transport.wait_seconds,
                      collect_seconds=transport.collect_seconds)
    for i, name in enumerate(output_names(recovered)):
        data = emu.outputs[i].tobytes()
        path = args.output / f'{name}.raw'
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        item = {'sha256': sha, 'bytes': len(data)}
        if expected:
            reference = expected.get(name, expected.get(name + '.raw'))
            item['equal'] = sha == reference.get('reference_sha256', reference['sha256'])
        if args.compare:
            item['equal'] = item.get('equal', True) and data == (args.compare / path.name).read_bytes()
        report['outputs'][path.name] = item
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(v.get('equal', True) for v in report['outputs'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
