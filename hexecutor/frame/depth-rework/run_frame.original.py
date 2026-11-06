#!/usr/bin/env python3
"""Run the packaged stereo graph with Frame HVX convolution or local packet tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import signal
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
    path = ROOT / 'frame-package.json'
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    if manifest['format'] != 'depth-model-frame-v1':
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
    ap.add_argument('--compare', type=Path, help='directory of simulator/reference native outputs')
    ap.add_argument('--timeout', type=float, default=120, help='seconds per DSP request')
    args = ap.parse_args()
    if args.smoke == bool(args.inputs):
        ap.error('choose --smoke or --inputs DIR')
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
    emu = htp_emu.Emulator(graph, constants, inputs)
    emu.mem = CompactMemory(graph, constants)
    emu.t = emu.mem.tensors
    transport = backend = None
    if args.backend == 'hvx':
        from frame_transport import FrameTransport
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        transport = FrameTransport(args.callback, args.payload, args.timeout)
    elif args.backend == 'native-test':
        transport = LocalTransport(conv_library=args.test_library)
    elif args.backend == 'packed-cpu':
        transport = LocalTransport()
    start = time.monotonic()
    if transport:
        with transport:
            backend = HvxConvolution(transport, verify=args.verify_matmuls, direct=args.backend != 'packed-cpu')
            emu.conv_backend = backend
            execute(emu, nodes, inputs, progress=True)
    else:
        execute(emu, nodes, inputs, progress=True)
    report = {'model': args.model, 'context_sha256': graph['context_sha256'], 'backend': args.backend,
              'hardware_executed': args.backend == 'hvx', 'seconds': time.monotonic() - start,
              'numpy_version': np.__version__,
              'convolutions': backend.calls if backend else 0,
              'integer_macs': backend.macs if backend else 0, 'outputs': {}}
    report['address_cache_bytes'] = sum(p.nbytes + o.nbytes for p, o in emu.mem._addr.values())
    if backend and backend.calls != 1036:
        raise RuntimeError(f'expected 1036 HVX convolutions, executed {backend.calls}')
    if args.backend == 'hvx':
        report.update(callback_sha256=transport.callback_sha256, payload_sha256=transport.payload_sha256,
                      dispatch_seconds=transport.dispatch_seconds,
                      bytes_uploaded=transport.bytes_uploaded)
    expected = json.loads((recovered / 'emulator_smoke.json').read_text()) if args.smoke else {}
    for i, name in enumerate(output_names(recovered)):
        data = emu.outputs[i].tobytes()
        path = args.output / f'{name}.raw'
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        item = {'sha256': sha, 'bytes': len(data)}
        if args.smoke:
            reference = expected.get(name, expected.get(name + '.raw'))
            item['equal'] = sha == reference['reference_sha256']
        if args.compare:
            item['equal'] = data == (args.compare / path.name).read_bytes()
        report['outputs'][path.name] = item
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(v.get('equal', True) for v in report['outputs'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
