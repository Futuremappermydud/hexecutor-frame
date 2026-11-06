#!/usr/bin/env python3
"""Run one staged HMX probe, retaining progress on timeout for diagnosis."""

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

import argparse
import json
from pathlib import Path
import struct
import sys

HERE = (PATHS.benchmarks)
sys.path.insert(0, str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport
from hexecutor.offline.hvx_protocol import STATUS_OFFSET

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=(0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14), default=0)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--case', choices=('random', 'identity', 'zero'), default='random')
parser.add_argument('--iterations', type=int, default=10000)
args = parser.parse_args()
if not 1 <= args.iterations <= 1000000:
    parser.error('--iterations must be between 1 and 1000000')
args.report.parent.mkdir(parents=True, exist_ok=True)
build = (PATHS.native / 'build')
report = {'stage': args.stage}
reference = None
if args.stage in (13, 14):
    import numpy as np
    rng = np.random.default_rng(args.seed)
    a = rng.integers(-3, 4, (32, 32), dtype=np.int32)
    b = rng.integers(-3, 4, (32, 32), dtype=np.int32)
    if args.case == 'identity': b = np.eye(32, dtype=np.int32)
    if args.case == 'zero': b[:] = 0
    reference = a.astype(np.int64) @ b.astype(np.int64)
    report['seed'] = args.seed
    report['case'] = args.case
t = FrameTransport(build / 'qd_loader.elf',
                   build / ('hmx_exec.elf' if args.stage >= 9 else 'hmx_bringup.elf'), 3)
try:
    with t:
        def prepare(seq):
            t.arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
            t.arena[2048:2240] = bytes(192)
            struct.pack_into('<II', t.arena, 8, seq, args.stage)
            if args.stage in (13, 14):
                for off, mat in ((0x20000, a), (0x21000, b)):
                    packed = mat.reshape(16, 2, 32).transpose(0, 2, 1).astype(np.float16).tobytes()
                    t.arena[off:off + len(packed)] = packed
            struct.pack_into('<I', t.arena, 16, args.iterations)
            t.fence()

        def collect():
            report['workers'] = [list(struct.unpack_from('<16I', t.arena, 2048 + 64*i)) for i in range(3)]
            report['status'] = [list(struct.unpack_from('<4I', t.arena, 128 + 16*i)) for i in range(3)]
            report['config'] = list(struct.unpack_from('<48I', t.arena, 0x10000))
            report['tlb'] = list(struct.unpack_from('<128Q', t.arena, 0x11000))
            report['rscc'] = list(struct.unpack_from('<4096I', t.arena, 0x12000))
            if args.stage in (10, 12, 13, 14):
                import numpy as np
                values = np.frombuffer(t.arena, dtype=np.float16, count=1024, offset=0x18000).copy()
                report['matrix_values'] = values.astype(float).tolist()
                if args.stage in (13, 14):
                    matrix = values.reshape(16, 32, 2).transpose(0, 2, 1).reshape(32, 32)
                    report['matrix_matches_reference'] = bool(np.array_equal(matrix, reference))
                    report['matrix_max_abs_error'] = float(np.max(np.abs(matrix.astype(float) - reference)))
                    report['matrix_reference'] = reference.ravel().tolist()
                else:
                    report['matrix_all_32'] = bool(np.all(values == 32))
            report['payload_sha256'] = t.payload_sha256
            if args.stage in (11, 12, 13, 14):
                w = report['workers'][0]
                report['power_cycle_completed'] = bool(w[2] == 0 and w[5] == 0 and w[7] == 32
                                                      and w[8] == 68 and w[15] == 24)
                if args.stage >= 12:
                    report['kernel_cycles'] = w[9] + (w[10] << 32)
                    report['iterations'] = w[11]
                    report['cycles_per_tile'] = report['kernel_cycles'] / w[11] if w[11] else None
            args.report.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: v for k, v in report.items() if k not in ('tlb', 'rscc', 'matrix_values', 'matrix_reference')}), flush=True)

        try:
            t.invoke(prepare, lambda: t.workers_done() and t.workers_exited(), collect)
            report['completed'] = True
        except BaseException as exc:
            report['error'] = str(exc)
            collect()
            raise
finally:
    args.report.write_text(json.dumps(report, indent=2) + '\n')

for check in ('power_cycle_completed', 'matrix_all_32', 'matrix_matches_reference'):
    if report.get(check) is False:
        raise SystemExit(f'HMX verification failed: {check}; see {args.report}')
