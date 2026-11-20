#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run CDSP microbenchmarks through the loader + bench payload."""

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

import json
from pathlib import Path
import struct
import sys
import time

HERE = (PATHS.benchmarks)
sys.path.insert(0, str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport
from hexecutor.offline.hvx_protocol import STATUS_OFFSET

CMD = struct.Struct('<8I')
OUT = struct.Struct('<8I')
ARENA_CACHED = 0x88800000
ARENA_UNCACHED = 0xf1800000
DATA = 0x10000  # arena offset of benchmark data


def bench(t, mode, iters, addr=0, nbytes=0, stride=128, threads=3):
    def prepare(seq):
        t.arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
        t.arena[2048:2048 + 96] = bytes(96)
        CMD.pack_into(t.arena, 0, 0x48434e42, seq, mode, iters, addr, nbytes, stride, threads)
    def collect():
        return [OUT.unpack_from(t.arena, 2048 + i * 32) for i in range(3)]
    start = time.monotonic()
    outs = t.invoke(prepare, t.workers_done, collect)
    wall = time.monotonic() - start
    cyc = [o[1] | o[2] << 32 for o in outs[:threads]]
    pk = [o[4] | o[5] << 32 for o in outs[:threads]]
    return {'mode': mode, 'iters': iters, 'addr': hex(addr), 'bytes': nbytes, 'stride': stride,
            'threads': threads, 'wall_ms': wall * 1e3, 'cycles': cyc, 'packets': pk,
            'result': [o[3] for o in outs[:threads]]}


def main():
    build = (PATHS.native / 'build')
    rows = []
    with FrameTransport(build / 'qd_loader.elf', build / sys.argv[1] if len(sys.argv) > 1 else build / 'bench_hvx.elf', 30) as t:
        t.arena[DATA:DATA + 0x400000] = bytes(range(256)) * (0x400000 // 256)
        def run(*a, **k):
            r = bench(t, *a, **k); rows.append(r); print(json.dumps(r), flush=True)
        vtcm = 0x31000000
        run(0, 0)
        run(7, 64, vtcm, 128 << 10, 128, threads=1)             # VTCM hot, 1 thread
        run(7, 64, vtcm, 128 << 10, 128)                        # VTCM, 3 threads (own slices)
        run(7, 64, 0x30000000, 128 << 10, 128, threads=1)       # L2-TCM
        run(3, 64, vtcm, 64 << 10, 4, threads=1)                # scalar from VTCM
        run(3, 64, 0x30000000, 64 << 10, 4, threads=1)          # scalar from L2-TCM
        run(6, 1, ARENA_CACHED + DATA, 1 << 20, threads=1)      # DDR->VTCM copy, 8 in flight
        run(6, 1, ARENA_CACHED + DATA, 1 << 20)                 # 3 threads
        run(6, 1, ARENA_UNCACHED + DATA, 1 << 20, threads=1)
    (build / 'bench-results.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
