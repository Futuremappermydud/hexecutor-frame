#!/usr/bin/env python3
"""Run CDSP microbenchmarks through the loader + bench payload."""
import json
from pathlib import Path
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from frame_transport import FrameTransport
from hvx_protocol import STATUS_OFFSET

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
    build = HERE.parent / 'build'
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
