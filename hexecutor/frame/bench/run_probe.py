#!/usr/bin/env python3
"""Run a one-shot probe payload once and print its per-worker result words."""
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from frame_transport import FrameTransport
from hvx_protocol import STATUS_OFFSET

build = HERE.parent / 'build'
with FrameTransport(build / 'qd_loader.elf', build / sys.argv[1], 10) as t:
    def prepare(seq):
        t.arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
        t.arena[2048:2048 + 192] = bytes(192)
        struct.pack_into('<I', t.arena, 8, seq)
    words = t.invoke(prepare, t.workers_done,
                     lambda: [struct.unpack_from('<16I', t.arena, 2048 + 64 * i) for i in range(3)])
    for tid, w in enumerate(words):
        print(tid, ' '.join(f'{x:08x}' for x in w))
    if len(sys.argv) > 2:  # also save a raw dump region: offset bytes
        off, n = (int(v, 0) for v in sys.argv[2:4])
        Path(sys.argv[4]).write_bytes(bytes(t.arena[off:off + n]))
