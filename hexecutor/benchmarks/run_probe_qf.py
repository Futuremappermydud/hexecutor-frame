#!/usr/bin/env python3
"""Check HVX qfloat/convert instructions against NumPy (probe_qf.c)."""

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

from pathlib import Path
import struct
import sys

import numpy as np

HERE = (PATHS.benchmarks)
sys.path.insert(0, str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport

rng = np.random.default_rng(1)
a = rng.integers(0, 16, 128, dtype=np.uint8)
b = rng.integers(-3000, 3000, 32, dtype=np.int32)
c = (rng.standard_normal(64) * 0.01).astype(np.float16)
d = (rng.standard_normal(32) * 3).astype(np.float32)
word = struct.unpack('<I', np.array([3, -5, 7, -128], np.int8).tobytes())[0]
dx = np.float16(0.0371)
with FrameTransport((PATHS.native / 'build/qd_loader.elf'), (PATHS.native / 'build/probe_qf.elf'), 10) as t:
    t.arena[0x4000:0x4200] = a.tobytes() + b.tobytes() + c.tobytes() + d.tobytes()
    def prepare(seq):
        t.arena[128:176] = bytes(48)
        struct.pack_into('<9I', t.arena, 0, 0x46515250, seq, word,
                         int(np.array([dx, dx], np.float16).view(np.uint32)[0]), 0, 0, 0, 0, 0)
    out = t.invoke(prepare, t.workers_done, lambda: bytes(t.arena[0x5000:0x5000 + 11 * 128]))
    ok = struct.unpack_from('<I', t.arena, 32)[0] == 0x600d
vec = [out[i * 128:(i + 1) * 128] for i in range(11)]
print('ran', ok)
w = np.array([3, -5, 7, -128], np.int32)
exp0 = (a.reshape(32, 4).astype(np.int32) * w).sum(1)
print('vrmpy ub*Rb', np.array_equal(np.frombuffer(vec[0], np.int32), exp0))
print('w->sf', np.allclose(np.frombuffer(vec[1], np.float32), b.astype(np.float32)))
lo, hi = np.frombuffer(vec[2], np.float32), np.frombuffer(vec[3], np.float32)
prod = c.astype(np.float32) * np.float32(dx)
print('hf*hf lo=even', np.allclose(lo, prod[0::2], rtol=1e-3), 'hi=odd', np.allclose(hi, prod[1::2], rtol=1e-3),
      'lo=first32', np.allclose(lo, prod[:32], rtol=1e-3))
print('sf*sf', np.allclose(np.frombuffer(vec[4], np.float32), d * d, rtol=1e-5))
print('qf32 acc', np.allclose(np.frombuffer(vec[5], np.float32), 2 * d * d, rtol=1e-5))
print('sf add', np.allclose(np.frombuffer(vec[6], np.float32), 2 * d, rtol=1e-6))
cl, ch = np.frombuffer(vec[7], np.float32), np.frombuffer(vec[10], np.float32)
cf = c.astype(np.float32)
print('hf->sf lo=even', np.allclose(cl, cf[0::2]), 'lo=first32', np.allclose(cl, cf[:32]), 'hi=odd', np.allclose(ch, cf[1::2]))
print('h->hf', np.allclose(np.frombuffer(vec[8], np.float16)[:8].astype(np.float32),
                           np.frombuffer(b.tobytes(), np.int16)[:8].astype(np.float32), rtol=1e-3))
print('max', np.allclose(np.frombuffer(vec[9], np.float32), np.maximum(d, 0)))
