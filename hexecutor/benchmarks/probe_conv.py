"""Run a few persistent-payload convolutions on the Frame and check them exactly."""

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

import sys, time, struct
from pathlib import Path
sys.path.insert(0, str((PATHS.native)))
import numpy as np
import hexecutor.offline.frame_transport as ft
from hexecutor.offline.hvx_protocol import STATUS, STATUS_OFFSET
b = (PATHS.native / 'build')
rng = np.random.default_rng(1)
with ft.FrameTransport((PATHS.native / 'build/qd_loader.elf'), (PATHS.native / 'build/qd_conv.elf'), 10) as t:
    orig = t._persistent_done
    log = []
    def pd():
        log.append((time.monotonic(), t.get(ft.CTRL), t.get(ft.CTRL+4), t.get(ft.CTRL+8),
                    [STATUS.unpack_from(t.arena, STATUS_OFFSET + i*16) for i in range(3)]))
        return orig()
    t._persistent_done = pd
    for xs, ws in [((10, 34, 128), (3, 3, 128, 128)), ((10, 258, 32), (3, 3, 32, 32)), ((9, 33, 512), (2, 2, 512, 128))] * int(sys.argv[1]):
        x = rng.integers(0, 256, xs, dtype=np.uint8); w = rng.integers(-128, 128, ws, dtype=np.int32)
        try:
            t0 = time.monotonic(); got = t.conv(x, w); dt = time.monotonic() - t0
        except Exception as e:
            print('FAIL', e); 
            for l in log[:3] + log[-3:]: print(l)
            print('cmd', struct.unpack_from('<20I', t.arena, 0), 'barrier', struct.unpack_from('<I', t.arena, 64))
            raise
        kh, kw = ws[:2]; oh, ow = xs[0]-kh+1, xs[1]-kw+1
        want = sum(x[ky:ky+oh, kx:kx+ow].astype(np.int64) @ w[ky, kx].astype(np.int64) for ky in range(kh) for kx in range(kw))
        tm = [struct.unpack_from('<4I', t.arena, 2048 + 16 * i)[:3] for i in range(3)]
        # Never re-pack into the live arena here: workers would run it as a command.
        print('equal', np.array_equal(got, want), f'{dt*1e3:.2f} ms', 'polls', len(log),
              'dsp copy/wait/compute us', [[round(c / 1900) for c in r] for r in tm]); log.clear()
