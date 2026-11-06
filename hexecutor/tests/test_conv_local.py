#!/usr/bin/env python3
"""Check the persistent conv payload's packing, indexing and blocking natively."""

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
import sys
import numpy as np
from hexecutor.offline.local_transport import LocalTransport

lib = (PATHS.native / 'build/libqd_conv_host.so')
t = LocalTransport(conv_library=lib)
rng = np.random.default_rng(7)
shapes = [((10, 130, 64), (3, 3, 64, 64)), ((10, 34, 128), (3, 3, 128, 128)), ((1, 2048, 12), (1, 1, 12, 32)),
          ((9, 33, 512), (2, 2, 512, 128)), ((1, 64, 20), (1, 1, 20, 128)), ((10, 10, 128), (3, 3, 128, 128)),
          ((10, 258, 32), (3, 3, 32, 32)), ((3, 5, 3), (2, 2, 3, 8)), ((9, 17, 512), (2, 2, 512, 128))]
for xs, ws in shapes:
    for boundary in (False, True):
        x = rng.integers(0, 256, xs, dtype=np.uint8)
        w = rng.integers(-128, 128, ws, dtype=np.int32)
        if boundary:
            x[:] = 255; w[:] = -128
        got = t.conv(x, w)
        kh, kw = ws[:2]
        oh, ow = xs[0] - kh + 1, xs[1] - kw + 1
        want = np.zeros((oh, ow, ws[3]), np.int64)
        for ky in range(kh):
            for kx in range(kw):
                want += x[ky:ky + oh, kx:kx + ow].astype(np.int64) @ w[ky, kx].astype(np.int64)
        assert np.array_equal(got, want), (xs, ws, boundary)
        print('ok', xs, ws, 'boundary' if boundary else 'random')
