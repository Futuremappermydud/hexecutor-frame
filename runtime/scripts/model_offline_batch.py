#!/usr/bin/env python3
"""Run recorded Model model inputs through the DSP model in ONE session (Frame only).

Use this instead of live per-frame inference for replay corpora: the live v2
recorder closes and reopens the cDSP session per frame, and that close path
failed on its 7th close in E02. A single session closed cleanly in all E02 runs.

Inputs: prefixes P such that P-left.u8 / P-right.u8 are 256x320 uint8 model
images. These are spherical (capture_model_inputs: frame-N-spherical) or pinhole
(frame-N-pinhole) inputs, or a v2 recording's frame-N.

Outputs per prefix, in --out:
  NAME-disparity.f32, NAME-right-disparity.f32 (plane 0/1, float32 320x256)
  NAME-uncertainty.u8, NAME-right-uncertainty.u8 (uint8 320x256)
  batch.json: input/output SHA256, timing, backend, package and repeat agreement.
Zero temporal state (t=0), matching the E00-E02 baseline (recurrence off).

Example on Frame:
  ~/frame-dsp/depth-model/.venv/bin/python scripts/model_offline_batch.py \\
      --package ~/frame-dsp/depth-model/wave1-20261101 --out /tmp/out \\
      capture/frame-1-spherical capture/frame-2-spherical
Afterwards check the DSP: frame/bench/chunk_map.py must report n_bad 0.
"""

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

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--package', type=Path, required=True)
ap.add_argument('--model', choices=('primary', 'alternate'), default='primary')
ap.add_argument('--backend', choices=('prepared-qnn', 'prepared'), default='prepared-qnn')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--repeats', type=int, default=1, help='>1 also checks bit-exact repeatability')
ap.add_argument('prefixes', nargs='+')
a = ap.parse_args()
if a.backend == 'prepared-qnn' and a.model != 'primary':
    ap.error('prepared-qnn supports primary only; use --backend prepared for alternate')
if a.repeats < 1:
    ap.error('--repeats must be positive')
root = a.package.resolve()
sys.path[:0] = [str(root / 'frame'), str(root / 'tools'), str((PATHS.repo / 'runtime/scripts'))]
from depth_model import offline_bridge as bridge  # noqa: E402

os.sched_setaffinity(0, {7})
a.out.mkdir(parents=True, exist_ok=True)
zero_h = np.zeros((1, 2, 320, 256), np.float32)
zero_t = np.zeros((1, 2, 4, 4), np.float32)
sha = lambda b: hashlib.sha256(b).hexdigest()
pairs = []
for p in a.prefixes:
    left = np.fromfile(p + '-left.u8', np.uint8)
    right = np.fromfile(p + '-right.u8', np.uint8)
    if left.size != 320 * 256 or right.size != 320 * 256:
        sys.exit(f'{p}: expected 320x256 uint8 images')
    pairs.append((p, np.stack([left.reshape(320, 256), right.reshape(320, 256)])))

report = {'package': str(root), 'backend': a.backend, 'model': a.model, 'temporal': False,
          'repeats': a.repeats, 'pairs': []}
engine = (bridge.QNN_ENGINE, bridge.QNN_BOOTSTRAP) if a.backend == 'prepared-qnn' else (bridge.PREPARED_ENGINE, bridge.PREPARED_BOOTSTRAP)
with bridge.prepared_session(root, a.model, *engine) as infer:
    for p, images in pairs:
        outs, ms = [], []
        for _ in range(a.repeats):
            t0 = time.monotonic()
            disparity, uncertainty = infer(images, zero_h, zero_t)
            ms.append((time.monotonic() - t0) * 1e3)
            outs.append((np.asarray(disparity, np.float32).reshape(2, 320, 256).copy(),
                         np.asarray(uncertainty).reshape(2, 320, 256).astype(np.uint8)))
        d, u = outs[0]
        exact = all(np.array_equal(d, o[0]) and np.array_equal(u, o[1]) for o in outs[1:])
        name = Path(p).name
        d[0].tofile(a.out / f'{name}-disparity.f32'); d[1].tofile(a.out / f'{name}-right-disparity.f32')
        u[0].tofile(a.out / f'{name}-uncertainty.u8'); u[1].tofile(a.out / f'{name}-right-uncertainty.u8')
        entry = {'prefix': p, 'input_sha256': sha(images.tobytes()), 'output_sha256': sha(d.tobytes() + u.tobytes()),
                 'ms': ms, 'repeats_bit_exact': exact}
        report['pairs'].append(entry)
        print(name, json.dumps({k: entry[k] for k in ('ms', 'repeats_bit_exact')}), flush=True)
(a.out / 'batch.json').write_text(json.dumps(report, indent=2) + '\n')
print('session closed', flush=True)
