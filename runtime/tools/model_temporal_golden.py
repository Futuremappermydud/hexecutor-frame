#!/usr/bin/env python3
"""Write reference warps from scripts/model_temporal.py for tools/test_model_temporal.cpp.

  model_temporal_golden.py DISPARITY.npy POSES.npy OUT [--every 10]

DISPARITY.npy is [frames, 2, 320, 256] model output. POSES.npy is [frames, 12]: the
camera-to-world rotation (row-major) then translation of each frame. Each case holds
the previous output, the motion, and the expected hidden and t, for spherical and
pinhole geometry.
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

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((PATHS.repo / 'runtime/scripts')))
from model_temporal import ModelTemporalState  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('disparity', type=Path)
parser.add_argument('poses', type=Path)
parser.add_argument('out', type=Path)
parser.add_argument('--every', type=int, default=10)
args = parser.parse_args()

disparity = np.load(args.disparity, mmap_mode='r')
poses = np.load(args.poses)
args.out.mkdir(parents=True, exist_ok=True)
yaw = np.radians(35.0)
fast = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
cases = 0
for frame in range(1, len(disparity), args.every):
    Rp, tp = poses[frame - 1, :9].reshape(3, 3), poses[frame - 1, 9:]
    Rc, tc = poses[frame, :9].reshape(3, 3), poses[frame, 9:]
    steps = [(Rc.T @ Rp, Rc.T @ (tp - tc))]
    if cases % 5 == 0:
        steps.append((fast, np.array([0.05, -0.02, 0.1])))
    for rotation, translation in steps:
        previous = np.asarray(disparity[frame - 1], np.float32)
        for degrees, name in ((0.29, 'spherical'), (None, 'pinhole')):
            state = ModelTemporalState(spherical_deg=degrees)
            state.save(previous)
            hidden, t = state.inputs((rotation, translation))
            stem = args.out / f'case-{cases}-{name}'
            previous.tofile(f'{stem}-previous.f32')
            np.concatenate([rotation.ravel(), translation]).astype('<f8').tofile(f'{stem}-motion.f64')
            np.asarray(hidden[0], '<f4').tofile(f'{stem}-hidden.f32')
            np.asarray(t[0], '<f4').tofile(f'{stem}-t.f32')
        cases += 1
(args.out / 'cases.txt').write_text(f'{cases}\n')
print(f'{cases} cases in {args.out}')
