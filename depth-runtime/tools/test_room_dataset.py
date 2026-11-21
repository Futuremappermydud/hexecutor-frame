#!/usr/bin/env python3
"""Offline contract check: captured arrays + poses -> depth v2 -> TSDF mesh."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = PATHS.repo / 'depth-runtime'


def write(path, value):
    path.write_text(json.dumps(value))


def main():
    with tempfile.TemporaryDirectory(prefix='room-dataset-test-') as temporary:
        root = Path(temporary)
        session = root / 'session'
        for name in ('rgb', 'model', 'ir'):
            (session / name).mkdir(parents=True)
        converter = root / 'model_batch_to_v2'
        replay = root / 'replay_room_tsdf'
        subprocess.run(['c++', '-O2', '-std=c++17', '-Iinclude', '-o', str(converter),
                        'tools/model_batch_to_v2.cpp'], cwd=REPO, check=True)
        subprocess.run(['c++', '-O2', '-std=c++17', '-Iinclude', '-o', str(replay),
                        'tools/replay_room_tsdf.cpp', 'src/tsdf_volume.cpp', '-lpthread', '-lm'],
                       cwd=REPO, check=True)
        write(session / 'session.json', {'version': 1, 'status': 'captured',
                                        'scene': 'fixture', 'split': 'development'})
        write(session / 'rgb/rectified-camera-in-head.json',
              {'rotation_row_major': [1, 0, 0, 0, 1, 0, 0, 0, 1], 'translation_m': [0.1, 0, 0]})
        write(session / 'rgb/frame-1.json',
              {'timed': True, 'left_end_of_readout_ns': 1_008_000_000,
               'capture_begin_ns': 1_000_000_000, 'capture_end_ns': 1_010_000_000,
               'camera_sequence': 1, 'spherical_step_rad': np.radians(0.29)})
        for eye in ('left', 'right'):
            np.full(616 * 616, 120, np.uint8).tofile(session / f'rgb/frame-1-rect-{eye}.u8')
            np.full(256 * 320, 120, np.uint8).tofile(session / f'rgb/frame-1-spherical-{eye}.u8')
        for prefix in ('', 'right-'):
            np.full(256 * 320, 5, np.float32).tofile(
                session / f'model/frame-1-spherical-{prefix}disparity.f32')
            np.full(256 * 320, 50, np.uint8).tofile(
                session / f'model/frame-1-spherical-{prefix}uncertainty.u8')
        with (session / 'head_pose.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mono_ns', 'raw_ns', 'valid', 'result'] +
                            [f'm{r}{c}' for r in range(3) for c in range(4)])
            writer.writerow([1_000_000_000, 990_000_000, 1, 200,
                             1, 0, 0, 0.2, 0, 1, 0, 0, 0, 0, 1, 0])
        subprocess.run([sys.executable, str((PATHS.repo / 'depth-runtime/scripts/room_dataset.py')), 'finalize',
                        str(session), '--skip-inference', '--skip-tags', '--converter',
                        str(converter), '--replay', str(replay)], cwd=REPO, check=True)
        meta = json.loads((session / 'depth/frame-1.json').read_text())
        summary = json.loads((session / 'session.json').read_text())
        manifest = json.loads((session / 'manifest.json').read_text())
        assert meta['pose_delta_ns'] == 0
        assert abs(meta['camera_to_world_translation_m'][0] - 0.3) < 1e-5
        assert summary['frames_meshed'] == 1 and summary['status'] == 'finalized'
        assert (session / 'mesh/mesh.ply').read_bytes().startswith(b'ply\n')
        assert manifest['defects'] == [] and 'depth/frame-1-depth.f32' in manifest['sha256']
        subprocess.run([sys.executable, str((PATHS.repo / 'depth-runtime/scripts/room_dataset.py')), 'index', str(root)],
                       cwd=REPO, check=True)
        dataset = json.loads((root / 'dataset.json').read_text())
        assert dataset['scene_splits'] == {'fixture': 'development'}
        print('room dataset end-to-end fixture: PASS')


if __name__ == '__main__':
    main()
