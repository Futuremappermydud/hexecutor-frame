#!/usr/bin/env python3
"""E12 paired full/incremental TSDF extraction on saved model outputs."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_reconstruction_replay import replay_stream  # noqa: E402
from temporal_event_test import EXPOSURE_OFFSET_MS, PATHS, frame_paths, write_json  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(manifest):
    row = {}
    for token in manifest['replay_stdout'].split():
        if '=' in token:
            key, value = token.split('=', 1)
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--converter', type=Path, default=PATHS.repo / 'depth-runtime/bin/model_batch_to_v2')
    parser.add_argument('--replay', type=Path, default=PATHS.repo / 'depth-runtime/bin/replay_room_tsdf')
    parser.add_argument('--mesh-every', type=int, default=5)
    parser.add_argument('--reference-mesh', type=Path)
    parser.add_argument('--reference-replay', type=Path)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    args.session, args.model, args.out = args.session.resolve(), args.model.resolve(), args.out.resolve()
    if args.mesh_every < 1:
        parser.error('--mesh-every must be positive')
    from room_dataset import load_poses, write_depth_metadata
    paths = frame_paths(args.session / 'rgb')
    depth = args.out / 'depth'
    depth.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([str(args.converter), str(args.session / 'rgb'), str(args.model), str(depth), str(len(paths))],
                       check=True)
        extrinsic = json.loads((args.session / 'rgb/rectified-camera-in-head.json').read_text())
        write_depth_metadata(paths, depth, load_poses(args.session / 'head_pose.csv'), extrinsic, EXPOSURE_OFFSET_MS)
        write_json(depth / 'capture.json', {
            'version': 2, 'model': 'primary', 'backend': 'prepared-qnn', 'projection': 'spherical',
            'temporal': 'off', 'model_shape': [320, 256], 'rectified_shape': [616, 616],
            'right_disparity': True, 'capture_clock': 'CLOCK_MONOTONIC', 'exposure_timestamp': True,
            'config_sha256': '',
        })
        manifests = {}
        arms = ('incremental', 'full') if args.reverse else ('full', 'incremental')
        for arm in arms:
            flags = ['--mesh-every', str(args.mesh_every)]
            if arm == 'incremental':
                flags.append('--incremental')
            manifests[arm] = replay_stream(depth, args.out / arm, args.replay, tsdf_args=flags)
        if args.reference_replay:
            manifests['pre_e12'] = replay_stream(depth, args.out / 'pre_e12', args.reference_replay,
                                                tsdf_args=['--mesh-every', str(args.mesh_every)])
        meshes = {arm: (args.out / arm / 'mesh.ply').read_bytes() for arm in manifests}
        exact = meshes['full'] == meshes['incremental']
        reference_replay_exact = meshes['full'] == meshes['pre_e12'] if args.reference_replay else None
        reference_exact = None
        if args.reference_mesh:
            reference_exact = meshes['full'] == args.reference_mesh.read_bytes()
        session = json.loads((args.session / 'session.json').read_text())
        rows = {arm: metrics(manifest) for arm, manifest in manifests.items()}
        failures = []
        if not exact:
            failures.append('incremental mesh differs from full extraction')
        if reference_exact is False:
            failures.append('refactored full mesh differs from the pre-E12 reference')
        if reference_replay_exact is False:
            failures.append('pre-E12 binary mesh differs from the refactored full extraction')
        report = {
            'id': 'E12-incremental-extraction',
            'status': 'complete',
            'parent_baseline': 'E02b observations; E08 3 cm / 12 cm / 4 m projective TSDF',
            'dataset_split': session.get('split', 'unknown'),
            'session': str(args.session),
            'changed_variables': ['full versus dirty-leaf-plus-neighbor extraction'],
            'parameters': {'voxel_m': 0.03, 'truncation_m': 0.12, 'max_depth_m': 4.0,
                           'mesh_every_frames': args.mesh_every, 'max_triangles': 480000,
                           'reverse_order': args.reverse},
            'backend': 'C++17 CPU replay over saved prepared-qnn outputs; no camera or DSP access',
            'precision': 'fp16 TSDF/weights, float32 mesh vertices',
            'quality_metrics': {'mesh_bytes_equal': exact, 'pre_e12_mesh_equal': reference_exact,
                                'pre_e12_replay_equal': reference_replay_exact,
                                'mesh_sha256': {arm: hashlib.sha256(data).hexdigest() for arm, data in meshes.items()},
                                'triangles': {arm: rows[arm].get('triangles') for arm in rows}},
            'timing_p50_p95': {arm: {'integrate_ms': [rows[arm].get('integrate_ms_p50'), rows[arm].get('integrate_ms_p95')],
                                     'mesh_ms': [rows[arm].get('mesh_ms_p50'), rows[arm].get('mesh_ms_p95')],
                                     'rebuilt_leaves': [rows[arm].get('rebuilt_leaves_p50'),
                                                        rows[arm].get('rebuilt_leaves_p95')]}
                               for arm in rows},
            'peak_memory': {arm: {'peak_leaves': rows[arm].get('peak_leaves'),
                                  'evicted_leaves': rows[arm].get('evicted'),
                                  'cached_leaves': rows[arm].get('cached_leaves'),
                                  'cache_vertex_bytes': rows[arm].get('cache_vertex_bytes')} for arm in rows},
            'source_sha256': {'replay_binary': sha256(args.replay), 'driver': sha256(Path(__file__).resolve())},
            'failures': failures,
            'decision': 'candidate only; parity and timing do not establish gate G quality',
            'next_experiment': 'repeat on another saved capture before considering live integration',
        }
        if args.reference_replay:
            report['source_sha256']['pre_e12_replay_binary'] = sha256(args.reference_replay)
        report['pass'] = not failures
        write_json(args.out / 'e12.json', report)
        print(json.dumps(report, indent=2))
        if failures:
            raise SystemExit(1)
    finally:
        shutil.rmtree(depth, ignore_errors=True)


if __name__ == '__main__':
    main()
