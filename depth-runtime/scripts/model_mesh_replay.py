#!/usr/bin/env python3
"""Pack a recorded v2 depth capture into the stream read by bin/replay_model_mesh.

The stream holds the model outputs (disparity, uncertainty), the rectified left luma and
the capture-time pose for each frame, so the replay covers the whole CPU path from the
network output to the published TSDF mesh. Frames without a valid capture-time pose are
skipped, as the live worker does not integrate them.

  model_mesh_replay.py pack SESSION/depth out.mmr [--max-frames N]
"""
import argparse
import json
import struct
import sys
from pathlib import Path

MODEL = 320 * 256
RECT = 616 * 616
MAGIC = 0x31524D4D  # 'MMR1'


def pack(directory, output, max_frames, max_uncertainty):
    capture = json.loads((directory / 'capture.json').read_text())
    if capture['version'] != 2 or capture['model_shape'] != [320, 256] or capture['projection'] != 'spherical':
        raise SystemExit('unsupported capture: need a version 2 spherical 320x256 recording')
    paths = sorted(directory.glob('frame-*.json'), key=lambda path: int(path.stem.split('-')[1]))
    frames = []
    step = None
    for path in paths:
        meta = json.loads(path.read_text())
        if not (meta.get('pose_valid') and meta.get('capture_time_pose')):
            continue
        if step is None:
            step = meta['spherical_step_rad']
        elif abs(step - meta['spherical_step_rad']) > 1e-9:
            raise SystemExit(f'spherical step changes at {path}')
        frames.append(meta)
        if max_frames and len(frames) == max_frames:
            break
    if not frames or not step:
        raise SystemExit('no usable frames')
    with output.open('wb') as stream:
        stream.write(struct.pack('<IIfI', MAGIC, len(frames), step, max_uncertainty))
        for meta in frames:
            transform = meta['camera_to_world_rotation'] + meta['camera_to_world_translation_m']
            stream.write(struct.pack('<12f', *transform))
            stream.write(struct.pack('<3f', *meta['position_m']))
            prefix = meta['files_prefix']
            for suffix, size in (('disparity.f32', MODEL * 4), ('uncertainty.u8', MODEL), ('rect-left.u8', RECT)):
                data = (directory / f'{prefix}-{suffix}').read_bytes()
                if len(data) != size:
                    raise SystemExit(f'bad size: {prefix}-{suffix}')
                stream.write(data)
    print(f'frames={len(frames)} skipped={len(paths) - len(frames)} step={step} bytes={output.stat().st_size}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    packer = commands.add_parser('pack')
    packer.add_argument('depth', type=Path)
    packer.add_argument('output', type=Path)
    packer.add_argument('--max-frames', type=int, default=0)
    packer.add_argument('--max-uncertainty', type=int, default=89)
    args = parser.parse_args()
    pack(args.depth, args.output, args.max_frames, args.max_uncertainty)


if __name__ == '__main__':
    sys.exit(main())
