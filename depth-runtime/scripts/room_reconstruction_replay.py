#!/usr/bin/env python3
"""Validate a v2 Model capture and replay its recorded metric depth into TSDF."""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 616, 616
MODEL_PIXELS = 256 * 320


def load_capture(directory):
    capture = json.loads((directory / 'capture.json').read_text())
    if capture['version'] != 2 or capture['model_shape'] != [320, 256]:
        raise ValueError('unsupported capture version or model shape')
    frames = []
    for meta_path in sorted(directory.glob('frame-*.json'), key=lambda path: int(path.stem.split('-')[1])):
        meta = json.loads(meta_path.read_text())
        prefix = meta['files_prefix']
        arrays = {}
        for name, size, dtype in (
            ('left.u8', MODEL_PIXELS, np.uint8), ('right.u8', MODEL_PIXELS, np.uint8),
            ('rect-left.u8', WIDTH * HEIGHT, np.uint8), ('rect-right.u8', WIDTH * HEIGHT, np.uint8),
            ('disparity.f32', MODEL_PIXELS, '<f4'), ('right-disparity.f32', MODEL_PIXELS, '<f4'),
            ('uncertainty.u8', MODEL_PIXELS, np.uint8), ('depth.f32', WIDTH * HEIGHT, '<f4'),
            ('confidence.f32', WIDTH * HEIGHT, '<f4'),
            ('validity.u8', WIDTH * HEIGHT, np.uint8),
        ):
            path = directory / f'{prefix}-{name}'
            if path.stat().st_size != size * np.dtype(dtype).itemsize:
                raise ValueError(f'bad shape: {path}')
            arrays[name] = np.fromfile(path, dtype=dtype)
        if meta['version'] != 2 or len(meta['camera_to_world_rotation']) != 9 or len(meta['camera_to_world_translation_m']) != 3:
            raise ValueError(f'invalid metadata: {meta_path}')
        if not np.array_equal(arrays['validity.u8'] != 0, arrays['depth.f32'] > 0):
            raise ValueError(f'validity disagrees with depth: {meta_path}')
        frames.append((meta, arrays))
    if not frames:
        raise ValueError('empty capture')
    return capture, frames


def replay(directory, output, executable, max_frames=None):
    capture, frames = load_capture(directory)
    if max_frames is not None:
        frames = frames[:max_frames]
    output.mkdir(parents=True, exist_ok=True)
    stream = output / 'frames.bin'
    with stream.open('wb') as file:
        file.write(struct.pack('<I', len(frames)))
        for meta, arrays in frames:
            file.write(struct.pack('<12f', *(meta['camera_to_world_rotation'] + meta['camera_to_world_translation_m'])))
            file.write(arrays['depth.f32'].tobytes())
            file.write(arrays['rect-left.u8'].tobytes())
            file.write(arrays['confidence.f32'].tobytes())
    result = subprocess.run([str(executable), str(stream), str(output / 'mesh.ply')], check=True, capture_output=True, text=True)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    manifest = {'version': 1, 'source': str(directory), 'input_sha256': hashes,
                'frames': len(frames), 'capture': capture, 'replay_stdout': result.stdout.strip()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def replay_stream(directory, output, executable, max_frames=None):
    """Replay without retaining all frames in RAM or a multi-gigabyte frames.bin."""
    capture = json.loads((directory / 'capture.json').read_text())
    if capture['version'] != 2 or capture['model_shape'] != [320, 256]:
        raise ValueError('unsupported capture version or model shape')
    paths = sorted(directory.glob('frame-*.json'), key=lambda path: int(path.stem.split('-')[1]))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths or len(paths) > 10000:
        raise ValueError('invalid replay frame count')
    output.mkdir(parents=True, exist_ok=True)
    mesh = output / 'mesh.ply'
    process = subprocess.Popen([str(executable), '-', str(mesh)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdin.write(struct.pack('<I', len(paths)))
        for meta_path in paths:
            meta = json.loads(meta_path.read_text())
            transform = meta['camera_to_world_rotation'] + meta['camera_to_world_translation_m']
            if meta['version'] != 2 or len(transform) != 12:
                raise ValueError(f'invalid pose: {meta_path}')
            process.stdin.write(struct.pack('<12f', *transform))
            for suffix, size in (('depth.f32', WIDTH * HEIGHT * 4),
                                 ('rect-left.u8', WIDTH * HEIGHT),
                                 ('confidence.f32', WIDTH * HEIGHT * 4)):
                source = directory / f'{meta["files_prefix"]}-{suffix}'
                if source.stat().st_size != size:
                    raise ValueError(f'bad shape: {source}')
                with source.open('rb') as frame:
                    for block in iter(lambda: frame.read(1024 * 1024), b''):
                        process.stdin.write(block)
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f'TSDF replay exited {process.returncode}: {stderr.decode(errors="replace")}')
    except BaseException:
        process.kill()
        process.wait()
        mesh.unlink(missing_ok=True)
        raise
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in directory.iterdir() if path.is_file()}
    manifest = {'version': 1, 'source': str(directory), 'input_sha256': hashes,
                'frames': len(paths), 'capture': capture,
                'replay_stdout': stdout.decode(errors='replace').strip(), 'streamed': True}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--stream', action='store_true', help='pipe frames to replay without a large temporary stream')
    args = parser.parse_args()
    method = replay_stream if args.stream else replay
    print(json.dumps(method(args.capture, args.output, args.executable, args.max_frames), indent=2))
