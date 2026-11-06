#!/usr/bin/env python3
"""Offline stereo inference. Never imports Frame transport or connects to hardware."""

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
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import htp_emu
import sim_trace
from run_htp_emu import execute

ROOT = (PATHS.model_data)
IMAGE_SHAPE = (320, 256)
STATE_SHAPE = (1, 2, 320, 256)
TRANSFORM_SHAPE = (1, 2, 4, 4)


def checked(value, shape, dtype, name):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f'{name}: expected shape {shape}, got {value.shape}')
    if value.dtype != np.dtype(dtype):
        raise ValueError(f'{name}: expected {np.dtype(dtype)}, got {value.dtype}')
    if not np.isfinite(value).all():
        raise ValueError(f'{name}: nonfinite values are not supported')
    return np.ascontiguousarray(value)


class StereoModel:
    """Reusable CPU session over the recovered compiled graph.

    Images are uint8 [320,256] in model coordinates. hidden and transforms are
    float32 [1,2,320,256] and [1,2,4,4]. Omitted values use experimental zero
    state and two identity matrices. No state is implicitly carried between
    calls: external motion compensation and state semantics remain unresolved.
    A session is mutable and must not be used concurrently.
    """
    def __init__(self, model='primary', root=ROOT):
        if model not in ('primary', 'alternate'):
            raise ValueError('model must be primary or alternate')
        self.model = model
        recovered = Path(root) / 'recovered' / model
        self.graph = sim_trace.load_graph(recovered)
        self.nodes = sim_trace.execution_nodes(self.graph)
        missing = {n['type_name'] for n in self.nodes} - htp_emu.KERNELS.keys()
        if missing:
            raise ValueError(f'Unsupported kernels: {sorted(missing)}')
        self.emu = htp_emu.Emulator(
            self.graph, (recovered / 'constants/constant_pool_5.bin').read_bytes(), {})

    def infer(self, left, right, hidden=None, transforms=None, progress=False):
        left = checked(left, IMAGE_SHAPE, np.uint8, 'left')
        right = checked(right, IMAGE_SHAPE, np.uint8, 'right')
        if hidden is None:
            hidden = np.zeros(STATE_SHAPE, np.float32)
        if transforms is None:
            transforms = np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1))
        hidden = checked(hidden, STATE_SHAPE, np.float32, 'hidden')
        transforms = checked(transforms, TRANSFORM_SHAPE, np.float32, 'transforms')
        start = time.monotonic()
        outputs = execute(self.emu, self.nodes,
                          dict(enumerate((left, right, hidden, transforms))), progress)
        # Copy so the next inference cannot alter an earlier result.
        auxiliary = outputs[0].reshape(STATE_SHAPE).copy()
        floating = outputs[1].reshape(STATE_SHAPE).copy()
        return {
            'float_output': floating,
            'auxiliary_u8': auxiliary,
            'auxiliary_dequantized': (auxiliary.astype(np.float32) - 136) * np.float32(0.05533868819475174),
            'seconds': time.monotonic() - start,
        }


def load_image(path, resize=False):
    if path.suffix.lower() == '.npy':
        return checked(np.load(path, allow_pickle=False), IMAGE_SHAPE, np.uint8, str(path))
    if path.suffix.lower() == '.raw':
        data = np.fromfile(path, np.uint8)
        if data.size != np.prod(IMAGE_SHAPE):
            raise ValueError(f'{path}: expected {np.prod(IMAGE_SHAPE)} raw bytes')
        return data.reshape(IMAGE_SHAPE)
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('L')
        size = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
        if image.size != size:
            if not resize:
                raise ValueError(f'{path}: expected image width/height {size}, got {image.size}; use --resize explicitly')
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.array(image, dtype=np.uint8)


def save_result(result, output, metadata):
    output.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output / 'outputs.npz', **arrays)
    for name, array in arrays.items():
        np.save(output / f'{name}.npy', array, allow_pickle=False)
    # PGM needs no imaging dependencies. Save both planes without assigning eye semantics.
    limits = []
    for plane, values in enumerate(result['float_output'][0]):
        lo, hi = (float(v) for v in np.percentile(values, [2, 98]))
        pixels = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
        pixels = (pixels * 255).astype(np.uint8)
        (output / f'float_plane_{plane}.pgm').write_bytes(b'P5\n256 320\n255\n' + pixels.tobytes())
        limits.append([lo, hi])
    report = dict(metadata, seconds=result['seconds'], preview_percentile_limits=limits,
                  outputs={k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                               'sha256': hashlib.sha256(v.tobytes()).hexdigest()}
                           for k, v in arrays.items()},
                  interpretation='Float output tracks synthetic horizontal disparity in pixels. Both planes retained; real-camera geometry and auxiliary confidence semantics unverified.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['primary', 'alternate'], default='primary')
    parser.add_argument('--left', type=Path, required=True, help='rectified image, uint8 .npy or native .raw')
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--hidden', type=Path, help='float32 .npy [1,2,320,256]')
    parser.add_argument('--transforms', type=Path, help='float32 .npy [1,2,4,4]')
    parser.add_argument('--resize', action='store_true', help='explicitly resize images; caller remains responsible for rectification/calibration')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        left, right = (load_image(p, args.resize) for p in (args.left, args.right))
        hidden = np.load(args.hidden, allow_pickle=False) if args.hidden else None
        transforms = np.load(args.transforms, allow_pickle=False) if args.transforms else None
        # Validate before allocating the graph's large address caches.
        if hidden is not None:
            checked(hidden, STATE_SHAPE, np.float32, 'hidden')
        if transforms is not None:
            checked(transforms, TRANSFORM_SHAPE, np.float32, 'transforms')
        session = StereoModel(args.model)
        result = session.infer(left, right, hidden, transforms, progress=True)
        report = save_result(result, args.output, {
            'model': args.model, 'context_sha256': session.graph['context_sha256'],
            'backend': 'offline-numpy-cpu', 'left': str(args.left.resolve()),
            'right': str(args.right.resolve()), 'resize_requested': args.resize,
            'zero_hidden_placeholder': hidden is None,
            'identity_transforms_placeholder': transforms is None,
        })
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f'depth-model: {exc}\n')
    print(f"Saved {args.output / 'outputs.npz'} ({report['seconds']:.2f}s)")


if __name__ == '__main__':
    main()
