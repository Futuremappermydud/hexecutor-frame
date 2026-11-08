#!/usr/bin/env python3
"""Create deterministic placed-depth reference scenes for the E01 evaluator."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from room_geometry_eval import evaluate


def scene_depth(name, width=616, height=616):
    rows, columns = np.indices((height, width))
    ray_x = (columns - width / 2) / 380.0
    ray_y = (rows - height / 2) / 380.0
    if name == 'wall':
        depth = np.full((height, width), 2.5)
    elif name == 'oblique':
        depth = 2.5 / (1 + 0.3 * ray_x + 0.12 * ray_y)
    elif name == 'corner':
        depth = np.minimum(2.5 / (1 + 0.4 * ray_x), 2.5 / (1 - 0.25 * ray_x))
    elif name == 'opening':
        depth = np.full((height, width), 2.5)
        depth[(np.abs(ray_x) < 0.22) & (ray_y > -0.2)] = 0
    elif name == 'thin':
        depth = np.full((height, width), 2.5)
        depth[np.abs(ray_x - 0.15) < 0.015] = 1.5
    elif name == 'curve':
        depth = np.full((height, width), 2.5)
        radius = ray_x ** 2 + ray_y ** 2
        depth[radius < 0.12] = 1.8 - 0.25 * np.sqrt(np.maximum(0, 1 - radius / 0.12))[radius < 0.12]
    else:
        raise ValueError(name)
    return depth.astype('<f4')


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=True)
    splits = {'development': ['wall', 'opening'], 'tuning': ['oblique', 'thin'],
              'held_out': ['corner', 'curve']}
    rng = np.random.default_rng(seed)
    scenes = []
    for split, names in splits.items():
        for name in names:
            reference = scene_depth(name)
            noisy = reference.copy()
            valid = reference > 0
            noisy[valid] += rng.normal(0, 0.005, valid.sum()).astype(np.float32)
            noisy[valid & (rng.random(reference.shape) < 0.01)] = 0
            for label, depth in (('reference', reference), ('noisy', noisy)):
                depth.tofile(output / f'{name}-{label}.f32')
            with (output / f'{name}-frames.bin').open('wb') as file:
                file.write(struct.pack('<I', 6))
                for _ in range(6):
                    file.write(struct.pack('<12f', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0))
                    file.write(noisy.tobytes())
                    file.write(np.full(reference.shape, 128, np.uint8).tobytes())
                    file.write(np.ones(reference.shape, np.float32).tobytes())
            scenes.append({'scene': name, 'split': split, 'reference': f'{name}-reference.f32',
                           'observation': f'{name}-noisy.f32', 'replay_stream': f'{name}-frames.bin',
                           'observation_metrics': evaluate(reference, noisy)})
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in output.iterdir() if path.suffix in ('.f32', '.bin')}
    manifest = {'version': 1, 'seed': seed, 'units': 'rectified camera Z metres',
                'shape': [616, 616], 'independent_scenes': True,
                'reference_transform': 'identity, camera at origin', 'reference_uncertainty_m': 0,
                'scenes': scenes, 'sha256': files}
    (output / 'dataset.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.seed), indent=2))
