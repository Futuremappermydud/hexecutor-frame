#!/usr/bin/env python3
"""CPU adapter regression checks against existing simulator hashes; no hardware."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model_depth import IMAGE_SHAPE, ROOT, STATE_SHAPE, StereoModel, checked, load_image, save_result
from stereo_probes import PAD, texture


class InputTests(unittest.TestCase):
    def test_reject_lossy_or_bad_inputs(self):
        for value in (np.zeros(IMAGE_SHAPE, np.float32), np.zeros((256, 320), np.uint8)):
            with self.assertRaises(ValueError):
                checked(value, IMAGE_SHAPE, np.uint8, 'image')
        bad = np.zeros(STATE_SHAPE, np.float32)
        bad.flat[0] = np.nan
        with self.assertRaises(ValueError):
            checked(bad, STATE_SHAPE, np.float32, 'state')

    def test_image_and_raw_loading(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = np.arange(320 * 256, dtype=np.uint8).reshape(IMAGE_SHAPE)
            Image.fromarray(a).save(root / 'left.png')
            a.tofile(root / 'left.raw')
            np.save(root / 'left.npy', a)
            for name in ('left.png', 'left.raw', 'left.npy'):
                np.testing.assert_array_equal(load_image(root / name), a)
            Image.new('L', (640, 480)).save(root / 'large.png')
            with self.assertRaises(ValueError):
                load_image(root / 'large.png')
            self.assertEqual(load_image(root / 'large.png', True).shape, IMAGE_SHAPE)
            (root / 'short.raw').write_bytes(b'x')
            with self.assertRaises(ValueError):
                load_image(root / 'short.raw')


def integration():
    records = {}
    refs = json.loads((ROOT / 'reference/simulator_output_hashes.json').read_text())['models']
    big = texture()
    left = big[PAD:PAD + 320, PAD:PAD + 256].copy()
    right = big[PAD:PAD + 320, PAD + 8:PAD + 8 + 256].copy()
    smoke = np.tile(np.arange(256, dtype=np.uint8), (320, 1))
    for model in ('primary', 'alternate'):
        session = StereoModel(model)
        rng = np.random.default_rng(1234)
        random = (rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
                  rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
                  rng.uniform(-5, 95, STATE_SHAPE).astype(np.float32),
                  rng.normal(size=(1, 2, 4, 4)).astype(np.float32))
        records[model] = []
        # Final smoke checks that arbitrary supplied state does not leak into next call.
        for name, inputs in [('smoke', (smoke, smoke)), ('w+8', (left, right)),
                             ('random', random), ('smoke', (smoke, smoke))]:
            result = session.infer(*inputs)
            actual = [hashlib.sha256(result[k].tobytes()).hexdigest()
                      for k in ('auxiliary_u8', 'float_output')]
            if name == 'w+8':
                expected = list(refs[model][name].values())
            else:
                expected = [r['sha256'] for r in json.loads(
                    (ROOT / f'recovered/{model}/emulator_{name}.json').read_text()).values()]
            assert sorted(actual) == sorted(expected), (model, name, actual, expected)
            with tempfile.TemporaryDirectory() as temp:
                save_result(result, Path(temp), {'model': model})
                with np.load(Path(temp) / 'outputs.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['float_output'], result['float_output'])
            records[model].append({'case': name, 'equal': True, 'hashes': actual, 'seconds': result['seconds']})
            print(model, name, 'PASS', round(result['seconds'], 2), flush=True)
        del session, result
        gc.collect()
    target = ROOT / 'portable/validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'backend': 'offline-numpy-cpu', 'models': records}, indent=2) + '\n')


if __name__ == '__main__':
    import sys
    if '--integration' in sys.argv:
        integration()
    else:
        unittest.main()
