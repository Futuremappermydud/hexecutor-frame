#!/usr/bin/env python3
"""Replay saved model inputs through the prepared QNN executor on the Frame."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from depth_model import offline_bridge as bridge


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scan', type=Path)
    ap.add_argument('--package', type=Path,
                    default=Path.home() / 'frame-dsp/depth-model/wave1-20261101')
    ap.add_argument('--frames', type=int, default=3)
    args = ap.parse_args()
    root = args.package.resolve()
    sys.path[:0] = [str(root / 'frame'), str(root / 'tools')]
    zero_hidden = np.zeros((1, 2, 320, 256), np.float32)
    zero_t = np.zeros((1, 2, 4, 4), np.float32)
    results = []
    with bridge.prepared_session(root, 'primary', bridge.QNN_ENGINE, bridge.QNN_BOOTSTRAP) as infer:
        for index in range(1, args.frames + 1):
            prefix = args.scan / 'depth' / f'frame-{index}'
            left = np.fromfile(f'{prefix}-left.u8', np.uint8).reshape(320, 256)
            right = np.fromfile(f'{prefix}-right.u8', np.uint8).reshape(320, 256)
            start = time.monotonic()
            disparity, uncertainty = infer((left, right), zero_hidden, zero_t)
            disparity = disparity.reshape(2, 320, 256)
            uncertainty = uncertainty.reshape(2, 320, 256)
            elapsed_ms = (time.monotonic() - start) * 1000
            expected_disparity = np.stack((
                np.fromfile(f'{prefix}-disparity.f32', '<f4').reshape(320, 256),
                np.fromfile(f'{prefix}-right-disparity.f32', '<f4').reshape(320, 256)))
            expected_uncertainty = np.fromfile(f'{prefix}-uncertainty.u8', np.uint8).reshape(320, 256)
            diff = np.abs(disparity[0] - expected_disparity[0])
            uncertainty_diff = uncertainty[0].astype(np.int16) - expected_uncertainty.astype(np.int16)
            item = {'frame': index, 'qnn_ms': round(elapsed_ms, 2),
                    'disparity_exact': bool(np.array_equal(disparity, expected_disparity)),
                    'disparity_max_abs': float(diff.max()),
                    'disparity_mean_abs': float(diff.mean()),
                    'disparity_abs_p95_p99': [float(x) for x in np.percentile(diff, (95, 99))],
                    'qnn_disparity_mean': float(disparity[0].mean()),
                    'reference_disparity_mean': float(expected_disparity[0].mean()),
                    'qnn_disparity_minmax': [float(disparity[0].min()), float(disparity[0].max())],
                    'reference_disparity_minmax': [float(expected_disparity[0].min()), float(expected_disparity[0].max())],
                    'disparity_corr': float(np.corrcoef(disparity[0].ravel(), expected_disparity[0].ravel())[0, 1]),
                    'uncertainty_exact': bool(np.array_equal(uncertainty[0], expected_uncertainty)),
                    'uncertainty_mean_abs': float(np.abs(uncertainty_diff).mean()),
                    'uncertainty_equal_fraction': float(np.mean(uncertainty_diff == 0))}
            results.append(item)
            print(json.dumps(item), flush=True)
    if not all(item['disparity_corr'] > 0.99 for item in results):
        raise SystemExit('QNN disparity is poorly correlated with the saved reference output')


if __name__ == '__main__':
    main()
