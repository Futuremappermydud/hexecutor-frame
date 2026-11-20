#!/usr/bin/env python3
"""E09 input test: keep only model pixels the image texture supports (E05 wall-mesh gate).

Offline development tool. Writes OUT/DST/model/ from OUT/SRC/model/ (the
temporal_event_test.py arm layout): disparities are hardlinked unchanged, and the left
uncertainty is set to 255 on rejected pixels, so depth conversion (uncertainty < 89) drops
them. Right planes are hardlinked.

Gate per model pixel (unchanged from E05 wall_meshes.py, declared there before its first
mesh): 9x9 ZNCC on the spherical pair at the model's rounded disparity >= 0.6, at least 0.1
above the ZNCC 3 px either side, and left window standard deviation >= 2. The side test
rejects blank wall and horizontal lines that match at any disparity.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_event_test import H, W, frame_paths, write_json  # noqa: E402


def gate(left, right, disparity):
    from numpy.lib.stride_tricks import sliding_window_view
    half = 4
    a = sliding_window_view(left, (9, 9)).reshape(H - 8, W - 8, 81)
    b = sliding_window_view(right, (9, 9)).reshape(H - 8, W - 8, 81)
    a = a - a.mean(-1, keepdims=True)
    na = np.sqrt((a * a).sum(-1))
    b = b - b.mean(-1, keepdims=True)
    nb = np.sqrt((b * b).sum(-1))
    d = np.rint(np.nan_to_num(disparity[half:H - half, half:W - half])).astype(np.int32)
    rows, columns = np.mgrid[0:H - 8, 0:W - 8]
    score, ok = {}, np.ones(d.shape, bool)
    for offset in (0, -3, 3):
        x = columns - (d + offset)
        inside = (x >= 0) & (x < W - 8)
        ok &= inside
        x = np.clip(x, 0, W - 9)
        with np.errstate(all='ignore'):
            score[offset] = (a * b[rows, x]).sum(-1) / (na * nb[rows, x])
    side = np.maximum(np.nan_to_num(score[-3], nan=1.0), np.nan_to_num(score[3], nan=1.0))
    keep = ok & (score[0] >= .6) & (score[0] - side >= .1) & (na >= 2 * 9)
    full = np.zeros((H, W), bool)
    full[half:H - half, half:W - half] = keep
    return full


def one(job):
    session, src, dst, name = job
    left = np.fromfile(session / 'rgb' / f'{name}-spherical-left.u8', np.uint8).reshape(H, W).astype(np.float32)
    right = np.fromfile(session / 'rgb' / f'{name}-spherical-right.u8', np.uint8).reshape(H, W).astype(np.float32)
    prefix = f'{name}-spherical'
    d = np.fromfile(src / f'{prefix}-disparity.f32', '<f4').reshape(H, W)
    u = np.fromfile(src / f'{prefix}-uncertainty.u8', np.uint8).reshape(H, W)
    keep = gate(left, right, d)
    np.where(keep, u, 255).astype(np.uint8).tofile(dst / f'{prefix}-uncertainty.u8')
    for suffix in ('disparity.f32', 'right-disparity.f32', 'right-uncertainty.u8'):
        target = dst / f'{prefix}-{suffix}'
        target.unlink(missing_ok=True)
        os.link(src / f'{prefix}-{suffix}', target)
    return float(keep.mean()), float((keep & (u < 89)).sum() / max((u < 89).sum(), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', default='off')
    parser.add_argument('--dest', default='gated')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    session, out = args.session.resolve(), args.out.resolve()
    src, dst = out / args.source / 'model', out / args.dest / 'model'
    dst.mkdir(parents=True, exist_ok=True)
    names = [p.stem for p in frame_paths(session / 'rgb')]
    with Pool(args.workers) as pool:
        kept = pool.map(one, [(session, src, dst, n) for n in names], chunksize=8)
    kept = np.array(kept)
    report = {'source': args.source, 'frames': len(names), 'kept_fraction_median': float(np.median(kept[:, 0])),
              'kept_of_valid_median': float(np.median(kept[:, 1]))}
    write_json(out / args.dest / 'texture_gate.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
