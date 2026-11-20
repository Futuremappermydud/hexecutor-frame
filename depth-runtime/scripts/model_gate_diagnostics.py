#!/usr/bin/env python3
"""Replay a FRAME_DEPTH_RECORD capture (temporal mode) and measure what the
modelMetricDepth gates remove: range, border, low contrast (<12 luma levels in
5x5), census disagreement (>8/24 bits), neighbour support (<3 of 8 within
max(4 cm, 4%)). Saves a few frames (luma, depth, gate reason, confidence) to npz."""

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

import argparse, json, sys
from pathlib import Path
import numpy as np

H, W = 320, 256
FB_MODEL = 24.3048 * W / 616


def windows(a, r):
    p = np.pad(a, r, mode='edge')
    return np.lib.stride_tricks.sliding_window_view(p, (2 * r + 1, 2 * r + 1))


def census(img):
    win = windows(img.astype(np.int16), 2).reshape(H, W, 25)
    centre = img.astype(np.int16)[..., None]
    bits = np.delete(win < centre, 12, axis=2)
    return np.packbits(bits, axis=2, bitorder='big').view('>u4')[..., 0] >> 8 if False else \
        (bits * (1 << np.arange(23, -1, -1, dtype=np.int64))).sum(2)


def gates(disp, left, right):
    reason = np.zeros((H, W), np.uint8)  # 0 kept,1 invalid/range,2 border,3 contrast,4 census,5 support
    z = np.where(disp > 0, FB_MODEL / np.maximum(disp, 1e-6), 0)
    ys, xs = np.mgrid[0:H, 0:W]
    inrange = (disp > 0) & (z >= 0.25) & (z <= 5)
    reason[~inrange] = 1
    rx = xs - disp
    border = (rx < 2) | (rx > W - 3) | (xs < 2) | (xs > W - 3) | (ys < 2) | (ys > H - 3)
    reason[(reason == 0) & border] = 2
    win = windows(left, 2)
    contrast = win.max((2, 3)).astype(int) - win.min((2, 3)).astype(int)
    reason[(reason == 0) & (contrast < 12)] = 3
    cl, cr = census(left), census(right)
    xr = np.clip(np.rint(rx).astype(int), 0, W - 1)
    ham = np.vectorize(lambda v: bin(int(v)).count('1'), otypes=[int])(cl ^ cr[ys, xr])
    reason[(reason == 0) & (ham > 8)] = 4
    measured = np.where(reason == 0, z, 0)
    pad = np.pad(measured, 1)
    support = np.zeros((H, W), int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            n = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            support += (n > 0) & (np.abs(n - measured) <= np.maximum(0.04, 0.04 * measured))
    reason[(reason == 0) & (support < 3)] = 5
    return z, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording'); ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--save', type=int, default=6)
    a = ap.parse_args()
    root = a.package.resolve(); here = (PATHS.repo / 'depth-runtime/scripts')
    sys.path[:0] = [str(root / 'frame'), str(root / 'tools'), str(here)]
    import hashlib
    from depth_model import offline_bridge as bridge
    from model_temporal import ModelTemporalState
    from model_temporal_replay import frames
    from depth_model.adapters import ResidentDSP
    seq = list(frames(a.recording))
    program = 'frame/native_program/examples/primary-v2.nprg'; image = 'frame/build/resident-primary.qdrx'
    for name in (bridge.QNN_ENGINE, bridge.QNN_BOOTSTRAP, 'frame/build/qd_loader.elf', program, image):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == bridge.PREPARED_SHA256[name], name
    names = ['kept', 'invalid/range', 'border', 'low contrast', 'census', 'no support']
    totals = np.zeros(6); saved = {}
    keep = set(np.linspace(10, len(seq) - 1, a.save).astype(int).tolist())
    state = ModelTemporalState()
    with ResidentDSP(root / image, root / 'recovered/primary/constants/constant_pool_5.bin',
                     engine=root / bridge.QNN_ENGINE, bootstrap=root / bridge.QNN_BOOTSTRAP,
                     prepared_program=root / program) as dsp:
        for i, (_, imgs, motion) in enumerate(seq):
            h, t = state.inputs(motion)
            outs = dsp.run([imgs[0], imgs[1], h, t])
            disp = outs[1].reshape(2, H, W); conf = outs[0].reshape(2, H, W)
            state.save(disp)
            z, reason = gates(disp[0], imgs[0], imgs[1])
            totals += np.bincount(reason.ravel(), minlength=6)
            if i in keep:
                saved[f'f{i}_left'] = imgs[0]; saved[f'f{i}_z'] = z.astype(np.float32)
                saved[f'f{i}_reason'] = reason; saved[f'f{i}_conf'] = conf[0]
    frac = totals / totals.sum()
    report = {n: round(float(f), 4) for n, f in zip(names, frac)}
    print(json.dumps(report, indent=1))
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'gate_frames.npz', **saved)
    (a.out / 'gate_report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
