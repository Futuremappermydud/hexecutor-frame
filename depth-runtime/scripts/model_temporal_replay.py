#!/usr/bin/env python3
"""Replay a FRAME_DEPTH_RECORD capture through the Model model with and without
the temporal state, in one prepared-qnn DSP session, and compare.

Metrics per mode (left disparity plane, model pixels, plausible depth 0.25-5 m):
  lr_consistent  fraction of pixels whose right-plane disparity at x-d agrees
                 within 1 px (independent of temporal smoothing)
  flicker_med    median |d_now - warp(d_prev)| where the warp lands (aligned pairs)
  flicker_gt1    fraction of those pixels changing by more than 1 px
"""

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

import argparse, json, struct, sys, time
from pathlib import Path
import numpy as np

HDR, BLOCK, PAIR = 16, 64, 2 * 256 * 320
FB_MODEL = 24.3048 * 256 / 616


def frames(path):
    data = Path(path).read_bytes(); step = HDR + BLOCK + PAIR
    for o in range(0, len(data) - step + 1, step):
        magic, seq, w, h = struct.unpack_from('<4I', data, o)
        assert magic == 0x36504451 and (w, h) == (256, 320)
        block = struct.unpack_from('<I12f3I', data, o + HDR)
        images = np.frombuffer(data, np.uint8, PAIR, o + HDR + BLOCK).reshape(2, 320, 256)
        motion = ((np.array(block[1:10]).reshape(3, 3), np.array(block[10:13]))
                  if block[0] & 1 else None)
        yield seq, images, motion


def plausible(d):
    return (d > FB_MODEL / 5.0) & (d < FB_MODEL / 0.25)


def lr_consistency(out):
    dl, dr = out[0], out[1]
    ys, xs = np.nonzero(plausible(dl))
    xr = np.rint(xs - dl[ys, xs]).astype(int)
    ok = (xr >= 0) & (xr < 256)
    agree = np.abs(dl[ys[ok], xs[ok]] - dr[ys[ok], xr[ok]]) < 1.0
    return float(agree.sum() / max(len(ys), 1)), int(len(ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording'); ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--report', type=Path)
    a = ap.parse_args()
    root = a.package.resolve()
    here = (PATHS.repo / 'depth-runtime/scripts')
    sys.path[:0] = [str(root / 'frame'), str(root / 'tools'), str(here)]
    from depth_model import offline_bridge as bridge
    from model_temporal import ModelTemporalState
    seq = list(frames(a.recording))
    print(f'{len(seq)} pairs, {sum(m is not None for _, _, m in seq)} with motion', flush=True)
    zero_h = np.zeros((1, 2, 320, 256), np.float32); zero_t = np.zeros((1, 2, 4, 4), np.float32)
    report = {'pairs': len(seq), 'modes': {}}
    with bridge.prepared_session(root, 'primary', bridge.QNN_ENGINE, bridge.QNN_BOOTSTRAP) as infer:
        infer(seq[0][1], zero_h, zero_t)  # warm-up outside the measurement
        for mode in ('zero', 'temporal'):
            state = ModelTemporalState()
            probe = ModelTemporalState(pyramid=0)  # full-res warp for the flicker metric
            lr, flick, gt1, ms, gate8 = [], [], [], [], 0
            prev = None
            for _, images, motion in seq:
                t0 = time.monotonic()
                if mode == 'temporal':
                    h, t = state.inputs(motion)
                    gate8 += int(state.stats.get('history', False) and np.abs(t).mean() < 7.5e-4)
                else:
                    h, t = zero_h, zero_t
                out = infer(images, h, t)[0].reshape(2, 320, 256)
                if mode == 'temporal': state.save(out)
                ms.append((time.monotonic() - t0) * 1e3)
                lr.append(lr_consistency(out)[0])
                if prev is not None and motion is not None:
                    probe.prev = prev
                    warped = probe.inputs(motion)[0][0, 0]
                    m = (warped > 0) & plausible(out[0]) & plausible(warped)
                    if m.sum() > 1000:
                        diff = np.abs(out[0][m] - warped[m])
                        flick.append(float(np.median(diff))); gt1.append(float((diff > 1).mean()))
                prev = out.copy()
            r = {'lr_consistent': float(np.mean(lr)), 'flicker_med_px': float(np.median(flick)) if flick else None,
                 'flicker_gt1': float(np.mean(gt1)) if gt1 else None, 'flicker_pairs': len(flick),
                 'ms_median': float(np.median(ms)), 'still_gate_frames': gate8}
            report['modes'][mode] = r
            print(mode, json.dumps(r), flush=True)
    if a.report: a.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
