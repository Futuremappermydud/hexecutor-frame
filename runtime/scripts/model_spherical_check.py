#!/usr/bin/env python3
"""Run the Model model on a rectified pinhole pair (616x616) two ways and compare:
  pinhole: the current anisotropic resize to 256x320, z = FB/d
  spherical: resampled into the Model-style spherical model, triangulated
Depth is compared along identical rays where both are valid (uncertainty < 89)."""

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

import argparse, hashlib, sys
from pathlib import Path
import numpy as np


def bilinear(img, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 2); y0 = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 2)
    a = np.clip(x - x0, 0, 1); b = np.clip(y - y0, 0, 1); f = img.astype(np.float32)
    top = f[y0, x0] * (1 - a) + f[y0, x0 + 1] * a; bot = f[y0 + 1, x0] * (1 - a) + f[y0 + 1, x0 + 1] * a
    return top * (1 - b) + bot * b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('left'); ap.add_argument('right'); ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--k-deg', type=float, default=0.29)
    a = ap.parse_args()
    root = a.package.resolve(); here = (PATHS.repo / 'runtime/scripts')
    sys.path[:0] = [str(root / 'frame'), str(root / 'tools'), str(here)]
    from depth_model import offline_bridge as bridge
    from model_spherical import SphericalModel, W, H
    from depth_model.adapters import ResidentDSP
    F, C, N = 380.0, 308.0, 616
    L = np.fromfile(a.left, np.uint8).reshape(N, N); R = np.fromfile(a.right, np.uint8).reshape(N, N)
    # current path: pixel-centre resize (modelResize)
    xs = np.clip((np.arange(W) + .5) * N / W - .5, 0, N - 1); ys = np.clip((np.arange(H) + .5) * N / H - .5, 0, N - 1)
    gx, gy = np.meshgrid(xs, ys)
    pin = [np.rint(bilinear(img, gx, gy)).astype(np.uint8) for img in (L, R)]
    m = SphericalModel(a.k_deg)
    sx, sy, ok = m.spherical_to_pinhole_map(F, C, C)
    inside = ok & (sx >= 0) & (sx <= N - 1) & (sy >= 0) & (sy <= N - 1)
    sph = [np.where(inside, np.rint(bilinear(img, sx, sy)), 0).astype(np.uint8) for img in (L, R)]
    program = 'frame/native_program/examples/primary-v2.nprg'; image = 'frame/build/resident-primary.qdrx'
    zero_h = np.zeros((1, 2, H, W), np.float32); zero_t = np.zeros((1, 2, 4, 4), np.float32)
    with ResidentDSP(root / image, root / 'recovered/primary/constants/constant_pool_5.bin',
                     engine=root / bridge.QNN_ENGINE, bootstrap=root / bridge.QNN_BOOTSTRAP,
                     prepared_program=root / program) as dsp:
        out = {}
        for name, imgs in (('pinhole', pin), ('spherical', sph)):
            o = dsp.run([imgs[0], imgs[1], zero_h, zero_t])
            out[name] = (o[1].reshape(2, H, W)[0], o[0].reshape(2, H, W)[0])
    # pinhole path depth on its own grid
    dp, up = out['pinhole']; zp = np.where((dp > 0) & (up < 89), 24.3048 * W / N / np.maximum(dp, 1e-6), 0)
    # spherical depth (z in rect frame) on its grid
    ds, us = out['spherical']
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    P = m.triangulate(uu, vv, ds)
    zs = np.where((ds > 0) & (us < 89) & inside, P[2], 0)
    # same rays: spherical pixel -> pinhole pixel -> pinhole-path model pixel
    px = np.clip(np.rint((sx + .5) * W / N - .5).astype(int), 0, W - 1); py = np.clip(np.rint((sy + .5) * H / N - .5).astype(int), 0, H - 1)
    zp_on_s = zp[py, px]
    both = (zs > 0.25) & (zs < 5) & (zp_on_s > 0.25) & (zp_on_s < 5) & inside
    rel = np.abs(zs[both] - zp_on_s[both]) / zp_on_s[both]
    print(f'k={a.k_deg} deg/px: spherical valid {np.mean((zs>0.25)&(zs<5)&inside):.3f} (of covered), pinhole valid {np.mean((zp>0.25)&(zp<5)):.3f}')
    print(f'same-ray comparison on {both.sum()} px: median rel diff {np.median(rel):.4f}, p90 {np.percentile(rel,90):.4f}, '
          f'median z spherical {np.median(zs[both]):.3f} m vs pinhole {np.median(zp_on_s[both]):.3f} m')
    print(f'uncertainty median: spherical {np.median(us[inside]):.1f}, pinhole {np.median(up):.1f}')
    np.savez_compressed('/tmp/model_spherical_check.npz', sph_left=sph[0], pin_left=pin[0], zs=zs, zp=zp, us=us, up=up)


if __name__ == '__main__':
    main()
