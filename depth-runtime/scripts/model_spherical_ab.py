#!/usr/bin/env python3
"""A/B of Model model inputs on one FRAME_DEPTH_RECORD capture made with
--model-spherical (sequence.bin = resized pinhole, sequence.bin.sph = spherical,
same captures and motion). Both run with temporal state in their own geometry
and are converted to the 616x616 pinhole depth the TSDF consumes."""

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

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

W, H, N, F, C = 256, 320, 616, 380.0, 308.0
B = 24.3048 / 380.0


def support(r):
    pad = np.pad(r, 1); s = np.zeros(r.shape, int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                n = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                s += (n > 0) & (np.abs(n - r) <= np.maximum(0.04, 0.04 * r))
    return np.where((r > 0) & (s >= 3), r, 0)


def pinhole_depth(d, u):
    xs = np.arange(W)[None, :]
    z = np.where(d > 0, 24.3048 * W / N / np.maximum(d, 1e-6), 0)
    ok = (d > 0) & (u < 89) & (z >= .25) & (z <= 5) & (xs - d >= 2) & (xs - d <= W - 3)
    ok[:2] = ok[-2:] = False; ok[:, :2] = ok[:, -2:] = False
    z = support(np.where(ok, z, 0))
    mx = np.minimum(W - 1, ((np.arange(N) + .5) * W / N).astype(int)); my = np.minimum(H - 1, ((np.arange(N) + .5) * H / N).astype(int))
    return z[my][:, mx]


def spherical_depth(d, u, m, idx, cosel):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    al = (uu - (W - 1) / 2) * m.k; ar = al - d * m.k
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = B / (np.tan(al) - np.tan(ar))
    z = rho * np.cos((vv - (H - 1) / 2) * m.k)
    ok = (d > 0) & (u < 89) & np.isfinite(rho) & (rho > 0) & (z >= .25) & (z <= 5) & (uu - d >= 2)
    ok[:2] = ok[-2:] = False; ok[:, :2] = ok[:, -2:] = False
    r = support(np.where(ok, rho, 0)).ravel()
    out = np.zeros(N * N, np.float32); valid = idx >= 0
    out[valid] = r[idx[valid]] * cosel[valid]
    return out.reshape(N, N)


def warp_depth(prev, R, t):
    """Forward-warp a 616 pinhole depth map by P_now = R P_prev + t (nearest, z-buffer)."""
    ys, xs = np.nonzero(prev > 0); z = prev[ys, xs]
    P = np.stack([(xs - C) / F * z, (ys - C) / F * z, z]); Q = R @ P + t[:, None]
    ok = Q[2] > .2; Q = Q[:, ok]
    u = np.rint(F * Q[0] / Q[2] + C).astype(int); v = np.rint(F * Q[1] / Q[2] + C).astype(int)
    inb = (u >= 0) & (u < N) & (v >= 0) & (v < N); u, v, zz = u[inb], v[inb], Q[2][inb]
    out = np.full(N * N, np.inf); np.minimum.at(out, v * N + u, zz)
    out[np.isinf(out)] = 0; return out.reshape(N, N)


def lr(dl, dr):
    ys, xs = np.nonzero(dl > 0.5); xr = np.rint(xs - dl[ys, xs]).astype(int); ok = (xr >= 0) & (xr < W)
    return float((np.abs(dl[ys[ok], xs[ok]] - dr[ys[ok], xr[ok]]) < 1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording'); ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--k-deg', type=float, default=0.29); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = a.package.resolve(); here = (PATHS.repo / 'depth-runtime/scripts')
    sys.path[:0] = [str(root / 'frame'), str(root / 'tools'), str(here)]
    from depth_model import offline_bridge as bridge
    from model_temporal import ModelTemporalState
    from model_temporal_replay import frames
    from model_spherical import SphericalModel
    from depth_model.adapters import ResidentDSP
    pin = list(frames(a.recording)); sph = list(frames(a.recording + '.sph'))
    assert len(pin) == len(sph)
    m = SphericalModel(a.k_deg, B)
    U, V = m.pinhole_to_spherical_map(F, C, C, N, N)
    ui, vi = np.rint(U).astype(int), np.rint(V).astype(int)
    idx = np.where((ui >= 0) & (ui < W) & (vi >= 0) & (vi < H), vi * W + ui, -1).ravel()
    cosel = np.cos(np.arctan2((np.arange(N)[:, None] - C) / F, 1.0) + 0 * np.arange(N)[None, :]).ravel()
    program = 'frame/native_program/examples/primary-v2.nprg'; image = 'frame/build/resident-primary.qdrx'
    res = {}; keep = {}; saved = set(np.linspace(20, len(pin) - 1, 5).astype(int).tolist())
    with ResidentDSP(root / image, root / 'recovered/primary/constants/constant_pool_5.bin',
                     engine=root / bridge.QNN_ENGINE, bootstrap=root / bridge.QNN_BOOTSTRAP,
                     prepared_program=root / program) as dsp:
        depths = {}
        for mode, seq in (('pinhole', pin), ('spherical', sph)):
            state = ModelTemporalState(spherical_deg=a.k_deg if mode == 'spherical' else None)
            valid, lrs, flick, maps = [], [], [], []
            prev = None
            for i, (_, imgs, motion) in enumerate(seq):
                h, t = state.inputs(motion)
                o = dsp.run([imgs[0], imgs[1], h, t])
                d = o[1].reshape(2, H, W); u = o[0].reshape(2, H, W)
                state.save(d)
                z = pinhole_depth(d[0], u[0]) if mode == 'pinhole' else spherical_depth(d[0], u[0], m, idx, cosel)
                valid.append(float((z > 0).mean())); lrs.append(lr(d[0], d[1]))
                if prev is not None and motion is not None:
                    w = warp_depth(prev, *motion); both = (w > 0) & (z > 0)
                    if both.sum() > 2000: flick.append(float(np.median(np.abs(z[both] - w[both]) / w[both])))
                prev = z; maps.append(z)
                if i in saved: keep[f'{mode}{i}_z'] = z.astype(np.float16); keep[f'{mode}{i}_img'] = imgs[0]
            depths[mode] = maps
            res[mode] = {'valid_fraction_616': round(float(np.mean(valid)), 4), 'lr_consistent': round(float(np.mean(lrs)), 4),
                         'flicker_median_rel': round(float(np.median(flick)), 4), 'flicker_pairs': len(flick)}
            print(mode, json.dumps(res[mode]), flush=True)
    agree = []
    for zp, zs in zip(depths['pinhole'], depths['spherical']):
        both = (zp > 0) & (zs > 0)
        if both.sum() > 2000: agree.append(float(np.median(np.abs(zs[both] - zp[both]) / zp[both])))
    res['same_ray_median_rel_diff'] = round(float(np.median(agree)), 4)
    res['median_depth_m'] = {k: round(float(np.median(np.concatenate([z[z > 0] for z in v[::10]]))), 3) for k, v in depths.items()}
    print(json.dumps({k: res[k] for k in ('same_ray_median_rel_diff', 'median_depth_m')}))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'ab.json').write_text(json.dumps(res, indent=1) + '\n'); np.savez_compressed(a.out / 'ab_frames.npz', **keep)


if __name__ == '__main__':
    main()
