"""CPU IR rectify + RGB-prior splat + ZNCC unwrap. No DSP."""
import json
import os
from collections import namedtuple
from pathlib import Path

import numpy as np

W_IR, H_IR = 1056, 1024
W, H = 256, 320
K = np.radians(0.29)
MAX_D, HALF, WIN, PERIOD, BLOCK = 72, 5, 1.7, 3.45, 16
MIN_SCORE = float(os.environ.get("IR_MIN_SCORE", "0.35"))
MIN_INLIERS = int(os.environ.get("IR_MIN_INLIERS", "80"))
MAX_UNC = int(os.environ.get("IR_MAX_UNC", "200"))
RECT_W = RECT_H = 616
RECT_F, RECT_CX, RECT_CY = 380.0, 308.0, 308.0
FOOT_N, FOOT_CELL = 64, 0.10  # must match kIrFootprintSize / kIrFootprintCell
FOOT_MIN_POINTS = 3
FRONT_MARGIN = 0.06  # dots this far in front of the plane are foreground, not noise
MIN_MINOR_STD = 0.06  # about a 20 cm extent; thinner strips leave the normal unconstrained

PlaneFit = namedtuple("PlaneFit", "n c cen radius inliers count rms u v")


class KB4:
    def __init__(self, intr):
        self.fx, self.fy, self.cx, self.cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        self.k = [intr["k1"], intr["k2"], intr["k3"], intr["k4"]]

    def project(self, P):
        x, y, z = P
        r = np.hypot(x, y)
        th = np.arctan2(r, z)
        t2 = th * th
        rd = th * (1 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(r > 1e-12, rd / r, 1.0)
        return self.fx * x * s + self.cx, self.fy * y * s + self.cy


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_rig(path=None):
    path = Path(path or os.environ.get("IR_CALIB", "/persist/xrservice.json"))
    cams = json.loads(path.read_text())["cameras"]
    rig = {}
    for i in (0, 1):
        e = cams[i]["extrinsics"]
        px, pz = np.array(e["plus_x"], float), np.array(e["plus_z"], float)
        R = np.stack([px, np.cross(pz, px), pz], 1)
        c = np.array(e["position"], float) / 1000.0
        rig[cams[i]["sourceCamera"]] = (R, c, KB4(cams[i]["intrinsics"][0]))
    return rig


class Rectifier:
    def __init__(self, rig=None):
        pitch_deg = float(os.environ.get("IR_PITCH", "0.25"))
        yaw = np.radians(float(os.environ.get("IR_YAW_DEG", "0")))
        roll = np.radians(float(os.environ.get("IR_ROLL_DEG", "0.4")))
        rig = rig or load_rig()
        (self.RL0, cL, self.kl), (self.RR0, cR, self.kr) = rig["slam_left"], rig["slam_right"]
        b = cR - cL
        self.baseline = float(np.linalg.norm(b))
        self.cL = cL
        x = b / self.baseline
        z = self.RL0[:, 2] + self.RR0[:, 2]
        z -= x * (x @ z)
        z /= np.linalg.norm(z)
        self.Rrect = np.stack([x, np.cross(z, x), z], 1)
        c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
        self.RL = self.RL0.T @ self.Rrect
        cy, sy, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
        self.RR = (self.RR0.T @ self.Rrect @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                   @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                   @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]))
        self.k = K
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        th, ph = (u - (W - 1) / 2) * self.k, (v - (H - 1) / 2) * self.k
        rays = np.stack([np.tan(th), np.sin(ph), np.cos(ph)]).reshape(3, -1)
        self.mapL = self._map(self.RL, self.kl, rays)
        self.mapR = self._map(self.RR, self.kr, rays)

    def _map(self, R, lens, rays):
        mx, my = lens.project(R @ rays)
        return mx.reshape(H, W), my.reshape(H, W)

    def _sample(self, img, m):
        mx, my = m
        x0, y0 = np.floor(mx).astype(int), np.floor(my).astype(int)
        fx, fy = mx - x0, my - y0
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < W_IR - 1) & (y0 < H_IR - 1)
        x0, y0 = np.clip(x0, 0, W_IR - 2), np.clip(y0, 0, H_IR - 2)
        a = img.astype(np.float32)
        out = (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
               + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)
        return np.where(ok, out, 0), ok

    def rectify(self, left, right):
        l, okl = self._sample(left, self.mapL)
        r, okr = self._sample(right, self.mapR)
        return l, r, okl, okr

    def triangulate(self, u, v, d):
        th, ph = (np.asarray(u) - (W - 1) / 2) * self.k, (np.asarray(v) - (H - 1) / 2) * self.k
        tr = th - np.asarray(d) * self.k
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = self.baseline / (np.tan(th) - np.tan(tr))
        return np.stack([rho * np.tan(th), rho * np.sin(ph), rho * np.cos(ph)])


def box(a, h=HALF):
    c = np.cumsum(np.cumsum(np.pad(a, ((h + 1, h), (h + 1, h))), 0), 1)
    k = 2 * h + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def zncc_volume(A, B, okA, okB):
    mA, mA2 = box(A), box(A * A)
    vol = np.full((MAX_D + 1, H, W), -1.0, np.float32)
    for d in range(MAX_D + 1):
        Bs = np.full_like(B, np.nan)
        oks = np.zeros_like(okB)
        Bs[:, d:] = B[:, :W - d]
        oks[:, d:] = okB[:, :W - d]
        valid = oks & okA
        Bz = np.where(valid, Bs, 0)
        mB, mB2, mAB = box(Bz), box(Bz * Bz), box(np.where(valid, A, 0) * Bz)
        cov = mAB - mA * mB
        den = np.sqrt(np.maximum(mA2 - mA * mA, 1e-6) * np.maximum(mB2 - mB * mB, 1e-6))
        s = cov / den
        s[box(valid.astype(float)) < 0.999] = -1
        vol[d] = s
    return vol


def subpixel(vol, idx):
    below = np.take_along_axis(vol, np.clip(idx - 1, 0, MAX_D)[None], 0)[0]
    peak = np.take_along_axis(vol, idx[None], 0)[0]
    above = np.take_along_axis(vol, np.clip(idx + 1, 0, MAX_D)[None], 0)[0]
    den = below - 2 * peak + above
    den_safe = np.where(np.abs(den) > 1e-6, den, 1.0)
    return np.where(np.abs(den) > 1e-6, idx + 0.5 * (below - above) / den_safe, idx.astype(float))


def search(vol, dd, prior, has):
    win = np.abs(dd - np.nan_to_num(prior, nan=-99)[None]) <= WIN
    masked = np.where(win, vol, -2)
    idx = masked.argmax(0)
    best = masked.max(0)
    below = np.take_along_axis(vol, np.clip(idx - 1, 0, MAX_D)[None], 0)[0]
    above = np.take_along_axis(vol, np.clip(idx + 1, 0, MAX_D)[None], 0)[0]
    peak = (best >= below) & (best >= above) & (idx > 0) & (idx < MAX_D)
    return subpixel(vol, idx), has & peak & (best >= MIN_SCORE)


def unwrapped_offset(off, ok):
    gh, gw = -(-H // BLOCK), -(-W // BLOCK)
    z = np.zeros((gh, gw), complex)
    n = np.zeros((gh, gw))
    v, u = np.nonzero(ok)
    np.add.at(z, (v // BLOCK, u // BLOCK), np.exp(2j * np.pi * off[v, u] / PERIOD))
    np.add.at(n, (v // BLOCK, u // BLOCK), 1)
    good = (n >= 8) & (np.abs(z) >= 0.4 * np.maximum(n, 1))
    phase = np.angle(z) / (2 * np.pi) * PERIOD
    val = np.full((gh, gw), np.nan)
    if not good.any():
        return np.zeros((H, W))
    start = np.unravel_index(np.argmax(np.where(good, n, -1)), n.shape)
    val[start] = phase[start]
    queue = [start]
    while queue:
        i, j = queue.pop(0)
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < gh and 0 <= b < gw and good[a, b] and np.isnan(val[a, b]):
                val[a, b] = val[i, j] + (phase[a, b] - val[i, j] + PERIOD / 2) % PERIOD - PERIOD / 2
                queue.append((a, b))
    while np.isnan(val).any():
        p = np.pad(val, 1, constant_values=np.nan)
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        cnt = np.isfinite(nb).sum(0)
        fill = np.where(cnt > 0, np.nansum(nb, 0) / np.maximum(cnt, 1), np.nan)
        val = np.where(np.isnan(val), fill, val)
    yc, xc = (np.arange(H) + 0.5) / BLOCK - 0.5, (np.arange(W) + 0.5) / BLOCK - 0.5
    y0, x0 = np.clip(np.floor(yc).astype(int), 0, gh - 2), np.clip(np.floor(xc).astype(int), 0, gw - 2)
    fy, fx = np.clip(yc - y0, 0, 1)[:, None], np.clip(xc - x0, 0, 1)[None]
    Y0, X0 = y0[:, None], x0[None]
    return ((val[Y0, X0] * (1 - fx) + val[Y0, X0 + 1] * fx) * (1 - fy)
            + (val[Y0 + 1, X0] * (1 - fx) + val[Y0 + 1, X0 + 1] * fx) * fy)


def entries(doc, key):
    return [np.array(e[key], float) for dev in doc["devices"] for hist in dev.values()
            if isinstance(hist, list) for e in hist if isinstance(e, dict) and key in e]


def medoid(es):
    def ang(a, b):
        return 2 * np.arccos(min(1.0, abs(float(a[3:] @ b[3:]))))
    return es[int(np.argmin([sum(ang(a, b) for b in es) for a in es]))]


def stereo_entry(es, policy=None):
    """Same choice as loadStereoTransform in src/depth_pipeline.cpp, so the prior and
    the output live in the frame the rectifier actually used."""
    policy = policy or os.environ.get("FRAME_STEREO_CALIBRATION_POLICY", "medoid")
    return es[-1] if policy == "latest" or len(es) <= 2 else medoid(es)


def rgb_basis(rl):
    R = quat_to_R(*rl[3:])
    t = rl[:3]
    x = -(R.T @ t)
    x /= np.linalg.norm(x)
    z = R[2] + np.array([0, 0, 1.0])
    z /= np.linalg.norm(z)
    z -= x * (x @ z)
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1), float(np.linalg.norm(t))


def load_rgb_chain(path=None):
    path = Path(path or os.environ.get(
        "FRAME_STEREO_CALIBRATION",
        "/home/steamos/.config/openvr/config/cv/xrservice/colorPassthroughCalibrationHistory.json"))
    doc = json.loads(path.read_text())
    rl = stereo_entry(entries(doc, "passthroughCam0FromPassthroughCam1Extrinsics"))
    fr = entries(doc, "frontCam0FromPassthroughCam0Extrinsics")[-1]
    return quat_to_R(*rl[3:]), rl[:3], quat_to_R(*fr[3:]), fr[:3], rgb_basis(rl)


def rgb_disparity_to_ir(disp, unc, rect, chain):
    R_rl, t_rl, R_fr, t_fr, (Rrgb, b_rgb) = chain
    ok = (unc < MAX_UNC) & (disp > 0.05)
    us = np.arange(2 * W) / 2 - 0.25
    vs = np.arange(2 * H) / 2 - 0.25
    U, V = np.meshgrid(us, vs)
    u0, v0 = np.clip(np.floor(U).astype(int), 0, W - 2), np.clip(np.floor(V).astype(int), 0, H - 2)
    fu, fv = np.clip(U - u0, 0, 1), np.clip(V - v0, 0, 1)
    corners = [(v0, u0, (1 - fu) * (1 - fv)), (v0, u0 + 1, fu * (1 - fv)),
               (v0 + 1, u0, (1 - fu) * fv), (v0 + 1, u0 + 1, fu * fv)]
    D = sum(disp[a, b] * w for a, b, w in corners)
    OK = np.all([ok[a, b] for a, b, _ in corners], 0)
    spread = np.max([disp[a, b] for a, b, _ in corners], 0) - np.min([disp[a, b] for a, b, _ in corners], 0)
    OK &= (spread < 1.5) & (D > 0.05)
    th, ph = (U - (W - 1) / 2) * K, (V - (H - 1) / 2) * K
    rho = b_rgb / (np.tan(th) - np.tan(th - D * K))
    P = np.stack([rho * np.tan(th), rho * np.sin(ph), rho * np.cos(ph)])[:, OK]
    P = R_fr @ (R_rl @ (Rrgb @ P) + t_rl[:, None]) + t_fr[:, None]
    Q = rect.Rrect.T @ (P - rect.cL[:, None])
    r = np.hypot(Q[1], Q[2])
    tq, pq = np.arctan2(Q[0], r), np.arctan2(Q[1], Q[2])
    dq = (tq - np.arctan(np.tan(tq) - rect.baseline / r)) / K
    ui = np.round(tq / K + (W - 1) / 2).astype(int)
    vi = np.round(pq / K + (H - 1) / 2).astype(int)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (Q[2] > 0)
    out = np.full(H * W, -np.inf)
    np.maximum.at(out, vi[inb] * W + ui[inb], dq[inb])
    out = out.reshape(H, W)
    have = np.isfinite(out)
    out[~have] = np.nan
    return out


def fit_dominant_plane(pts, inlier_thresh=0.040, min_inliers=None, iterations=150, rng=None):
    """Vectorized RANSAC with a two-step least-squares refit. n.X = c with c > 0, so
    the normal faces away from the camera whatever the surface's tilt. Returns a
    PlaneFit, or None."""
    if min_inliers is None:
        min_inliers = MIN_INLIERS
    if len(pts) < min_inliers:
        print(f"[IrDotPlane] reject: pts {len(pts)} < {min_inliers}", flush=True)
        return None
    rng = rng or np.random.default_rng()
    triplets = rng.integers(0, len(pts), (iterations, 3))
    p0, p1, p2 = pts[triplets[:, 0]], pts[triplets[:, 1]], pts[triplets[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-6
    if not valid.any():
        return None
    normals = normals[valid] / norms[valid]
    p0 = p0[valid]

    dists = np.abs(np.einsum("kd,nd->kn", normals, pts) - np.einsum("kd,kd->k", normals, p0)[:, None])
    inlier_counts = (dists < inlier_thresh).sum(axis=1)
    best_idx = int(np.argmax(inlier_counts))
    if inlier_counts[best_idx] < min_inliers:
        print(f"[IrDotPlane] reject: best inliers {inlier_counts[best_idx]} < {min_inliers} (of {len(pts)} pts)", flush=True)
        return None

    inliers = dists[best_idx] < inlier_thresh
    for _ in range(2):
        pts_in = pts[inliers]
        cen = pts_in.mean(0)
        w, v = np.linalg.eigh((pts_in - cen).T @ (pts_in - cen))
        n = v[:, 0]
        c = float(np.dot(n, cen))
        refit = np.abs(pts @ n - c) < inlier_thresh
        if refit.sum() < min_inliers:
            break
        inliers = refit
    pts_in = pts[inliers]
    cen = pts_in.mean(0)
    w, v = np.linalg.eigh((pts_in - cen).T @ (pts_in - cen))
    n, u = v[:, 0], v[:, 2]
    c = float(np.dot(n, cen))
    if c < 0:
        n, c = -n, -c
    if c <= 0.25:
        print(f"[IrDotPlane] reject: distance c={c:.3f} <= 0.25", flush=True)
        return None
    minor_std = float(np.sqrt(max(w[1], 0.0) / len(pts_in)))
    if minor_std < MIN_MINOR_STD:
        print(f"[IrDotPlane] reject: surface extent std {minor_std:.3f} < {MIN_MINOR_STD}", flush=True)
        return None
    rms = float(np.sqrt(np.mean((pts_in @ n - c) ** 2)))
    radius = float(np.percentile(np.linalg.norm(pts_in - cen, axis=1), 95))
    return PlaneFit(n, c, cen, radius, inliers, int(inliers.sum()), rms, u, np.cross(n, u))


def plane_cells(q, fit):
    d = q - fit.cen
    return (np.floor(d @ fit.v / FOOT_CELL).astype(int) + FOOT_N // 2,
            np.floor(d @ fit.u / FOOT_CELL).astype(int) + FOOT_N // 2)


def plane_footprint(pts, fit):
    """FOOT_N^2 grid over the plane, indexed [v, u]: 0 unobserved, 1 dots on the plane,
    2 dots in front of it. Foreground dots are cast onto the plane along their camera
    ray, so a chair marks the part of the wall it hides from this view."""
    def count(q):
        iv, iu = plane_cells(q, fit)
        ok = (iv >= 0) & (iv < FOOT_N) & (iu >= 0) & (iu < FOOT_N)
        grid = np.zeros((FOOT_N, FOOT_N), int)
        np.add.at(grid, (iv[ok], iu[ok]), 1)
        return grid

    on_plane = count(pts[fit.inliers])
    along_n = pts @ fit.n
    front = ~fit.inliers & (along_n > 0.1) & (along_n < fit.c - FRONT_MARGIN)
    in_front = count(pts[front] * (fit.c / along_n[front])[:, None])
    good = on_plane >= FOOT_MIN_POINTS
    p = np.pad(good, 1)
    grown = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | good
    grid = grown.astype(np.uint8)
    grid[(in_front >= FOOT_MIN_POINTS) & (in_front > on_plane)] = 2
    return grid


def plane_depth(fit, grid):
    """Depth of the plane at every RGB pixel whose footprint cell has dots on it."""
    xx, yy = np.meshgrid(np.arange(RECT_W), np.arange(RECT_H))
    rx, ry = (xx - RECT_CX) / RECT_F, (yy - RECT_CY) / RECT_F
    denom = fit.n[0] * rx + fit.n[1] * ry + fit.n[2]
    zp = np.where(denom > 0.05, fit.c / np.maximum(denom, 0.05), 0.0)
    q = np.stack([rx * zp, ry * zp, zp], -1).reshape(-1, 3)
    iv, iu = plane_cells(q, fit)
    ok = (iv >= 0) & (iv < FOOT_N) & (iu >= 0) & (iu < FOOT_N)
    cell = np.zeros(len(q), np.uint8)
    cell[ok] = grid[iv[ok], iu[ok]]
    return np.where(cell.reshape(RECT_H, RECT_W) == 1, zp, 0.0).astype(np.float32)


def ir_points_to_rgb_depth(rect, u, v, d, chain, rng=None):
    """Returns depth (RECT_H x RECT_W, 0 = none), plus (PlaneFit, footprint) or None."""
    R_rl, t_rl, R_fr, t_fr, (Rrgb, _) = chain
    Q = rect.triangulate(u, v, d)
    P = rect.Rrect @ Q + rect.cL[:, None]
    Pr = R_fr.T @ (P - t_fr[:, None])
    Pl = R_rl.T @ (Pr - t_rl[:, None])
    Prgb = (Rrgb.T @ Pl).T  # N x 3
    z = Prgb[:, 2]
    ok = (z > 0.25) & (z < 5.0)
    x = RECT_F * Prgb[:, 0] / z + RECT_CX
    y = RECT_F * Prgb[:, 1] / z + RECT_CY
    ui = np.round(x).astype(int)
    vi = np.round(y).astype(int)
    inb = ok & (ui >= 0) & (ui < RECT_W) & (vi >= 0) & (vi < RECT_H)
    pts = Prgb[inb]
    fit = fit_dominant_plane(pts, rng=rng)
    plane_info = None
    depth = np.zeros((RECT_H, RECT_W), np.float32)
    stamp = np.ones(len(pts), bool)
    if fit is not None:
        grid = plane_footprint(pts, fit)
        plane_info = (fit, grid)
        depth = plane_depth(fit, grid)
        stamp = ~fit.inliers  # the fit already carries the on-plane dots
    return stamp_points(depth, ui[inb][stamp], vi[inb][stamp], pts[stamp, 2].astype(np.float32)), plane_info


def stamp_points(depth, ui, vi, z):
    """3x3 stamp, far to near, so a nearer dot wins over a farther value."""
    order = np.argsort(-z)
    ui, vi, z = ui[order], vi[order], z[order]
    for dy in (0, -1, 1):
        for dx in (0, -1, 1):
            vy = np.clip(vi + dy, 0, RECT_H - 1)
            ux = np.clip(ui + dx, 0, RECT_W - 1)
            cur = depth[vy, ux]
            take = (cur == 0.0) | ((dx == 0) & (dy == 0) & (z < cur))
            depth[vy[take], ux[take]] = z[take]
    return depth
