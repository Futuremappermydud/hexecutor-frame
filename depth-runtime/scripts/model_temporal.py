"""Model-style temporal state for the stereo model (hidden + t inputs).

- No history (first frame, lost pose, reset): hidden = 0 and t = 0.
- saveFrame: each disparity plane is 2x2 box-downsampled and its values halved.
- Next frame: every saved pixel is triangulated in its own eye, moved by the
  rigid motion (previous rectified-left camera -> current), projected into both
  current eyes at full resolution, and its disparity uL - uR is max-splatted into
  a half-resolution buffer of its own eye (holes stay 0), then nearest-upsampled.
  Values are full-resolution model-pixel disparities, like the model output.
- t[e] = T_e(now) - T_e(prev) as column-major 4x4. The graph only uses it as a
  gate (mean|t| < ~7.5e-4 -> 0.8, else 0.5), so the relative motion stands in
  for the Model's world-frame poses.

Geometry: rectified pinhole images (rect_w x rect_h, focal, cx, cy, baseline)
resized to the model's 256x320 with independent pixel-center scales.
"""
import numpy as np

W, H = 256, 320


class ModelTemporalState:
    def __init__(self, focal=380.0, cx=308.0, cy=308.0, baseline=24.3048 / 380.0,
                 rect_w=616, rect_h=616, pyramid=1, spherical_deg=None):
        self.f, self.cx, self.cy, self.b = float(focal), float(cx), float(cy), float(baseline)
        self.sx, self.sy = W / rect_w, H / rect_h
        self.p = pyramid
        # spherical_deg: model images are Model-style rectified spherical
        # (model_spherical.SphericalModel) instead of the resized pinhole.
        self.sph = None
        if spherical_deg:
            from model_spherical import SphericalModel
            self.sph = SphericalModel(spherical_deg, baseline)
        self.prev = None
        self.stats = {}

    def reset(self):
        self.prev = None

    def save(self, disparity):
        """disparity: [2,H,W] (left, right) model output planes."""
        d = np.asarray(disparity, np.float32).reshape(2, H, W)
        k = 1 << self.p
        low = d.reshape(2, H // k, k, W // k, k).mean(axis=(2, 4))
        self.prev = np.ascontiguousarray(low / k, np.float32)

    def inputs(self, motion):
        """motion: None or (R[3,3], t[3]) mapping previous rectified-left camera
        coordinates (metres, x right, y down, z forward) to current ones.
        Returns hidden [1,2,H,W] and t [1,2,4,4] float32."""
        hidden = np.zeros((1, 2, H, W), np.float32)
        tmat = np.zeros((1, 2, 4, 4), np.float32)
        if self.prev is None or motion is None:
            self.stats = {'history': False}
            return hidden, tmat
        R = np.asarray(motion[0], np.float64).reshape(3, 3)
        t = np.asarray(motion[1], np.float64).reshape(3)
        bvec = np.array([self.b, 0.0, 0.0])
        for e, te in enumerate((t, R @ bvec + t - bvec)):
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, te
            tmat[0, e] = (M - np.eye(4)).T  # row-major storage of column-major M - I
        k = 1 << self.p
        h, w = H // k, W // k
        splat = 0
        for e in (0, 1):
            low = self.prev[e]
            i, j = np.nonzero(low > 0.25 / k)
            d = low[i, j].astype(np.float64) * k        # full-resolution model disparity
            u = j * k + (k - 1) / 2.0                   # block centre, full-res model pixels
            v = i * k + (k - 1) / 2.0
            if self.sph is not None:
                P = R @ self.sph.triangulate(u, v, d, right_image=bool(e)) + t[:, None]
                P = P[:, np.isfinite(P).all(0) & (P[2] > 0.05)]
                ul, ur, vv, disp = self.sph.project(P)
                col = np.floor(ur if e else ul).astype(np.int64)
                row = np.floor(vv).astype(np.int64)
            else:
                xr = (u + 0.5) / self.sx - 0.5
                yr = (v + 0.5) / self.sy - 0.5
                z = self.f * self.b / (d / self.sx)
                x = (xr - self.cx) * z / self.f + (self.b if e else 0.0)
                y = (yr - self.cy) * z / self.f
                P = R @ np.stack([x, y, z]) + t[:, None]
                ok = P[2] > 0.05
                P = P[:, ok]
                ulr = self.f * P[0] / P[2] + self.cx
                urr = self.f * (P[0] - self.b) / P[2] + self.cx
                vr = self.f * P[1] / P[2] + self.cy
                disp = (ulr - urr) * self.sx
                col = np.floor(((urr if e else ulr) + 0.5) * self.sx - 0.5).astype(np.int64)
                row = np.floor((vr + 0.5) * self.sy - 0.5).astype(np.int64)
            inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            col, row, disp = col[inb] >> self.p, row[inb] >> self.p, disp[inb]
            buf = np.zeros(h * w, np.float32)
            order = np.argsort(disp, kind='stable')     # ascending: last write = max
            buf[(row * w + col)[order]] = disp[order]
            splat += int((buf > 0).sum())
            hidden[0, e] = np.repeat(np.repeat(buf.reshape(h, w), k, 0), k, 1)
        self.stats = {'history': True, 'splat_fraction': splat / (2 * h * w),
                      'mean_abs_t': float(np.abs(tmat).mean())}
        return hidden, tmat
