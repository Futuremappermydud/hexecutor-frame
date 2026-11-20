"""Model-style rectified spherical camera for the stereo model.

Model pixel (u, v), W=256 x H=320, angular step k (rad/px, same on both axes):
  theta = (u - (W-1)/2) * k   azimuth inside the epipolar plane
  phi   = (v - (H-1)/2) * k   elevation of the epipolar plane about the baseline (+x)
  ray   ~ (tan theta, sin phi, cos phi) in the rectified left frame
Disparity d (model px) = (thetaL - thetaR) / k; with baseline B along +x,
  rho = B / (tan thetaL - tan thetaR),  P = rho * (tan thetaL, sin phi, cos phi).
"""
import numpy as np

W, H = 256, 320


class SphericalModel:
    def __init__(self, k_deg=0.29, baseline=24.3048 / 380.0):
        self.k = np.radians(k_deg)
        self.b = baseline
        self.cu, self.cv = (W - 1) / 2.0, (H - 1) / 2.0

    def angles(self, u, v):
        return (np.asarray(u) - self.cu) * self.k, (np.asarray(v) - self.cv) * self.k

    def pixel(self, theta, phi):
        return theta / self.k + self.cu, phi / self.k + self.cv

    def triangulate(self, u, v, d, right_image=False):
        """Point in the rectified left frame for a left- (or right-) image pixel."""
        theta, phi = self.angles(u, v)
        if right_image:
            tr, tl = theta, theta + d * self.k
        else:
            tl, tr = theta, theta - d * self.k
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = self.b / (np.tan(tl) - np.tan(tr))
        return np.stack([rho * np.tan(tl), rho * np.sin(phi), rho * np.cos(phi)])

    def project(self, P):
        """Left column, right column, row and disparity of left-frame points [3,N]."""
        rho = np.hypot(P[1], P[2])
        phi = np.arctan2(P[1], P[2])
        tl = np.arctan2(P[0], rho)
        tr = np.arctan2(P[0] - self.b, rho)
        ul, v = self.pixel(tl, phi)
        ur, _ = self.pixel(tr, phi)
        return ul, ur, v, (tl - tr) / self.k

    def pinhole_to_spherical_map(self, focal, cx, cy, rect_w, rect_h):
        """For each pinhole pixel: model (u, v) along the same ray."""
        x, y = np.meshgrid(np.arange(rect_w), np.arange(rect_h))
        rx, ry = (x - cx) / focal, (y - cy) / focal
        theta = np.arctan2(rx, np.hypot(ry, 1.0))
        phi = np.arctan2(ry, 1.0)
        return self.pixel(theta, phi)

    def spherical_to_pinhole_map(self, focal, cx, cy):
        """For each model pixel: pinhole (x, y) along the same ray (for resampling)."""
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        theta, phi = self.angles(u, v)
        d = np.stack([np.tan(theta), np.sin(phi), np.cos(phi)])
        return focal * d[0] / d[2] + cx, focal * d[1] / d[2] + cy, d[2] > 0
