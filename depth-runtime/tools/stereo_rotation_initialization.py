"""Offline robust camera-center initialization with a supplied rotation."""
import numpy as np
from scipy.optimize import least_squares


def initialize_center(world, pixels, intrinsic, camera_to_world, minimum_inliers=12):
    world, pixels = np.asarray(world, float), np.asarray(pixels, float)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsic).T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    directions = rays @ camera_to_world.T
    projectors = np.eye(3)[None]-directions[:, :, None]*directions[:, None, :]
    def solve(indices):
        a = projectors[indices].sum(axis=0)
        if np.linalg.cond(a) > 1e10:
            return None
        b = np.einsum('nij,nj->ni', projectors[indices], world[indices]).sum(axis=0)
        return np.linalg.solve(a, b)
    def residual(center):
        local = (world-center) @ camera_to_world
        q = local @ intrinsic.T
        z = np.where(np.abs(local[:, 2]) > 1e-8, local[:, 2], 1e-8)
        return q[:, :2]/z[:, None]-pixels, local[:, 2] > 0
    if len(world) < max(3, minimum_inliers):
        return None
    rng = np.random.default_rng(0)
    best, key = None, (-1, -np.inf)
    for _ in range(200):
        center = solve(rng.choice(len(world), 3, replace=False))
        if center is None: continue
        errors, positive = residual(center); norm = np.linalg.norm(errors, axis=1)
        inliers = positive & (norm <= 3.)
        score = (int(inliers.sum()), -float(np.minimum(norm, 3.).sum()))
        if score > key:best, key = inliers, score
    if best is None or int(best.sum()) < minimum_inliers:
        return None
    center = solve(np.flatnonzero(best))
    if center is None:return None
    fit = least_squares(lambda x: residual(x)[0][best].ravel(), center, loss='soft_l1', f_scale=1., max_nfev=30)
    errors, positive = residual(fit.x)
    if not fit.success or np.sum(positive & (np.linalg.norm(errors, axis=1) <= 3.)) < minimum_inliers:
        return None
    return fit.x
