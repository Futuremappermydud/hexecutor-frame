"""Explicit offline equal-track robust objective; no independence claim."""
import numpy as np


def equal_track_soft_l1(track_ids):
    """Unit-mean weights, equal total per surviving track, four scalars/row.

    Weight rho after evaluating it at the original pixel robust scale. Scaling
    residuals before soft-L1 would instead change each track's outlier threshold.
    """
    ids = np.asarray(track_ids)
    if ids.ndim != 1 or not len(ids):
        raise ValueError('nonempty one-dimensional track IDs required')
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = np.repeat(len(ids)/(len(counts)*counts[inverse]), 4)

    def loss(z):
        z = np.asarray(z)
        if z.shape != weights.shape:
            raise ValueError('four scalar residuals per observation required')
        root = np.sqrt(1.+z)
        return np.array([2.*(root-1.), 1./root, -.5/root**3])*weights
    return loss
