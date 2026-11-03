#!/usr/bin/env python3
"""Plot evaluate_occlusion_completion outputs; requires numpy and matplotlib."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path, dtype):
    data = np.fromfile(path, dtype=dtype)
    if data.size != 616 * 616:
        raise ValueError(f"Expected 616x616 pixels: {path}")
    return data.reshape(616, 616)


def mesh_coverage(depth):
    valid = np.isfinite(depth) & (depth >= 0.25) & (depth <= 5)
    lo = np.full((307, 307), 5.0)
    hi = np.zeros_like(lo)
    supported = np.ones_like(lo, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            samples = depth[dy:614+dy:2, dx:614+dx:2]
            supported &= valid[dy:614+dy:2, dx:614+dx:2]
            lo = np.minimum(lo, samples)
            hi = np.maximum(hi, samples)
    supported &= (hi - lo) <= np.maximum(0.04, lo * 0.04)
    coverage = np.zeros_like(valid)
    coverage[:614, :614] = supported.repeat(2, axis=0).repeat(2, axis=1)
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("baseline_prefix", type=Path)
    parser.add_argument("new_prefix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    left = read(args.left, np.uint8)
    old = read(f"{args.baseline_prefix}-old-surface.raw", np.float32)
    new = read(f"{args.new_prefix}-completed.raw", np.float32)
    measured = read(f"{args.new_prefix}-measured.raw", np.float32)
    old_mesh, new_mesh = mesh_coverage(old), mesh_coverage(new)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].imshow(left, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Captured left camera (rectified luma)")
    for axis, depth, title in (
        (axes[0, 1], old, "Previous surface"),
        (axes[0, 2], new, "New surface with plane completion"),
    ):
        shown = axis.imshow(np.ma.masked_less_equal(depth, 0), cmap="turbo_r", vmin=0.25, vmax=5)
        axis.set_facecolor("black")
        axis.set_title(f"{title}\n{100*np.count_nonzero(depth > 0)/depth.size:.1f}% depth coverage")
    fig.colorbar(shown, ax=axes[0, 1:], shrink=0.7, label="Camera depth (metres)")
    for axis, mask, title in (
        (axes[1, 0], old_mesh, "Previous supported mesh cells"),
        (axes[1, 1], new_mesh, "New supported mesh cells"),
    ):
        axis.imshow(left, cmap="gray", vmin=0, vmax=255)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 0.8, 1, 0.65]
        axis.imshow(overlay)
        axis.set_title(f"{title}\n{100*mask.mean():.1f}% camera-view coverage")
    added = (new > 0) & ~(measured > 0)
    axes[1, 2].imshow(left, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*added.shape, 4))
    overlay[added] = [1, 0.6, 0, 0.65]
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Estimated interiors (orange)\nIncludes enclosed-hole filling")
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle("Same camera pair: stereo confidence + image-guided plane completion\n"
                 "Coverage diagnostic only; no ground-truth depth or display-eye alignment", fontsize=13)
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
