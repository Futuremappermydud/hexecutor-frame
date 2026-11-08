#!/usr/bin/env python3
"""Evaluate placed metric depth against one fixed reference depth map."""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(reference, candidate, focal=380.0, boundary_threshold=0.08):
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError('depth shapes differ')
    valid_reference = np.isfinite(reference) & (reference > 0)
    valid_candidate = np.isfinite(candidate) & (candidate > 0)
    overlap = valid_reference & valid_candidate
    if not overlap.any():
        raise ValueError('no overlapping depth')
    errors = np.abs(candidate[overlap] - reference[overlap])
    reference_edges = np.zeros(reference.shape, bool)
    candidate_edges = np.zeros(candidate.shape, bool)
    for source, valid, edges in ((reference, valid_reference, reference_edges),
                                 (candidate, valid_candidate, candidate_edges)):
        horizontal = valid[:, 1:] & valid[:, :-1] & (np.abs(source[:, 1:] - source[:, :-1]) > boundary_threshold)
        vertical = valid[1:, :] & valid[:-1, :] & (np.abs(source[1:, :] - source[:-1, :]) > boundary_threshold)
        edges[:, 1:] |= horizontal
        edges[1:, :] |= vertical
        edges[:, 1:] |= valid[:, 1:] ^ valid[:, :-1]
        edges[1:, :] |= valid[1:, :] ^ valid[:-1, :]
    boundary_px = None
    if reference_edges.any() and candidate_edges.any():
        reference_points = np.argwhere(reference_edges)
        candidate_points = np.argwhere(candidate_edges)
        distances = []
        for point in reference_points:
            distances.append(np.min(np.sum((candidate_points - point) ** 2, axis=1)) ** 0.5)
        boundary_px = float(np.mean(distances))
    return {
        'mean_abs_z_m': float(errors.mean()),
        'p95_abs_z_m': float(np.percentile(errors, 95)),
        'signed_z_bias_m': float(np.mean(candidate[overlap] - reference[overlap])),
        'completeness': float(overlap.sum() / valid_reference.sum()),
        'false_fill_fraction': float((~valid_reference & valid_candidate).sum() / max((~valid_reference).sum(), 1)),
        'boundary_displacement_px': boundary_px,
        'reference_boundary_pixels': int(reference_edges.sum()),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--height', type=int, default=616)
    parser.add_argument('--width', type=int, default=616)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    shape = (args.height, args.width)
    reference = np.fromfile(args.reference, '<f4').reshape(shape)
    candidate = np.fromfile(args.candidate, '<f4').reshape(shape)
    result = evaluate(reference, candidate)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
