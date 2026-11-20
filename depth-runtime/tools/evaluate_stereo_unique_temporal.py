#!/usr/bin/env python3
"""Explicit offline uniqueness-gated temporal replay; baseline modules unchanged.

The wrapper applies the same optional matcher to stereo births, both temporal
propagations and stereo closure. Context changes are restored even on failure.
"""
import argparse
import json
from pathlib import Path
import stereo_direct_tracks as tracks
import evaluate_stereo_temporal_lens as temporal
import evaluate_stereo_long_trajectory as long
from stereo_unique_matching import refine_matches


def run(manifest, frozen, poses, config, exposure_offset_ms, external_tag_manifest=None, long_trajectory=False,
        fixed_rotation_initialization=False):
    if fixed_rotation_initialization and not long_trajectory:
        raise ValueError('fixed-rotation initialization ablation requires long-trajectory mode')
    old_refiner, old_builder = tracks.refine_matches, temporal.track_window
    def builder(frames):
        adjusted, counts = [], []
        for frame in frames:
            selected = refine_matches(frame['images'], frame['masks'], frame['stereo'])
            adjusted.append(dict(frame, stereo=selected))
            counts.append({'before': len(frame['stereo']), 'after': len(selected)})
        rows, diagnostics = old_builder(adjusted)
        diagnostics['uniqueness_stereo_birth_counts'] = counts
        return rows, diagnostics
    tracks.refine_matches, temporal.track_window = refine_matches, builder
    try:
        if long_trajectory:
            result = long.run(manifest, frozen, poses, config, exposure_offset_ms, external_tag_manifest,
                              prune_outlier_support=True, fixed_rotation_initialization=fixed_rotation_initialization)
        else:
            result = temporal.run(manifest, frozen, poses, config, exposure_offset_ms, external_tag_manifest,
                                  prune_support=True, prune_outlier_support=True)
    finally:
        tracks.refine_matches, temporal.track_window = old_refiner, old_builder
    result.update(driver_before_uniqueness_sha256=result['source_sha256'], source_sha256=temporal.sha(__file__),
                  unique_matcher_sha256=temporal.sha(Path(__file__).with_name('stereo_unique_matching.py')),
                  matching_mode='experimental_two_direction_local_zncc_uniqueness',
                  uniqueness_settings={'template_size':15, 'search_size':41, 'excluded_peak_radius_px':2.,
                                       'minimum_zncc':.8, 'minimum_peak_margin':.05})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'frozen', 'poses', 'config', 'output'):
        parser.add_argument(name)
    parser.add_argument('--exposure-offset-ms', type=float, required=True)
    parser.add_argument('--external-tag-manifest')
    parser.add_argument('--long-trajectory', action='store_true')
    parser.add_argument('--fixed-rotation-initialization', action='store_true')
    args = parser.parse_args()
    result = run(args.manifest, args.frozen, args.poses, args.config, args.exposure_offset_ms,
                 args.external_tag_manifest, args.long_trajectory, args.fixed_rotation_initialization)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
