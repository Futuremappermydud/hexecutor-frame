#!/usr/bin/env python3
"""Explicit offline direct-track replay; frozen inputs and scoring are preserved."""
import argparse
import json
from pathlib import Path
import evaluate_stereo_temporal_calibration as temporal
from stereo_photometric_matching import refine_matches
from stereo_direct_tracks import track_window, filter_left_parallax
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest'); ap.add_argument('v2_result'); ap.add_argument('output')
    ap.add_argument('--start', action='append', type=int, required=True)
    ap.add_argument('--window-length', type=int, default=12)
    ap.add_argument('--external-tag-manifest')
    ap.add_argument('--temporal-model', choices=('relative4', 'rotation3'), default='relative4')
    ap.add_argument('--head-poses')
    ap.add_argument('--device-config')
    ap.add_argument('--exposure-offset-ms', type=float)
    ap.add_argument('--use-head-translation', action='store_true',
                    help='fit one translation scale; retain calibrated lever offsets')
    ap.add_argument('--min-left-parallax-deg', type=float,
                    help='retain whole tracks with >=3 views and this left-ray parallax; requires head poses')
    args = ap.parse_args()
    rotations, rotation_provenance, motion = None, None, None
    if args.min_left_parallax_deg is not None and (
            args.head_poses is None or not np.isfinite(args.min_left_parallax_deg) or
            not 0 < args.min_left_parallax_deg < 180):
        ap.error('--min-left-parallax-deg requires head poses and a finite angle between 0 and 180')
    if args.use_head_translation and args.head_poses is None:
        ap.error('--use-head-translation requires recorded head poses')
    pose_options = (args.head_poses, args.device_config, args.exposure_offset_ms)
    if any(value is not None for value in pose_options):
        if any(value is None for value in pose_options):
            ap.error('--head-poses, --device-config and --exposure-offset-ms are required together')
        from stereo_pose_constraints import recorded_motion
        rotations, translations, levers, rotation_provenance = recorded_motion(
            args.manifest, args.head_poses, args.device_config, args.exposure_offset_ms)
        if args.use_head_translation:
            motion = {'translation': translations, 'lever_arm': levers}
    original = temporal.refine_matches
    def track_filter(rows, ids):
        return filter_left_parallax(rows, np.array([rotations[i] for i in ids]), temporal.K,
                                    minimum_degrees=args.min_left_parallax_deg)
    temporal.refine_matches = refine_matches
    try:
        result = temporal.evaluate(args.manifest, args.v2_result, args.start,
                                   args.window_length, args.external_tag_manifest,
                                   track_builder=track_window, temporal_model=args.temporal_model,
                                   camera_rotations_by_ordinal=rotations,
                                   camera_motion_by_ordinal=motion,
                                   track_filter=track_filter if args.min_left_parallax_deg is not None else None)
    finally:
        temporal.refine_matches = original
    result.update(matching_mode='experimental_direct_tracks_roi_lk',
                  method='Direct temporal pyramidal LK proposals, ROI-normalized bidirectional refinement, independent stereo closure <=0.5px; no ORB endpoint snapping. Frozen stereo seeds, separated births >=8px in both eyes. Tracks >=2 frames; id %5 held out; explicit selected temporal model and optional recorded-motion constraints; original scoring.',
                  tracking_source_sha256=temporal.sha(Path(__file__).with_name('stereo_direct_tracks.py')),
                  driver_sha256=temporal.sha(__file__),
                  refinement_source_sha256=temporal.sha(Path(__file__).with_name('stereo_photometric_matching.py')))
    result['rotation_constraints'] = rotation_provenance
    result['head_translation_used_up_to_scale'] = args.use_head_translation
    result['minimum_left_parallax_degrees'] = args.min_left_parallax_deg
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
