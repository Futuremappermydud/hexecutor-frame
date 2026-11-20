#!/usr/bin/env python3
"""Explicit offline ROI-normalized temporal replay; baseline stays unchanged."""
import argparse
import json
from pathlib import Path
import evaluate_stereo_temporal_calibration as temporal
from stereo_photometric_matching import refine_matches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest'); ap.add_argument('v2_result'); ap.add_argument('output')
    ap.add_argument('--start', action='append', type=int, required=True)
    ap.add_argument('--window-length', type=int, default=12)
    ap.add_argument('--external-tag-manifest')
    args = ap.parse_args()
    original = temporal.refine_matches
    temporal.refine_matches = refine_matches
    try:
        result = temporal.evaluate(args.manifest, args.v2_result, args.start,
                                   args.window_length, args.external_tag_manifest)
    finally:
        temporal.refine_matches = original
    result.update(matching_mode='experimental_roi_normalized_lk',
                  driver_sha256=temporal.sha(__file__),
                  refinement_source_sha256=temporal.sha(Path(__file__).with_name('stereo_photometric_matching.py')))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
