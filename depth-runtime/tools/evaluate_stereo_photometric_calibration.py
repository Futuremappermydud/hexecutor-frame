#!/usr/bin/env python3
"""Compare a separately tested appearance-invariant matcher on frozen captures.

Only the v2 evaluator's refinement function is replaced in memory. Its image
selection, masks, split, seed matching and calibration models remain unchanged.
No capture or live calibration changes are performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_stereo_self_calibration as baseline
from stereo_photometric_matching import refine_matches


def run(manifest_path, frozen_path):
    frozen = json.loads(Path(frozen_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if baseline.sha(baseline.__file__) != frozen['source_sha256']:
        raise ValueError('baseline evaluator source mismatch')
    if baseline.sha(manifest_path) != frozen['manifest_sha256']:
        raise ValueError('frozen manifest mismatch')
    matcher_path = Path(__file__).with_name('stereo_photometric_matching.py')
    matcher_hash = baseline.sha(matcher_path)
    old_refinement = baseline.refine_matches
    comparisons = []

    def compare(images, masks, seeds, return_indices=False):
        old, old_ids = old_refinement(images, masks, seeds, return_indices=True)
        new, new_ids = refine_matches(images, masks, seeds, return_indices=True)
        common, old_order, new_order = np.intersect1d(old_ids, new_ids, return_indices=True)
        delta = new[new_order, 2:] - old[old_order, 2:]
        ordinal = manifest['frames'][len(comparisons)]['ordinal']
        comparisons.append({'ordinal': ordinal, 'old_count': len(old), 'new_count': len(new),
                            'common_seed_count': len(common),
                            'right_x_change_on_common': baseline.summary(delta[:, 0]),
                            'right_y_change_on_common': baseline.summary(delta[:, 1])})
        return (new, new_ids) if return_indices else new

    baseline.refine_matches = compare
    try:
        result = baseline.run(manifest_path, matching='lk')
    finally:
        baseline.refine_matches = old_refinement
    for key in ('source_sha256', 'estimator_sha256', 'manifest_sha256', 'rectifier_sha256',
                'calibration_sha256', 'observations_sha256', 'pairs_sha256', 'input_sha256'):
        if result[key] != frozen[key]:
            raise ValueError('frozen provenance changed: ' + key)
    if baseline.sha(matcher_path) != matcher_hash:
        raise ValueError('matcher source changed during evaluation')
    result.update(driver_sha256=baseline.sha(__file__), refinement_source_sha256=matcher_hash,
                  frozen_comparison_sha256=baseline.sha(frozen_path),
                  matching_mode='experimental_normalized_patch_refinement',
                  refinement='Separate appearance-invariant two-dimensional refinement; source identified by refinement_source_sha256. Original v2 LK is retained only as a same-seed comparison.',
                  refinement_comparison_by_frame=comparisons)
    result['caveats'].append('Experimental matcher changes retained support; independent tag support and split are unchanged. Passing synthetic controls does not validate real stereo photometry.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('frozen_result'); parser.add_argument('output')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.manifest, args.frozen_result), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
