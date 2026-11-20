#!/usr/bin/env python3
"""Known-translation control for the frozen v2 ORB/LK matcher.

Instrument its verified source only in memory. The right image is an affine
translation of the rectified left image, not an independent stereo capture.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SHIFTS = ((-12.25, .625), (-24.5, -.375))
OUTLIER_PX = 1.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(points, shift):
    if not len(points):
        return {'count': 0}
    error = points[:, 2:4] - points[:, :2] - np.asarray(shift)
    distance = np.linalg.norm(error, axis=1)
    def axis(v):
        return {'mean_px': float(np.mean(v)), 'median_px': float(np.median(v)),
                'p90_abs_px': float(np.percentile(np.abs(v), 90))}
    return {'count': len(points), 'x_error': axis(error[:, 0]), 'y_error': axis(error[:, 1]),
            'euclidean_p90_px': float(np.percentile(distance, 90)),
            'outlier_threshold_px': OUTLIER_PX, 'outlier_count': int(np.sum(distance > OUTLIER_PX))}


def spatial(points, shift):
    result = {}
    for y in range(2):
        for x in range(2):
            keep = ((points[:, 0] >= x*308) & (points[:, 0] < (x+1)*308) &
                    (points[:, 1] >= y*308) & (points[:, 1] < (y+1)*308))
            result[f'{x},{y}'] = stats(points[keep], shift)
    return result


def run(manifest_path, frozen_path, ordinals):
    here = Path(__file__).resolve().parent
    eval_path = here / 'evaluate_stereo_self_calibration.py'
    source = eval_path.read_text()
    frozen = json.loads(Path(frozen_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if sha(eval_path) != frozen['source_sha256'] or sha(here/'stereo_self_calibration.py') != frozen['estimator_sha256'] or sha(manifest_path) != frozen['manifest_sha256']:
        raise ValueError('frozen evaluator, estimator or manifest hash mismatch')
    selected = [f for f in manifest['frames'] if f['ordinal'] in ordinals]
    if len(selected) != len(ordinals) or any(f['split'] != 'train' for f in selected):
        raise ValueError('ordinals must select distinct training frames')
    for f in selected:
        for side in ('left', 'right'):
            path = str(Path(f[side]).resolve())
            if sha(path) != frozen['input_sha256'][path]:
                raise ValueError('raw input changed: ' + path)
    patches = {
        "    for frame in manifest['frames']:\n": "    for frame in manifest['frames']:\n        if frame['ordinal'] not in audit_ordinals:\n            continue\n",
        '        k0, d0 = orb.detectAndCompute(images[0], masks[0])': '        images[1], masks[1] = audit_shift(images[0], masks[0])\n        k0, d0 = orb.detectAndCompute(images[0], masks[0])',
        '        parts[split].append(pts)': '        audit_callback(ordinal, seeds.copy(), seeds[selected].copy(), pts.copy())\n        parts[split].append(pts)',
        "    for model in ('rotation3', 'relative4', 'full6'):": '    for model in ():'}
    for old, new in patches.items():
        if source.count(old) != 1:
            raise ValueError('source instrumentation anchor changed')
        source = source.replace(old, new)
    cases = []
    for shift in SHIFTS:
        records = []
        def translated(image, mask):
            matrix = np.array([[1., 0., shift[0]], [0., 1., shift[1]]], dtype=np.float32)
            height, width = image.shape
            out = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Conservative interpolation support: require neighboring source mask pixels valid.
            safe = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
            out_mask = cv2.warpAffine(safe, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return out, out_mask
        def capture(ordinal, seeds, kept_seeds, refined):
            records.append({'ordinal': ordinal, 'all_orb_seeds': stats(seeds, shift),
                            'retained_orb_seeds': stats(kept_seeds, shift),
                            'refined': stats(refined, shift),
                            'refined_spatial_quadrants': spatial(refined, shift)})
        namespace = {'__name__': 'matching_control', '__file__': str(eval_path),
                     'audit_ordinals': set(ordinals), 'audit_shift': translated,
                     'audit_callback': capture}
        exec(compile(source, str(eval_path), 'exec'), namespace)
        observed = namespace['run'](str(manifest_path), matching='lk')
        for key in ('source_sha256', 'estimator_sha256', 'rectifier_sha256', 'calibration_sha256',
                    'observations_sha256', 'pairs_sha256'):
            if observed[key] != frozen[key]:
                raise ValueError('source provenance mismatch: ' + key)
        expected_paths = {str(Path(f[side]).resolve()) for f in selected for side in ('left', 'right')}
        if set(observed['input_sha256']) != expected_paths or any(observed['input_sha256'][p] != frozen['input_sha256'][p] for p in expected_paths):
            raise ValueError('observed selected frame hashes do not match v2')
        cases.append({'right_translation_px': list(shift), 'frames': records,
                      'observed_input_sha256': observed['input_sha256']})
    return {'audit_source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path),
            'v2_result_sha256': sha(frozen_path),
            'source_sha256': frozen['source_sha256'], 'estimator_sha256': frozen['estimator_sha256'],
            'rectifier_sha256': observed['rectifier_sha256'], 'calibration_sha256': observed['calibration_sha256'],
            'observations_sha256': observed['observations_sha256'], 'pairs_sha256': observed['pairs_sha256'],
            'selection_ordinals': ordinals, 'cases': cases,
            'method': 'Synthetic right = bilinear affine translation of rectified saved left. Left mask is the unchanged v2 mask; right mask is translated from left after an additional 3x3 erosion for interpolation support. Frozen ORB mutual matching, LK forward/back and ZNCC follow unchanged.',
            'caveats': ['Translation is not physical stereo, lighting variation, occlusion, depth variation or an independent image.',
                        'Only returned correspondences are scored; selection and discarded seeds can bias observed errors.',
                        'Tag masks can miss undetected tags; no tag geometry or model fit is used.',
                        'OpenCV interpolation quantizes subpixel warp positions; these are not noise-free continuous image shifts.']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest'); ap.add_argument('v2_result'); ap.add_argument('output')
    ap.add_argument('--ordinal', action='append', type=int, required=True)
    args = ap.parse_args()
    result = run(args.manifest, args.v2_result, args.ordinal)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
