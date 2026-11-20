#!/usr/bin/env python3
"""Known image translations with fixed synthetic cross-camera appearance changes.

This is a controlled stress test of the frozen ORB/LK matcher, not a simulation
of measured Frame sensor characteristics. No stereo calibration is fitted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CASES = ('identity', 'gain_offset', 'anisotropic_blur', 'illumination_gradient',
         'noise', 'combined', 'occlusion')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(image, case):
    """Change appearance at fixed pixel coordinates with predeclared settings."""
    out = image.astype(np.float32)
    if case in ('anisotropic_blur', 'combined'):
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2, sigmaY=.4)
    if case in ('gain_offset', 'combined'):
        out = .65 * out + 20.
    if case == 'illumination_gradient':
        out = out * np.linspace(.6, 1.4, image.shape[1])[None, :] + 15.
    if case in ('noise', 'combined'):
        out += np.random.default_rng(731).normal(0., 4., image.shape)
    if case == 'occlusion':
        h, w = image.shape
        out[h//2-78:h//2+78, w//2-38:w//2+38] = 100.
    if case not in CASES:
        raise ValueError('unknown photometric case')
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def run(manifest, frozen, ordinals, matcher='v2'):
    # Keep the checked v3 translation-control extraction/matching path. Only
    # replace its synthetic right-image return, with a guarded single anchor.
    control = Path(__file__).with_name('audit_stereo_selfcal_matching.py')
    source = control.read_text()
    anchor = '            return out, out_mask'
    if source.count(anchor) != 1:
        raise ValueError('translation control source anchor changed')
    source = source.replace(anchor, '            return photo_control(out), out_mask')
    replacement = None
    matcher_hash = None
    if matcher == 'normalized':
        from stereo_photometric_matching import refine_matches
        replacement = refine_matches
        matcher_hash = sha(Path(__file__).with_name('stereo_photometric_matching.py'))
        anchor = "        observed = namespace['run'](str(manifest_path), matching='lk')"
        if source.count(anchor) != 1:
            raise ValueError('matcher replacement source anchor changed')
        source = source.replace(anchor, "        namespace['refine_matches'] = photo_matcher\n" + anchor)
    elif matcher != 'v2':
        raise ValueError('unknown matcher')
    results = {}
    for case in CASES:
        namespace = {'__name__': 'photometric_control', '__file__': str(control),
                     'photo_control': lambda image, name=case: transform(image, name),
                     'photo_matcher': replacement}
        exec(compile(source, str(control), 'exec'), namespace)
        results[case] = namespace['run'](manifest, frozen, ordinals)
        print(f'completed {case}', flush=True)
    return {'source_sha256': sha(__file__), 'translation_control_sha256': sha(control),
            'matcher': matcher, 'replacement_matcher_sha256': matcher_hash,
            'opencv_version': cv2.__version__, 'numpy_version': np.__version__,
            'settings': {'shifts_px': [[-12.25, .625], [-24.5, -.375]],
                         'gain': .65, 'offset': 20., 'blur_sigma_xy': [1.2, .4],
                         'gradient_endpoints': [.6, 1.4], 'gradient_offset': 15.,
                         'noise_sigma': 4., 'noise_seed': 731,
                         'occlusion_width_height': [76, 156], 'occlusion_intensity': 100},
            'cases': results,
            'caveats': ['Synthetic appearance changes are not measured sensor parameters.',
                        'Translations lack physical stereo perspective, depth and distortion.',
                        'Occlusion is deliberately not supplied to matcher masks.',
                        'Noise is deterministic and shared across cases; counts are correlated.',
                        'Selection gates affect which correspondences survive; no metric acceptance.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('frozen_result'); parser.add_argument('output')
    parser.add_argument('--ordinal', action='append', type=int, required=True)
    parser.add_argument('--matcher', choices=('v2', 'normalized'), default='v2')
    args = parser.parse_args()
    result = run(args.manifest, args.frozen_result, args.ordinal, args.matcher)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
