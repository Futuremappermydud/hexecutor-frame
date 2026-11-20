#!/usr/bin/env python3
"""Independent projection tests of offline update-consistency guard."""
import copy
import unittest

import numpy as np

from stereo_selfcal_validation import assess_update_stability

K_L = np.array([[450., 0., 302.], [0., 460., 307.], [0., 0., 1.]])
K_R = np.array([[600., 0., 320.], [0., 590., 240.], [0., 0., 1.]])
BASELINE = 0.105


def fit(rotation=(0., 0., 0.), scale=1., direction=(1., 0., 0.), accepted=True):
    return {'accepted': accepted, 'parameters': {'rotation_vector_rad': list(rotation),
            'focal_scale': scale, 'baseline_direction_left': list(direction),
            'rectification_rotation_vector_rad': [0., 0., 0.]}}


def projected_x(f):
    """Independent physical right-camera pinhole projection at frozen probes."""
    p = f['parameters']
    v = np.array(p['rotation_vector_rad'])
    angle = np.linalg.norm(v)
    if angle:
        x, y, z = v / angle
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rot = np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * cross @ cross
    else:
        rot = np.eye(3)
    world = np.array([[x*z, y*z, z] for z in (0.7, 1.5, 2.5)
                       for y in (-0.5, 0., 0.5) for x in (-0.5, 0., 0.5)])
    camera = (world - BASELINE * np.array(p['baseline_direction_left'])) @ rot
    return K_R[0, 0] * p['focal_scale'] * camera[:, 0] / camera[:, 2] + K_R[0, 2]


def independent_spread(fits):
    x = np.array([projected_x(f) for f in fits])
    return np.max(np.ptp(x, axis=0))


class StabilityTests(unittest.TestCase):
    def assess(self, candidate, subsets, **kwargs):
        return assess_update_stability(candidate, subsets, K_L, K_R, BASELINE, **kwargs)

    def test_equal_fits_pass_without_metric_acceptance(self):
        one = fit()
        result = self.assess(one, [copy.deepcopy(one), copy.deepcopy(one)])
        self.assertTrue(result['stability_passed'], result)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')
        self.assertAlmostEqual(result['max_horizontal_spread_px'], 0., delta=1e-12)
        self.assertEqual(result['grid']['point_count'], 27)

    def test_known_yaw_small_and_large(self):
        base, small, large = fit(), fit(rotation=(0., 0.0001, 0.)), fit(rotation=(0., 0.002, 0.))
        for other, passed in ((small, True), (large, False)):
            with self.subTest(passed=passed):
                fits = [base, other, base]
                result = self.assess(fits[0], fits[1:])
                self.assertEqual(result['stability_passed'], passed, result)
                self.assertAlmostEqual(result['max_horizontal_spread_px'], independent_spread(fits), delta=1e-9)

    def test_focal_and_baseline_direction_differences(self):
        direction = np.array([1., 0., 0.1]); direction /= np.linalg.norm(direction)
        for changed in (fit(scale=1.004), fit(direction=direction)):
            with self.subTest(changed=changed):
                fits = [fit(), changed, fit()]
                result = self.assess(fits[0], fits[1:])
                self.assertFalse(result['stability_passed'], result)
                self.assertAlmostEqual(result['max_horizontal_spread_px'], independent_spread(fits), delta=1e-9)

    def test_rejected_or_missing_subsets_and_nonfinite_fit(self):
        good = fit()
        for candidate, subsets in ((good, []), (good, [good]), (fit(accepted=False), [good, good]),
                                    (good, [good, fit(accepted=False)]),
                                    (good, [good, fit(rotation=(float('nan'), 0., 0.))]),
                                    (None, [good, good]),
                                    ({'accepted': True, 'parameters': None}, [good, good]),
                                    ({'accepted': True, 'parameters': []}, [good, good]),
                                    (fit(rotation=(0., np.pi, 0.)), [good, good])):
            with self.subTest(candidate=candidate, subsets=subsets):
                result = self.assess(candidate, subsets)
                self.assertFalse(result['stability_passed'], result)
                self.assertTrue(result['reasons'])
                self.assertEqual(result['metric_acceptance'], 'unevaluated')

    def test_rejects_invalid_calibration_baseline_and_threshold(self):
        good = fit()
        for kl, kr, baseline, threshold in ((np.eye(2), K_R, BASELINE, .3),
                (K_L, np.zeros((3, 3)), BASELINE, .3),
                (K_L, K_R, 0., .3), (K_L, K_R, float('nan'), .3),
                (K_L, K_R, BASELINE, 0.), (K_L, K_R, BASELINE, float('nan'))):
            with self.subTest(baseline=baseline, threshold=threshold):
                with self.assertRaises(ValueError):
                    assess_update_stability(good, [good, good], kl, kr, baseline,
                                            max_horizontal_spread_px=threshold)

    def test_rectification_gauge_does_not_affect_original_right_projection(self):
        base, shifted = fit(), fit(rotation=(0., 0.0001, 0.))
        reference = self.assess(base, [shifted, base])
        altered = copy.deepcopy(shifted)
        altered['parameters']['rectification_rotation_vector_rad'] = [0.2, -0.1, 0.3]
        actual = self.assess(base, [altered, base])
        self.assertEqual(actual['stability_passed'], reference['stability_passed'])
        self.assertAlmostEqual(actual['max_horizontal_spread_px'], reference['max_horizontal_spread_px'], delta=1e-12)


if __name__ == '__main__':
    unittest.main()
