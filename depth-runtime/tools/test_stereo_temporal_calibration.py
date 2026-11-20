#!/usr/bin/env python3
"""Independent moving-camera 3-D projection tests for offline temporal BA."""
import unittest

import numpy as np

from stereo_temporal_calibration import fit_temporal_calibration

K = np.array([[600., 0., 640.], [0., 610., 480.], [0., 0., 1.]])
BASELINE = 0.065
ROTATION = np.array([0.004, -0.008, 0.003])
SCALE = 1.008


def rot(v):
    angle = np.linalg.norm(v)
    if not angle:
        return np.eye(3)
    x, y, z = np.asarray(v) / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * cross @ cross


def pixels(p, scale=1.):
    return np.column_stack((K[0, 0] * scale * p[:, 0] / p[:, 2] + K[0, 2],
                            K[1, 1] * scale * p[:, 1] / p[:, 2] + K[1, 2]))


def synthetic(seed=501, n=60, frames=5, scene_scale=1., stationary=False, depth=None, spatial_scale=1.):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.8, 2.8, n) if depth is None else np.full(n, depth)
    points = np.column_stack((rng.uniform(-0.5, 0.5, n) * z * spatial_scale,
                              rng.uniform(-0.35, 0.35, n) * z * spatial_scale, z)) * scene_scale
    observations = []
    for frame in range(frames):
        # The first left camera defines the world origin and orientation.
        i = 0 if stationary else frame
        camera_to_world = rot(np.array([0.006, -0.011, 0.004]) * i)
        center = np.array([0.045, 0.012, 0.008]) * i * scene_scale
        left = (points - center) @ camera_to_world
        right = (left - [BASELINE * scene_scale, 0., 0.]) @ rot(ROTATION)
        lp, rp = pixels(left), pixels(right, SCALE)
        for track in range(n):
            observations.append([frame, track, *lp[track], *rp[track]])
    return np.array(observations), points


def independent_right(points_left, result, baseline):
    p = result['parameters']
    camera = (points_left - [baseline, 0., 0.]) @ rot(p['rotation_vector_rad'])
    return pixels(camera, p['focal_scale'])


class TemporalCalibrationTests(unittest.TestCase):
    def test_recorded_translation_pattern_fits_scale_and_preserves_lever_arm(self):
        data, points = synthetic()
        rotations = np.array([rot(np.array([.006, -.011, .004])*i) for i in range(5)])
        centers = np.array([[.045, .012, .008]]) * np.arange(5)[:, None]
        lever = np.array([r @ [.05, -.01, .02] for r in rotations]); lever -= lever[0]
        # Tracker units deliberately differ by 2.5; lever offsets remain metric.
        motion = {'translation': (centers-lever)/2.5, 'lever_arm': lever}
        result = fit_temporal_calibration(data, K, K, BASELINE,
            camera_rotations=rotations, camera_motion=motion)
        self.assertTrue(result['accepted'], result)
        self.assertTrue(result['diagnostics']['motion_trajectory_fixed_up_to_scale'])
        self.assertAlmostEqual(result['diagnostics']['motion_translation_scale'], 2.5, delta=.01)
        np.testing.assert_allclose([p['translation_m'] for p in result['poses']], centers, atol=.001)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION, atol=.001, rtol=0)
        self.assertAlmostEqual(result['parameters']['focal_scale'], SCALE, delta=.002)
        estimated = {p['track_id']: p['position_m'] for p in result['points']}
        np.testing.assert_allclose([estimated[i] for i in range(len(points))], points, atol=.03)

    def test_invalid_recorded_motion(self):
        data, _ = synthetic()
        rotations = np.tile(np.eye(3), (5, 1, 1))
        for motion in ({}, {'translation': np.zeros((4, 3)), 'lever_arm': np.zeros((5, 3))},
                       {'translation': np.ones((5, 3)), 'lever_arm': np.zeros((5, 3))}):
            with self.assertRaises(ValueError):
                fit_temporal_calibration(data, K, K, BASELINE, camera_rotations=rotations, camera_motion=motion)
        with self.assertRaises(ValueError):
            fit_temporal_calibration(data, K, K, BASELINE, camera_motion={'translation': np.zeros((5, 3)), 'lever_arm': np.zeros((5, 3))})

    def test_independent_camera_rotations_with_free_translation_scale(self):
        data, points = synthetic(scene_scale=2.)
        rotations = np.array([rot(np.array([.006, -.011, .004])*i) for i in range(5)])
        result = fit_temporal_calibration(data, K, K, 2*BASELINE, camera_rotations=rotations)
        self.assertTrue(result['accepted'], result)
        self.assertTrue(result['diagnostics']['camera_rotations_fixed'])
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION, atol=.001, rtol=0)
        self.assertAlmostEqual(result['parameters']['focal_scale'], SCALE, delta=.002)
        for pose, expected in zip(result['poses'], rotations):
            np.testing.assert_allclose(rot(pose['rotation_vector_rad']), expected, atol=1e-10)
        estimated = {p['track_id']: p['position_m'] for p in result['points']}
        np.testing.assert_allclose(np.array([estimated[i] for i in range(len(points))]), points, atol=.06, rtol=0)

    def test_invalid_independent_rotations(self):
        data, _ = synthetic()
        valid = np.tile(np.eye(3), (5, 1, 1))
        reflected = valid.copy(); reflected[1, 0, 0] = -1
        shifted = valid.copy(); shifted[0] = rot([0., .1, 0.])
        nonfinite = valid.copy(); nonfinite[1, 0, 0] = np.nan
        for bad in (valid[:4], reflected, shifted, nonfinite):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    fit_temporal_calibration(data, K, K, BASELINE, camera_rotations=bad)

    def test_fixed_focal_recovery_and_three_parameter_profile(self):
        data, points = synthetic()
        data[:, 4:6] = (data[:, 4:6] - K[:2, 2]) / SCALE + K[:2, 2]
        result = fit_temporal_calibration(data, K, K, BASELINE, model='rotation3')
        self.assertTrue(result['accepted'], result)
        self.assertEqual(result['model'], 'temporal_rotation3')
        self.assertEqual(result['parameters']['focal_scale'], 1.)
        self.assertFalse(result['diagnostics']['focal_estimated'])
        self.assertEqual(len(result['diagnostics']['calibration_profile_singular_values']), 3)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION, atol=.001, rtol=0)
        expected = pixels((points - [BASELINE, 0., 0.]) @ rot(ROTATION))
        self.assertLess(np.max(np.abs(independent_right(points, result, BASELINE) - expected)), .2)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')

    def test_fixed_focal_rejection_and_invalid_model(self):
        data, _ = synthetic(stationary=True)
        data[:, 4:6] = (data[:, 4:6] - K[:2, 2]) / SCALE + K[:2, 2]
        result = fit_temporal_calibration(data, K, K, BASELINE, model='rotation3')
        self.assertFalse(result['accepted'], result)
        self.assertIn('insufficient_translation', result['diagnostics']['reasons'])
        early = fit_temporal_calibration(data[:3], K, K, BASELINE, model='rotation3')
        self.assertEqual(early['model'], 'temporal_rotation3')
        self.assertEqual(early['diagnostics']['calibration_parameter_count'], 3)
        with self.assertRaises(ValueError):
            fit_temporal_calibration(data, K, K, BASELINE, model='invalid')

    def test_clean_recovery_and_original_right_projection(self):
        data, points = synthetic()
        result = fit_temporal_calibration(data, K, K, BASELINE)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION,
                                   rtol=0, atol=0.001)
        self.assertAlmostEqual(result['parameters']['focal_scale'], SCALE, delta=0.002)
        expected = pixels((points - [BASELINE, 0., 0.]) @ rot(ROTATION), SCALE)
        prediction = independent_right(points, result, BASELINE)
        self.assertLess(np.median(np.abs(prediction - expected)), 0.2)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')
        pose0 = next(p for p in result['poses'] if p['frame_id'] == 0)
        np.testing.assert_allclose(pose0['rotation_vector_rad'], 0., atol=1e-12)
        np.testing.assert_allclose(pose0['translation_m'], 0., atol=1e-12)
        estimated_points = {p['track_id']: p['position_m'] for p in result['points']}
        self.assertEqual(len(estimated_points), len(points))
        np.testing.assert_allclose(np.array([estimated_points[i] for i in range(len(points))]),
                                   points, atol=0.03, rtol=0)

    def test_nonzero_skew_intrinsics(self):
        data, _ = synthetic()
        skewed_k = K.copy(); skewed_k[0, 1] = 30.
        # Standard K projects u = fx*x/z + skew*y/z + cx; focal scale
        # multiplies both right focal axes and skew, leaving cy fixed.
        data[:, 2] += 30. * (data[:, 3] - K[1, 2]) / K[1, 1]
        data[:, 4] += 30. * (data[:, 5] - K[1, 2]) / K[1, 1]
        result = fit_temporal_calibration(data, skewed_k, skewed_k, BASELINE)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION, atol=.001, rtol=0)
        self.assertAlmostEqual(result['parameters']['focal_scale'], SCALE, delta=.002)

    def test_noise_and_limited_outliers(self):
        data, points = synthetic(seed=520)
        rng = np.random.default_rng(521)
        data[:, 2:] += rng.normal(0., 0.08, data[:, 2:].shape)
        bad = rng.choice(len(data), size=int(len(data) * .05), replace=False)
        data[bad, 4:6] += rng.normal(0., 8., (len(bad), 2))
        result = fit_temporal_calibration(data, K, K, BASELINE)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION,
                                   rtol=0, atol=0.003)
        self.assertAlmostEqual(result['parameters']['focal_scale'], SCALE, delta=0.005)
        expected = pixels((points - [BASELINE, 0., 0.]) @ rot(ROTATION), SCALE)
        self.assertLess(np.median(np.abs(independent_right(points, result, BASELINE) - expected)), 0.5)

    def test_known_baseline_sets_scale_not_metric_acceptance(self):
        first, _ = synthetic()
        doubled, doubled_points = synthetic(scene_scale=2.)
        np.testing.assert_allclose(first, doubled, atol=1e-10, rtol=0)
        result = fit_temporal_calibration(doubled, K, K, 2. * BASELINE)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION,
                                   rtol=0, atol=0.001)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')
        estimated = {p['track_id']: p['position_m'] for p in result['points']}
        np.testing.assert_allclose(np.array([estimated[i] for i in range(len(doubled_points))]),
                                   doubled_points, atol=0.06, rtol=0)

    def test_stationary_disconnected_and_insufficient_support_reject(self):
        stationary, _ = synthetic(stationary=True)
        noisy_stationary = stationary.copy()
        noisy_stationary[:, 2:] += np.random.default_rng(17).normal(0., 0.08, noisy_stationary[:, 2:].shape)
        disconnected, _ = synthetic()
        # Make each frame its own disjoint track set despite plausible pixels.
        disconnected[:, 1] += disconnected[:, 0] * 10000
        few, _ = synthetic(n=6)
        for data in (stationary, noisy_stationary, disconnected, few):
            with self.subTest(count=len(data)):
                result = fit_temporal_calibration(data, K, K, BASELINE)
                self.assertFalse(result['accepted'], result)
                self.assertTrue(result['diagnostics']['reasons'])

    def test_noise_and_spatial_support_diagnostics(self):
        full, _ = synthetic(seed=731)
        central, _ = synthetic(seed=731, spatial_scale=.25)
        rng = np.random.default_rng(732)
        unit_noise = rng.normal(0., 1., full[:, 2:].shape)
        low, high = full.copy(), full.copy()
        low[:, 2:] += .08 * unit_noise
        high[:, 2:] += .3 * unit_noise
        narrow = central.copy(); narrow[:, 2:] += .08 * unit_noise
        results = [fit_temporal_calibration(a, K, K, BASELINE) for a in (low, high, narrow)]
        for result in results:
            diag = result['diagnostics']
            for key in ('scalar_residual_mad_sigma_px', 'nominal_independent_noise_max_probe_std_px'):
                self.assertTrue(np.isfinite(diag[key]), result)
                self.assertGreaterEqual(diag[key], 0.)
            self.assertIn('not covariance', diag['nominal_noise_diagnostic_assumptions'])
        lowd, highd, narrowd = [r['diagnostics'] for r in results]
        self.assertGreater(highd['scalar_residual_mad_sigma_px'], lowd['scalar_residual_mad_sigma_px'])
        self.assertGreater(highd['nominal_independent_noise_max_probe_std_px'], lowd['nominal_independent_noise_max_probe_std_px'])
        self.assertGreater(narrowd['nominal_independent_noise_max_probe_std_px'], lowd['nominal_independent_noise_max_probe_std_px'])
        self.assertLess(max(narrowd['spatial_support']['normalized_xy_weighted_5_95_span']),
                        min(lowd['spatial_support']['normalized_xy_weighted_5_95_span']))

    def test_profile_sensitivity_reports_pixel_scale(self):
        data, _ = synthetic()
        reduced = data.copy(); reduced[:, 2:] *= 0.1
        scaled_k = K.copy(); scaled_k[:2, :] *= 0.1
        native = fit_temporal_calibration(data, K, K, BASELINE)
        small = fit_temporal_calibration(reduced, scaled_k, scaled_k, BASELINE)
        self.assertTrue(native['accepted'], native)
        self.assertTrue(small['accepted'], small)
        a, b = native['diagnostics'], small['diagnostics']
        ratio_key = 'calibration_profile_singular_ratio'
        weak_key = 'calibration_profile_weak_direction_rms_px_per_unit'
        np.testing.assert_allclose(b[ratio_key], a[ratio_key], rtol=.002, atol=0.)
        np.testing.assert_allclose(b[weak_key], a[weak_key] * .1, rtol=.002, atol=0.)

    def test_stereo_consensus_requires_both_views(self):
        data, _ = synthetic(seed=620)
        rng = np.random.default_rng(621)
        # Nearly every left view is accurate, while right vertical measurements
        # are independently corrupted. Prevent hard pruning from concealing the
        # distinction between per-view and complete-stereo consensus.
        data[:, 5] += rng.choice([-2.5, 2.5], size=len(data))
        result = fit_temporal_calibration(data, K, K, BASELINE,
                                          config={'refit_max_view_error_px': 100.})
        self.assertFalse(result['accepted'], result)
        self.assertIn('insufficient_inlier_fraction', result['diagnostics']['reasons'])
        self.assertGreater(result['diagnostics']['per_view_inlier_fraction'],
                           result['diagnostics']['inlier_fraction'])

    def test_pruned_support_must_still_meet_minimum(self):
        data, _ = synthetic(seed=520)
        rng = np.random.default_rng(521)
        data[:, 2:] += rng.normal(0., 0.08, data[:, 2:].shape)
        bad = rng.choice(len(data), size=15, replace=False)
        data[bad, 4:6] += rng.normal(0., 8., (len(bad), 2))
        result = fit_temporal_calibration(data, K, K, BASELINE, config={'min_observations': 300})
        self.assertFalse(result['accepted'], result)
        self.assertIn('insufficient_support_after_outlier_rejection', result['diagnostics']['reasons'])

    def test_invalid_configuration_rejected(self):
        data, _ = synthetic()
        for config in ({'min_frames': 3.5}, {'min_tracks': True},
                       {'min_frames': 13}, {'min_observations': float('nan')}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    fit_temporal_calibration(data, K, K, BASELINE, config=config)

    def test_invalid_observations_rejected(self):
        data, _ = synthetic()
        broken = data.copy(); broken[0, 2] = np.nan
        duplicate = np.vstack((data, data[0]))
        fractional_id = data.copy(); fractional_id[0, 1] = 0.5
        for bad in (data[:, :5], broken, duplicate, fractional_id):
            with self.assertRaises(ValueError):
                fit_temporal_calibration(bad, K, K, BASELINE)
        for baseline in (0., float('nan')):
            with self.assertRaises(ValueError):
                fit_temporal_calibration(data, K, K, baseline)


if __name__ == '__main__':
    unittest.main()
