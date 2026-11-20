#!/usr/bin/env python3
"""Independent 3-D pinhole projection checks for the offline rectifier.

Tolerances are fixed from the synthetic camera and noise model, before running
against the estimator. These tests do not establish real-camera metric accuracy.
"""
import copy
import unittest

import numpy as np

from stereo_self_calibration import fit_calibration, rectify_points, project_right_points


K = np.array([[600., 0., 640.], [0., 610., 480.], [0., 0., 1.]])
ROTATION = np.array([0.006, -0.012, 0.004])  # right-camera rays to left axes, radians
FOCAL_SCALE = 1.012


def rotation_matrix(vector):
    """Construct the physical camera orientation using Rodrigues' formula."""
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def scene(seed=42, n=600, rotation=ROTATION, scale=FOCAL_SCALE, baseline=0.065):
    """Sample actual 3-D points, then independently project both cameras."""
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.7, 3.0, n)
    points = np.column_stack((rng.uniform(-0.65, 0.65, n) * depth,
                              rng.uniform(-0.42, 0.42, n) * depth, depth))
    return project(points, rotation, scale, baseline)


def project(points, rotation=ROTATION, scale=FOCAL_SCALE, baseline=0.065):
    left = points
    # Right camera is displaced along left x; its axes are physically rotated.
    shifted = points - np.array([baseline, 0., 0.])
    right = shifted @ rotation_matrix(rotation)
    ideal = shifted

    def pixels(p, s=1.):
        return np.column_stack((K[0, 0] * s * p[:, 0] / p[:, 2] + K[0, 2],
                                K[1, 1] * s * p[:, 1] / p[:, 2] + K[1, 2]))
    return np.column_stack((pixels(left), pixels(right, scale))), pixels(ideal)


def general_scene(seed=222, n=700, depths=None):
    """Project physical world points through two oriented and displaced cameras."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.7, 3.0, n) if depths is None else np.broadcast_to(depths, (n,)).copy()
    left_points = np.column_stack((rng.uniform(-0.6, 0.6, n) * z,
                                   rng.uniform(-0.4, 0.4, n) * z, z))
    # Define a global frame with a nonidentity left pose, then transform back
    # as a camera would: all reported stereo parameters remain in left axes.
    left_to_world = rotation_matrix([0.2, -0.1, 0.07])
    center_world = np.array([1.3, -0.2, 0.4])
    world = left_points @ left_to_world.T + center_world
    left = (world - center_world) @ left_to_world
    baseline = np.array([0.065, 0.003, -0.002])
    right = (left - baseline) @ rotation_matrix(ROTATION)
    def px(p, scale=1.):
        return np.column_stack((K[0, 0] * scale * p[:, 0] / p[:, 2] + K[0, 2],
                                K[1, 1] * scale * p[:, 1] / p[:, 2] + K[1, 2]))
    return left, baseline, np.column_stack((px(left), px(right, FOCAL_SCALE)))


def align_baseline_to_x(baseline):
    """Independent Rodrigues construction of minimal rotation to the x axis."""
    u = baseline / np.linalg.norm(baseline)
    target = np.array([1., 0., 0.])
    v = np.cross(u, target)
    cross = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + cross + cross @ cross / (1. + np.dot(u, target))


class StereoSelfCalibrationTests(unittest.TestCase):
    def test_full6_general_geometry_and_original_frame_projection(self):
        points, baseline, matches = general_scene()
        result = fit_calibration(matches, K, K, model='full6')
        self.assertTrue(result['accepted'], result)
        p = result['parameters']
        np.testing.assert_allclose(p['rotation_vector_rad'], ROTATION, atol=3e-5, rtol=0)
        self.assertAlmostEqual(p['focal_scale'], FOCAL_SCALE, delta=3e-5)
        np.testing.assert_allclose(p['baseline_slopes_yz'], baseline[1:] / baseline[0], atol=3e-5, rtol=0)
        np.testing.assert_allclose(p['baseline_direction_left'], baseline / np.linalg.norm(baseline), atol=3e-5, rtol=0)
        corrected = rectify_points(matches, K, K, result)
        q = align_baseline_to_x(baseline)
        left_rect = points @ q.T
        right_rect = (points - baseline) @ q.T
        def px(v):
            return np.column_stack((K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2],
                                    K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]))
        np.testing.assert_allclose(corrected, np.column_stack((px(left_rect), px(right_rect))), atol=0.02, rtol=0)
        self.assertLess(np.max(np.abs(corrected[:, 1] - corrected[:, 3])), 0.02)
        # Scoring must use original left-frame points and original right image,
        # not the rectified left points, which are in a different gauge.
        predicted = project_right_points(points, K, result, np.linalg.norm(baseline))
        np.testing.assert_allclose(predicted, matches[:, 2:], atol=0.02, rtol=0)
        alternate_gauge = copy.deepcopy(result)
        alternate_gauge['parameters']['rectification_rotation_vector_rad'] = [0.08, -0.03, 0.01]
        np.testing.assert_allclose(project_right_points(points, K, alternate_gauge, np.linalg.norm(baseline)),
                                   predicted, atol=1e-12, rtol=0)
        self.assertGreater(np.median(np.abs(corrected[:, 1] - matches[:, 1])), 1.)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')

    def test_full6_robust_noise_and_outliers(self):
        points, baseline, clean = general_scene(seed=411)
        rng = np.random.default_rng(412)
        matches = clean + rng.normal(0., 0.05, clean.shape)
        bad = rng.choice(len(matches), 70, replace=False)
        matches[bad, 2:] = matches[rng.permutation(bad), 2:]
        result = fit_calibration(matches, K, K, model='full6')
        self.assertTrue(result['accepted'], result)
        p = result['parameters']
        np.testing.assert_allclose(p['rotation_vector_rad'], ROTATION, atol=0.001, rtol=0)
        self.assertAlmostEqual(p['focal_scale'], FOCAL_SCALE, delta=0.002)
        np.testing.assert_allclose(p['baseline_slopes_yz'], baseline[1:] / baseline[0], atol=0.002, rtol=0)
        predicted = project_right_points(points, K, result, np.linalg.norm(baseline))
        self.assertLess(np.median(np.abs(predicted - clean[:, 2:])), 0.25)

    def test_full6_planar_and_narrow_depth_rejected(self):
        for depths in (1.5, np.linspace(1.499, 1.501, 700)):
            with self.subTest(depths=depths if np.isscalar(depths) else 'narrow'):
                _, _, matches = general_scene(depths=depths)
                result = fit_calibration(matches, K, K, model='full6')
                self.assertFalse(result['accepted'], result)
                self.assertTrue(result['diagnostics']['reasons'])
        # A tilted plane has substantial depth spread but affine inverse depth;
        # spread by itself is not independent 3-D support.
        rng = np.random.default_rng(991)
        x = rng.uniform(-0.6, 0.6, 700)
        y = rng.uniform(-0.4, 0.4, 700)
        z = 1.5 / (1. - 0.3 * x + 0.2 * y)
        points = np.column_stack((x * z, y * z, z))
        baseline = np.array([0.065, 0.003, -0.002])
        r = (points - baseline) @ rotation_matrix(ROTATION)
        left = np.column_stack((600. * x + 640., 610. * y + 480.))
        right = np.column_stack((600. * FOCAL_SCALE * r[:, 0] / r[:, 2] + 640.,
                                 610. * FOCAL_SCALE * r[:, 1] / r[:, 2] + 480.))
        result = fit_calibration(np.column_stack((left, right)), K, K, model='full6')
        self.assertFalse(result['accepted'], result)
        # Noise must not masquerade as physical depth diversity.
        _, _, planar = general_scene(depths=1.5)
        rng = np.random.default_rng(990)
        for sigma in (0.05, 0.2):
            noisy = planar + rng.normal(0., sigma, planar.shape)
            result = fit_calibration(noisy, K, K, model='full6')
            self.assertFalse(result['accepted'], result)

    def test_physical_rotation_focal_and_corrected_rays(self):
        # Exercise both signs, including a distinct principal-point-independent yaw.
        for rotation in (ROTATION, -ROTATION):
            with self.subTest(rotation=rotation):
                matches, ideal = scene(rotation=rotation)
                result = fit_calibration(matches, K, K)
                self.assertTrue(result['accepted'], result)
                np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], rotation,
                                           atol=2e-5, rtol=0)
                self.assertAlmostEqual(result['parameters']['focal_scale'], FOCAL_SCALE, delta=2e-5)
                corrected = rectify_points(matches, K, K, result)
                np.testing.assert_allclose(corrected[:, :2], matches[:, :2], atol=1e-9, rtol=0)
                np.testing.assert_allclose(corrected[:, 2:], ideal, atol=0.01, rtol=0)

    def test_unequal_intrinsics_use_normalized_rays(self):
        kr = np.array([[730., 0., 590.], [0., 570., 455.], [0., 0., 1.]])
        rng = np.random.default_rng(72)
        z = rng.uniform(0.7, 3., 600)
        world = np.column_stack((rng.uniform(-0.6, 0.6, 600) * z,
                                 rng.uniform(-0.4, 0.4, 600) * z, z))
        relative = world - [0.065, 0., 0.]
        camera = relative @ rotation_matrix(ROTATION)
        left = np.column_stack((K[0, 0] * world[:, 0] / z + K[0, 2],
                                K[1, 1] * world[:, 1] / z + K[1, 2]))
        right = np.column_stack((kr[0, 0] * FOCAL_SCALE * camera[:, 0] / camera[:, 2] + kr[0, 2],
                                 kr[1, 1] * FOCAL_SCALE * camera[:, 1] / camera[:, 2] + kr[1, 2]))
        matches = np.column_stack((left, right))
        result = fit_calibration(matches, K, kr)
        self.assertTrue(result['accepted'], result)
        corrected = rectify_points(matches, K, kr, result)
        expected_right = np.column_stack((kr[0, 0] * relative[:, 0] / relative[:, 2] + kr[0, 2],
                                          kr[1, 1] * relative[:, 1] / relative[:, 2] + kr[1, 2]))
        np.testing.assert_allclose(corrected[:, 2:], expected_right, atol=0.01, rtol=0)
        yleft = (corrected[:, 1] - K[1, 2]) / K[1, 1]
        yright = (corrected[:, 3] - kr[1, 2]) / kr[1, 1]
        self.assertLess(np.max(np.abs(yleft - yright)), 2e-5)

    def test_noise_and_mismatched_correspondences(self):
        clean, ideal = scene(seed=81)
        rng = np.random.default_rng(91)
        noisy = clean + rng.normal(0., 0.08, clean.shape)
        bad = rng.choice(len(noisy), size=90, replace=False)
        noisy[bad, 2:] = noisy[rng.permutation(bad), 2:]
        result = fit_calibration(noisy, K, K)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters']['rotation_vector_rad'], ROTATION,
                                   atol=0.001, rtol=0)
        self.assertAlmostEqual(result['parameters']['focal_scale'], FOCAL_SCALE, delta=0.002)
        corrected = rectify_points(clean, K, K, result)
        self.assertLess(np.median(np.abs(corrected[:, 3] - ideal[:, 1])), 0.15)
        self.assertLess(np.median(np.abs(corrected[:, 2] - ideal[:, 0])), 0.35)

    def test_insufficient_or_spatially_degenerate_support_rejected(self):
        matches, _ = scene()
        self.assertFalse(fit_calibration(matches[:20], K, K)['accepted'])
        # Hundreds of duplicates cannot manufacture independent spatial support.
        duplicates = np.repeat(matches[:1], 300, axis=0)
        self.assertFalse(fit_calibration(duplicates, K, K)['accepted'])
        # A handful of distributed observations repeated over frames still do
        # not constitute 100 distinct feature correspondences.
        repeated = np.tile(matches[:6], (100, 1))
        self.assertFalse(fit_calibration(repeated, K, K)['accepted'])
        rng = np.random.default_rng(4)
        z = rng.uniform(0.7, 3.0, 300)
        points = np.column_stack((0.001 * rng.normal(size=300),
                                  0.001 * rng.normal(size=300), z))
        narrow, _ = project(points)
        self.assertFalse(fit_calibration(narrow, K, K)['accepted'])

    def test_narrow_horizon_is_not_observable(self):
        rng = np.random.default_rng(123)
        z = rng.uniform(0.7, 3., 500)
        world = np.column_stack((rng.uniform(-0.6, 0.6, 500) * z,
                                 rng.normal(0., 0.0001, 500) * z, z))
        matches, _ = project(world)
        self.assertFalse(fit_calibration(matches, K, K)['accepted'])

    def test_out_of_model_baseline_does_not_certify_metric_depth(self):
        rng = np.random.default_rng(202)
        z = rng.uniform(0.7, 3., 600)
        world = np.column_stack((rng.uniform(-0.6, 0.6, 600) * z,
                                 rng.uniform(-0.4, 0.4, 600) * z, z))
        right = (world - [0.065, 0.004, 0.]) @ rotation_matrix(ROTATION)
        left_px = np.column_stack((600. * world[:, 0] / z + 640., 610. * world[:, 1] / z + 480.))
        right_px = np.column_stack((600. * FOCAL_SCALE * right[:, 0] / right[:, 2] + 640.,
                                    610. * FOCAL_SCALE * right[:, 1] / right[:, 2] + 480.))
        matches = np.column_stack((left_px, right_px))
        result = fit_calibration(matches, K, K)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')
        self.assertIn('diagnostics', result)
        # Depth-dependent epipolar misalignment cannot be removed by a pure
        # relative rotation and focal-scale correction.
        if result['accepted']:
            corrected = rectify_points(matches, K, K, result)
            self.assertGreater(np.percentile(np.abs(corrected[:, 1] - corrected[:, 3]), 95), 0.15)

    def test_no_consensus_is_rejected(self):
        matches, _ = scene(seed=105)
        rng = np.random.default_rng(106)
        # Destroy all correspondence association while preserving both marginal
        # image distributions, so coverage alone must not imply acceptance.
        matches[:, 2:] = matches[rng.permutation(len(matches)), 2:]
        result = fit_calibration(matches, K, K)
        self.assertFalse(result['accepted'], result)

    def test_invalid_arrays_and_intrinsics(self):
        matches, _ = scene()
        for bad, left, right in ((matches[:, :3], K, K),
                                  (matches.copy(), np.zeros((3, 3)), K),
                                  (matches.copy(), K, np.eye(2))):
            with self.subTest(shape=bad.shape, kshape=left.shape):
                with self.assertRaises((ValueError, TypeError)):
                    fit_calibration(bad, left, right)
        broken = matches.copy()
        broken[0, 0] = np.nan
        with self.assertRaises((ValueError, TypeError)):
            fit_calibration(broken, K, K)

    def test_rectification_cannot_certify_metric_depth(self):
        # Scaling the whole scene and baseline produces exactly the same pixels.
        # Therefore even perfect row alignment cannot measure the physical baseline.
        a, _ = scene(seed=12, baseline=0.065)
        rng = np.random.default_rng(12)
        z = rng.uniform(0.7, 3.0, 600)
        points = np.column_stack((rng.uniform(-0.65, 0.65, 600) * z,
                                  rng.uniform(-0.42, 0.42, 600) * z, z))
        b, _ = project(points * 2., baseline=0.13)
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
        result = fit_calibration(a, K, K)
        self.assertTrue(result['accepted'], result)
        self.assertEqual(result['metric_acceptance'], 'unevaluated')
        corrected = rectify_points(a, K, K, result)
        self.assertLess(np.max(np.abs(corrected[:, 1] - corrected[:, 3])), 0.01)


if __name__ == '__main__':
    unittest.main()
