import unittest
import numpy as np
from stereo_rotation_initialization import initialize_center
from test_stereo_temporal_calibration import rot
from test_stereo_temporal_lens import fixture
from test_stereo_lens_calibration import K
from stereo_temporal_calibration import fit_temporal_calibration


class RotationInitializationTests(unittest.TestCase):
    def test_known_center_skew_noise_and_outliers(self):
        rng = np.random.default_rng(1121)
        local = np.column_stack((rng.uniform(-.5, .5, 60), rng.uniform(-.4, .4, 60), rng.uniform(1., 3., 60)))
        rotation, center = rot([.03, -.07, .02]), np.array([.13, .02, .03])
        world = local @ rotation.T+center
        intrinsic = K.copy(); intrinsic[0, 1] = 15.
        q = local @ intrinsic.T; pixels = q[:, :2]/q[:, 2, None]
        pixels += rng.normal(0., .1, pixels.shape); pixels[:15, 0] += 20.
        prediction = initialize_center(world, pixels, intrinsic, rotation)
        self.assertIsNotNone(prediction)
        np.testing.assert_allclose(prediction, center, atol=.002)

    def test_degenerate_parallel_rays_reject(self):
        points = np.column_stack((np.zeros(15), np.zeros(15), np.linspace(1., 3., 15)))
        pixels = np.tile(K[:2, 2], (15, 1))
        self.assertIsNone(initialize_center(points, pixels, K, np.eye(3)))

    def test_physical_temporal_recovery_with_fixed_rotation_initialization(self):
        rows, _, rotations, cameras, truth = fixture()
        fit = fit_temporal_calibration(rows, K, K, .105, model='physical6', camera_rotations=rotations,
            camera_ray_models=cameras, fixed_rotation_initialization=True)
        self.assertTrue(fit['accepted'], fit)
        self.assertEqual(fit['diagnostics']['initialization_strategy'], 'fixed_rotation')
        np.testing.assert_allclose(fit['parameters_vector'], truth, atol=.001)


if __name__ == '__main__':
    unittest.main()
