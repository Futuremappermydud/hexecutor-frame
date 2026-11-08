"""Independent angular-geometry tests; no captured data or camera/DSP access."""
import unittest
import numpy as np

from ir_yaw_normalization import normalize_matches, yaw_sensitivity


def rotated_projection(u, v, yaw_deg, roll_deg=.4):
    # Independent axis-angle reference, using unit directions instead of the
    # production Rz/Ry matrix composition and nonunit spherical rays.
    theta, phi = (u-255.5)*np.radians(.145), (v-319.5)*np.radians(.145)
    p = np.stack([np.sin(theta), np.cos(theta)*np.sin(phi),
                  np.cos(theta)*np.cos(phi)], axis=-1)
    roll = np.radians(roll_deg)
    axis = np.array([np.sin(roll), np.cos(roll), 0])
    angle = -np.radians(yaw_deg)
    q = p*np.cos(angle) + np.cross(axis, p)*np.sin(angle)
    q += (p@axis)[:, None]*axis*(1-np.cos(angle))
    return np.arctan2(q[:, 0], np.hypot(q[:, 1], q[:, 2]))/np.radians(.145)+255.5


class GeometryTests(unittest.TestCase):
    def test_small_angle_agrees_with_analytic_cosine_across_image(self):
        u, v = np.meshgrid(np.linspace(15, 495, 25), np.linspace(15, 625, 31))
        expected = np.cos((v-319.5)*np.radians(.145))/.145
        actual = yaw_sensitivity(u, v, roll_deg=0, delta_deg=-1e-5)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertGreater(np.ptp(actual), 1.0)  # A global scale cannot satisfy this field.

    def test_finite_angle_matches_independent_axis_angle_rotation(self):
        rng = np.random.default_rng(8)
        u, v = rng.uniform(-20, 510, 1000), rng.uniform(10, 630, 1000)
        for roll in [0, .4, -1.2]:
            for delta in [-.3, .1]:
                expected = (u-rotated_projection(u, v, delta, roll))/delta
                actual = yaw_sensitivity(u, v, roll_deg=roll, delta_deg=delta)
                np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

    def test_known_yaw_recovery_across_image_and_disparity(self):
        rng = np.random.default_rng(13)
        u, v = rng.uniform(15, 495, 1000), rng.uniform(15, 625, 1000)
        d0 = rng.uniform(2, 70, 1000)
        for true_yaw in [-.7, -.35, 0, .1]:
            expected = d0 + (u-d0)-rotated_projection(u-d0, v, true_yaw)
            samples = np.column_stack([np.arange(1000), np.zeros(1000), u, v,
                                       np.ones(1000), expected, d0, d0, np.ones(1000), np.ones(1000)])
            estimated, _ = normalize_matches(samples)
            self.assertLess(np.max(np.abs(estimated-true_yaw)), .005)

    def test_rejects_invalid_input_instead_of_dropping_matches(self):
        for samples in [np.zeros((0, 10)), np.zeros((2, 9)), np.full((2, 10), np.nan)]:
            with self.assertRaises(ValueError):
                normalize_matches(samples)
        with self.assertRaises(ValueError):
            yaw_sensitivity(np.array([1]), np.array([1]), delta_deg=0)


if __name__ == '__main__':
    unittest.main()
