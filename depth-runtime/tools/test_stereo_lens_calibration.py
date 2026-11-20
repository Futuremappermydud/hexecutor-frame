import unittest
import cv2
import numpy as np
from stereo_lens_calibration import CameraRayModel, fit_lens_calibration, project_right
from test_stereo_temporal_calibration import rot

K = np.array([[380., 0., 308.], [0., 380., 308.], [0., 0., 1.]])
LENS = np.array([329., 331., 528., 510., .03, -.008, -.003, -.0002])


def independent_projection(points, rotation, delta=0., scale=1., principal_shift=(0.,0.)):
    physical_k = np.array([[LENS[0]*scale, 0., LENS[2]], [0., LENS[1]*scale, LENS[3]], [0., 0., 1.]])
    coefficients = LENS[4:].copy(); coefficients[0] += delta
    physical = np.asarray(points) @ rotation.T
    nominal_k=physical_k.copy();nominal_k[:2,:2]/=scale
    physical_k[:2,2]+=principal_shift
    raw, _ = cv2.fisheye.projectPoints(physical.reshape(1, -1, 3), np.zeros(3), np.zeros(3), physical_k, coefficients)
    return cv2.fisheye.undistortPoints(raw, nominal_k, LENS[4:], R=rotation.T, P=K).reshape(-1, 2)


class PhysicalLensCalibrationTests(unittest.TestCase):
    def test_camera_roundtrip_and_independent_opencv_projection(self):
        rotation = rot([.05, .3, .02])
        camera = CameraRayModel(K, rotation, LENS)
        points = np.array([[x, y, 1.] for y in (-.4, 0., .4) for x in (-.5, 0., .5)])
        for delta, scale in [(0., 1.), (.003, 1.008), (-.003, .99)]:
            expected = independent_projection(points, rotation, delta, scale)
            projected = camera.project(points, delta, scale)
            np.testing.assert_allclose(projected, expected, atol=1e-8)
            np.testing.assert_allclose(camera.rays(projected, delta, scale), points, atol=1e-9)

    def test_independent_multidepth_lens_and_rotation_recovery(self):
        rng = np.random.default_rng(914)
        z = rng.uniform(.7, 2.5, 600)
        points = np.column_stack((rng.uniform(-.5, .5, 600)*z, rng.uniform(-.4, .4, 600)*z, z))
        left_r, right_r, relative = rot([.05, .3, .02]), rot([-.04, -.3, .03]), np.array([.004, -.008, .003])
        left = CameraRayModel(K, left_r, LENS); right = CameraRayModel(K, right_r, LENS)
        lp = independent_projection(points, left_r, .002)
        rp = independent_projection((points-[.105, 0., 0.]) @ rot(relative), right_r, -.003, 1.008)
        fit = fit_lens_calibration(np.column_stack((lp, rp)), left, right)
        self.assertTrue(fit['accepted'], fit)
        np.testing.assert_allclose(fit['parameters_vector'], [*relative, np.log(1.008), .002, -.003], atol=1e-5)
        np.testing.assert_allclose(project_right(points, right, fit['parameters_vector'], .105), rp, atol=.01)
        self.assertEqual(fit['metric_acceptance'], 'unevaluated')

    def test_weak_support_and_out_of_range_rays(self):
        camera = CameraRayModel(K, np.eye(3), LENS)
        fit = fit_lens_calibration(np.tile([308., 308., 280., 308.], (200, 1)), camera, camera)
        self.assertFalse(fit['accepted'])
        self.assertIn('insufficient_unique_support', fit['diagnostics']['reasons'])
        with self.assertRaises(ValueError):
            camera.project([[0., 0., -1.]])
        for point in ([[0., 0., 0.]], [[np.nan, 0., 1.]]):
            with self.assertRaises(ValueError):
                camera.project(point)
        for scale in (0., np.nan):
            with self.assertRaises(ValueError):
                camera.rays([[308., 308.]], focal_scale=scale)


if __name__ == '__main__':
    unittest.main()
