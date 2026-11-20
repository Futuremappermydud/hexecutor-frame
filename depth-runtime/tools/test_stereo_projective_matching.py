import unittest
import numpy as np
from audit_stereo_projective_matching import plane_homography, apply_homography
from test_stereo_temporal_calibration import rot


class ProjectiveControlTests(unittest.TestCase):
    def test_independent_physical_plane_projection(self):
        kl = np.array([[380., 10., 308.], [0., 390., 300.], [0., 0., 1.]])
        kr = np.array([[382., 5., 307.], [0., 391., 309.], [0., 0., 1.]])
        rotation = np.array([.004, -.008, .003])
        rays = np.array([[x, y, 1.] for x in (-.5, 0., .5) for y in (-.4, 0., .4)])
        for normal in ([0., 0., 1.], [.5, 0., 1.], [0., -.5, 1.]):
            unit = np.array(normal)/np.linalg.norm(normal)
            points = rays*(1.5/(rays @ unit))[:, None]
            left = points @ kl.T; left = left[:, :2]/left[:, 2, None]
            right = (points-[.105, 0., 0.]) @ rot(rotation) @ kr.T
            expected = right[:, :2]/right[:, 2, None]
            h = plane_homography(kl, kr, .105, rotation, normal, 1.5)
            np.testing.assert_allclose(apply_homography(left, h), expected, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
