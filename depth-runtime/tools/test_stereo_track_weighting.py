import unittest
import numpy as np
from stereo_track_weighting import equal_track_soft_l1

class TrackWeightingTests(unittest.TestCase):
    def test_equal_total_weight_and_original_robust_threshold(self):
        ids=np.array([3,3,3,9]);loss=equal_track_soft_l1(ids)
        z=np.linspace(.1,4,16);rho=loss(z);weights=loss(np.zeros(16))[1]
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertAlmostEqual(weights[:12].sum(),weights[12:].sum())
        np.testing.assert_allclose(rho[1]/weights,1/np.sqrt(1+z))
        h=1e-5
        np.testing.assert_allclose((loss(z+h)[0]-loss(z-h)[0])/(2*h),rho[1],rtol=1e-8)
        np.testing.assert_allclose((loss(z+h)[1]-loss(z-h)[1])/(2*h),rho[2],rtol=1e-8)

    def test_track_duplication_preserves_relative_robust_objective(self):
        # Repeating only one track changes total objective scale, not its optimum.
        from scipy.optimize import least_squares
        def solve(y,ids):
            return least_squares(lambda x: np.repeat(x[0]-y,4),[0.],loss=equal_track_soft_l1(ids)).x
        np.testing.assert_allclose(solve(np.array([0.,2.]),[0,1]),
                                   solve(np.array([0.,0.,0.,2.]),[0,0,0,1]),atol=1e-6)

    def test_balanced_physical_fit_recovers_independent_camera_truth(self):
        from test_stereo_temporal_lens import fixture, K
        from stereo_temporal_calibration import fit_temporal_calibration
        rows, _, rotations, cameras, truth = fixture()
        result = fit_temporal_calibration(rows, K, K, .105, model='physical6',
            camera_rotations=rotations, camera_ray_models=cameras,
            config={'balance_track_weights': True})
        self.assertTrue(result['accepted'], result['diagnostics'])
        self.assertTrue(result['diagnostics']['balance_track_weights'])
        np.testing.assert_allclose(result['parameters_vector'], truth, atol=.001)
        with self.assertRaises(ValueError):
            fit_temporal_calibration(rows, K, K, .105, model='physical6',
                camera_ray_models=cameras, config={'balance_track_weights': 1})

if __name__=='__main__':unittest.main()
