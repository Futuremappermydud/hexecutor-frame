import unittest
import numpy as np
from test_stereo_temporal_lens import fixture
from stereo_fixed_geometry import fit_fixed_geometry
from evaluate_stereo_temporal_lens import heldout_prediction

class FixedGeometryTests(unittest.TestCase):
    def test_independent_nonzero_geometry_and_disjoint_tracks(self):
        rows,_,rotations,cameras,truth=fixture()
        support=rows[rows[:,1]%5!=0];evaluation=rows[rows[:,1]%5==0]
        frozen=truth.copy();fit=fit_fixed_geometry(support,rotations,cameras,.105,frozen)
        self.assertTrue(fit['accepted'],fit['diagnostics'])
        np.testing.assert_array_equal(fit['parameters_vector'],truth)
        np.testing.assert_array_equal(frozen,truth)
        np.testing.assert_allclose([p['translation_m'] for p in fit['poses']],[[.045*i,.012*i,.008*i] for i in range(5)],atol=1e-5)
        self.assertFalse(set(p['track_id'] for p in fit['points'])&set(evaluation[:,1]))
        report=heldout_prediction(evaluation,fit,*cameras,.105)
        self.assertLess(max(report['scores']['candidate_right']['rms_xy_px']),1e-3)
        altered=evaluation.copy();altered[:,4:]+=[5.,-4.]
        changed=heldout_prediction(altered,fit,*cameras,.105)
        self.assertEqual(report['scores']['left_reprojection'],changed['scores']['left_reprojection'])

    def test_known_calibration_can_have_noisy_weak_motion_prediction_error(self):
        # Independent image projection with exact calibration and poses. The
        # nuisance acceptance flag must not imply a metric accuracy certificate.
        rows,_,r,c,truth=fixture(motion=(.015,.004,.003))
        rows[:,2:]+=np.random.default_rng(35001).normal(0,.35,(len(rows),4))
        fit=fit_fixed_geometry(rows[rows[:,1]%5!=0],r,c,.105,truth)
        self.assertTrue(fit['accepted'])
        np.testing.assert_array_equal(fit['parameters_vector'],truth)
        report=heldout_prediction(rows[rows[:,1]%5==0],fit,*c,.105)
        self.assertGreater(report['scores']['candidate_right']['rms_xy_px'][0],.3)
        self.assertEqual(fit['metric_acceptance'],'unevaluated')

    def test_disconnected_support_rejects(self):
        rows,_,r,c,p=fixture()
        second=rows.copy();second[:,0]+=5;second[:,1]+=100
        fit=fit_fixed_geometry(np.vstack((rows,second)),np.concatenate((r,r)),c,.105,p)
        self.assertFalse(fit['accepted'])
        self.assertEqual(fit['diagnostics']['reasons'],['disconnected_support'])

    def test_insufficient_and_invalid(self):
        rows,_,r,c,p=fixture()
        self.assertFalse(fit_fixed_geometry(rows[:20],r[:1],c,.105,p)['accepted'])
        with self.assertRaises(ValueError):fit_fixed_geometry(rows,r,c,.105,np.r_[p,0,0])

if __name__=='__main__':unittest.main()
