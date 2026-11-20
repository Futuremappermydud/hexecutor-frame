import unittest
import cv2
import numpy as np
from test_stereo_temporal_lens import fixture,K
from test_stereo_lens_calibration import independent_projection
from stereo_temporal_calibration import fit_temporal_calibration
from stereo_lens_calibration import calibrated_left_camera,project_right,epipolar_residual
from stereo_support_refit import physical_inlier_mask
from evaluate_stereo_grouped_lens import probe_stability
from evaluate_stereo_temporal_lens import heldout_prediction
from evaluate_stereo_lens_calibration import tag_score
from stereo_fixed_geometry import fit_fixed_geometry
from stereo_residual_audit import residual_structure

class LeftFocalTests(unittest.TestCase):
    def test_independent_recovery_and_all_projection_consumers(self):
        for scale in (.998,1.002):
            rows,points,rotations,cameras,truth=fixture()
            for i,r in enumerate(rotations):
                local=(points-np.array([.045,.012,.008])*i)@r
                rows[rows[:,0]==i,2:4]=independent_projection(local,cameras[0].rotation,truth[4],scale)
            expected=np.r_[truth,0.,0.,0.,0.,np.log(scale)]
            fit=fit_temporal_calibration(rows,K,K,.105,model='physical_focal7',camera_rotations=rotations,camera_ray_models=cameras)
            self.assertTrue(fit['accepted'],fit['diagnostics'])
            np.testing.assert_allclose(fit['parameters_vector'],expected,atol=1e-4)
            left=calibrated_left_camera(cameras[0],expected)
            pixels=left.project(points,truth[4])
            np.testing.assert_allclose(left.rays(pixels,truth[4]),points/points[:,2,None],atol=1e-8)
            rp=project_right(points,cameras[1],expected,.105)
            np.testing.assert_allclose(epipolar_residual(np.c_[pixels,rp],*cameras,expected),0,atol=1e-7)
            self.assertTrue(np.all(physical_inlier_mask(rows,cameras,.105,fit)))
            self.assertTrue(probe_stability([fit,fit,fit],*cameras,.105)['stability_passed'])
            report=heldout_prediction(rows,fit,*cameras,.105)
            self.assertLess(max(report['scores']['candidate_right']['rms_xy_px']),.001)
            self.assertLess(max(residual_structure(rows,fit,cameras,.105)['overall']['rms_lx_ly_rx_ry_px']),.001)
            fixed=fit_fixed_geometry(rows[rows[:,1]%5!=0],rotations,cameras,.105,expected)
            self.assertTrue(fixed['accepted'],fixed['diagnostics'])
            held=heldout_prediction(rows[rows[:,1]%5==0],fixed,*cameras,.105)
            self.assertLess(max(held['scores']['candidate_right']['rms_xy_px']),.001)
            corners=np.array([[-.06,-.06,1.4],[.06,-.06,1.4],[.06,.06,1.4],[-.06,.06,1.4]])
            left_pixels=independent_projection(corners,cameras[0].rotation,truth[4],scale)
            right_pixels=independent_projection((corners-[.105,0,0])@cv2.Rodrigues(truth[:3])[0],cameras[1].rotation,truth[5],np.exp(truth[3]))
            external=tag_score([{'left':left_pixels,'right':right_pixels}],*cameras,expected,.105,.12)
            self.assertLess(max(external['rms_xy_px']),1e-5)


    def test_stationary_unknown_left_focal_remains_rejected(self):
        rows,_,rotations,cameras,_=fixture(stationary=True)
        fit=fit_temporal_calibration(rows,K,K,.105,model='physical_focal7',camera_rotations=rotations,camera_ray_models=cameras)
        self.assertFalse(fit['accepted'],fit['diagnostics'])
        self.assertEqual(fit['metric_acceptance'],'unevaluated')

if __name__=='__main__':unittest.main()
