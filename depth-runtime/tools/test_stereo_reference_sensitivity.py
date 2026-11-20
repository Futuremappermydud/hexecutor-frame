import unittest
import cv2
import numpy as np
from stereo_reference_sensitivity import left_pose_prediction,reference_sensitivity
from test_stereo_temporal_lens import fixture
from test_stereo_lens_calibration import independent_projection

class ReferenceSensitivityTests(unittest.TestCase):
    def test_independent_projection_and_right_pixel_independence(self):
        _,_,_,cameras,p=fixture()
        obj=np.array([[-.06,-.06,0],[.06,-.06,0],[.06,.06,0],[-.06,.06,0]])
        points=obj@cv2.Rodrigues(np.array([.2,-.3,.1]))[0].T+[.1,.02,1.5]
        pixels=independent_projection(points,cameras[0].rotation,p[4])
        expected=independent_projection((points-[.105,0,0])@cv2.Rodrigues(p[:3])[0],cameras[1].rotation,p[5],np.exp(p[3]))
        actual,g=left_pose_prediction(pixels,*cameras,p,.105,.12)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        first=reference_sensitivity([{'left':pixels,'right':expected}],*cameras,p,.105,.12,draws=8,sigmas=(0.,.1))
        second=reference_sensitivity([{'left':pixels,'right':expected+1000}],*cameras,p,.105,.12,draws=8,sigmas=(0.,.1))
        self.assertEqual(first,second)
        np.testing.assert_allclose(first['observations'][0]['perturbations'][0]['rms_prediction_change_xy_px'],0.,atol=1e-12)
        self.assertGreater(first['observations'][0]['perturbations'][1]['rms_prediction_change_xy_px'][0],0.)
if __name__=='__main__':unittest.main()
