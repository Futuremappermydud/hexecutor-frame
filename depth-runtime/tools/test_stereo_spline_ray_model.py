import unittest
import numpy as np
from stereo_lens_calibration import CameraRayModel, shifted_principal_camera, calibrated_left_camera
from test_stereo_kb_spline import INTRINSICS

class SplineRayTest(unittest.TestCase):
    def test_distinct_nominal_actual_and_wrappers(self):
        intrinsic=np.array([[380.,0,308],[0,380.,308],[0,0,1.]])
        nominal=[328.,329.,530.,511.,.025,-.006,-.004,0.]
        plain=CameraRayModel(intrinsic,np.eye(3),nominal)
        camera=CameraRayModel(intrinsic,np.eye(3),nominal,INTRINSICS)
        points=np.array([[.1,.2,1],[-.4,.3,1],[0.,0.,1.]])
        pixels=camera.project(points)
        self.assertGreater(np.max(np.abs(pixels-plain.project(points))),1.)
        # Independent composition: spline raw -> original KB inverse -> pinhole.
        raw=np.array(camera.actual_spline.project(points.T)).T
        ray=plain._unproject_physical(raw)
        expected=(ray@intrinsic.T)[:,:2]/ray[:,2,None]
        np.testing.assert_allclose(pixels,expected,atol=1e-10)
        for current in (camera,shifted_principal_camera(camera,[.3,-.2]),
                        calibrated_left_camera(camera,np.r_[np.zeros(10),.002])):
            pixels=current.project(points,.001,1.002)
            recovered=current.rays(pixels,.001,1.002)
            np.testing.assert_allclose(recovered,points,atol=1e-10)
            trial,valid=current.project_for_optimization(points,.001,1.002)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(trial,pixels,atol=1e-10)

if __name__=='__main__':unittest.main()
