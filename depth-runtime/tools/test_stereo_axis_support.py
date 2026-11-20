"""Support cleanup must retain the shared physical camera/head-axis model."""
import unittest
import numpy as np
from stereo_temporal_calibration import fit_temporal_calibration
from stereo_support_refit import refit_physical_support,refit_grouped_physical_support
from test_stereo_temporal_lens import fixture
from test_stereo_temporal_calibration import rot
from test_stereo_lens_calibration import K

class AxisSupportTests(unittest.TestCase):
    def test_single_and_grouped_cleanup_preserve_injected_axis_correction(self):
        rows,_,true_rotations,cameras,truth=fixture(varying_rotation=True)
        axis=np.array([.02,-.01,.015]);d=rot(axis);prior=np.array([d@r@d.T for r in true_rotations])
        rows[(rows[:,1]==0)&(rows[:,0]>0),4]+=30.
        config={'block_precondition':True}
        for grouped in (False,True):
            obs=rows;rotations=prior;groups=None
            if grouped:
                second=rows.copy();second[:,0]+=10;second[:,1]+=10000
                obs=np.vstack((rows,second));rotations=np.concatenate((prior,prior));groups=[0]*5+[1]*5
            initial=fit_temporal_calibration(obs,K,K,.105,model='physical6',config=config,
                camera_rotations=rotations,camera_ray_models=cameras,frame_groups=groups,
                global_rotation_initialization=True,refine_camera_rotation=True)
            self.assertFalse(initial['accepted'],initial)
            if grouped:
                fit,report=refit_grouped_physical_support(obs,rotations,groups,cameras,.105,initial,config,global_initialization=True)
            else:fit,report=refit_physical_support(obs,rotations,cameras,.105,initial,config)
            self.assertTrue(report['attempted'],report);self.assertTrue(fit['accepted'],fit)
            np.testing.assert_allclose(fit['camera_rotation_refinement_vector_rad'],axis,atol=.003,rtol=0)
            np.testing.assert_allclose(fit['parameters_vector'],truth,atol=.001,rtol=0)
            self.assertTrue(fit['diagnostics']['camera_rotations_fixed'])
            self.assertFalse(fit['diagnostics']['rotation_initialization_only'])

if __name__=='__main__':unittest.main()
