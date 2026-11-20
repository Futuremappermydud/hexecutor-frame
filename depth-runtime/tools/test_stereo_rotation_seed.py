"""Supplied rotations seed geometry, without fixing the optimized rotations."""
import unittest
import numpy as np
import cv2
from stereo_temporal_calibration import fit_temporal_calibration
from test_stereo_temporal_lens import fixture
from test_stereo_lens_calibration import K
from test_stereo_temporal_calibration import rot

class RotationSeedTests(unittest.TestCase):
    def test_perturbed_seed_rotations_are_released_in_two_gauges(self):
        rows,_,truth_rot,cameras,truth=fixture(varying_rotation=True)
        second=rows.copy();second[:,0]+=10;second[:,1]+=10000
        seeds=np.array([r@rot(np.array([.001,-.0015,.0005])*i) for i,r in enumerate(truth_rot)])
        result=fit_temporal_calibration(np.vstack((rows,second)),K,K,.105,model='physical6',
            camera_rotations=np.concatenate((seeds,seeds)),camera_ray_models=cameras,
            frame_groups=[0]*5+[1]*5,global_rotation_initialization=True,rotation_initialization_only=True)
        self.assertTrue(result['accepted'],result)
        self.assertFalse(result['diagnostics']['camera_rotations_fixed'])
        self.assertTrue(result['diagnostics']['rotation_initialization_only'])
        np.testing.assert_allclose(result['parameters_vector'],truth,atol=1e-4,rtol=0)
        for pose in result['poses']:
            i=pose['frame_id']%10
            actual=cv2.Rodrigues(np.array(pose['rotation_vector_rad']))[0]
            np.testing.assert_allclose(actual,truth_rot[i],atol=1e-5,rtol=0)
        self.assertGreater(np.max(np.abs(seeds-truth_rot)),.004)

    def test_support_cleanup_preserves_free_seed_mode(self):
        from stereo_support_refit import refit_physical_support,refit_grouped_physical_support
        rows,_,rotations,cameras,truth=fixture()
        rows[(rows[:,1]==0)&(rows[:,0]>0),4]+=30.
        for grouped in (False,True):
            observations=rows.copy();seeds=rotations;groups=None
            if grouped:
                second=rows.copy();second[:,0]+=10;second[:,1]+=10000
                observations=np.vstack((rows,second));seeds=np.concatenate((rotations,rotations));groups=[0]*5+[1]*5
            initial=fit_temporal_calibration(observations,K,K,.105,model='physical6',camera_rotations=seeds,
                camera_ray_models=cameras,frame_groups=groups,global_rotation_initialization=True,rotation_initialization_only=True)
            self.assertFalse(initial['accepted'],initial)
            if grouped:
                fit,report=refit_grouped_physical_support(observations,seeds,groups,cameras,.105,initial,
                    global_initialization=True,use_recorded_rotations=False,rotation_initialization_only=True)
            else:
                fit,report=refit_physical_support(observations,seeds,cameras,.105,initial,
                    use_recorded_rotations=False,rotation_initialization_only=True)
            self.assertTrue(report['attempted'],report)
            self.assertTrue(fit['accepted'],fit)
            self.assertFalse(fit['diagnostics']['camera_rotations_fixed'])
            self.assertTrue(fit['diagnostics']['rotation_initialization_only'])
            np.testing.assert_allclose(fit['parameters_vector'],truth,atol=1e-4,rtol=0)

    def test_invalid_or_constrained_seed_modes_reject(self):
        rows,_,rotations,cameras,_=fixture()
        options=dict(model='physical6',camera_ray_models=cameras,rotation_initialization_only=True)
        for kwargs in ({},{'camera_rotations':rotations},
                       {'camera_rotations':rotations,'global_rotation_initialization':True,'refine_camera_rotation':True}):
            with self.assertRaises(ValueError):fit_temporal_calibration(rows,K,K,.105,**options,**kwargs)

if __name__=='__main__':unittest.main()
