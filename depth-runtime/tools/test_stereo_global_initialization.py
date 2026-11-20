import unittest
import numpy as np
from test_stereo_lens_calibration import K,LENS
from stereo_lens_calibration import CameraRayModel
from stereo_temporal_calibration import fit_temporal_calibration
from test_stereo_temporal_calibration import rot

class GlobalInitializationTests(unittest.TestCase):
    def test_connected_graph_without_sequential_pnp_overlap(self):
        rng=np.random.default_rng(2221);points=np.c_[rng.uniform(-.4,.4,(50,2)),rng.uniform(1.,2.5,50)]
        rotations=np.array([rot([.004*i,-.009*i,.003*i]) for i in range(5)])
        centers=np.array([[.04*i,.007*i,.012*i] for i in range(5)])
        rows=[]
        for t,point in enumerate(points):
            edge=t//10
            for f in (edge,(edge+1)%5):
                left=(point-centers[f])@rotations[f];right=left-[.105,0,0]
                lp=K@left;rp=K@right
                rows.append([f,t,*(lp[:2]/lp[2]),*(rp[:2]/rp[2])])
        cameras=(CameraRayModel(K,np.eye(3),LENS),)*2
        ordinary=fit_temporal_calibration(rows,K,K,.105,model='physical4',camera_rotations=rotations,camera_ray_models=cameras)
        self.assertFalse(ordinary['accepted']);self.assertIn('pose_initialization_failed',ordinary['diagnostics']['reasons'])
        result=fit_temporal_calibration(rows,K,K,.105,model='physical4',camera_rotations=rotations,camera_ray_models=cameras,global_rotation_initialization=True)
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['parameters_vector'],0.,atol=1e-5)
        np.testing.assert_allclose([p['translation_m'] for p in result['poses']],centers,atol=1e-5)

    def test_nonzero_physical_calibration_and_explicit_rotation_requirement(self):
        from test_stereo_temporal_lens import fixture
        rows,_,rotations,cameras,truth=fixture()
        result=fit_temporal_calibration(rows,K,K,.105,model='physical6',camera_rotations=rotations,camera_ray_models=cameras,global_rotation_initialization=True)
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['parameters_vector'],truth,atol=.001)
        with self.assertRaises(ValueError):
            fit_temporal_calibration(rows,K,K,.105,global_rotation_initialization=True)

if __name__=='__main__':unittest.main()
