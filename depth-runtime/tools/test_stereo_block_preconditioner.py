import unittest
import numpy as np
from scipy.sparse import csc_matrix
from stereo_block_preconditioner import block_rotation

class BlockPreconditionerTests(unittest.TestCase):
    def test_orthogonal_map_preserves_objective_bounds_and_column_span(self):
        rng=np.random.default_rng(6801);j=rng.normal(size=(90,11));j[:,2:5]@=np.array([[1,3,1],[0,.1,3],[0,0,.03]])
        bounds=(np.r_[-.1,-.2,np.full(9,-np.inf)],np.r_[.1,.2,np.full(9,np.inf)])
        t,d=block_rotation(csc_matrix(j),[(2,3),(5,6)],bounds);t=t.toarray()
        np.testing.assert_allclose(t.T@t,np.eye(11),atol=1e-14,rtol=0)
        np.testing.assert_array_equal(t[:2],np.eye(11)[:2])
        x=rng.normal(size=11);q=t.T@x
        np.testing.assert_allclose(j@x,(j@t)@q,atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(x),np.linalg.norm(q),atol=1e-14)
        for start,width in ((2,3),(5,6)):
            b=(j@t)[:,start:start+width];gram=b.T@b
            np.testing.assert_allclose(gram-np.diag(np.diag(gram)),0.,atol=1e-8)
        u,s,_=np.linalg.svd(j[:,2:],full_matrices=False);v,ss,_=np.linalg.svd((j@t)[:,2:],full_matrices=False)
        np.testing.assert_allclose(u@u.T,v@v.T,atol=1e-10,rtol=0)
        self.assertEqual(d['variables_removed'],0)

    def test_solver_restores_physical_solution_jacobian_and_gradient(self):
        from stereo_bundle_solver import solve_bundle
        rng=np.random.default_rng(6802);a=rng.normal(size=(80,8))
        a[:,2:5]=a[:,2:5]@np.array([[1.,3.,1.],[0.,.1,3.],[0.,0.,.03]])
        truth=np.array([.1,-.2,.3,.4,-.2,.1,-.1,.2]);target=a@truth
        bounds=(np.r_[-.5,-.5,np.full(6,-np.inf)],np.r_[.5,.5,np.full(6,np.inf)])
        result,d=solve_bundle(lambda x:a@x-target,np.zeros(8),csc_matrix(a!=0),bounds,
            'linear',1.,100,nuisance_blocks=[(2,3),(5,3)])
        self.assertTrue(result.success,result.message)
        np.testing.assert_allclose(result.x,truth,atol=1e-6,rtol=0)
        np.testing.assert_allclose(result.jac.toarray(),a,atol=1e-6,rtol=0)
        np.testing.assert_allclose(result.fun,a@result.x-target,atol=1e-12)
        np.testing.assert_allclose(result.grad,a.T@result.fun,atol=1e-8,rtol=0)
        self.assertEqual(d['block_preconditioner']['rotated_blocks'],2)

    def test_temporal_known_calibration_with_free_seeded_rotations(self):
        from test_stereo_temporal_lens import fixture
        from test_stereo_lens_calibration import K
        from stereo_temporal_calibration import fit_temporal_calibration
        rows,_,rotations,cameras,truth=fixture(varying_rotation=True)
        fit=fit_temporal_calibration(rows,K,K,.105,model='physical6',
            config={'block_precondition':True},camera_rotations=rotations,camera_ray_models=cameras,
            global_rotation_initialization=True,rotation_initialization_only=True)
        self.assertTrue(fit['accepted'],fit)
        np.testing.assert_allclose(fit['parameters_vector'],truth,atol=1e-4,rtol=0)
        self.assertTrue(all(a['block_preconditioner']['variables_removed']==0 for a in fit['diagnostics']['least_squares_solver_attempts']))

    def test_null_blocks_preserved_and_unsafe_maps_reject(self):
        j=csc_matrix(np.zeros((20,6)));t,d=block_rotation(j,[(0,3),(3,3)],(-np.inf,np.inf))
        np.testing.assert_array_equal(t.toarray(),np.eye(6));self.assertEqual(d['rank_deficient_blocks_unchanged'],2)
        for blocks,bounds in (([(0,3),(2,3)],(-np.inf,np.inf)), ([(0,3)],(0.,np.inf)), ([(0,4)],(-np.inf,np.inf))):
            with self.assertRaises(ValueError):block_rotation(j,blocks,bounds)

if __name__=='__main__':unittest.main()
