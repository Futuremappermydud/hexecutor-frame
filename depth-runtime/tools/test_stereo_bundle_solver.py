import unittest
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix,eye,issparse
from stereo_bundle_solver import colored_dense_jacobian,solve_bundle

class BundleSolverTests(unittest.TestCase):
    def test_colored_jacobian_at_bound_matches_analytic_derivative(self):
        fun=lambda x:np.array([x[0]**2+2*x[1],np.sin(x[1]),x[0]*x[2],np.exp(x[2])])
        pattern=csr_matrix([[1,1,0],[0,1,0],[1,0,1],[0,0,1]])
        x=np.array([0.,.3,-.2]);bounds=(np.array([0.,-1.,-1.]),np.ones(3))
        expected=np.array([[0,2,0],[0,np.cos(.3),0],[-.2,0,0],[0,0,np.exp(-.2)]])
        np.testing.assert_allclose(colored_dense_jacobian(fun,x,pattern,bounds),expected,atol=1e-7,rtol=0)

    def test_exact_solution_matches_independent_analytic_bounded_robust_solve(self):
        a=np.array([[1.,2.],[3.,-.5],[0.,1.],[2.,1.]])
        target=np.array([2.,1.,.1,4.]);fun=lambda x:a@x-target
        bounds=(np.zeros(2),np.ones(2));start=np.array([.4,.3])
        expected=least_squares(fun,start,jac=lambda x:a.copy(),bounds=bounds,loss='soft_l1',
             f_scale=1.,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7,max_nfev=100)
        got,d=solve_bundle(fun,start,csr_matrix(a!=0),bounds,'soft_l1',1.,100,True)
        self.assertEqual(d['solver'],'dense_exact');self.assertTrue(issparse(got.jac))
        np.testing.assert_allclose(got.x,expected.x,atol=1e-6,rtol=0)
        self.assertAlmostEqual(got.cost,expected.cost,places=10)

    def test_temporal_calibration_and_support_cleanup_keep_exact_mode(self):
        from test_stereo_temporal_lens import fixture
        from test_stereo_lens_calibration import K
        from stereo_temporal_calibration import fit_temporal_calibration
        from stereo_support_refit import refit_physical_support
        rows,_,rotations,cameras,truth=fixture()
        rows[(rows[:,1]==0)&(rows[:,0]>0),4]+=30.
        config={'exact_small_problem':True}
        initial=fit_temporal_calibration(rows,K,K,.105,model='physical6',config=config,
            camera_rotations=rotations,camera_ray_models=cameras,global_rotation_initialization=True,
            rotation_initialization_only=True)
        self.assertFalse(initial['accepted'],initial)
        self.assertIn('insufficient_support_after_outlier_rejection',initial['diagnostics']['reasons'])
        fit,report=refit_physical_support(rows,rotations,cameras,.105,initial,config,
            use_recorded_rotations=False,rotation_initialization_only=True)
        self.assertTrue(report['attempted']);self.assertTrue(fit['accepted'],fit)
        self.assertTrue(all(a['solver']=='dense_exact' for a in fit['diagnostics']['least_squares_solver_attempts']))
        np.testing.assert_allclose(fit['parameters_vector'],truth,atol=1e-4,rtol=0)

    def test_large_problem_stays_sparse_under_explicit_entry_limit(self):
        n=1500
        got,d=solve_bundle(lambda x:x-.2,np.ones(n),eye(n,format='csr'),
            (-np.inf,np.inf),'soft_l1',1.,100,True)
        self.assertEqual(d['solver'],'sparse_lsmr');self.assertTrue(issparse(got.jac))
        self.assertTrue(got.success);np.testing.assert_allclose(got.x,.2,atol=1e-7)

if __name__=='__main__':unittest.main()
