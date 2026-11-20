import unittest
import numpy as np
from scipy.sparse import csc_matrix
from stereo_profile_projection import profile_calibration_columns

class ProfileProjectionTests(unittest.TestCase):
    def test_dense_oracle_with_extreme_units_null_columns_and_redundancy(self):
        rng=np.random.default_rng(2611)
        basis,_=np.linalg.qr(rng.normal(size=(140,30)))
        scales=10.**np.linspace(-7,7,30)
        nuisance=basis*scales
        nuisance=np.c_[nuisance,nuisance[:,4],np.zeros(140)]
        calibration=rng.normal(size=(140,8))
        actual,diagnostics=profile_calibration_columns(csc_matrix(np.c_[calibration,nuisance]),8)
        expected=calibration-basis@(basis.T@calibration)
        np.testing.assert_allclose(actual,expected,atol=1e-8)
        self.assertTrue(all(x in (0,1,2,4,5) for x in diagnostics['stops']))
        self.assertEqual(diagnostics['zero_nuisance_columns'],1)

    def test_iteration_exhaustion_recovery_matches_dense_svd(self):
        rng=np.random.default_rng(6201)
        q,_=np.linalg.qr(rng.normal(size=(100,24)))
        v,_=np.linalg.qr(rng.normal(size=(24,24)))
        nuisance=q@np.diag(np.geomspace(1.,.01,24))@v.T
        nuisance=np.c_[nuisance,nuisance[:,3],np.zeros(100)]
        calibration=rng.normal(size=(100,4))
        jac=csc_matrix(np.c_[calibration,nuisance])
        norms=np.linalg.norm(nuisance,axis=0)
        scaled=nuisance/np.where(norms>0,norms,1.)
        u,singular,_=np.linalg.svd(scaled,full_matrices=False)
        basis=u[:,singular>1e-12]
        expected=calibration-basis@(basis.T@calibration)
        failed,d0=profile_calibration_columns(jac,4,max_iterations=1,recovery_iterations=1)
        self.assertEqual(d0['stops'],[7]*4)
        self.assertGreater(np.max(np.abs(failed-expected)),.01)
        recovered,d=profile_calibration_columns(jac,4,max_iterations=1,recovery_iterations=500)
        self.assertTrue(all(len(a)==2 for a in d['attempts']))
        self.assertTrue(all(stop in (0,1,2,4,5) for stop in d['stops']))
        np.testing.assert_allclose(recovered,expected,atol=2e-6,rtol=0)
        with self.assertRaises(ValueError):profile_calibration_columns(jac,4,recovery_iterations=True)

if __name__=='__main__':unittest.main()
