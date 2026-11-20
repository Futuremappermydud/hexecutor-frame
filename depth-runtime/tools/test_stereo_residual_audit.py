import unittest
import numpy as np
from stereo_residual_audit import residual_structure
from test_stereo_temporal_lens import fixture

class ResidualAuditTests(unittest.TestCase):
    def test_independent_projection_and_known_age_error(self):
        rows,points,rotations,cameras,truth=fixture(baseline_slopes=(.003,-.002))
        import cv2
        fit={'accepted':False,'parameters_vector':truth.tolist(),
             'poses':[{'frame_id':i,'rotation_vector_rad':cv2.Rodrigues(r)[0].ravel().tolist(),
                       'translation_m':(np.array([.045,.012,.008])*i).tolist()} for i,r in enumerate(rotations)],
             'points':[{'track_id':i,'position_m':p.tolist()} for i,p in enumerate(points)]}
        base=residual_structure(rows,fit,cameras,.105)
        np.testing.assert_allclose(base['overall']['rms_lx_ly_rx_ry_px'],0,atol=1e-7)
        self.assertFalse(base['training_accepted'])
        shifted=rows.copy();shifted[:,4]+=shifted[:,0]*.2
        report=residual_structure(shifted,fit,cameras,.105)
        for age in report['by_track_age']:
            self.assertAlmostEqual(age['mean_lx_ly_rx_ry_px'][2],-.2*age['age_frames'],places=7)
        fit['poses']=fit['poses'][1:]
        missing=residual_structure(shifted,fit,cameras,.105)
        self.assertEqual(missing['omitted_missing_pose_or_point'],60)
        self.assertEqual(missing['overall']['observations'],240)

    def test_physical_principal_shift_uses_actual_optical_center(self):
        import cv2
        rows, points, rotations, cameras, truth = fixture(right_principal_shift=(1.3, -.8))
        fit = {'accepted': True, 'parameters_vector': truth.tolist(),
               'poses': [{'frame_id': i, 'rotation_vector_rad': cv2.Rodrigues(r)[0].ravel().tolist(),
                          'translation_m': (np.array([.045, .012, .008])*i).tolist()}
                         for i, r in enumerate(rotations)],
               'points': [{'track_id': i, 'position_m': p.tolist()} for i, p in enumerate(points)]}
        report = residual_structure(rows, fit, cameras, .105)
        np.testing.assert_allclose(report['overall']['rms_lx_ly_rx_ry_px'], 0, atol=1e-7)

    def test_empty_or_invalid(self):
        rows,_,_,cameras,_=fixture()
        self.assertFalse(residual_structure(rows,{'accepted':False},cameras,.105)['available'])
        with self.assertRaises(ValueError):residual_structure(np.vstack((rows,rows[:1])),{},cameras,.105)

if __name__=='__main__':unittest.main()
