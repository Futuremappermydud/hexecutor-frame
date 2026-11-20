import unittest
import numpy as np
from audit_stereo_temporal_pose import rotation_comparison, associate_poses
from test_stereo_temporal_calibration import rot


class TemporalPoseAuditTests(unittest.TestCase):
    def test_unknown_world_and_head_axes_cancel_in_angles(self):
        world, extrinsic = rot([.3, -.1, .4]), rot([-.2, .5, .1])
        head = np.array([rot([.02*i, -.01*i, .005*i]) for i in range(5)])
        camera = np.array([world @ r @ extrinsic for r in head])
        report = rotation_comparison(camera, head)
        self.assertLess(report['max_abs_angle_difference_deg'], 1e-12)
        wrong = camera.copy(); wrong[3] = wrong[3] @ rot([.01, .02, .03])
        self.assertGreater(rotation_comparison(wrong, head)['max_abs_angle_difference_deg'], .5)

    def test_bracket_interpolation_recovers_known_motion_and_rejects_bad_endpoints(self):
        rows=[]
        for time, angle, translation in [(100,0.,0.),(200,.2,2.)]:
            pose=np.column_stack((rot([0.,angle,0.]),[translation,0.,0.]))
            row={'mono_ns':str(time),'valid':'1','result':'200'}
            row.update({f'm{a}{b}':str(pose[a,b]) for a in range(3) for b in range(4)})
            rows.append(row)
        interpolated,diag=associate_poses(rows,[125],100,interpolate=True)
        np.testing.assert_allclose(interpolated[0,:,:3],rot([0.,.05,0.]),atol=1e-12)
        np.testing.assert_allclose(interpolated[0,:,3],[.5,0.,0.],atol=1e-12)
        self.assertEqual(diag[0]['interpolation_fraction'],.25)
        nearest,_=associate_poses(rows,[125],100)
        self.assertAlmostEqual(nearest[0,0,3],0.)
        exact,_=associate_poses(rows,[100],0,interpolate=True)
        np.testing.assert_array_equal(exact,nearest)
        with self.assertRaises(ValueError):associate_poses(rows,[99],100,interpolate=True)
        with self.assertRaises(ValueError):associate_poses(rows,[125],50,interpolate=True)
        # Nearest is valid, but the farther bracket endpoint may not be skipped.
        rows[1]['valid']='0'
        with self.assertRaises(ValueError):associate_poses(rows,[125],100,interpolate=True)
        rows[1]['valid']='1';rows[1]['m00']='2'
        with self.assertRaises(ValueError):associate_poses(rows,[125],100,interpolate=True)

    def test_nearest_pose_validity_and_clock_boundaries(self):
        rows = []
        for time in [100, 200, 300]:
            row = {'mono_ns': str(time), 'valid': '1', 'result': '200'}
            row.update({f'm{r}{c}': str(float(r == c)) for r in range(3) for c in range(4)})
            rows.append(row)
        selected, diag = associate_poses(rows, [101, 290], 10)
        self.assertEqual(selected.shape, (2, 3, 4))
        self.assertEqual([d['gap_ns'] for d in diag], [1, 10])
        with self.assertRaises(ValueError):
            associate_poses(rows, [311], 10)
        rows[1]['valid'] = '0'
        with self.assertRaises(ValueError):
            associate_poses(rows, [199], 100)
        rows[1]['valid'] = '1'; rows[1]['result'] = '201'
        with self.assertRaises(ValueError):
            associate_poses(rows, [199], 100)
        rows[1]['result'] = '200'; rows[1]['m00'] = '-1'
        with self.assertRaises(ValueError):
            associate_poses(rows, [199], 100)


if __name__ == '__main__':
    unittest.main()
