import unittest
from unittest.mock import patch
import numpy as np
from stereo_temporal_calibration import fit_temporal_calibration
from stereo_support_refit import refit_physical_support
from stereo_lens_calibration import project_right
from test_stereo_temporal_lens import fixture
from test_stereo_lens_calibration import K, independent_projection
from test_stereo_temporal_calibration import rot


class SupportRefitTests(unittest.TestCase):
    def test_drop_orphaned_track_and_refit_with_original_minima(self):
        rows, points, rotations, cameras, truth = fixture()
        rows[(rows[:, 1] == 0) & (rows[:, 0] > 0), 4] += 30.
        initial = fit_temporal_calibration(rows, K, K, .105, model='physical6',
                    camera_rotations=rotations, camera_ray_models=cameras)
        self.assertFalse(initial['accepted'], initial)
        self.assertIn('insufficient_support_after_outlier_rejection', initial['diagnostics']['reasons'])
        fit, report = refit_physical_support(rows, rotations, cameras, .105, initial)
        self.assertTrue(report['attempted'])
        retained = rows[report['retained_input_observation_indices']]
        self.assertEqual(len(retained), report['retained_stereo_observations'])
        self.assertNotIn(0, retained[:, 1])
        from evaluate_stereo_grouped_long import retained_problem
        reconstructed, selected_rotations = retained_problem(rows, rotations, report)
        np.testing.assert_array_equal(reconstructed, retained)
        np.testing.assert_allclose(selected_rotations[0], np.eye(3), atol=1e-12)
        self.assertTrue(fit['accepted'], fit)
        self.assertNotIn(0, [p['track_id'] for p in fit['points']])
        self.assertGreaterEqual(fit['diagnostics']['track_count'], 30)
        self.assertGreaterEqual(fit['diagnostics']['observation_count'], 90)
        expected = independent_projection((points-[.105, 0., 0.]) @ rot(truth[:3]), cameras[1].rotation, truth[5], np.exp(truth[3]))
        self.assertLess(np.max(np.abs(project_right(points, cameras[1], fit['parameters_vector'], .105)-expected)), .3)
        # Cleanup must preserve the caller's free-rotation ablation.
        with patch('stereo_support_refit.fit_temporal_calibration', wraps=fit_temporal_calibration) as fitter:
            free, free_report = refit_physical_support(rows, rotations, cameras, .105, initial,
                                                     use_recorded_rotations=False)
        self.assertIsNone(fitter.call_args.kwargs['camera_rotations'])
        self.assertTrue(free_report['attempted'])
        self.assertTrue(free['accepted'], free)


    def test_grouped_cleanup_preserves_separate_gauges_and_shared_calibration(self):
        from stereo_support_refit import refit_grouped_physical_support
        first,_,r1,cameras,truth=fixture()
        second,_,r2,_,_=fixture(seed=2222,motion=(-.025,.03,.02))
        first[(first[:,1]==0)&(first[:,0]>0),4]+=30.
        second[(second[:,1]==0)&(second[:,0]>0),4]+=30.
        second[:,0]+=10;second[:,1]+=10000
        rows=np.vstack((first,second));rotations=np.concatenate((r1,r2));groups=[11]*5+[29]*5
        initial=fit_temporal_calibration(rows,K,K,.105,model='physical6',camera_rotations=rotations,camera_ray_models=cameras,frame_groups=groups,global_rotation_initialization=True)
        self.assertFalse(initial['accepted'],initial)
        result,report=refit_grouped_physical_support(rows,rotations,groups,cameras,.105,initial,global_initialization=True)
        self.assertTrue(report['attempted'],report)
        self.assertTrue(result['accepted'],result)
        self.assertEqual({p['group_id'] for p in result['poses']},{11,29})
        np.testing.assert_allclose(result['parameters_vector'],truth,atol=.001)

    def test_grouped_cleanup_preserves_free_rotations(self):
        from stereo_support_refit import refit_grouped_physical_support
        first,_,r1,cameras,truth=fixture()
        second,_,r2,_,_=fixture(seed=2222,motion=(-.025,.03,.02))
        first[(first[:,1]==0)&(first[:,0]>0),4]+=30.
        second[(second[:,1]==0)&(second[:,0]>0),4]+=30.
        second[:,0]+=10;second[:,1]+=10000
        rows=np.vstack((first,second));rotations=np.concatenate((r1,r2));groups=[11]*5+[29]*5
        initial=fit_temporal_calibration(rows,K,K,.105,model='physical6',camera_rotations=rotations,camera_ray_models=cameras,frame_groups=groups,global_rotation_initialization=True)
        with patch('stereo_support_refit.fit_temporal_calibration',wraps=fit_temporal_calibration) as fitter:
            fit,report=refit_grouped_physical_support(rows,rotations,groups,cameras,.105,initial,use_recorded_rotations=False)
        self.assertIsNone(fitter.call_args.kwargs['camera_rotations'])
        self.assertTrue(report['attempted'])
        self.assertTrue(fit['accepted'],fit['diagnostics'])
        self.assertFalse(fit['diagnostics']['camera_rotations_fixed'])
        np.testing.assert_allclose(fit['parameters_vector'],truth,atol=.001)
        with self.assertRaises(ValueError):
            refit_grouped_physical_support(rows,rotations,groups,cameras,.105,initial,global_initialization=True,use_recorded_rotations=False)

    def test_grouped_cleanup_rebases_scaled_motion_after_gauge_frame_removal(self):
        from stereo_support_refit import refit_grouped_physical_support
        first,_,r1,cameras,truth=fixture()
        second,_,r2,_,_=fixture(seed=2512,motion=(-.025,.03,.02))
        first[first[:,0]==0,4]+=30.
        second[:,0]+=10;second[:,1]+=10000
        rows=np.vstack((first,second));rotations=np.concatenate((r1,r2));groups=[0]*5+[1]*5
        centers=np.array([np.array(v)*i for v in ((.045,.012,.008),(-.025,.03,.02)) for i in range(5)])
        lever=np.array([(r-np.eye(3))@np.array([.035,-.02,.01]) for r in rotations])
        motion={'translation':(centers-lever)/1.7,'lever_arm':lever}
        initial=fit_temporal_calibration(rows,K,K,.105,model='physical6',camera_rotations=rotations,camera_ray_models=cameras,frame_groups=groups,camera_motion=motion,global_rotation_initialization=True)
        self.assertFalse(initial['accepted'],initial)
        fit,report=refit_grouped_physical_support(rows,rotations,groups,cameras,.105,initial,global_initialization=True,camera_motion=motion)
        self.assertTrue(report['attempted'],report)
        self.assertTrue(fit['accepted'],fit)
        self.assertNotIn(0,[p['frame_id'] for p in fit['poses']])
        self.assertAlmostEqual(fit['diagnostics']['motion_translation_scale'],1.7,delta=.001)
        np.testing.assert_allclose(fit['parameters_vector'],truth,atol=.001)

    def test_unrelated_rejections_cannot_be_revived(self):
        rows, _, rotations, cameras, _ = fixture()
        for reason in ('rank_or_conditioning_failure', 'insufficient_translation', 'lens_parameter_bound_reached'):
            initial = {'accepted': False, 'diagnostics': {'reasons': [reason], 'converged': True}, 'poses': [], 'points': []}
            result, report = refit_physical_support(rows, rotations, cameras, .105, initial)
            self.assertIs(result, initial)
            self.assertFalse(report['attempted'])


if __name__ == '__main__':
    unittest.main()
