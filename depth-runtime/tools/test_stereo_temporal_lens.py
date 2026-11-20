"""Temporal physical-lens recovery from independent OpenCV camera projections."""
import unittest
import numpy as np
from stereo_lens_calibration import CameraRayModel, project_right
from stereo_temporal_calibration import fit_temporal_calibration
from test_stereo_lens_calibration import K, LENS, independent_projection
from test_stereo_temporal_calibration import rot
from evaluate_stereo_temporal_lens import heldout_prediction


def fixture(lens_changes=True, stationary=False, seed=1007, motion=(.045, .012, .008), varying_rotation=False, baseline_slopes=None, right_principal_shift=None):
    rng = np.random.default_rng(seed)
    z = rng.uniform(.8, 2.8, 60)
    points = np.column_stack((rng.uniform(-.5, .5, 60)*z, rng.uniform(-.35, .35, 60)*z, z))
    left_r, right_r = rot([.05, .3, .02]), rot([-.04, -.3, .03])
    relative = np.array([.004, -.008, .003]); scale = 1.008
    dl, dr = (.002, -.003) if lens_changes else (0., 0.)
    direction = np.array([1., *(baseline_slopes or (0., 0.))]); direction /= np.linalg.norm(direction)
    rows, rotations = [], []
    for frame in range(5):
        i = 0 if stationary else frame
        rate = np.array([.006, -.011, .004])*i
        if varying_rotation:
            rate += np.array([.003, .001, -.002])*i*i
        rotation = rot(rate)
        rotations.append(rotation)
        center = np.array(motion)*i
        left = (points-center) @ rotation
        lp = independent_projection(left, left_r, dl)
        rp = independent_projection((left-.105*direction) @ rot(relative), right_r, dr, scale, right_principal_shift or (0.,0.))
        rows.extend([frame, track, *l, *r] for track, (l, r) in enumerate(zip(lp, rp)))
    truth=np.array([*relative, np.log(scale), dl, dr, *(baseline_slopes or ())])
    if right_principal_shift is not None:truth=np.r_[truth,0.,0.,right_principal_shift]
    return np.array(rows), points, np.array(rotations), (CameraRayModel(K, left_r, LENS), CameraRayModel(K, right_r, LENS)), truth


class TemporalPhysicalLensTests(unittest.TestCase):
    def test_independent_nonhorizontal_baseline_recovery(self):
        for model in ('physical_baseline6', 'physical_baseline8'):
            with self.subTest(model=model):
                rows, points, rotations, cameras, truth = fixture(
                    lens_changes=model == 'physical_baseline8', baseline_slopes=(.003, -.002))
                result = fit_temporal_calibration(rows, K, K, .105, model=model,
                    camera_rotations=rotations, camera_ray_models=cameras)
                self.assertTrue(result['accepted'], result)
                np.testing.assert_allclose(result['parameters_vector'], truth, atol=.001)
                direction = np.array([1., .003, -.002]); direction /= np.linalg.norm(direction)
                expected = independent_projection((points-.105*direction) @ rot(truth[:3]),
                    cameras[1].rotation, truth[5], np.exp(truth[3]))
                np.testing.assert_allclose(project_right(points, cameras[1], truth, .105), expected, atol=1e-8)
                from stereo_lens_calibration import epipolar_residual
                pair=np.c_[cameras[0].project(points,truth[4]),expected]
                np.testing.assert_allclose(epipolar_residual(pair,*cameras,truth),0.,atol=1e-8)
                self.assertLess(np.max(np.abs(project_right(points, cameras[1], result['parameters_vector'], .105)-expected)), .3)
                self.assertAlmostEqual(np.linalg.norm(result['parameters']['baseline_direction_left']), 1.)

    def test_free_camera_rotations_recover_physical_yaw(self):
        rows, _, _, cameras, truth = fixture(varying_rotation=True)
        result = fit_temporal_calibration(rows, K, K, .105, model='physical6', camera_ray_models=cameras)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters_vector'], truth, atol=.001)

    def test_right_physical_principal_shift_recovery(self):
        rows,points,rotations,cameras,truth=fixture(right_principal_shift=(.4,-.3))
        fit=fit_temporal_calibration(rows,K,K,.105,model='physical_center8',camera_rotations=rotations,camera_ray_models=cameras)
        self.assertTrue(fit['accepted'],fit)
        np.testing.assert_allclose(fit['parameters_vector'],truth,atol=.01)
        expected=independent_projection((points-[.105,0,0])@rot(truth[:3]),cameras[1].rotation,truth[5],np.exp(truth[3]),(.4,-.3))
        np.testing.assert_allclose(project_right(points,cameras[1],truth,.105),expected,atol=1e-8)
        from stereo_lens_calibration import shifted_principal_camera
        corrected=shifted_principal_camera(cameras[1],(.4,-.3))
        right_points=(points-[.105,0,0])@rot(truth[:3])
        np.testing.assert_allclose(corrected.rays(expected,truth[5],np.exp(truth[3])),right_points/right_points[:,2,None],atol=1e-8)
        finite,valid=corrected.project_for_optimization(right_points,truth[5],np.exp(truth[3]))
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(finite,expected,atol=1e-8)
        from stereo_lens_calibration import epipolar_residual
        pair=np.c_[cameras[0].project(points,truth[4]),expected]
        np.testing.assert_allclose(epipolar_residual(pair,*cameras,truth),0.,atol=1e-8)

    def test_shared_head_motion_scale_with_independent_world_gauges(self):
        first,_,r1,cameras,truth=fixture()
        second,_,r2,_,_=fixture(seed=2511,motion=(-.025,.03,.02))
        second[:,0]+=10;second[:,1]+=10000
        rotations=np.concatenate((r1,r2));centers=np.array([np.array(v)*i for v in ((.045,.012,.008),(-.025,.03,.02)) for i in range(5)])
        origin=np.array([.035,-.02,.01]);lever=np.array([(r-np.eye(3))@origin for r in rotations])
        translation=(centers-lever)/1.7
        motion={'translation':translation,'lever_arm':lever}
        result=fit_temporal_calibration(np.vstack((first,second)),K,K,.105,model='physical6',camera_rotations=rotations,camera_ray_models=cameras,frame_groups=[0]*5+[1]*5,camera_motion=motion,global_rotation_initialization=True)
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['parameters_vector'],truth,atol=.001)
        self.assertAlmostEqual(result['diagnostics']['motion_translation_scale'],1.7,delta=.001)
        bad=translation.copy();bad[5,0]=.01
        with self.assertRaises(ValueError):
            fit_temporal_calibration(np.vstack((first,second)),K,K,.105,model='physical6',camera_rotations=rotations,camera_ray_models=cameras,frame_groups=[0]*5+[1]*5,camera_motion={'translation':bad,'lever_arm':lever})

    def test_shared_camera_head_axis_refinement(self):
        rows, _, true_rotations, cameras, truth = fixture(varying_rotation=True)
        correction = rot([.02, -.01, .015])
        prior = correction @ true_rotations @ correction.T
        result = fit_temporal_calibration(rows, K, K, .105, model='physical6',
            camera_rotations=prior, camera_ray_models=cameras, refine_camera_rotation=True)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['parameters_vector'], truth, atol=.001)
        np.testing.assert_allclose(result['camera_rotation_refinement_vector_rad'], [.02, -.01, .015], atol=.003)
        for pose, rotation in zip(result['poses'], true_rotations):
            np.testing.assert_allclose(rot(pose['rotation_vector_rad']), rotation, atol=.001)
        self.assertEqual(len(result['diagnostics']['calibration_profile_singular_values']), 6)

    def test_independent_windows_share_calibration_with_separate_gauges(self):
        first, _, r1, cameras, truth = fixture()
        second, _, r2, _, _ = fixture(seed=1009, motion=(-.025, .03, .02))
        second[:, 0] += 10; second[:, 1] += 10000
        observations = np.vstack((first, second))
        rotations = np.concatenate((r1, r2))
        groups = [11]*5+[29]*5
        result = fit_temporal_calibration(observations, K, K, .105, model='physical6',
            camera_rotations=rotations, camera_ray_models=cameras, frame_groups=groups)
        self.assertTrue(result['accepted'], result)
        self.assertEqual(result['diagnostics']['frame_group_count'], 2)
        np.testing.assert_allclose(result['parameters_vector'], truth, atol=.001)
        for frame in (0, 10):
            gauge = next(p for p in result['poses'] if p['frame_id'] == frame)
            np.testing.assert_allclose(gauge['translation_m'], 0., atol=1e-12)
            np.testing.assert_allclose(gauge['rotation_vector_rad'], 0., atol=1e-12)
        self.assertEqual({p['group_id'] for p in result['points']}, {11, 29})
        # Sharing a track across separately fixed world gauges is invalid.
        shared = observations.copy(); shared[shared[:, 0] >= 10, 1] -= 10000
        with self.assertRaises(ValueError):
            fit_temporal_calibration(shared, K, K, .105, model='physical6',
                camera_rotations=rotations, camera_ray_models=cameras, frame_groups=groups)

    def test_stationary_group_cannot_borrow_other_window_motion(self):
        first, _, r1, cameras, _ = fixture()
        second, _, r2, _, _ = fixture(stationary=True, seed=1010)
        second[:, 0] += 10; second[:, 1] += 10000
        result = fit_temporal_calibration(np.vstack((first, second)), K, K, .105, model='physical6',
            camera_rotations=np.concatenate((r1, r2)), camera_ray_models=cameras, frame_groups=[0]*5+[1]*5)
        self.assertFalse(result['accepted'], result)
        self.assertIn('insufficient_translation', result['diagnostics']['reasons'])

    def test_heldout_left_only_physical_triangulation(self):
        rows, _, rotations, cameras, truth = fixture()
        fit = {'accepted': True, 'parameters_vector': truth.tolist(),
               'poses': [{'frame_id': i, 'rotation_vector_rad': [.006*i, -.011*i, .004*i],
                          'translation_m': [.045*i, .012*i, .008*i]} for i in range(5)]}
        report = heldout_prediction(rows, fit, *cameras, .105)
        self.assertEqual(report['available_tracks'], 60)
        self.assertLess(max(report['scores']['candidate_right']['rms_xy_px']), 1e-7)
        altered = rows.copy(); altered[:, 4] += 3.
        shifted = heldout_prediction(altered, fit, *cameras, .105)
        self.assertEqual(report['available_tracks'], shifted['available_tracks'])
        self.assertEqual(report['skipped_tracks'], shifted['skipped_tracks'])
        self.assertAlmostEqual(shifted['scores']['candidate_right']['median_xy_px'][0], -3., delta=1e-7)
        self.assertEqual(report['scores']['left_reprojection'], shifted['scores']['left_reprojection'])
        missing_pose = dict(fit, poses=[p for p in fit['poses'] if p['frame_id'] != 2])
        partial = heldout_prediction(rows, missing_pose, *cameras, .105)
        self.assertEqual(partial['available_tracks'], 60)
        self.assertEqual(partial['omitted_observations_missing_training_pose'], 60)
        self.assertLess(max(partial['scores']['candidate_right']['rms_xy_px']), 1e-7)

    def test_nonlinear_heldout_refinement_uses_left_only(self):
        rows, _, _, cameras, truth = fixture(right_principal_shift=(.4, -.3))
        fit = {'accepted': True, 'parameters_vector': truth.tolist(),
               'poses': [{'frame_id': i, 'rotation_vector_rad': [.006*i, -.011*i, .004*i],
                          'translation_m': [.045*i, .012*i, .008*i]} for i in range(5)]}
        rows[:, 2:4] += np.random.default_rng(221).normal(0, .35, (len(rows), 2))
        linear = heldout_prediction(rows, fit, *cameras, .105)
        refined = heldout_prediction(rows, fit, *cameras, .105, refine_left_points=True)
        self.assertEqual(refined['available_tracks'], linear['available_tracks'])
        self.assertTrue(any(t['used'] for t in refined['left_point_refinement']['tracks']))
        self.assertLess(sum(np.square(refined['scores']['left_reprojection']['rms_xy_px'])),
                        sum(np.square(linear['scores']['left_reprojection']['rms_xy_px'])))
        altered = rows.copy(); altered[:, 4:] += [17., -9.]
        shifted = heldout_prediction(altered, fit, *cameras, .105, refine_left_points=True)
        self.assertEqual(refined['left_point_refinement'], shifted['left_point_refinement'])
        self.assertEqual(refined['scores']['left_reprojection'], shifted['scores']['left_reprojection'])

    def test_noisy_physical_recovery_on_independent_probes(self):
        rows, _, rotations, cameras, truth = fixture()
        rows[:, 2:] += np.random.default_rng(1008).normal(0., .05, rows[:, 2:].shape)
        fit = fit_temporal_calibration(rows, K, K, .105, model='physical6',
                    camera_rotations=rotations, camera_ray_models=cameras)
        self.assertTrue(fit['accepted'], fit)
        probes = np.array([[x*z, y*z, z] for z in (.7, 1.5, 2.5) for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
        expected = independent_projection((probes-[.105, 0., 0.]) @ rot(truth[:3]), cameras[1].rotation, truth[5], np.exp(truth[3]))
        prediction = project_right(probes, cameras[1], fit['parameters_vector'], .105)
        self.assertLess(np.max(np.abs(prediction[:, 0]-expected[:, 0])), .3)

    def test_independent_recovery_of_physical_models(self):
        for model in ('physical4', 'physical6'):
            with self.subTest(model=model):
                rows, points, rotations, cameras, truth = fixture(model == 'physical6')
                result = fit_temporal_calibration(rows, K, K, .105, model=model,
                    camera_rotations=rotations, camera_ray_models=cameras)
                self.assertTrue(result['accepted'], result)
                self.assertEqual(result['model'], 'temporal_'+model)
                self.assertNotIn('focal_scale', result['parameters'])
                self.assertEqual(result['diagnostics']['projection_domain_fraction'], 1.)
                estimated = np.array(result['parameters_vector'])
                np.testing.assert_allclose(estimated, truth, atol=.001)
                expected = independent_projection((points-[.105, 0., 0.]) @ rot(truth[:3]), cameras[1].rotation, truth[5], np.exp(truth[3]))
                self.assertLess(np.max(np.abs(project_right(points, cameras[1], estimated, .105)-expected)), .3)
                self.assertEqual(result['metric_acceptance'], 'unevaluated')

    def test_stationary_and_missing_camera_models_reject(self):
        rows, _, rotations, cameras, _ = fixture(stationary=True)
        result = fit_temporal_calibration(rows, K, K, .105, model='physical6',
                    camera_rotations=rotations, camera_ray_models=cameras)
        self.assertFalse(result['accepted'], result)
        self.assertIn('insufficient_translation', result['diagnostics']['reasons'])
        with self.assertRaises(ValueError):
            fit_temporal_calibration(rows, K, K, .105, model='physical6')
        early = fit_temporal_calibration(rows[:3], K, K, .105, model='physical6', camera_ray_models=cameras)
        self.assertFalse(early['accepted'])
        self.assertEqual(early['parameters_vector'], [0.]*6)

    def test_finite_trial_extension_matches_exact_projection_only_in_domain(self):
        camera = CameraRayModel(K, rot([.05, .3, .02]), LENS)
        good = np.array([[.1, -.1, 1.], [-.4, .3, 1.]])
        for delta in (-.02, 0., .02):
            for scale in (.952, 1., 1.051):
                projected, valid = camera.project_for_optimization(good, delta, scale)
                self.assertTrue(valid.all())
                np.testing.assert_allclose(projected, camera.project(good, delta, scale), atol=1e-9)
                extended, valid = camera.project_for_optimization(np.array([[0., 0., -1.], [100., 0., .1], [0., 0., 0.]]), delta, scale)
                self.assertTrue(np.isfinite(extended).all())
                self.assertFalse(valid.any())


if __name__ == '__main__':
    unittest.main()
