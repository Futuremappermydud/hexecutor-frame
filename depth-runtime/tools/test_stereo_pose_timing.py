import unittest
import numpy as np
from audit_stereo_pose_timing import score_left_tracks
from test_stereo_temporal_calibration import synthetic, rot, K


class LeftPoseTimingTests(unittest.TestCase):
    def test_known_geometry_and_exclusion_of_right_pixels(self):
        rows, _ = synthetic()
        rotations = np.array([rot(np.array([.006, -.011, .004])*i) for i in range(5)])
        centers = np.arange(5)[:, None] * np.array([[.045, .012, .008]])
        result = score_left_tracks(rows, rotations, centers, K)
        self.assertEqual(result['training']['valid_tracks'], 48)
        self.assertEqual(result['heldout']['valid_tracks'], 12)
        self.assertLess(result['training']['rms_px'], 1e-7)
        altered = rows.copy(); altered[:, 4:6] += 100.
        self.assertEqual(result, score_left_tracks(altered, rotations, centers, K))
        # A different trajectory direction cannot be repaired by point depths.
        wrong = centers.copy(); wrong[2:, 1] += .02
        self.assertGreater(score_left_tracks(rows, rotations, wrong, K)['training']['rms_px'], .5)

    def test_scale_ambiguity_and_stationary_rejection(self):
        rows, _ = synthetic()
        rotations = np.array([rot(np.array([.006, -.011, .004])*i) for i in range(5)])
        centers = np.arange(5)[:, None] * np.array([[.045, .012, .008]])
        scaled = score_left_tracks(rows, rotations, centers*3., K)
        self.assertLess(scaled['training']['rms_px'], 1e-7)
        stationary, _ = synthetic(stationary=True)
        rejected = score_left_tracks(stationary, np.tile(np.eye(3), (5, 1, 1)), np.zeros((5, 3)), K)
        self.assertEqual(rejected['training']['valid_tracks'], 0)
        self.assertTrue(all(t['error'] == 'weak_parallax' for t in rejected['tracks']))


if __name__ == '__main__':
    unittest.main()
