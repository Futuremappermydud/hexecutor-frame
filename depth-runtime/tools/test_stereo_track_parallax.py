import unittest
import numpy as np
from stereo_direct_tracks import filter_left_parallax
from test_stereo_temporal_calibration import synthetic, rot, pixels, K


class TrackParallaxTests(unittest.TestCase):
    def test_known_motion_and_right_pixel_independence(self):
        rows, _ = synthetic()
        rotations = np.array([rot(np.array([.006, -.011, .004])*i) for i in range(5)])
        filtered, report = filter_left_parallax(rows, rotations, K)
        np.testing.assert_array_equal(filtered, rows)
        self.assertEqual(report['left_parallax_filter']['retained_tracks'], 60)
        altered = rows.copy(); altered[:, 4:] += 100.
        changed, new_report = filter_left_parallax(altered, rotations, K)
        np.testing.assert_array_equal(changed[:, :4], filtered[:, :4])
        self.assertEqual(report, new_report)
        few, _ = filter_left_parallax(rows[rows[:, 0] < 2], rotations, K)
        self.assertEqual(len(few), 0)

    def test_pure_rotation_is_not_parallax(self):
        _, points = synthetic()
        rows, rotations = [], []
        for i in range(5):
            rotation = rot(np.array([.05, -.08, .04])*i)
            rotations.append(rotation)
            uv = pixels(points @ rotation)
            rows.extend([i, track, *point, *(point-[20., 0.])] for track, point in enumerate(uv))
        filtered, report = filter_left_parallax(rows, rotations, K)
        self.assertEqual(len(filtered), 0)
        self.assertLess(max(t['parallax_degrees'] for t in report['left_parallax_filter']['tracks']), 1e-5)

    def test_empty_and_invalid(self):
        rotations = np.tile(np.eye(3), (3, 1, 1))
        result, _ = filter_left_parallax(np.empty((0, 6)), rotations, K)
        self.assertEqual(result.shape, (0, 6))
        for threshold in (0., np.nan, 180.):
            with self.assertRaises(ValueError):
                filter_left_parallax(np.empty((0, 6)), rotations, K, threshold)
        bad = rotations.copy(); bad[1, 0, 0] = -1
        with self.assertRaises(ValueError):
            filter_left_parallax(np.empty((0, 6)), bad, K)
        with self.assertRaises(ValueError):
            filter_left_parallax([[3, 0, 10, 20, 0, 20]], rotations, K)


if __name__ == '__main__':
    unittest.main()
