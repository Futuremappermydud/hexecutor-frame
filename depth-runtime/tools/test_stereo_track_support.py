import unittest
import json
import numpy as np
from stereo_track_support import prune_training_support
from test_stereo_temporal_calibration import synthetic, rot, K


class TrackSupportTests(unittest.TestCase):
    def test_sparse_endpoint_and_orphan_track_are_removed_without_lowering_minima(self):
        rows, _ = synthetic()
        endpoint = rows[(rows[:, 0] == 4) & (rows[:, 1] < 5)].copy(); endpoint[:, 0] = 5
        orphan = rows[(rows[:, 0] >= 3) & (rows[:, 1] == 0)].copy(); orphan[:, 1] = 1000
        tail = endpoint[endpoint[:, 1] == 0].copy(); tail[:, 1] = 1000
        data = np.vstack((rows, endpoint, orphan, tail))
        rotations = np.array([rot(np.array([.006, -.011, .004])*i) for i in range(5)] + [rot(np.array([.006, -.011, .004])*4)])
        pruned, frames, report = prune_training_support(data, rotations, K)
        self.assertEqual(frames, [0, 1, 2, 3, 4])
        self.assertTrue(report['support_pruning']['eligible'])
        json.dumps({'frames': frames, 'diagnostics': report}, allow_nan=False)
        self.assertNotIn(1000, pruned[:, 1])
        np.testing.assert_array_equal(pruned, rows)
        self.assertTrue(all(r['output_observations'] <= r['input_observations'] for r in report['support_pruning']['rounds']))
        changed = data.copy(); changed[:, 4:6] += 100.
        other, selected, other_report = prune_training_support(changed, rotations, K)
        self.assertEqual(frames, selected); self.assertEqual(report, other_report)
        np.testing.assert_array_equal(pruned[:, :4], other[:, :4])

    def test_connected_component_selection_uses_training_support_only(self):
        rows = []
        for start, count, track_offset in [(0, 30, 0), (3, 40, 100)]:
            for frame in range(start, start+3):
                for track in range(count):
                    # Near K's principal point and 16-pixel total movement:
                    # every track clears the independent 0.5-degree gate.
                    x, y = 620+track%8*10+(frame-start)*8, 440+track//8*10
                    rows.append([frame, track+track_offset, x, y, x-20, y])
        selected, frames, report = prune_training_support(rows, np.tile(np.eye(3), (6, 1, 1)), K)
        self.assertEqual(frames, [3, 4, 5])
        self.assertEqual(len(selected), 120)
        self.assertTrue(report['support_pruning']['eligible'])
        self.assertTrue(np.all(selected[:, 1] >= 100))

    def test_empty_weak_and_invalid_settings(self):
        rows = np.empty((0, 6)); rotations = np.tile(np.eye(3), (3, 1, 1))
        pruned, frames, report = prune_training_support(rows, rotations, K)
        self.assertEqual(pruned.shape, (0, 6)); self.assertEqual(frames, [])
        self.assertFalse(report['support_pruning']['eligible'])
        for kwargs in ({'minimum_frame_tracks': True}, {'minimum_track_frames': 2}, {'minimum_frames': 0}):
            with self.assertRaises(ValueError):
                prune_training_support(rows, rotations, K, **kwargs)


if __name__ == '__main__':
    unittest.main()
