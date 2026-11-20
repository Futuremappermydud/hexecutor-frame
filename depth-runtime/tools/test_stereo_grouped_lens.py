import unittest
import numpy as np
from pathlib import Path
from evaluate_stereo_grouped_lens import probe_stability, combine_windows, stereo_frame_identities
from stereo_lens_calibration import CameraRayModel
from test_stereo_lens_calibration import K, LENS
from test_stereo_temporal_calibration import rot


class GroupedLensTests(unittest.TestCase):
    def test_raw_stereo_frame_identity_uses_hashes_not_renamed_paths(self):
        frames = [{'left': 'firstL', 'right': 'firstR'}, {'left': 'renamedL', 'right': 'renamedR'}]
        hashes = {str(Path(name).resolve()): digest for name, digest in
                  [('firstL', 'Lhash'), ('firstR', 'Rhash'), ('renamedL', 'Lhash'), ('renamedR', 'Rhash')]}
        with self.assertRaises(ValueError):
            stereo_frame_identities({'frames': frames}, {'input_sha256': hashes})
        hashes[str(Path('renamedR').resolve())] = 'differentRhash'
        self.assertEqual(len(stereo_frame_identities({'frames': frames}, {'input_sha256': hashes})), 2)

    def test_probe_guard_checks_left_lens_as_well_as_right(self):
        cameras = [CameraRayModel(K, rot([.05, .3, .02]), LENS), CameraRayModel(K, rot([-.04, -.3, .03]), LENS)]
        fit = {'accepted': True, 'parameters_vector': [0.]*6}
        self.assertTrue(probe_stability([fit, fit, fit], *cameras, .105)['stability_passed'])
        left_change = {'accepted': True, 'parameters_vector': [0., 0., 0., 0., .01, 0.]}
        report = probe_stability([fit, fit, left_change], *cameras, .105)
        self.assertFalse(report['stability_passed'])
        self.assertEqual(report['max_horizontal_spread_px']['right'], 0.)
        self.assertGreater(report['max_horizontal_spread_px']['left'], .3)
        self.assertFalse(probe_stability([fit, fit], *cameras, .105)['stability_passed'])
        self.assertFalse(probe_stability([fit, fit, dict(fit, accepted=False)], *cameras, .105)['stability_passed'])

    def test_independent_ids_and_nonconsecutive_window_indices(self):
        rows = np.array([[0, 3, 100, 100, 80, 100], [1, 3, 102, 100, 82, 100]], float)
        prepared = [(None, rows, None, np.tile(np.eye(3), (2, 1, 1)), None, None, None)]*3
        combined, rotations, groups = combine_windows(prepared, [0, 2])
        np.testing.assert_array_equal(combined[:, 0], [0, 1, 200, 201])
        np.testing.assert_array_equal(combined[:, 1], [3, 3, 2000003, 2000003])
        self.assertEqual(groups, [0, 0, 2, 2])
        self.assertEqual(rotations.shape, (4, 3, 3))
        np.testing.assert_array_equal(rows[:, 0], [0, 1])


if __name__ == '__main__':
    unittest.main()
