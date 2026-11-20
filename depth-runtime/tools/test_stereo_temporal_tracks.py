#!/usr/bin/env python3
"""Independent correspondence graph checks for temporal track extraction."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evaluate_stereo_temporal_calibration import build_tracks, heldout_left_prediction, select_external_tags, evaluate, K
from test_stereo_temporal_calibration import rot


class TemporalTrackTests(unittest.TestCase):
    def test_two_eye_agreement_and_consecutive_association(self):
        # Three stereo features; track 0 persists, track 1 ends after frame 1,
        # and mismatched right association prevents linking feature 2 at all.
        first = np.array([[50., 50., 30., 50.], [90., 60., 70., 60.],
                          [130., 80., 110., 80.]])
        second = first + [3., 1., 3., 1.]
        third = first + [6., 2., 6., 2.]
        left01 = np.column_stack((first[:, :2], second[:, :2]))
        right01 = np.column_stack((first[:, 2:], second[:, 2:]))
        # Deliberately send third feature's right temporal observation to the
        # second feature, contradicting its left link.
        right01[2, 2:] = second[1, 2:]
        left12 = np.column_stack((second[:1, :2], third[:1, :2]))
        right12 = np.column_stack((second[:1, 2:], third[:1, 2:]))
        rows = build_tracks([first, second, third], [left01, left12], [right01, right12])
        tracks = {int(i): rows[rows[:, 1] == i] for i in np.unique(rows[:, 1])}
        # The ambiguous endpoint may also eliminate feature 1 via mutual
        # uniqueness; the feature 0 chain must remain and feature 2 must not.
        self.assertTrue(any(np.array_equal(a[:, 0], [0., 1., 2.]) for a in tracks.values()))
        self.assertFalse(any(np.any(a[:, 2] == 130.) for a in tracks.values()))
        for a in tracks.values():
            self.assertEqual(len(a), len(np.unique(a[:, 0])))

    def test_heldout_scoring_uses_only_left_to_triangulate(self):
        baseline = 0.105
        relative = np.array([0.004, -0.008, 0.003])
        focal_scale = 1.008
        candidate = {'parameters': {'rotation_vector_rad': relative.tolist(), 'focal_scale': focal_scale}}
        world = np.array([[0.1, -0.1, 1.4], [-0.3, 0.2, 2.0], [0.4, 0.3, 2.5]])
        poses, rows = [], []
        for frame in range(4):
            v = np.array([.004, -.01, .002]) * frame
            center = np.array([.09, .01, .005]) * frame
            poses.append({'frame_id': frame, 'rotation_vector_rad': v.tolist(), 'translation_m': center.tolist()})
            left = (world - center) @ rot(v)
            right = (left - [baseline, 0., 0.]) @ rot(relative)
            for track, (l, r) in enumerate(zip(left, right)):
                uv_l = (K @ l)[:2] / l[2]
                uv_r = focal_scale * (K[:2, :2] @ r[:2] / r[2]) + K[:2, 2]
                rows.append([frame, track, *uv_l, *uv_r])
        rows = np.asarray(rows)
        ba = {'accepted': True, 'poses': poses}
        initial = heldout_left_prediction(rows, ba, {'model': candidate}, baseline)
        self.assertEqual(initial['available_tracks'], 3)
        self.assertLess(initial['scores']['model']['x_p90_abs_px'], 1e-7)
        self.assertLess(initial['scores']['model']['y_p90_abs_px'], 1e-7)
        shifted = rows.copy(); shifted[:, 4] += 3.
        altered = heldout_left_prediction(shifted, ba, {'model': candidate}, baseline)
        self.assertEqual(initial['available_tracks'], altered['available_tracks'])
        self.assertEqual(initial['skipped_tracks'], altered['skipped_tracks'])
        self.assertAlmostEqual(altered['scores']['model']['x_median_px'] - initial['scores']['model']['x_median_px'], -3., delta=1e-9)
        rejected = heldout_left_prediction(rows, {'accepted': False, 'poses': poses}, {'model': candidate}, baseline)
        self.assertEqual(rejected['available_tracks'], 0)
        self.assertEqual(rejected['skipped_tracks']['rejected_or_missing_poses'], 3)
        missing = heldout_left_prediction(rows, {'accepted': True, 'poses': []}, {'model': candidate}, baseline)
        self.assertEqual(missing['available_tracks'], 0)
        self.assertEqual(missing['skipped_tracks']['rejected_or_missing_poses'], 3)
        stationary = {'accepted': True, 'poses': [dict(p, rotation_vector_rad=[0., 0., 0.], translation_m=[0., 0., 0.]) for p in poses]}
        repeated = rows.copy()
        for track in range(3):
            repeated[repeated[:, 1] == track, 2:4] = repeated[repeated[:, 1] == track, 2:4][0]
        weak = heldout_left_prediction(repeated, stationary, {'model': candidate}, baseline)
        self.assertEqual(weak['available_tracks'], 0)
        self.assertEqual(weak['skipped_tracks']['weak_parallax'], 3)

    def test_external_tags_map_source_pairs_not_dense_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairfile, obsfile, manifest = root / 'pairs', root / 'obs', root / 'manifest'
            pairfile.write_text('\n'.join(json.dumps({'left_file': f'{n}.raw', 'right_file': f'{n}r.raw', 'pair': n}) for n in (500, 600, 700)))
            observations = [{'ordinal': i, 'tag': 4} for i in (1, 2, 4)]
            obsfile.write_text(json.dumps(observations))
            manifest.write_text(json.dumps({'pairs': str(pairfile), 'observations': str(obsfile),
                'frames': [{'ordinal': 1, 'split': 'heldout', 'left': '/x/500.raw', 'right': '/x/500r.raw'},
                           {'ordinal': 2, 'split': 'heldout', 'left': '/x/600.raw', 'right': '/x/600r.raw'},
                           {'ordinal': 4, 'split': 'train', 'left': '/x/700.raw', 'right': '/x/700r.raw'}]}))
            # Dense ordinal 1 collides with the external ordinal 1, but its
            # source pair is 500; only actual pair 600 is excluded.
            selected, provenance = select_external_tags(manifest, {1, 600})
            self.assertEqual(selected, [observations[0]])
            self.assertIn('manifest_sha256', provenance)
            malformed = json.loads(manifest.read_text())
            malformed['frames'][0]['right'] = '/x/600r.raw'
            manifest.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                select_external_tags(manifest, {600})

    def test_window_length_boundaries_and_default(self):
        self.assertEqual(inspect.signature(evaluate).parameters['window_length'].default, 5)
        for length in (1, 13):
            with self.assertRaises(ValueError):
                evaluate('unused', 'unused', [], window_length=length)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, f, o, p = (root / n for n in ('manifest', 'frozen', 'obs', 'pairs'))
            o.write_text('[]'); f.write_text('{}')
            p.write_text(json.dumps({'left_file': 'left.raw', 'left_raw_ns': 0}) + '\n')
            m.write_text(json.dumps({'observations': str(o), 'pairs': str(p),
                                     'frames': [{'ordinal': 0, 'left': '/x/left.raw'}]}))
            observed = {'baseline_m': .065, 'source_sha256': 'x', 'rectifier_sha256': 'x',
                        'calibration_sha256': 'x', 'observations_sha256': 'x',
                        'pairs_sha256': 'x', 'input_sha256': {}}
            with patch('evaluate_stereo_temporal_calibration.extract', return_value=({}, observed)):
                for length in (2, 5, 12):
                    output = evaluate(m, f, [0], window_length=length)
                    self.assertEqual(output['window_length'], length)
                    self.assertEqual(output['windows'][0]['error'], 'missing frame(s)')

    def test_empty_and_bad_temporal_counts(self):
        empty = np.empty((0, 4))
        self.assertEqual(build_tracks([empty, empty], [empty], [empty]).shape, (0, 6))
        with self.assertRaises(ValueError):
            build_tracks([empty, empty], [], [empty])


if __name__ == '__main__':
    unittest.main()
