import unittest
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
from stereo_pose_constraints import rectified_in_head, recorded_motion
from test_stereo_temporal_calibration import rot


class PoseConstraintBasisTests(unittest.TestCase):
    def test_recorded_motion_clock_basis_and_lever_transform(self):
        def axes(r, position):
            return {'plus_x': r[:, 0].tolist(), 'plus_z': r[:, 2].tolist(), 'position': position, '__units': 'mm'}
        calibration = {'cameras': [
            {'sourceCamera': 'slam_right', 'extrinsics': axes(np.eye(3), [0, 0, 0])},
            {'sourceCamera': 'slam_left', 'extrinsics': axes(np.eye(3), [-100, 0, 0])}]}
        extrinsic = rot([.2, -.3, .1])
        config = {'cv': {'cad_from_cal': axes(extrinsic, [.02, .03, .04])}, 'head': axes(np.eye(3), [0, 0, 0])}
        head_rotations = [rot([.1, .2, -.3]), rot([.2, .3, -.2])]
        head_centers = [[.2, .3, .4], [.3, .4, .5]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('cal', calibration), ('config', config)]:
                (root/name).write_text(json.dumps(value))
            pairs = [{'left_file': f'{i}L', 'right_file': f'{i}R', 'left_monotonic_ns': (i+1)*1000000+100000} for i in range(2)]
            (root/'pairs').write_text('\n'.join(json.dumps(p) for p in pairs))
            (root/'manifest').write_text(json.dumps({'calibration': str(root/'cal'), 'pairs': str(root/'pairs'),
                'frames': [{'ordinal': i, 'left': f'{i}L', 'right': f'{i}R'} for i in range(2)]}))
            with (root/'poses').open('w', newline='') as stream:
                fields = ['mono_ns', 'valid', 'result'] + [f'm{r}{c}' for r in range(3) for c in range(4)]
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                for i, (rotation, center) in enumerate(zip(head_rotations, head_centers)):
                    pose = np.column_stack((rotation, center))
                    row = {'mono_ns': (i+1)*1000000, 'valid': '1', 'result': '200'}
                    row.update({f'm{r}{c}': pose[r, c] for r in range(3) for c in range(4)})
                    writer.writerow(row)
            rotations, translations, levers, provenance = recorded_motion(root/'manifest', root/'poses', root/'config', .1)
            origin = extrinsic @ [-.1, 0, 0] + [.02, .03, .04]
            for i in range(2):
                np.testing.assert_allclose(rotations[i], head_rotations[i] @ extrinsic, atol=1e-12)
                np.testing.assert_allclose(translations[i], head_centers[i], atol=1e-12)
                np.testing.assert_allclose(levers[i], head_rotations[i] @ origin, atol=1e-12)
            self.assertTrue(all(a['gap_ns'] == 0 for a in provenance['association']))

    def test_known_nontrivial_calibration_and_head_axes(self):
        def axes(r, position):
            return {'plus_x': r[:, 0].tolist(), 'plus_z': r[:, 2].tolist(), 'position': position}
        # Baseline x, optical z, identity left/right axes => identity rectified basis.
        calibration = {'cameras': [
            {'sourceCamera': 'slam_right', 'extrinsics': axes(np.eye(3), [0, 0, 0])},
            {'sourceCamera': 'slam_left', 'extrinsics': axes(np.eye(3), [-100, 0, 0])}]}
        cad_from_cal, cad_from_head = rot([.2, .1, -.3]), rot([-.2, .4, .05])
        config = {'cv': {'cad_from_cal': axes(cad_from_cal, [0, 0, 0])}, 'head': axes(cad_from_head, [0, 0, 0])}
        expected = cad_from_head.T @ cad_from_cal
        np.testing.assert_allclose(rectified_in_head(calibration, config), expected, atol=1e-12)
        # A nontrivial relative rig rotation changes the rectified basis while
        # leaving the head mapping unchanged. Construct its bisector independently.
        left = rot([.1, 0., 0.])
        calibration['cameras'][1]['extrinsics'] = axes(left, [-100, 0, 0])
        z = left[:, 2] + [0., 0., 1.]; z[0] = 0; z /= np.linalg.norm(z)
        rect = np.column_stack(([1., 0., 0.], np.cross(z, [1., 0., 0.]), z))
        np.testing.assert_allclose(rectified_in_head(calibration, config), expected @ rect, atol=1e-12)
        calibration['cameras'][0]['sourceCamera'] = 'unknown'
        with self.assertRaises(ValueError):
            rectified_in_head(calibration, config)


if __name__ == '__main__':
    unittest.main()
