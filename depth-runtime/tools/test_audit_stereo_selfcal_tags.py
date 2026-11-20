#!/usr/bin/env python3
"""Independent synthetic pose and provenance checks for the tag audit."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    from audit_stereo_selfcal_tags import K, audit, sha


@unittest.skipIf(cv2 is None, 'OpenCV required')
class TagAuditTests(unittest.TestCase):
    def test_two_pose_branches_scale_invariance_and_identity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            size, baseline = 0.12, 0.065
            h = size / 2
            square = np.array([[-h, -h, 0.], [h, -h, 0.], [h, h, 0.], [-h, h, 0.]])
            rot, _ = cv2.Rodrigues(np.array([0.25, -0.2, 0.05]))
            points = square @ rot.T + [0.08, -0.06, 1.4]
            def project(p):
                q = p @ K.T
                return (q[:, :2] / q[:, 2, None]).tolist()
            observations = [{'ordinal': 1, 'tag': 7, 'left': project(points),
                             'right': project(points - [baseline, 0., 0.])}]
            paths = {name: root / name for name in ('obs', 'calib', 'rectifier', 'manifest', 'result')}
            paths['obs'].write_text(json.dumps(observations))
            paths['calib'].write_text('calibration')
            paths['rectifier'].write_text('snapshot')
            manifest = {'observations': str(paths['obs']), 'calibration': str(paths['calib']),
                        'rectifier_source': str(paths['rectifier']), 'tag_size_m': size,
                        'frames': [{'ordinal': 1, 'split': 'heldout'}]}
            paths['manifest'].write_text(json.dumps(manifest))
            result = {'manifest_sha256': sha(paths['manifest']), 'observations_sha256': sha(paths['obs']),
                      'calibration_sha256': sha(paths['calib']), 'rectifier_sha256': sha(paths['rectifier']),
                      'baseline_m': baseline, 'models': {}}
            paths['result'].write_text(json.dumps(result))
            data = audit(paths['manifest'], paths['result'])
            records = data['records']
            self.assertEqual({r['branch'] for r in records}, {0, 1})
            pose = next(r for r in records if r['branch'] == 0 and r['baseline_factor'] == r['size_factor'] == 1.)
            self.assertLess(pose['left_rms_px'], 1e-6)
            np.testing.assert_allclose(pose['right_mean_error_px'], 0., atol=1e-6)
            for branch in (0, 1):
                equal = [r for r in records if r['branch'] == branch and r['baseline_factor'] == r['size_factor']]
                for r in equal[1:]:
                    np.testing.assert_allclose(r['right_mean_error_px'], equal[0]['right_mean_error_px'], atol=1e-9)
            for name in ('obs', 'calib', 'rectifier', 'manifest'):
                original = paths[name].read_text()
                paths[name].write_text(original + ' ')
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    audit(paths['manifest'], paths['result'])
                paths[name].write_text(original)


if __name__ == '__main__':
    unittest.main()
