
# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str((PATHS.repo / 'depth-runtime/scripts')))
from room_geometry_eval import evaluate


class GeometryEvaluatorTest(unittest.TestCase):
    def test_injected_plane_shift(self):
        reference = np.full((32, 32), 2.0, np.float32)
        result = evaluate(reference, reference + 0.04)
        self.assertAlmostEqual(result['mean_abs_z_m'], 0.04, places=5)
        self.assertAlmostEqual(result['signed_z_bias_m'], 0.04, places=5)

    def test_closed_hole(self):
        reference = np.full((32, 32), 2.0, np.float32)
        reference[10:20, 10:20] = 0
        candidate = reference.copy()
        candidate[10:20, 10:20] = 2.0
        result = evaluate(reference, candidate)
        self.assertGreater(result['false_fill_fraction'], 0.99)

    def test_rounded_corner_and_boundary_shift(self):
        reference = np.full((32, 32), 2.0, np.float32)
        reference[:, 16:] = 3.0
        candidate = np.full((32, 32), 2.0, np.float32)
        candidate[:, 18:] = 3.0
        result = evaluate(reference, candidate)
        self.assertGreater(result['boundary_displacement_px'], 1.0)
        rounded = reference.copy()
        rounded[:, 15:18] = np.array([2.25, 2.5, 2.75])
        self.assertGreater(evaluate(reference, rounded)['mean_abs_z_m'], 0.01)


if __name__ == '__main__':
    unittest.main()
