#!/usr/bin/env python3
"""Decision and split-isolation regression tests for the offline workflow."""

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

import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str((PATHS.repo / 'depth-runtime/scripts')))
from analyze_stereo_composite_experiment import FIELDS, gate, select_development
from run_stereo_composite_experiment import make_plan
from stereo_composite_common import (AMBIGUOUS, ARMS, DEVELOPMENT_SEEDS, EVALUATION_SEEDS,
    NOISY, REGRESSION_SEEDS, SCENES, config, config_id, identity, load_phase, pooled, write_json)


class CompositeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "binary"
        self.binary.write_bytes(b"test-binary")

    def tearDown(self):
        self.temporary.cleanup()

    def fixtures(self):
        reference = {s: dict({field: 1. for field in FIELDS}, supported_pct=50.) for s in SCENES}
        for scene in AMBIGUOUS:
            reference[scene]["supported_pct"] = 0.
        reference["unrelated-pair"]["supported_pct"] = 10.
        common = {s: dict(common_reference_mae_px=.2, common_candidate_mae_px=.1) for s in SCENES}
        return reference, common

    def test_evaluation_is_separate_from_every_development_and_regression_seed(self):
        self.assertFalse(set(EVALUATION_SEEDS) & set(DEVELOPMENT_SEEDS + REGRESSION_SEEDS))
        plan = make_plan("smoke", self.root, None, self.binary)
        self.assertTrue(all(j["case"] == "synthetic-1709" for j in plan["jobs"]))

    def test_missing_or_changed_freeze_never_produces_evaluation_plan(self):
        with self.assertRaises(FileNotFoundError):
            make_plan("evaluation", self.root, None, self.binary)
        write_json(self.root / "frozen.json", dict(identity={}, finalists=[]))
        with self.assertRaisesRegex(ValueError, "different source/binary"):
            make_plan("evaluation", self.root, None, self.binary)
        self.assertFalse((self.root / "evaluation").exists())

    def test_partial_job_is_rejected_before_analysis(self):
        folder = self.root / "smoke"
        folder.mkdir()
        write_json(folder / "plan.json", dict(jobs=[dict(id="unfinished")]))
        with self.assertRaises(FileNotFoundError):
            load_phase(self.root, "smoke")

    def test_pixel_pooling_excludes_undefined_accuracy_without_zeroing_it(self):
        self.assertAlmostEqual(pooled([dict(v="1", n="10"), dict(v="3", n="30"),
                                       dict(v="nan", n="0")], "v", "n"), 2.5)
        self.assertIsNone(pooled([dict(v="nan", n="0")], "v", "n"))

    def test_empty_common_mask_and_rejected_all_depth_do_not_pass(self):
        reference, common = self.fixtures()
        candidate = copy.deepcopy(reference)
        candidate[NOISY[0]]["accepted_bad_depth_pct"] = None
        candidate[NOISY[0]]["missing_gt_pct"] = 100.
        common[NOISY[0]]["common_candidate_mae_px"] = None
        reasons = gate(candidate, reference, common)
        self.assertTrue(any("accepted_bad_depth_pct" in r for r in reasons))
        self.assertTrue(any("missing_gt_pct" in r for r in reasons))
        self.assertTrue(any("no common-mask MAE gain" in r for r in reasons))

    def test_false_support_and_boundary_leakage_override_precision_gain(self):
        reference, common = self.fixtures()
        candidate = copy.deepcopy(reference)
        candidate["unrelated-pair"]["supported_pct"] = 11.
        candidate["thin-silhouette"]["background_boundary_leakage_of_all_pct"] = 2.
        candidate["blank-ambiguous"]["supported_pct"] = .01
        reasons = gate(candidate, reference, common)
        self.assertTrue(any("false support" in r for r in reasons))
        self.assertTrue(any("background_boundary" in r for r in reasons))
        self.assertTrue(any("ambiguity support" in r for r in reasons))

    def test_failed_candidates_are_not_selected_because_mae_is_lower(self):
        reference, common = self.fixtures()
        arms = {arm: dict(passed=True, scenes=reference, common=common) for arm in ARMS}
        base = dict(config=config(), arms=arms)
        candidate = dict(config=config("composite-raw", .25), arms=copy.deepcopy(arms))
        candidate["arms"]["integer"]["passed"] = False
        result = select_development({config_id(base["config"]): base,
                                     config_id(candidate["config"]): candidate}, dict(identity=identity(self.binary)))
        self.assertEqual(result["finalists"], [])
        self.assertEqual(result["status"], "failed-quality-gates")


if __name__ == "__main__":
    unittest.main()
