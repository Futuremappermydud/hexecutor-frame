#!/usr/bin/env python3
"""Independent aggregate evidence, omission, and identity checks."""
import json
from pathlib import Path
import tempfile
import unittest

import prepare_dense_stereo_replay as dense


def write(path, obj):
    path.write_text(json.dumps(obj))


def setup(root):
    plan = {'windows': [{'anchor_v2_ordinal': a, 'frames': [{'source_pair_id': a * 100 + i}
                                                              for i in range(12)]}
                         for a in (0, 10, 30)],
            'sources': {'frozen_manifest': {'sha256': 'original-manifest-hash'}, 'observations': {'sha256': 'obs-hash'}, 'pair_metadata': {'sha256': 'pairs-hash'}}}
    plan_path = root / 'plan.json'; write(plan_path, plan)
    out = root / 'materialized'; out.mkdir()
    results = []
    replay = {}
    for key in ('source_sha256', 'temporal_sha256', 'estimator_sha256', 'evaluator_sha256', 'validation_sha256'):
        source = root / key; source.write_text(key)
        replay[key] = {'path': str(source), 'sha256': dense.sha(source)}
    for w in plan['windows']:
        a = w['anchor_v2_ordinal']
        m = out / f'window-{a}.json'
        write(m, {'source_plan_sha256': dense.sha(plan_path), 'frames': w['frames'], 'replay_sources': replay})
        baseline = out / f'window-{a}-baseline.json'; write(baseline, {'anchor': a})
        result = {'manifest_sha256': dense.sha(m), 'v2_result_sha256': dense.sha(baseline),
                  **{key: record['sha256'] for key, record in replay.items()}, 'baseline_m': .065, 'window_length': 12,
                  'windows': [{'start': 0, 'external_tag_provenance': {'manifest_sha256': 'original-manifest-hash', 'observations_sha256': 'obs-hash', 'pairs_sha256': 'pairs-hash'},
                               'temporal_relative4': {'fit': {'accepted': True, 'parameters': {
                                   'rotation_vector_rad': [0., 0., 0.], 'focal_scale': 1.}}}}]}
        p = root / f'result-{a}.json'; write(p, result); results.append(p)
    return plan_path, results


class AggregateTests(unittest.TestCase):
    def test_missing_and_rejected_cannot_pass_session(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, results = setup(Path(temp))
            complete = dense.aggregate(plan, results)
            self.assertTrue(complete['session_stability_passed'])
            self.assertEqual(complete['metric_acceptance'], 'unevaluated')
            missing = dense.aggregate(plan, results[:2])
            self.assertFalse(missing['session_stability_passed'])
            self.assertEqual(missing['missing_result_anchors'], [30])
            r = json.loads(results[2].read_text())
            r['windows'][0]['temporal_relative4']['fit']['accepted'] = False
            write(results[2], r)
            rejected = dense.aggregate(plan, results)
            self.assertFalse(rejected['session_stability_passed'])
            self.assertEqual(rejected['failed_or_rejected_anchors'], [30])

    def test_wrong_anchor_duplicate_baseline_and_plan_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan, results = setup(root)
            with self.assertRaises(ValueError):
                dense.aggregate(plan, [results[0], results[0]])
            original = results[0].read_text()
            value = json.loads(original); value['manifest_sha256'] = 'unknown'
            write(results[0], value)
            with self.assertRaises(ValueError): dense.aggregate(plan, results)
            results[0].write_text(original)
            value = json.loads(original); del value['windows'][0]['external_tag_provenance']
            write(results[0], value)
            with self.assertRaisesRegex(ValueError, 'provenance missing'): dense.aggregate(plan, results)
            results[0].write_text(original)
            source = root / 'validation_sha256'; old_source = source.read_text(); source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'replay source mismatch'): dense.aggregate(plan, results)
            source.write_text(old_source)
            (root/'materialized'/'window-0-baseline.json').write_text('changed')
            with self.assertRaises(ValueError): dense.aggregate(plan, results)
            # Fresh fixture verifies plan mutation is also detected.
        with tempfile.TemporaryDirectory() as temp:
            plan, results = setup(Path(temp))
            plan.write_text(plan.read_text() + ' ')
            with self.assertRaises(ValueError): dense.aggregate(plan, results)


if __name__ == '__main__':
    unittest.main()
