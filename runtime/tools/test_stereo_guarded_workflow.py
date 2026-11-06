#!/usr/bin/env python3

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str((PATHS.repo / 'runtime/scripts')))
from stereo_guarded_common import jobs,verify_complete
from stereo_composite_common import DEVELOPMENT_SEEDS,EVALUATION_SEEDS,digest,write_json


class WorkflowTests(unittest.TestCase):
    def test_frozen_grid_and_seeds(self):
        plan=jobs(Path('/binary'),Path('/inputs'),Path('/output'))
        self.assertEqual(len(plan),104)
        self.assertEqual(len({j['name'] for j in plan}),104)
        self.assertEqual({j['guard'] for j in plan},set(range(4)))
        self.assertEqual({j['exposure'] for j in plan},{False,True})
        seeds={int(j['case'].split('-')[1]) for j in plan if j['case'].startswith('synthetic-')}
        self.assertEqual(seeds,set(DEVELOPMENT_SEEDS))
        self.assertFalse(seeds & set(EVALUATION_SEEDS))
        for job in plan:
            self.assertEqual('--support-exposure' in job['command'],job['exposure'])
            self.assertEqual('--metrics-only' not in job['command'],job['case'] in ['synthetic-1709','frame-70'])

    def fixture(self,root):
        job=dict(name='test')
        (root/'metrics.csv').write_text('original\n')
        write_json(root/'complete.json',dict(job=job,output_sha256={'metrics.csv':digest(root/'metrics.csv')}))
        return job

    def test_verified_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            self.assertEqual(verify_complete(root,job)['job'],job)

    def test_changed_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            (root/'metrics.csv').write_text('tampered\n')
            with self.assertRaisesRegex(ValueError,'Output changed'):verify_complete(root,job)

    def test_changed_inventory_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=self.fixture(root)
            (root/'unexpected.csv').write_text('extra')
            with self.assertRaisesRegex(ValueError,'inventory changed'):verify_complete(root,job)

    def test_job_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            with self.assertRaisesRegex(ValueError,'Completion mismatch'):verify_complete(root,dict(name='other'))


if __name__=='__main__':unittest.main()
