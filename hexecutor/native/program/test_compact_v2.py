#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""NPRG2 geometry dedup and block delta-stream equivalence tests."""

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

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str((PATHS.native / 'program')))
import compact_v2 as v2
import prepared_program as pp

ROOT = (PATHS.native)
L2_PLAN = 768 * 1024  # 1 MiB less three 64 KiB stacks and 64 KiB state reserve


class CompactV2Tests(unittest.TestCase):
    def test_both_models_roundtrip_and_fit_planning_ceiling(self):
        for model, command_count in (('primary', 5068), ('alternate', 5040)):
            source = ((PATHS.native / f'build/resident-{model}.qdrx')).read_bytes()
            prepared = v2.lower(source)
            result = v2.verify_equivalence(source, prepared)
            decoded = v2.decode(prepared)
            self.assertEqual(result['commands'], command_count)
            self.assertLessEqual(len(prepared), L2_PLAN)
            self.assertFalse(hashlib.sha256(source).hexdigest() != decoded['source_sha256'])
            self.assertLess(result['geometries'], result['tensors'] // 7)

    def test_crc_mutation_rejected(self):
        source = ((PATHS.native / 'build/resident-primary.qdrx')).read_bytes()
        prepared = bytearray(v2.lower(source)); prepared[-1] ^= 0x20
        with self.assertRaisesRegex(ValueError, 'CRC'):
            v2.decode(bytes(prepared))

    def test_approved_conv_features_are_shape_specific(self):
        source = ((PATHS.native / 'build/resident-primary.qdrx')).read_bytes()
        program = v2.decode(v2.lower(source))
        index = next(i for i,c in enumerate(program['commands']) if c[1] == 6)
        one = dict(program, commands=[program['commands'][index]])
        features = pp.command_features(one, 0)
        resolver = pp.exact_feature_capability({pp.feature_fingerprint(features)}, scratch_bytes=1024)
        pp.preflight_native_program(one, {6: resolver}, scratch_budget=16*1024)
        import copy
        changed = copy.deepcopy(one)
        ti = changed['commands'][0][4][0]
        dims = changed['tensor_features'][ti]['shape']['dimensions']
        changed['tensor_features'][ti]['shape']['dimensions'] = (dims[0]+1,) + dims[1:]
        with self.assertRaisesRegex(ValueError, 'no numerical proof'):
            pp.preflight_native_program(changed, {6: resolver}, scratch_budget=16*1024)


if __name__ == '__main__':
    unittest.main()
