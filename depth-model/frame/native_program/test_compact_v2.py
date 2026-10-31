#!/usr/bin/env python3
"""NPRG2 geometry dedup and block delta-stream equivalence tests."""
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact_v2 as v2
import prepared_program as pp

ROOT = Path(__file__).resolve().parents[1]
L2_PLAN = 768 * 1024  # 1 MiB less three 64 KiB stacks and 64 KiB state reserve


class CompactV2Tests(unittest.TestCase):
    def test_both_models_roundtrip_and_fit_planning_ceiling(self):
        for model, command_count in (('primary', 5068), ('alternate', 5040)):
            source = (ROOT / 'build' / f'resident-{model}.qdrx').read_bytes()
            prepared = v2.lower(source)
            result = v2.verify_equivalence(source, prepared)
            decoded = v2.decode(prepared)
            self.assertEqual(result['commands'], command_count)
            self.assertLessEqual(len(prepared), L2_PLAN)
            self.assertFalse(hashlib.sha256(source).hexdigest() != decoded['source_sha256'])
            self.assertLess(result['geometries'], result['tensors'] // 7)

    def test_crc_mutation_rejected(self):
        source = (ROOT / 'build' / 'resident-primary.qdrx').read_bytes()
        prepared = bytearray(v2.lower(source)); prepared[-1] ^= 0x20
        with self.assertRaisesRegex(ValueError, 'CRC'):
            v2.decode(bytes(prepared))

    def test_approved_conv_features_are_shape_specific(self):
        source = (ROOT / 'build' / 'resident-primary.qdrx').read_bytes()
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
