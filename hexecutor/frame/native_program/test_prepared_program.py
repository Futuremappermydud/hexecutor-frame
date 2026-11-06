#!/usr/bin/env python3
"""Offline NPRG format and QDRX-equivalence checks."""
import sys
import copy
import struct
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepared_program as pp

ROOT = Path(__file__).resolve().parents[1]


class PreparedProgramTests(unittest.TestCase):
    def test_v1_roundtrip_but_fails_stack_aware_l2_plan(self):
        for model in ('primary', 'alternate'):
            source = (ROOT / 'build' / f'resident-{model}.qdrx').read_bytes()
            prepared = pp.lower(source)
            result = pp.verify_equivalence(source, prepared)
            decoded = pp.decode(prepared)
            self.assertEqual(result['commands'], 5068 if model == 'primary' else 5040)
            self.assertEqual(decoded['source_sha256'], __import__('hashlib').sha256(source).hexdigest())
            self.assertGreater(len(prepared), pp.L2_CEILING)

    def test_crc_and_source_binding_reject_mutations(self):
        source = (ROOT / 'build' / 'resident-primary.qdrx').read_bytes()
        prepared = bytearray(pp.lower(source))
        prepared[-1] ^= 0x40
        with self.assertRaisesRegex(ValueError, 'CRC'):
            pp.decode(bytes(prepared))
        prepared = pp.lower(source)
        other = (ROOT / 'build' / 'resident-alternate.qdrx').read_bytes()
        with self.assertRaisesRegex(ValueError, 'source QDRX binding'):
            pp.verify_equivalence(other, prepared)

    def test_native_preflight_fails_closed(self):
        import compact_v2 as v2
        source = (ROOT / 'build' / 'resident-primary.qdrx').read_bytes()
        decoded = v2.decode(v2.lower(source))
        with self.assertRaisesRegex(ValueError, 'no per-command kernel resolvers'):
            pp.preflight_native_program(decoded, {})

    def test_same_opcode_requires_exact_convolution_features(self):
        import compact_v2 as v2
        source = (ROOT / 'build' / 'resident-primary.qdrx').read_bytes()
        decoded = v2.decode(v2.lower(source))
        conv_opcode = 6  # k_hmx_conv in the QDRX ABI
        index = next(i for i, cmd in enumerate(decoded['commands']) if cmd[1] == conv_opcode)
        fixture = copy.copy(decoded)
        fixture['commands'] = [decoded['commands'][index]]
        features = pp.command_features(fixture, 0)
        # Test-only allowlist entry: this proves that the gate distinguishes
        # command feature tuples. It is not a claim of numeric kernel proof.
        resolver = pp.exact_feature_capability({pp.feature_fingerprint(features)}, scratch_bytes=4096)
        accepted = pp.preflight_native_program(fixture, {conv_opcode: resolver}, scratch_budget=8192)
        self.assertEqual(accepted['peak_scratch_bytes'], 4096)

        changed = copy.deepcopy(fixture)
        activation_index = fixture['commands'][0][4][0]
        dims = changed['tensor_features'][activation_index]['shape']['dimensions']
        changed['tensor_features'][activation_index]['shape']['dimensions'] = (dims[0]+1,) + dims[1:]
        changed_features = pp.command_features(changed, 0)
        self.assertEqual(changed_features['opcode'], features['opcode'])
        self.assertNotEqual(changed_features['inputs'][0]['shape'], features['inputs'][0]['shape'])
        with self.assertRaisesRegex(ValueError, 'no numerical proof'):
            pp.preflight_native_program(changed, {conv_opcode: resolver}, scratch_budget=8192)

        # A proof that still permits legacy VTCM fallback is not native-ready.
        fallback = lambda _: pp.Capability(True, scratch_bytes=1, may_fallback=True)
        with self.assertRaisesRegex(ValueError, 'legacy fallback'):
            pp.preflight_native_program(fixture, {conv_opcode: fallback}, scratch_budget=8192)
        with self.assertRaisesRegex(ValueError, 'budget'):
            pp.preflight_native_program(fixture,
                {conv_opcode: pp.exact_feature_capability({pp.feature_fingerprint(features)}, scratch_bytes=9000)},
                scratch_budget=8192)


if __name__ == '__main__':
    unittest.main()
