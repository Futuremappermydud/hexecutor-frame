#!/usr/bin/env python3
"""IR integrity, fail-closed lowering and real recovered descriptor checks."""

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

import copy
from pathlib import Path
import sys
import os
import unittest
sys.path.insert(0, str((PATHS.native)))
import depth_model.lowering.resident_program as rp
class ResidentProgramTests(unittest.TestCase):
    def test_codec_preserves_wide_arguments_and_float_bits(self):
        value = {'args': [None, False, True, -(1 << 63), (1 << 64)-1, -0.0, b'\0\xff', 'tensor']}
        raw = rp.encode(value)
        self.assertEqual(raw, rp.encode(rp.decode(raw)))
        for i in range(len(raw)):
            with self.assertRaises(ValueError): rp.decode(raw[:i])

    def test_codec_rejects_invalid_data(self):
        for raw in (b'x', b'l\xff\xff\xff\xff', b'ntrailing', b'i\x02'+bytes(8)):
            with self.assertRaises(ValueError): rp.decode(raw)
        for value in (1 << 64, float('nan'), {1: 'bad key'}):
            with self.assertRaises(ValueError): rp.encode(value)

    def test_registry_rejects_unimplemented_handler(self):
        with self.assertRaises(ValueError):
            rp.kernel_registry('@kernel("new")\ndef k_unimplemented(e,n): pass')

    @classmethod
    def setUpClass(cls):
        root = (PATHS.repo)
        cls.root = root
        cls.registry = rp.kernel_registry(((PATHS.model_tools / 'htp_emu.py')).read_text())
        cls.sources = rp.read_sources((PATHS.model_data / 'recovered/primary'))
        cls.program = rp.compile_program(*cls.sources, cls.registry)
        cls.raw = rp.pack(cls.program)

    def test_both_graphs_roundtrip_every_descriptor_and_schedule_entry(self):
        for model, count, duplicates in [('primary', 5068, 25), ('alternate', 5040, 19)]:
            with self.subTest(model=model):
                sources = rp.read_sources(PATHS.model_data/'recovered'/model)
                program = rp.compile_program(*sources, self.registry)
                restored = rp.unpack(rp.pack(program))
                rp.verify_sources(restored, *sources, self.registry)
                self.assertEqual(len(restored['commands']), count)
                self.assertEqual(sum(c['duplicate_input_pad'] for c in restored['commands']), duplicates)
                self.assertFalse(restored['memory']['hardware_layout_verified'])
                self.assertGreater(restored['memory']['naive_pool_shortfall_bytes'], 0)

    def test_corruption_and_truncation_rejected(self):
        for raw in (self.raw[:10], self.raw[:-1], self.raw+b'x', self.raw[:100]+bytes([self.raw[100]^1])+self.raw[101:]):
            with self.assertRaises(ValueError): rp.unpack(raw)

    def test_unknown_operation_and_missing_tensor_rejected(self):
        for field, value in [('type_name', 'NewUnimplementedOp'), ('inputs', [999999])]:
            graph = copy.deepcopy(self.sources[0])
            seq = self.sources[1]['nodes'][0]
            next(n for n in graph['nodes'] if n['sequence']==seq)[field] = value
            with self.assertRaises(ValueError):
                rp.compile_program(graph, *self.sources[1:], self.registry)

    def test_missing_recorded_arithmetic_arguments_rejected(self):
        calls = copy.deepcopy(self.sources[2])
        command = next(c for c in self.program['commands'] if c['opcode'] == rp.OPCODES['k_add_u8'])
        del calls['calls'][str(command['node']['sequence'])]
        with self.assertRaises(ValueError):
            rp.compile_program(*self.sources[:2], calls, self.sources[3], self.registry)

    def test_wrong_context_rejected(self):
        order = dict(self.sources[1], context_sha256='00'*32)
        with self.assertRaises(ValueError):
            rp.compile_program(self.sources[0], order, *self.sources[2:], self.registry)

    def test_out_of_bounds_alias_rejected(self):
        program = copy.deepcopy(self.program)
        tensor = next(t for t in program['graph']['tensors'] if (t.get('storage') or {}).get('blocks'))
        tensor['storage']['blocks'][0] = {'pool': 1, 'offset': 8 << 20}
        with self.assertRaises(ValueError): rp.memory_requirements(program)

    def test_backward_dependency_cannot_be_removed(self):
        program = copy.deepcopy(self.program)
        program['commands'][1]['after'] = -1
        with self.assertRaises(ValueError): rp.unpack(rp.pack(program))

    def test_mutated_descriptor_fails_source_comparison(self):
        program = copy.deepcopy(self.program)
        program['graph']['tensors'][0]['quantization']['zero_point'] = 1
        with self.assertRaises(ValueError): rp.verify_sources(program, *self.sources, self.registry)


if __name__ == '__main__': unittest.main()
