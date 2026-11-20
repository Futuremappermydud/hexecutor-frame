#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Source semantics, native fail-closed validation and bounded dispatch tests."""

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
import ctypes as ct
import hashlib
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = (PATHS.repo)
sys.path.insert(0, str((PATHS.native)))
import depth_model.lowering.resident_program as rp
import depth_model.lowering.resident_image as ri
class View(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('bytes', ct.c_uint32),
                ('offset', ct.c_uint32*6), ('count', ct.c_uint32*6), ('pools', ct.c_uint32*7)]


class Status(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in ('code', 'command', 'tensor', 'detail')]


Kernel = ct.CFUNCTYPE(ct.c_uint32, ct.POINTER(View), ct.c_uint32, ct.c_void_p)
Cancel = ct.CFUNCTYPE(ct.c_uint32, ct.c_void_p)


def checksum(raw):
    raw = bytearray(raw)
    struct.pack_into('<I', raw, 16, 0)
    struct.pack_into('<I', raw, 16, zlib.crc32(raw))
    return bytes(raw)


def mutate(raw, offset, value):
    out = bytearray(raw)
    struct.pack_into('<I', out, offset, value)
    return checksum(out)


class ResidentImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        library = Path(cls.tmp.name)/'resident.so'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-shared', '-fPIC', str((PATHS.native / 'qd_resident.c')), '-o', str(library)], check=True)
        cls.lib = ct.CDLL(str(library))
        cls.lib.qdr_open.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(View), ct.POINTER(Status)]
        cls.lib.qdr_open.restype = ct.c_uint32
        cls.lib.qdr_dispatch.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(Kernel), ct.c_uint32,
                                        ct.c_uint32, Cancel, ct.c_void_p, ct.POINTER(Status)]
        cls.lib.qdr_dispatch.restype = ct.c_uint32
        cls.programs, cls.images = {}, {}
        registry = rp.kernel_registry(((PATHS.model_tools / 'htp_emu.py')).read_text())
        for model in ('primary', 'alternate'):
            p = rp.compile_program(*rp.read_sources((PATHS.model_data / 'recovered')/model), registry)
            cls.programs[model] = p
            cls.images[model] = ri.pack(p)
        cls.raw = cls.images['primary']
        cls.header = struct.unpack_from('<64I', cls.raw)
        cls.sections = [cls.header[37+i*2] for i in range(6)]

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def native(self, raw):
        # Also exercise an unaligned base. No native struct casts are allowed.
        buf = ct.create_string_buffer(b'x'+raw)
        view, status = View(), Status()
        code = self.lib.qdr_open(ct.addressof(buf)+1, len(raw), ct.byref(view), ct.byref(status))
        self.assertEqual(code, status.code)
        if code:
            self.assertFalse(view.data)
            self.assertEqual(view.bytes, 0)
        return code, status

    def test_generated_registry_and_emulator_registry_agree(self):
        self.assertEqual(((PATHS.native / 'qd_resident_schema.h')).read_text(), ri.c_registry())
        registry = rp.kernel_registry(((PATHS.model_tools / 'htp_emu.py')).read_text())
        self.assertEqual(registry, dict(zip(ri.schema.VARIANTS, ri.schema.VARIANT_HANDLERS)))

    def test_both_images_retain_all_execution_semantics_and_validate_natively(self):
        for model, raw in self.images.items():
            with self.subTest(model=model):
                program = self.programs[model]
                ri.verify_sources(raw, program)
                self.assertEqual(self.native(raw)[0], 0)
                self.assertLess(len(raw), 2 << 20)
                decoded = ri.unpack(raw)
                self.assertEqual(len(decoded['commands']), len(program['commands']))
                for a, b in zip(decoded['commands'], ri.semantic_commands(program)):
                    self.assertEqual(a, b)
                # Exact source pool/offsets preserve mutable aliases and dynamic
                # weight/Fi storage; no tensors are promoted to constants.
                for a, b in zip(decoded['tensors'], ri.semantic_tensors(program)):
                    self.assertEqual(a, b)
                self.assertFalse(program['memory']['hardware_layout_verified'])

    def test_deterministic_interning_and_u64_call_arguments(self):
        self.assertEqual(self.raw, ri.pack(self.programs['primary']))
        self.assertEqual(hashlib.sha256(self.raw).digest(), hashlib.sha256(ri.pack(self.programs['primary'])).digest())
        self.assertLess(self.header[40], sum(len((t.get('storage') or {}).get('blocks', []))
                                           for t in self.programs['primary']['graph']['tensors']))
        calls = [call for c in ri.unpack(self.raw)['commands'] for call in c['calls']]
        self.assertTrue(any(isinstance(a, int) and a > (1 << 63) for _, r, s in calls for a in r+s))
        self.assertTrue(any(a == 'P' for _, r, s in calls for a in r+s))

    def test_crc_truncation_and_header_rejected(self):
        for raw in (self.raw[:255], self.raw[:-1], self.raw+b'x',
                    self.raw[:300]+bytes([self.raw[300]^1])+self.raw[301:]):
            self.assertNotEqual(self.native(raw)[0], 0)
            with self.assertRaises(ValueError): ri.unpack(raw)
        for word in (0, 1, 2, 3, 5, 6, 7, 49, 63):
            self.assertNotEqual(self.native(mutate(self.raw, word*4, ri.NONE))[0], 0)

    def test_every_section_rejects_overflow_overlap_and_out_of_bounds(self):
        for i in range(6):
            for value in (0, 257, len(self.raw)+1, ri.NONE):
                self.assertNotEqual(self.native(mutate(self.raw, (37+2*i)*4, value))[0], 0)
            self.assertNotEqual(self.native(mutate(self.raw, (38+2*i)*4, ri.NONE))[0], 0)

    def test_descriptor_reference_and_call_corruptions_rejected(self):
        tensor, blocks, command, refs, words, calls = self.sections
        for offset, value in ((tensor, 0), (tensor+4, 99), (tensor+8, 0), (tensor+12, 128),
                              (tensor+16, ri.NONE), (tensor+120, 1),
                              (blocks, (8 << 20)|1), (blocks, 6), (blocks, 129),
                              (command+4, 0), (command+8, ri.NONE), (command+12, 1),
                              (command+16, 0), (command+20, ri.NONE),
                              (command+24, ri.NONE), (command+28, ri.NONE),
                              (command+72, 1), (refs, ri.NONE), (calls, 0),
                              (calls+4, 1024), (calls+8, 1), (24*4, 1), (25*4, ri.NONE)):
            with self.subTest(offset=offset, value=value):
                self.assertNotEqual(self.native(mutate(self.raw, offset, value))[0], 0)

    def test_bad_spill_and_zap_pool_ranges_rejected(self):
        for opcode in ('k_spill_fill', 'k_block_zap'):
            commands = self.programs['primary']['commands']
            index = next(i for i, c in enumerate(commands) if c['opcode'] == rp.OPCODES[opcode])
            c = struct.unpack_from('<24I', self.raw, self.sections[2]+index*96)
            # Spill offset or the first zap source.
            at = c[14] + (3 if opcode == 'k_spill_fill' else 2)
            self.assertNotEqual(self.native(mutate(self.raw, self.sections[4]+at*4, ri.NONE))[0], 0)

    def test_arithmetic_call_cannot_be_removed(self):
        index = next(i for i, c in enumerate(self.programs['primary']['commands'])
                     if c['opcode'] == rp.OPCODES['k_add_u8'])
        raw = bytearray(self.raw)
        struct.pack_into('<2I', raw, self.sections[2]+index*96+64, 0, 0)
        code, status = self.native(checksum(raw))
        self.assertEqual(code, 8)
        self.assertEqual(status.command, index)

    def test_shared_span_reuse_cannot_exhaust_validator(self):
        raw = bytearray(self.raw)
        # Reusing the whole word table thousands of times is in bounds, but
        # would multiply work without an aggregate validator budget.
        for i, command in enumerate(self.programs['primary']['commands']):
            if command['opcode'] not in (rp.OPCODES['k_spill_fill'], rp.OPCODES['k_block_zap']):
                struct.pack_into('<2I', raw, self.sections[2]+i*96+56, 0, self.header[46])
        self.assertEqual(self.native(checksum(raw))[0], 12)

    def dispatch(self, raw=None, handlers=None, budget=65536, cancel=None):
        raw = raw or self.raw
        buf, status = ct.create_string_buffer(raw), Status()
        handlers = handlers or (Kernel * 76)()
        cancel = cancel or Cancel()
        code = self.lib.qdr_dispatch(buf, len(raw), handlers, len(handlers), budget,
                                     cancel, None, ct.byref(status))
        return code, status

    def test_missing_variant_is_rejected_before_any_kernel_executes(self):
        visited = []
        callback = Kernel(lambda v, i, ctx: visited.append(i) or 0)
        handlers = (Kernel * 76)(*([callback]*76))
        # Remove a late variant, after earlier implemented operations.
        variant = ri.VARIANTS['Dequantize@Ff.CH']
        handlers[variant] = Kernel()
        code, status = self.dispatch(handlers=handlers)
        self.assertEqual(code, 10)
        self.assertEqual(status.detail, variant)
        self.assertGreater(status.command, 0)
        self.assertEqual(visited, [])

    def test_serial_dispatch_budget_cancellation_and_kernel_error(self):
        visited = []
        callback = Kernel(lambda v, i, ctx: visited.append(i) or 0)
        handlers = (Kernel * 76)(*([callback]*76))
        self.assertEqual(self.dispatch(handlers=handlers, budget=1)[0], 12)
        self.assertEqual(visited, [])
        cancel = Cancel(lambda ctx: len(visited) >= 7)
        code, status = self.dispatch(handlers=handlers, cancel=cancel)
        self.assertEqual(code, 11)
        self.assertEqual(len(visited), 7)
        self.assertEqual(visited, sorted(visited))
        visited.clear()
        self.assertEqual(self.dispatch(handlers=handlers)[0], 0)
        expected = [i for i, c in enumerate(self.programs['primary']['commands']) if c['opcode'] != rp.OPCODES['k_nop']]
        self.assertEqual(visited, expected)
        error = Kernel(lambda v, i, ctx: 123)
        handlers = (Kernel * 76)(*([error]*76))
        code, status = self.dispatch(handlers=handlers)
        self.assertEqual((code, status.command, status.detail), (13, 0, 123))

    def test_unencodable_call_and_unknown_variant_fail_lowering(self):
        for target in ('call', 'variant'):
            program = copy.deepcopy(self.programs['primary'])
            if target == 'variant': program['commands'][0]['node']['type_name'] = 'new'
            else:
                c = next(c for c in program['commands'] if c['calls'])
                c['calls'][0][0] = 'new'
            with self.assertRaises(ValueError): ri.pack(program)

    def test_random_structural_mutations_return_diagnostics(self):
        # Recompute CRC to reach the native structural validator. Random changes
        # to immediate values can be structurally valid, so do not insist every
        # mutation fails; source identity remains the host's responsibility.
        rng = random.Random(1941)
        for _ in range(80):
            offset = rng.choice(self.sections) + 4*rng.randrange(8)
            raw = mutate(self.raw, offset, rng.getrandbits(32))
            code, _ = self.native(raw)
            self.assertIn(code, range(14))


if __name__ == '__main__': unittest.main()
