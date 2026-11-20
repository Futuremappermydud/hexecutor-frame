#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Host-only proof checks for fixture layout, oracle mapping, and retain decisions."""

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

import hashlib, importlib.util, struct, unittest
import contextlib, io, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch

HERE=(PATHS.native / 'kernels/convolution')
spec=importlib.util.spec_from_file_location('probe',(PATHS.native / 'kernels/convolution/run_xwb1022_frame_probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class MockTransport:
    def __init__(self): self.retained=None
    def retain_session(self,reason): self.retained=reason

class HostProof(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess
        subprocess.run(['python3',str((PATHS.native / 'kernels/convolution/make_xwb1022_frame_fixture.py'))],check=True,capture_output=True,text=True)
        cls.fixture=((PATHS.native / 'kernels/convolution/xwb1022_frame_fixture.bin')).read_bytes()

    def test_fixture_sizes_hashes_and_bounds(self):
        parts=probe.parse_fixture(self.fixture)
        self.assertEqual(tuple(map(len,parts)),probe.SIZES)
        self.assertLess(probe.FIXTURE_OFFSET+len(self.fixture),0x80000)
        self.assertLess(probe.ECHO_OFFSET+len(parts[0]),0x170000)

    def test_physical_mapping_matches_reference_deltas(self):
        inp,w,fi,sim,cpu=probe.parse_fixture(self.fixture)
        r=probe.validate_outputs(sim,cpu,sim)
        self.assertTrue(r['simulator_physical_exact'])
        self.assertTrue(r['cpu_0_or_plus1'])
        self.assertEqual(r['logical_delta_counts'],{'0':63095,'1':2441})

    def test_bad_fixture_rejected(self):
        bad=bytearray(self.fixture);bad[-1]^=1
        with self.assertRaises(ValueError): probe.parse_fixture(bytes(bad))

    def test_restore_retain_on_tlb_power_or_vtcm_failure(self):
        good=[0]*32;good[17]=1;good[18]=1;good[19:23]=probe.EXPECTED_TLB;good[23:27]=probe.EXPECTED_TLB
        self.assertFalse(probe.restore_problem(good))
        cases=[]
        for pos,val in [(10,1),(11,1),(17,0),(19,0),(23,0)]:
            row=good.copy();row[pos]=val;cases.append(row)
        for row in cases:
            transport=MockTransport()
            if probe.restore_problem(row): transport.retain_session('retain for recovery')
            self.assertEqual(transport.retained,'retain for recovery')

    def exercise_runner(self, fault=None):
        parts=probe.parse_fixture(self.fixture); inp,w,fi,sim,cpu=parts; objects=[]
        class FakeTransport:
            def __init__(self,*_):
                self.arena=bytearray(0x800000);self.payload_sha256='mock-payload';self.retained=None;objects.append(self)
            def __enter__(self): return self
            def __exit__(self,*_): return False
            def fence(self): pass
            def retain_session(self,reason): self.retained=reason
            def workers_done(self): return True
            def workers_exited(self): return True
            def invoke(self,prepare,wait,collect):
                prepare(7);self.assert_packet=struct.unpack_from('<6I',self.arena)
                self.assert_ready=bool(wait())
                words=[0]*32;words[0]=probe.MAGIC;words[17]=1;words[18]=1
                words[19:23]=probe.EXPECTED_TLB;words[23:27]=probe.EXPECTED_TLB
                if fault=='power': words[10]=1
                if fault=='restore': words[17]=0
                if fault=='tlb': words[23]^=1
                for i,data in zip((28,29,30),(inp,w,fi)): words[i]=probe.fnv32(data)
                struct.pack_into('<32I',self.arena,2048,*words)
                self.arena[probe.OUTPUT_OFFSET:probe.OUTPUT_OFFSET+len(sim)]=sim
                if fault=='output': self.arena[probe.OUTPUT_OFFSET]^=1
                self.arena[probe.ECHO_OFFSET:probe.ECHO_OFFSET+len(inp)]=inp
                return collect()
        ft=types.ModuleType('frame_transport');ft.FrameTransport=FakeTransport
        bench=types.ModuleType('bench');vr=types.ModuleType('bench.verify_resident_decoder');vr.snapshot=lambda:{'mock':True}
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules,{'frame_transport':ft,'bench':bench,'bench.verify_resident_decoder':vr}), patch('sys.argv',['probe','--report',td+'/r.json']),contextlib.redirect_stdout(io.StringIO()):
            if fault:
                message='3x3 probe failed validation' if fault=='output' else 'HMX/power/TLB/VTCM restore verification failed'
                with self.assertRaisesRegex(RuntimeError,message): probe.main()
            else: probe.main()
            import json
            report=json.loads(Path(td+'/r.json').read_text())
            self.assertEqual(report['passed'],fault is None)
            self.assertEqual(Path(td+'/r.output.bin').stat().st_size,65536)
            self.assertEqual(Path(td+'/r.input-echo.bin').stat().st_size,probe.IN_BYTES)
        self.assertEqual(objects[0].assert_packet,probe.struct.unpack('<6I',probe.make_packet(7)))
        self.assertTrue(objects[0].assert_ready)
        self.assertEqual(objects[0].retained is not None,fault in ('power','restore','tlb'))

    def test_runner_collects_valid_mock_once(self): self.exercise_runner()
    def test_runner_retains_on_shutdown_failure(self): self.exercise_runner('power')
    def test_runner_retains_on_vtcm_restore_failure(self): self.exercise_runner('restore')
    def test_runner_retains_on_changed_restore_tlb(self): self.exercise_runner('tlb')
    def test_runner_preserves_numerical_failure_evidence(self): self.exercise_runner('output')

    def test_device_packet_is_single_call_and_bounded(self):
        packet=probe.make_packet(7)
        self.assertEqual(struct.unpack('<6I',packet),(probe.MAGIC,1,7,probe.FIXTURE_OFFSET,0x700000,1))
        self.assertEqual(hashlib.sha256(self.fixture).hexdigest(),hashlib.sha256(((PATHS.native / 'kernels/convolution/xwb1022_frame_fixture.bin')).read_bytes()).hexdigest())

if __name__=='__main__': unittest.main()
