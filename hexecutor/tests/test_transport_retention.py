# SPDX-License-Identifier: LGPL-3.0-or-later
"""A failed memory restore must retain state even after workers are released."""

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

from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport,CTRL,WORK
from hexecutor.offline.hvx_protocol import ARENA_OFFSET,ARENA_BYTES,CODE_BASE,CODE_BYTES,PAYLOAD_BASE,PAYLOAD_BYTES

class Memory(bytearray):
    def close(self):self.closed=True

def fixture(active=False):
    t=FrameTransport.__new__(FrameTransport)
    mem=Memory(0x2000000);t.mem=mem;t.arena=memoryview(mem)[ARENA_OFFSET:ARENA_OFFSET+ARENA_BYTES]
    t.persistent=active;t.job_active=active;t.installed=True;t.registered=True;t._stuck=False;t._retain_reason=None
    t.fds=[];t.socket=None;t.lock=None;t.eye_stopped=False;t.original_eye=None
    t.saved_work=bytes(32);t.saved_code=bytes(CODE_BYTES);t.saved_payload=bytes(PAYLOAD_BYTES)
    struct.pack_into('<I',mem,CTRL,1);struct.pack_into('<I',mem,CTRL+8,1);struct.pack_into('<I',mem,WORK+24,CODE_BASE)
    code=CODE_BASE-0x87000000;payload=PAYLOAD_BASE-0x87000000
    mem[code:code+CODE_BYTES]=b'\x11'*CODE_BYTES
    mem[payload:payload+PAYLOAD_BYTES]=b'\x22'*PAYLOAD_BYTES
    t.arena[:]=b'\x33'*ARENA_BYTES
    # Clock/HMX state is restored before the memory-lease result is inspected.
    t.arena[256:260]=bytes(4);t.arena[512:516]=bytes(4)
    def unexpected(*args,**kwargs):raise AssertionError('retained session dispatched/released a worker')
    t.stop_persistent=unexpected;t.wait=unexpected
    return t,mem,code,payload

class Retention(unittest.TestCase):
    def test_retains_after_worker_release_and_during_job(self):
        for active in (False,True):
            with self.subTest(active=active):
                t,mem,code,payload=fixture(active)
                before=bytes(mem);t.eye_stopped=True;t.original_eye=('do not restart',)
                t.retain_session('restore was not verified')
                with self.assertRaisesRegex(RuntimeError,'explicit recovery'):
                    t._close()
                self.assertEqual(bytes(mem),before)
                self.assertTrue(mem.closed)
                self.assertIsNone(t.mem)

    def test_ordinary_completed_session_cleans_regions(self):
        t,mem,code,payload=fixture()
        t._close()
        self.assertEqual(struct.unpack_from('<I',mem,WORK+24)[0],0)
        self.assertEqual(mem[code:code+CODE_BYTES],bytes(CODE_BYTES))
        self.assertEqual(mem[payload:payload+PAYLOAD_BYTES],bytes(PAYLOAD_BYTES))
        self.assertEqual(mem[ARENA_OFFSET:ARENA_OFFSET+ARENA_BYTES],bytes(ARENA_BYTES))

if __name__=='__main__':unittest.main()
