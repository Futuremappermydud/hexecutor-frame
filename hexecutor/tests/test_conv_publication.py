"""Regression tests for publishing commands to live polling workers."""

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

import struct
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str((PATHS.native)))
from hexecutor.offline.conv_protocol import BARRIER_OFFSET, COMMAND, pack_conv, pack_exit
from hexecutor.offline.hvx_protocol import ARENA_BYTES


class PublicationTests(unittest.TestCase):
    def test_live_sequence_never_changes_before_fence(self):
        arena=bytearray(ARENA_BYTES)
        struct.pack_into('<I',arena,8,41)
        struct.pack_into('<3I',arena,BARRIER_OFFSET,41,41,41)
        seen=[]
        def fence():
            seen.append(struct.unpack_from('<I',arena,8)[0])
        pack_conv(arena,np.full((3,5,3),255,np.uint8),np.full((2,2,3,8),-128,np.int8),42,fence)
        self.assertEqual(seen,[41,42])
        self.assertEqual(struct.unpack_from('<3I',arena,BARRIER_OFFSET),(41,41,41))
        self.assertEqual(COMMAND.unpack_from(arena)[16],20000)
        seen.clear()
        pack_exit(arena,43,fence)
        self.assertEqual(seen,[42,43])
        self.assertEqual(COMMAND.unpack_from(arena)[9],(~43)&0xffffffff)

    def test_clock_opt_out_is_explicit(self):
        arena=bytearray(ARENA_BYTES)
        x=np.zeros((1,1,4),np.uint8);w=np.zeros((1,1,4,8),np.int8)
        pack_conv(arena,x,w,1,raise_clock=False)
        self.assertEqual(COMMAND.unpack_from(arena)[17],1)
        pack_conv(arena,x,w,2)
        self.assertEqual(COMMAND.unpack_from(arena)[17],0)


if __name__=='__main__': unittest.main()
