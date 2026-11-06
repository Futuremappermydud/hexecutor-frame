"""Native packets match the original packer, including odd channels and padding."""

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

import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str((PATHS.native)))
from depth_model.reference.native_reference import native_library
from hexecutor.offline.conv_protocol import pack_conv,conv_result,NB
from hexecutor.offline.hvx_protocol import ARENA_BYTES,STATUS,STATUS_OFFSET


class NativePackets(unittest.TestCase):
    def test_packets_and_result_collection(self):
        rng=np.random.default_rng(128);native=native_library();cache={}
        for xs,ws in [((10,34,128),(3,3,128,128)),((3,5,3),(2,2,3,8)),((1,1024,20),(1,1,20,128))]:
            x=rng.integers(0,256,xs,dtype=np.uint8);w=rng.integers(-128,128,ws,dtype=np.int8)
            for immutable in (False,True,True):
                if immutable:w.flags.writeable=False
                a=bytearray(ARENA_BYTES);b=bytearray(ARENA_BYTES)
                layout=pack_conv(a,x,w,1)
                pack_conv(b,x,w,1,native=native,weight_cache=cache)
                self.assertEqual(a,b)
                raw=np.frombuffer(a,'<i4',layout['cb']//4,layout['co'])
                raw[:]=rng.integers(-(2**30),2**30,raw.size,dtype=np.int32)
                units=layout['oh']*-(-layout['owp']//64)*(layout['n']//NB)
                for tid in range(3):STATUS.pack_into(a,STATUS_OFFSET+tid*16,1,0,len(range(tid,units,3)),0)
                np.testing.assert_array_equal(conv_result(a,layout,1,native),conv_result(a,layout,1))
        for xs,ws in [((3,5,3),(2,2,3,32)),((1,129,20),(1,1,20,32)),((10,34,128),(3,3,128,128))]:
            x=rng.integers(0,256,xs,dtype=np.uint8);w=rng.integers(-128,128,ws,dtype=np.int8)
            q=np.tile(np.array([0,1537,4],np.int32),(ws[-1],1))
            a=bytearray(ARENA_BYTES);b=bytearray(ARENA_BYTES)
            L=pack_conv(a,x,w,1,quantization=q,hmx=True)
            pack_conv(b,x,w,1,native=native,quantization=q,hmx=True)
            self.assertEqual(a,b)
            raw=np.frombuffer(a,np.uint8,L['cb'],L['co']);raw[:]=rng.integers(0,256,raw.size,dtype=np.uint8)
            units=L['oh']*-(-L['owp']//64)*(L['n']//NB)
            for tid in range(3):STATUS.pack_into(a,STATUS_OFFSET+tid*16,1,0,len(range(tid,units,3)),0)
            np.testing.assert_array_equal(conv_result(a,L,1,native),conv_result(a,L,1))
        self.assertGreater(len(cache),0)


if __name__=='__main__':unittest.main()
