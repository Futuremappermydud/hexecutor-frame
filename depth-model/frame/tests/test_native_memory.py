"""Compare native blocked access against the calibrated NumPy address oracle."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'frame')]
import htp_emu,sim_trace
from fast_depth import NativeMemory,native_library


class NativeMemoryTests(unittest.TestCase):
    def test_all_tensor_layouts_and_write_regions(self):
        rng=np.random.default_rng(913)
        for model in ('primary','alternate'):
            graph=sim_trace.load_graph(ROOT/'recovered'/model)
            constants=(ROOT/'recovered'/model/'constants/constant_pool_5.bin').read_bytes()
            original=htp_emu.Memory(graph,constants)
            native=NativeMemory(graph,constants,native_library())
            for p in original.pools:
                data=rng.integers(0,256,original.pools[p].size,dtype=np.uint8)
                original.pools[p][:]=data;native.pools[p][:]=data
            # Every storage geometry, shape and padding combination is checked.
            checked=set()
            for t in graph['tensors']:
                if not t.get('shape') or not t.get('storage'):continue
                if t['dtype'] not in htp_emu.NP:continue
                if not (t['storage'].get('direct') or t['storage'].get('blocks')):continue
                signature=(t['class_name'],str(t['shape']),str(t['storage']))
                if signature in checked:continue
                checked.add(signature);tid=t['id']
                for full in (False,True):
                    expected=original.read(tid,full)
                    actual=native.read(tid,full)
                    np.testing.assert_array_equal(actual.view(np.uint8),expected.view(np.uint8),err_msg=f'{model} {tid} {full}')
                value=original.read(tid)
                original.write(tid,value);native.write(tid,value)
                if htp_emu.is_crouton(t):
                    value=original.read(tid,full=True)
                    original.write_region(tid,value);native.write_region(tid,value)
                # Avoid retaining the original oracle's enormous address maps.
                original._addr.clear()
            for p in original.pools:
                np.testing.assert_array_equal(original.pools[p],native.pools[p],err_msg=f'{model} pool {p}')
            print(model,len(checked),'tensor descriptors checked',flush=True)


if __name__=='__main__':unittest.main()
