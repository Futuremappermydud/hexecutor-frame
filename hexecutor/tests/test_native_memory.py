"""Compare native blocked access against the calibrated NumPy address oracle."""

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
ROOT=(PATHS.repo)
sys.path[:0]=[str((PATHS.model_tools)),str((PATHS.native))]
import htp_emu,sim_trace
from depth_model.reference.native_reference import NativeMemory,native_library


class NativeMemoryTests(unittest.TestCase):
    def test_all_tensor_layouts_and_write_regions(self):
        rng=np.random.default_rng(913)
        for model in ('primary','alternate'):
            graph=sim_trace.load_graph((PATHS.model_data / 'recovered')/model)
            constants=((PATHS.model_data / 'recovered')/model/'constants/constant_pool_5.bin').read_bytes()
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
