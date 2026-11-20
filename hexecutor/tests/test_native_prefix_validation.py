# SPDX-License-Identifier: LGPL-3.0-or-later

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

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

FRAME=(PATHS.native)
sys.path.insert(0,str((PATHS.benchmarks)))
spec=importlib.util.spec_from_file_location('native_prefix',(PATHS.benchmarks / 'verify_native_resident_prefix.py'))
prefix=importlib.util.module_from_spec(spec);spec.loader.exec_module(prefix)


class NativePrefixValidation(unittest.TestCase):
    def test_only_target_allows_plus_one(self):
        expected=np.array([3,4,5,6],dtype=np.uint8);mask=np.array([False,True,True,False])
        self.assertTrue(prefix.compare_pool(np.array([3,5,5,6],dtype=np.uint8),expected,mask)['passed'])
        self.assertFalse(prefix.compare_pool(np.array([4,5,5,6],dtype=np.uint8),expected,mask)['passed'])
        self.assertFalse(prefix.compare_pool(np.array([3,3,5,6],dtype=np.uint8),expected,mask)['passed'])
        self.assertFalse(prefix.compare_pool(np.array([3,6,5,6],dtype=np.uint8),expected,mask)['passed'])

    def test_cropped_rows_are_outside_native_write_footprint(self):
        output={'class_name':'CB','dtype':'uint8',
                'shape':{'dimensions':[1,2,8,32],'max_dimensions':[1,8,8,32],'padding':[0,0,0,0]},
                'storage':{'slice_offset':1,'blocks':[{'pool':1,'offset':0},{'pool':1,'offset':2048}]}}
        weight={'shape':{'max_dimensions':[1,1,32,32]}}
        input_tensor={'shape':{'max_dimensions':[1,2,8,32],'padding':[0,0,0,0]}}
        mask=prefix.output_footprint(output,weight,input_tensor,{1:4096})[1]
        self.assertEqual(int(mask.sum()),512)
        self.assertFalse(mask[:2048].any())
        self.assertTrue(mask[2048:2560].all())
        self.assertFalse(mask[2560:].any())

    def test_real_first_command_has_full_expected_output(self):
        graph=prefix.resident_image.unpack(((PATHS.native / 'build/resident-primary.qdrx')).read_bytes())
        tensors={t['id']:t for t in graph['tensors']}
        command=next(c for c in graph['commands'] if c['sequence']==1195)
        xt,wt=[tensors[i] for i in command['inputs'][:2]]
        output=tensors[command['outputs'][0]]
        masks=prefix.output_footprint(output,wt,xt,{1:8<<20,3:8<<20,4:8<<20,5:16<<20,6:8<<20})
        self.assertEqual(sum(int(m.sum()) for m in masks.values()),65536)

    def test_padded_ci_uses_physical_output_footprint(self):
        graph=prefix.resident_image.unpack(((PATHS.native / 'build/resident-primary.qdrx')).read_bytes())
        tensors={t['id']:t for t in graph['tensors']}
        command=next(c for c in graph['commands'] if c['sequence']==115)
        xt,wt=[tensors[i] for i in command['inputs'][:2]]
        output=tensors[command['outputs'][0]]
        masks=prefix.output_footprint(output,wt,xt,{1:8<<20,3:8<<20,4:8<<20,5:16<<20,6:8<<20})
        self.assertEqual(sum(int(m.sum()) for m in masks.values()),65536)
        wt['shape']['max_dimensions'][2]=16
        with self.assertRaises(ValueError):
            prefix.output_footprint(output,wt,xt,{1:8<<20})


if __name__=='__main__':unittest.main()
