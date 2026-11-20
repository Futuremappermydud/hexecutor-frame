# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exhaust every u8 input pair for each recovered add/requant parameter set."""

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
import sys
import numpy as np
ROOT=(PATHS.repo);sys.path[:0]=[str((PATHS.native)),str((PATHS.model_tools))]
import htp_emu,sim_trace
from depth_model.reference.native_reference import native_library
native=native_library();seen=set();count=0
a=np.repeat(np.arange(256,dtype=np.uint8),256);b=np.tile(np.arange(256,dtype=np.uint8),256)
for model in ('primary','alternate'):
    recovered=(PATHS.model_data / 'recovered')/model;g=sim_trace.load_graph(recovered)
    emu=htp_emu.Emulator(g,(recovered/'constants/constant_pool_5.bin').read_bytes(),{})
    for n in g['nodes']:
        kind=n['type_name']
        if kind not in ('Add.tcm@CB*3','linearclip_simple_rq@CB*2.fi'):continue
        add=kind.startswith('Add');r,s=htp_emu.call_args(emu,n,'elementwise_add_crouton_u8' if add else 'requantize_crouton_u8')
        params=(r[4],r[5],*s[:3]) if add else tuple(r[3:6]);key=(add,params)
        if key in seen:continue
        seen.add(key);got=np.empty_like(a)
        if add:
            assert s[1]==0
            status=native.qdm_add_u8(got.ctypes.data,a.ctypes.data,b.ctypes.data,a.size,r[4],r[5],s[0])
            expected=htp_emu.add_u8(a,b,*params)
        else:
            assert r[4]==r[5]==0
            status=native.qdm_requant_u8(got.ctypes.data,a.ctypes.data,a.size,r[3])
            expected=htp_emu.requantize_u8(a,*params)
        assert status==0
        np.testing.assert_array_equal(got,expected);count+=got.size
print(f'PASS: {len(seen)} parameter sets, {count:,} elementwise comparisons')
