"""Exhaust every u8 input pair for each recovered add/requant parameter set."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'frame'),str(ROOT/'tools')]
import htp_emu,sim_trace
from fast_depth import native_library
native=native_library();seen=set();count=0
a=np.repeat(np.arange(256,dtype=np.uint8),256);b=np.tile(np.arange(256,dtype=np.uint8),256)
for model in ('primary','alternate'):
    recovered=ROOT/'recovered'/model;g=sim_trace.load_graph(recovered)
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
