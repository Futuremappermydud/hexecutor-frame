# SPDX-License-Identifier: LGPL-3.0-or-later

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
import struct,sys
ROOT=(PATHS.repo);sys.path.insert(0,str((PATHS.model_tools)))
import htp_emu,sim_trace
adds=set();affine=set()
for model in ('primary','alternate'):
    rec=(PATHS.model_data / 'recovered')/model;g=sim_trace.load_graph(rec)
    emu=htp_emu.Emulator(g,(rec/'constants/constant_pool_5.bin').read_bytes(),{})
    for node in g['nodes']:
        kind=node['type_name']
        if kind=='Add.tcm@CH*3':
            r,s=htp_emu.call_args(emu,node,'elementwise_add_crouton_u16');adds.add((r[4],r[5],s[0]+1))
        elif kind=='linearclip_simple_rq@CH*2.fi':
            r,s=htp_emu.call_args(emu,node,'requantize_crouton_u16');affine.add((r[3]&65535,r[4]&65535,r[4]>>16,15-(r[3]>>24),0))
        elif kind in ('linearclip@CH*2.fi','linearclip@CH.CB.fi'):
            if emu.t[node['inputs'][0]]['dtype']!='uint16':continue
            p=emu.mem.read(node['inputs'][1]).reshape(-1)
            P,zp=int(p[10])&0xffffffff,int(p[11])&0xffffffff
            m=P&65535;m=-m if p[6]<0 else m
            affine.add((m,zp&65535,zp>>16,15-(P>>24),emu.t[node['outputs'][0]]['dtype']=='uint8'))
def fits(k):
    m,z,o,s,_=k
    return 1<=s<=31 and min(-z*m+(1<<(s-1)),(65535-z)*m+(1<<(s-1)))>=-2**31 and max(-z*m+(1<<(s-1)),(65535-z)*m+(1<<(s-1)))<2**31
valid=[k for k in sorted(affine) if fits(k)]
validadd=[k for k in sorted(adds) if 1<=k[2]<=31 and min(k[:2])>=0 and sum(k[:2])*65535+(1<<(k[2]-1))<2**32]
# Include signed multipliers, endpoint zero points and shift/boundary cases.
extra=[(m,z,o,s,t) for m,z,o,s in ((-32767,32768,32768,15),(65535,32768,1,16),(0,0,65535,1),(1,65535,65535,1),(16384,0,0,31)) for t in (0,1)]
valid=sorted(set(valid+extra));validadd=sorted(set(validadd+[(0,0,1),(65535,0,16),(32768,32768,16)]))
((PATHS.native / 'build/glue-u16-cases.bin')).write_bytes(struct.pack('<2I',len(valid),len(validadd))+b''.join(struct.pack('<5i',*k) for k in valid)+b''.join(struct.pack('<3I',*k) for k in validadd))
print(f'affine: {len(affine)} recovered, {sum(map(fits,affine))} proven; add: {len(adds)} recovered, {len(validadd)-3} proven; with boundaries {len(valid)}/{len(validadd)}')
