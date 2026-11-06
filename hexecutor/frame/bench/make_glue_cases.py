"""Extract every recovered u8 add/requant epilogue for the HVX oracle."""
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools')]
import htp_emu,sim_trace
seen=set()
for model in ('primary','alternate'):
    recovered=ROOT/'recovered'/model;g=sim_trace.load_graph(recovered)
    emu=htp_emu.Emulator(g,(recovered/'constants/constant_pool_5.bin').read_bytes(),{})
    for node in g['nodes']:
        kind=node['type_name']
        if kind=='Add.tcm@CB*3':
            r,s=htp_emu.call_args(emu,node,'elementwise_add_crouton_u8');seen.add((r[4],r[5],s[0]+8))
        elif kind=='linearclip_simple_rq@CB*2.fi':
            r,s=htp_emu.call_args(emu,node,'requantize_crouton_u8');seen.add((r[3]&65535,0,15-(r[3]>>24)))
cases=sorted(seen)
(ROOT/'frame/build/glue-u8-cases.bin').write_bytes(struct.pack('<I',len(cases))+b''.join(struct.pack('<3I',*c) for c in cases))
print(f'{len(cases)} recovered parameter sets')
