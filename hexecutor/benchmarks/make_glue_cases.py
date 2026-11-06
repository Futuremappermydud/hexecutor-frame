"""Extract every recovered u8 add/requant epilogue for the HVX oracle."""

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
ROOT=(PATHS.repo);sys.path[:0]=[str((PATHS.model_tools))]
import htp_emu,sim_trace
seen=set()
for model in ('primary','alternate'):
    recovered=(PATHS.model_data / 'recovered')/model;g=sim_trace.load_graph(recovered)
    emu=htp_emu.Emulator(g,(recovered/'constants/constant_pool_5.bin').read_bytes(),{})
    for node in g['nodes']:
        kind=node['type_name']
        if kind=='Add.tcm@CB*3':
            r,s=htp_emu.call_args(emu,node,'elementwise_add_crouton_u8');seen.add((r[4],r[5],s[0]+8))
        elif kind=='linearclip_simple_rq@CB*2.fi':
            r,s=htp_emu.call_args(emu,node,'requantize_crouton_u8');seen.add((r[3]&65535,0,15-(r[3]>>24)))
cases=sorted(seen)
((PATHS.native / 'build/glue-u8-cases.bin')).write_bytes(struct.pack('<I',len(cases))+b''.join(struct.pack('<3I',*c) for c in cases))
print(f'{len(cases)} recovered parameter sets')
