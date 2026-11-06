"""Sweep L2 cache fields 0..5 on Frame; one restored callback per size."""

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

import array
import json
import struct
import sys
from pathlib import Path
sys.path.insert(0,str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport

FRAME=(PATHS.native)
size=int(sys.argv[1])
if size not in range(6):raise SystemExit('size field must be 0..5')
with FrameTransport((PATHS.native / 'build/qd_loader.elf'),(PATHS.native / 'build/qd_l2sweep_probe.elf'),30,raise_clock=False) as t:
    def prepare(seq):
        t.arena[:4096]=bytes(4096)
        struct.pack_into('<II',t.arena,8,seq,0)
        struct.pack_into('<I',t.arena,64,size)
        values=array.array('I',((i*0x9e3779b1)&0xffffffff for i in range(262144)))
        t.arena[0x300000:0x400000]=values.tobytes()
        t.fence()
    out=t.invoke(prepare,t.workers_done,lambda:struct.unpack_from('<97I',t.arena,2048))
if out[0]!=0x5057324c:raise SystemExit(f'probe did not run: {out[:8]}')
rows=[{'footprint_bytes':n*4,'first_cycles':out[8+4*i],'second_cycles':out[9+4*i],
       'sum_equal':out[10+4*i]==out[11+4*i]}
      for i,n in enumerate((2048,16384,65536,131072,196608,262144))]
rep={'requested_field':size,'syscfg_before':hex(out[2]),'syscfg_cache':hex(out[4]),
     'syscfg_restored':hex(out[5]),'ccr_before':hex(out[3]),'ccr_restored':hex(out[7]),
     'changed_tcm_words':out[6],'changed_per_16k':list(out[32:96]),'rows':rows,'error':out[96]}
print(json.dumps(rep,indent=1))
if len(sys.argv)>2:Path(sys.argv[2]).write_text(json.dumps(rep,indent=1)+'\n')
if rep['error'] or not all(r['sum_equal'] for r in rows):raise SystemExit('L2 sweep validation failed')
