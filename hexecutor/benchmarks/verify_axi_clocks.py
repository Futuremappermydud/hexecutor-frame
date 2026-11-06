"""Bounded 750/1000/750 MHz AXI tests; DDR-controller clock stays unchanged."""

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

import json
from pathlib import Path
import statistics
import struct
import sys
import numpy as np
FRAME=(PATHS.native);sys.path.insert(0,str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport
from verify_resident_decoder import snapshot
report={'before':snapshot(),'runs':[],'kind':'Q6 AXI-master clock, not DDR clock'}
for mhz in (750,1000,750):
    with FrameTransport((PATHS.native / 'build/qd_loader.elf'),(PATHS.native / f'build/qd_axi_{mhz}.elf'),20) as t:
        ring=np.arange(262144,dtype=np.uint32)+1;ring[4095]=0
        def prepare(seq):
            t.arena[128:176]=bytes(48);t.arena[2048:2304]=bytes(256)
            t.arena[0x100000:0x200000]=ring.tobytes()
            struct.pack_into('<I',t.arena,8,seq);t.fence()
        words=t.invoke(prepare,t.workers_done,lambda:struct.unpack_from('<28I',t.arena,2048))
        assert words[0]==mhz and words[1]==words[5]==words[6]==0,words
        row={'mhz':mhz,'pll_l':hex(words[2]),'pll_alpha':hex(words[3]),'cfg':hex(words[4]),
             'vector_cycles':list(words[8::2]),'scalar_cycles':list(words[9::2])}
        row['vector_gbps']=1048576/(statistics.median(row['vector_cycles'])/1843200000)/1e9
        row['dependent_scalar_ns']=statistics.median(row['scalar_cycles'])/1843200000*1e9/4096
    row['after']=snapshot();report['runs'].append(row);print(json.dumps(row),flush=True)
report['passed']=True
((PATHS.native / 'build/axi-750-1000-validation.json')).write_text(json.dumps(report,indent=2)+'\n')
