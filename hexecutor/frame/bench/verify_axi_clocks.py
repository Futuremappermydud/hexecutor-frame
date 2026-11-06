"""Bounded 750/1000/750 MHz AXI tests; DDR-controller clock stays unchanged."""
import json
from pathlib import Path
import statistics
import struct
import sys
import numpy as np
FRAME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(FRAME))
from frame_transport import FrameTransport
from verify_resident_decoder import snapshot
report={'before':snapshot(),'runs':[],'kind':'Q6 AXI-master clock, not DDR clock'}
for mhz in (750,1000,750):
    with FrameTransport(FRAME/'build/qd_loader.elf',FRAME/f'build/qd_axi_{mhz}.elf',20) as t:
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
(FRAME/'build/axi-750-1000-validation.json').write_text(json.dumps(report,indent=2)+'\n')
