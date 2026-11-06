"""Run the read-only L2 map probe (build/qd_l2map_probe.elf): stack pointers,
TLB entries and L2-TCM occupancy at callback entry."""
import json
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frame_transport import FrameTransport

FRAME = Path(__file__).resolve().parent.parent
with FrameTransport(FRAME/'build/qd_loader.elf', FRAME/'build/qd_l2map_probe.elf', 30, raise_clock=False) as t:
    def prepare(seq):
        t.arena[:4096] = bytes(4096); struct.pack_into('<I', t.arena, 8, seq); t.fence()
    out = t.invoke(prepare, t.workers_done, lambda: struct.unpack_from('<448I', t.arena, 2048))
if out[0] != 0x50414d32:
    raise SystemExit(f'probe did not run: {out[:8]}')
tlb = []
for i in range(128):
    lo, hi = out[64+2*i], out[65+2*i]
    e = (hi << 32) | lo
    if e:
        tlb.append({'i': i, 'raw': f'{e:#018x}'})
rep = {'sp': [hex(v) for v in out[8:11]], 'tlb': tlb,
       'l2tcm_nonzero_words_per_16k': list(out[320:384]), 'l2tcm_hash_per_16k': [f'{v:08x}' for v in out[384:448]]}
print(json.dumps(rep, indent=1))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(rep, indent=1) + '\n')
