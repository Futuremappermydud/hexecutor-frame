"""Run the bounded hardware L2 cache carve-out probe (built separately)."""

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
import struct
import sys
from pathlib import Path
sys.path.insert(0, str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport

FRAME = (PATHS.native)
with FrameTransport((PATHS.native / 'build/qd_loader.elf'), (PATHS.native / 'build/qd_l2cache_probe.elf'), 30, raise_clock=False) as t:
    def prepare(seq):
        t.arena[:4096] = bytes(4096)
        struct.pack_into('<I', t.arena, 8, seq)
        for i in range(8192):
            struct.pack_into('<I', t.arena, 0x40000+i*4, (i*0x9e3779b1)&0xffffffff)
        t.fence()
    out = t.invoke(prepare, t.workers_done, lambda: struct.unpack_from('<96I', t.arena, 2048))
if out[0] != 0x4341324c:
    raise SystemExit(f'probe did not run: {out[:8]}')
rep = dict(syscfg_before=hex(out[1]), ccr_before=hex(out[2]), syscfg_cache=hex(out[3]),
           first_pass_cycles=out[4], second_pass_cycles=out[5],
           first_sum=out[6], second_sum=out[7], syscfg_restored=hex(out[8]),
           changed_words_full_tcm=out[9], first_changed_word=out[10],
           ccr_restored=hex(out[11]), error=out[12], changed_per_16k=list(out[32:96]))
print(json.dumps(rep, indent=1))
if len(sys.argv)>1:
    Path(sys.argv[1]).write_text(json.dumps(rep, indent=1)+'\n')
if rep['error'] or out[6]!=out[7] or out[8]!=out[1] or out[11]!=out[2]:
    raise SystemExit('L2 cache probe failed validation')
