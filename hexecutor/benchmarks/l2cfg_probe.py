"""Run the read-only L2 configuration probe (build/qd_l2cfg_probe.elf) and print
SYSCFG/CCR/SSR/CFGBASE and the core configuration table."""

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
NAMES = {0x10: 'l2_cfg_base', 0x2c: 'tlb_count', 0x30: 'coproc_type', 0x34: 'coproc_ctx', 0x38: 'vtcm_base',
         0x3c: 'tcm_size', 0x40: 'l2_tag_size', 0x44: 'l2_array_size', 0x48: 'thread_mask'}

with FrameTransport((PATHS.native / 'build/qd_loader.elf'), (PATHS.native / 'build/qd_l2cfg_probe.elf'), 30, raise_clock=False) as t:
    def prepare(seq):
        t.arena[:4096] = bytes(4096); struct.pack_into('<I', t.arena, 8, seq); t.fence()
    out = t.invoke(prepare, t.workers_done, lambda: struct.unpack_from('<48I', t.arena, 2048))
if out[0] != 0x47464332:
    raise SystemExit(f'probe did not run: {out[:8]}')
syscfg = out[1]
rep = {'syscfg': hex(syscfg), 'l2cfg_field_16_18': (syscfg >> 16) & 7, 'ccr': hex(out[2]), 'ssr': hex(out[3]),
       'cfgbase': hex(out[4]), 'rev': hex(out[5]),
       'cfg_table': {NAMES.get(i*4, hex(i*4)): hex(out[16+i]) for i in range(32)}}
print(json.dumps(rep, indent=1))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(rep, indent=1) + '\n')
