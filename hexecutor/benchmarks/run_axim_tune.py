#!/usr/bin/env python3
"""Run axim_tune.elf once: HVX DDR read bandwidth with the Q6 AXI master at XO,
raised (750 MHz, or 1000 MHz for axim_tune_1000.elf) and restored.

  run_axim_tune.py [REPORT.json] [PAYLOAD.elf]
"""

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
import json
import struct
import sys

HERE = (PATHS.benchmarks)
sys.path.insert(0, str((PATHS.native)))
from hexecutor.offline.frame_transport import FrameTransport
from hexecutor.offline.hvx_protocol import STATUS_OFFSET

DUMP = 0x10000
READ_BYTES = 1 << 20
CORE_HZ = 1843.2e6
FIELDS = ('progress', 'error', 'xo_cycles', 'cfg_before', 'pll_mode_locked', 'cfg_raised',
          'cmd_raised', 'raised_cycles', 'cfg_restored', 'pll_mode_restored', 'restored_cycles', 'sink',
          'xo_overlap_cycles', 'raised_overlap_cycles', 'restored_overlap_cycles')
ERRORS = {0: None, 2: 'core2 PLL already in use', 3: 'PLL lock timeout',
          4: 'RCG switch timeout', 5: 'RCG restore timeout'}


def gbps(cycles):
    return round(READ_BYTES / (cycles / CORE_HZ) / 1e9, 3) if cycles else None


def main():
    build = (PATHS.native / 'build')
    with FrameTransport(build / 'qd_loader.elf', build / (sys.argv[2] if len(sys.argv) > 2 else 'axim_tune.elf'), 20) as t:
        def prepare(seq):
            t.arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
            t.arena[DUMP:DUMP + 4 * len(FIELDS)] = bytes(4 * len(FIELDS))
            struct.pack_into('<I', t.arena, 8, seq)
        words = t.invoke(prepare, t.workers_done,
                         lambda: struct.unpack_from(f'<{len(FIELDS)}I', t.arena, DUMP))
    r = dict(zip(FIELDS, words))
    report = {k: (f'{v:#x}' if k.startswith(('cfg', 'cmd', 'pll')) else v) for k, v in r.items()}
    report['error_text'] = ERRORS.get(r['error'], r['error'])
    report['finished'] = r['progress'] == 0xffffffff
    for k in ('xo', 'raised', 'restored'):
        report[f'{k}_gbps_serial'] = gbps(r[f'{k}_cycles'])
        report[f'{k}_gbps_overlap'] = gbps(r[f'{k}_overlap_cycles'])
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
