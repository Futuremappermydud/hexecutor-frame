#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run qwen_ops_test.elf once: HVX Qwen phases against their scalar references."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
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


def f32(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


def main():
    build = (PATHS.native / 'build')
    with FrameTransport(build / 'qd_loader.elf', build / 'qwen_ops_test.elf', 30) as t:
        def prepare(seq):
            t.arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
            t.arena[DUMP:DUMP + 352] = bytes(352)
            struct.pack_into('<I', t.arena, 8, seq)
        w = t.invoke(prepare, t.workers_done, lambda: struct.unpack_from('<88I', t.arena, DUMP))
    signed = lambda u: u - (1 << 32) if u >> 31 else u
    report = {
        'finished': w[0] == 0xffffffff,
        'axi_raise_error': w[7],
        'rmsnorm': {'inexact': w[8], 'max_rel_err': f32(w[9])},
        'residual_add': {'inexact': w[16], 'max_rel_err': f32(w[17])},
        'quantize': {'code_mismatches': w[24], 's8_mismatches': w[25], 'scale_mismatches': w[26],
                     'first_mismatch': None if w[27] == 0xffffffff else
                     {'index': w[27], 'ref': signed(w[28]), 'hvx': signed(w[29]), 'x': f32(w[30])}},
        'silu': {'inexact': w[32], 'max_rel_err': f32(w[33])},
        'rope_head': {'inexact': w[40], 'max_rel_err': f32(w[41])},
        'argmax_failures': w[48],
        'softmax': {'inexact': w[52], 'max_rel_err': f32(w[53]), 'nonzero_tail': w[54]},
        'cycles': {'quantize_3072 (hvx = one worker)': [w[56], w[57]], 'rmsnorm_1024': [w[58], w[59]],
                   'silu_3072': [w[60], w[61]], 'order': '[scalar, hvx]'},
        'qk_block_cycles': {'ddr_cdsp_mem': w[80], 'l2_tcm': w[81], 'vtcm': w[82], 'stage_ddr_to_l2_copy': w[83]},
    }
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
