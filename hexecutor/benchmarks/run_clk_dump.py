#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run clk_dump.elf once and decode the Turing clock registers it read."""

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
TURING_CC, Q6_CC = 0x32008000, 0x32348000
# (name, kind, address) in clk_dump.c order; rcg entries take two words (CMD, CFG).
FIELDS = [
    ('turing_cc+0x404 q6ss_q6_axim', 'branch', TURING_CC + 0x404),
    ('turing_cc+0x22c (Valve NSPNOC)', 'branch', TURING_CC + 0x22c),
    ('turing_cc+0x414 (Valve AHBS_AON)', 'branch', TURING_CC + 0x414),
    ('q6_cc+0x20 core', 'rcg', Q6_CC + 0x20),
    ('q6_cc+0x180 hmx', 'rcg', Q6_CC + 0x180),
    ('turing_cc+0x248 nspnoc', 'branch', TURING_CC + 0x248),
    ('turing_cc+0x410 q6ss_ahbs_aon', 'branch', TURING_CC + 0x410),
    ('turing_cc+0x254 ceng/q6_axim', 'rcg', TURING_CC + 0x254),
    ('turing_cc+0x25c ceng_nsp', 'branch', TURING_CC + 0x25c),
    ('turing_cc+0x264 ceng_proc', 'branch', TURING_CC + 0x264),
    ('turing_cc+0x124 dma_nspnoc', 'rcg', TURING_CC + 0x124),
    ('turing_cc+0x270 dma_nspnoc', 'branch', TURING_CC + 0x270),
    ('turing_cc+0x130 vapss_q6_axi', 'branch', TURING_CC + 0x130),
]
# Raw words read after FIELDS: (label, address).
PLL_REGS = ('MODE', 'OPMODE', 'STATE', 'STATUS', 'L_VAL', 'ALPHA_VAL', 'USER_CTL', 'USER_CTL_U')
RAW = ([('q6_axim CMD_DFSR', TURING_CC + 0x254 + 0x14)] +
       [(f'q6_axim DFS level {i} CFG', TURING_CC + 0x254 + 0x1c + 4 * i) for i in range(8)] +
       [(f'core{p}_pll {r}', 0x32000000 + 0x1000 * p + 4 * i)
        for p in range(4) for i, r in enumerate(PLL_REGS)])


def decode(fields, words):
    rows, i = [], 0
    for name, kind, _ in fields:
        if kind == 'rcg':
            cmd, cfg = words[i], words[i + 1]
            i += 2
            row = {'name': name, 'cmd': f'{cmd:#x}', 'root_off': bool(cmd >> 31),
                   'cfg': f'{cfg:#x}', 'src': (cfg >> 8) & 7, 'div': ((cfg & 0x1f) + 1) / 2,
                   'xo': (cfg >> 8) & 7 == 0, 'hw_ctrl': bool(cfg >> 20 & 1)}
        else:
            w = words[i]
            i += 1
            row = {'name': name, 'cbcr': f'{w:#x}', 'enabled': bool(w & 1), 'clk_off': bool(w >> 31)}
        rows.append(row)
    return rows


def main():
    build = (PATHS.native / 'build')
    nfield = sum(2 if k == 'rcg' else 1 for _, k, _ in FIELDS)
    nwords = nfield + len(RAW)
    with FrameTransport(build / 'qd_loader.elf', build / 'clk_dump.elf', 10) as t:
        def prepare(seq):
            t.arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
            t.arena[DUMP:DUMP + 4 * (1 + nwords)] = bytes(4 * (1 + nwords))
            struct.pack_into('<I', t.arena, 8, seq)
        last, *words = t.invoke(prepare, t.workers_done,
                                lambda: struct.unpack_from(f'<{1 + nwords}I', t.arena, DUMP))
    done = last == 0xffffffff
    # A stalled read leaves out[0] at its address (an RCG CFG is CMD + 4);
    # decode only the fields read completely before it.
    stall = next((i for i, (_, k, a) in enumerate(FIELDS)
                  if a <= last <= a + (4 if k == 'rcg' else 0)), len(FIELDS))
    fields = FIELDS if done else FIELDS[:stall]
    raw_stall = next((i for i, (_, a) in enumerate(RAW) if a == last), len(RAW))
    raw = RAW if done else RAW[:raw_stall] if stall == len(FIELDS) else []
    report = {'finished': done, 'stalled_at': None if done else f'{last:#x}',
              'registers': decode(fields, words),
              'raw': {label: f'{w:#010x}' for (label, _), w in zip(raw, words[nfield:])}}
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
