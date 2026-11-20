#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run bounded one-hot HMX spatial tap probes for setter-derived tuple."""

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
import re
import subprocess
import sys

ROOT = (PATHS.repo)
SDK = (PATHS.model_data / '.work/hexagon-sdk/6.6.0.0')
FRAME = (PATHS.native)
BIN = 'native_kernels/qnn_hmx_generic_field_probe_sim.elf'
CLANG = '/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang'
SIM = '/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim'
BASE = ['docker', 'run', '--rm', '--platform', 'linux/amd64',
        '-v', f'{SDK}:/sdk:ro', '-v', f'{FRAME}:/src', '-w', '/src',
        'depth-model-hmx-tools:latest']
COMMON = ['-mv75', '-mhmx', '-mhvx', '-mhvx-length=128b', '-O2', '-G0',
          '-ffreestanding', '-fno-builtin', '-fno-stack-protector', '-Wall', '-Wextra',
          '-DKH=3', '-DKW=3', '-DSETTER_R3=2', '-DSETTER_R5=6',
          '-DSETTER_FLAGS=0x700', '-DSETTER_ARG7=8']
SOURCES = ['native_kernels/qnn_hmx_generic_field_probe_sim.c', 'qd_hmx_ops.S',
           'native_kernels/qd_qnn_hmx_stride1_generic.S', 'native_kernels/qd_set_hmx_params_stride1.S']
for tap in range(9):
    args = COMMON + [f'-DTAPY={tap // 3}', f'-DTAPX={tap % 3}'] + SOURCES + ['-o', BIN]
    subprocess.run(BASE + [CLANG] + args, cwd=ROOT, check=True)
    p = subprocess.run(BASE + [SIM, '-mv75', '--quiet', BIN], cwd=ROOT,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    m = re.search(r'output c0:\n([^\n]+)', p.stdout)
    first = m.group(1).split() if m else []
    cfg = re.search(r'cfg=([^\n]+)', p.stdout)
    print(f'tap={tap} status={p.returncode} cfg={cfg.group(1) if cfg else "?"} row0={first}')
    if 'extra physical crouton bytes written=' in p.stdout:
        print(re.search(r'extra physical crouton bytes written=\d+', p.stdout).group(0))
    for failure in re.findall(r'FAIL [^\n]+', p.stdout):
        print(failure)
    sys.stdout.flush()
