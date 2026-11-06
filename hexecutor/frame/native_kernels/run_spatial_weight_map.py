#!/usr/bin/env python3
"""Run bounded one-hot HMX spatial tap probes for setter-derived tuple."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / '.work/hexagon-sdk/6.6.0.0'
FRAME = ROOT / 'frame'
BIN = 'native_kernels/qnn_hmx_spatial_field_probe_sim.elf'
CLANG = '/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang'
SIM = '/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim'
BASE = ['docker', 'run', '--rm', '--platform', 'linux/amd64',
        '-v', f'{SDK}:/sdk:ro', '-v', f'{FRAME}:/src', '-w', '/src',
        'depth-model-hmx-tools:latest']
COMMON = ['-mv75', '-mhmx', '-mhvx', '-mhvx-length=128b', '-O2', '-G0',
          '-ffreestanding', '-fno-builtin', '-fno-stack-protector', '-Wall', '-Wextra',
          '-DKH=3', '-DKW=3', '-DSETTER_R3=2', '-DSETTER_R5=6',
          '-DSETTER_FLAGS=0x700', '-DSETTER_ARG7=8']
SOURCES = ['native_kernels/qnn_hmx_spatial_field_probe_sim.c', 'qd_hmx_ops.S',
           'native_kernels/qd_qnn_hmx_3x3.S', 'native_kernels/qd_set_hmx_params_stride1.S']
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
    if 'FAIL completion=' in p.stdout:
        print(re.search(r'FAIL completion=[^\n]+', p.stdout).group(0))
