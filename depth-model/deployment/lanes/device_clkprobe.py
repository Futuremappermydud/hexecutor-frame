#!/usr/bin/env python3
"""Run a QDS_CLK_PROBE engine on smoke and print the clock registers read during the run.
usage: device_clkprobe.py NAME (from the frame deploy root)"""
import json, struct, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO/'depth-model/python'), str(REPO/'depth-model/tools'), str(REPO/'hexecutor/benchmarks')]
from hexecutor.paths import workspace
PATHS = workspace()

name = sys.argv[1]
import verify_prepared_full_model as v
from pathlib import Path
orig = v.cases
def cases(model, only, expect):
    names, todo = orig(model, only, expect)
    h = json.loads((PATHS.native/'build/ab/investigation-stream-real-20261105/host_hashes.json').read_text())
    return names, [(l, d, {n + '.raw': h[l][i] for i, n in enumerate(names)}) for l, d, e in todo]
v.cases = cases
snap = []
run0 = v.ResidentDSP.run
def run(self, *a, **k):
    r = run0(self, *a, **k)
    snap.append(struct.unpack('<50I', bytes(self.mem[0x7810000:0x7810000 + 200])))
    return r
v.ResidentDSP.run = run
sys.argv = ['x', '--model', 'primary', '--repeat', '3', '--case', 'smoke', '--engine', str(PATHS.native/f'build/ab/planes-20261105/eng-{name}.elf'), '--bootstrap', str(PATHS.native/f'build/ab/planes-20261105/boot-{name}.elf'), '--prepared-program', str(PATHS.model_deployment/'programs/primary-v2.nprg'), '--report', f'/tmp/clk-{name}.json']
try: v.main()
except SystemExit: pass
s = snap[-1]
L = ['tcc+404 axim_br', 'tcc+22c nspnoc?', 'tcc+414 ahbs?', 'tcc+248 nspnoc_br', 'tcc+410 ahbs_br', 'tcc+254 AXIM_RCG_CMD', 'tcc+258 AXIM_RCG_CFG', 'tcc+25c ceng_nsp', 'tcc+264 ceng_proc', 'tcc+124 dma_rcg_cmd', 'tcc+128 dma_rcg_cfg', 'tcc+270 dma_br', 'tcc+130 vapss', 'q6+20 CORE_CMD', 'q6+24 CORE_CFG', 'q6+180 HMX_CMD', 'q6+184 HMX_CFG', 'q6+1a0 HMX_BR', 'mxu_pll+0', 'mxu_pll+4', 'mxu_pll+10 L', 'mxu_pll+14 ALPHA', 'core2_pll+0', 'core2_pll+4', 'core2_pll+10 L', 'core2_pll+14 ALPHA']
for i, n in enumerate(L): print(f'{n:24s} {s[i]:#010x}')
TK = 5.425347222222222e-07
print('uncached DDR 256 KiB ms', [round(s[40 + r] * TK, 3) for r in range(3)], 'GB/s', [round(0.262144 / (s[40 + r] * TK) , 2) for r in range(3)])
print('VTCM 256 KiB ms', round(s[48] * TK, 3), 'GB/s', round(0.262144 / (s[48] * TK), 2))
