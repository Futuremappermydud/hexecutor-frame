"""Write recovered/<model>/hw_<case>.json: the silicon-rounding oracle (V75 HMX
conversion, NPRG2_HW_ROUNDING=1) from the host scalar reference, in the format
of emulator_<case>.json. Run from depth-model after building .work/full/test_hw:
  OUT=.work/full/test_hw EXTRA=-DNPRG2_HW_ROUNDING=1 sh .work/merge/build_host.sh
"""
import json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CASES = {'smoke': 'frame/build/native-smoke', 'random': 'frame/build/depth-inputs/random'}
for model in ('primary', 'alternate'):
    names = json.loads((ROOT / f'recovered/{model}/emulator_random.json').read_text())  # has bytes for both models
    by_bytes = {v['bytes']: k for k, v in names.items()}
    for case, inputs in CASES.items():
        env = dict(os.environ, QD_EXPECT='print', QD_INPUTS=str(ROOT / inputs))
        out = subprocess.run([str(ROOT / '.work/full/test_hw'), model, 'reference', 'whole'], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout
        sha = dict(re.findall(r'whole output (\d) sha256 ([0-9a-f]{64})', out))
        assert len(sha) == 2, out[-500:]
        doc = {by_bytes[163840]: {'sha256': sha['0'], 'bytes': 163840},
               by_bytes[655360]: {'sha256': sha['1'], 'bytes': 655360}}
        dst = ROOT / f'recovered/{model}/hw_{case}.json'
        dst.write_text(json.dumps(doc, indent=2) + '\n')
        print(model, case, sha['0'][:16], sha['1'][:16], '->', dst.relative_to(ROOT))
