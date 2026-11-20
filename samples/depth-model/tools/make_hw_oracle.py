"""Write recovered/<model>/hw_<case>.json: the silicon-rounding oracle (V75 HMX
conversion, NPRG2_HW_ROUNDING=1) from the host scalar reference, in the format
of emulator_<case>.json. Run from depth-model after building .work/full/test_hw:
  OUT=.work/full/test_hw EXTRA=-DNPRG2_HW_ROUNDING=1 sh .work/merge/build_host.sh
"""

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

import json, os, re, subprocess, sys
from pathlib import Path
ROOT = (PATHS.model_data)
CASES = {'smoke': str(PATHS.native/'build/native-smoke'), 'random': str(PATHS.native/'build/depth-inputs/random')}
for model in ('primary', 'alternate'):
    names = json.loads(((PATHS.model_data / f'recovered/{model}/emulator_random.json')).read_text())  # has bytes for both models
    by_bytes = {v['bytes']: k for k, v in names.items()}
    for case, inputs in CASES.items():
        env = dict(os.environ, QD_EXPECT='print', QD_INPUTS=str(ROOT / inputs))
        out = subprocess.run([str((PATHS.model_data / '.work/full/test_hw')), model, 'reference', 'whole'], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout
        sha = dict(re.findall(r'whole output (\d) sha256 ([0-9a-f]{64})', out))
        assert len(sha) == 2, out[-500:]
        doc = {by_bytes[163840]: {'sha256': sha['0'], 'bytes': 163840},
               by_bytes[655360]: {'sha256': sha['1'], 'bytes': 655360}}
        dst = (PATHS.model_data / f'recovered/{model}/hw_{case}.json')
        dst.write_text(json.dumps(doc, indent=2) + '\n')
        print(model, case, sha['0'][:16], sha['1'][:16], '->', dst.relative_to(ROOT))
