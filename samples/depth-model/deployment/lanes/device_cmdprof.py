#!/usr/bin/env python3
"""Run device_check with a QDS_LANE_CMDPROF engine and save per-command handler time.
usage: device_cmdprof.py OUT.json <device_check args>  (from the frame deploy root)"""
import json, statistics, struct, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO/'samples/depth-model/python'), str(REPO/'samples/depth-model/tools'), str(REPO/'hexecutor/benchmarks')]
from hexecutor.paths import workspace
PATHS = workspace()

out = sys.argv.pop(1)
import verify_prepared_full_model as v
all_cases = '--all-cases' in sys.argv
if all_cases: sys.argv.remove('--all-cases')
from pathlib import Path
original = v.cases
def cases(model, only, expect):
    names, todo = original(model, None if all_cases else only, expect)
    hashes = json.loads((PATHS.native/'build/ab/investigation-stream-real-20261105/host_hashes.json').read_text())
    return names, [(label, d, {n + '.raw': hashes[label][i] for i, n in enumerate(names)}) for label, d, e in todo]
v.cases = cases
samples = []
orig_run = v.ResidentDSP.run
def run(self, *a, **k):
    r = orig_run(self, *a, **k)
    raw = bytes(self.mem[0x7800000:0x7800000 + 5068 * 4])
    samples.append(struct.unpack('<5068I', raw))
    hp = self.handler_profile
    samples_ms.append(sum(x['run_ms'] + x['validate_ms'] for x in hp))
    return r
samples_ms = []
v.ResidentDSP.run = run
v.main()
warm = samples[1:]; wms = samples_ms[1:]
ticks = [sum(s) for s in warm]
scale = statistics.median(m / t for m, t in zip(wms, ticks) if t)
per = [statistics.median(s[i] for s in warm) * scale for i in range(5068)]
json.dump(per, open(out, 'w'))
print('samples', len(warm), 'sum ms', round(sum(per), 3), 'handler sum', round(statistics.median(wms), 3), 'ms/tick', scale, 'zero', sum(1 for x in per if x == 0))
