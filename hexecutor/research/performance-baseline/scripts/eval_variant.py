"""Compare a host dump variant with simulator and Model on all baseline cases."""
import json
import sys
from pathlib import Path
import numpy as np

work, variant = Path(sys.argv[1]), sys.argv[2]
reference = Path('.work/model-qnn-meta-20261030/full-validation/primary')
cases = json.loads(Path('.work/model-qnn-meta-20261030/full-validation.json').read_text())['models']['primary']['runs']
def metrics(a, b):
    assert a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    d = a.astype(np.float64) - b
    return dict(mae=float(np.abs(d).mean()), bias=float(d.mean()), p95=float(np.percentile(np.abs(d), 95)), max=float(np.abs(d).max()))
rows = {}
for case in cases:
    name = case['case']
    row = {}
    for i, (head, dtype) in enumerate((('u8', 'u1'), ('f32', '<f4'))):
        ref = np.fromfile(reference / name / f'cpu-{head}.raw', dtype)
        actual = np.fromfile(work / 'out' / variant / name / f'out{i}.raw', dtype)
        model = np.fromfile(reference / name / f'{head}.raw', dtype)
        row[head] = metrics(actual, ref)
        row['model_' + head] = metrics(model, ref)
    rows[name] = row
out = work / f'metrics_{variant}.json'
out.write_text(json.dumps(rows, indent=2) + '\n')
for metric in ('mae', 'bias', 'p95', 'max'):
    passed = sum(abs(r['f32'][metric]) <= abs(r['model_f32'][metric]) for r in rows.values())
    print(metric, 'median', np.median([abs(r['f32'][metric]) for r in rows.values()]), 'at or below Model', passed, '/', len(rows))
print(out)
