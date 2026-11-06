"""Report exact differences; this calibration is not a passing backend gate."""
from pathlib import Path
import json
import numpy as np
BUILD=Path(__file__).resolve().parents[1]/'build'
r=np.load(BUILD/'native-hmx-expected.npz');meta=r['meta'];want=r['output']
got=np.fromfile(BUILD/'native-hmx-output.bin',np.uint8).reshape(-1,16,32,4).transpose(0,1,3,2).reshape(-1,64,32)
if got.shape!=want.shape:raise ValueError('incomplete simulator output')
rows=[]
for mode in np.unique(meta[:,2]):
    p=meta[:,2]==mode;d=got[p].astype(np.int16)-want[p]
    row={'mode':int(mode),'different':int(np.count_nonzero(d)),'total':d.size,'min':int(d.min()),'max':int(d.max())}
    row['exact']=row['different']==0;rows.append(row);print(json.dumps(row))
(BUILD/'native-hmx-conversion-validation.json').write_text(json.dumps({'purpose':'direct native conversion calibration; no hardware/backend integration','modes':rows},indent=2)+'\n')
