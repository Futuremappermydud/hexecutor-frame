#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline capacity proof for per-call direct3 A staging; never accesses device."""

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

import json, sys
from pathlib import Path
ROOT=(PATHS.repo)
sys.path.insert(0,str((PATHS.native)))
from depth_model.lowering.resident_image import unpack
PAGE=2048; TAIL=512*1024; CAP=251  # usable page budget after tail runtime reservations
census=json.loads(((PATHS.native / 'kernels/convolution/qd_native_3x3_graph_eligibility_census.json')).read_text())
for model in ('primary','alternate'):
    graph=unpack(((PATHS.native / f'build/resident-{model}.qdrx')).read_bytes())
    ts={t['id']:t for t in graph['tensors']}; eligible=set(census[model]['eligible_sequence_ids'])
    rows=[]
    for ci,c in enumerate(graph['commands']):
        if c.get('sequence') not in eligible: continue
        x=ts[c['inputs'][0]]; shape=x['shape']; maxd=shape['max_dimensions']
        # 3x3 adapter table follows [maxH/8, maxW/8, CI/32] pages.
        w=ts[c['inputs'][1]]['shape']['max_dimensions']; pages=(maxd[1]//8)*(maxd[2]//8)*(w[2]//32)
        blocks=x['storage']['blocks'][x['storage'].get('slice_offset',0):][:pages]
        refs=[(b['pool'],b['offset']) for b in blocks]
        assert len(refs)==pages and all(p==1 and o%PAGE==0 for p,o in refs), (model,c['sequence'])
        unique=set(refs)
        rows.append(dict(index=ci,sequence=c['sequence'],refs=pages,unique=len(unique),aliases=pages-len(unique),bytes=len(unique)*PAGE))
    assert len(rows)==len(eligible)
    worst=max(rows,key=lambda r:r['unique'])
    over=[r for r in rows if r['unique']>CAP]
    print(json.dumps(dict(model=model,commands=len(rows),capacity_pages=CAP,capacity_bytes=CAP*PAGE,physical_tail_bytes=TAIL,
      max=worst,over_capacity=len(over),max_refs=max(r['refs'] for r in rows),total_copy_bytes=sum(r['bytes'] for r in rows),
      total_copy_mib=round(sum(r['bytes'] for r in rows)/(1024*1024),2),
      unique_histogram={str(n):sum(r['unique']==n for r in rows) for n in sorted({r['unique'] for r in rows})}),indent=2))
