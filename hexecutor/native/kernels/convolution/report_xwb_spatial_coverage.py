#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inventory exact 3x3 CI32/CO32 CB*2 Xwb graph footprints."""

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

import collections,hashlib,json
from pathlib import Path
ROOT=(PATHS.repo)
source=(PATHS.model_data / 'recovered/primary/graph.json');g=json.loads(source.read_text());t={x['id']:x for x in g['tensors']}
rows=[];bad=[]
for n in g['nodes']:
 if n['type_name']!='ConvLayer_s1.opt@CB*2.Xwb.Fi.t':continue
 x,w,fi=(t[i] for i in n['inputs'][:3]);y=t[n['outputs'][0]]
 if (w['shape']['dimensions'] != [3,3,32,32] or
     x['shape']['dimensions'] != [1,10,258,32] or
     y['shape']['dimensions'] != [1,8,256,32]):continue
 xb=x['storage'].get('blocks',[]);yb=y['storage'].get('blocks',[]);so=y['storage'].get('slice_offset',0)
 active=yb[so:so+32]
 inrefs={(b.get('pool'),b.get('offset')) for b in xb if isinstance(b,dict) and b.get('pool')==1}
 outrefs={(b.get('pool'),b.get('offset')) for b in active if isinstance(b,dict) and b.get('pool')==1}
 row={'sequence':n['sequence'],'input':x['id'],'weights':w['id'],'fi':fi['id'],'output':y['id'],
      'input_table_blocks':len(xb),'input_slice_offset':x['storage'].get('slice_offset',0),
      'output_table_blocks':len(yb),'output_slice_offset':so,'active_output_blocks':len(active),
      'input_output_page_aliases':len(inrefs & outrefs)}
 rows.append(row)
 if (tuple(x['shape']['max_dimensions'])!=(1,24,272,32) or tuple(x['shape']['padding'])!=(0,7,7,0) or
     x.get('quantization',{}).get('zero_point')!=0 or len(xb)!=102 or x['storage'].get('slice_offset',0)!=0 or
     tuple(y['shape']['max_dimensions'])!=(1,8,256,32) or tuple(y['shape']['padding'])!=(0,0,0,0) or
     len(active)!=32 or so+32>len(yb) or len(outrefs)!=32 or inrefs&outrefs or
     w['class_name']!='Xwb' or w['dtype']!='int8' or fi['class_name']!='Fi' or fi['dtype']!='int32'):
  bad.append(row)
report={'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'filter':'3x3 stride-1 Xwb, CB input/output, CI=CO=32',
 'matched_graph_nodes':len(rows),'parameter_pairs':{
  f"W{wid}/Fi{fid}":sum(r['weights']==wid and r['fi']==fid for r in rows)
  for wid,fid in sorted({(r['weights'],r['fi']) for r in rows})},
 'logical_input':[1,10,258,32],'input_max':[1,24,272,32],'input_origin':[0,7,7,0],
 'input_physical_pages':102,'input_physical_bytes':102*2048,'input_endcoords_for_leaf':[265,17],
 'logical_output':[1,8,256,32],'output_max':[1,8,256,32],'output_physical_pages':32,'output_bytes':32*2048,
 'validated_footprint_conditions':'Input table has 102 active pages with slice offset 0; output logical/max shape exactly 8x256x32 (32 active pages), each output table slice has all 32 addresses, output active pages are distinct and do not alias input active pages; no output padding/crop-external bytes.',
 'invalid_rows':bad,'all_footprints_pass':not bad,
 'covered_sequence_by_specific_actual_weight_FI_pair':1022,
 'coverage_scope':'Shape/footprint inventory only; simulator full-oracle proof is sequence 1022 pair W960/Fi933. Other weight/FI parameter pairs share geometry but were not numerically run.'}
out=(PATHS.native / 'kernels/convolution/xwb_spatial_ci32_graph_coverage.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if bad:raise SystemExit(1)
