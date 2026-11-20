# SPDX-License-Identifier: LGPL-3.0-or-later
"""Count graph 1x1 shapes eligible by the proven bbb aligned leaf geometry."""

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

from collections import Counter
import json
from pathlib import Path
root=(PATHS.repo)
graph=json.loads(((PATHS.model_data / 'recovered/primary/graph.json')).read_text())
tensors={t['id']:t for t in graph['tensors']}
counts=Counter()
for node in graph['nodes']:
    if 'ConvLayer_s1.opt' not in node['type_name'] or len(node['inputs'])<2 or not node['outputs']:
        continue
    x,w,y=(tensors.get(i,{}) for i in (node['inputs'][0],node['inputs'][1],node['outputs'][0]))
    xd=x.get('shape',{}).get('dimensions',[]);wd=w.get('shape',{}).get('dimensions',[]);yd=y.get('shape',{}).get('dimensions',[])
    if len(wd)!=4 or wd[:2]!=[1,1]: continue
    sig=(tuple(xd),x.get('dtype'),tuple(wd),tuple(yd),y.get('dtype'))
    counts[sig]+=1
eligible=[];excluded=[]
for sig,count in counts.items():
    xd,xtype,wd,yd,ytype=sig
    # Proven leaf's bbb type is uint8 activation, s8 filter, uint8 output.
    # It requires channels divisible by 32. Graph's outer shape is NHWC.
    good=(xtype=='uint8' and ytype=='uint8' and len(xd)==4 and len(yd)==4 and
          wd[2]%32==0 and wd[3]%32==0 and xd[3]==wd[2] and yd[3]==wd[3])
    kh,kw=wd[0],wd[1]
    if len(xd)==4 and len(yd)==4:
        oh=min(((yd[1]+7)//8)*8,yd[1],xd[1]-kh+1)
        ow=min(((yd[2]+7)//8)*8,yd[2],xd[2]-kw+1)
        input_exec=[oh+kh-1,ow+kw-1]
        padded_crouton=[((oh+7)//8)*8,((ow+7)//8)*8]
    else: oh=ow=0;input_exec=[];padded_crouton=[]
    item={'count':count,'input_shape_raw':xd,'input_dtype':xtype,'filter_shape':wd,
          'output_shape_raw':yd,'output_dtype':ytype,'output_hw_after_k_hmx_conv_clamp':[oh,ow],
          'input_hw_reached_by_convolution':[input_exec[0],input_exec[1]] if input_exec else [],
          'physical_crouton_allocation_hw':[padded_crouton[0],padded_crouton[1]] if padded_crouton else []}
    (eligible if good else excluded).append(item)
report={'provider_leaf':'hmx_v73_convbbb1x1_stride1','graph':'recovered/primary/graph.json',
        'eligible_nodes':sum(i['count'] for i in eligible),'one_by_one_nodes':sum(counts.values()),
        'eligible_signatures':eligible,'excluded_signatures':excluded,
        'interpretation':'Eligibility is shape/dtype only. k_hmx_conv rounds oh up to 8 then clamps to the recorded output height and input bound; ow is rounded up to 8 then clamped. The simulator checks the entire physical 8x8-crouton allocation, including crop-external rows/columns, with poison table tails and canaries; this does not validate graph weight quantization or all runtime config branches.'}
out=(PATHS.native / 'kernels/convolution/1x1-graph-coverage.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(f"eligible 1x1 nodes={report['eligible_nodes']} / total 1x1 nodes={report['one_by_one_nodes']}")
for i in eligible: print(i['count'],i['input_shape_raw'],i['filter_shape'],i['output_shape_raw'],'effective=',i['output_hw_after_k_hmx_conv_clamp'],'footprint=',i['physical_crouton_allocation_hw'])
