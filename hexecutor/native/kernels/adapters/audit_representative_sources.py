#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit exact raw weight/FI sources for representative compiled QDRX 1x1s."""

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

import argparse, hashlib, json, sys
from pathlib import Path
FRAME=(PATHS.native)
sys.path.insert(0,str(FRAME))
import depth_model.lowering.resident_image as resident_image
SAMPLES=(1195,2380,4136,4137,4151,4154,4169,4173,4269)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',type=Path,default=(PATHS.native / 'build/resident-primary.qdrx'))
    ap.add_argument('--constants-dir',type=Path,default=(PATHS.model_data / 'recovered/primary/constants'))
    ap.add_argument('--output',type=Path,default=(PATHS.native / 'kernels/adapters/representative-source-audit.json'))
    a=ap.parse_args(); raw=a.image.read_bytes(); graph=resident_image.unpack(raw)
    tensors={t['id']:t for t in graph['tensors']}; producers={tid:c for c in graph['commands'] for tid in c['outputs']}
    pools={5:(a.constants_dir/'constant_pool_5.bin').read_bytes()}
    rows=[]
    for seq in SAMPLES:
        c=next(c for c in graph['commands'] if c['sequence']==seq)
        x,w,fi=(tensors[i] for i in c['inputs'][:3]); y=tensors[c['outputs'][0]]
        def source(t):
            p=producers.get(t['id'])
            if p is None:return t,None
            return tensors[p['inputs'][0]],p
        ws,wp=source(w); fs,fp=source(fi)
        def raw_bytes(t):
            ref=t['storage'].get('direct')
            if not ref:return None
            pool,off=ref['pool'],ref['offset']; n=1
            for d in t['shape']['dimensions']:n*=d
            n*=t['element_bytes']; buf=pools.get(pool)
            if buf is None:return None
            if off+n>len(buf):raise ValueError(f'tensor {t["id"]} source outside pool')
            return buf[off:off+n]
        wb,fb=raw_bytes(ws),raw_bytes(fs)
        rows.append({'sequence':seq,'variant':c['variant'],'input_id':x['id'],'output_id':y['id'],
          'input_hw':x['shape']['dimensions'][1:3],'output_hw':y['shape']['dimensions'][1:3],
          'input_max_hw':x['shape']['max_dimensions'][1:3],'output_max_hw':y['shape']['max_dimensions'][1:3],
          'ci':w['shape']['dimensions'][2],'co':w['shape']['dimensions'][3],
          'weight_tensor':w['id'],'weight_source_tensor':ws['id'],'weight_source':ws['storage'].get('direct'),
          'weight_producer':wp['variant'] if wp else None,'weight_bytes':len(wb) if wb else None,
          'weight_sha256':hashlib.sha256(wb).hexdigest() if wb else None,
          'fi_tensor':fi['id'],'fi_source_tensor':fs['id'],'fi_source':fs['storage'].get('direct'),
          'fi_producer':fp['variant'] if fp else None,'fi_bytes':len(fb) if fb else None,
          'fi_sha256':hashlib.sha256(fb).hexdigest() if fb else None,
          'slice_offsets':[x['storage'].get('slice_offset',0),y['storage'].get('slice_offset',0)],
          'static_raw_weight_fi':wb is not None and fb is not None})
    report={'qdrx_image':str(a.image),'qdrx_sha256':hashlib.sha256(raw).hexdigest(),
            'constant_pool_sha256':{str(i):hashlib.sha256(p).hexdigest() for i,p in pools.items()},
            'samples':rows,'note':'Command 2380 Fi is runtime-generated; source audit records its producer but does not claim raw final Fi validation.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for r in rows:print(f"{r['sequence']}: {r['input_hw']} {r['variant']} slices={r['slice_offsets']} rawFi={r['static_raw_weight_fi']}")
    print(f'wrote {a.output}')
if __name__=='__main__':main()
