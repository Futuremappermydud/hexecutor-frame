#!/usr/bin/env python3
"""Capture the first production Quantize@FH.Ff as a guarded CPU-prefix fixture."""
from __future__ import annotations
import argparse, hashlib, json, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'frame'), str(ROOT/'tools')]
from native_graph import NativeGraph
import resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

VARIANT = 'Quantize@FH.Ff'

def capture(model: str, case: str, outdir: Path) -> None:
    image = ROOT/f'frame/build/resident-{model}.qdrx'
    q = resident_image.unpack(image.read_bytes())
    hits = [(i,c) for i,c in enumerate(q['commands']) if c['opcode']==28 and c['variant']==VARIANT]
    if not hits: raise ValueError(f'{model}: missing {VARIANT}')
    index,c = hits[0]
    ts = {t['id']:t for t in q['tensors']}
    if len(c['inputs']) != 1 or len(c['outputs']) != 1 or len(c['calls']) != 1:
        raise ValueError('Quantize command arity changed')
    x,y = (ts[c['inputs'][0]], ts[c['outputs'][0]])
    if x['dtype']!='float32' or x['class_name']!='Ff' or y['dtype']!='uint16' or y['class_name']!='FH':
        raise ValueError('Quantize type/class profile changed')
    if x['shape'] != y['shape'] or x['shape']['padding'] != [0]*4:
        raise ValueError('Quantize shape/padding profile changed')
    if 'direct' not in x['storage'] or 'direct' not in y['storage']:
        raise ValueError('Quantize tensors must be direct storage')
    if x['storage']['direct']['pool'] != 1 or y['storage']['direct']['pool'] != 1:
        raise ValueError('Quantize tensors must reside in pool 1')
    call = c['calls'][0]
    src_off=x['storage']['direct']['offset']; dst_off=y['storage']['direct']['offset']
    count=1
    for d in x['shape']['dimensions']: count*=d
    if call[0] != 'quantize_sf_to_u16' or call[1][1] != y['quantization']['zero_point'] or call[1][3] != count:
        raise ValueError('Quantize leaf-call count/zero-point differs from tensor metadata')
    src_bytes=count*4; dst_bytes=count*2
    if src_off < dst_off+dst_bytes and dst_off < src_off+src_bytes: raise ValueError('input/output alias')
    case_dir=next(row[1] for row in cases(model,[case])[1] if row[0]==case)
    graph=NativeGraph(image,ROOT/f'recovered/{model}/constants/constant_pool_5.bin')
    try:
        graph.use_cpu(); graph.set_hoist(False); graph.set_block_output(False)
        graph.run(load_inputs(case_dir),0,index)
        before=graph.pools[1].copy()
        source=before[src_off:src_off+src_bytes].tobytes()
        graph.run(load_inputs(case_dir),index,index+1)
        expected=graph.pools[1][dst_off:dst_off+dst_bytes].tobytes()
        outdir.mkdir(parents=True,exist_ok=True)
        stem=f'{model}-{case}'
        (outdir/f'{stem}.f32').write_bytes(source)
        (outdir/f'{stem}.u16').write_bytes(expected)
        (outdir/f'{stem}.pool1').write_bytes(before.tobytes())
        meta={'format':'NQFX1','model':model,'case':case,'qdrx_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
              'command_index':index,'sequence':c['sequence'],'opcode':c['opcode'],'variant':VARIANT,
              'variant_id':resident_image.VARIANTS[VARIANT],
              'input_tensor':x['id'],'output_tensor':y['id'],'shape':x['shape']['dimensions'],
              'input_pool':1,'input_offset':src_off,'input_bytes':src_bytes,
              'output_pool':1,'output_offset':dst_off,'output_bytes':dst_bytes,
              'zero_point':y['quantization']['zero_point'],'scale_f32':y['quantization']['scale'],
              'inverse_scale_f32':struct.unpack('<f',struct.pack('<f',1.0/y['quantization']['scale']))[0],
              'leaf_call':call,'source_sha256':hashlib.sha256(source).hexdigest(),
              'expected_sha256':hashlib.sha256(expected).hexdigest(),
              'pool1_before_sha256':hashlib.sha256(before.tobytes()).hexdigest(),
              'pool1_bytes':before.nbytes,'non_output_pool1_unchanged':bool((before[:dst_off]==graph.pools[1][:dst_off]).all() and (before[dst_off+dst_bytes:]==graph.pools[1][dst_off+dst_bytes:]).all())}
        (outdir/f'{stem}.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f"{model}/{case}: command={index} seq={c['sequence']} count={count} scale={meta['scale_f32']:.17g} zp={meta['zero_point']} source={meta['source_sha256']} output={meta['expected_sha256']}")
    finally: graph.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',choices=('smoke','random'),default='smoke'); ap.add_argument('--out',type=Path,default=Path(__file__).parent/'fixtures'); a=ap.parse_args()
    for model in ('primary','alternate'): capture(model,a.case,a.out)
if __name__=='__main__': main()
