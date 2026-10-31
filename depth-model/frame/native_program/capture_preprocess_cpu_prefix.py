#!/usr/bin/env python3
"""Capture CPU-prefix oracle states for the exact early InputSlice and Im2col ops."""
from __future__ import annotations
import argparse, struct, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'frame'));sys.path.insert(0,str(ROOT/'tools'))
from native_graph import NativeGraph
import resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

def unique_output_ranges(t, op):
    st=t['storage']
    if 'direct' in st:
        n=1
        for d in t['shape']['dimensions']:n*=d
        return [(st['direct']['offset'],n*4)]
    seen=set();out=[]
    for b in st['blocks']:
        if b['pool']!=1:raise ValueError('preprocess output page outside pool1')
        if b['offset'] not in seen:seen.add(b['offset']);out.append((b['offset'],2048))
    return sorted(out)

def capture(model,case,outdir):
    image=ROOT/f'frame/build/resident-{model}.qdrx';gjson=resident_image.unpack(image.read_bytes());ts={t['id']:t for t in gjson['tensors']}
    targets=[]
    for i,c in enumerate(gjson['commands']):
        if (c['opcode']==3 and c['variant']=='*InputSlice@Ff.s4*6.') or (c['opcode']==5 and c['variant']=='ConvLayer.opt.inconv_im2col.tcm@CB.FB.s4.t'):
            targets.append((i,c))
    if len(targets)<2:raise ValueError(f'missing target ops for {model}')
    # First InputSlice is command 0; first strict CB Im2col is the useful 3x3 profile.
    targets=[next(x for x in targets if x[1]['opcode']==3),next(x for x in targets if x[1]['opcode']==5)]
    constants=ROOT/f'recovered/{model}/constants/constant_pool_5.bin';case_dir=next(row[1] for row in cases(model,[case])[1] if row[0]==case);inputs=load_inputs(case_dir)
    outdir.mkdir(parents=True,exist_ok=True);graph=NativeGraph(image,constants)
    try:
        graph.use_cpu();graph.set_hoist(False);graph.set_block_output(False)
        for index,c in targets:
            graph.run(inputs,0,index)
            before={p:(graph.pools[p].copy() if p in graph.pools else bytes()) for p in range(7)}
            io_idx=c['io_index'];io_data=inputs[io_idx].tobytes() if c['opcode']==3 else b''
            graph.run(inputs,index,index+1)
            ranges=unique_output_ranges(ts[c['outputs'][0]],c['opcode'])
            path=outdir/f'{model}-{case}-op{c["opcode"]}.ppcpu'
            sizes=[len(before[p]) for p in range(7)]
            with path.open('wb') as f:
                f.write(struct.pack('<4s15I',b'PPFX',1,c['opcode'],resident_image.VARIANTS[c['variant']],c['sequence'],index,io_idx,len(io_data),len(ranges),*sizes))
                for p in range(7):f.write(before[p])
                f.write(io_data)
                for off,n in ranges:f.write(struct.pack('<II',off,n));f.write(graph.pools[1][off:off+n].tobytes())
            print(f'{model}/{case}: op={c["opcode"]} command={index} patches={len(ranges)} bytes={path.stat().st_size}',flush=True)
    finally:graph.close()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=('smoke','random'),default='smoke');ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'fixtures');args=ap.parse_args()
    for m in ('primary','alternate'):capture(m,args.case,args.out)
if __name__=='__main__':main()
