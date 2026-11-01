#!/usr/bin/env python3
"""Capture each Sign command's physical CPU pool prefix and changed pool-1 pages."""
from __future__ import annotations
import argparse, struct, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"frame"),str(ROOT/"tools")]
from native_graph import NativeGraph
import resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases

def capture(model: str, case: str, outdir: Path) -> int:
    image=ROOT/f"frame/build/resident-{model}.qdrx"
    decoded=resident_image.unpack(image.read_bytes())
    targets=[(i,c) for i,c in enumerate(decoded["commands"]) if c["opcode"]==30]
    variant_ids={name:n for name,n in resident_image.VARIANTS.items()}
    if len(targets)!=13: raise ValueError(f"expected 13 Sign commands in {model}, got {len(targets)}")
    case_dir=next(row[1] for row in cases(model,[case])[1] if row[0]==case)
    inputs=load_inputs(case_dir)
    graph=NativeGraph(image,ROOT/f"recovered/{model}/constants/constant_pool_5.bin")
    try:
        graph.use_cpu(); graph.set_hoist(False); graph.set_block_output(False)
        graph.run(inputs,0,targets[0][0])
        outdir.mkdir(parents=True,exist_ok=True)
        for index,cmd in targets:
            before=[graph.pools[p].copy() if p in graph.pools else np.zeros(0,np.uint8) for p in range(7)]
            graph.run(inputs,index,index+1)
            after=[graph.pools[p].copy() if p in graph.pools else np.zeros(0,np.uint8) for p in range(7)]
            for p in range(7):
                if p!=1 and not np.array_equal(before[p],after[p]): raise ValueError(f"Sign command {index} changed pool {p}")
            changes=np.flatnonzero(before[1]!=after[1])
            pages=np.unique(changes//2048)*2048 if changes.size else np.zeros(0,dtype=np.uint32)
            if any(int(x)+2048>len(after[1]) for x in pages): raise ValueError("truncated changed page")
            path=outdir/f"{model}-{case}-sign-{index}.epcpu"
            with path.open("wb") as f:
                f.write(struct.pack("<4s16I",b"EPCP",1,30,variant_ids[cmd["variant"]],cmd["sequence"],index,
                                    0xffffffff,0,0,len(pages),*(len(p) for p in before)))
                for pool in before:f.write(pool.tobytes())
                for page in pages:
                    f.write(struct.pack("<I",int(page)))
                    f.write(after[1][int(page):int(page)+2048].tobytes())
            print(f"{model}/{case}: command={index} variant={variant_ids[cmd['variant']]} pages={len(pages)} fixture={path}",flush=True)
            if index+1<len(decoded["commands"]): graph.run(inputs,index+1,next((j for j,_ in targets if j>index),len(decoded["commands"])))
        return len(targets)
    finally:graph.close()

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--case",default="smoke");ap.add_argument("--out",type=Path,default=ROOT/"frame/native_program/fixtures")
    a=ap.parse_args();n=0
    for model in ("primary","alternate"):n+=capture(model,a.case,a.out)
    print(f"captured {n} Sign CPU prefixes")
