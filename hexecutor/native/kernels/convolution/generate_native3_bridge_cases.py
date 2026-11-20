#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Emit the exact T/C shape and storage metadata for five real QDRX ops."""

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

from pathlib import Path
import sys

ROOT=(PATHS.repo)
sys.path[:0]=[str((PATHS.native)),str((PATHS.model_tools))]
from depth_model.lowering.resident_image import unpack

SEQUENCES=(247,874,3589,3527,5019)
IMAGE=(PATHS.native / 'build/resident-primary.qdrx')
OUT=(PATHS.native / 'kernels/convolution/native3_bridge_cases_generated.h')
DTYPE={"uint8":1,"int8":2,"int32":5}
FLAG={"CB":2,"DIRECT":4}

def cls(name):
    return int.from_bytes(name.encode()[:4].ljust(4,b"\0"),"little")

def array(v):
    return "{"+",".join(str(int(x)) for x in v)+"}"

def main():
    graph=unpack(IMAGE.read_bytes());tensors={t["id"]:t for t in graph["tensors"]}
    commands={c["sequence"]:c for c in graph["commands"]}
    rows=[]
    for seq in SEQUENCES:
        c=commands[seq];x,w,fi=(tensors[i] for i in c["inputs"][:3]);y=tensors[c["outputs"][0]]
        items=(x,w,fi,y)
        row=[]
        for t in items:
            shape=t["shape"];storage=t["storage"]
            if t["class_name"] in ("CB","cb"):
                flags=2;pool=1;offset=0;blocks=len(storage.get("blocks",[]));slice_=storage.get("slice_offset",0)
            else:
                flags=4;direct=storage["direct"];pool=direct["pool"];offset=direct["offset"];blocks=0;slice_=0
            row.append({"dtype":DTYPE[t["dtype"]],"flags":flags,"esize":t["element_bytes"],
                "crouton":1 if t["class_name"] in ("CB","cb") else 0,"cls":cls(t["class_name"]),
                "d":shape["dimensions"],"m":shape["max_dimensions"],"p":shape["padding"],
                "pool":pool,"offset":offset,"nblocks":blocks,"slice":slice_,
                "zp":t.get("quantization",{}).get("zero_point",0)})
        rows.append((seq,row))
    text=["/* Generated from primary QDRX; do not hand-edit. */",
          "static const uint32_t k_bridge_sequences[5]={"+",".join(str(s) for s,_ in rows)+"};",
          f"static const uint32_t k_bridge_pool1_bytes={int(graph['pool_bytes']['1'])}u;",
          "static const uint32_t k_bridge_metadata[5][4][22]={"]
    for _,row in rows:
        vals=[]
        for t in row:
            vals += [t["dtype"],t["flags"],t["esize"],t["crouton"],t["cls"],*t["d"],*t["m"],*t["p"],t["pool"],t["offset"],t["nblocks"],t["slice"],t["zp"]]
        assert len(vals)==4*22
        text.append("  {"+",".join("{"+",".join(map(str,vals[i:i+22]))+"}" for i in range(0,len(vals),22))+"},")
    text += ["};",""]
    OUT.write_text("\n".join(text))
    print(f"generated={OUT} sequences={','.join(str(s) for s,_ in rows)}")

if __name__=="__main__":main()
