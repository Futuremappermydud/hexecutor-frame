#!/usr/bin/env python3
"""Capture a CPU-prefix S2 X/W/FI and complete post-command Y page grid."""

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

import hashlib
import struct
import sys
from pathlib import Path

HERE = (PATHS.native / 'kernels/stride2')
FRAME = (PATHS.native)
ROOT = FRAME.parent
sys.path[:0] = [str(FRAME), str(ROOT / "tools")]
from depth_model.adapters import NativeGraph  # noqa: E402
from depth_model.lowering.resident_image import unpack  # noqa: E402
from check_kernels import load_inputs  # noqa: E402

MAGIC = 0x3253324E  # N2S2
DEFAULT_SEQUENCE = 224


def blocks(t, pools):
    return b"".join(bytes(pools[e["pool"]][e["offset"]:e["offset"] + 2048])
                   for e in t["storage"]["blocks"])


def direct(t, pools):
    p = t["storage"]["direct"]
    n = t["element_bytes"]
    for d in t["shape"]["dimensions"]:
        n *= d
    return bytes(pools[p["pool"]][p["offset"]:p["offset"] + n])


def main():
    sequence = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEQUENCE
    image = FRAME / "build/resident-primary.qdrx"
    graph = unpack(image.read_bytes())
    command = next(c for c in graph["commands"] if c["sequence"] == sequence)
    if command["variant"] != "ConvLayer_s2.opt@CB*2.Xwb.Fi":
        raise SystemExit(f"unexpected target variant {command['variant']}")
    index = graph["commands"].index(command)
    tensors = {t["id"]: t for t in graph["tensors"]}
    x, w, fi = (tensors[i] for i in command["inputs"][:3])
    y = tensors[command["outputs"][0]]
    host = NativeGraph(image, ROOT / "recovered/primary/constants/constant_pool_5.bin")
    try:
        host.use_cpu()
        inputs = load_inputs(FRAME / "build/depth-inputs/random")
        host.run(inputs, last=index)
        xb = blocks(x, host.pools)
        wb = direct(w, host.pools)
        fb = direct(fi, host.pools)
        host.run(inputs, first=index, last=index + 1)
        yb = blocks(y, host.pools)
        vals = [MAGIC, sequence]
        for key in ("dimensions", "max_dimensions", "padding"):
            vals.extend(x["shape"][key])
        vals.extend([x["storage"]["slice_offset"], len(x["storage"]["blocks"])])
        for key in ("dimensions", "max_dimensions", "padding"):
            vals.extend(y["shape"][key])
        vals.extend([y["storage"]["slice_offset"], len(y["storage"]["blocks"])])
        vals.extend(w["shape"]["dimensions"])
        vals.extend([2 if w["class_name"] == "Xpb" else 1, len(wb), len(fb),
                     x.get("quantization", {}).get("zero_point", 0),
                     y.get("quantization", {}).get("zero_point", 0)])
        vals += [0] * (48 - len(vals))
        raw = struct.pack("<48I", *vals) + xb + wb + fb + yb
        out = (PATHS.native / 'kernels/stride2/s2_cpu_prefix_fixture.bin')
        out.write_bytes(raw)
        print(f"fixture={out} sequence={sequence} bytes={len(raw)} X={len(xb)} W={len(wb)} FI={len(fb)} Y={len(yb)}")
        print(f"fixture_sha256={hashlib.sha256(raw).hexdigest()}")
        print(f"image_sha256={hashlib.sha256(image.read_bytes()).hexdigest()}")
        print(f"input={x['shape']} output={y['shape']} weights={w['shape']} fi={fi['shape']}")
    finally:
        host.close()


if __name__ == "__main__":
    main()
