#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture real QDRX 247 input/W/FI and post-command physical output pages."""

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

import hashlib
import struct
import sys
from pathlib import Path

HERE = (PATHS.native / 'kernels/convolution')
FRAME = (PATHS.native)
ROOT = FRAME.parent
sys.path[:0] = [str(FRAME), str(ROOT / "tools")]
from depth_model.adapters import NativeGraph  # noqa: E402
from depth_model.lowering.resident_image import unpack  # noqa: E402
from check_kernels import load_inputs  # noqa: E402

MAGIC = 0x324B5748  # HWK2
SEQUENCE = 247


def blocks(tensor, pools):
    return b"".join(bytes(pools[e["pool"]][e["offset"]:e["offset"] + 2048])
                   for e in tensor["storage"]["blocks"])


def direct(tensor, pools):
    p = tensor["storage"]["direct"]
    size = tensor["element_bytes"]
    for d in tensor["shape"]["dimensions"]:
        size *= d
    return bytes(pools[p["pool"]][p["offset"]:p["offset"] + size])


def main():
    image = FRAME / "build/resident-primary.qdrx"
    graph = unpack(image.read_bytes())
    command = next(c for c in graph["commands"] if c["sequence"] == SEQUENCE)
    index = graph["commands"].index(command)
    tensors = {t["id"]: t for t in graph["tensors"]}
    x, w, fi = (tensors[i] for i in command["inputs"][:3])
    y = tensors[command["outputs"][0]]
    host = NativeGraph(image, ROOT / "recovered/primary/constants/constant_pool_5.bin")
    try:
        host.use_cpu()
        host.run(load_inputs(FRAME / "build/depth-inputs/random"), last=index)
        xb = blocks(x, host.pools)
        wb = direct(w, host.pools)
        fb = direct(fi, host.pools)
        # Execute only the target node, then capture its full output block table.
        host.run(load_inputs(FRAME / "build/depth-inputs/random"), first=index, last=index + 1)
        yb = blocks(y, host.pools)
        values = [MAGIC, SEQUENCE,
                  x["shape"]["dimensions"][1], x["shape"]["dimensions"][2],
                  x["shape"]["max_dimensions"][1], x["shape"]["max_dimensions"][2],
                  x["storage"]["slice_offset"], len(x["storage"]["blocks"]),
                  y["shape"]["dimensions"][1], y["shape"]["dimensions"][2],
                  y["shape"]["max_dimensions"][1], y["shape"]["max_dimensions"][2],
                  y["storage"]["slice_offset"], len(y["storage"]["blocks"]),
                  w["shape"]["dimensions"][2], w["shape"]["dimensions"][3],
                  2 if w["class_name"] == "Xwb" else 1, len(wb), len(fb),
                  x["shape"]["padding"][1], x["shape"]["padding"][2],
                  y["shape"]["padding"][1], y["shape"]["padding"][2],
                  x.get("quantization", {}).get("zero_point", 0)]
        values += [0] * (32 - len(values))
        raw = struct.pack("<32I", *values) + xb + yb + wb + fb
        out = (PATHS.native / 'kernels/convolution/xwb_ci64_qdrx247_cpu_prefix_fixture.bin')
        out.write_bytes(raw)
        print(f"fixture={out} bytes={len(raw)} input={len(xb)} output={len(yb)} "
              f"weights={len(wb)} fi={len(fb)} output_slice={values[12]}")
        print(f"fixture_sha256={hashlib.sha256(raw).hexdigest()}")
        print(f"image_sha256={hashlib.sha256(image.read_bytes()).hexdigest()}")
    finally:
        host.close()


if __name__ == "__main__":
    main()
