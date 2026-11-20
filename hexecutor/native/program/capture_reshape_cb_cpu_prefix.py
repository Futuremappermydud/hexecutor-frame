#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture Reshape.hw@CB*2 CPU-prefix fixtures for each CPU model.

By default captures the first command; --all-pads captures the first command
for each source H padding value 0..7 (each occurs 10 times per model)."""
from __future__ import annotations

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


import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = (PATHS.repo)
sys.path.insert(0, str((PATHS.native)))
sys.path.insert(0, str((PATHS.model_tools)))
from depth_model.adapters import NativeGraph
import depth_model.lowering.resident_image as resident_image
from check_kernels import load_inputs
from bench.verify_native_graph import cases


def capture(model: str, case: str, outdir: Path, pad: int | None = None) -> Path:
    image = (PATHS.native / f'build/resident-{model}.qdrx')
    decoded = resident_image.unpack(image.read_bytes())
    hits = [(i, c) for i, c in enumerate(decoded["commands"])
            if c["opcode"] == 16 and c["variant"] == "Reshape.hw@CB*2"]
    if len(hits) != 80:
        raise ValueError(f"expected 80 CB reshapes in {model}, got {len(hits)}")
    tensors = {t["id"]: t for t in decoded["tensors"]}
    if pad is not None:
        hits = [(i, c) for i, c in hits
                if tuple(tensors[c["inputs"][0]]["shape"]["padding"]) == (0, pad, 0, 0)]
        if len(hits) != 10:
            raise ValueError(f"expected 10 CB reshapes with pad {pad} in {model}, got {len(hits)}")
    index, command = hits[0]
    output = tensors[command["outputs"][0]]
    st = output["storage"]
    pages = st["blocks"][st["slice_offset"]:st["slice_offset"] + 4]
    if len(pages) != 4 or any(p["pool"] != 1 for p in pages):
        raise ValueError("unexpected CB reshape output pages")
    offsets = [p["offset"] for p in pages]
    constants = (PATHS.model_data / f'recovered/{model}/constants/constant_pool_5.bin')
    case_dir = next(row[1] for row in cases(model, [case])[1] if row[0] == case)
    inputs = load_inputs(case_dir)
    graph = NativeGraph(image, constants)
    try:
        graph.use_cpu()
        graph.set_hoist(False)
        graph.set_block_output(False)
        graph.run(inputs, 0, index)
        before = [graph.pools[p].copy() if p in graph.pools else np.zeros(0, np.uint8)
                  for p in range(7)]
        graph.run(inputs, index, index + 1)
        after = [graph.pools[p].copy() if p in graph.pools else np.zeros(0, np.uint8)
                 for p in range(7)]
        allowed = np.zeros(len(after[1]), dtype=bool)
        for off in offsets:
            allowed[off:off + 2048] = True
        if np.any((before[1] != after[1]) & ~allowed):
            raise ValueError("CPU reshape wrote outside output pages")
        if any(not np.array_equal(before[p], after[p]) for p in range(7) if p != 1):
            raise ValueError("CPU reshape wrote another pool")
        outdir.mkdir(parents=True, exist_ok=True)
        suffix = "" if pad is None else f"-pad{pad}"
        path = outdir / f"{model}-{case}-reshape-cb{suffix}.epcpu"
        with path.open("wb") as f:
            f.write(struct.pack("<4s16I", b"EPCP", 1, 16, 49, command["sequence"], index,
                                0xffffffff, 0, 0, 4, *(len(p) for p in before)))
            for pool in before:
                f.write(pool.tobytes())
            for off in offsets:
                f.write(struct.pack("<I", off))
                f.write(after[1][off:off + 2048].tobytes())
        print(f"{model}/{case}: command={index} sequence={command['sequence']} fixture={path}", flush=True)
        return path
    finally:
        graph.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="smoke")
    ap.add_argument("--out", type=Path, default=(PATHS.native / 'program/fixtures'))
    ap.add_argument("--all-pads", action="store_true")
    args = ap.parse_args()
    for model_name in ("primary", "alternate"):
        if args.all_pads:
            for row_pad in range(8):
                capture(model_name, args.case, args.out, row_pad)
        else:
            capture(model_name, args.case, args.out)
