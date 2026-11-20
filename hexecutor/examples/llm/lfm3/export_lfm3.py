#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export LFM3-1.7B stages through ExecuTorch (torch.export -> HexecutorPartitioner -> HexecutorBackend) to HXFG.

Usage: export_lfm3.py MODEL_DIR OUT_DIR [--lmax N] [--stages N] [--only I] [--q8 KINDS] [--q8-w1-layers N]
Writes OUT_DIR/stage<i>.{hxfg,bin} and OUT_DIR/manifest.json (shapes, state names, sizes).
"""
import argparse, json, pathlib, sys, time
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lfm"))
import lfm3_stages as L
from gen_executorch_decode_golden import export_and_capture

ap = argparse.ArgumentParser()
ap.add_argument("model"), ap.add_argument("out")
ap.add_argument("--lmax", type=int, default=256)
ap.add_argument("--stages", type=int, default=4)
ap.add_argument("--only", type=int)
ap.add_argument("--q8", default="q k v o in_b in_c in_x out")
ap.add_argument("--q8-w1-layers", type=int, default=0, help="w1 stays Q8 in the first N layers")
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ck = L.Checkpoint(a.model)
manifest = {"lmax": a.lmax, "stages": [], "q8": a.q8, "q8_w1_layers": a.q8_w1_layers}
for i, (lo, hi) in enumerate(L.stage_ranges(a.stages)):
    if a.only is not None and i != a.only:
        continue
    t0 = time.time()
    stage = L.Stage(ck, lo, hi, a.lmax, final=(hi == 24), q8=a.q8.split(),
                    q8_names=[f"l{j}_w1" for j in range(a.q8_w1_layers)], log=lambda *_: None).eval()
    example = (torch.zeros(1, 1, L.D), torch.zeros(1, 1), torch.zeros(1, 1, 1, a.lmax), torch.zeros(1, 1, 1, a.lmax))
    (summary, image, constants), _ = export_and_capture(stage, example)
    (out / f"stage{i}.hxfg").write_bytes(image)
    (out / f"stage{i}.bin").write_bytes(constants)
    manifest["stages"].append({"index": i, "layers": [lo, hi], "arena_bytes": summary["arena_bytes"],
                               "state_bytes": summary["state_bytes"], "constant_bytes": summary["constant_bytes"],
                               "ops": summary["operations"], "state_names": summary["state_names"]})
    print(f"stage {i} layers {lo}-{hi}: {summary['operations']} ops, {summary['constant_bytes'] / 2**20:.1f} MiB constants, "
          f"{summary['state_bytes'] / 2**20:.2f} MiB state, {summary['arena_bytes'] / 2**20:.2f} MiB arena, "
          f"{time.time() - t0:.0f} s", flush=True)
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
