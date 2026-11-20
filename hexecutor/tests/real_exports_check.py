#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Real PyTorch exports through the float ONNX lowerer, compared on the reference runner.

Standalone report; the unit tests do not run it (it needs torch, which the onnx test
environment lacks). From the repository root:

    python3 hexecutor/tests/real_exports_check.py [--work DIR] [--only NAME ...]

Requirements: torch, onnx, onnxruntime, numpy; torchvision for the mobilenet_v2 and
resnet18 rows (they report SKIP without it). Models are random-weight, eval mode,
batch one, static shapes. Each is exported twice with torch.onnx.export(dynamo=False,
opset_version=17): with do_constant_folding=True (the usual export) and with it False
(exposes the Shape/Gather/Identity hygiene chains). Each export is lowered by
hexecutor/offline/lower_onnx_float.py, run on the portable reference runner
(HEXECUTOR_TENSOR_RUN, default build/bin/hexecutor_tensor_run) and every output is
compared with onnxruntime on the exported file: |got - want| <= 1e-3 * (1 + |want|).
Files go under --work (a fresh temporary directory by default).
"""

import argparse
import collections
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNNER = Path(os.environ.get("HEXECUTOR_TENSOR_RUN", ROOT / "build/bin/hexecutor_tensor_run"))
TOL = 1e-3


# ------------------------------------------------------------------ models
def _models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MBV3Block(nn.Module):
        """MobileNetV3-style: 1x1 expand + HardSwish, depthwise stride-2 + HardSwish, SE with a
        hard-sigmoid gate, 1x1 project (no residual: the stride changes the shape)."""

        def __init__(self):
            super().__init__()
            self.expand = nn.Conv2d(16, 64, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.act1 = nn.Hardswish()
            self.dw = nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.act2 = nn.Hardswish()
            self.se_pool = nn.AdaptiveAvgPool2d(1)
            self.se1 = nn.Conv2d(64, 16, 1)
            self.se_relu = nn.ReLU()
            self.se2 = nn.Conv2d(16, 64, 1)
            self.se_gate = nn.Hardsigmoid()
            self.project = nn.Conv2d(64, 16, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(16)

        def forward(self, x):
            y = self.act1(self.bn1(self.expand(x)))
            y = self.act2(self.bn2(self.dw(y)))
            s = self.se_gate(self.se2(self.se_relu(self.se1(self.se_pool(y)))))
            return self.bn3(self.project(y * s))

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()

            def block(ci, co):
                return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                                     nn.ReLU())

            self.enc1 = block(3, 16)
            self.down = nn.MaxPool2d(2)
            self.enc2 = block(16, 32)
            self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
            self.dec1 = block(48, 16)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec2 = block(16, 8)
            self.head = nn.Conv2d(8, 4, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down(e1))
            d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
            return self.head(self.dec2(self.up2(d1)))

    class ShapeChain(nn.Module):
        """Size-from-shape upsampling and a view-based flatten (Shape/Gather/Concat chains)."""

        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 8, 3, padding=1)
            self.fc = nn.Linear(16 * 16 * 8, 10)

        def forward(self, x):
            y = F.max_pool2d(F.relu(self.c1(x)), 2)
            y = F.interpolate(y, size=(y.shape[2] * 2, y.shape[3] * 2), mode="nearest")
            return self.fc(self.c2(y).view(y.size(0), -1))

    class ScalarAffine(nn.Module):
        """BatchNorm after a residual Add (cannot fold), scalar Mul/Add/Sub, Clip(0, 6)."""

        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(16)

        def forward(self, x):
            y = self.c1(x)
            z = self.bn(y + self.c2(y))
            z = F.hardtanh(z * 0.5 + 1.0, 0.0, 6.0)
            return z - 2.0

    class Hygiene(nn.Module):
        """Exporter hygiene: float-to-float cast, eval Dropout, clamp (Clip), mean over spatial
        axes with keepdim, Squeeze with constant axes, then a linear head."""

        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 12, 3, padding=1)
            self.fc = nn.Linear(12, 5)

        def forward(self, x):
            y = F.dropout(self.c(x.float()), p=0.5, training=False).clamp(0.0, 6.0)
            m = y.mean(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1)
            return self.fc(m)

    class OnesLikeAdd(nn.Module):
        """A full-size constant operand (ones_like folds to a constant tensor): expected to be
        rejected, since only per-channel constants are lowered."""

        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            y = self.c(x)
            return y + torch.ones_like(y) * 0.25

    def torchvision_model(name):
        import torchvision

        return getattr(torchvision.models, name)(weights=None)

    return {
        "mobilenet_v2": (lambda: torchvision_model("mobilenet_v2"), (1, 3, 96, 96)),
        "resnet18": (lambda: torchvision_model("resnet18"), (1, 3, 64, 64)),
        "mbv3_block": (MBV3Block, (1, 16, 32, 32)),
        "tiny_unet": (TinyUNet, (1, 3, 32, 32)),
        "shape_chain": (ShapeChain, (1, 3, 16, 16)),
        "scalar_affine": (ScalarAffine, (1, 3, 16, 16)),
        "hygiene": (Hygiene, (1, 3, 16, 16)),
        "ones_like_add": (OnesLikeAdd, (1, 3, 8, 8)),
    }


# ------------------------------------------------------------------ comparison
def _hwc(o):
    """An ONNX batch-one output as the runner's layout: HWC for rank 4, [C] for rank 2."""
    if o.ndim == 3:  # [1, C, L] is NCHW with H = 1: HWC is [L, C]
        return o[0].transpose(1, 0)
    return o[0].transpose(1, 2, 0) if o.ndim == 4 else o[0]


def _merge(parts, got, flats):
    """Reassemble channel-split runner outputs (HWC, flat) into the given HWC/flat shapes."""
    merged, i = [], 0
    for flat in flats:
        width, chunks = flat.shape[-1], []
        while sum(c.shape[-1] for c in chunks) < width:
            lo, hi = parts[i]["channels"]
            chunks.append(got[i].reshape(-1, hi - lo))
            i += 1
        merged.append(np.concatenate(chunks, axis=-1).reshape(flat.shape))
    return merged


def _hxfg_op_kinds(prefix):
    image = Path(f"{prefix}.hxfg").read_bytes()
    tensors, ops = struct.unpack_from("<2I", image, 12)
    return [struct.unpack_from("<I", image, 256 + 32 * tensors + 64 * i)[0] for i in range(ops)]


def check(name, build, fold, work):
    import onnx
    import onnxruntime
    import torch
    from hexecutor.offline import lower_onnx_float as lowering

    tag = f"{name}{'' if fold else '_nofold'}"
    row = {"model": tag, "nodes": "-", "hxfg": "-", "fp16": "-", "err": "-", "range": "-",
           "result": ""}
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    path, (factory, shape) = d / "model.onnx", build
    try:
        torch.manual_seed(0)
        model = factory()
        for m in model.modules():  # non-trivial running statistics, so outputs are not ~0
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 2.0)
        model.eval()
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (torch.randn(shape),), str(path), dynamo=False,
                              opset_version=17, do_constant_folding=fold,
                              input_names=["x"], output_names=["y"])
    except ImportError:  # torchvision missing: the caller reports SKIP
        raise
    except Exception as e:  # noqa: BLE001 - reported in the table
        row["result"] = f"EXPORT FAILED: {type(e).__name__}: {str(e).splitlines()[0]}"
        return row, []
    row["nodes"] = len(onnx.load(str(path)).graph.node)
    x = np.random.default_rng(11).normal(0, 1, shape).astype(np.float32)
    want = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"x": x})
    try:
        manifest = lowering.lower(path, d / "g")
    except Exception as e:  # noqa: BLE001 - the reason is the result
        row["result"] = f"LOWER FAILED: {type(e).__name__}: {e}"
        return row, []
    plan = manifest["fp16_plan"]
    row["hxfg"] = len(_hxfg_op_kinds(d / "g"))
    row["fp16"] = "eligible" if plan["eligible"] else f"no ({len(plan['blockers'])} blockers)"
    data = _hwc(x)
    (d / "in.f32").write_bytes(np.ascontiguousarray(data, np.float32).tobytes())
    try:
        subprocess.run([str(RUNNER), str(d / "g.hxfg"), str(d / "g.bin"), str(d / "in.f32"),
                        str(d / "out_")], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        row["result"] = f"RUNNER FAILED: {e.stderr.decode(errors='replace').strip()[-200:]}"
        return row, plan["blockers"]
    got, i = [], 0
    while (d / f"out_{i}.f32").exists():
        got.append(np.fromfile(d / f"out_{i}.f32", np.float32))
        i += 1
    if len(got) != len(manifest["output_tensors"]):
        row["result"] = "FAIL: runner produced a different number of outputs"
        return row, plan["blockers"]
    flats = [_hwc(o) for o in want]
    merged = _merge(manifest["output_tensors"], got, flats)
    err = max(float(np.max(np.abs(m - w))) for m, w in zip(merged, flats))
    ok = all(np.all(np.abs(m - w) <= TOL * (1 + np.abs(w))) for m, w in zip(merged, flats))
    row["err"] = f"{err:.2e}"
    row["range"] = f"{max(float(np.max(np.abs(w))) for w in flats):.2g}"
    row["result"] = "PASS" if ok else "FAIL: outputs differ"
    return row, plan["blockers"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", type=Path, help="scratch directory (default: a temporary one)")
    p.add_argument("--only", nargs="*", help="model names to run (default: all)")
    args = p.parse_args()
    if not RUNNER.exists():
        sys.exit(f"reference runner not found: {RUNNER} (set HEXECUTOR_TENSOR_RUN)")
    work = args.work or Path(tempfile.mkdtemp(prefix="real-exports-"))
    work.mkdir(parents=True, exist_ok=True)
    models = _models()
    names = args.only or list(models)
    rows, blockers = [], {}
    for name in names:
        if name not in models:
            sys.exit(f"unknown model {name}; choose from {', '.join(models)}")
        for fold in (True, False):
            tag = name + ("" if fold else "_nofold")
            try:
                row, reasons = check(name, models[name], fold, work)
            except ImportError as e:  # torch, torchvision or onnxruntime missing
                row = {"model": tag, "nodes": "-", "hxfg": "-", "fp16": "-", "err": "-",
                       "range": "-", "result": f"SKIP: {e}"}
                reasons = []
            rows.append(row)
            if reasons:
                blockers[row["model"]] = reasons
    cols = ["model", "nodes", "hxfg", "fp16", "err", "range", "result"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if blockers:
        print("\nFP16 plan blockers (the graph still runs on the FP32 path):")
        for tag, reasons in blockers.items():
            for reason in reasons:
                print(f"  {tag}: {reason}")
    print(f"\nfiles: {work}")
    failed = [r for r in rows if r["result"].split(":")[0] in
              ("EXPORT FAILED", "LOWER FAILED", "RUNNER FAILED", "FAIL")]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
