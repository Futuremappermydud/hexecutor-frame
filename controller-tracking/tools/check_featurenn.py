#!/usr/bin/env python3
"""Compare the lowered FeatureNN HXFG graph against onnxruntime on the QLinear model.

Offline only. onnxruntime on the quantized ONNX model stands in for XRService's
own CPU fallback; the HXFG graph is run through Hexecutor's portable CPU
reference (`hexecutor_tensor_run` with four arguments), or through the DSP when
--device arguments are given on the Frame. Inputs are raw 0..1 stereo crops; the
host-side normalization `(x - 0.449) / 0.226` and the reshape/split/clip/exp head
are reproduced here exactly as the ONNX graph defines them.

Reports per-output differences for the Gemm vector and the decoded heads
(center, uncertainty, plus_x, plus_z). Nothing here is used at runtime.
"""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import onnx
import onnxruntime as ort

GEMM = "/head/fc/Gemm_output_0"


def onnx_reference(model_path):
    """Session returning the Gemm vector and the four named outputs."""
    model = onnx.load(str(model_path))
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(GEMM, onnx.TensorProto.FLOAT, [1, 552])
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in session.get_outputs()]
    return lambda image: dict(zip(names, session.run(None, {"Image": image})))


def normalization(model_path):
    """Read the Sub/Div constants that precede QuantizeLinear."""
    model = onnx.load(str(model_path))
    init = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    sub = next(n for n in model.graph.node if n.op_type == "Sub")
    div = next(n for n in model.graph.node if n.op_type == "Div")
    return float(init[sub.input[1]]), float(init[div.input[1]])


def decode(gemm):
    """The ONNX head: Reshape [2,92,3], Split 4x23, Clip(max 10), *2, Exp."""
    r = gemm.reshape(2, 92, 3)
    center, log_unc, plus_x, plus_z = (r[:, 23 * k : 23 * (k + 1)] for k in range(4))
    return {
        "center": center,
        "uncertainty": np.exp(np.minimum(log_unc, 10.0) * 2.0),
        "plus_x": plus_x,
        "plus_z": plus_z,
    }


def inputs(sample, rng, count):
    """Stereo 128x128 crops: uniform noise, then shifted natural crops."""
    out = [("noise", rng.random((1, 2, 128, 128), dtype=np.float32))]
    if sample:
        raw = sample.read_bytes()
        header = raw.split(b"\n", 3)
        w, h = map(int, header[1].split())
        img = np.frombuffer(header[3][: w * h], np.uint8).reshape(h, w) / 255.0
        for k in range(count):
            y, x = rng.integers(0, h - 140), rng.integers(0, w - 140)
            left = img[y : y + 128, x : x + 128]
            right = img[y : y + 128, x + 6 : x + 134]  # small stereo disparity
            out.append((f"crop{k}", np.stack([left, right])[None].astype(np.float32)))
    return out


def run_hxfg(runner, prefix, x, device, work):
    """Run the lowered graph; input is NHWC with 3 compact channels."""
    nhwc = np.zeros((128, 128, 3), np.float32)
    nhwc[..., :2] = x[0].transpose(1, 2, 0)
    path = work / "input.f32"
    path.write_bytes(nhwc.tobytes())
    cmd = [str(runner), str(prefix.with_suffix(".hxfg")), str(prefix.with_suffix(".bin")),
           str(path), str(work / "out_")] + device
    subprocess.run(cmd, check=True, capture_output=True)
    # A wide head may be split into several outputs (see output_parts).
    parts = json.loads(prefix.with_suffix(".json").read_text())["output_parts"]
    y = np.zeros(parts[-1]["channels"][1], np.float32)
    for i, part in enumerate(parts):
        lo, hi = part["channels"]
        y[lo:hi] = np.fromfile(work / f"out_{i}.f32", np.float32)
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--lowered", type=Path, required=True, help="HXFG prefix")
    p.add_argument("--runner", type=Path, required=True, help="hexecutor_tensor_run")
    p.add_argument("--sample", type=Path, help="binary PGM for natural crops")
    p.add_argument("--crops", type=int, default=8)
    p.add_argument("--device", nargs=4, metavar=("LOADER", "ENGINE", "BOOTSTRAP", "PROBE"),
                   default=[])
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    reference = onnx_reference(a.model)
    mean, std = normalization(a.model)
    rng = np.random.default_rng(0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, x in inputs(a.sample, rng, a.crops):
            want = reference(x)
            got = run_hxfg(a.runner, a.lowered, (x - mean) / std, a.device, Path(tmp))
            ref_gemm = want[GEMM].reshape(-1)
            mine, ref = decode(got), decode(ref_gemm)
            for k in ref:  # the decode must match onnxruntime's own head exactly
                if not np.allclose(ref[k], want[k][...], rtol=1e-5, atol=1e-6):
                    raise SystemExit(f"head decode mismatch for {k}")
            row = {"input": name,
                   "gemm_max_abs": float(np.abs(got - ref_gemm).max()),
                   "gemm_ref_abs_mean": float(np.abs(ref_gemm).mean())}
            for k in ("center", "plus_x", "plus_z"):
                d = np.abs(mine[k] - ref[k])
                row[f"{k}_max_abs"], row[f"{k}_mean_abs"] = float(d.max()), float(d.mean())
            ratio = mine["uncertainty"] / ref["uncertainty"]
            row["uncertainty_ratio_range"] = [float(ratio.min()), float(ratio.max())]
            row["center_ref_range"] = [float(ref["center"].min()), float(ref["center"].max())]
            rows.append(row)
            print(json.dumps(row))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
