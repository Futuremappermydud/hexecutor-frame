#!/usr/bin/env python3
"""Check recovered convolution parameters against QAIRT's own HTP compiler and simulator.

For every stored convolution, build an equivalent one-operator QNN Conv2d graph
from the recovered weights, scales and int32 bias, then:

1. let the SDK prepare it and search the new context for the original compiled
   `Fi` (requantization/bias) and packed weight bytes; and
2. execute it on random inputs in the x86 HTP simulator and fit the exact
   integer requantization that produced the outputs.

Packed `Xpb` weights are tested as 1x1 convolutions over their im2col input,
which is the arithmetic the compiled kernel performs. Padding is fixed by this
test graph and is not recovered here.

Requires Docker, the qnn_op_harness binary (see depth-model/sim/), and NumPy.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

import numpy as np

IMAGE = "depth-model-qnn-sim"


def load(recovered):
    g = json.loads((recovered / "graph.json").read_text())
    tensors = {t["id"]: t for t in g["tensors"]}
    layers = json.loads((recovered / "constants" / "layers.json").read_text())
    return g, tensors, layers


def describe(g, tensors, layer):
    sid = layer["source_id"]
    comp = [n for n in g["nodes"] if n["source_id"] == sid and n["type_name"].startswith("ConvLayer_s")]
    node = comp[0]
    act_in, weight, fi = node["inputs"][0], node["inputs"][1], node["inputs"][2]
    out = node["outputs"][0]
    q_in, q_out = tensors[act_in]["quantization"], tensors[out]["quantization"]
    sixteen = tensors[act_in]["class_name"] == "CH"
    hwio = layer["hwio_shape"]
    packed = tensors[weight]["class_name"] == "Xpb"
    stride = 2 if node["type_name"].startswith("ConvLayer_s2") else 1
    return {
        "source_id": sid, "name": layer["name"], "kernel": node["type_name"],
        "hwio": hwio, "packed": packed, "stride": stride, "sixteen_bit": sixteen,
        "in_scale": q_in["scale"], "in_zero_point": q_in["zero_point"],
        "out_scale": q_out["scale"], "out_zero_point": q_out["zero_point"],
        "weight_scale": layer["weight_quantization"]["scale"],
        "weight_file": layer["weight_file"], "fi_tensor": layer["compiled_parameter_tensors"][0],
        "weight_tensor": layer["weight_tensor"],
    }


def split_fi(fi, cout):
    """Fi rows hold [multiplier x32, bias x32] for each group of 32 output channels."""
    rows = fi.reshape(-1, 64)
    return rows[:, :32].reshape(-1)[:cout], rows[:, 32:].reshape(-1)[:cout]


def write_case(d, info, recovered, rng):
    d.mkdir(parents=True, exist_ok=True)
    w = np.load(recovered / "constants" / info["weight_file"])
    kh, kw, cin, cout = w.shape
    w.astype(np.int8).tofile(d / "w.bin")
    np.zeros(cout, np.int32).tofile(d / "b0.bin")
    h, wd = 16, 32
    if info["stride"] == 2:
        h, wd = 2 * h, 2 * wd
    adt, nbytes = ("ufxp16", np.uint16) if info["sixteen_bit"] else ("ufxp8", np.uint8)
    hi = 65536 if info["sixteen_bit"] else 256
    x = rng.integers(0, hi, (1, h, wd, cin)).astype(nbytes)
    x.tofile(d / "x.raw")
    ph, pw = kh // 2, kw // 2
    oh, ow = (h + 2 * ph - kh) // info["stride"] + 1, (wd + 2 * pw - kw) // info["stride"] + 1
    s_in, s_w, s_out = info["in_scale"], info["weight_scale"], info["out_scale"]
    spec = f"""tensor x app_write {adt} {s_in!r} {-info['in_zero_point']} 4 1 {h} {wd} {cin}
tensor w static sfxp8 {s_w!r} 0 4 {kh} {kw} {cin} {cout} /work/{d.name}/w.bin
tensor b static sfxp32 {s_in * s_w!r} 0 1 {cout} /work/{d.name}/BIAS
tensor y app_read {adt} {s_out!r} {-info['out_zero_point']} 4 1 {oh} {ow} {cout}
param_tensor conv stride 1 2 {info['stride']} {info['stride']}
param_tensor conv pad_amount 2 2 2 {ph} {ph} {pw} {pw}
param_tensor conv dilation 1 2 1 1
param_scalar conv group u32 1
node conv Conv2d in 3 x w b out 1 y
"""
    (d / "spec0.txt").write_text(spec.replace("BIAS", "b0.bin"))
    (d / "spec.txt").write_text(spec.replace("BIAS", "b.bin"))
    return {"input_shape": [1, h, wd, cin], "output_shape": [1, oh, ow, cout], "pad": [ph, pw]}


def model_bias(fi, ctx0, cout, sixteen_bit):
    """Invert the compiler's bias mapping using a zero-bias calibration context.

    8-bit:  Fi bias = bias + fold(zp_out)
    16-bit: Fi bias = floor(bias / 256) + fold(zp_out)
    Returns an int32 bias that compiles to the original Fi. For 16-bit layers the
    original model bias is only determined to within [0, 255] of this value.
    """
    fi = fi.reshape(-1).astype("<i4")
    pos = ctx0.find(fi.reshape(-1, 64)[0, :32].tobytes())
    if pos < 0:
        raise ValueError("calibration context lacks the original multiplier row")
    fi0 = np.frombuffer(ctx0[pos:pos + fi.nbytes], "<i4")
    _, bias = split_fi(fi, cout)
    _, bias0 = split_fi(fi0, cout)
    return (bias.astype(np.int64) - bias0) * (256 if sixteen_bit else 1)


def run_harness(args, work, cases, spec, execute):
    script = "\n".join(
        f"/h/qnn_op_harness /sdk/lib/x86_64-linux-clang/libQnnHtp.so /work/{d.name}/{spec} /work/{d.name} "
        f"--save-context /work/{d.name}/{spec[:-4]}.ctx"
        + (f" --input x=/work/{d.name}/x.raw" if execute else "")
        + f" > /work/{d.name}/{spec[:-4]}.log 2>&1 || echo FAIL {d.name}" for d, _ in cases)
    subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64",
                    "-e", "LD_LIBRARY_PATH=/sdk/lib/x86_64-linux-clang:/usr/lib/llvm-14/lib",
                    "-v", f"{args.sdk.resolve()}:/sdk:ro", "-v", f"{args.harness.resolve().parent}:/h:ro",
                    "-v", f"{work}:/work", IMAGE, "bash", "-c", script], check=True)


def accumulate(x, w, bias, stride, pad, zp_in):
    x = x.astype(np.int64) - zp_in
    kh, kw, cin, cout = w.shape
    xp = np.pad(x[0], ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
    oh = (xp.shape[0] - kh) // stride + 1
    ow = (xp.shape[1] - kw) // stride + 1
    acc = np.zeros((oh, ow, cout), np.int64)
    w = w.astype(np.int64)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[ky:ky + stride * oh:stride, kx:kx + stride * ow:stride]
            acc += patch @ w[ky, kx]
    return acc + bias.astype(np.int64)


def requantize(acc_with_fi_bias, real_multiplier, sixteen_bit):
    """Closed-form HTP convolution requantization recovered with this tool.

    acc_with_fi_bias = sum((x - zp_in) * w) + Fi_bias * (256 if sixteen_bit else 1),
    where the compiled Fi bias already folds in the output zero point.
    The scale is applied as an fp16 value after a right shift p.
    """
    p = 8 if sixteen_bit else math.floor(-math.log2(real_multiplier)) - 3
    scale = Fraction(float(np.float16(real_multiplier * 2 ** p)))
    m, s = scale.numerator, scale.denominator.bit_length() - 1
    rounding = (1 << (s - 1)) if s > 0 else 0
    qmax = 65535 if sixteen_bit else 255
    return np.clip(((acc_with_fi_bias >> p) * m + rounding) >> s, 0, qmax)


def fit_requant(acc, y, zp_out, qmax):
    """Find pre-shift p and the exact rational m/2^s with
    y = clip(((acc >> p) * m + 2^(s-1)) >> s) + zp_out."""
    inner = (y > 0) & (y < qmax)
    for p in range(0, 12):
        a = (acc >> p)[inner]
        target = y[inner].astype(np.float64) - zp_out
        pos, neg = a > 0, a < 0
        for s in range(8, 40):
            lo = hi = None
            scale = 2.0 ** s
            bounds_lo = np.concatenate([((target - 0.5) * scale / a)[pos], ((target + 0.5) * scale / a)[neg]])
            bounds_hi = np.concatenate([((target + 0.5) * scale / a)[pos], ((target - 0.5) * scale / a)[neg]])
            lo, hi = np.ceil(bounds_lo.max()), np.ceil(bounds_hi.min()) - 1
            if lo <= hi:
                m = int(lo)
                pred = np.clip(((acc >> p) * m + (1 << (s - 1)) >> s) + zp_out, 0, qmax)
                if np.array_equal(pred, y):
                    return {"pre_shift": p, "multiplier": m, "shift": s}
        # try next pre-shift
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovered", type=Path)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--only", type=int, nargs="*", help="source IDs to test")
    args = parser.parse_args()
    recovered, work = args.recovered.resolve(), args.work.resolve()
    g, tensors, layers = load(recovered)
    rng = np.random.default_rng(7)
    pool = (recovered / "constants" / "constant_pool_5.bin").read_bytes()
    cases = []
    for layer in layers:
        if args.only and layer["source_id"] not in args.only:
            continue
        info = describe(g, tensors, layer)
        d = work / f"layer_{info['source_id']}"
        info.update(write_case(d, info, recovered, rng))
        cases.append((d, info))

    run_harness(args, work, cases, "spec0.txt", execute=False)
    for d, info in cases:
        fi = np.load(recovered / "constants" / "arrays" / f"tensor_{info['fi_tensor']:05d}.npy")
        bias = model_bias(fi, (d / "spec0.ctx").read_bytes(), info["hwio"][3], info["sixteen_bit"])
        bias.astype(np.int32).tofile(d / "b.bin")
    run_harness(args, work, cases, "spec.txt", execute=True)

    results = []
    for d, info in cases:
        r = dict(info)
        ctx = (d / "spec.ctx").read_bytes() if (d / "spec.ctx").exists() else b""
        fi = np.load(recovered / "constants" / "arrays" / f"tensor_{info['fi_tensor']:05d}.npy")
        r["model_bias_int32"] = np.fromfile(d / "b.bin", np.int32).tolist()
        r["fi_reproduced"] = bool(ctx) and ctx.find(fi.astype("<i4").tobytes()) >= 0
        wt = tensors[info["weight_tensor"]]
        off = wt["storage"]["direct"]["offset"]
        size = int(np.prod(wt["shape"]["dimensions"]))
        r["packed_weights_reproduced"] = bool(ctx) and ctx.find(pool[off:off + size]) >= 0
        out_path = d / "y.raw"
        if out_path.exists():
            dt = np.uint16 if info["sixteen_bit"] else np.uint8
            y = np.fromfile(out_path, dt).reshape(info["output_shape"])[0].astype(np.int64)
            x = np.fromfile(d / "x.raw", dt).reshape(info["input_shape"])
            w = np.fromfile(d / "w.bin", np.int8).reshape(info["hwio"])
            qmax = 65535 if info["sixteen_bit"] else 255
            # Kernel view: the compiled Fi bias already contains the output zero point.
            _, fi_bias = split_fi(fi.reshape(-1), info["hwio"][3])
            kernel_bias = fi_bias.astype(np.int64) * (256 if info["sixteen_bit"] else 1)
            acc = accumulate(x, w, kernel_bias, info["stride"], info["pad"], info["in_zero_point"])
            r["requant_fit"] = fit_requant(acc, y, 0, qmax)
            M = info["in_scale"] * info["weight_scale"] / info["out_scale"]
            r["real_multiplier"] = M
            r["closed_form_exact"] = bool(np.array_equal(requantize(acc, M, info["sixteen_bit"]), y))
            b = np.fromfile(d / "b.bin", np.int32)
            acc_model = accumulate(x, w, b, info["stride"], info["pad"], info["in_zero_point"])
            ref = np.clip(np.rint(acc_model * M) + info["out_zero_point"], 0, qmax)
            r["float_reference_exact_fraction"] = float((ref == y).mean())
        results.append(r)
        fit = r.get("requant_fit")
        print(f"{info['source_id']:4d} {info['kernel']:34s} Fi={'OK ' if r['fi_reproduced'] else 'NO '}"
              f"W={'OK ' if r['packed_weights_reproduced'] else 'NO '}"
              f"closed_form={'EXACT' if r.get('closed_form_exact') else 'MISMATCH'} "
              f"float={r.get('float_reference_exact_fraction', 0):.4f} fit={fit}")
    (work / "conv_oracle.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
