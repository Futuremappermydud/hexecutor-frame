#!/usr/bin/env python3
"""Offline, fail-closed lowering of a static QLinear ONNX graph to float HXFG v1.

Supports the operator-level quantized form written by onnxruntime's static
quantizer (QuantizeLinear, QLinearConv, QLinearAdd, DequantizeLinear) plus a
float spatial ReduceMean and Gemm head. Weights are dequantized offline; the
DSP runs the float/FP16 tensor backend, so results approximate rather than
reproduce the u8 arithmetic.

The caller names the float tensor that becomes the graph input (after any
preprocessing the host performs) and the float tensor that becomes the graph
output (before any host-side postprocessing). Fused activations are inferred
from each quantized output's representable range:

  zero point 0, range 6.0      -> ReLU6
  zero point 0, other range    -> ReLU (the range clamp is not reproduced)
  nonzero zero point           -> none (linear; saturation not reproduced)

Every inferred clamp is recorded in the manifest so callers can see where the
float graph may differ from the quantized one.

Requires numpy and onnx. Nothing here is used by the native host or engine.
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

try:
    from . import hxfg_writer as hx
except ImportError:  # run as a script from hexecutor/offline
    import hxfg_writer as hx


def lower(source, destination, input_name, output_name, pad_input_channels=0):
    model = onnx.load(str(source))
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    dims = {}
    for v in list(inferred.value_info) + list(inferred.input) + list(inferred.output):
        shape = v.type.tensor_type.shape.dim
        if any(not d.HasField("dim_value") for d in shape):
            continue
        dims[v.name] = [d.dim_value for d in shape]

    def const(name, what):
        if name not in init:
            raise ValueError(f"{what} must be a constant initializer")
        return init[name]

    def scalar(name, what):
        v = const(name, what)
        if v.size != 1:
            raise ValueError(f"{what} must be per-tensor")
        return v.reshape(()).item()

    ids, shapes = {}, {}
    alias = {}  # quantized or float name -> canonical tensor name

    def canon(name):
        while name in alias:
            name = alias[name]
        return name

    def tid(name):
        name = canon(name)
        if name not in ids:
            ids[name] = len(ids)
        return ids[name]

    def hwc_of_onnx(name):
        s = dims.get(name)
        if s is None:
            raise ValueError(f"no static shape for {name}")
        if len(s) == 4:
            if s[0] != 1:
                raise ValueError("only batch one is supported")
            return [s[2], s[3], s[1]]
        if len(s) == 2 and s[0] == 1:
            return [1, 1, s[1]]
        raise ValueError(f"unsupported rank for {name}: {s}")

    def set_shape(name, hwc):
        shapes[tid(name)] = list(map(int, hwc))

    graph_input = tid(input_name)
    in_hwc = hwc_of_onnx(input_name)
    if pad_input_channels:
        if pad_input_channels < in_hwc[2]:
            raise ValueError("--pad-input-channels is smaller than the input")
        in_hwc[2] = pad_input_channels
    set_shape(input_name, in_hwc)
    quant = {}  # canonical name -> (scale, zero point)

    def stride(i):
        c = shapes[i][2]
        return 3 if i == graph_input and c == 3 else (c + 31) & ~31

    def hwc(i):
        return shapes[i]

    def activation(scale, zero_point, dtype):
        if dtype != np.uint8:
            raise ValueError("only uint8 activations are supported")
        lo, hi = (0 - zero_point) * scale, (255 - zero_point) * scale
        if zero_point == 0:
            act = hx.ACT_RELU6 if abs(hi - 6.0) < 1e-3 else hx.ACT_RELU
        else:
            act = hx.ACT_NONE
        return act, [float(lo), float(hi)]

    pool = hx.ConstantPool()
    ops, source_ops, clamps = [], [], []
    output_parts = None
    started, finished = False, False
    for j, node in enumerate(g.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if finished:
            break
        if not started:
            if node.op_type == "QuantizeLinear" and canon(node.input[0]) == canon(
                input_name
            ):
                started = True
            else:
                continue
        op = node.op_type
        p = [0] * 11
        if op == "QuantizeLinear":
            if canon(node.input[0]) != canon(input_name):
                raise ValueError("QuantizeLinear is only supported at the graph input")
            alias[node.output[0]] = node.input[0]
            quant[canon(input_name)] = (
                scalar(node.input[1], "input scale"),
                scalar(node.input[2], "input zero point"),
            )
            continue
        if op == "DequantizeLinear":
            if canon(node.input[0]) not in quant:
                raise ValueError("DequantizeLinear of an unknown tensor")
            alias[node.output[0]] = node.input[0]
            if node.output[0] == output_name:
                finished = True
            continue
        out = node.output[0]
        if op == "QLinearConv":
            if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
                raise ValueError("auto_pad is unsupported")
            if list(attrs.get("dilations", [1, 1])) != [1, 1]:
                raise ValueError("dilated convolution unsupported")
            x = node.input[0]
            xs = scalar(node.input[1], "x scale")
            w = const(node.input[3], "weights").astype(np.float32)
            ws = const(node.input[4], "weight scale").astype(np.float32).reshape(-1)
            wzp = const(node.input[5], "weight zero point").astype(np.float32).reshape(-1)
            ys = scalar(node.input[6], "y scale")
            yzp_arr = const(node.input[7], "y zero point")
            yzp = int(yzp_arr.reshape(()).item())
            bias = (
                const(node.input[8], "bias").astype(np.float64)
                if len(node.input) > 8 and node.input[8]
                else np.zeros(w.shape[0])
            )
            co, ci_g, kh, kw = w.shape
            if ws.size == 1:
                ws = np.full(co, ws[0], np.float32)
            if wzp.size == 1:
                wzp = np.full(co, wzp[0], np.float32)
            if ws.size != co or wzp.size != co or bias.size != co:
                raise ValueError("per-channel quantization size mismatch")
            wf = (w - wzp[:, None, None, None]) * ws[:, None, None, None]
            bf = (bias * xs * ws.astype(np.float64)).astype(np.float32)
            sh, sw = attrs.get("strides", [1, 1])
            pt, pl, pb, pr = attrs.get("pads", [0, 0, 0, 0])
            ih, iw, ic = hwc(tid(x))
            oh, ow = (ih + pt + pb - kh) // sh + 1, (iw + pl + pr - kw) // sw + 1
            if out in dims and [oh, ow, co] != hwc_of_onnx(out):
                raise ValueError(f"convolution shape mismatch at node {j}")
            set_shape(out, [oh, ow, co])
            ostride = stride(tid(out))
            group = attrs.get("group", 1)
            act, rng = activation(ys, yzp, yzp_arr.dtype)
            if group == 1:
                kind = hx.CONV
                if ci_g < ic:  # padded graph input channels
                    wf = np.pad(wf, ((0, 0), (0, ic - ci_g), (0, 0), (0, 0)))
                elif ci_g != ic:
                    raise ValueError("input channel mismatch")
                wk = np.pad(
                    wf.transpose(2, 3, 1, 0), ((0, 0), (0, 0), (0, 0), (0, ostride - co))
                )
            elif group == co == ic and ci_g == 1:
                kind = hx.DEPTHWISE
                wk = np.pad(wf[:, 0].transpose(1, 2, 0), ((0, 0), (0, 0), (0, ostride - co)))
            else:
                raise ValueError("only dense or multiplier-one depthwise groups supported")
            p[:9] = [
                kh,
                kw,
                sh,
                sw,
                pt,
                pl,
                pool.emit(wk),
                pool.emit(np.pad(bf, (0, ostride - co))),
                act,
            ]
            if kind == hx.CONV:
                packed, tiles = hx.pack_hmx_weights(wk)
                if hx.hmx_fits(tiles, ostride):
                    p[9], p[10] = pool.emit(packed, "<f2"), tiles
            quant[canon(out)] = (ys, yzp)
            clamps.append({"node": j, "op": op, "activation": act, "quantized_range": rng})
            ops.append([kind, tid(x), hx.NONE, hx.NONE, tid(out)] + p)
        elif op == "QLinearAdd":
            a, b = node.input[0], node.input[3]
            if hwc(tid(a)) != hwc(tid(b)) or (
                out in dims and hwc(tid(a)) != hwc_of_onnx(out)
            ):
                raise ValueError("broadcast add unsupported")
            ys = scalar(node.input[6], "y scale")
            yzp_arr = const(node.input[7], "y zero point")
            yzp = int(yzp_arr.reshape(()).item())
            act, rng = activation(ys, yzp, yzp_arr.dtype)
            set_shape(out, hwc(tid(a)))
            p[0] = act
            quant[canon(out)] = (ys, yzp)
            clamps.append({"node": j, "op": op, "activation": act, "quantized_range": rng})
            ops.append([hx.ADD, tid(a), tid(b), hx.NONE, tid(out)] + p)
        elif op == "ReduceMean":
            axes = list(attrs.get("axes", []))
            if axes != [2, 3] or attrs.get("keepdims", 1) != 0:
                raise ValueError("only spatial ReduceMean without keepdims supported")
            c = hwc(tid(node.input[0]))[2]
            set_shape(out, [1, 1, c])
            ops.append([hx.MEAN, tid(node.input[0]), hx.NONE, hx.NONE, tid(out)] + p)
        elif op == "Gemm":
            if (
                attrs.get("transA", 0) != 0
                or attrs.get("transB", 0) != 1
                or attrs.get("alpha", 1.0) != 1.0
                or attrs.get("beta", 1.0) != 1.0
            ):
                raise ValueError("only Gemm with transB=1 and unit alpha/beta supported")
            x = node.input[0]
            wmat = const(node.input[1], "Gemm B").astype(np.float32)
            n, k = wmat.shape
            if hwc(tid(x)) != [1, 1, k]:
                raise ValueError("Gemm input must be a 1x1 spatial vector")
            bias = const(node.input[2], "Gemm C").astype(np.float32).reshape(-1)
            tiles = (k + 31) // 32
            # The FP16 path needs HMX weights on every convolution, and one
            # convolution's packed weights must fit the per-op HMX budget.
            # Split wide outputs into several graph outputs; the host joins them.
            chunk = max(
                (
                    c
                    for c in range(32, min((n + 31) & ~31, hx.MAX_CHANNELS) + 1, 32)
                    if hx.hmx_fits(tiles, c)
                ),
                default=0,
            )
            if not chunk:
                raise ValueError("Gemm input is too deep for the HMX budget")
            if chunk >= n:
                parts = [(out, 0, n)]
            else:
                if out != output_name:
                    raise ValueError("a split Gemm must produce the graph output")
                parts = [
                    (f"{out}#{i}", lo, min(lo + chunk, n))
                    for i, lo in enumerate(range(0, n, chunk))
                ]
            for name, lo, hi in parts:
                set_shape(name, [1, 1, hi - lo])
                ostride = stride(tid(name))
                wk = np.pad(
                    wmat[lo:hi].T[None, None],
                    ((0, 0), (0, 0), (0, 0), (0, ostride - (hi - lo))),
                )
                q = [0] * 11
                q[:9] = [
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    pool.emit(wk),
                    pool.emit(np.pad(bias[lo:hi], (0, ostride - (hi - lo)))),
                    hx.ACT_NONE,
                ]
                packed, t = hx.pack_hmx_weights(wk)
                q[9], q[10] = pool.emit(packed, "<f2"), t
                ops.append([hx.CONV, tid(x), hx.NONE, hx.NONE, tid(name)] + q)
            output_parts = [{"tensor": name, "channels": [lo, hi]} for name, lo, hi in parts]
        else:
            raise ValueError(f"unsupported operation {j}: {op}")
        source_ops.append({"onnx_node": j, "kind": op, "output": tid(out)})
        if out == output_name:
            finished = True
    if not started:
        raise ValueError(f"no QuantizeLinear consumes {input_name}")
    if not finished:
        raise ValueError(f"output {output_name} was not reached")
    if output_parts is None:
        output_parts = [{"tensor": output_name, "channels": [0, hwc(tid(output_name))[2]]}]
    out_ids = [tid(part["tensor"]) for part in output_parts]
    if any(op[1] in out_ids or op[2] in out_ids for op in ops):
        raise ValueError("the output tensor must not feed later lowered operations")
    full_shapes = {i: list(s) for i, s in shapes.items()}
    return hx.write(
        ops,
        [graph_input],
        out_ids,
        hwc,
        stride,
        pool,
        destination,
        {
            "source": str(Path(source).resolve()),
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "ops": source_ops,
            "shapes": full_shapes,
            "extra": {
                "input_tensor": input_name,
                "output_tensor": output_name,
                "input_channels": hwc_of_onnx(input_name)[2],
                "output_parts": output_parts,
                "activation_ranges": clamps,
            },
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path, help="destination prefix (.hxfg/.bin/.json)")
    p.add_argument("--input", required=True, help="float ONNX tensor fed to the graph")
    p.add_argument("--output-tensor", required=True, help="float ONNX tensor returned")
    p.add_argument(
        "--pad-input-channels",
        type=int,
        default=0,
        help="zero-pad graph input channels (3 selects the compact input layout)",
    )
    a = p.parse_args()
    m = lower(a.model, a.output, a.input, a.output_tensor, a.pad_input_channels)
    print(
        f"{len(m['ops'])} operations, arena {m['arena_bytes']} bytes, "
        f"constants {m['constant_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
