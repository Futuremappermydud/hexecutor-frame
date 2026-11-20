#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline, fail-closed lowering of a static float TFLite graph to HXFG v1.

Requires numpy and the generated `tflite` Python schema package. No TFLite,
FlatBuffers, or Python dependency is needed by the native host or DSP engine.
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import tflite

try:
    from . import hxfg_writer
except ImportError:  # run as a script from hexecutor/offline
    import hxfg_writer

NONE = hxfg_writer.NONE
KINDS = {
    name: i + 1
    for i, name in enumerate(
        (
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "ADD",
            "PRELU",
            "MAX_POOL_2D",
            "PAD",
            "RESIZE_BILINEAR",
            "RESHAPE",
            "CONCATENATION",
            "MEAN",
            "LOGISTIC",
        )
    )
}
# Operations added after the first eleven; values are the HX_* kinds in graph.h.
for _name in ("HARD_SWISH", "TANH", "LEAKY_RELU", "RELU", "RELU6"):
    KINDS[_name] = hxfg_writer.ACT
for _name in ("SUB", "MUL"):
    KINDS[_name] = hxfg_writer.BINARY
KINDS["AVERAGE_POOL_2D"] = hxfg_writer.AVGPOOL
ACT_FUNCTIONS = {
    "HARD_SWISH": hxfg_writer.FN_HARDSWISH,
    "TANH": hxfg_writer.FN_TANH,
    "LEAKY_RELU": hxfg_writer.FN_LEAKY,
    "RELU": hxfg_writer.FN_RELU,
    "RELU6": hxfg_writer.FN_RELU6,
}


def lower(source, destination):
    model = tflite.Model.GetRootAsModel(source.read_bytes(), 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("exactly one subgraph is supported")
    g = model.Subgraphs(0)
    constants = {}
    shapes = {}
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        if t.IsVariable():
            raise ValueError("variable tensors are unsupported")
        shape = list(map(int, t.ShapeAsNumpy()))
        shapes[i] = shape
        raw = model.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(raw, np.ndarray) and raw.size:
            dtype = {
                tflite.TensorType.FLOAT32: "<f4",
                tflite.TensorType.FLOAT16: "<f2",
                tflite.TensorType.INT32: "<i4",
            }.get(t.Type())
            if dtype is None:
                raise ValueError(f"unsupported constant type {t.Type()}")
            constants[i] = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)
    names = {
        v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)
    }
    ops, source_ops = [], []
    pool = hxfg_writer.ConstantPool()
    emit = pool.emit

    def options(op, cls):
        expected = getattr(tflite.BuiltinOptions, cls.__name__)
        if op.BuiltinOptionsType() != expected:
            raise ValueError("incorrect builtin options")
        table = op.BuiltinOptions()
        obj = cls()
        obj.Init(table.Bytes, table.Pos)
        return obj

    def hwc(i):
        s = shapes[i]
        if not 1 <= len(s) <= 4 or any(v <= 0 for v in s):
            raise ValueError(f"non-static shape {s}")
        padded = [1] * (4 - len(s)) + s
        if padded[0] != 1:
            raise ValueError("only batch one is supported")
        return padded[1:]

    graph_inputs = set(map(int, g.InputsAsNumpy()))

    def stride(i):
        return 3 if i in graph_inputs and hwc(i)[2] == 3 else (hwc(i)[2] + 31) & ~31

    def channel_constant(i, count):
        a = np.asarray(constants[i], dtype=np.float32).reshape(-1)
        if a.size != count:
            raise ValueError("channel constant shape mismatch")
        return np.pad(a, (0, ((count + 31) & ~31) - count))

    def activation(value):
        if value not in (0, 1, 2, 3):
            raise ValueError(f"unsupported fused activation {value}")
        return value

    def padding(a, b, kh, kw, sh, sw, mode):
        ih, iw, _ = hwc(a)
        oh, ow, _ = hwc(b)
        if mode == tflite.Padding.SAME:
            if (oh, ow) != ((ih + sh - 1) // sh, (iw + sw - 1) // sw):
                raise ValueError("invalid SAME shape")
            return (
                max(0, (oh - 1) * sh + kh - ih) // 2,
                max(0, (ow - 1) * sw + kw - iw) // 2,
            )
        if mode == tflite.Padding.VALID and (oh, ow) == (
            (ih - kh) // sh + 1,
            (iw - kw) // sw + 1,
        ):
            return 0, 0
        raise ValueError("unsupported padding/shape")

    for j in range(g.OperatorsLength()):
        op = g.Operators(j)
        code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        name = names.get(code, str(code))
        ins, outs = list(map(int, op.InputsAsNumpy())), list(
            map(int, op.OutputsAsNumpy())
        )
        if len(outs) != 1:
            raise ValueError("one output per operation required")
        out = outs[0]
        if name == "DEQUANTIZE":
            if (
                len(ins) != 1
                or ins[0] not in constants
                or constants[ins[0]].dtype != np.float16
            ):
                raise ValueError("only constant fp16 dequantization is supported")
            constants[out] = constants[ins[0]].astype(np.float32)
            continue
        if g.Tensors(out).Type() != tflite.TensorType.FLOAT32:
            raise ValueError("only float32 activations supported")
        kind, x, y, z = KINDS.get(name), ins[0], NONE, NONE
        p = [0] * 11
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            expected_inputs = 3
            if len(ins) != expected_inputs or any(i not in constants for i in ins[1:]):
                raise ValueError("constant weights and bias required")
            w = constants[ins[1]]
            co = hwc(out)[2]
            if name == "FULLY_CONNECTED":
                cfg = options(op, tflite.FullyConnectedOptions)
                if (
                    cfg.WeightsFormat() != 0
                    or cfg.KeepNumDims()
                    or cfg.AsymmetricQuantizeInputs()
                ):
                    raise ValueError("unsupported fully connected options")
                w = w[:, None, None, :]
                sh = sw = 1
                ph = pw = 0
                kind = KINDS["CONV_2D"]
                kh = kw = 1
            else:
                cls = (
                    tflite.Conv2DOptions
                    if name == "CONV_2D"
                    else tflite.DepthwiseConv2DOptions
                )
                cfg = options(op, cls)
                if cfg.DilationHFactor() != 1 or cfg.DilationWFactor() != 1:
                    raise ValueError("dilated convolution unsupported")
                if name == "DEPTHWISE_CONV_2D" and cfg.DepthMultiplier() != 1:
                    raise ValueError("depth multiplier must be one")
                kh, kw = w.shape[1:3]
                sh, sw = cfg.StrideH(), cfg.StrideW()
                ph, pw = padding(x, out, kh, kw, sh, sw, cfg.Padding())
            if name == "DEPTHWISE_CONV_2D":
                w = np.pad(w[0], ((0, 0), (0, 0), (0, stride(out) - co)))
            else:
                w = np.pad(
                    w.transpose(1, 2, 3, 0),
                    ((0, 0), (0, 0), (0, 0), (0, stride(out) - co)),
                )
            p[:9] = [
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                emit(w),
                emit(channel_constant(ins[2], co)),
                activation(cfg.FusedActivationFunction()),
            ]
            if name != "DEPTHWISE_CONV_2D":
                packed, tiles = hxfg_writer.pack_hmx_weights(w)
                p[9], p[10] = emit(packed, "<f2"), tiles
        elif name in ("ADD", "SUB", "MUL"):
            if len(ins) != 2:
                raise ValueError(f"invalid {name}")
            cfg_cls = {
                "ADD": tflite.AddOptions,
                "SUB": tflite.SubOptions,
                "MUL": tflite.MulOptions,
            }[name]
            fused = activation(options(op, cfg_cls).FusedActivationFunction())
            opcode = {"ADD": hxfg_writer.BIN_ADD, "SUB": hxfg_writer.BIN_SUB,
                      "MUL": hxfg_writer.BIN_MUL}[name]
            a, b = ins
            oshape = hwc(out)

            def fits(small):  # each of h, w, c is the output's or one
                return all(s in (1, o) for s, o in zip(hwc(small), oshape))

            if a in constants and b in constants:
                raise ValueError("both operands constant")
            if a in constants or b in constants:
                const_i, tensor = (a, b) if a in constants else (b, a)
                if hwc(tensor) != oshape or hwc(const_i)[:2] != [1, 1] or not fits(const_i):
                    raise ValueError("constant operand must be per-channel")
                vec = np.broadcast_to(
                    np.asarray(constants[const_i], dtype=np.float32).reshape(-1),
                    (oshape[2],),
                )
                x, y, kind = tensor, NONE, hxfg_writer.BINARY
                p[:4] = [opcode, 1, emit(np.pad(vec, (0, ((oshape[2] + 31) & ~31) - oshape[2]))),
                         int(const_i == a)]
            elif name == "ADD" and shapes[a] == shapes[b] == shapes[out]:
                x, y, kind = a, b, hxfg_writer.ADD
                p[0] = fused
                fused = 0
            elif hwc(a) == oshape and fits(b):
                x, y, kind = a, b, hxfg_writer.BINARY
                p[:4] = [opcode, 0, 0, 0]
            elif hwc(b) == oshape and fits(a):
                x, y, kind = b, a, hxfg_writer.BINARY
                p[:4] = [opcode, 0, 0, int(opcode == hxfg_writer.BIN_SUB)]
            else:
                raise ValueError("operands broadcast against each other")
            if fused:
                # Binary operations carry no fused activation: run it as a separate operation
                # on a fresh tensor that takes the place of the output.
                mid = max(shapes) + 1
                shapes[mid] = list(shapes[out])
                ops.append([kind, x, y, z, mid] + list(map(int, p)))
                source_ops.append({"tflite_op": j, "kind": name, "output": mid})
                kind, x, y, p = hxfg_writer.ACT, mid, NONE, [0] * 11
                p[0] = hxfg_writer.FN_RELU if fused == 1 else hxfg_writer.FN_RELU6
                if fused == 2:
                    raise ValueError("RELU_N1_TO_1 is unsupported after a binary operation")
        elif name == "PRELU":
            if len(ins) != 2:
                raise ValueError("invalid PReLU")
            p[0] = emit(channel_constant(ins[1], hwc(x)[2]))
        elif name == "MAX_POOL_2D":
            cfg = options(op, tflite.Pool2DOptions)
            if cfg.FusedActivationFunction() != 0:
                raise ValueError("pool activation unsupported")
            kh, kw, sh, sw = (
                cfg.FilterHeight(),
                cfg.FilterWidth(),
                cfg.StrideH(),
                cfg.StrideW(),
            )
            ph, pw = padding(x, out, kh, kw, sh, sw, cfg.Padding())
            p[:6] = [kh, kw, sh, sw, ph, pw]
        elif name == "PAD":
            pads = constants[ins[1]]
            if pads.shape != (4, 2) or np.any(pads < 0) or np.any(pads[0]):
                raise ValueError("unsupported padding")
            if (np.array(shapes[x]) + pads.sum(axis=1)).tolist() != shapes[out]:
                raise ValueError("incorrect pad shape")
            p[:3] = pads[1:, 0].tolist()
        elif name == "RESIZE_BILINEAR":
            cfg = options(op, tflite.ResizeBilinearOptions)
            if constants[ins[1]].tolist() != hwc(out)[:2]:
                raise ValueError("incorrect resize shape")
            p[:2] = [int(cfg.AlignCorners()), int(cfg.HalfPixelCenters())]
        elif name == "RESHAPE":
            if np.prod(shapes[x]) != np.prod(shapes[out]):
                raise ValueError("invalid reshape")
        elif name == "CONCATENATION":
            cfg = options(op, tflite.ConcatenationOptions)
            if len(ins) != 2 or cfg.FusedActivationFunction() != 0:
                raise ValueError("only two-input plain concatenation supported")
            y = ins[1]
            axis = cfg.Axis() % len(shapes[out])
            p[0] = axis + 3 - len(shapes[out])
            if p[0] < 0:
                raise ValueError("batch concatenation unsupported")
        elif name == "MEAN":
            cfg = options(op, tflite.ReducerOptions)
            if constants[ins[1]].tolist() != [1, 2] or hwc(out) != [1, 1, hwc(x)[2]]:
                raise ValueError("only spatial mean supported")
        elif name in ACT_FUNCTIONS:
            p[0] = ACT_FUNCTIONS[name]
            if name == "LEAKY_RELU":
                p[1] = hxfg_writer.f32_bits(options(op, tflite.LeakyReluOptions).Alpha())
        elif name == "AVERAGE_POOL_2D":
            cfg = options(op, tflite.Pool2DOptions)
            if cfg.FusedActivationFunction() != 0:
                raise ValueError("pool activation unsupported")
            kh, kw, sh, sw = (
                cfg.FilterHeight(),
                cfg.FilterWidth(),
                cfg.StrideH(),
                cfg.StrideW(),
            )
            ph, pw = padding(x, out, kh, kw, sh, sw, cfg.Padding())
            p[:7] = [kh, kw, sh, sw, ph, pw, 0]  # TFLite averages over valid taps only
        elif name != "LOGISTIC":
            raise ValueError(f"unsupported operation {j}: {name}")
        ops.append([kind, x, y, z, out] + list(map(int, p)))
        source_ops.append({"tflite_op": j, "kind": name, "output": out})
    inputs, outputs = list(map(int, g.InputsAsNumpy())), list(
        map(int, g.OutputsAsNumpy())
    )
    return hxfg_writer.write(
        ops,
        inputs,
        outputs,
        hwc,
        stride,
        pool,
        destination,
        {
            "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "ops": source_ops,
            "shapes": shapes,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument(
        "output", type=Path, help="output prefix, in an ignored artifact directory"
    )
    args = p.parse_args()
    result = lower(args.model, args.output)
    print(
        f'{len(result["ops"])} ops, {result["arena_bytes"]} arena bytes, {result["constant_bytes"]} constant bytes'
    )


if __name__ == "__main__":
    main()
