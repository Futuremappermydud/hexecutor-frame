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
    "EXP": hxfg_writer.FN_EXP,
    "SQRT": hxfg_writer.FN_SQRT,
    "ABS": hxfg_writer.FN_ABS,
    "NEG": hxfg_writer.FN_NEG,
    "GELU": hxfg_writer.FN_GELU,
    "ELU": hxfg_writer.FN_ELU,
    "LOG": hxfg_writer.FN_LOG,
    "FLOOR": hxfg_writer.FN_FLOOR,
    "CEIL": hxfg_writer.FN_CEIL,
    "SIN": hxfg_writer.FN_SIN,
    "COS": hxfg_writer.FN_COS,
    "ROUND": hxfg_writer.FN_ROUND,
}
for _name in ACT_FUNCTIONS:
    KINDS[_name] = hxfg_writer.ACT
# HX_BINARY operations by TFLite builtin; the first group carries a fused activation option.
BINARY_OPS = {
    "ADD": hxfg_writer.BIN_ADD,
    "SUB": hxfg_writer.BIN_SUB,
    "MUL": hxfg_writer.BIN_MUL,
    "DIV": hxfg_writer.BIN_DIV,
    "MINIMUM": hxfg_writer.BIN_MIN,
    "MAXIMUM": hxfg_writer.BIN_MAX,
    "POW": hxfg_writer.BIN_POW,
    "EQUAL": hxfg_writer.BIN_EQUAL,
    "GREATER": hxfg_writer.BIN_GREATER,
    "LESS": hxfg_writer.BIN_LESS,
    "GREATER_EQUAL": hxfg_writer.BIN_GE,
    "LESS_EQUAL": hxfg_writer.BIN_LE,
}
COMPARISONS = ("EQUAL", "GREATER", "LESS", "GREATER_EQUAL", "LESS_EQUAL")
COMMUTATIVE = (hxfg_writer.BIN_ADD, hxfg_writer.BIN_MUL, hxfg_writer.BIN_MIN, hxfg_writer.BIN_MAX,
               hxfg_writer.BIN_EQUAL)
REDUCERS = {
    "MEAN": hxfg_writer.RED_MEAN,
    "SUM": hxfg_writer.RED_SUM,
    "REDUCE_MAX": hxfg_writer.RED_MAX,
    "REDUCE_MIN": hxfg_writer.RED_MIN,
}
# Operations that may produce (or pass through) int32 or bool tensors; all others are float32 only.
NON_FLOAT_OK = COMPARISONS + (
    "ARG_MAX", "ARG_MIN", "CAST", "GATHER", "SELECT", "SELECT_V2", "TILE", "RESHAPE",
)


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
                tflite.TensorType.INT64: "<i8",
                tflite.TensorType.BOOL: "?",
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
        if len(s) > 4 or any(v <= 0 for v in s):  # a scalar is one pixel, one channel
            raise ValueError(f"non-static shape {s}")
        padded = [1] * (4 - len(s)) + s
        if padded[0] != 1:
            raise ValueError("only batch one is supported")
        return padded[1:]

    graph_inputs = set(map(int, g.InputsAsNumpy()))
    dtype_codes = {
        tflite.TensorType.FLOAT32: hxfg_writer.DTYPE_FLOAT32,
        tflite.TensorType.INT32: hxfg_writer.DTYPE_INT32,
        tflite.TensorType.BOOL: hxfg_writer.DTYPE_BOOL,
    }
    temp_dtypes = {}

    def dtype(i):
        if i in temp_dtypes:
            return temp_dtypes[i]
        if i >= g.TensorsLength():  # an activation temporary: float32 unless declared
            return hxfg_writer.DTYPE_FLOAT32
        if g.Tensors(i).Type() not in dtype_codes:
            raise ValueError(f"unsupported tensor element type {g.Tensors(i).Type()}")
        return dtype_codes[g.Tensors(i).Type()]

    def new_temp(shape, code=hxfg_writer.DTYPE_FLOAT32):
        i = max(shapes) + 1
        shapes[i] = list(shape)
        temp_dtypes[i] = code
        return i

    def hwc_axis(name, axis, rank):
        """HWC axis of a TFLite tensor axis. Shapes align to the trailing (h, w, c) axes, so a
        rank-4 NHWC tensor keeps its batch axis (rejected) and lower ranks gain leading ones."""
        a = axis + rank if axis < 0 else axis
        if not 0 <= a < rank:
            raise ValueError(f"{name}: axis {axis} out of range for rank {rank}")
        if a + 3 - rank < 0:
            raise ValueError(f"{name}: batch axis unsupported")
        return a + 3 - rank

    def pad4(values, fill):
        return [fill] * (4 - len(values)) + list(map(int, values))

    def const_ints(name, i):
        if i not in constants or constants[i].dtype.kind not in "iu":
            raise ValueError(f"{name}: operand must be a constant integer tensor")
        return constants[i].reshape(-1).tolist()

    def compact_conv(op, i):
        """True when op is a dense square 3x3/5x5 CONV_2D reading tensor i, with equal strides of
        at most 2 and no dilation: the only consumer the compact three-channel input layout allows."""
        code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        if names.get(code) != "CONV_2D" or int(op.Inputs(0)) != i or op.InputsLength() < 2:
            return False
        w = shapes[int(op.Inputs(1))]  # [O, kh, kw, I]
        cfg = options(op, tflite.Conv2DOptions)
        return (len(w) == 4 and w[1] == w[2] and w[1] in (3, 5) and cfg.StrideH() == cfg.StrideW()
                and cfg.StrideH() <= 2 and cfg.DilationHFactor() == 1 and cfg.DilationWFactor() == 1)

    compact_inputs = set()
    for i in graph_inputs:
        if len(shapes[i]) == 4 and shapes[i][-1] == 3:
            users = [g.Operators(j) for j in range(g.OperatorsLength())
                     if i in map(int, g.Operators(j).InputsAsNumpy())]
            if len(users) == 1 and compact_conv(users[0], i):
                compact_inputs.add(i)

    def stride(i):
        # A three-channel graph input stays compact (stride 3) only when its sole consumer is a
        # convolution the compact-RGB path accepts; otherwise it is channel-padded like any tensor.
        return 3 if i in compact_inputs else (hwc(i)[2] + 31) & ~31

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
        if len(outs) != 1 and name not in ("SPLIT", "SPLIT_V"):
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
        if g.Tensors(out).Type() != tflite.TensorType.FLOAT32 and name not in NON_FLOAT_OK:
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
        elif name in BINARY_OPS:
            if len(ins) != 2:
                raise ValueError(f"invalid {name}")
            if any(dtype(i) != hxfg_writer.DTYPE_FLOAT32 for i in ins):
                raise ValueError(f"{name}: only float32 operands are supported")
            cfg_cls = {
                "ADD": tflite.AddOptions,
                "SUB": tflite.SubOptions,
                "MUL": tflite.MulOptions,
                "DIV": tflite.DivOptions,
            }.get(name)
            fused = activation(options(op, cfg_cls).FusedActivationFunction()) if cfg_cls else 0
            opcode = BINARY_OPS[name]
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
                p[:4] = [opcode, 0, 0, int(opcode not in COMMUTATIVE)]
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
        elif (
            name == "MEAN"
            and len(shapes[x]) == 4
            and ins[1] in constants
            and constants[ins[1]].tolist() == [1, 2]
            and hwc(out) == [1, 1, hwc(x)[2]]
        ):
            options(op, tflite.ReducerOptions)  # spatial mean keeps its dedicated operation
        elif name in REDUCERS:
            cfg = options(op, tflite.ReducerOptions)
            rank, mask = len(shapes[x]), 0
            for axis in const_ints(name, ins[1]):
                bit = 1 << hwc_axis(name, axis, rank)
                if mask & bit:
                    raise ValueError(f"{name}: repeated axes")
                mask |= bit
            if not mask:
                raise ValueError(f"{name}: empty axes are unsupported")
            hin = hwc(x)
            keep = [1 if mask & (1 << k) else hin[k] for k in range(3)]
            kind, p[:2] = hxfg_writer.REDUCE, [REDUCERS[name], mask]
            if hwc(out) != keep:
                if cfg.KeepDims():
                    raise ValueError(f"{name}: output shape disagrees with the reduction")
                # keep_dims=false drops axes: reduce into the keep-dims layout, then re-shape.
                mid = new_temp(keep)
                ops.append([kind, x, NONE, NONE, mid] + list(map(int, p)))
                source_ops.append({"tflite_op": j, "kind": name, "output": mid})
                kind, x, p = hxfg_writer.RESHAPE, mid, [0] * 11
        elif name == "RSQRT":
            mid = new_temp(shapes[out])
            ops.append([hxfg_writer.ACT, x, NONE, NONE, mid, hxfg_writer.FN_SQRT] + [0] * 10)
            source_ops.append({"tflite_op": j, "kind": name, "output": mid})
            kind, x, p[0] = hxfg_writer.ACT, mid, hxfg_writer.FN_RECIPROCAL
        elif name in ACT_FUNCTIONS:
            p[0] = ACT_FUNCTIONS[name]
            if name == "LEAKY_RELU":
                p[1] = hxfg_writer.f32_bits(options(op, tflite.LeakyReluOptions).Alpha())
            elif name == "ELU":
                p[1] = hxfg_writer.f32_bits(1.0)
            elif name == "GELU" and options(op, tflite.GeluOptions).Approximate():
                raise ValueError("GELU: only the exact (erf) form is supported")
        elif name == "SOFTMAX":
            if options(op, tflite.SoftmaxOptions).Beta() != 1.0:
                raise ValueError("SOFTMAX: beta must be one")
            kind = hxfg_writer.SOFTMAX  # per pixel over channels, TFLite's last axis
        elif name == "RESIZE_NEAREST_NEIGHBOR":
            cfg = options(op, tflite.ResizeNearestNeighborOptions)
            if cfg.AlignCorners() and cfg.HalfPixelCenters():
                raise ValueError("RESIZE_NEAREST_NEIGHBOR: align_corners with half_pixel_centers")
            if constants[ins[1]].tolist() != hwc(out)[:2] or hwc(out)[2] != hwc(x)[2]:
                raise ValueError("incorrect resize shape")
            # TFLite rounds half up (align corners) or takes floor((i + 0.5) * scale): ONNX's
            # round_prefer_ceil; plain TFLite is asymmetric with floor.
            coord = 1 if cfg.AlignCorners() else (2 if cfg.HalfPixelCenters() else 0)
            kind, p[:6] = hxfg_writer.RESIZE, [0, 0, 1, coord, 0 if coord == 0 else 2, 0]
        elif name == "TRANSPOSE":
            perm = const_ints(name, ins[1])
            rank = len(shapes[x])
            if sorted(perm) != list(range(rank)):
                raise ValueError("TRANSPOSE: invalid permutation")
            perm4 = list(range(4 - rank)) + [a + 4 - rank for a in perm]
            if perm4[0] != 0:
                raise ValueError("TRANSPOSE: batch axis must stay first")
            kind, p[:3] = hxfg_writer.TRANSPOSE, [a - 1 for a in perm4[1:]]
        elif name in ("SLICE", "STRIDED_SLICE"):
            dims = pad4(shapes[x], 1)
            begin = pad4(const_ints(name, ins[1]), 0)
            if name == "SLICE":
                size = pad4(const_ints(name, ins[2]), 1)
                size = [d - b if n < 0 else n for n, b, d in zip(size, begin, dims)]
            else:
                cfg = options(op, tflite.StridedSliceOptions)
                if cfg.EllipsisMask() or cfg.NewAxisMask() or cfg.ShrinkAxisMask() or cfg.Offset():
                    raise ValueError("STRIDED_SLICE: ellipsis, new-axis, shrink-axis masks "
                                     "and offset are unsupported")
                if any(v != 1 for v in const_ints(name, ins[3])):
                    raise ValueError("STRIDED_SLICE: only unit strides are supported")
                rank = len(shapes[x])
                end = pad4(const_ints(name, ins[2]), 1)
                begin_mask, end_mask = cfg.BeginMask() << (4 - rank), cfg.EndMask() << (4 - rank)
                begin_mask |= (1 << (4 - rank)) - 1  # padded axes are whole
                end_mask |= (1 << (4 - rank)) - 1
                begin = [0 if begin_mask >> k & 1 else (b + d if b < 0 else b)
                         for k, (b, d) in enumerate(zip(begin, dims))]
                end = [d if end_mask >> k & 1 else (e + d if e < 0 else e)
                       for k, (e, d) in enumerate(zip(end, dims))]
                begin = [min(max(b, 0), d) for b, d in zip(begin, dims)]
                end = [min(max(e, 0), d) for e, d in zip(end, dims)]
                size = [e - b for b, e in zip(begin, end)]
            if begin[0] != 0 or size[0] != 1 or min(size) <= 0:
                raise ValueError(f"{name}: batch slicing and empty slices are unsupported")
            if size[1:] != hwc(out):
                raise ValueError(f"{name}: output shape disagrees with the slice")
            kind, p[:3] = hxfg_writer.SLICE, begin[1:]
        elif name in ("SPLIT", "SPLIT_V"):
            if name == "SPLIT":
                x, axis, sizes = ins[1], const_ints(name, ins[0])[0], None
                options(op, tflite.SplitOptions)
            else:
                axis, sizes = const_ints(name, ins[2])[0], const_ints(name, ins[1])
                options(op, tflite.SplitVOptions)
            rank = len(shapes[x])
            a = hwc_axis(name, axis, rank)
            if sizes is None:
                sizes = [hwc(x)[a] // len(outs)] * len(outs)
            if -1 in sizes:
                sizes[sizes.index(-1)] = hwc(x)[a] - sum(v for v in sizes if v != -1)
            if len(sizes) != len(outs) or sum(sizes) != hwc(x)[a] or min(sizes) <= 0:
                raise ValueError(f"{name}: sizes must be positive and cover the axis")
            at = 0
            for o, n in zip(outs, sizes):
                begin = [0, 0, 0]
                begin[a] = at
                want = list(hwc(x))
                want[a] = n
                if hwc(o) != want or g.Tensors(o).Type() != tflite.TensorType.FLOAT32:
                    raise ValueError(f"{name}: output shape disagrees with the split")
                ops.append([hxfg_writer.SLICE, x, NONE, NONE, o] + begin + [0] * 8)
                source_ops.append({"tflite_op": j, "kind": name, "output": o})
                at += n
            continue
        elif name == "BATCH_MATMUL":
            cfg = options(op, tflite.BatchMatMulOptions)
            if cfg.AsymmetricQuantizeInputs() or len(ins) != 2:
                raise ValueError("BATCH_MATMUL: unsupported options")
            if ins[0] in constants or ins[1] in constants:
                raise ValueError("BATCH_MATMUL: both operands must be runtime tensors")
            y = ins[1]
            (ba, ma, ka), (bb, kb, nb) = hwc(x), hwc(y)
            if cfg.AdjX():
                ma, ka = ka, ma
            if cfg.AdjY():
                kb, nb = nb, kb
            if ka != kb or hwc(out) != [ba, ma, nb] or bb not in (1, ba):
                raise ValueError("BATCH_MATMUL: operand shapes disagree")
            kind, p[:4] = hxfg_writer.MATMUL, [int(cfg.AdjY()), 0, ba, int(cfg.AdjX())]
        elif name == "TRANSPOSE_CONV":
            if len(ins) not in (3, 4) or ins[1] not in constants or (
                len(ins) == 4 and ins[3] >= 0 and ins[3] not in constants
            ):
                raise ValueError("TRANSPOSE_CONV: constant weights and bias required")
            cfg = options(op, tflite.TransposeConvOptions)
            w, x = constants[ins[1]], ins[2]  # [co, kh, kw, ci]
            (ih, iw, ci), (oh, ow, co) = hwc(x), hwc(out)
            if w.shape[0] != co or w.shape[3] != ci:
                raise ValueError("TRANSPOSE_CONV: weights disagree with the channels")
            kh, kw = w.shape[1:3]
            sh, sw = cfg.StrideH(), cfg.StrideW()
            same = cfg.Padding() == tflite.Padding.SAME
            tot_h = max(0, (ih - 1) * sh + kh - oh) if same else 0
            tot_w = max(0, (iw - 1) * sw + kw - ow) if same else 0
            # Extra output rows/columns past the kernel footprint (stride above the kernel).
            oph, opw = oh - ((ih - 1) * sh + kh - tot_h), ow - ((iw - 1) * sw + kw - tot_w)
            zt, zl = kh - 1 - tot_h // 2, kw - 1 - tot_w // 2
            zb, zr = kh - 1 - (tot_h - tot_h // 2), kw - 1 - (tot_w - tot_w // 2)
            if min(oph, opw, zt, zl, zb, zr) < 0:
                raise ValueError("TRANSPOSE_CONV: unsupported padding/shape")
            zid = new_temp([zt + (ih - 1) * sh + 1 + zb + oph, zl + (iw - 1) * sw + 1 + zr + opw, ci])
            ops.append([hxfg_writer.ZEROINSERT, x, NONE, NONE, zid, sh, sw, zt, zl] + [0] * 7)
            source_ops.append({"tflite_op": j, "kind": name, "output": zid})
            bias = np.zeros(co, np.float32) if len(ins) == 3 or ins[3] < 0 else np.asarray(
                constants[ins[3]], np.float32).reshape(-1)
            if bias.size != co:
                raise ValueError("TRANSPOSE_CONV: bias shape mismatch")
            # Dense kernel on the zero-inserted input: flipped in both spatial axes.
            w = np.pad(
                np.asarray(w, np.float32)[:, ::-1, ::-1, :].transpose(1, 2, 3, 0),
                ((0, 0), (0, 0), (0, 0), (0, stride(out) - co)),
            )
            packed, tiles = hxfg_writer.pack_hmx_weights(w)
            kind, x = hxfg_writer.CONV, zid
            p[:11] = [kh, kw, 1, 1, 0, 0, emit(w), emit(np.pad(bias, (0, stride(out) - co))),
                      activation(cfg.FusedActivationFunction()), emit(packed, "<f2"), tiles]
        elif name == "GATHER":
            cfg = options(op, tflite.GatherOptions)
            table = constants.get(ins[0])
            if table is None or table.ndim != 2 or ins[1] in constants or cfg.BatchDims():
                raise ValueError("GATHER: only a constant [vocab, width] table with runtime "
                                 "indices is supported")
            if cfg.Axis() != 0 or dtype(ins[1]) != hxfg_writer.DTYPE_INT32:
                raise ValueError("GATHER: only axis 0 with int32 indices is supported")
            vocab, width = table.shape
            row = (width + 31) & ~31
            rows = np.pad(np.asarray(table, np.float32), ((0, 0), (0, row - width)))
            if hwc(out)[2] != width or int(np.prod(hwc(out)[:2])) != int(np.prod(shapes[ins[1]])):
                raise ValueError("GATHER: output shape disagrees with the indices")
            x, y = ins[1], NONE
            kind, p[:10] = hxfg_writer.GATHER, [1, emit(rows), vocab, 0, 0, width, 1, row, 0, 0]
        elif name in ("ARG_MAX", "ARG_MIN"):
            axis = const_ints(name, ins[1])[0]
            rank = len(shapes[x])
            if axis % rank != rank - 1:
                raise ValueError(f"{name}: only the last (channel) axis is supported")
            if g.Tensors(out).Type() != tflite.TensorType.INT32:
                raise ValueError(f"{name}: output type must be int32")
            kind, p[:2] = (hxfg_writer.ARGMAX if name == "ARG_MAX" else hxfg_writer.ARGMIN), [0, 1]
            h_, w_, _ = hwc(x)
            if hwc(out) != [h_, w_, 1]:  # the axis is dropped: arg into [h, w, 1], then re-shape
                mid = new_temp([h_, w_, 1], hxfg_writer.DTYPE_INT32)
                ops.append([kind, x, NONE, NONE, mid] + list(map(int, p)))
                source_ops.append({"tflite_op": j, "kind": name, "output": mid})
                kind, x, p = hxfg_writer.RESHAPE, mid, [0] * 11
        elif name == "TILE":
            reps = pad4(const_ints(name, ins[1]), 1)
            hi, wi, ci = hwc(x)
            ho, wo, co = hwc(out)
            if reps[0] != 1 or dtype(x) != dtype(out) or (ho, wo, co) != (hi * reps[1], wi * reps[2], ci * reps[3]):
                raise ValueError("TILE: unsupported repeats or element type")
            kind = hxfg_writer.TILE
        elif name == "CAST":
            src, dst = dtype(x), dtype(out)
            modes = {(0, 1): 0, (1, 0): 1, (2, 0): 2, (0, 2): 3, (1, 2): 4, (2, 1): 5}
            kind, p[0] = hxfg_writer.CAST, modes.get((src, dst), 6 if src == dst else None)
            if p[0] is None:
                raise ValueError("CAST: unsupported element types")
        elif name in ("SELECT", "SELECT_V2"):
            if len(ins) != 3 or ins[0] in constants or dtype(ins[0]) != hxfg_writer.DTYPE_BOOL:
                raise ValueError(f"{name}: a runtime bool condition is required")
            if not dtype(ins[1]) == dtype(ins[2]) == dtype(out):
                raise ValueError(f"{name}: operand element types differ")
            if name == "SELECT" and not hwc(ins[0]) == hwc(ins[1]) == hwc(ins[2]) == hwc(out):
                raise ValueError("SELECT: operands must share one shape (use SELECT_V2)")
            oshape, code = hwc(out), dtype(out)

            def operand(i):
                if i not in constants:
                    return 0, 0, i
                arr = np.asarray(constants[i], np.float32 if code == 0 else np.int32)
                if arr.size == 1:
                    bits = hxfg_writer.f32_bits(arr.reshape(-1)[0]) if code == 0 else int(
                        arr.reshape(-1)[0]) & 0xFFFFFFFF
                    return 1, bits, NONE
                full = np.broadcast_to(arr.reshape(pad4(shapes[i], 1)), pad4(oshape, 1))[0]
                pad = ((oshape[2] + 31) & ~31) - oshape[2]
                return 3, emit(np.pad(full, ((0, 0), (0, 0), (0, pad)))), NONE

            (xm, xv, y), (ym, yv, z) = operand(ins[1]), operand(ins[2])
            x, kind, p[:4] = ins[0], hxfg_writer.WHERE, [xm, xv, ym, yv]
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
        dtype,
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
