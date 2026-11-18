#!/usr/bin/env python3
"""Offline, fail-closed lowering of a static float32 ONNX graph to HXFG v1.

This is the front end for ordinary (non-QLinear) ONNX exports, for example PyTorch
`torch.onnx.export` output. Shapes must be static with batch one. Everything that
HXFG cannot express raises ValueError naming the node; nothing falls back silently.

BatchNorm folding: an inference BatchNormalization (and a per-channel Mul, Add, Sub or
Div by a constant) that directly follows a Conv or Gemm/MatMul whose output has no other
user and no activation yet is folded into that layer's weights and bias. A BatchNorm
that cannot be folded (after an Add, a pool, a graph input, or a layer with several
users) becomes a per-channel multiply and add.

Pattern rewrites: Sigmoid feeding a Mul with its own input becomes SiLU; Relu and
Clip(0, 6) or Clip(0, inf) fold into a preceding single-user Conv, depthwise Conv or
same-shape Add, and otherwise run as a standalone activation. Reshape-like nodes
(Flatten, Reshape, Squeeze, Unsqueeze, Identity, Dropout) are aliases when their HWC shape is
unchanged; otherwise they become an NCHW flatten and unflatten (HX_RESHAPE flags 1 and 2), so
spatial and channel order are preserved although HXFG stores tensors as HWC. Rank-4 batch-one
Transpose with a batch-first perm is HX_TRANSPOSE.

Supported operators: Conv, Gemm, MatMul (constant right operand), BatchNormalization,
Add, Sub, Mul, Div (constant divisor), Relu, Clip, LeakyRelu, PRelu, Sigmoid, Tanh,
HardSigmoid, HardSwish, Gelu (exact), Exp, Sqrt, Abs, Neg, Erf (standalone activations),
Softmax (axis 1 only), InstanceNormalization, GroupNormalization (opset 18 or 21 scale form), LayerNormalization (over [1, C] channels, or over (C, H, W) with axis 1; Mean and InvStdDev unused), MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool, ReduceMean/ReduceSum/ReduceMax/ReduceMin over any non-batch axes (keepdims layout; the (2, 3) mean is HX_MEAN), Concat, Pad, Resize (linear, or nearest with asymmetric coordinates and floor), Identity, Dropout, Flatten,
Reshape, Squeeze, Unsqueeze, Transpose (rank 4, batch first), Slice (unit steps), Split (channel
axis) and grouped Conv (each group is a Slice, a dense Conv and a channel Concat chain).

Exporter hygiene: before lowering, every node whose inputs are all constants (Identity-wrapped
initializers, Shape/Gather/Unsqueeze/Concat size chains, scalar arithmetic; see `FOLDABLE`) is
evaluated with onnx.reference and replaced by an initializer, Shape of a statically shaped
tensor becomes a constant, and nodes whose outputs are no longer read are dropped. Shape
inference runs with data propagation and again on the folded graph, so computed Resize sizes
resolve. Cast to float32 is an alias. A squeezed rank-3 `[1, C, L]` intermediate is NCHW with
H = 1. The packed constant pool is limited to 32 MiB, the tensor runtime's own limit.

The manifest's `fp16_plan` entry lists what keeps the graph out of the DSP's blocked
FP16 plan (see `fp16_blockers`); such a graph still lowers and runs on the FP32 path.

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

NONE = hx.NONE
# hx_graph_open (hexecutor/native/tensor/graph.c) rejects constant pools above 32 MiB.
CONSTANT_POOL_LIMIT = 0x4000000


def _pad32(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.pad(values, (0, ((values.size + 31) & ~31) - values.size))


class Op:
    """One HXFG operation under construction. Layers keep unpacked weights until the end."""

    def __init__(self, kind, x, y, out, p=(), node=None):
        self.kind, self.x, self.y, self.out = kind, x, y, out
        self.p = (list(p) + [0] * 11)[:11]
        self.node = node
        self.weights = None  # layers: float32 [out, in/group, kh, kw]
        self.bias = None
        self.foldable = False  # a following affine node may be folded in
        self.fused = False  # an activation has been fused
        self.dilation = (1, 1)  # conv/depthwise only; encoded as the HXFG z field

    @property
    def z(self):
        """HXFG z: NONE for dilation (1, 1), else (dh << 8) | dw."""
        dh, dw = self.dilation
        return NONE if (dh, dw) == (1, 1) else (dh << 8) | dw


def _infer_shapes(model):
    """Shape inference with data propagation, so Shape/Gather/Concat size chains resolve."""
    return onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True).graph


def _static_dims(graph):
    """Tensor name -> static dimensions (None for non-float32 values, absent when dynamic)."""
    dims = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        shape = v.type.tensor_type.shape.dim
        if v.type.tensor_type.elem_type not in (0, onnx.TensorProto.FLOAT):
            dims[v.name] = None  # non-float32 value
        elif all(d.HasField("dim_value") for d in shape):
            dims[v.name] = [d.dim_value for d in shape]
    return dims


class Lowerer:
    def __init__(self, model, compact_input="auto"):
        self.model = model
        self.graph = model.graph
        inferred = _infer_shapes(model)
        self.init = {i.name: numpy_helper.to_array(i) for i in self.graph.initializer}
        self.nodes = []
        for node in self.graph.node:  # Constant nodes become initializers
            if node.op_type == "Constant":
                self.init[node.output[0]] = self._constant_value(node)
            else:
                self.nodes.append(node)
        self.dims = _static_dims(inferred)
        self._fold_constants()
        self._refresh_dims()
        self.compact_input = compact_input
        self.alias, self.ids, self.shapes, self.ops = {}, {}, {}, []
        self.producer = {}  # tensor id -> Op
        self.pool = hx.ConstantPool()
        self.source_ops = []
        self.inputs = [i.name for i in self.graph.input if i.name not in self.init]
        self.outputs = [o.name for o in self.graph.output]
        self.users = {}
        for n in self.nodes:
            for name in n.input:
                if name:
                    self.users[name] = self.users.get(name, 0) + 1
        for name in self.outputs:
            self.users[name] = self.users.get(name, 0) + 1
        self.output_parts = None
        self.synthetic = 0

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _constant_value(node):
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
            if a.name in ("value_float", "value_int"):
                return np.asarray(onnx.helper.get_attribute_value(a))
            if a.name in ("value_floats", "value_ints"):
                return np.asarray(list(onnx.helper.get_attribute_value(a)))
        raise ValueError("unsupported Constant node")

    # Exporter hygiene: nodes whose inputs are all constants (Identity-wrapped initializers,
    # Shape/Gather/Unsqueeze/Concat shape chains, scalar arithmetic) are evaluated offline.
    FOLDABLE = frozenset({
        "Identity", "Cast", "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat", "Reshape",
        "Slice", "Flatten", "Transpose", "Expand", "Tile", "ConstantOfShape", "Range",
        "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Reciprocal", "Neg", "Abs", "Floor",
        "Ceil", "Exp", "Log", "Erf", "Sign", "Mod", "Sum", "Max", "Min", "Clip", "Relu",
        "Equal", "Greater", "Less", "Not", "Where", "ReduceSum", "ReduceMean", "ReduceProd",
    })

    def _fold_constants(self):
        """Replace constant subexpressions by initializers, then drop nodes whose outputs
        nothing reads any more. Shape of a tensor with static dimensions becomes a constant."""
        graph_outputs = {o.name for o in self.graph.output}
        kept = []
        for node in self.nodes:
            outs = [o for o in node.output if o]
            if not outs or len(outs) != len(node.output) or graph_outputs & set(outs):
                kept.append(node)
                continue
            if node.op_type == "Shape" and isinstance(self.dims.get(node.input[0]), list):
                dims = self.dims[node.input[0]]
                a = self.attrs(node)
                start, end = a.get("start", 0), a.get("end", len(dims))
                self.init[outs[0]] = np.asarray(dims[start:end], dtype=np.int64)
                continue
            names = [i for i in node.input if i]
            if (node.op_type in self.FOLDABLE and all(i in self.init for i in names)):
                try:
                    values = self._evaluate(node, {i: self.init[i] for i in names}, outs)
                except Exception:  # not foldable here: lowering reports it if unsupported
                    kept.append(node)
                    continue
                for name, value in zip(outs, values):
                    self.init[name] = np.asarray(value)
                continue
            kept.append(node)
        # Dead nodes (outputs unread after folding, e.g. Shape chains that fed a folded
        # Reshape) are removed; they cannot change any graph output.
        while True:
            used = set(graph_outputs)
            for n in kept:
                used.update(i for i in n.input if i)
            live = [n for n in kept if any(o in used for o in n.output if o)]
            if len(live) == len(kept):
                break
            kept = live
        self.nodes = kept

    def _refresh_dims(self):
        """Folding turns computed sizes (Resize from a Shape chain, Upsample scales) into
        constants that shape inference can read: infer again on the folded graph and add the
        dimensions that were unknown before."""
        inits = [numpy_helper.from_array(np.asarray(v), k) for k, v in self.init.items()]
        graph = onnx.helper.make_graph(self.nodes, "folded", list(self.graph.input),
                                       list(self.graph.output), inits)
        model = onnx.helper.make_model(graph, opset_imports=self.model.opset_import)
        model.ir_version = self.model.ir_version
        try:
            inferred = _static_dims(_infer_shapes(model))
        except Exception:  # keep the dimensions of the original graph
            return
        for name, dims in inferred.items():
            self.dims.setdefault(name, dims)

    def _evaluate(self, node, feeds, outs):
        """Run one node with onnx.reference on its constant inputs."""
        from onnx.reference import ReferenceEvaluator

        inputs = [
            onnx.helper.make_tensor_value_info(
                n, onnx.helper.np_dtype_to_tensor_dtype(feeds[n].dtype), list(feeds[n].shape))
            for n in feeds
        ]
        outputs = [onnx.helper.make_tensor_value_info(o, 0, None) for o in outs]
        graph = onnx.helper.make_graph([node], "fold", inputs, outputs)
        model = onnx.helper.make_model(graph, opset_imports=self.model.opset_import)
        model.ir_version = self.model.ir_version
        return ReferenceEvaluator(model).run(None, feeds)

    def fail(self, node, why):
        raise ValueError(f"{node.op_type} '{node.name or node.output[0]}': {why}")

    def canon(self, name):
        while name in self.alias:
            name = self.alias[name]
        return name

    def const(self, name):
        return self.init.get(name)

    def hwc(self, name):
        s = self.dims.get(name)
        if s is None:
            raise ValueError(f"no static float32 shape for {name}")
        if len(s) == 4 and s[0] == 1:
            return [s[2], s[3], s[1]]
        if len(s) == 3 and s[0] == 1:  # squeezed intermediate: NCHW with H = 1
            return [1, s[2], s[1]]
        if len(s) == 2 and s[0] == 1:
            return [1, 1, s[1]]
        raise ValueError(f"unsupported shape {s} for {name} (batch-one rank 2, 3 or 4 only)")

    def tid(self, name):
        name = self.canon(name)
        if name not in self.ids:
            self.ids[name] = len(self.ids)
            if name in self.dims:
                self.shapes[self.ids[name]] = self.hwc(name)
        return self.ids[name]

    def temp(self, like, shape):
        """A new internal tensor with an explicit HWC shape."""
        self.synthetic += 1
        name = f"{like}#t{self.synthetic}"
        self.ids[name] = len(self.ids)
        self.shapes[self.ids[name]] = list(shape)
        return self.ids[name]

    def add(self, op):
        self.ops.append(op)
        self.producer[op.out] = op
        return op

    def sole_user(self, name):
        return self.users.get(name, 0) == 1 and name not in self.outputs

    # ------------------------------------------------------------ pads / pools
    @staticmethod
    def attrs(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def dilation(self, node, a):
        """(dh, dw) of a convolution from its dilations attribute; (1, 1) when absent."""
        d = list(a.get("dilations", [1, 1]))
        if len(d) != 2 or min(d) < 1 or max(d) > 8:
            self.fail(node, "dilations must be between 1 and 8 on each axis")
        return d[0], d[1]

    def spatial(self, node, a, x, out, k, dilated=False):
        """(stride, pad_top, pad_left) of a windowed operation; checks the output size.
        k is the kernel size; dilated=True is for convolutions, which may dilate the window."""
        ih, iw, _ = self.hwc(x)
        oh, ow, _ = self.hwc(out)
        s = list(a.get("strides", [1, 1]))
        if dilated:
            dh, dw = self.dilation(node, a)
        else:
            if list(a.get("dilations", [1, 1])) != [1, 1]:
                self.fail(node, "dilation is unsupported")
            dh, dw = 1, 1
        # Effective extents (k - 1) * d + 1 are what the output size and padding are based on.
        ke = [(k[0] - 1) * dh + 1, (k[1] - 1) * dw + 1]
        mode = a.get("auto_pad", b"NOTSET")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        if mode in ("SAME_UPPER", "SAME_LOWER"):
            tot = [max((oh - 1) * s[0] + ke[0] - ih, 0), max((ow - 1) * s[1] + ke[1] - iw, 0)]
            top = [t // 2 if mode == "SAME_UPPER" else t - t // 2 for t in tot]
            pads = top + [tot[0] - top[0], tot[1] - top[1]]
        elif mode == "VALID":
            pads = [0, 0, 0, 0]
        else:
            pads = list(a.get("pads", [0, 0, 0, 0]))
        if (ih + pads[0] + pads[2] - ke[0]) // s[0] + 1 != oh or (
            iw + pads[1] + pads[3] - ke[1]
        ) // s[1] + 1 != ow:
            self.fail(node, "output size disagrees with the window geometry")
        if pads[0] >= ke[0] or pads[1] >= ke[1]:
            self.fail(node, "leading padding must be smaller than the kernel")
        if max(k) > 15 or max(s) > 8:
            self.fail(node, "kernel above 15 or stride above 8")
        return s, pads[0], pads[1]

    # ------------------------------------------------------------ constants
    def channel_vector(self, node, value, out_name):
        """Per-channel float32 vector from a constant broadcast against `out_name`, else None."""
        shape = self.dims[out_name]
        c = shape[1]
        v = np.asarray(value, dtype=np.float32)
        padded = (1,) * (len(shape) - v.ndim) + tuple(v.shape)
        if len(padded) != len(shape) or padded[0] != 1:
            return None
        if len(shape) == 4 and (padded[2] != 1 or padded[3] != 1):
            return None
        if padded[1] not in (1, c):
            return None
        return np.broadcast_to(v.reshape(-1), (c,)).astype(np.float32).copy()

    # ------------------------------------------------------------ main driver
    def run(self):
        for name in self.inputs:
            self.tid(name)
        silu = self._find_silu()
        for node in self.nodes:
            if id(node) in silu["skip"]:
                continue
            handler = getattr(self, "op_" + node.op_type, None)
            if handler is None:
                self.fail(node, "unsupported operator")
            if id(node) in silu["mul"]:
                self._emit_act(node, silu["mul"][id(node)], hx.FN_SILU)
            else:
                handler(node)
            self.source_ops.append({"onnx_op": node.op_type, "output": node.output[0]})
        self._finish_layers()
        return self._write

    def _find_silu(self):
        by_output = {n.output[0]: n for n in self.nodes}
        skip, mul = set(), {}
        for n in self.nodes:
            if n.op_type != "Mul" or len(n.input) != 2:
                continue
            for s, x in (n.input, n.input[::-1]):
                prod = by_output.get(s)
                if (
                    prod is not None
                    and prod.op_type == "Sigmoid"
                    and prod.input[0] == x
                    and self.users.get(s, 0) == 1
                    and s not in self.outputs
                    and x not in self.init
                ):
                    skip.add(id(prod))
                    mul[id(n)] = x
                    break
        return {"skip": skip, "mul": mul}

    # ------------------------------------------------------------ layers
    def _new_layer(self, node, kind, x_name, out_name, weights, bias, p, dilation=(1, 1)):
        op = Op(kind, self.tid(x_name), NONE, self.tid(out_name), p, node)
        op.dilation = dilation
        op.weights, op.bias = weights.astype(np.float32), bias.astype(np.float32)
        op.foldable = True
        return self.add(op)

    def op_Conv(self, node):
        a = self.attrs(node)
        x, w = node.input[0], self.const(node.input[1])
        if w is None or w.ndim != 4:
            self.fail(node, "weights must be a constant 4-D tensor")
        bias = (
            self.const(node.input[2])
            if len(node.input) > 2 and node.input[2]
            else np.zeros(w.shape[0], np.float32)
        )
        if bias is None:
            self.fail(node, "bias must be constant")
        co, cig, kh, kw = w.shape
        _, _, ic = self.hwc(x)
        group = a.get("group", 1)
        dil = self.dilation(node, a)
        s, pt, pl = self.spatial(node, a, x, node.output[0], [kh, kw], dilated=True)
        if group == 1 and cig == ic:
            kind = hx.CONV
        elif group == co == ic and cig == 1:
            kind = hx.DEPTHWISE
        elif group > 1 and cig * group == ic and co % group == 0:
            return self._grouped_conv(node, x, w, np.asarray(bias).reshape(-1), group,
                                      [kh, kw, s[0], s[1], pt, pl], dil)
        else:
            self.fail(node, "groups must divide the channels and the output channels")
        self._new_layer(node, kind, x, node.output[0], w, np.asarray(bias).reshape(-1),
                        [kh, kw, s[0], s[1], pt, pl], dil)

    def _grouped_conv(self, node, x, w, bias, group, p, dilation=(1, 1)):
        """Grouped convolution as one dense convolution per channel group: a Slice of the
        group's input channels, the group's weights, then a channel Concat chain. The per-group
        layers are plain layers, so BatchNorm and ReLU do not fold into them (the Concat is
        their producer)."""
        co, cig = w.shape[0], w.shape[1]
        ih, iw, _ = self.hwc(x)
        oh, ow, _ = self.hwc(node.output[0])
        cog = co // group
        xid, acc_id, acc_c = self.tid(x), None, 0
        for g in range(group):
            piece = self.temp(node.output[0], [ih, iw, cig])
            self.add(Op(hx.SLICE, xid, NONE, piece, [0, 0, g * cig], node))
            part = self.temp(node.output[0], [oh, ow, cog])
            layer = Op(hx.CONV, piece, NONE, part, p, node)
            layer.dilation = dilation
            layer.weights = w[g * cog:(g + 1) * cog].astype(np.float32)
            layer.bias = bias[g * cog:(g + 1) * cog].astype(np.float32)
            self.add(layer)
            if acc_id is None:
                acc_id, acc_c = part, cog
                continue
            acc_c += cog
            merged = (self.tid(node.output[0]) if g == group - 1
                      else self.temp(node.output[0], [oh, ow, acc_c]))
            self.add(Op(hx.CONCAT, acc_id, part, merged, [2], node))
            acc_id = merged

    def op_ConvTranspose(self, node):
        """ConvTranspose as zero insertion (stride s, leading pad k-1-p) followed by a dense
        stride-1 convolution with the spatially flipped kernel and swapped channel axes. The
        convolution is an ordinary layer, so BatchNorm folding and ReLU fusion apply to it."""
        a = self.attrs(node)
        x, w = node.input[0], self.const(node.input[1])
        if w is None or w.ndim != 4:
            self.fail(node, "weights must be a constant 4-D tensor")
        if a.get("group", 1) != 1:
            self.fail(node, "groups other than 1 are unsupported")
        if list(a.get("dilations", [1, 1])) != [1, 1]:
            self.fail(node, "dilation is unsupported")
        if "output_shape" in a:
            self.fail(node, "output_shape is unsupported")
        mode = a.get("auto_pad", b"NOTSET")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        if mode != "NOTSET":
            self.fail(node, "auto_pad is unsupported")
        if len(node.input) > 2 and node.input[2]:
            bias = self.const(node.input[2])
            if bias is None:
                self.fail(node, "bias must be constant")
        else:
            bias = np.zeros(w.shape[1], np.float32)
        ci, co, kh, kw = w.shape
        ih, iw, ic = self.hwc(x)
        if ic != ci:
            self.fail(node, "weights disagree with the input channels")
        if max(kh, kw) > 15:
            self.fail(node, "kernel above 15")
        s = list(a.get("strides", [1, 1]))
        if min(s) < 1 or max(s) > 8:
            self.fail(node, "stride must be between 1 and 8")
        pt, pl, pb, pr = (list(a.get("pads", [0, 0, 0, 0])) + [0, 0, 0, 0])[:4]
        op_h, op_w = (list(a.get("output_padding", [0, 0])) + [0, 0])[:2]
        # Leading and trailing zeros of the zero-inserted input (pads larger than k-1 are
        # rejected: the padded input would need a negative leading border).
        zt, zl, zb, zr = kh - 1 - pt, kw - 1 - pl, kh - 1 - pb, kw - 1 - pr
        if min(zt, zl, zb, zr) < 0:
            self.fail(node, "pads must not exceed the kernel size minus one")
        oh, ow, _ = self.hwc(node.output[0])
        if (oh, ow) != ((ih - 1) * s[0] + kh - pt - pb + op_h,
                        (iw - 1) * s[1] + kw - pl - pr + op_w):
            self.fail(node, "output size disagrees with the transposed geometry")
        zh = zt + (ih - 1) * s[0] + 1 + zb + op_h
        zw = zl + (iw - 1) * s[1] + 1 + zr + op_w
        zid = self.temp(node.output[0], [zh, zw, ic])
        self.add(Op(hx.ZEROINSERT, self.tid(x), NONE, zid, [s[0], s[1], zt, zl], node))
        # Equivalent dense kernel: [ci, co, kh, kw] -> [co, ci, kh, kw], flipped in both axes.
        dense = np.ascontiguousarray(
            np.asarray(w, np.float32).transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
        layer = Op(hx.CONV, zid, NONE, self.tid(node.output[0]), [kh, kw, 1, 1, 0, 0], node)
        layer.weights = dense
        layer.bias = np.asarray(bias, np.float32).reshape(-1)
        layer.foldable = True
        self.add(layer)

    def _linear(self, node, x, w, bias, out):
        """Gemm/MatMul as a 1x1 convolution on a [1, 1, K] tensor; wide outputs are split."""
        if self.hwc(x)[:2] != [1, 1]:
            self.fail(node, "input must be a flat [1, K] tensor")
        n, k = w.shape
        tiles = (k + 31) // 32
        chunk = max(
            (c for c in range(32, min((n + 31) & ~31, hx.MAX_CHANNELS) + 1, 32)
             if hx.hmx_fits(tiles, c)),
            default=0,
        )
        if not chunk:
            self.fail(node, "input is too deep for the HMX budget")
        w4 = w.reshape(n, k, 1, 1)
        if chunk >= n:
            self._new_layer(node, hx.CONV, x, out, w4, bias, [1, 1, 1, 1, 0, 0])
            return
        if out not in self.outputs:  # mid-graph: split into chunks plus a concat at the end
            self._new_layer(node, hx.CONV, x, out, w4, bias, [1, 1, 1, 1, 0, 0])
            return
        parts = []
        for i, lo in enumerate(range(0, n, chunk)):
            hi = min(lo + chunk, n)
            name = f"{out}#{i}"
            self.dims[name] = [1, hi - lo]
            op = self._new_layer(node, hx.CONV, x, name, w4[lo:hi], bias[lo:hi],
                                 [1, 1, 1, 1, 0, 0])
            op.foldable = False
            parts.append({"tensor": name, "channels": [lo, hi]})
        self.output_parts = {out: parts}

    def op_Gemm(self, node):
        a = self.attrs(node)
        if a.get("transA", 0) != 0:
            self.fail(node, "transA is unsupported")
        b = self.const(node.input[1])
        if b is None or b.ndim != 2:
            self.fail(node, "B must be a constant matrix")
        w = (b if a.get("transB", 0) else b.T) * np.float32(a.get("alpha", 1.0))
        n = w.shape[0]
        bias = np.zeros(n, np.float32)
        if len(node.input) > 2 and node.input[2]:
            c = self.const(node.input[2])
            if c is None:
                self.fail(node, "C must be constant")
            bias = np.broadcast_to(np.asarray(c).reshape(-1), (n,)) * np.float32(
                a.get("beta", 1.0)
            )
        self._linear(node, node.input[0], w.astype(np.float32), bias, node.output[0])

    def op_MatMul(self, node):
        b = self.const(node.input[1])
        if b is None or b.ndim != 2:
            self.fail(node, "right operand must be a constant matrix")
        self._linear(node, node.input[0], b.T.astype(np.float32),
                     np.zeros(b.shape[1], np.float32), node.output[0])

    # ------------------------------------------------------------ affine / BN
    def _affine(self, node, x_name, out_name, scale, shift):
        """y = x * scale + shift per channel: fold into the producing layer, else emit ops."""
        x = self.canon(x_name)
        prod = self.producer.get(self.ids.get(x, -1))
        if (
            prod is not None
            and prod.foldable
            and not prod.fused
            and prod.weights is not None
            and self.sole_user(x_name)
            and prod.weights.shape[0] == scale.size
        ):
            prod.weights = prod.weights * scale.reshape(-1, 1, 1, 1)
            prod.bias = prod.bias * scale + shift
            self.alias[out_name] = x
            return
        xid, shape = self.tid(x_name), self.hwc(x_name)
        cur = xid
        if not np.all(scale == 1):
            nxt = self.tid(out_name) if np.all(shift == 0) else self.temp(out_name, shape)
            self.add(Op(hx.BINARY, cur, NONE, nxt,
                        [hx.BIN_MUL, 1, self.pool.emit(_pad32(scale)), 0], node))
            cur = nxt
        if not np.all(shift == 0):
            nxt = self.tid(out_name)
            self.add(Op(hx.BINARY, cur, NONE, nxt,
                        [hx.BIN_ADD, 1, self.pool.emit(_pad32(shift)), 0], node))
            cur = nxt
        if cur == xid:  # identity
            self.alias[out_name] = x
        elif cur != self.tid(out_name):  # pragma: no cover
            raise ValueError("internal error in affine lowering")

    def op_BatchNormalization(self, node):
        a = self.attrs(node)
        if a.get("training_mode", 0):
            self.fail(node, "training mode is unsupported")
        if len(node.output) > 1 and any(node.output[1:]):
            self.fail(node, "extra outputs are unsupported")
        scale, bias, mean, var = (self.const(n) for n in node.input[1:5])
        if any(v is None for v in (scale, bias, mean, var)):
            self.fail(node, "scale, bias, mean and variance must be constants")
        inv = (scale / np.sqrt(var + a.get("epsilon", 1e-5))).astype(np.float64)
        self._affine(node, node.input[0], node.output[0], inv.astype(np.float32),
                     (bias - mean * inv).astype(np.float32))

    # ------------------------------------------------------------ elementwise binary
    def _binary(self, node, opcode):
        a_name, b_name = node.input
        out = node.output[0]
        ca, cb = self.const(a_name), self.const(b_name)
        if ca is not None and cb is not None:
            self.fail(node, "both operands are constants")
        if ca is not None or cb is not None:
            const, tensor, rev = (ca, b_name, 1) if ca is not None else (cb, a_name, 0)
            if self.dims[tensor] != self.dims[out]:
                self.fail(node, "a constant operand would broadcast the tensor")
            vec = self.channel_vector(node, const, out)
            if vec is None:
                self.fail(node, "a constant operand must be a per-channel vector")
            if opcode == hx.BIN_MUL and not rev:
                return self._affine(node, tensor, out, vec, np.zeros_like(vec))
            if opcode == hx.BIN_ADD:
                return self._affine(node, tensor, out, np.ones_like(vec), vec)
            if opcode == hx.BIN_SUB and not rev:
                return self._affine(node, tensor, out, np.ones_like(vec), -vec)
            if opcode == hx.BIN_MUL:  # c * x
                return self._affine(node, tensor, out, vec, np.zeros_like(vec))
            # c - x: negate, then add c
            neg = self.temp(out, self.hwc(out))
            self.add(Op(hx.BINARY, self.tid(tensor), NONE, neg,
                        [hx.BIN_MUL, 1, self.pool.emit(_pad32(-np.ones_like(vec))), 0], node))
            self.add(Op(hx.BINARY, neg, NONE, self.tid(out),
                        [hx.BIN_ADD, 1, self.pool.emit(_pad32(vec)), 0], node))
            return None
        sa, sb, so = self.dims[a_name], self.dims[b_name], self.dims[out]
        if len(sa) != len(sb):
            self.fail(node, "operands of different rank")

        def fits(small, big):
            return len(small) == len(big) and small[0] == 1 and all(
                s in (1, o) for s, o in zip(small, big)
            )

        if sa == so and fits(sb, so):
            x, y, rev = a_name, b_name, 0
        elif sb == so and fits(sa, so):
            x, y, rev = b_name, a_name, int(opcode == hx.BIN_SUB)
        else:
            self.fail(node, "operands broadcast against each other")
        if opcode == hx.BIN_ADD and self.dims[x] == self.dims[y]:
            self.add(Op(hx.ADD, self.tid(x), self.tid(y), self.tid(out), [0], node))
            self.producer[self.tid(out)].foldable = False
        else:
            self.add(Op(hx.BINARY, self.tid(x), self.tid(y), self.tid(out),
                        [opcode, 0, 0, rev], node))

    def op_Add(self, node):
        self._binary(node, hx.BIN_ADD)

    def op_Sub(self, node):
        self._binary(node, hx.BIN_SUB)

    def op_Mul(self, node):
        self._binary(node, hx.BIN_MUL)

    def op_Div(self, node):
        d = self.const(node.input[1])
        if d is None or self.const(node.input[0]) is not None:
            self.fail(node, "only division of a tensor by a constant is supported")
        vec = self.channel_vector(node, 1.0 / np.asarray(d, dtype=np.float64), node.output[0])
        if vec is None or self.dims[node.input[0]] != self.dims[node.output[0]]:
            self.fail(node, "the divisor must be a per-channel vector")
        self._affine(node, node.input[0], node.output[0], vec, np.zeros_like(vec))

    # ------------------------------------------------------------ activations
    @staticmethod
    def _act_code(node, a, consts):
        """ReLU-family fused code (1 ReLU, 3 ReLU6) for Relu and Clip, else None."""
        if node.op_type == "Relu":
            return hx.ACT_RELU
        if node.op_type != "Clip":
            return None
        lo = consts[0] if consts[0] is not None else a.get("min")
        hi = consts[1] if consts[1] is not None else a.get("max")
        lo = None if lo is None else float(np.asarray(lo).reshape(()))
        hi = None if hi is None else float(np.asarray(hi).reshape(()))
        if lo == 0 and (hi is None or hi == np.inf):
            return hx.ACT_RELU
        if lo == 0 and hi == 6:
            return hx.ACT_RELU6
        return None

    def _fuse(self, node, code):
        """Fold a ReLU-family activation into a single-user producer, if possible."""
        x = node.input[0]
        prod = self.producer.get(self.ids.get(self.canon(x), -1))
        if prod is None or prod.fused or not self.sole_user(x):
            return False
        if prod.kind in (hx.CONV, hx.DEPTHWISE) and prod.weights is not None:
            prod.p[8] = code
        elif prod.kind == hx.ADD and prod.foldable is False and prod.p[0] == 0:
            prod.p[0] = code
        else:
            return False
        prod.fused = True
        self.alias[node.output[0]] = self.canon(x)
        return True

    def _emit_act(self, node, x_name, fn, alpha=0.0, beta=0.0):
        self.add(Op(hx.ACT, self.tid(x_name), NONE, self.tid(node.output[0]),
                    [fn, hx.f32_bits(alpha) if alpha else 0, hx.f32_bits(beta) if beta else 0],
                    node))

    def _relu_like(self, node):
        a = self.attrs(node)
        consts = [self.const(n) if n else None for n in node.input[1:3]] + [None, None]
        code = self._act_code(node, a, consts)
        if code is None:
            self.fail(node, "only ReLU, Clip(0, 6) and Clip(0, inf) are supported")
        if not self._fuse(node, code):
            self.add(Op(hx.ACT, self.tid(node.input[0]), NONE, self.tid(node.output[0]),
                        [hx.FN_RELU if code == hx.ACT_RELU else hx.FN_RELU6], node))

    op_Relu = _relu_like
    op_Clip = _relu_like

    def op_LeakyRelu(self, node):
        self._emit_act(node, node.input[0], hx.FN_LEAKY, self.attrs(node).get("alpha", 0.01))

    def op_Tanh(self, node):
        self._emit_act(node, node.input[0], hx.FN_TANH)

    def op_HardSwish(self, node):
        self._emit_act(node, node.input[0], hx.FN_HARDSWISH)

    def op_HardSigmoid(self, node):
        a = self.attrs(node)
        self._emit_act(node, node.input[0], hx.FN_HARDSIGMOID, a.get("alpha", 0.2),
                       a.get("beta", 0.5))

    def op_Gelu(self, node):
        approx = self.attrs(node).get("approximate", "none")
        approx = approx.decode() if isinstance(approx, bytes) else approx
        if approx != "none":
            self.fail(node, "only exact (erf) GELU is supported")
        self._emit_act(node, node.input[0], hx.FN_GELU)

    def op_Exp(self, node):
        self._emit_act(node, node.input[0], hx.FN_EXP)

    def op_Sqrt(self, node):
        self._emit_act(node, node.input[0], hx.FN_SQRT)

    def op_Abs(self, node):
        self._emit_act(node, node.input[0], hx.FN_ABS)

    def op_Neg(self, node):
        self._emit_act(node, node.input[0], hx.FN_NEG)

    def op_Erf(self, node):
        self._emit_act(node, node.input[0], hx.FN_ERF)

    def op_Sigmoid(self, node):
        self.add(Op(hx.LOGISTIC, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                    node))

    def op_Softmax(self, node):
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) not in (2, 4):
            self.fail(node, "only batch-one rank 2 or 4 float32 inputs are supported")
        rank = len(dims)
        # Opset 13+ defaults to axis -1, which is W for rank 4: that case is rejected below.
        axis = int(self.attrs(node).get("axis", -1))
        axis = axis + rank if axis < 0 else axis
        if axis != 1:
            self.fail(node, f"only softmax over the channel axis (axis 1) is supported; "
                            f"got axis {axis} for rank {rank} (set axis=1 explicitly)")
        if self.const(node.input[0]) is not None:
            self.fail(node, "constant inputs are unsupported")
        self.add(Op(hx.SOFTMAX, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                    node))

    # ------------------------------------------------------------ normalisation
    def _norm(self, node, x, out, mode, groups, eps, gamma, beta):
        """One HX_NORM: mode and groups as in graph.h, epsilon as float32 bits, and gamma and beta
        as per-channel constant vectors (zero-padded to 32 lanes, 128-byte aligned in the pool)."""
        if self.const(x) is not None:
            self.fail(node, "constant inputs are unsupported")
        if self.hwc(out) != self.hwc(x):
            self.fail(node, "output shape must equal the input shape")
        self.add(Op(hx.NORM, self.tid(x), NONE, self.tid(out),
                    [mode, groups, hx.f32_bits(eps), self.pool.emit(_pad32(gamma)),
                     self.pool.emit(_pad32(beta))], node))

    def _norm_vector(self, node, value, c, what):
        """A float32 vector of exactly c elements: one normalisation scale or bias per channel."""
        v = np.asarray(value, dtype=np.float32)
        if v.size != c:
            self.fail(node, f"{what} must have {c} elements, one per channel; got shape {list(v.shape)}")
        return v.reshape(-1).copy()

    def _spatial_vector(self, node, value, c, h, w, what):
        """Per-channel vector from a [C, H, W] (or broadcastable [C, 1, 1]) parameter, which must
        be constant over H and W. This is LayerNorm's scale or bias over (C, H, W)."""
        v = np.asarray(value, dtype=np.float32)
        if v.ndim == 4 and v.shape[0] == 1:
            v = v[0]
        try:
            full = np.broadcast_to(v, (c, h, w))
        except ValueError:
            self.fail(node, f"{what} must have shape [C, H, W] or [C, 1, 1]; got {list(v.shape)}")
        vec = full[:, 0, 0].copy()
        if not np.array_equal(full, np.broadcast_to(vec[:, None, None], (c, h, w))):
            self.fail(node, f"{what} must be constant over the spatial axes")
        return vec

    def _optional_constant(self, node, index, what):
        """A constant input that may be absent (None) for LayerNormalization's Scale and Bias."""
        if len(node.input) <= index or not node.input[index]:
            return None
        value = self.const(node.input[index])
        if value is None:
            self.fail(node, f"{what} must be constant")
        return value

    def op_InstanceNormalization(self, node):
        """Instance normalisation: per channel over the spatial axes of a [1, C, H, W] input,
        which is HX_NORM with one group per channel."""
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) != 4:
            self.fail(node, "only batch-one rank-4 float32 inputs are supported")
        c = dims[1]
        scale = self._optional_constant(node, 1, "scale")
        bias = self._optional_constant(node, 2, "bias")
        if scale is None or bias is None:
            self.fail(node, "scale and bias are required and must be constant")
        eps = float(self.attrs(node).get("epsilon", 1e-5))
        self._norm(node, node.input[0], node.output[0], hx.NORM_GROUP, c, eps,
                   self._norm_vector(node, scale, c, "scale"),
                   self._norm_vector(node, bias, c, "bias"))

    def op_LayerNormalization(self, node):
        """LayerNorm over the channel axis of a [1, C] tensor (HX_NORM, channel mode), or over
        (C, H, W) of a [1, C, H, W] tensor with axis 1 (HX_NORM, one group). Scale and bias must
        be constant per channel (spatially constant for rank 4). Mean and InvStdDev are unused."""
        if any(self.users.get(name, 0) for name in node.output[1:]):
            self.fail(node, "the Mean and InvStdDev outputs are unsupported")
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) not in (2, 4):
            self.fail(node, "only batch-one rank-2 or rank-4 float32 inputs are supported")
        axis = int(self.attrs(node).get("axis", -1)) % len(dims)
        eps = float(self.attrs(node).get("epsilon", 1e-5))
        scale = self._optional_constant(node, 1, "scale")
        bias = self._optional_constant(node, 2, "bias")
        if len(dims) == 2:
            if dims[0] != 1 or axis != 1:
                self.fail(node, "rank-2 LayerNormalization must normalise a [1, C] tensor over "
                                "axis 1 or -1")
            c = dims[1]
            gamma = (np.ones(c, np.float32) if scale is None
                     else self._norm_vector(node, scale, c, "scale"))
            beta = (np.zeros(c, np.float32) if bias is None
                    else self._norm_vector(node, bias, c, "bias"))
            self._norm(node, node.input[0], node.output[0], hx.NORM_CHANNEL, 0, eps, gamma, beta)
            return
        if axis != 1:
            self.fail(node, "rank-4 LayerNormalization is supported over axis 1 (C, H, W) only")
        _, c, h, w = dims
        gamma = (np.ones(c, np.float32) if scale is None
                 else self._spatial_vector(node, scale, c, h, w, "scale"))
        beta = (np.zeros(c, np.float32) if bias is None
                else self._spatial_vector(node, bias, c, h, w, "bias"))
        self._norm(node, node.input[0], node.output[0], hx.NORM_GROUP, 1, eps, gamma, beta)

    def op_GroupNormalization(self, node):
        """GroupNormalization over a [1, C, H, W] input. Scale and bias have one value per group
        (opset 18) or per channel (opset 21); group values are repeated over their channels."""
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) != 4:
            self.fail(node, "only batch-one rank-4 float32 inputs are supported")
        c = dims[1]
        a = self.attrs(node)
        groups = int(a.get("num_groups", 0))
        if groups < 1 or c % groups:
            self.fail(node, f"num_groups {groups} must divide the {c} channels")
        scale = self._optional_constant(node, 1, "scale")
        bias = self._optional_constant(node, 2, "bias")
        if scale is None or bias is None:
            self.fail(node, "scale and bias are required and must be constant")
        eps = float(a.get("epsilon", 1e-5))

        def per_channel(value, what):
            v = np.asarray(value, dtype=np.float32).reshape(-1)
            if v.size == c:
                return v.copy()
            if v.size == groups:
                return np.repeat(v, c // groups)
            self.fail(node, f"{what} must have {groups} (per group) or {c} (per channel) elements")

        self._norm(node, node.input[0], node.output[0], hx.NORM_GROUP, groups, eps,
                   per_channel(scale, "scale"), per_channel(bias, "bias"))

    def op_PRelu(self, node):
        slope = self.const(node.input[1])
        if slope is None:
            self.fail(node, "slope must be constant")
        vec = self.channel_vector(node, slope, node.output[0])
        if vec is None:
            self.fail(node, "slope must be per channel or a scalar")
        self.add(Op(hx.PRELU, self.tid(node.input[0]), NONE, self.tid(node.output[0]),
                    [self.pool.emit(_pad32(vec))], node))

    # ------------------------------------------------------------ pooling
    def _pool(self, node, kind):
        a = self.attrs(node)
        if a.get("ceil_mode", 0):
            self.fail(node, "ceil mode is unsupported")
        if len(node.output) > 1 and any(node.output[1:]):
            self.fail(node, "pool indices are unsupported")
        k = list(a["kernel_shape"])
        s, pt, pl = self.spatial(node, a, node.input[0], node.output[0], k)
        p = [k[0], k[1], s[0], s[1], pt, pl]
        if kind == hx.AVGPOOL:
            p.append(int(a.get("count_include_pad", 0)))
        self.add(Op(kind, self.tid(node.input[0]), NONE, self.tid(node.output[0]), p, node))

    def op_MaxPool(self, node):
        self._pool(node, hx.MAXPOOL)

    def op_AveragePool(self, node):
        self._pool(node, hx.AVGPOOL)

    def _mean(self, node):
        x, out = node.input[0], node.output[0]
        if len(self.dims[x]) != 4 or self.hwc(out)[:2] != [1, 1]:
            self.fail(node, "only spatial reduction of a rank-4 tensor is supported")
        self.add(Op(hx.MEAN, self.tid(x), NONE, self.tid(out), [], node))

    def op_GlobalAveragePool(self, node):
        self._mean(node)

    def _reduce_axes(self, node):
        """Axes from the attribute (opset < 13 or 18) or a constant second input."""
        a = self.attrs(node)
        if "axes" in a:
            return [int(v) for v in a["axes"]]
        if len(node.input) > 1 and node.input[1]:
            value = self.const(node.input[1])
            if value is None:
                self.fail(node, "axes must be constant")
            return [int(v) for v in np.asarray(value).reshape(-1)]
        return []

    def _reduce(self, node, operation, axes):
        """HX_REDUCE over rank-4 NCHW (or rank-2) axes. Batch is rejected; empty axes (reduce
        everything) are rejected. The result must match the keepdims layout of the input with
        each reduced axis set to one, so keepdims=0 is accepted only when that layout is
        representable as the output's HWC shape."""
        x, out = node.input[0], node.output[0]
        dims = self.dims.get(x)
        if dims is None or len(dims) not in (2, 4):
            self.fail(node, "only static rank-2 or rank-4 float32 inputs are supported")
        if self.const(x) is not None:
            self.fail(node, "constant inputs are unsupported")
        if not axes:
            self.fail(node, "empty axes (reduce everything) are unsupported")
        rank = len(dims)
        bits = {1: 4, 2: 1, 3: 2} if rank == 4 else {1: 4}  # ONNX C, H, W -> HWC c, h, w
        mask = 0
        for v in axes:
            v = int(v) + rank if int(v) < 0 else int(v)
            if v == 0:
                self.fail(node, "reducing the batch axis is unsupported")
            if v not in bits:
                self.fail(node, f"axis {v} is not a reducible axis of rank {rank}")
            if mask & bits[v]:
                self.fail(node, "repeated axes")
            mask |= bits[v]
        hin = self.hwc(x)
        expect = [1 if mask & 1 else hin[0], 1 if mask & 2 else hin[1], 1 if mask & 4 else hin[2]]
        try:
            got = self.hwc(out)
        except ValueError:
            self.fail(node, f"output shape {self.dims.get(out)} has no HWC form; "
                            "use keepdims=1 or reduce fewer axes")
        if got != expect:
            self.fail(node, f"output HWC {got} differs from the keepdims layout {expect}; "
                            "use keepdims=1")
        if operation == hx.RED_MEAN and mask == 3 and len(dims) == 4:
            self._mean(node)  # spatial mean keeps the dedicated, channel-preserving operation
            return
        self.add(Op(hx.REDUCE, self.tid(x), NONE, self.tid(out), [operation, mask], node))

    def op_ReduceMean(self, node):
        self._reduce(node, hx.RED_MEAN, self._reduce_axes(node))

    def op_ReduceSum(self, node):
        self._reduce(node, hx.RED_SUM, self._reduce_axes(node))

    def op_ReduceMax(self, node):
        self._reduce(node, hx.RED_MAX, self._reduce_axes(node))

    def op_ReduceMin(self, node):
        self._reduce(node, hx.RED_MIN, self._reduce_axes(node))

    def op_GlobalMaxPool(self, node):
        self._reduce(node, hx.RED_MAX, [2, 3])

    # ------------------------------------------------------------ shape operators
    def _alias_shape(self, node):
        """Identity, Dropout, Flatten, Reshape, Squeeze and Unsqueeze. An unchanged HWC shape is
        an alias; any other change is a row-major (NCHW) reshape, lowered as an NCHW flatten to
        1x1xN (HX_RESHAPE flag 1) and an NCHW unflatten (flag 2), each skipped when already 1x1."""
        x, out = node.input[0], node.output[0]
        src, dst = self.hwc(x), self.hwc(out)
        if src == dst:
            self.alias[out] = self.canon(x)
            return
        n = src[0] * src[1] * src[2]
        if n != dst[0] * dst[1] * dst[2]:
            self.fail(node, "the input and output element counts differ")
        source = self.tid(x)
        if src[:2] != [1, 1]:
            if n > hx.MAX_CHANNELS:
                self.fail(node, "the flattened size exceeds the channel limit")
            if dst[:2] == [1, 1]:
                self.add(Op(hx.RESHAPE, source, NONE, self.tid(out), [1], node))
                return
            flat = self.temp(out, [1, 1, n])
            self.add(Op(hx.RESHAPE, source, NONE, flat, [1], node))
            source = flat
        self.add(Op(hx.RESHAPE, source, NONE, self.tid(out), [2], node))

    op_Identity = op_Dropout = op_Flatten = _alias_shape
    op_Reshape = op_Squeeze = op_Unsqueeze = _alias_shape

    def op_Cast(self, node):
        """Exporters cast float32 to float32 (for example `x.float()`): an alias. Other target
        types are not float32 tensors and cannot be lowered."""
        if self.dims.get(node.output[0]) is None:
            self.fail(node, "only casts to float32 are lowered")
        self._alias_shape(node)

    def op_Transpose(self, node):
        """Rank-4 batch-one transposes whose perm keeps the batch axis first. ONNX axes are
        (1=C, 2=H, 3=W); HWC axes are (H=0, W=1, C=2). Output h, w and c are output ONNX axes
        2, 3 and 1, so they read source axes perm[2], perm[3] and perm[1]."""
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) != 4:
            self.fail(node, "only rank-4 batch-one tensors with perm [0, a, b, c] are supported")
        perm = [int(v) for v in self.attrs(node).get("perm", [3, 2, 1, 0])]
        if sorted(perm) != [0, 1, 2, 3] or perm[0] != 0:
            self.fail(node, "the batch axis must stay first: perm must be [0, a, b, c]")
        if self.const(node.input[0]) is not None:
            self.fail(node, "constant inputs are unsupported")
        hwc_of = {1: 2, 2: 0, 3: 1}
        p = [hwc_of[perm[2]], hwc_of[perm[3]], hwc_of[perm[1]]]
        if p == [0, 1, 2]:
            self.alias[node.output[0]] = self.canon(node.input[0])
            return
        self.add(Op(hx.TRANSPOSE, self.tid(node.input[0]), NONE, self.tid(node.output[0]), p, node))

    def op_Concat(self, node):
        rank = len(self.dims[node.output[0]])
        axis = int(self.attrs(node)["axis"]) % rank
        if rank == 4 and axis == 0 or rank not in (2, 4):
            self.fail(node, "unsupported axis or rank")
        hwc_axis = 2 if rank == 2 else {1: 2, 2: 0, 3: 1}[axis]
        names = [n for n in node.input]
        if any(self.const(n) is not None for n in names):
            self.fail(node, "constant inputs are unsupported")
        acc_id, acc_shape = self.tid(names[0]), list(self.hwc(names[0]))
        for i, n in enumerate(names[1:], 1):
            shape = list(self.hwc(n))
            merged = list(acc_shape)
            merged[hwc_axis] += shape[hwc_axis]
            last = i == len(names) - 1
            out = self.tid(node.output[0]) if last else self.temp(node.output[0], merged)
            self.add(Op(hx.CONCAT, acc_id, self.tid(n), out, [hwc_axis], node))
            acc_id, acc_shape = out, merged

    def op_Slice(self, node):
        """Slice with constant starts/ends (opset 10+) and unit steps. Channel and spatial
        axes become SLICE offsets; any other axis must be taken whole."""
        if len(node.input) < 3 or not node.input[1] or not node.input[2]:
            self.fail(node, "starts and ends must be inputs (opset 10 or later)")
        data = node.input[0]
        dims = self.dims.get(data)
        if dims is None or len(dims) not in (2, 4):
            self.fail(node, "only static rank-2 or rank-4 tensors are supported")
        rank = len(dims)

        def const_input(i, default):
            if len(node.input) <= i or not node.input[i]:
                return default
            value = self.const(node.input[i])
            if value is None:
                self.fail(node, "starts, ends, axes and steps must be constant")
            return [int(v) for v in np.asarray(value).reshape(-1)]

        starts, ends = const_input(1, None), const_input(2, None)
        if len(starts) != len(ends):
            self.fail(node, "starts and ends must have the same length")
        axes = const_input(3, list(range(len(starts))))
        steps = const_input(4, [1] * len(starts))
        if len(axes) != len(starts) or len(steps) != len(starts) or any(s != 1 for s in steps):
            self.fail(node, "only unit steps are supported")
        hwc_axis = {1: 2, 2: 0, 3: 1} if rank == 4 else {1: 2}  # ONNX axis -> HWC index
        sizes = list(self.hwc(data))
        offsets, out_shape, seen = [0, 0, 0], list(sizes), set()
        for axis, start, end in zip(axes, starts, ends):
            axis = axis % rank
            if axis in seen:
                self.fail(node, "an axis appears twice")
            seen.add(axis)
            dim = dims[axis]
            lo = min(max(start + dim if start < 0 else start, 0), dim)
            hi = min(max(end + dim if end < 0 else end, 0), dim)
            if axis not in hwc_axis:
                if lo != 0 or hi != dim:
                    self.fail(node, f"slicing axis {axis} is unsupported")
                continue
            if hi <= lo:
                self.fail(node, "empty slices are unsupported")
            offsets[hwc_axis[axis]], out_shape[hwc_axis[axis]] = lo, hi - lo
        if out_shape != list(self.hwc(node.output[0])):
            self.fail(node, "output shape disagrees with the slice")
        if offsets == [0, 0, 0] and out_shape == sizes:
            self.alias[node.output[0]] = self.canon(data)
            return
        self.add(Op(hx.SLICE, self.tid(data), NONE, self.tid(node.output[0]), offsets, node))

    def op_Split(self, node):
        """Split along the channel axis: one SLICE per output."""
        a = self.attrs(node)
        data = node.input[0]
        dims = self.dims.get(data)
        if dims is None or len(dims) not in (2, 4):
            self.fail(node, "only static rank-2 or rank-4 tensors are supported")
        if int(a.get("axis", 0)) % len(dims) != 1:
            self.fail(node, "only splits along the channel axis (1) are supported")
        ih, iw, ic = self.hwc(data)
        if len(node.input) > 1 and node.input[1]:
            sizes = self.const(node.input[1])
            if sizes is None:
                self.fail(node, "split sizes must be constant")
            sizes = [int(v) for v in np.asarray(sizes).reshape(-1)]
        elif "split" in a:
            sizes = [int(v) for v in a["split"]]
        else:
            if ic % len(node.output):
                self.fail(node, "channels do not divide evenly among the outputs")
            sizes = [ic // len(node.output)] * len(node.output)
        if len(sizes) != len(node.output) or min(sizes) <= 0 or sum(sizes) != ic:
            self.fail(node, "split sizes must be positive, cover the channels and match outputs")
        xid, offset = self.tid(data), 0
        for out, size in zip(node.output, sizes):
            if self.hwc(out) != [ih, iw, size]:
                self.fail(node, "output shape disagrees with the split sizes")
            self.add(Op(hx.SLICE, xid, NONE, self.tid(out), [0, 0, offset], node))
            offset += size

    def op_Pad(self, node):
        a = self.attrs(node)
        mode = a.get("mode", b"constant")
        if (mode.decode() if isinstance(mode, bytes) else mode) != "constant":
            self.fail(node, "only constant padding is supported")
        pads = a.get("pads")
        if pads is None:
            pads = self.const(node.input[1]) if len(node.input) > 1 and node.input[1] else None
        if pads is None or len(node.input) > 3 and node.input[3]:
            self.fail(node, "pads must be constant and axes unsupported")
        value = self.const(node.input[2]) if len(node.input) > 2 and node.input[2] else a.get(
            "value", 0.0
        )
        if value is None or float(np.asarray(value).reshape(())) != 0.0:
            self.fail(node, "only a zero pad value is supported")
        pads = [int(v) for v in pads]
        if len(self.dims[node.input[0]]) != 4 or len(pads) != 8 or min(pads) < 0 or pads[0] or pads[4]:
            self.fail(node, "only non-negative padding of rank-4 tensors, not the batch")
        ih, iw, ic = self.hwc(node.input[0])
        if self.hwc(node.output[0]) != [ih + pads[2] + pads[6], iw + pads[3] + pads[7],
                                        ic + pads[1] + pads[5]]:
            self.fail(node, "output shape disagrees with the pads")
        self.add(Op(hx.PAD, self.tid(node.input[0]), NONE, self.tid(node.output[0]),
                    [pads[2], pads[3], pads[1]], node))

    def op_Resize(self, node):
        a = self.attrs(node)
        text = lambda k, d: (a.get(k, d).decode() if isinstance(a.get(k, d), bytes) else a.get(k, d))
        mode = text("mode", "nearest")
        if mode not in ("linear", "nearest"):
            self.fail(node, "only linear (bilinear) and nearest resize are supported")
        if a.get("antialias", 0):
            self.fail(node, "antialias is unsupported")
        coord = text("coordinate_transformation_mode", "half_pixel")
        ih, iw, ic = self.hwc(node.input[0])
        oh, ow, oc = self.hwc(node.output[0])
        if ic != oc:
            self.fail(node, "channels must not change")
        if mode == "nearest":
            # HX_RESIZE p[2]: source index floor(out * in / out) per axis.
            rounding = text("nearest_mode", "round_prefer_floor")
            if coord != "asymmetric" or rounding != "floor":
                self.fail(node, "nearest resize needs coordinate_transformation_mode=asymmetric "
                                f"and nearest_mode=floor (got {coord}, {rounding})")
            flags = [0, 0, 1]
        elif coord == "align_corners":
            flags = [1, 0, 0]
        elif coord == "half_pixel" or (coord == "pytorch_half_pixel" and oh > 1 and ow > 1):
            flags = [0, 1, 0]
        else:
            self.fail(node, f"coordinate transformation mode {coord} is unsupported")
        self.add(Op(hx.RESIZE, self.tid(node.input[0]), NONE, self.tid(node.output[0]), flags,
                    node))

    # ------------------------------------------------------------ finishing
    def _split_wide(self, op):
        """A dense convolution whose packed weights exceed the HMX budget becomes several
        convolutions over output-channel chunks joined by channel Concats."""
        co, cig, kh, kw = op.weights.shape
        tiles, ostride = (kh * kw * cig + 31) // 32, (co + 31) & ~31
        if hx.hmx_fits(tiles, ostride):
            return [op]
        chunk = max(
            (c for c in range(32, ostride + 1, 32) if hx.hmx_fits(tiles, c)), default=0
        )
        if not chunk:  # input too deep: reported by fp16_blockers, runs unpacked on FP32
            return [op]
        oh, ow, _ = self.shapes[op.out]
        out, acc_id, acc_c = [], None, 0
        for lo in range(0, co, chunk):
            hi = min(lo + chunk, co)
            last = hi == co
            part = op.out if (last and lo == 0) else self.temp("split", [oh, ow, hi - lo])
            piece = Op(hx.CONV, op.x, NONE, part, op.p, op.node)
            piece.dilation = op.dilation
            piece.weights, piece.bias = op.weights[lo:hi], op.bias[lo:hi]
            out.append(piece)
            if acc_id is None:
                acc_id, acc_c = part, hi - lo
                continue
            acc_c += hi - lo
            merged = op.out if last else self.temp("split", [oh, ow, acc_c])
            out.append(Op(hx.CONCAT, acc_id, part, merged, [2], op.node))
            acc_id = merged
        return out

    def _finish_layers(self):
        """Split oversized layers, then pack weights (after all folding) into the constant
        pool and the p[] fields."""
        ops = []
        for op in self.ops:
            ops.extend(self._split_wide(op) if op.kind == hx.CONV and op.weights is not None
                       else [op])
        self.ops = ops
        for op in self.ops:
            if op.weights is None:
                continue
            w = op.weights
            co = w.shape[0]
            ostride = (co + 31) & ~31
            if op.kind == hx.CONV:
                wk = np.pad(w.transpose(2, 3, 1, 0), ((0, 0),) * 3 + ((0, ostride - co),))
            else:
                wk = np.pad(w[:, 0].transpose(1, 2, 0), ((0, 0), (0, 0), (0, ostride - co)))
            op.p[6] = self.pool.emit(wk)
            op.p[7] = self.pool.emit(_pad32(op.bias))
            if op.kind == hx.CONV:
                packed, tiles = hx.pack_hmx_weights(wk)
                if hx.hmx_fits(tiles, ostride):
                    op.p[9], op.p[10] = self.pool.emit(packed, "<f2"), tiles

    def _compact_conv(self, n, name):
        w = self.init.get(n.input[1]) if n.op_type == "Conv" and len(n.input) > 1 else None
        if w is None or n.input[0] != name or self.attrs(n).get("group", 1) != 1:
            return False
        a = self.attrs(n)
        if self.dilation(n, a) != (1, 1):  # the compact-RGB path reads undilated windows
            return False
        k = w.shape[2:]
        s = list(a.get("strides", [1, 1]))
        return k[0] == k[1] and k[0] in (3, 5) and s[0] == s[1] and s[0] <= 2

    def _compact_inputs(self):
        compact = set()
        if self.compact_input == "off":
            return compact
        for name in self.inputs:
            s = self.dims.get(name)
            if not s or len(s) != 4 or s[1] != 3:
                continue
            users = [n for n in self.nodes if name in n.input]
            # The DSP's compact-RGB fast path takes exactly one square 3x3 or 5x5 dense
            # convolution with equal strides of at most 2; anything else reads a blocked,
            # channel-padded input instead.
            if len(users) == 1 and self._compact_conv(users[0], name):
                compact.add(self.ids[name])
        return compact

    def _write(self, destination, source):
        for name in self.outputs:
            if self.canon(name) in self.inputs:
                raise ValueError(f"graph output {name} is just a graph input")
        out_ids, out_parts = [], []
        for name in self.outputs:
            if self.output_parts and name in self.output_parts:
                for part in self.output_parts[name]:
                    out_ids.append(self.tid(part["tensor"]))
                    out_parts.append(part)
            else:
                out_ids.append(self.tid(name))
                out_parts.append({"tensor": name, "channels": [0, self.hwc(name)[2]]})
        for i in out_ids:
            if i not in self.producer:
                raise ValueError("every graph output must be produced by a lowered operation")
        input_ids = [self.ids[n] for n in self.inputs]
        compact = self._compact_inputs()

        def hwc(i):
            return self.shapes[i]

        def stride(i):
            return 3 if i in compact else (self.shapes[i][2] + 31) & ~31

        ops = [[o.kind, o.x, o.y, o.z, o.out] + o.p for o in self.ops]
        blockers = fp16_blockers(ops, input_ids, out_ids, hwc, stride)
        if len(self.pool.data) > CONSTANT_POOL_LIMIT:
            raise ValueError(
                f"constant pool is {len(self.pool.data)} bytes (FP32 weights and packed FP16 "
                f"weights); the tensor runtime accepts at most {CONSTANT_POOL_LIMIT} bytes")
        return hx.write(
            ops, input_ids, out_ids, hwc, stride, self.pool, destination,
            {
                "source": str(Path(source).resolve()),
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "ops": self.source_ops,
                "shapes": {i: list(s) for i, s in self.shapes.items()},
                "extra": {
                    "input_tensors": self.inputs,
                    "output_tensors": out_parts,
                    "fp16_plan": {"eligible": not blockers, "blockers": blockers},
                },
            },
        )


def fp16_blockers(ops, inputs, outputs, hwc, stride):
    """Reasons the DSP's blocked FP16 plan would reject this graph.

    Mirrors `supported()` in hexecutor/native/tensor/fp16.c. A graph with blockers still
    lowers and runs on the FP32 path; it just loses the FP16 speed. VTCM capacity and the
    live-tensor limit are checked by the DSP at session open and are not predicted here.
    """
    reasons, compact_conv = [], False
    for n, op in enumerate(ops):
        kind, x, p = op[0], op[1], op[5:]
        if stride(x) == 3 and kind != hx.CONV:
            reasons.append(f"op {n}: compact RGB input feeds a non-convolution")
        if kind == hx.CONV:
            if not p[10]:
                reasons.append(f"op {n}: convolution has no HMX packing (weights too large)")
            elif stride(x) == 3:
                if p[0] != p[1] or p[0] not in (3, 5) or p[2] != p[3] or p[2] > 2:
                    reasons.append(f"op {n}: compact-input convolution must be 3x3 or 5x5, stride <= 2")
                if compact_conv:
                    reasons.append(f"op {n}: second compact-input convolution")
                compact_conv = True
            elif (p[0], p[1], p[2], p[3], p[4], p[5]) != (1, 1, 1, 1, 0, 0) and p[10] > 126:
                reasons.append(f"op {n}: K = {p[0] * p[1] * hwc(x)[2]} exceeds 4032 (im2col tile)")
    return reasons


def lower(source, destination, compact_input="auto"):
    model = onnx.load(str(source))
    lowerer = Lowerer(model, compact_input)
    write = lowerer.run()
    return write(Path(destination), source)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path, help="destination prefix (.hxfg/.bin/.json)")
    p.add_argument(
        "--compact-input",
        choices=("auto", "off"),
        default="auto",
        help="store a 3-channel input compactly when only dense convolutions read it",
    )
    a = p.parse_args()
    m = lower(a.model, a.output, a.compact_input)
    print(
        f"{len(m['ops'])} operations, arena {m['arena_bytes']} bytes, "
        f"constants {m['constant_bytes']} bytes"
    )
    plan = m["fp16_plan"]
    if plan["eligible"]:
        print("blocked FP16 plan: eligible")
    else:
        print("blocked FP16 plan: NOT eligible (FP32 path only):")
        for r in plan["blockers"]:
            print("  -", r)


if __name__ == "__main__":
    main()
