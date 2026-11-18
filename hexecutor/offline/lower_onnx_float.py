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
(Flatten, Reshape, Squeeze, Unsqueeze, Identity, Dropout) are aliases when they only
move a 1x1 spatial extent; anything that would mix spatial and channel order is
rejected because HXFG stores tensors as HWC while ONNX is NCHW.

Supported operators: Conv, Gemm, MatMul (constant right operand), BatchNormalization,
Add, Sub, Mul, Div (constant divisor), Relu, Clip, LeakyRelu, PRelu, Sigmoid, Tanh,
HardSigmoid, HardSwish, Gelu (exact), MaxPool, AveragePool, GlobalAveragePool,
ReduceMean over (2, 3), Concat, Pad, Resize (linear), Identity, Dropout, Flatten,
Reshape, Squeeze, Unsqueeze.

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


class Lowerer:
    def __init__(self, model, compact_input="auto"):
        self.model = model
        self.graph = model.graph
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
        self.init = {i.name: numpy_helper.to_array(i) for i in self.graph.initializer}
        self.nodes = []
        for node in self.graph.node:  # Constant nodes become initializers
            if node.op_type == "Constant":
                self.init[node.output[0]] = self._constant_value(node)
            else:
                self.nodes.append(node)
        self.dims = {}
        for v in list(inferred.value_info) + list(inferred.input) + list(inferred.output):
            dims = v.type.tensor_type.shape.dim
            if v.type.tensor_type.elem_type not in (0, onnx.TensorProto.FLOAT):
                self.dims[v.name] = None  # non-float32 value
            elif all(d.HasField("dim_value") for d in dims):
                self.dims[v.name] = [d.dim_value for d in dims]
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
        if len(s) == 2 and s[0] == 1:
            return [1, 1, s[1]]
        raise ValueError(f"unsupported shape {s} for {name} (batch-one rank 2 or 4 only)")

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

    def spatial(self, node, a, x, out, k):
        """(stride, pad_top, pad_left) of a windowed operation; checks the output size."""
        ih, iw, _ = self.hwc(x)
        oh, ow, _ = self.hwc(out)
        s = list(a.get("strides", [1, 1]))
        if list(a.get("dilations", [1, 1])) != [1, 1]:
            self.fail(node, "dilation is unsupported")
        mode = a.get("auto_pad", b"NOTSET")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        if mode in ("SAME_UPPER", "SAME_LOWER"):
            tot = [max((oh - 1) * s[0] + k[0] - ih, 0), max((ow - 1) * s[1] + k[1] - iw, 0)]
            top = [t // 2 if mode == "SAME_UPPER" else t - t // 2 for t in tot]
            pads = top + [tot[0] - top[0], tot[1] - top[1]]
        elif mode == "VALID":
            pads = [0, 0, 0, 0]
        else:
            pads = list(a.get("pads", [0, 0, 0, 0]))
        if (ih + pads[0] + pads[2] - k[0]) // s[0] + 1 != oh or (
            iw + pads[1] + pads[3] - k[1]
        ) // s[1] + 1 != ow:
            self.fail(node, "output size disagrees with the window geometry")
        if pads[0] >= k[0] or pads[1] >= k[1]:
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
    def _new_layer(self, node, kind, x_name, out_name, weights, bias, p):
        op = Op(kind, self.tid(x_name), NONE, self.tid(out_name), p, node)
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
        s, pt, pl = self.spatial(node, a, x, node.output[0], [kh, kw])
        if group == 1 and cig == ic:
            kind = hx.CONV
        elif group == co == ic and cig == 1:
            kind = hx.DEPTHWISE
        else:
            self.fail(node, "only dense or multiplier-one depthwise groups are supported")
        self._new_layer(node, kind, x, node.output[0], w, np.asarray(bias).reshape(-1),
                        [kh, kw, s[0], s[1], pt, pl])

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
        if out not in self.outputs:
            self.fail(node, "an output wider than the HMX budget must be a graph output")
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

    def op_Sigmoid(self, node):
        self.add(Op(hx.LOGISTIC, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                    node))

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

    def op_ReduceMean(self, node):
        a = self.attrs(node)
        axes = a.get("axes")
        if axes is None and len(node.input) > 1:
            axes = self.const(node.input[1])
        if axes is None or sorted(int(v) % 4 for v in axes) != [2, 3]:
            self.fail(node, "only a reduction over axes (2, 3) is supported")
        self._mean(node)

    # ------------------------------------------------------------ shape operators
    def _alias_shape(self, node):
        x, out = node.input[0], node.output[0]
        if self.hwc(x) != self.hwc(out):
            self.fail(node, "it mixes spatial and channel order, which HWC storage cannot alias")
        self.alias[out] = self.canon(x)

    op_Identity = op_Dropout = op_Flatten = _alias_shape
    op_Reshape = op_Squeeze = op_Unsqueeze = _alias_shape

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
        if text("mode", "nearest") != "linear":
            self.fail(node, "only linear (bilinear) resize is supported")
        if a.get("antialias", 0):
            self.fail(node, "antialias is unsupported")
        coord = text("coordinate_transformation_mode", "half_pixel")
        ih, iw, ic = self.hwc(node.input[0])
        oh, ow, oc = self.hwc(node.output[0])
        if ic != oc:
            self.fail(node, "channels must not change")
        if coord == "align_corners":
            flags = [1, 0]
        elif coord == "half_pixel" or (coord == "pytorch_half_pixel" and oh > 1 and ow > 1):
            flags = [0, 1]
        else:
            self.fail(node, f"coordinate transformation mode {coord} is unsupported")
        self.add(Op(hx.RESIZE, self.tid(node.input[0]), NONE, self.tid(node.output[0]), flags,
                    node))

    # ------------------------------------------------------------ finishing
    def _finish_layers(self):
        """Pack layer weights (after all folding) into the constant pool and p[] fields."""
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

    def _compact_inputs(self):
        compact = set()
        if self.compact_input == "off":
            return compact
        for name in self.inputs:
            s = self.dims.get(name)
            if not s or len(s) != 4 or s[1] != 3:
                continue
            users = [n for n in self.nodes if name in n.input]
            if users and all(
                n.op_type == "Conv" and n.input[0] == name and self.attrs(n).get("group", 1) == 1
                and self.init.get(n.input[1]) is not None
                for n in users
            ):
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

        ops = [[o.kind, o.x, o.y, NONE, o.out] + o.p for o in self.ops]
        blockers = fp16_blockers(ops, input_ids, out_ids, hwc, stride)
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
            elif (p[0], p[1], p[2], p[3], p[4], p[5]) != (1, 1, 1, 1, 0, 0):
                reasons.append(
                    f"op {n}: {p[0]}x{p[1]} stride-{p[2]} convolution (only 1x1 is blocked-FP16 "
                    "except on the compact input)"
                )
        if kind == hx.DEPTHWISE and (
            p[0] != p[1] or p[0] not in (3, 5) or p[2] != p[3] or p[2] not in (1, 2)
        ):
            reasons.append(f"op {n}: depthwise must be square 3x3 or 5x5 with stride 1 or 2")
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
