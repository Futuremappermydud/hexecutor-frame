#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
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

Transformer lowering (batch one, static shapes): a rank-3 `[1, L, K]` tensor read or written
by MatMul, last-axis LayerNormalization or last-axis Softmax is a token matrix, HWC with h = L,
w = 1, c = K, not NCHW with H = 1. Token status flows through elementwise operators, Concat,
Gather, reductions and reshape-like operators. A MatMul with a constant right operand is a 1x1
convolution over the L rows; two runtime operands lower to HX_MATMUL (p[0] = transB from the
operand's layout, a transposed operand being a logical view). Attention is recognised from the
exporter chain: Reshape to a `[1, L, H, dh]` view, Transpose to `[1, H, L, dh]` (per-head channel
slices; K transposed with `[0, 2, 3, 1]` is a transposed operand), one HX_MATMUL per head, Mul or
Div by a scalar folded into the QK product's p[1], Softmax over the last axis (HX_SOFTMAX per
row), the value product, and a Transpose `[0, 2, 1, 3]` merged back by a channel Concat before
the Reshape to `[1, L, D]`. A Transpose `[0, 2, 1]` of a conv-style `[1, C, L]` (patch
embedding) is a logical reshape. Gather of a contiguous run on the sequence axis is a slice.
Anything else on token or head tensors raises an error naming the node.

The manifest's `fp16_plan` entry lists what keeps the graph out of the DSP's blocked
FP16 plan (see `fp16_blockers`); such a graph still lowers and runs on the FP32 path.

Requires numpy and onnx. Nothing here is used by the native host or engine.
"""

import argparse
import collections
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
MATMUL = hx.MATMUL
RMSNORM = hx.RMSNORM
SDPA = hx.SDPA
ATTN_SOFTMAX = hx.ATTN_SOFTMAX
# hx_graph_open (hexecutor/native/tensor/graph.c) rejects constant pools above 32 MiB.
CONSTANT_POOL_LIMIT = 0x4000000
# hx_graph_open rejects more operations than this (HXFG_MAX_OPS).
OPERATION_LIMIT = 1024

# One matrix of a runtime tensor: `rows` x `cols` physical elements of tensor `id` (h = rows,
# c = cols). `trans` reads it transposed (logical [cols, rows]); `op` produced it, or None.
Mat = collections.namedtuple("Mat", "id rows cols trans op")
ELEMENTWISE = frozenset({
    "Add", "Sub", "Mul", "Div", "Relu", "Clip", "LeakyRelu", "PRelu", "Sigmoid", "Tanh",
    "HardSigmoid", "HardSwish", "Gelu", "Exp", "Sqrt", "Abs", "Neg", "Erf", "Identity",
    "Dropout", "Cast", "Silu",
})
REDUCES = frozenset({"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"})
# Operators that pass token status from one of their inputs to their outputs.
TOKEN_LINKED = ELEMENTWISE | REDUCES | frozenset({
    "Concat", "Gather", "GatherElements", "GatherND", "ScatterND", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "LayerNormalization",
    "RMSNormalization", "RMSNorm", "SimplifiedLayerNormalization", "Softmax", "Split", "Slice",
})
TOKEN_OPS = TOKEN_LINKED | frozenset({"MatMul", "Transpose", "Einsum", "ScaledDotProductAttention", "sdpa"})
HEAD_OPS = frozenset({"MatMul", "Mul", "Div", "Softmax", "Transpose", "Identity", "Einsum", "ScaledDotProductAttention", "sdpa"})
SPLIT_OPS = frozenset({"Transpose", "Reshape", "Identity"})


def is_causal_mask(const):
    arr = np.asarray(const, dtype=np.float32)
    if arr.ndim < 2:
        return False
    h, w = arr.shape[-2], arr.shape[-1]
    if h != w or h < 2:
        return False
    for i in range(h):
        for j in range(w):
            val = arr[..., i, j]
            if j <= i:
                if not np.allclose(val, 0.0, atol=1e-3):
                    return False
            else:
                if not np.all(val <= -100.0):
                    return False
    return True


def _narrow_int64(arr, name=""):
    if isinstance(arr, np.ndarray) and arr.dtype == np.int64:
        if np.any(arr < -2147483648) or np.any(arr > 2147483647):
            raise ValueError(f"int64 constant {name} out of int32 range: [{arr.min()}, {arr.max()}]")
        return arr.astype(np.int32)
    return arr


def _pad32(values):
    arr = np.asarray(values)
    if arr.dtype.kind in ("i", "u", "b"):
        arr = arr.astype(np.int32).reshape(-1)
        pad = ((arr.size + 31) & ~31) - arr.size
        return np.pad(arr, (0, pad))
    values = arr.astype(np.float32).reshape(-1)
    return np.pad(values, (0, ((values.size + 31) & ~31) - values.size))


class Op:
    """One HXFG operation under construction. Layers keep unpacked weights until the end."""

    def __init__(self, kind, x, y, out, p=(), node=None, z=NONE):
        self.kind, self.x, self.y, self.out = kind, x, y, out
        self.p = (list(p) + [0] * 11)[:11]
        self.node = node
        self.weights = None  # layers: float32 [out, in/group, kh, kw]
        self.bias = None
        self.foldable = False  # a following affine node may be folded in
        self.fused = False  # an activation has been fused
        self.dilation = (1, 1)  # conv/depthwise only; encoded as the HXFG z field
        self.z_id = z if z != NONE else None  # tensor id for ops with a third input or output

    @property
    def z(self):
        """HXFG z: tensor id for ops like SDPA, else NONE or (dh << 8) | dw for conv."""
        if self.z_id is not None:
            return self.z_id
        dh, dw = self.dilation
        return NONE if (dh, dw) == (1, 1) else (dh << 8) | dw


def _infer_shapes(model):
    """Shape inference with data propagation, so Shape/Gather/Concat size chains resolve."""
    return onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True).graph


def _static_dims(graph):
    """Tensor name -> static dimensions (None for unsupported types, absent when dynamic)."""
    dims = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        shape = v.type.tensor_type.shape.dim
        elem_type = v.type.tensor_type.elem_type
        if elem_type not in (0, onnx.TensorProto.FLOAT):
            if elem_type in (onnx.TensorProto.BOOL, onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                if all(d.HasField("dim_value") for d in shape):
                    dims[v.name] = [d.dim_value for d in shape]
            else:
                dims[v.name] = None  # non-float32 value
        elif all(d.HasField("dim_value") for d in shape):
            dims[v.name] = [d.dim_value for d in shape]
    return dims


def _elem_types(graph):
    """Tensor name -> element type enum."""
    types = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        types[v.name] = v.type.tensor_type.elem_type
    return types


class Lowerer:
    def __init__(self, model, compact_input="auto"):
        self.model = model
        self.graph = model.graph
        inferred = _infer_shapes(model)
        self.init = {i.name: _narrow_int64(numpy_helper.to_array(i), i.name) for i in self.graph.initializer}
        self.nodes = []
        for node in self.graph.node:  # Constant nodes become initializers
            if node.op_type == "Constant":
                self.init[node.output[0]] = _narrow_int64(self._constant_value(node), node.output[0])
            else:
                self.nodes.append(node)
        self.dims = _static_dims(inferred)
        self.types = _elem_types(inferred)
        self.dtypes = {}
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
        # Transformer layout state (see the docstring): token tensors, rank-4 head splits of a
        # token tensor (canonical name -> (H, dh)), per-head matrices and transposed token views.
        self.token, self.split_of, self.heads, self.ttrans, self.multihead = set(), {}, {}, {}, set()
        self.head_view, self.causal_masked = set(), set()
        self._classify_tokens()

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
                    self.init[name] = _narrow_int64(np.asarray(value), name)
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
        inits = [
            numpy_helper.from_array(
                v.astype(np.int64) if hasattr(v, "dtype") and v.dtype == np.int32 else np.asarray(v), k
            )
            for k, v in self.init.items()
        ]
        graph = onnx.helper.make_graph(self.nodes, "folded", list(self.graph.input),
                                       list(self.graph.output), inits)
        model = onnx.helper.make_model(graph, opset_imports=self.model.opset_import)
        model.ir_version = self.model.ir_version
        try:
            inferred_graph = _infer_shapes(model)
            inferred = _static_dims(inferred_graph)
            inferred_types = _elem_types(inferred_graph)
        except Exception:  # keep the dimensions of the original graph
            return
        for name, dims in inferred.items():
            self.dims.setdefault(name, dims)
        for name, elem_type in inferred_types.items():
            self.types.setdefault(name, elem_type)
        # Forward shape propagation for ops unhandled or broken by ONNX shape inference
        changed = True
        while changed:
            changed = False
            for n in self.nodes:
                op = n.op_type
                inputs = n.input
                out = n.output[0] if n.output else None
                if not out or out in self.dims:
                    continue
                in0 = inputs[0] if inputs else None
                s0 = self.dims.get(in0)
                if s0 is None:
                    continue
                attrs = self.attrs(n)
                if op in ("RMSNormalization", "RMSNorm", "SimplifiedLayerNormalization", "LayerNormalization", "InstanceNormalization", "Relu", "Sigmoid",
                          "Tanh", "Gelu", "Identity", "Dropout", "Cast", "Softmax"):
                    self.dims[out] = list(s0)
                    changed = True
                elif op in ("Add", "Sub", "Mul", "Div"):
                    self.dims[out] = list(s0)
                    changed = True
                elif op == "Resize":
                    sizes = self.const(inputs[3]) if len(inputs) > 3 and inputs[3] else None
                    scales = self.const(inputs[2]) if len(inputs) > 2 and inputs[2] else (
                        self.const(inputs[1]) if len(inputs) > 1 and inputs[1] and not attrs.get("roi") else None
                    )
                    if scales is None and "scales" in attrs:
                        scales = attrs["scales"]
                    if sizes is None and "sizes" in attrs:
                        sizes = attrs["sizes"]
                    if sizes is not None:
                        self.dims[out] = [int(v) for v in sizes]
                        changed = True
                    elif scales is not None and s0 is not None:
                        self.dims[out] = [int(np.floor(s0[i] * float(scales[i]))) for i in range(len(s0))]
                        changed = True
                elif op == "MatMul":
                    w = self.const(inputs[1]) if len(inputs) > 1 else None
                    if w is not None and len(s0) in (2, 3):
                        self.dims[out] = list(s0[:-1]) + [int(w.shape[-1])]
                        changed = True
                    elif len(inputs) > 1 and inputs[1] in self.dims:
                        s1 = self.dims[inputs[1]]
                        self.dims[out] = list(s0[:-1]) + [s1[-1]]
                        changed = True
                elif op == "Transpose":
                    perm = [int(v) for v in attrs.get("perm", [])]
                    if len(perm) == len(s0):
                        self.dims[out] = [s0[p] for p in perm]
                        changed = True
                elif op == "Split":
                    sizes = self.const(inputs[1]) if len(inputs) > 1 and inputs[1] else attrs.get("split")
                    if sizes is not None:
                        axis = int(attrs.get("axis", 0)) % len(s0)
                        for out_i, sz in zip(n.output, sizes):
                            sh = list(s0)
                            sh[axis] = int(sz)
                            self.dims[out_i] = sh
                        changed = True
                elif op == "Reshape":
                    sh = self.const(inputs[1]) if len(inputs) > 1 else None
                    if sh is not None:
                        sh = [int(v) for v in sh]
                        tot = int(np.prod(s0))
                        if -1 in sh:
                            idx = sh.index(-1)
                            known = int(np.prod([v for v in sh if v != -1]))
                            sh[idx] = tot // known if known else 1
                        self.dims[out] = sh
                        changed = True
                elif op == "Conv":
                    w = self.const(inputs[1]) if len(inputs) > 1 else None
                    if w is not None and len(s0) == 4:
                        co = int(w.shape[0])
                        kh, kw = int(w.shape[2]), int(w.shape[3])
                        strides = attrs.get("strides", [1, 1])
                        pads = attrs.get("pads", [0, 0, 0, 0])
                        dil = attrs.get("dilations", [1, 1])
                        ho = (s0[2] + pads[0] + pads[2] - dil[0] * (kh - 1) - 1) // strides[0] + 1
                        wo = (s0[3] + pads[1] + pads[3] - dil[1] * (kw - 1) - 1) // strides[1] + 1
                        self.dims[out] = [s0[0], co, ho, wo]
                        changed = True
                elif op == "Einsum":
                    eq = attrs.get("equation", "")
                    if isinstance(eq, bytes):
                        eq = eq.decode("utf-8")
                    eq = "".join(eq.split())
                    if "->" in eq and len(inputs) == 2 and in0 in self.dims and inputs[1] in self.dims:
                        s1 = self.dims[inputs[1]]
                        lhs, rhs = eq.split("->")
                        in_eqs = lhs.split(",")
                        dim_map = {}
                        for sym, d in zip(in_eqs[0], s0):
                            dim_map[sym] = d
                        for sym, d in zip(in_eqs[1], s1):
                            dim_map[sym] = d
                        if all(sym in dim_map for sym in rhs):
                            self.dims[out] = [dim_map[sym] for sym in rhs]
                            changed = True
                elif op in ("ScaledDotProductAttention", "sdpa"):
                    if len(inputs) >= 3 and in0 in self.dims and inputs[2] in self.dims:
                        dv = self.dims[inputs[2]]
                        self.dims[out] = list(s0[:-1]) + [dv[-1]]
                        changed = True

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
        val = self.init.get(name)
        if val is not None and isinstance(val, np.ndarray) and val.dtype == np.int64:
            return _narrow_int64(val, name)
        return val

    def _is_multihead(self, name):
        return self.canon(name) in self.multihead

    def _is_head_view(self, name):
        return self.canon(name) in self.head_view

    def hwc(self, name):
        name = self.canon(name)
        s = self.dims.get(name)
        if s is None:
            raise ValueError(f"no static float32 shape for {name}")
        if name in self.heads:
            raise ValueError(f"{name} is a per-head attention tensor with no HWC form")
        if (self._is_multihead(name) or self._is_head_view(name)) and len(s) == 4 and s[0] == 1:
            return [s[1], s[2], s[3]]
        if len(s) == 3 and s[0] == 1 and self._is_token(name):
            return [s[1], 1, s[2]]
        if len(s) == 4 and s[0] == 1 and name in self.split_of:  # heads merged into channels
            return [s[1], 1, s[2] * s[3]]
        if name in self.ttrans:  # logical transpose: only ever a MatMul operand
            return [s[1], 1, s[2]]
        # Higher-rank tensor (e.g. rank-5 [1, 1, H, L, D] or rank-6 [1, 1, 1, 2, 2, 1024]):
        if len(s) in (5, 6) and s[0] == 1:
            if all(d == 1 for d in s[:-3]):
                return list(s[-3:])
            return [int(np.prod(s[1:-2])), int(s[-2]), int(s[-1])]
        if len(s) == 4 and s[0] == 1:
            return [s[2], s[3], s[1]]
        if len(s) == 3 and s[0] == 1:  # squeezed intermediate: NCHW with H = 1
            return [1, s[2], s[1]]
        if len(s) == 2 and s[0] == 1:
            return [1, 1, s[1]]
        raise ValueError(f"unsupported shape {s} for {name} (batch-one rank 2..6 only)")

    def set_dtype(self, name_or_tid, dtype):
        if isinstance(name_or_tid, str):
            canon = self.canon(name_or_tid)
            if canon in self.heads or canon in self.ttrans:
                return
            tid = self.tid(name_or_tid)
        else:
            tid = name_or_tid
        self.dtypes[tid] = dtype

    def get_dtype(self, name_or_tid):
        if isinstance(name_or_tid, str):
            canon = self.canon(name_or_tid)
            if canon in self.heads or canon in self.ttrans:
                return hx.DTYPE_FLOAT32
            if canon in self.ids:
                return self.dtypes.get(self.ids[canon], hx.DTYPE_FLOAT32)
            elem_type = self.types.get(canon, self.types.get(name_or_tid))
            if elem_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                return hx.DTYPE_INT32
            elif elem_type == onnx.TensorProto.BOOL:
                return hx.DTYPE_BOOL
            return hx.DTYPE_FLOAT32
        return self.dtypes.get(name_or_tid, hx.DTYPE_FLOAT32)

    def tid(self, name):
        name = self.canon(name)
        if name in self.heads or name in self.ttrans:
            raise ValueError(f"internal: {name} has no single tensor id")
        if name not in self.ids:
            tid = len(self.ids)
            self.ids[name] = tid
            if name in self.dims:
                self.shapes[tid] = self.hwc(name)
            elem_type = self.types.get(name)
            if elem_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                self.dtypes.setdefault(tid, hx.DTYPE_INT32)
            elif elem_type == onnx.TensorProto.BOOL:
                self.dtypes.setdefault(tid, hx.DTYPE_BOOL)
            else:
                self.dtypes.setdefault(tid, hx.DTYPE_FLOAT32)
        return self.ids[name]

    def temp(self, like, shape, dtype=None):
        """A new internal tensor with an explicit HWC shape."""
        self.synthetic += 1
        name = f"{like}#t{self.synthetic}"
        tid = len(self.ids)
        self.ids[name] = tid
        self.shapes[tid] = list(shape)
        if dtype is not None:
            self.dtypes[tid] = dtype
        elif isinstance(like, int):
            self.dtypes[tid] = self.dtypes.get(like, hx.DTYPE_FLOAT32)
        elif isinstance(like, str):
            canon = self.canon(like)
            if canon in self.ids:
                self.dtypes[tid] = self.dtypes.get(self.ids[canon], hx.DTYPE_FLOAT32)
            else:
                self.dtypes[tid] = hx.DTYPE_FLOAT32
        return tid

    def add(self, op):
        self.ops.append(op)
        self.producer[op.out] = op
        if op.kind == hx.TOPK and op.z_id is not None and op.z_id != NONE:
            self.producer[op.z_id] = op
        return op

    def _is_token(self, name):
        """True for a token tensor [1, L, K] (or a merged head tensor) with the token HWC form."""
        c = self.canon(name)
        if c in self.ttrans or c in self.heads:
            return False
        return c in self.token or (c in self.split_of and len(self.dims.get(c) or ()) == 4)

    def _producer(self, name):
        return self.producer.get(self.tid(name))

    def _mats(self, name):
        """The matrices of a runtime operand: one for a token tensor or transposed view, one per
        head for a per-head tensor. None when the operand is not a matrix."""
        c = self.canon(name)
        if c in self.heads:
            return self.heads[c]
        if c in self.ttrans:
            base = self.canon(self.ttrans[c])
            rows, _, cols = self.hwc(base)
            return [Mat(self.tid(base), rows, cols, True, self._producer(base))]
        if self._is_token(c):
            rows, _, cols = self.hwc(c)
            return [Mat(self.tid(c), rows, cols, False, self._producer(c))]
        return None

    def _emit_matmul(self, node, a, b, out_id):
        """One HX_MATMUL: out = a . b, with b read as [N, K] when it is a transposed matrix."""
        if a.trans:
            self.fail(node, "a transposed left operand is unsupported")
        trans_b = 1 if b.trans else 0
        inner, cols = (b.cols, b.rows) if trans_b else (b.rows, b.cols)
        if inner != a.cols:
            self.fail(node, f"inner dimensions disagree ({a.cols} and {inner})")
        op = self.add(Op(MATMUL, a.id, b.id, out_id, [trans_b, 0], node))
        return Mat(out_id, a.rows, cols, False, op)

    def _emit_batched_matmul(self, node, a_name, b_name, out_name, transB=0, transA=0, scale=0.0):
        da = self.dims.get(a_name)
        db = self.dims.get(b_name)
        if da is None or db is None:
            self.fail(node, "runtime operands require static dimensions")

        scale_bits = 0 if scale == 0.0 else (scale if isinstance(scale, int) else hx.f32_bits(scale))

        if len(da) == 4 and len(db) == 4:
            H = da[1]
            if db[1] != H:
                self.fail(node, f"head counts disagree ({H} vs {db[1]})")
            Ka = da[2] if transA else da[3]
            Kb = db[3] if transB else db[2]
            if Ka != Kb:
                self.fail(node, f"inner dimensions disagree ({Ka} vs {Kb})")

            self.multihead.add(self.canon(a_name))
            self.multihead.add(self.canon(b_name))
            self.multihead.add(self.canon(out_name))
            op = self.add(Op(MATMUL, self.tid(a_name), self.tid(b_name), self.tid(out_name),
                             [transB, scale_bits, H, transA], node))
            return op

        if len(da) == 3 and len(db) == 3:
            Ka = da[1] if transA else da[2]
            Kb = db[2] if transB else db[1]
            if Ka != Kb:
                self.fail(node, f"inner dimensions disagree ({Ka} vs {Kb})")
            self.token.add(self.canon(a_name))
            self.token.add(self.canon(b_name))
            self.token.add(self.canon(out_name))
            op = self.add(Op(MATMUL, self.tid(a_name), self.tid(b_name), self.tid(out_name),
                             [transB, scale_bits, 1, transA], node))
            return op

        self.fail(node, f"unsupported batched MatMul operands shapes {da} and {db}")

    def _runtime_matmul(self, node):
        """MatMul of two runtime operands: batched multi-head, token tensors or per-head attention."""
        a_name, b_name, out = node.input[0], node.input[1], node.output[0]
        da = self.dims.get(a_name)
        db = self.dims.get(b_name)
        if da is not None and db is not None and len(da) == 4 and len(db) == 4 and self.canon(a_name) not in self.heads:
            Ka = da[3]
            if db[2] == Ka:
                transB = 0
            elif db[3] == Ka:
                transB = 1
            else:
                self.fail(node, f"inner dimensions disagree ({Ka} vs {db[2]},{db[3]})")
            return self._emit_batched_matmul(node, a_name, b_name, out, transB=transB)

        if da is not None and db is not None and len(da) == 3 and len(db) == 3 and self.canon(a_name) not in self.heads and self.canon(b_name) not in self.heads:
            transA_val = 1 if self.canon(a_name) in self.ttrans else 0
            transB_val = 1 if self.canon(b_name) in self.ttrans else 0
            real_a = self.ttrans[self.canon(a_name)] if transA_val else a_name
            real_b = self.ttrans[self.canon(b_name)] if transB_val else b_name
            Ka = da[1] if transA_val else da[2]
            Kb = db[2] if transB_val else db[1]
            if Ka == Kb:
                return self._emit_batched_matmul(node, real_a, real_b, out, transB=transB_val, transA=transA_val)

        A, B = self._mats(a_name), self._mats(b_name)
        if A is None or B is None:
            self.fail(node, "runtime operands must be token tensors [1, L, K] or per-head "
                            "attention tensors")
        heads = self.canon(a_name) in self.heads
        if (self.canon(b_name) in self.heads) != heads or len(A) != len(B):
            self.fail(node, "operands must both be per-head tensors of one head count, or both "
                            "token tensors")
        if heads:
            shape = self.dims[out]
            if len(shape) != 4 or shape[1] != len(A):
                self.fail(node, f"output shape {shape} disagrees with the head count")
            new = []
            for a, b in zip(A, B):
                cols = b.rows if b.trans else b.cols
                if [shape[2], shape[3]] != [a.rows, cols]:
                    self.fail(node, "output shape disagrees with the per-head product")
                t = self.temp(out, [a.rows, 1, cols])
                new.append(self._emit_matmul(node, a, b, t))
            self.heads[out] = new
            return
        if out not in self.token or len(A) != 1:
            self.fail(node, "the product of two token tensors must be a token tensor")
        cols = B[0].rows if B[0].trans else B[0].cols
        if self.hwc(out) != [A[0].rows, 1, cols]:
            self.fail(node, "output shape disagrees with the product")
        self._emit_matmul(node, A[0], B[0], self.tid(out))

    def _fold_scale(self, name, out, factor):
        """Fold a scalar multiplier into the HX_MATMUL that produced `name` (p[1]), when that
        product has no other user and no scale yet. The output then aliases the product."""
        if factor == 0.0 or self.canon(name) in self.ttrans or not self.sole_user(name):
            return False
        mats = self._mats(name)
        if mats is not None and all(m.op is not None and m.op.kind == MATMUL and not m.op.p[1] for m in mats):
            bits = 0 if factor == 1.0 else hx.f32_bits(factor)
            for m in mats:
                m.op.p[1] = bits
            self.alias[out] = self.canon(name)
            return True
        c = self.canon(name)
        prod = self.producer.get(self.ids.get(c, -1))
        if prod is not None and prod.kind == MATMUL and not prod.p[1]:
            bits = 0 if factor == 1.0 else hx.f32_bits(factor)
            prod.p[1] = bits
            self.alias[out] = c
            if self._is_multihead(name):
                self.multihead.add(out)
            return True
        return False

    def _scale(self, node, name, out, factor):
        """out = name * factor for a scalar factor on a token or per-head tensor: folded into
        the producing product when possible, else one multiply per matrix."""
        if self._fold_scale(name, out, factor):
            return
        c = self.canon(name)
        if c in self.heads:
            new = []
            for m in self.heads[c]:
                vec = np.full(m.cols, factor, np.float32)
                t = self.temp(out, [m.rows, 1, m.cols])
                op = self.add(Op(hx.BINARY, m.id, NONE, t,
                                 [hx.BIN_MUL, 1, self.pool.emit(_pad32(vec)), 0], node))
                new.append(Mat(t, m.rows, m.cols, m.trans, op))
            self.heads[out] = new
            return
        vec = np.full(self.hwc(out)[2], factor, np.float32)
        self._affine(node, name, out, vec, np.zeros_like(vec))

    def _check_operands(self, node):
        """Fail closed when a token or per-head tensor reaches an operator that cannot take it."""
        for name in node.input:
            if not name or name in self.init:
                continue
            c = self.canon(name)
            rank = len(self.dims.get(name) or ())
            if c in self.ttrans:
                ok, kind = node.op_type == "MatMul", "transposed token"
            elif c in self.heads:
                ok, kind = node.op_type in HEAD_OPS, "per-head attention"
            elif rank == 4 and c in self.split_of:
                ok, kind = node.op_type in SPLIT_OPS, "[1, L, H, dh] head view"
            elif rank == 3 and self._is_token(name):
                ok, kind = node.op_type in TOKEN_OPS, "token [1, L, K]"
            else:
                continue
            if not ok:
                self.fail(node, f"{name} is a {kind} tensor, which this operator does not accept")

    def _classify_tokens(self):
        """Mark the rank-3 tensors in token layout. Seeds: every operand and result of a MatMul,
        the input of a last-axis LayerNormalization or Softmax, and the input of a reduction.
        Token status then flows through TOKEN_LINKED operators until nothing changes."""
        def r3(name):
            return bool(name) and name not in self.init and len(self.dims.get(name) or ()) == 3

        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                names = [n for n in list(node.input) + list(node.output) if r3(n)]
                if not names:
                    continue
                op = node.op_type
                last_axis = (op in ("LayerNormalization", "RMSNormalization", "RMSNorm", "SimplifiedLayerNormalization", "Softmax", "Split") and r3(node.input[0])
                             and int(self.attrs(node).get("axis", -1)) % 3 == 2)
                seed = op == "MatMul" or last_axis or (op in REDUCES and r3(node.input[0])) or (op == "Gather" and r3(node.output[0]))
                link = op in TOKEN_LINKED and any(n in self.token for n in names)
                if seed or link:
                    for n in names:
                        if n not in self.token:
                            self.token.add(n)
                            changed = True
        for m in self._find_rmsnorm_decomposed()["fused"].values():
            if r3(m["x"]):
                self.token.add(self.canon(m["x"]))
            if r3(m["out"]):
                self.token.add(self.canon(m["out"]))

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
        """Per-channel float32 vector from a constant broadcast against `out_name`, else None.
        For a token tensor [1, L, K] the channels are the last axis and the constant must not vary
        over the sequence."""
        shape = self.dims[out_name]
        token = self._is_token(out_name)
        c = shape[2] if token else shape[1]
        v = np.asarray(value, dtype=np.float32)
        padded = (1,) * (len(shape) - v.ndim) + tuple(v.shape)
        if len(padded) != len(shape) or padded[0] != 1:
            return None
        if token:
            if padded[1] != 1 or padded[2] not in (1, c):
                return None
        else:
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
        rmsnorms = self._find_rmsnorm_decomposed()
        for node in self.nodes:
            if id(node) in silu["skip"] or id(node) in rmsnorms["skip"]:
                continue
            handler = getattr(self, "op_" + node.op_type, None)
            if handler is None and id(node) not in rmsnorms["fused"]:
                self.fail(node, "unsupported operator")
            self._check_operands(node)
            if id(node) in silu["mul"]:
                self._emit_act(node, silu["mul"][id(node)], hx.FN_SILU)
            elif id(node) in rmsnorms["fused"]:
                self._emit_rmsnorm_fused(node, rmsnorms["fused"][id(node)])
            else:
                handler(node)
            self.source_ops.append({"onnx_op": node.op_type, "output": node.output[0]})
        self._finish_layers()
        return self._write

    def _find_rmsnorm_decomposed(self):
        by_output = {n.output[0]: n for n in self.nodes if n.output}
        skip = set()
        fused = {}

        def get_const(name):
            return self.const(name)

        def is_last_axis_reduce(n, rank):
            axes = self.attrs(n).get("axes")
            if axes is None and len(n.input) > 1:
                axes_c = get_const(n.input[1])
                if axes_c is not None:
                    axes = [int(a) for a in axes_c.reshape(-1)]
            if axes is None:
                axes = [-1]
            return any(int(a) % rank == (rank - 1) for a in axes)

        for n in self.nodes:
            # Form 1: Div(x, Sqrt(Add(ReduceMean(Pow(x, 2)), eps)))
            if n.op_type == "Div" and len(n.input) == 2:
                x, denom = n.input[0], n.input[1]
                sqrt_node = by_output.get(denom)
                if (sqrt_node is not None and sqrt_node.op_type == "Sqrt"
                        and self.users.get(denom, 0) == 1 and denom not in self.outputs):
                    add_node = by_output.get(sqrt_node.input[0])
                    if (add_node is not None and add_node.op_type == "Add" and len(add_node.input) == 2
                            and self.users.get(sqrt_node.input[0], 0) == 1 and sqrt_node.input[0] not in self.outputs):
                        mean_in, eps_in = None, None
                        for a, b in ((add_node.input[0], add_node.input[1]), (add_node.input[1], add_node.input[0])):
                            c_val = get_const(b)
                            if c_val is not None and c_val.size == 1:
                                mean_in, eps_in = a, float(c_val.reshape(-1)[0])
                                break
                        if mean_in is not None:
                            mean_node = by_output.get(mean_in)
                            rank = len(self.dims.get(x) or ())
                            if (mean_node is not None and mean_node.op_type == "ReduceMean"
                                    and rank in (2, 3) and is_last_axis_reduce(mean_node, rank)
                                    and self.users.get(mean_in, 0) == 1 and mean_in not in self.outputs):
                                pow_node = by_output.get(mean_node.input[0])
                                is_pow2 = False
                                if (pow_node is not None and self.users.get(mean_node.input[0], 0) == 1
                                        and mean_node.input[0] not in self.outputs):
                                    if pow_node.op_type == "Pow" and len(pow_node.input) == 2:
                                        p_in, exp_in = pow_node.input[0], get_const(pow_node.input[1])
                                        if p_in == x and exp_in is not None and np.allclose(exp_in, 2.0):
                                            is_pow2 = True
                                    elif pow_node.op_type == "Mul" and len(pow_node.input) == 2:
                                        if pow_node.input[0] == x and pow_node.input[1] == x:
                                            is_pow2 = True
                                if is_pow2:
                                    div_out = n.output[0]
                                    weight = None
                                    final_out = div_out
                                    final_node = n
                                    if self.users.get(div_out, 0) == 1 and div_out not in self.outputs:
                                        consumer = [user_node for user_node in self.nodes if div_out in user_node.input]
                                        if len(consumer) == 1 and consumer[0].op_type == "Mul" and len(consumer[0].input) == 2:
                                            mul_node = consumer[0]
                                            w_name = mul_node.input[1] if mul_node.input[0] == div_out else mul_node.input[0]
                                            w_val = get_const(w_name)
                                            if w_val is not None:
                                                weight = w_val
                                                final_out = mul_node.output[0]
                                                final_node = mul_node
                                                skip.add(id(n))
                                    skip.add(id(pow_node))
                                    skip.add(id(mean_node))
                                    skip.add(id(add_node))
                                    skip.add(id(sqrt_node))
                                    fused[id(final_node)] = {
                                        "x": x,
                                        "out": final_out,
                                        "eps": eps_in,
                                        "weight": weight,
                                        "node": final_node,
                                    }

            # Form 2: Mul(x, Rsqrt(Add(ReduceMean(Pow(x, 2)), eps)))
            if n.op_type == "Mul" and len(n.input) == 2 and id(n) not in fused and id(n) not in skip:
                for x_cand, rsqrt_in in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
                    rsqrt_node = by_output.get(rsqrt_in)
                    if (rsqrt_node is not None and rsqrt_node.op_type in ("Rsqrt",)
                            and self.users.get(rsqrt_in, 0) == 1 and rsqrt_in not in self.outputs):
                        add_node = by_output.get(rsqrt_node.input[0])
                        if (add_node is not None and add_node.op_type == "Add" and len(add_node.input) == 2
                                and self.users.get(rsqrt_node.input[0], 0) == 1 and rsqrt_node.input[0] not in self.outputs):
                            mean_in, eps_in = None, None
                            for a, b in ((add_node.input[0], add_node.input[1]), (add_node.input[1], add_node.input[0])):
                                c_val = get_const(b)
                                if c_val is not None and c_val.size == 1:
                                    mean_in, eps_in = a, float(c_val.reshape(-1)[0])
                                    break
                            if mean_in is not None:
                                mean_node = by_output.get(mean_in)
                                rank = len(self.dims.get(x_cand) or ())
                                if (mean_node is not None and mean_node.op_type == "ReduceMean"
                                        and rank in (2, 3) and is_last_axis_reduce(mean_node, rank)
                                        and self.users.get(mean_in, 0) == 1 and mean_in not in self.outputs):
                                    pow_node = by_output.get(mean_node.input[0])
                                    is_pow2 = False
                                    if (pow_node is not None and self.users.get(mean_node.input[0], 0) == 1
                                            and mean_node.input[0] not in self.outputs):
                                        if pow_node.op_type == "Pow" and len(pow_node.input) == 2:
                                            p_in, exp_in = pow_node.input[0], get_const(pow_node.input[1])
                                            if p_in == x_cand and exp_in is not None and np.allclose(exp_in, 2.0):
                                                is_pow2 = True
                                        elif pow_node.op_type == "Mul" and len(pow_node.input) == 2:
                                            if pow_node.input[0] == x_cand and pow_node.input[1] == x_cand:
                                                is_pow2 = True
                                    if is_pow2:
                                        mul_x_out = n.output[0]
                                        weight = None
                                        final_out = mul_x_out
                                        final_node = n
                                        if self.users.get(mul_x_out, 0) == 1 and mul_x_out not in self.outputs:
                                            consumer = [user_node for user_node in self.nodes if mul_x_out in user_node.input]
                                            if len(consumer) == 1 and consumer[0].op_type == "Mul" and len(consumer[0].input) == 2:
                                                w_mul = consumer[0]
                                                w_name = w_mul.input[1] if w_mul.input[0] == mul_x_out else w_mul.input[0]
                                                w_val = get_const(w_name)
                                                if w_val is not None:
                                                    weight = w_val
                                                    final_out = w_mul.output[0]
                                                    final_node = w_mul
                                                    skip.add(id(n))
                                        skip.add(id(pow_node))
                                        skip.add(id(mean_node))
                                        skip.add(id(add_node))
                                        skip.add(id(rsqrt_node))
                                        fused[id(final_node)] = {
                                            "x": x_cand,
                                            "out": final_out,
                                            "eps": eps_in,
                                            "weight": weight,
                                            "node": final_node,
                                        }
                                        break
        return {"skip": skip, "fused": fused}

    def _emit_rmsnorm_fused(self, node, m):
        x, out = m["x"], m["out"]
        if out not in self.dims and x in self.dims:
            self.dims[out] = list(self.dims[x])
        dims = self.dims.get(x)
        if dims is None:
            self.fail(node, "unknown shape for fused RMSNorm")
        if len(dims) == 3 and dims[0] == 1:
            self.token.add(self.canon(x))
            self.token.add(self.canon(out))
            c = dims[2]
        elif len(dims) == 2 and dims[0] == 1:
            c = dims[1]
        elif len(dims) == 2:
            self.token.add(self.canon(x))
            self.token.add(self.canon(out))
            c = dims[1]
        else:
            c = dims[-1]
        eps = m["eps"]
        weight = m["weight"]
        gamma_off = self.pool.emit(_pad32(self._norm_vector(node, weight, c, "scale"))) if weight is not None else NONE
        p = [hx.f32_bits(eps), gamma_off, NONE] + [0] * 8
        self.add(Op(hx.RMSNORM, self.tid(x), NONE, self.tid(out), p, node))

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
        dil = self.dilation(node, a)
        dh, dw = dil
        ke_h = (kh - 1) * dh + 1
        ke_w = (kw - 1) * dw + 1
        pt, pl, pb, pr = (list(a.get("pads", [0, 0, 0, 0])) + [0, 0, 0, 0])[:4]
        op_h, op_w = (list(a.get("output_padding", [0, 0])) + [0, 0])[:2]
        # Leading and trailing zeros of the zero-inserted input (pads larger than ke-1 are
        # rejected: the padded input would need a negative leading border).
        zt, zl, zb, zr = ke_h - 1 - pt, ke_w - 1 - pl, ke_h - 1 - pb, ke_w - 1 - pr
        if min(zt, zl, zb, zr) < 0:
            self.fail(node, "pads must not exceed the effective kernel size minus one")
        oh, ow, _ = self.hwc(node.output[0])
        if (oh, ow) != ((ih - 1) * s[0] + ke_h - pt - pb + op_h,
                        (iw - 1) * s[1] + ke_w - pl - pr + op_w):
            self.fail(node, "output size disagrees with the transposed geometry")
        zh = zt + (ih - 1) * s[0] + 1 + zb + op_h
        zw = zl + (iw - 1) * s[1] + 1 + zr + op_w
        zid = self.temp(node.output[0], [zh, zw, ic])
        self.add(Op(hx.ZEROINSERT, self.tid(x), NONE, zid, [s[0], s[1], zt, zl], node))
        # Equivalent dense kernel: [ci, co, kh, kw] -> [co, ci, kh, kw], flipped in both axes.
        dense = np.ascontiguousarray(
            np.asarray(w, np.float32).transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
        layer = Op(hx.CONV, zid, NONE, self.tid(node.output[0]), [kh, kw, 1, 1, 0, 0], node)
        if dil != (1, 1):
            layer.dilation = dil
        layer.weights = dense
        layer.bias = np.asarray(bias, np.float32).reshape(-1)
        layer.foldable = True
        self.add(layer)

    def _linear(self, node, x, w, bias, out):
        """Gemm/MatMul as a 1x1 convolution: on a [1, 1, K] tensor, or on a token tensor [1, L, K]
        as a convolution over its L rows (w = 1). Wide outputs are split."""
        if self.canon(x) in self.ttrans:
            self.fail(node, "a transposed token tensor is only a MatMul operand")
        n, k = w.shape
        token = self._is_token(x)
        if token:
            rows, width, ic = self.hwc(x)
            if width != 1 or ic != k:
                self.fail(node, f"input {[rows, width, ic]} does not match a {k}-input weight")
        elif self.hwc(x)[:2] != [1, 1]:
            self.fail(node, "input must be a flat [1, K] tensor")
        tiles = (k + 31) // 32
        chunk = max(
            (c for c in range(32, min((n + 31) & ~31, hx.MAX_CHANNELS) + 1, 32)
             if hx.hmx_fits(tiles, c)),
            default=0,
        )
        if not chunk:
            self.fail(node, "input is too deep for the HMX budget")
        w4 = w.reshape(n, k, 1, 1)
        if token:  # one layer; _split_wide splits an oversized output into channel chunks
            self._new_layer(node, hx.CONV, x, out, w4, bias, [1, 1, 1, 1, 0, 0])
            return
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
        if self.const(node.input[0]) is not None:
            self.fail(node, "constant left operands are unsupported")
        b = self.const(node.input[1])
        if b is None:
            return self._runtime_matmul(node)
        if b.ndim != 2:
            self.fail(node, "right operand must be a constant matrix")
        self._linear(node, node.input[0], b.T.astype(np.float32),
                     np.zeros(b.shape[1], np.float32), node.output[0])

    def op_Einsum(self, node):
        eq = self.attrs(node).get("equation", "")
        if isinstance(eq, bytes):
            eq = eq.decode("utf-8")
        eq = "".join(eq.split())
        if "->" not in eq or len(node.input) != 2:
            self.fail(node, f"only 2-operand Einsum with explicit '->' is supported; got '{eq}'")
        lhs, rhs = eq.split("->")
        in_eqs = lhs.split(",")
        if len(in_eqs) != 2:
            self.fail(node, f"only 2-operand Einsum is supported; got '{eq}'")

        a_name, b_name, out_name = node.input[0], node.input[1], node.output[0]
        eq_a, eq_b, eq_out = in_eqs[0], in_eqs[1], rhs

        contracted = [c for c in eq_a if c in eq_b and c not in eq_out]
        if len(contracted) != 1:
            self.fail(node, f"unsupported Einsum pattern '{eq}': expected exactly 1 contraction axis")
        c_idx = contracted[0]

        idx_a = eq_a.index(c_idx)
        if idx_a == len(eq_a) - 1:
            transA = 0
            m_char = eq_a[-2] if len(eq_a) >= 2 else None
        elif idx_a == len(eq_a) - 2:
            transA = 1
            m_char = eq_a[-1]
        else:
            self.fail(node, f"unsupported Einsum contraction axis position in operand A: '{eq}'")

        idx_b = eq_b.index(c_idx)
        if idx_b == len(eq_b) - 2:
            transB = 0
            n_char = eq_b[-1]
        elif idx_b == len(eq_b) - 1:
            transB = 1
            n_char = eq_b[-2] if len(eq_b) >= 2 else None
        else:
            self.fail(node, f"unsupported Einsum contraction axis position in operand B: '{eq}'")

        batch_a = [c for c in eq_a if c != c_idx and c != m_char]
        batch_b = [c for c in eq_b if c != c_idx and c != n_char]
        batch_out = [c for c in eq_out if c != m_char and c != n_char]
        if batch_a != batch_b or batch_a != batch_out:
            self.fail(node, f"Einsum batch axes mismatch: '{eq}'")

        return self._emit_batched_matmul(node, a_name, b_name, out_name, transB=transB, transA=transA)

    def op_ScaledDotProductAttention(self, node):
        q, k, v = node.input[0], node.input[1], node.input[2]
        mask = node.input[3] if len(node.input) > 3 and node.input[3] else None
        scale_in = node.input[4] if len(node.input) > 4 and node.input[4] else None
        scale = float(self.attrs(node).get("scale", 0.0))
        if scale_in:
            c = self.const(scale_in)
            if c is not None:
                scale = float(c)
        is_causal = int(self.attrs(node).get("is_causal", 0))

        scale_bits = 0 if scale == 0.0 else hx.f32_bits(scale)
        mask_id = self.tid(mask) if mask else hx.NONE
        out = node.output[0]

        dq = self.dims.get(q)
        if dq is None:
            self.fail(node, "SDPA query operand requires static dimensions")
        if len(dq) == 4:
            H = dq[1]
            self.multihead.add(self.canon(q))
            self.multihead.add(self.canon(k))
            self.multihead.add(self.canon(v))
            self.multihead.add(self.canon(out))
            self.add(Op(SDPA, self.tid(q), self.tid(k), self.tid(out),
                        [mask_id, scale_bits, is_causal, H, 0], node, z=self.tid(v)))
        elif len(dq) == 3:
            self.token.add(self.canon(q))
            self.token.add(self.canon(k))
            self.token.add(self.canon(v))
            self.token.add(self.canon(out))
            self.add(Op(SDPA, self.tid(q), self.tid(k), self.tid(out),
                        [mask_id, scale_bits, is_causal, 1, 0], node, z=self.tid(v)))
        else:
            self.fail(node, f"unsupported SDPA query shape {dq}")

    op_sdpa = op_ScaledDotProductAttention

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
        is_comparison = opcode in (
            hx.BIN_EQUAL, hx.BIN_GREATER, hx.BIN_LESS, hx.BIN_GE, hx.BIN_LE, hx.BIN_AND, hx.BIN_OR
        )
        if ca is not None or cb is not None:
            const, tensor, rev = (ca, b_name, 1) if ca is not None else (cb, a_name, 0)
            const_dtype = np.asarray(const).dtype
            tensor_dtype = self.get_dtype(tensor)
            is_int = (tensor_dtype == hx.DTYPE_INT32) or (const_dtype in (np.int32, np.int64))
            is_bool = (tensor_dtype == hx.DTYPE_BOOL) or (const_dtype == np.bool_)
            if is_comparison:
                self.set_dtype(out, hx.DTYPE_BOOL)
            elif is_int:
                self.set_dtype(out, hx.DTYPE_INT32)
            elif is_bool:
                self.set_dtype(out, hx.DTYPE_BOOL)
            else:
                self.set_dtype(out, hx.DTYPE_FLOAT32)

            if not is_int and not is_bool and not is_comparison:
                if self._is_token(tensor) or self._is_multihead(tensor) or self.canon(tensor) in self.heads:
                    if opcode == hx.BIN_MUL and np.asarray(const).size == 1:
                        return self._scale(node, tensor, out,
                                           float(np.asarray(const, np.float64).reshape(())))
                    if self.canon(tensor) in self.heads:
                        self.fail(node, "per-head attention tensors take only a scalar multiplier")
                if self._is_multihead(tensor) and opcode == hx.BIN_ADD and is_causal_mask(const):
                    self.causal_masked.add(self.canon(out))
                    self.alias[out] = self.canon(tensor)
                    self.multihead.add(self.canon(out))
                    return
            if self.dims[tensor] != self.dims[out]:
                self.fail(node, "a constant operand would broadcast the tensor")
            vec = self.channel_vector(node, const, out)
            if vec is not None:
                if not is_int and not is_bool and not is_comparison:
                    if opcode == hx.BIN_MUL and not rev:
                        return self._affine(node, tensor, out, vec, np.zeros_like(vec))
                    if opcode == hx.BIN_ADD:
                        return self._affine(node, tensor, out, np.ones_like(vec), vec)
                    if opcode == hx.BIN_SUB and not rev:
                        return self._affine(node, tensor, out, np.ones_like(vec), -vec)
                    if opcode == hx.BIN_MUL:  # c * x
                        return self._affine(node, tensor, out, vec, np.zeros_like(vec))
                    if opcode == hx.BIN_SUB and rev:
                        # c - x: negate, then add c
                        neg = self.temp(out, self.hwc(out))
                        self.add(Op(hx.BINARY, self.tid(tensor), NONE, neg,
                                    [hx.BIN_MUL, 1, self.pool.emit(_pad32(-np.ones_like(vec))), 0], node))
                        self.add(Op(hx.BINARY, neg, NONE, self.tid(out),
                                    [hx.BIN_ADD, 1, self.pool.emit(_pad32(vec)), 0], node))
                        return None
                return self.add(Op(hx.BINARY, self.tid(tensor), NONE, self.tid(out),
                                   [opcode, 1, self.pool.emit(_pad32(vec)), rev], node))
            # Full-size constant operand (mode 2)
            c_dtype = np.int32 if (is_int or (is_comparison and const_dtype in (np.int32, np.int64))) else (
                np.bool_ if is_bool else np.float32
            )
            try:
                c_arr = np.broadcast_to(np.asarray(const, dtype=c_dtype), self.dims[out])
            except ValueError:
                self.fail(node, f"cannot broadcast constant of shape {const.shape} to {self.dims[out]}")
            h, w, c = self.hwc(out)
            stride = (c + 31) & ~31
            if len(self.dims[out]) == 4:
                hwc_arr = c_arr[0].transpose(1, 2, 0)
            elif len(self.dims[out]) == 3:
                hwc_arr = c_arr[0].transpose(1, 0).reshape(1, w, c)
            else:
                hwc_arr = c_arr.reshape(h, w, c)
            if stride > c:
                hwc_arr = np.pad(hwc_arr, ((0, 0), (0, 0), (0, stride - c)))
            offset = self.pool.emit(hwc_arr.reshape(-1))
            self.add(Op(hx.BINARY, self.tid(tensor), NONE, self.tid(out),
                        [opcode, 2, offset, rev], node))
            return None
        if self.canon(a_name) in self.heads or self.canon(b_name) in self.heads:
            self.fail(node, "elementwise operations on per-head attention tensors are unsupported")
        sa, sb, so = self.dims[a_name], self.dims[b_name], self.dims[out]
        if len(sa) != len(sb):
            self.fail(node, "operands of different rank")

        def fits(small, big):
            return len(small) == len(big) and small[0] == 1 and all(
                s in (1, o) for s, o in zip(small, big)
            )

        non_commutative = (
            hx.BIN_SUB, hx.BIN_DIV, hx.BIN_POW,
            hx.BIN_GREATER, hx.BIN_LESS, hx.BIN_GE, hx.BIN_LE
        )
        if sa == so and fits(sb, so):
            x, y, rev = a_name, b_name, 0
        elif sb == so and fits(sa, so):
            x, y, rev = b_name, a_name, int(opcode in non_commutative)
        else:
            self.fail(node, "operands broadcast against each other")

        da, db = self.get_dtype(a_name), self.get_dtype(b_name)
        is_int = (da == hx.DTYPE_INT32 or db == hx.DTYPE_INT32)
        is_bool = (da == hx.DTYPE_BOOL or db == hx.DTYPE_BOOL)
        if is_comparison:
            self.set_dtype(out, hx.DTYPE_BOOL)
        elif is_int:
            self.set_dtype(out, hx.DTYPE_INT32)
        elif is_bool:
            self.set_dtype(out, hx.DTYPE_BOOL)
        else:
            self.set_dtype(out, hx.DTYPE_FLOAT32)

        if opcode == hx.BIN_ADD and self.dims[x] == self.dims[y] and not is_int:
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
        source = node.input[0]
        d0 = self.const(node.input[1])
        if (d0 is not None and self.const(node.input[0]) is None and np.asarray(d0).size == 1 and
                (self._is_token(source) or self.canon(source) in self.heads)):
            return self._scale(node, source, node.output[0],
                               1.0 / float(np.asarray(d0, np.float64).reshape(())))
        ca, cb = self.const(node.input[0]), self.const(node.input[1])
        if ca is not None and cb is not None:
            self.fail(node, "both operands are constants")
        if cb is not None and ca is None:
            try:
                recip = 1.0 / np.asarray(cb, dtype=np.float64)
                vec = self.channel_vector(node, recip, node.output[0])
                if vec is not None and self.dims[node.input[0]] == self.dims[node.output[0]]:
                    return self._affine(node, node.input[0], node.output[0], vec, np.zeros_like(vec))
            except (ZeroDivisionError, FloatingPointError):
                pass
        self._binary(node, hx.BIN_DIV)

    def op_Equal(self, node):
        self._binary(node, hx.BIN_EQUAL)

    def op_Greater(self, node):
        self._binary(node, hx.BIN_GREATER)

    def op_Less(self, node):
        self._binary(node, hx.BIN_LESS)

    def op_GreaterOrEqual(self, node):
        self._binary(node, hx.BIN_GE)

    def op_LessOrEqual(self, node):
        self._binary(node, hx.BIN_LE)

    def op_And(self, node):
        self._binary(node, hx.BIN_AND)

    def op_Or(self, node):
        self._binary(node, hx.BIN_OR)

    def op_Not(self, node):
        self.set_dtype(node.output[0], hx.DTYPE_BOOL)
        self._emit_act(node, node.input[0], hx.FN_NOT)

    def op_Min(self, node):
        self._binary(node, hx.BIN_MIN)

    def op_Max(self, node):
        self._binary(node, hx.BIN_MAX)

    def op_Pow(self, node):
        exp = self.const(node.input[1])
        if exp is not None and np.allclose(exp, 2.0):
            x = node.input[0]
            out = node.output[0]
            self.add(Op(hx.BINARY, self.tid(x), self.tid(x), self.tid(out), [hx.BIN_MUL, 0, 0, 0], node))
            return
        self._binary(node, hx.BIN_POW)

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

    def op_Silu(self, node):
        self._emit_act(node, node.input[0], hx.FN_SILU)

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

    def op_Log(self, node):
        self._emit_act(node, node.input[0], hx.FN_LOG)

    def op_Reciprocal(self, node):
        self._emit_act(node, node.input[0], hx.FN_RECIPROCAL)

    def op_Floor(self, node):
        self._emit_act(node, node.input[0], hx.FN_FLOOR)

    def op_Ceil(self, node):
        self._emit_act(node, node.input[0], hx.FN_CEIL)

    def op_Sin(self, node):
        self._emit_act(node, node.input[0], hx.FN_SIN)

    def op_Cos(self, node):
        self._emit_act(node, node.input[0], hx.FN_COS)

    def op_Not(self, node):
        self._emit_act(node, node.input[0], hx.FN_NOT)

    def op_Round(self, node):
        self._emit_act(node, node.input[0], hx.FN_ROUND)

    def op_Sigmoid(self, node):
        self.add(Op(hx.LOGISTIC, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                    node))

    def op_Softmax(self, node):
        x, out = node.input[0], node.output[0]
        if self.const(x) is not None:
            self.fail(node, "constant inputs are unsupported")
        dims = self.dims.get(x)
        if dims is None:
            self.fail(node, "only static float32 inputs are supported")
        rank = len(dims)
        axis = int(self.attrs(node).get("axis", -1))
        axis = axis + rank if axis < 0 else axis

        if self.canon(x) in self.heads:  # per-head [L, L] scores: a row softmax per head
            if axis != rank - 1:
                self.fail(node, "per-head Softmax must be over the last axis (-1)")
            new = []
            for m in self.heads[self.canon(x)]:
                if m.trans:
                    self.fail(node, "a transposed per-head tensor cannot be the softmax input")
                t = self.temp(out, [m.rows, 1, m.cols])
                op = self.add(Op(hx.SOFTMAX, m.id, NONE, t, [], node))
                new.append(Mat(t, m.rows, m.cols, False, op))
            self.heads[out] = new
            return

        causal = 1 if (x in self.causal_masked or self.canon(x) in self.causal_masked) else 0
        if self._is_multihead(x) or (rank == 4 and dims[0] == 1 and self.canon(x) in self.multihead):
            axis_map = {1: 0, 2: 1, 3: 2}
            if axis not in axis_map:
                self.fail(node, f"unsupported softmax axis {axis} for multihead shape {dims}")
            hwc_axis = axis_map[axis]
            self.multihead.add(self.canon(out))
            self.add(Op(ATTN_SOFTMAX, self.tid(x), NONE, self.tid(out), [hwc_axis, causal, 0], node))
            return

        if self._is_token(x):  # HX_SOFTMAX is per pixel over channels: per row of [L, K]
            axis_map = {1: 0, 2: 2}
            if axis not in axis_map:
                self.fail(node, f"unsupported softmax axis {axis} for token shape {dims}")
            hwc_axis = axis_map[axis]
            self.token.add(self.canon(out))
            if hwc_axis == 2:
                self.add(Op(hx.SOFTMAX, self.tid(x), NONE, self.tid(out), [], node))
            else:
                self.add(Op(ATTN_SOFTMAX, self.tid(x), NONE, self.tid(out), [hwc_axis, 0, 0], node))
            return

        if rank == 4 and dims[0] == 1:
            if axis != 1:
                self.fail(node, f"only softmax over the channel axis (axis 1) is supported; "
                                f"got axis {axis} for rank {rank} (set axis=1 explicitly)")
            self.add(Op(hx.SOFTMAX, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                        node))
            return

        if rank == 2 and dims[0] == 1:
            if axis != 1:
                self.fail(node, f"unsupported softmax axis {axis} for rank 2 shape {dims}")
            self.add(Op(hx.SOFTMAX, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                        node))
            return

        if rank in (5, 6) and dims[0] == 1:
            if axis == rank - 1:
                self.add(Op(hx.SOFTMAX, self.tid(node.input[0]), NONE, self.tid(node.output[0]), [],
                            node))
            elif axis == rank - 2:
                self.add(Op(ATTN_SOFTMAX, self.tid(node.input[0]), NONE, self.tid(node.output[0]),
                            [1, 0, 0], node))
            elif axis == rank - 3:
                self.add(Op(ATTN_SOFTMAX, self.tid(node.input[0]), NONE, self.tid(node.output[0]),
                            [0, 0, 0], node))
            else:
                self.fail(node, f"unsupported softmax axis {axis} for rank {rank}")
            return

        self.fail(node, f"unsupported input shape {dims} or axis {axis}")

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
        """Instance normalisation: per channel over the spatial axes of a [1, C, H, W] or [1, C, L] input,
        which is HX_NORM with one group per channel."""
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) not in (3, 4):
            self.fail(node, "only batch-one rank-3 or rank-4 float32 inputs are supported")
        c = dims[1]
        scale = self._optional_constant(node, 1, "scale")
        bias = self._optional_constant(node, 2, "bias")
        eps = float(self.attrs(node).get("epsilon", 1e-5))
        gamma = np.ones(c, np.float32) if scale is None else self._norm_vector(node, scale, c, "scale")
        beta = np.zeros(c, np.float32) if bias is None else self._norm_vector(node, bias, c, "bias")
        self._norm(node, node.input[0], node.output[0], hx.NORM_GROUP, c, eps, gamma, beta)

    def op_LayerNormalization(self, node):
        """LayerNorm over the channel axis of a [1, C] tensor (HX_NORM, channel mode), or over
        (C, H, W) of a [1, C, H, W] tensor with axis 1 (HX_NORM, one group). Scale and bias must
        be constant per channel (spatially constant for rank 4). Mean and InvStdDev are unused."""
        if any(self.users.get(name, 0) for name in node.output[1:]):
            self.fail(node, "the Mean and InvStdDev outputs are unsupported")
        if self._is_token(node.input[0]):  # [1, L, K] normalised over K: per row, channel mode
            if int(self.attrs(node).get("axis", -1)) % 3 != 2:
                self.fail(node, "token LayerNormalization must normalise the last axis")
            c = self.dims[node.input[0]][2]
            eps = float(self.attrs(node).get("epsilon", 1e-5))
            scale = self._optional_constant(node, 1, "scale")
            bias = self._optional_constant(node, 2, "bias")
            gamma = (np.ones(c, np.float32) if scale is None
                     else self._norm_vector(node, scale, c, "scale"))
            beta = (np.zeros(c, np.float32) if bias is None
                    else self._norm_vector(node, bias, c, "bias"))
            self._norm(node, node.input[0], node.output[0], hx.NORM_CHANNEL, 0, eps, gamma, beta)
            return
        dims = self.dims.get(node.input[0])
        if dims is None or len(dims) not in (2, 3, 4):
            self.fail(node, "only batch-one rank-2, rank-3 token, or rank-4 float32 inputs are supported")
        if len(dims) == 3:
            if dims[0] != 1 or int(self.attrs(node).get("axis", -1)) % 3 != 2:
                self.fail(node, "token LayerNormalization must normalise the last axis")
            self.token.add(self.canon(node.input[0]))
            self.token.add(self.canon(node.output[0]))
            c = dims[2]
            eps = float(self.attrs(node).get("epsilon", 1e-5))
            scale = self._optional_constant(node, 1, "scale")
            bias = self._optional_constant(node, 2, "bias")
            gamma = (np.ones(c, np.float32) if scale is None
                     else self._norm_vector(node, scale, c, "scale"))
            beta = (np.zeros(c, np.float32) if bias is None
                    else self._norm_vector(node, bias, c, "bias"))
            self._norm(node, node.input[0], node.output[0], hx.NORM_CHANNEL, 0, eps, gamma, beta)
            return
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

    def op_RMSNormalization(self, node):
        """RMSNorm over the last axis of a token tensor or [1, C] (HX_RMSNORM).
        Scale (gamma) must be constant (or optional None). Optional bias (beta) defaults to None."""
        x = node.input[0]
        out = node.output[0]
        if out not in self.dims and x in self.dims:
            self.dims[out] = list(self.dims[x])
        dims = self.dims.get(x)
        if dims is None:
            self.fail(node, "unknown shape")
        if len(dims) == 3 and dims[0] == 1:
            self.token.add(self.canon(x))
            self.token.add(self.canon(out))
            c = dims[2]
        elif len(dims) == 2 and dims[0] == 1:
            c = dims[1]
        elif len(dims) == 2:
            self.token.add(self.canon(x))
            self.token.add(self.canon(out))
            c = dims[1]
        else:
            c = dims[-1]
        eps = float(self.attrs(node).get("epsilon", 1e-5))
        scale = self._optional_constant(node, 1, "scale")
        gamma_off = self.pool.emit(_pad32(self._norm_vector(node, scale, c, "scale"))) if scale is not None else NONE
        beta = None
        if len(node.input) > 2 and node.input[2]:
            b_val = self._optional_constant(node, 2, "bias")
            if b_val is not None:
                beta = self._norm_vector(node, b_val, c, "bias")
        beta_off = self.pool.emit(_pad32(beta)) if beta is not None else NONE
        p = [hx.f32_bits(eps), gamma_off, beta_off] + [0] * 8
        self.add(Op(hx.RMSNORM, self.tid(x), NONE, self.tid(out), p, node))

    op_RMSNorm = op_RMSNormalization
    op_SimplifiedLayerNormalization = op_RMSNormalization

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
        dil = self.dilation(node, a)
        s, pt, pl = self.spatial(node, a, node.input[0], node.output[0], k, dilated=True)
        p = [k[0], k[1], s[0], s[1], pt, pl]
        if kind == hx.AVGPOOL:
            p.append(int(a.get("count_include_pad", 0)))
        op = Op(kind, self.tid(node.input[0]), NONE, self.tid(node.output[0]), p, node)
        if dil != (1, 1):
            op.dilation = dil
        self.add(op)

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
        token = self._is_token(x)
        if dims is None or (len(dims) not in (2, 4) and not token):
            self.fail(node, "only static rank-2 or rank-4 float32 inputs are supported")
        if self.const(x) is not None:
            self.fail(node, "constant inputs are unsupported")
        if not axes:
            self.fail(node, "empty axes (reduce everything) are unsupported")
        rank = len(dims)
        if token:  # [1, L, K]: the sequence is h (bit 1), the channels are c (bit 4)
            bits = {1: 1, 2: 4}
        else:
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
        dims_out = self.dims.get(out) or []
        has_legacy_split = any(n.op_type == "Transpose" and [int(v) for v in self.attrs(n).get("perm", [])] == [0, 2, 3, 1] for n in self.nodes)
        if self._is_token(x) and len(dims_out) == 4:  # [1, L, K] -> [1, L, H, dh]: a head view
            rows, _, width = self.hwc(x)
            if dims_out[:2] != [1, rows] or dims_out[2] * dims_out[3] != width:
                self.fail(node, "a token tensor may only be reshaped to [1, L, H, dh] with "
                                "H * dh = K")
            split, cx = (dims_out[2], dims_out[3]), self.canon(x)
            if has_legacy_split:
                if self.split_of.get(cx, split) != split:
                    self.fail(node, "the same tensor is split into two different head layouts")
                self.split_of[cx] = split
                self.alias[out] = cx
                return
            else:
                self.head_view.add(self.canon(out))
                self.add(Op(hx.RESHAPE, self.tid(x), NONE, self.tid(out), [0], node))
                return
        if self._is_head_view(x) and len(dims_out) == 3:
            self.token.add(self.canon(out))
            self.add(Op(hx.RESHAPE, self.tid(x), NONE, self.tid(out), [0], node))
            return
        src, dst = self.hwc(x), self.hwc(out)
        if src == dst:
            self.alias[out] = self.canon(x)
            return
        if self._is_token(x) and self._is_token(out):
            if src[0] * src[1] * src[2] != dst[0] * dst[1] * dst[2]:
                self.fail(node, "the input and output element counts differ")
            self.add(Op(hx.RESHAPE, self.tid(x), NONE, self.tid(out), [0], node))
            return
        if (self._is_token(x) or self._is_token(out)) and not (self._is_token(x) and self._is_token(out)):
            if src[0] * src[1] == dst[0] * dst[1] and src[2] == dst[2]:
                self.alias[out] = self.canon(x)
                return
            if src[0] * src[1] * src[2] == dst[0] * dst[1] * dst[2]:
                self.add(Op(hx.RESHAPE, self.tid(x), NONE, self.tid(out), [0], node))
                return
            self.fail(node, "a token tensor reshape requires matching elements")
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

    def op_Transpose(self, node):
        if not self._transpose_view(node):
            self._transpose_nchw(node)

    def _transpose_view(self, node):
        """Transposes of token and per-head tensors. Returns False for an NCHW transpose of a
        rank-4 or rank-3 tensor that is not in token layout (the caller handles it)."""
        x, out = node.input[0], node.output[0]
        cx = self.canon(x)
        dims_x = self.dims.get(x)
        perm = [int(v) for v in self.attrs(node).get("perm", [])]
        if cx in self.ttrans:
            self.fail(node, "a transposed token tensor cannot be transposed again")
        if cx in self.heads:
            self._heads_transpose(node, cx, perm)
            return True
        has_legacy_split = any(n.op_type == "Transpose" and [int(v) for v in self.attrs(n).get("perm", [])] == [0, 2, 3, 1] for n in self.nodes)
        if dims_x is not None and len(dims_x) == 4 and cx in self.split_of:
            if has_legacy_split:
                self._split_transpose(node, cx, perm)
                return True
        if dims_x is not None and len(dims_x) == 4 and dims_x[0] == 1:
            if perm == [0, 2, 1, 3]:
                if self._is_head_view(x) or cx in self.split_of:
                    self.multihead.add(self.canon(out))
                    self.add(Op(hx.TRANSPOSE, self.tid(x), NONE, self.tid(out), [1, 0, 2], node))
                    return True
                elif self._is_multihead(x):
                    self.head_view.add(self.canon(out))
                    self.add(Op(hx.TRANSPOSE, self.tid(x), NONE, self.tid(out), [1, 0, 2], node))
                    return True
            elif perm == [0, 1, 3, 2] and self._is_multihead(x):
                self.multihead.add(self.canon(out))
                self.add(Op(hx.TRANSPOSE, self.tid(x), NONE, self.tid(out), [0, 2, 1], node))
                return True
        if dims_x is not None and len(dims_x) in (5, 6) and dims_x[0] == 1:
            if out not in self.dims and len(perm) == len(dims_x):
                self.dims[out] = [dims_x[p] for p in perm]
            src, dst = self.hwc(x), self.hwc(out)
            if src == dst or (src[0] * src[1] == dst[0] * dst[1] and src[2] == dst[2]):
                self.alias[out] = cx
                return True
        if dims_x is not None and len(dims_x) == 3:
            if perm == [1, 0, 2]:
                self.fail(node, "sequence-first [L, 1, K] tensors are unsupported (torch's "
                                "nn.MultiheadAttention works this way); export a batch-first block")
            if perm != [0, 2, 1]:
                self.fail(node, "rank-3 transposes must be [0, 2, 1]")
            if out in self.outputs:
                self.fail(node, "a transposed view cannot be a graph output")
            if self._is_token(x):  # K^T of a token tensor: a logical view, used by MatMul only
                self.ttrans[out] = cx
                return True
            if not self._is_token(out):
                self.fail(node, "a rank-3 transpose is supported only into token layout")
            # [1, C, L] (NCHW with H = 1) and [1, L, C] share one HWC order: a logical reshape.
            self.add(Op(hx.RESHAPE, self.tid(x), NONE, self.tid(out), [0], node))
            return True
        return False

    def _split_transpose(self, node, cx, perm):
        """[1, L, H, dh] head view -> per-head matrices: [0, 2, 1, 3] gives [L, dh] heads, and
        [0, 2, 3, 1] gives the transposed heads [dh, L] (a logical view of the same slices)."""
        heads, dh = self.split_of[cx]
        rows = self.hwc(cx)[0]
        out = node.output[0]
        if perm == [0, 2, 1, 3]:
            trans, expect = False, [1, heads, rows, dh]
        elif perm == [0, 2, 3, 1]:
            trans, expect = True, [1, heads, dh, rows]
        else:
            self.fail(node, "a head view transposes as [0, 2, 1, 3] or [0, 2, 3, 1]")
        if list(self.dims[out]) != expect:
            self.fail(node, "output shape disagrees with the head split")
        base = self.tid(cx)
        mats = []
        for h in range(heads):
            piece = self.temp(out, [rows, 1, dh])
            op = self.add(Op(hx.SLICE, base, NONE, piece, [0, 0, h * dh], node))
            mats.append(Mat(piece, rows, dh, trans, op))
        self.heads[out] = mats

    def _heads_transpose(self, node, cx, perm):
        """Per-head matrices: [0, 1, 3, 2] flips each matrix (K^T); [0, 2, 1, 3] merges the heads
        back into channels, one token block per head, as a [1, L, H, dh] head view."""
        mats, out = self.heads[cx], node.output[0]
        if perm == [0, 1, 3, 2]:
            self.heads[out] = [m._replace(trans=not m.trans) for m in mats]
            return
        if perm != [0, 2, 1, 3] or any(m.trans for m in mats):
            self.fail(node, "per-head tensors transpose as [0, 1, 3, 2] or, for untransposed "
                            "heads, [0, 2, 1, 3]")
        heads, rows, dh = len(mats), mats[0].rows, mats[0].cols
        if list(self.dims[out]) != [1, rows, heads, dh]:
            self.fail(node, "output shape disagrees with the merged heads")
        self.split_of[out] = (heads, dh)
        if heads == 1:
            self.ids[out] = mats[0].id
            return
        acc, width = mats[0].id, dh
        for i, m in enumerate(mats[1:], 1):
            width += dh
            merged = self.tid(out) if i == heads - 1 else self.temp(out, [rows, 1, width])
            self.add(Op(hx.CONCAT, acc, m.id, merged, [2], node))
            acc = merged

    def _transpose_nchw(self, node):
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
        if self._is_token(node.output[0]):  # sequence (1) joins rows, channels (2) join c
            axis = int(self.attrs(node)["axis"]) % 3
            if axis not in (1, 2):
                self.fail(node, "token Concat must join the sequence (1) or channel (2) axis")
            hwc_axis = 0 if axis == 1 else 2
        else:
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

    def _gather_sequence(self, node):
        """Gather along the sequence axis of a token tensor [1, L, K] with constant indices that
        form a contiguous run: a scalar index gives [1, K], a run of m gives [1, m, K] (HX_SLICE)."""
        data, index, out = node.input[0], node.input[1], node.output[0]
        if not self._is_token(data) or len(self.dims[data]) != 3:
            self.fail(node, "only Gather over the sequence axis of a token tensor [1, L, K] is "
                            "supported")
        if int(self.attrs(node).get("axis", 0)) % 3 != 1:
            self.fail(node, "only the sequence axis (1) is supported")
        value = self.const(index)
        if value is None:
            self.fail(node, "indices must be constant")
        idx = np.asarray(value).astype(np.int64)
        if idx.ndim > 1:
            self.fail(node, "indices must be a scalar or a vector")
        rows, _, width = self.hwc(data)
        idx = idx.reshape(-1)
        idx = np.where(idx < 0, idx + rows, idx)
        if idx.size == 0 or idx.min() < 0 or idx.max() >= rows or np.any(np.diff(idx) != 1):
            self.fail(node, "indices must be a contiguous run of sequence positions")
        if self.hwc(out) != [idx.size, 1, width]:
            self.fail(node, "output shape disagrees with the indices")
        self.add(Op(hx.SLICE, self.tid(data), NONE, self.tid(out), [int(idx[0]), 0, 0], node))

    def op_Slice(self, node):
        """Slice with constant starts/ends (opset 10+) and unit steps. Channel and spatial
        axes become SLICE offsets; any other axis must be taken whole."""
        if len(node.input) < 3 or not node.input[1] or not node.input[2]:
            self.fail(node, "starts and ends must be inputs (opset 10 or later)")
        data = node.input[0]
        dims = self.dims.get(data)
        if dims is None or len(dims) not in (2, 3, 4):
            self.fail(node, "only static rank-2, rank-3 or rank-4 tensors are supported")
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
        if rank == 4:
            hwc_axis = {1: 2, 2: 0, 3: 1}
        elif rank == 3:
            hwc_axis = {1: 0, 2: 2} if self._is_token(data) else {1: 2}
        else:
            hwc_axis = {1: 2}

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
        if dims is None or len(dims) not in (2, 3, 4):
            self.fail(node, "only static rank-2, rank-3 or rank-4 tensors are supported")
        is_token = self._is_token(data) or (len(dims) == 3 and int(a.get("axis", 0)) % 3 == 2)
        exp_axis = 2 if is_token else 1
        if int(a.get("axis", 0)) % len(dims) != exp_axis:
            self.fail(node, f"only splits along the channel axis ({exp_axis}) are supported")
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
        if mode not in ("linear", "nearest", "cubic"):
            self.fail(node, "only linear, nearest, and cubic resize are supported")
        if a.get("antialias", 0):
            self.fail(node, "antialias is unsupported")
        coord = text("coordinate_transformation_mode", "half_pixel")
        out_name = node.output[0]
        if out_name not in self.dims:
            sizes = self.const(node.input[3]) if len(node.input) > 3 and node.input[3] else None
            scales = self.const(node.input[2]) if len(node.input) > 2 and node.input[2] else (
                self.const(node.input[1]) if len(node.input) > 1 and node.input[1] and not a.get("roi") else None
            )
            if sizes is None and "sizes" in a:
                sizes = a["sizes"]
            if scales is None and "scales" in a:
                scales = a["scales"]
            in_shape = self.dims[node.input[0]]
            if sizes is not None:
                self.dims[out_name] = [int(v) for v in sizes]
            elif scales is not None:
                self.dims[out_name] = [int(np.floor(in_shape[i] * float(scales[i]))) for i in range(len(in_shape))]
            else:
                self.fail(node, "output shape or scales/sizes must be specified")
        ih, iw, ic = self.hwc(node.input[0])
        oh, ow, oc = self.hwc(node.output[0])
        if ic != oc:
            self.fail(node, "channels must not change")
        coord_map = {
            "asymmetric": 0,
            "align_corners": 1,
            "half_pixel": 2,
            "pytorch_half_pixel": 3,
        }
        if coord not in coord_map:
            self.fail(node, f"coordinate transformation mode {coord} is unsupported")
        coord_code = coord_map[coord]
        if mode == "nearest":
            near_map = {
                "floor": 0,
                "round_prefer_floor": 1,
                "round_prefer_ceil": 2,
                "ceil": 3,
            }
            rounding = text("nearest_mode", "round_prefer_floor")
            if rounding not in near_map:
                self.fail(node, f"nearest mode {rounding} is unsupported")
            near_code = near_map[rounding]
            flags = [0, 0, 1, coord_code, near_code, 0]
        elif mode == "cubic":
            flags = [0, 0, 0, coord_code, 0, 1]
        else:  # linear
            flags = [1 if coord == "align_corners" else 0,
                     1 if coord in ("half_pixel", "pytorch_half_pixel") else 0,
                     0, coord_code, 0, 0]
        self.add(Op(hx.RESIZE, self.tid(node.input[0]), NONE, self.tid(node.output[0]), flags,
                    node))

    def _where_operand(self, name, out):
        c = self.const(name)
        if c is None:
            return 0, 0, self.tid(name)
        out_dtype = self.get_dtype(out)
        is_int_or_bool = out_dtype in (hx.DTYPE_INT32, hx.DTYPE_BOOL)
        arr = np.asarray(c, dtype=np.int32 if is_int_or_bool else np.float32)
        if arr.size == 1:
            val_bits = int(arr.reshape(-1)[0]) & 0xFFFFFFFF if is_int_or_bool else hx.f32_bits(float(arr.reshape(-1)[0]))
            return 1, val_bits, NONE
        vec = self.channel_vector(None, arr, out)
        if vec is not None:
            return 2, self.pool.emit(_pad32(vec)), NONE
        h, w, ch = self.hwc(out)
        stride = (ch + 31) & ~31
        barr = np.broadcast_to(arr, self.dims[out])
        if len(self.dims[out]) == 4:
            barr = barr[0].transpose(1, 2, 0)
        elif len(self.dims[out]) == 3:
            barr = barr[0].transpose(1, 0).reshape(1, w, ch)
        else:
            barr = barr.reshape(h, w, ch)
        if stride > ch:
            barr = np.pad(barr, ((0, 0), (0, 0), (0, stride - ch)))
        return 3, self.pool.emit(barr.reshape(-1)), NONE

    def op_Where(self, node):
        cond, x, y = node.input
        out = node.output[0]
        out_elem = self.types.get(out)
        if out_elem in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            out_dtype = hx.DTYPE_INT32
        elif out_elem == onnx.TensorProto.BOOL:
            out_dtype = hx.DTYPE_BOOL
        elif out_elem == onnx.TensorProto.FLOAT:
            out_dtype = hx.DTYPE_FLOAT32
        else:
            x_dt = self.get_dtype(x) if self.const(x) is None else (
                hx.DTYPE_INT32 if np.asarray(self.const(x)).dtype in (np.int32, np.int64)
                else (hx.DTYPE_BOOL if np.asarray(self.const(x)).dtype == np.bool_ else hx.DTYPE_FLOAT32)
            )
            y_dt = self.get_dtype(y) if self.const(y) is None else (
                hx.DTYPE_INT32 if np.asarray(self.const(y)).dtype in (np.int32, np.int64)
                else (hx.DTYPE_BOOL if np.asarray(self.const(y)).dtype == np.bool_ else hx.DTYPE_FLOAT32)
            )
            out_dtype = x_dt if x_dt != hx.DTYPE_FLOAT32 else y_dt
        self.set_dtype(out, out_dtype)
        x_mode, x_val, y_tid = self._where_operand(x, out)
        y_mode, y_val, z_tid = self._where_operand(y, out)
        op = Op(hx.WHERE, self.tid(cond), y_tid, self.tid(out),
                [x_mode, x_val, y_mode, y_val], node)
        op.z_id = z_tid
        self.add(op)

    def op_Gather(self, node):
        data = node.input[0]
        if (data in self.dims and len(self.dims[data]) == 3 and self._is_token(data)
                and int(self.attrs(node).get("axis", 0)) % 3 == 1):
            return self._gather_sequence(node)
        return self._gather_embedding(node)

    def _gather_embedding(self, node):
        a = self.attrs(node)
        data, indices = node.input[0], node.input[1]
        out = node.output[0]
        axis = int(a.get("axis", 0))
        table_const = self.const(data)
        indices_const = self.const(indices)
        d_shape = list(table_const.shape if table_const is not None else (self.dims.get(data) or ()))
        if axis < 0 and d_shape:
            axis += len(d_shape)
        vocab = int(d_shape[axis]) if d_shape and len(d_shape) > axis else 1
        outer_dim = int(np.prod(d_shape[:axis])) if axis > 0 and d_shape else 1
        inner_dim = int(np.prod(d_shape[axis + 1:])) if d_shape and axis + 1 < len(d_shape) else 1

        idx_type = self.types.get(indices)
        idx_dtype_code = 0 if idx_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64) else 1

        i_dims = self.dims.get(indices)
        if i_dims is None and indices_const is not None:
            i_dims = list(indices_const.shape)
        num_indices = int(np.prod(i_dims)) if i_dims is not None else 1

        stride = (inner_dim + 31) & ~31

        if out in self.dims:
            self.shapes[self.tid(out)] = self.hwc(out)
        elif axis == 0:
            self.shapes[self.tid(out)] = [1, num_indices, inner_dim]
        else:
            self.shapes[self.tid(out)] = [outer_dim, num_indices, inner_dim]

        if table_const is not None and indices_const is None:
            tbl = np.asarray(table_const, dtype=np.float32).reshape(outer_dim, vocab, inner_dim)
            if stride > inner_dim:
                tbl = np.pad(tbl, ((0, 0), (0, 0), (0, stride - inner_dim)))
            offset = self.pool.emit(tbl.reshape(-1))
            mode = 1
            x_tid = self.tid(indices)
            y_tid = NONE
            table_stride = stride
            const_idx_off = 0
            const_idx_num = 0
            if idx_dtype_code == 0:
                self.set_dtype(x_tid, hx.DTYPE_INT32)
            if x_tid not in self.shapes:
                if i_dims is not None and len(i_dims) == 3:
                    self.shapes[x_tid] = [int(i_dims[0]), int(i_dims[1]), int(i_dims[2])]
                elif i_dims is not None and len(i_dims) == 2:
                    self.shapes[x_tid] = [1, int(i_dims[0]), int(i_dims[1])]
                else:
                    self.shapes[x_tid] = [1, num_indices, 1]
        elif indices_const is not None:
            idx_arr = np.asarray(indices_const, dtype=np.int32).reshape(-1)
            const_idx_off = self.pool.emit(idx_arr, "<i4")
            const_idx_num = idx_arr.size
            mode = 2
            x_tid = self.tid(data)
            y_tid = NONE
            offset = 0
            table_stride = 0
            idx_dtype_code = 0
        else:
            mode = 0
            offset = 0
            x_tid = self.tid(indices)
            y_tid = self.tid(data)
            table_stride = 0
            const_idx_off = 0
            const_idx_num = 0
            if idx_dtype_code == 0:
                self.set_dtype(x_tid, hx.DTYPE_INT32)
            if x_tid not in self.shapes:
                if i_dims is not None and len(i_dims) == 3:
                    self.shapes[x_tid] = [int(i_dims[0]), int(i_dims[1]), int(i_dims[2])]
                elif i_dims is not None and len(i_dims) == 2:
                    self.shapes[x_tid] = [1, int(i_dims[0]), int(i_dims[1])]
                else:
                    self.shapes[x_tid] = [1, num_indices, 1]

        self.add(Op(hx.GATHER, x_tid, y_tid, self.tid(out),
                    [mode, offset, vocab, idx_dtype_code, axis, inner_dim, outer_dim, table_stride,
                     const_idx_off, const_idx_num], node))

    def op_GatherElements(self, node):
        a = self.attrs(node)
        data, indices = node.input[0], node.input[1]
        out = node.output[0]
        axis = int(a.get("axis", 0))
        d_shape = self.dims.get(data)
        if d_shape is not None and axis < 0:
            axis += len(d_shape)
        d_rank = len(d_shape) if d_shape is not None else 3
        if d_rank == 2:
            hwc_axis = axis + 1
        elif d_rank == 1:
            hwc_axis = 2
        elif d_rank == 4:
            hwc_axis = max(0, axis - 1)
        else:
            hwc_axis = axis

        idx_type = self.types.get(indices)
        idx_dtype_code = 0 if idx_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64) else 1

        idx_const = self.const(indices)
        table_const = self.const(data)

        if idx_const is not None:
            mode = 2
            idx_arr = np.asarray(idx_const, dtype=np.int32)
            offset = self.pool.emit(idx_arr.reshape(-1), "<i4")
            y_tid = NONE
            idx_dtype_code = 0
        elif table_const is not None:
            mode = 1
            arr = np.asarray(table_const, dtype=np.float32)
            c = arr.shape[-1] if arr.ndim > 0 else 1
            stride = (c + 31) & ~31
            if stride > c:
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, stride - c)]
                arr = np.pad(arr, pad_width)
            offset = self.pool.emit(arr.reshape(-1))
            y_tid = self.tid(indices)
            if idx_dtype_code == 0:
                self.set_dtype(y_tid, hx.DTYPE_INT32)
        else:
            mode = 0
            offset = 0
            y_tid = self.tid(indices)
            if idx_dtype_code == 0:
                self.set_dtype(y_tid, hx.DTYPE_INT32)

        if out in self.dims:
            self.shapes[self.tid(out)] = self.hwc(out)
        else:
            out_dims = self.dims.get(indices, [1, 1, 1])
            self.shapes[self.tid(out)] = [1, int(np.prod(out_dims[:-1])), int(out_dims[-1])]
        if y_tid != NONE and y_tid not in self.shapes:
            self.shapes[y_tid] = self.shapes[self.tid(out)]

        self.add(Op(hx.GATHER_ELEMENTS, self.tid(data), y_tid, self.tid(out),
                    [hwc_axis, idx_dtype_code, mode, offset], node))

    def op_GatherND(self, node):
        a = self.attrs(node)
        data, indices = node.input[0], node.input[1]
        out = node.output[0]
        batch_dims = int(a.get("batch_dims", 0))
        d_shape = self.dims.get(data)
        i_shape = self.dims.get(indices)
        idx_const = self.const(indices)
        table_const = self.const(data)
        if i_shape is None and idx_const is not None:
            i_shape = list(idx_const.shape)
        tuple_len = int(i_shape[-1]) if i_shape is not None else 1
        slice_elements = int(np.prod(d_shape[tuple_len:])) if d_shape is not None and len(d_shape) > tuple_len else 1

        idx_type = self.types.get(indices)
        idx_dtype_code = 0 if idx_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64) else 1

        if idx_const is not None:
            mode = 2
            idx_arr = np.asarray(idx_const, dtype=np.int32)
            offset = self.pool.emit(idx_arr.reshape(-1), "<i4")
            y_tid = NONE
            idx_dtype_code = 0
        elif table_const is not None:
            mode = 1
            arr = np.asarray(table_const, dtype=np.float32)
            c = arr.shape[-1] if arr.ndim > 0 else 1
            stride = (c + 31) & ~31
            if stride > c:
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, stride - c)]
                arr = np.pad(arr, pad_width)
            offset = self.pool.emit(arr.reshape(-1))
            y_tid = self.tid(indices)
            if idx_dtype_code == 0:
                self.set_dtype(y_tid, hx.DTYPE_INT32)
        else:
            mode = 0
            offset = 0
            y_tid = self.tid(indices)
            if idx_dtype_code == 0:
                self.set_dtype(y_tid, hx.DTYPE_INT32)

        if out in self.dims:
            self.shapes[self.tid(out)] = self.hwc(out)
        else:
            self.shapes[self.tid(out)] = [1, int(np.prod(i_shape[:-1])), slice_elements]

        if y_tid != NONE and y_tid not in self.shapes:
            num_idx = int(np.prod(i_shape[:-1])) if i_shape is not None else 1
            self.shapes[y_tid] = [1, num_idx, tuple_len]

        self.add(Op(hx.GATHER_ND, self.tid(data), y_tid, self.tid(out),
                    [batch_dims, idx_dtype_code, mode, offset, tuple_len, slice_elements], node))

    def op_ScatterND(self, node):
        a = self.attrs(node)
        data, indices, updates = node.input[0], node.input[1], node.input[2]
        out = node.output[0]
        reduction_val = a.get("reduction", b"none")
        if isinstance(reduction_val, bytes):
            reduction_str = reduction_val.decode("utf-8")
        else:
            reduction_str = str(reduction_val)
        reduction_map = {"none": 0, "add": 1, "mul": 2, "max": 3, "min": 4}
        reduction_code = reduction_map.get(reduction_str.lower(), 0)

        idx_const = self.const(indices)
        upd_const = self.const(updates)

        i_shape = self.dims.get(indices)
        if i_shape is None and idx_const is not None:
            i_shape = list(idx_const.shape)
        tuple_len = int(i_shape[-1]) if i_shape is not None else 1

        d_shape = self.dims.get(data)
        slice_elements = int(np.prod(d_shape[tuple_len:])) if d_shape is not None and len(d_shape) > tuple_len else 1

        idx_type = self.types.get(indices)
        idx_dtype_code = 0 if idx_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64) else 1

        mode_flags = 0
        p6_idx_off = 0
        p7_upd_off = 0
        num_idx = int(np.prod(i_shape[:-1])) if i_shape is not None else 1

        if idx_const is not None:
            mode_flags |= 1
            idx_arr = np.asarray(idx_const, dtype=np.int32)
            p6_idx_off = self.pool.emit(idx_arr.reshape(-1), "<i4")
            y_tid = NONE
            idx_dtype_code = 0
        else:
            y_tid = self.tid(indices)
            if idx_dtype_code == 0:
                self.set_dtype(y_tid, hx.DTYPE_INT32)

        if upd_const is not None:
            mode_flags |= 2
            upd_arr = np.asarray(upd_const, dtype=np.float32)
            p7_upd_off = self.pool.emit(upd_arr.reshape(-1))
            z_tid = NONE
        else:
            z_tid = self.tid(updates)

        if out in self.dims:
            self.shapes[self.tid(out)] = self.hwc(out)
        else:
            self.shapes[self.tid(out)] = self.hwc(data)

        if y_tid != NONE and y_tid not in self.shapes:
            self.shapes[y_tid] = [1, num_idx, tuple_len]
        if z_tid != NONE and z_tid not in self.shapes:
            if updates in self.dims:
                self.shapes[z_tid] = self.hwc(updates)
            else:
                u_shape = self.dims.get(updates)
                if u_shape is not None:
                    if len(u_shape) == 3:
                        self.shapes[z_tid] = [int(u_shape[0]), int(u_shape[1]), int(u_shape[2])]
                    elif len(u_shape) == 2:
                        self.shapes[z_tid] = [1, int(u_shape[0]), int(u_shape[1])]
                    elif len(u_shape) == 1:
                        self.shapes[z_tid] = [1, 1, int(u_shape[0])]
                    else:
                        self.shapes[z_tid] = [1, int(np.prod(u_shape[:-1])), int(u_shape[-1])]
                else:
                    self.shapes[z_tid] = [1, num_idx, slice_elements]

        op = Op(hx.SCATTER_ND, self.tid(data), y_tid, self.tid(out),
                [0, idx_dtype_code, tuple_len, slice_elements, reduction_code, mode_flags, p6_idx_off, p7_upd_off, num_idx],
                node, z=z_tid)
        self.add(op)

    def op_ArgMax(self, node):
        a = self.attrs(node)
        x = node.input[0]
        out = node.output[0]
        h, w, c = self.hwc(x)
        self.shapes[self.tid(out)] = [h, w, 1]
        select_last = int(a.get("select_last_index", 0))
        self.set_dtype(out, hx.DTYPE_INT32)
        self.add(Op(hx.ARGMAX, self.tid(x), NONE, self.tid(out), [select_last, 1], node))

    def op_ArgMin(self, node):
        a = self.attrs(node)
        x = node.input[0]
        out = node.output[0]
        h, w, c = self.hwc(x)
        self.shapes[self.tid(out)] = [h, w, 1]
        select_last = int(a.get("select_last_index", 0))
        self.set_dtype(out, hx.DTYPE_INT32)
        self.add(Op(hx.ARGMIN, self.tid(x), NONE, self.tid(out), [select_last, 1], node))

    def op_TopK(self, node):
        a = self.attrs(node)
        x = node.input[0]
        k_val = self.const(node.input[1]) if len(node.input) > 1 and node.input[1] else a.get("k")
        if k_val is None:
            self.fail(node, "K must be constant")
        k = int(np.asarray(k_val).reshape(-1)[0])
        largest = int(a.get("largest", 1))
        sorted_flag = int(a.get("sorted", 1))
        h, w, c = self.hwc(x)
        values_out = node.output[0]
        self.shapes[self.tid(values_out)] = [h, w, k]
        indices_out = node.output[1] if len(node.output) > 1 and node.output[1] else None
        z_id = NONE
        idx_dtype = 0
        if indices_out:
            z_id = self.tid(indices_out)
            self.shapes[z_id] = [h, w, k]
            idx_elem = self.types.get(indices_out)
            if idx_elem in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                self.set_dtype(indices_out, hx.DTYPE_INT32)
                idx_dtype = 1
            else:
                self.set_dtype(indices_out, hx.DTYPE_FLOAT32)
                idx_dtype = 0
        op = Op(hx.TOPK, self.tid(x), NONE, self.tid(values_out),
                [k, largest, sorted_flag, idx_dtype], node)
        op.z_id = z_id
        self.add(op)

    def op_Tile(self, node):
        x = node.input[0]
        repeats = self.const(node.input[1])
        if repeats is None:
            self.fail(node, "repeats must be constant")
        out = node.output[0]
        hi, wi, ci = self.hwc(x)
        ho, wo, co = self.hwc(out)
        if ho % hi != 0 or wo % wi != 0 or co % ci != 0:
            self.fail(node, f"output shape {[ho, wo, co]} is not a tile multiple of {[hi, wi, ci]}")
        self.add(Op(hx.TILE, self.tid(x), NONE, self.tid(out), [], node))

    def op_Expand(self, node):
        x = node.input[0]
        out = node.output[0]
        if self.dims.get(x) == self.dims.get(out):
            self.alias[out] = self.canon(x)
            return
        hi, wi, ci = self.hwc(x)
        ho, wo, co = self.hwc(out)
        if ho % hi == 0 and wo % wi == 0 and co % ci == 0:
            self.add(Op(hx.TILE, self.tid(x), NONE, self.tid(out), [], node))
        else:
            self.fail(node, f"Expand from {[hi, wi, ci]} to {[ho, wo, co]} is not integer tile")

    def op_Cast(self, node):
        a = self.attrs(node)
        to_type = int(a.get("to", onnx.TensorProto.FLOAT))
        from_type = self.types.get(node.input[0], onnx.TensorProto.FLOAT)
        valid_types = (onnx.TensorProto.FLOAT, onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.BOOL)
        if to_type not in valid_types or from_type not in valid_types:
            self.fail(node, f"only casts between float32, int32 and bool are lowered (got {from_type} -> {to_type})")
        if to_type == onnx.TensorProto.INT64:
            to_type = onnx.TensorProto.INT32
        if from_type == onnx.TensorProto.INT64:
            from_type = onnx.TensorProto.INT32
        if to_type == from_type:
            self.alias[node.output[0]] = self.canon(node.input[0])
            self.set_dtype(node.output[0], self.get_dtype(node.input[0]))
            return
        out_name = node.output[0]
        inp_name = node.input[0]
        if from_type == onnx.TensorProto.FLOAT and to_type == onnx.TensorProto.INT32:
            mode = 0  # float -> int
            out_dtype = hx.DTYPE_INT32
        elif from_type == onnx.TensorProto.INT32 and to_type == onnx.TensorProto.FLOAT:
            mode = 1  # int -> float
            out_dtype = hx.DTYPE_FLOAT32
        elif from_type == onnx.TensorProto.BOOL and to_type == onnx.TensorProto.FLOAT:
            mode = 2  # bool -> float
            out_dtype = hx.DTYPE_FLOAT32
        elif from_type == onnx.TensorProto.FLOAT and to_type == onnx.TensorProto.BOOL:
            mode = 3  # float -> bool
            out_dtype = hx.DTYPE_BOOL
        elif from_type == onnx.TensorProto.INT32 and to_type == onnx.TensorProto.BOOL:
            mode = 4  # int32 -> bool
            out_dtype = hx.DTYPE_BOOL
        elif from_type == onnx.TensorProto.BOOL and to_type == onnx.TensorProto.INT32:
            mode = 5  # bool -> int32
            out_dtype = hx.DTYPE_INT32
        else:
            mode = 6  # identity / bitcast
            out_dtype = (
                hx.DTYPE_INT32 if to_type == onnx.TensorProto.INT32
                else (hx.DTYPE_BOOL if to_type == onnx.TensorProto.BOOL else hx.DTYPE_FLOAT32)
            )
        self.set_dtype(out_name, out_dtype)
        self.add(Op(hx.CAST, self.tid(inp_name), NONE, self.tid(out_name), [mode], node))

    # ------------------------------------------------------------ finishing
    def _split_deep(self, op):
        """A dense convolution whose K = kh * kw * cin > 4032 is split into slices of input channels
        joined by HX_ADD."""
        if op.kind != hx.CONV or op.weights is None:
            return [op]
        co, cin, kh, kw = op.weights.shape
        k = kh * kw * cin
        if k <= 4032:
            return [op]
        max_cin_slice = 4032 // (kh * kw)
        if max_cin_slice < 1:
            return [op]
        chunk = (max_cin_slice // 32) * 32 if max_cin_slice >= 32 else max_cin_slice
        ih, iw, _ = self.shapes[op.x]
        oh, ow, _ = self.shapes[op.out]
        out = []
        acc_id = None
        for lo in range(0, cin, chunk):
            hi = min(lo + chunk, cin)
            last = (hi == cin)
            part_in = self.temp("deep_slice", [ih, iw, hi - lo])
            out.append(Op(hx.SLICE, op.x, NONE, part_in, [0, 0, lo], op.node))
            w_slice = op.weights[:, lo:hi, :, :]
            b_slice = op.bias if lo == 0 else np.zeros_like(op.bias)
            part_out = self.temp("deep_conv", [oh, ow, co])
            p_slice = list(op.p)
            p_slice[8] = 0
            conv_piece = Op(hx.CONV, part_in, NONE, part_out, p_slice, op.node)
            conv_piece.dilation = op.dilation
            conv_piece.weights = w_slice
            conv_piece.bias = b_slice
            out.append(conv_piece)
            if acc_id is None:
                acc_id = part_out
                continue
            sum_out = op.out if last else self.temp("deep_sum", [oh, ow, co])
            add_act = op.p[8] if last else 0
            add_op = Op(hx.ADD, acc_id, part_out, sum_out, [add_act], op.node)
            out.append(add_op)
            acc_id = sum_out
        return out
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
            ops.extend(self._split_deep(op))
        self.ops = ops
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
            if op.kind == hx.CONV and not getattr(op, "no_hmx", False):
                wk = np.pad(w.transpose(2, 3, 1, 0), ((0, 0),) * 3 + ((0, ostride - co),))
            else:
                wk = np.pad(w[:, 0].transpose(1, 2, 0), ((0, 0), (0, 0), (0, ostride - co)))
            op.p[6] = self.pool.emit(wk)
            op.p[7] = self.pool.emit(_pad32(op.bias))
            if op.kind == hx.CONV and not getattr(op, "no_hmx", False):
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
        if len(ops) > OPERATION_LIMIT:
            raise ValueError(f"{len(ops)} operations exceed the runtime limit of {OPERATION_LIMIT}; "
                             "fewer heads or a shorter sequence")
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
            dtypes=self.dtypes,
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
        if kind < 1 or kind > 37:
            reasons.append(f"op {n}: op kind {kind} is not supported in blocked FP16 plan")
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
