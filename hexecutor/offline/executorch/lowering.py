# SPDX-License-Identifier: LGPL-3.0-or-later
"""Edge-dialect ExecuTorch graph -> HXFG v1 lowering for the hexecutor backend.

Shared by the partitioner (which asks `kind_of` / `select_nodes`) and the
backend (which calls `lower_program`). Every rule is fail-closed: a node that
cannot be expressed exactly in the tensor_f32 HXFG operation set raises
`Unsupported`; the partitioner leaves such nodes on the CPU.

Layout: PyTorch tensors are NCHW with batch one. HXFG stores each tensor as
HWC with channels padded to 32 floats; graph inputs whose consumers are all
convolutions may use compact 3-channel storage (or stride-1 storage for one channel feeding a
single 3x3/5x5 convolution). A rank-2 `[1, C]` tensor is
stored as `[1, 1, C]`. The delegate blob asks the runtime to transpose rank-4
external I/O between NCHW and NHWC; nothing inside HXFG is NCHW.

Assumptions (also in README.md):
- Max pooling ignores padded positions (equivalent to -inf padding), as the
  TFLite SAME lowering assumes for the same tensor_f32 operation.
- ReLU and ReLU6 are only supported as fused activations of a single-user
  convolution, depthwise convolution or same-shape add; elsewhere they lower to a
  standalone HX_ACT.
"""

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

from .. import hxfg_writer as hx
from . import ops as _lfm_ops  # noqa: F401  registers the hexecutor::lfm_* operators

NONE = hx.NONE
MIN_OPS = 2  # partitions with fewer delegated operations stay on the CPU
MAX_CHANNELS = hx.MAX_CHANNELS
ACT_RELU, ACT_RELU6 = hx.ACT_RELU, hx.ACT_RELU6

KINDS = {
    "convolution.default": "conv",
    "linear.default": "linear",
    "add.Tensor": "add",
    "prelu.default": "prelu",
    "_prelu_kernel.default": "prelu",
    "max_pool2d_with_indices.default": "maxpool",
    "constant_pad_nd.default": "pad",
    "upsample_bilinear2d.vec": "resize",
    "upsample_bilinear2d.default": "resize",
    "upsample_nearest2d.vec": "resize",
    "upsample_nearest2d.default": "resize",
    "view_copy.default": "reshape",
    "reshape.default": "reshape",  # the pre-decomposition spellings, seen by ops_to_not_decompose
    "view.default": "reshape",
    "permute_copy.default": "transpose",
    "permute.default": "transpose",
    "transpose.int": "transpose",  # pre-decomposition spelling; edge has permute_copy
    # Shape-only views; each lowers like a reshape (a row-preserving HWC copy when the HWC shape is unchanged).
    "unsqueeze.default": "reshape",
    "unsqueeze_copy.default": "reshape",
    "squeeze.dim": "reshape",
    "squeeze_copy.dim": "reshape",
    "squeeze.dims": "reshape",
    "squeeze_copy.dims": "reshape",
    "clone.default": "reshape",
    "_clone_dim_order.default": "reshape",
    "alias_copy.default": "reshape",
    "alias.default": "reshape",  # the pre-edge spelling the partitioner judges before the edge conversion
    # Casts of a constant (ExecuTorch wraps scalar operands in a rank-0 tensor plus a dim-order copy)
    # fold into their consumer; a cast of a runtime tensor is kind "cast" (kind_of tells them apart).
    "_to_copy.default": "constcast",
    "_to_dim_order_copy.default": "constcast",
    "cat.default": "concat",
    "embedding.default": "embedding",
    "index_select.default": "index_select",
    "index.Tensor": "index",
    "gather.default": "gather_elements",
    "index_put.default": "scatter_rows",
    "slice_scatter.default": "scatter_rows",
    "select_scatter.default": "scatter_rows",
    "_native_batch_norm_legit_no_training.default": "bn",
    "mean.dim": "mean",
    "sum.dim_IntList": "reduce",
    "amax.default": "reduce",
    "amin.default": "reduce",
    "sigmoid.default": "logistic",
    "_softmax.default": "softmax",
    "softmax.int": "softmax",
    "relu.default": "activation",
    "hardtanh.default": "activation",
    "clamp.default": "activation",
    "getitem": "getitem",
    "native_layer_norm.default": "norm",
    "native_group_norm.default": "norm",
    "instance_norm.default": "norm",
    "native_instance_norm.default": "norm",
    "rms_norm.default": "rmsnorm",
    "_native_batch_norm_legit.no_stats": "norm",
    "_native_batch_norm_legit.default": "norm",
    "addmm.default": "addmm",
    "slice_copy.Tensor": "slice",
    "slice.Tensor": "slice",
    "split_with_sizes_copy.default": "split",
    "split_with_sizes.default": "split",
    "split_copy.Tensor": "split",
    "split.Tensor": "split",
    "sub.Tensor": "binary",
    "mul.Tensor": "binary",
    "bmm.default": "bmm",
    "where.self": "where",
    "gt.Scalar": "binary", "lt.Scalar": "binary", "ge.Scalar": "binary", "le.Scalar": "binary",
    "eq.Scalar": "binary",
    "ne.Tensor": "ne", "ne.Scalar": "ne",  # EQUAL then NOT
    "argmax.default": "argmax", "argmin.default": "argmin",
    "topk.default": "topk",
    "max.dim": "maxdim", "min.dim": "maxdim",  # values via REDUCE, indices via ARGMAX/ARGMIN
    "expand_copy.default": "tile", "expand.default": "tile",
    "rsqrt.default": "rsqrt",  # SQRT then RECIPROCAL
    "log_sigmoid.default": "logsigmoid",  # one output; the export keeps this spelling
    "var.correction": "var",  # mean, centre, square, mean, scale
    # Constant factories fold into their consumers (no operation is emitted).
    "full.default": "constfactory", "full_like.default": "constfactory", "scalar_tensor.default": "constfactory",
    "arange.default": "constfactory", "arange.start": "constfactory", "arange.start_step": "constfactory",
    "scaled_dot_product_attention.default": "sdpa",
    "lfm_linear_q.default": "lfm_linear_q",
    "lfm_rope.default": "lfm_rope",
    "lfm_shortconv.default": "lfm_shortconv",
    "lfm_kv_write.default": "lfm_kv_write",
    "div.Tensor": "binary",
    "minimum.default": "binary",
    "maximum.default": "binary",
    "pow.Tensor_Tensor": "binary",
    "pow.Tensor_Scalar": "binary",
    "eq.Tensor": "binary",
    "gt.Tensor": "binary",
    "lt.Tensor": "binary",
    "ge.Tensor": "binary",
    "le.Tensor": "binary",
    "bitwise_and.Tensor": "binary",
    "bitwise_or.Tensor": "binary",
    "hardswish.default": "act",
    "hardsigmoid.default": "act",
    "silu.default": "act",
    "tanh.default": "act",
    "leaky_relu.default": "act",
    "gelu.default": "act",
    "exp.default": "act",
    "sqrt.default": "act",
    "abs.default": "act",
    "neg.default": "act",
    "bitwise_not.default": "act",  # logical NOT on bool tensors only (checked)
    "erf.default": "act",
    "sin.default": "act",
    "cos.default": "act",
    "log.default": "act",
    "reciprocal.default": "act",
    "floor.default": "act",
    "ceil.default": "act",
    "round.default": "act",
    "elu.default": "act",
    "selu.default": "act",
    "celu.default": "act",
    "softplus.default": "act",
    "mish.default": "act",
    "sinh.default": "act",
    "cosh.default": "act",
    "sign.default": "act",
    "avg_pool2d.default": "avgpool",
    "_adaptive_avg_pool2d.default": "gap",
}


class Unsupported(ValueError):
    """Raised for any node or graph that the HXFG operation set cannot express."""


# ---------------------------------------------------------------- op names


def op_name(target):
    """'convolution.default'-style names for ATen overloads and Edge overloads."""
    schema = getattr(target, "_schema", None)
    if schema is not None and getattr(schema, "name", None):
        base = schema.name.split("::")[-1]
        return f"{base}.{getattr(schema, 'overload_name', '') or 'default'}"
    name = getattr(target, "__name__", None) or str(target)
    if "aten_" in name:  # Edge names: ..._ops_aten_convolution_default
        tail = name.split("aten_", 1)[1]
        for overload in ("default", "Tensor", "dim", "vec"):
            if tail.endswith("_" + overload):
                return tail[: -len(overload) - 1] + "." + overload
    return name


def kind_name(node):
    if node.op != "call_function":
        return None
    return KINDS.get(op_name(node.target))


def _is_nearest(node):
    """upsample_nearest2d has no align flag; its scale arguments follow the output size."""
    return op_name(node.target).startswith("upsample_nearest2d")


# ---------------------------------------------------------------- metadata


def static_shape(node):
    """Static tensor shape from the exported FakeTensor metadata."""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise Unsupported(f"{node.name}: not a tensor")
    if val.dtype not in (torch.float32, torch.int32, torch.int64, torch.bool):
        raise Unsupported(f"{node.name}: dtype {val.dtype} is not supported")
    if not all(isinstance(d, int) for d in val.shape):
        raise Unsupported(f"{node.name}: dynamic shape")
    return tuple(int(d) for d in val.shape)


def hwc_of(shape, name="tensor", multihead=False):
    """HXFG [H, W, C] for a batch-one rank-4 or rank-2/3 PyTorch shape, or multi-head [B, H, L, D] / rank-3 [B, L, C]."""
    shape = tuple(shape)
    if multihead:
        if len(shape) == 4:
            h, w, c = shape[0] * shape[1], shape[2], shape[3]
        elif len(shape) == 3:
            h, w, c = shape[0], shape[1], shape[2]
        else:
            raise Unsupported(f"{name}: multihead shape {shape} must be rank 3 or 4")
    elif len(shape) == 4 and shape[0] == 1:
        h, w, c = shape[2], shape[3], shape[1]
    elif len(shape) == 3:
        h, w, c = shape[0] * shape[1], 1, shape[2]
    elif len(shape) == 2:
        h, w, c = shape[0], 1, shape[1]
    elif len(shape) == 1:
        h, w, c = 1, shape[0], 1
    else:
        raise Unsupported(f"{name}: shape {shape} is not rank 2, 3 or batch-one rank 4")
    if min(h, w, c) < 1 or c > MAX_CHANNELS:
        raise Unsupported(f"{name}: unsupported size {shape}")
    return [h, w, c]


def _constant_fqn(ep, node):
    """Qualified name of the constant behind a placeholder or constant-transform node, or None."""
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function":
        return ("folded:" + node.name) if node in _constant_nodes(ep) else None
    if node.op != "placeholder":
        return None
    sig = ep.graph_signature
    return (
        sig.inputs_to_parameters.get(node.name)
        or sig.inputs_to_buffers.get(node.name)
        or sig.inputs_to_lifted_tensor_constants.get(node.name)
    )


_CONST_CASTS = ("_to_copy.default", "_to_dim_order_copy.default")
_SHAPE_VIEWS = ("view_copy.default", "view.default", "reshape.default", "alias_copy.default", "clone.default",
               "_clone_dim_order.default", "squeeze.dim", "squeeze_copy.dim", "squeeze.dims", "squeeze_copy.dims",
               "unsqueeze.default", "unsqueeze_copy.default")
_CONST_PASSTHROUGH = ("repeat.default",) + _SHAPE_VIEWS + _CONST_CASTS
_CONST_FACTORIES = ("full.default", "full_like.default", "scalar_tensor.default",
                    "arange.default", "arange.start", "arange.start_step")
_NUMPY_DTYPES = {torch.float32: np.float32, torch.int32: np.int32, torch.int64: np.int64}


def _factory_value(node):
    """Values of a constant factory (full, scalar_tensor, arange) from its literal arguments."""
    val = node.meta.get("val")
    dtype = _NUMPY_DTYPES.get(getattr(val, "dtype", None))
    if dtype is None:
        raise Unsupported(f"{node.name}: constant dtype {getattr(val, 'dtype', None)} is not supported")
    name = op_name(node.target)
    if name == "full_like.default":  # the shape of the input (metadata only), the fill value literal
        fill = node.args[1] if len(node.args) > 1 else node.kwargs.get("fill_value")
        if not isinstance(fill, (int, float)) or isinstance(fill, bool):
            raise Unsupported(f"{node.name}: fill value must be a literal number")
        return np.full(static_shape(node.args[0]), fill, dtype=dtype)
    args = list(node.args)
    if not all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in args if not isinstance(a, list)):
        raise Unsupported(f"{node.name}: factory arguments must be literal numbers")
    try:
        if name == "full.default":
            return np.full([int(v) for v in args[0]], args[1], dtype=dtype)
        if name == "scalar_tensor.default":
            return np.asarray(args[0], dtype=dtype)
        return np.arange(*args, dtype=np.float64).astype(dtype)  # arange.default(end) and start/step forms
    except (TypeError, ValueError) as error:
        raise Unsupported(f"{node.name}: {error}") from None


def _constant_nodes(ep):
    """Call nodes that fold into their consumers: constant factories, and casts or views of folded nodes or
    of constant placeholders. A candidate stays only when every user is a candidate too or reads it at an
    argument position its emitter takes as a constant (_CONST_ARGS). Otherwise the value must exist as a
    CPU tensor, so the partition would need a tensor id that nothing produces. Computed once per graph."""
    graph = ep.graph
    cached = getattr(graph, "_hexecutor_constant_nodes", None)
    if cached is not None:
        return cached
    candidates = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        name = op_name(node.target)
        if name in _CONST_FACTORIES:
            candidates.add(node)
        elif name in _CONST_PASSTHROUGH:
            if name in _CONST_CASTS and not _float_cast(node):
                continue
            base = node.args[0]
            if (isinstance(base, torch.fx.Node) and base in candidates) or (
                    isinstance(base, torch.fx.Node) and base.op == "placeholder" and _constant_fqn(ep, base)):
                candidates.add(node)
    changed = True
    while changed:
        changed = False
        for node in list(candidates):
            for user in node.users:
                if user in candidates:
                    continue
                if user.op != "call_function" or not _reads_as_constant(user, node):
                    candidates.discard(node)
                    changed = True
                    break
    graph._hexecutor_constant_nodes = candidates
    return candidates


def _reads_as_constant(user, node):
    """True when every argument of `user` that is `node` is a constant position of its emitter."""
    allowed = _CONST_ARGS.get(kind_name(user), ())
    positions = [i for i, arg in enumerate(user.args)
                 if arg is node or (isinstance(arg, (list, tuple)) and any(a is node for a in arg))]
    return bool(positions) and all(i in allowed for i in positions)


def _float_cast(node):
    """A cast to float32 that keeps the default dense layout (dim order is checked by the shape alone)."""
    val = node.meta.get("val")
    allowed = {"dtype", "dim_order", "layout", "device", "pin_memory", "non_blocking", "memory_format"}
    return (isinstance(val, torch.Tensor) and val.dtype == torch.float32 and set(node.kwargs) <= allowed
            and list(node.kwargs.get("dim_order") or range(val.dim())) == list(range(val.dim())))


def _is_constant(ep, node):
    return isinstance(node, torch.fx.Node) and _constant_fqn(ep, node) is not None


def _const_value(ep, node):
    """The constant data behind a placeholder or constant transform (lowering only; checks never read data)."""
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function":
        name = op_name(node.target)
        if name in _CONST_FACTORIES:
            return _factory_value(node)
        if name in _CONST_PASSTHROUGH:
            base = _const_value(ep, node.args[0])
            if base is None:
                return None
            if name == "repeat.default":
                repeats = [int(v) for v in node.args[1]]
                return np.tile(base, repeats)
            if name in _SHAPE_VIEWS:  # the same elements in row-major order under the node's shape
                return base.reshape(static_shape(node))
            return base
    fqn = _constant_fqn(ep, node)
    if fqn is None:
        return None
    store = dict(ep.state_dict)
    store.update(getattr(ep, "constants", None) or {})
    if fqn not in store:
        raise Unsupported(f"constant {fqn} is missing")
    return store[fqn].detach().cpu().numpy()


def _require_constant(ep, arg, what):
    if not _is_constant(ep, arg):
        raise Unsupported(f"{what} must be a constant")


def _constant(ep, arg, what):
    _require_constant(ep, arg, what)
    return np.asarray(_const_value(ep, arg), dtype=np.float32)


def _ints(values, n):
    values = [int(v) for v in values]
    if len(values) == 1 and n > 1:
        values = values * n
    if len(values) != n:
        raise Unsupported(f"expected {n} integers, got {values}")
    return values


def _pad32(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.pad(values, (0, ((values.size + 31) & ~31) - values.size))


# ---------------------------------------------------------------- activations


def act_code(node):
    """Activation implied by a ReLU/ReLU6-style node, or None."""
    name = op_name(node.target)
    if name == "relu.default":
        return ACT_RELU
    if name == "hardtanh.default" and len(node.args) == 3:
        lo, hi = node.args[1], node.args[2]
    elif name == "clamp.default":
        lo = node.args[1] if len(node.args) > 1 else None
        hi = node.args[2] if len(node.args) > 2 else None
    else:
        return None
    if lo == 0 and hi is None:
        return ACT_RELU
    if lo == 0 and hi == 6:
        return ACT_RELU6
    return None


FUSABLE = ("convolution.default", "add.Tensor")


def fusable_producer(node, done, activation_of, ep):
    """The delegated producer a ReLU-like node folds into, or None."""
    if act_code(node) is None or not isinstance(node.args[0], torch.fx.Node):
        return None
    prod = node.args[0]
    via = None
    if kind_name(prod) == "getitem" and isinstance(prod.args[0], torch.fx.Node) and kind_name(prod.args[0]) == "bn":
        bn = prod.args[0]  # conv -> batch norm -> ReLU: the norm folds into the conv, so the ReLU can too
        conv = bn.args[0]
        if (isinstance(conv, torch.fx.Node) and conv.op == "call_function"
                and op_name(conv.target) == "convolution.default" and bn in done and prod in done
                and len(prod.users) == 1 and _folding_bn(conv, ep) is bn):
            via, prod = prod, conv
    if prod not in done or prod in activation_of:
        return None
    if op_name(prod.target) not in FUSABLE:
        return None
    if op_name(prod.target) == "convolution.default" and len(static_shape(prod.args[0])) == 3:
        return None  # a conv1d's output is transposed back after the convolution; no fused activation
    if op_name(prod.target) == "convolution.default":
        groups = int(prod.args[8]) if len(prod.args) > 8 else 1
        wshape = static_shape(prod.args[1])
        ishape = static_shape(prod.args[0])
        depthwise = groups == wshape[0] == ishape[1] and wshape[1] == 1 and groups > 1
        if groups > 1 and not depthwise:
            return None  # grouped conv decomposes into slice+conv+concat
    if op_name(prod.target) == "add.Tensor" and not _plain_add(prod, ep):
        return None  # lowered to HX_BINARY, which has no fused activation
    if via is None and (len(prod.users) != 1 or next(iter(prod.users)) is not node):
        return None
    return prod



# ---------------------------------------------------------------- elementwise helpers

ACT_FUNCTIONS = {
    "hardswish.default": lambda n: (hx.FN_HARDSWISH, 0.0, 0.0),
    "hardsigmoid.default": lambda n: (hx.FN_HARDSIGMOID, 1.0 / 6.0, 0.5),
    "silu.default": lambda n: (hx.FN_SILU, 0.0, 0.0),
    "tanh.default": lambda n: (hx.FN_TANH, 0.0, 0.0),
    "leaky_relu.default": lambda n: (
        hx.FN_LEAKY,
        float(n.args[1]) if len(n.args) > 1 else float(n.kwargs.get("negative_slope", 0.01)),
        0.0,
    ),
    "gelu.default": lambda n: (hx.FN_GELU, 0.0, 0.0),
    "exp.default": lambda n: (hx.FN_EXP, 0.0, 0.0),
    "sqrt.default": lambda n: (hx.FN_SQRT, 0.0, 0.0),
    "abs.default": lambda n: (hx.FN_ABS, 0.0, 0.0),
    "neg.default": lambda n: (hx.FN_NEG, 0.0, 0.0),
    "bitwise_not.default": lambda n: (hx.FN_NOT, 0.0, 0.0),
    "erf.default": lambda n: (hx.FN_ERF, 0.0, 0.0),
    "sin.default": lambda n: (hx.FN_SIN, 0.0, 0.0),
    "cos.default": lambda n: (hx.FN_COS, 0.0, 0.0),
    "log.default": lambda n: (hx.FN_LOG, 0.0, 0.0),
    "reciprocal.default": lambda n: (hx.FN_RECIPROCAL, 0.0, 0.0),
    "floor.default": lambda n: (hx.FN_FLOOR, 0.0, 0.0),
    "ceil.default": lambda n: (hx.FN_CEIL, 0.0, 0.0),
    "round.default": lambda n: (hx.FN_ROUND, 0.0, 0.0),
    "elu.default": lambda n: _elu_like(n, hx.FN_ELU),
    "celu.default": lambda n: _elu_like(n, hx.FN_CELU),
    "selu.default": lambda n: (hx.FN_SELU, 0.0, 0.0),
    "softplus.default": lambda n: _softplus(n),
    "mish.default": lambda n: (hx.FN_MISH, 0.0, 0.0),
    "sinh.default": lambda n: (hx.FN_SINH, 0.0, 0.0),
    "cosh.default": lambda n: (hx.FN_COSH, 0.0, 0.0),
    "sign.default": lambda n: (hx.FN_SIGN, 0.0, 0.0),
}


def _elu_like(n, fn):
    """elu(x, alpha, scale, input_scale) and celu(x, alpha): only unit scales are expressible."""
    alpha = float(n.args[1]) if len(n.args) > 1 else float(n.kwargs.get("alpha", 1.0))
    scales = [float(v) for v in n.args[2:4]] + [1.0, 1.0]
    if fn == hx.FN_ELU and scales[:2] != [1.0, 1.0]:
        raise Unsupported("elu with scale or input_scale other than 1")
    return fn, alpha, 0.0


def _softplus(n):
    beta = float(n.args[1]) if len(n.args) > 1 else float(n.kwargs.get("beta", 1.0))
    threshold = float(n.args[2]) if len(n.args) > 2 else float(n.kwargs.get("threshold", 20.0))
    if beta != 1.0 or threshold != 20.0:
        raise Unsupported("only softplus with beta 1 and threshold 20")
    return hx.FN_SOFTPLUS, 0.0, 0.0


def _constant_operand(ep, arg, data):
    """(shape, values) of a constant tensor or Python number, else None.

    Partition-time checks run on fake tensors, so values are read only when `data` is set.
    """
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return (), (np.asarray(arg, dtype=np.float32) if data else None)
    if _is_constant(ep, arg):
        shape = static_shape(arg)
        return shape, (np.asarray(_const_value(ep, arg), dtype=np.float32) if data else None)
    return None


def _binary_plan(node, ep, data=False):
    """(operation, x node, y node or None, constant per-channel vector or None, reversed).

    The output always has x's shape; y is a tensor whose h, w, c are each the output's or one,
    or a per-channel constant. Raises Unsupported for anything else.
    """
    name = op_name(node.target)
    opcode = {
        "add.Tensor": hx.BIN_ADD, "sub.Tensor": hx.BIN_SUB, "mul.Tensor": hx.BIN_MUL,
        "div.Tensor": hx.BIN_DIV, "minimum.default": hx.BIN_MIN, "maximum.default": hx.BIN_MAX,
        "pow.Tensor_Tensor": hx.BIN_POW, "pow.Tensor_Scalar": hx.BIN_POW,
        "eq.Tensor": hx.BIN_EQUAL, "gt.Tensor": hx.BIN_GREATER, "lt.Tensor": hx.BIN_LESS,
        "ge.Tensor": hx.BIN_GE, "le.Tensor": hx.BIN_LE,
        "eq.Scalar": hx.BIN_EQUAL, "gt.Scalar": hx.BIN_GREATER, "lt.Scalar": hx.BIN_LESS,
        "ge.Scalar": hx.BIN_GE, "le.Scalar": hx.BIN_LE,
        "ne.Tensor": hx.BIN_EQUAL, "ne.Scalar": hx.BIN_EQUAL,  # ne is EQUAL followed by NOT (kind "ne")
        "bitwise_and.Tensor": hx.BIN_AND, "bitwise_or.Tensor": hx.BIN_OR,
    }[name]
    a, b = node.args[0], node.args[1]
    alpha = node.args[2] if len(node.args) > 2 else node.kwargs.get("alpha", 1)
    if alpha != 1:
        raise Unsupported(f"{name} alpha must be one")
    out = static_shape(node)
    hwc_of(out, node.name)
    if _dtype_of(node) in (torch.int32, torch.int64):
        raise Unsupported(f"{name}: integer arithmetic results are unsupported (comparisons give bool)")
    ca, cb = _constant_operand(ep, a, data), _constant_operand(ep, b, data)
    if (ca is None and not isinstance(a, torch.fx.Node)) or (
        cb is None and not isinstance(b, torch.fx.Node)
    ):
        raise Unsupported("unsupported operand")
    if ca is not None and cb is not None:
        raise Unsupported("both operands constant")
    if ca is not None or cb is not None:
        tensor, (cshape, const), reversed_ = (b, ca, 1) if ca is not None else (a, cb, 0)
        if static_shape(tensor) != out:
            raise Unsupported("constant operand would broadcast the tensor")
        if _is_int_node(tensor):  # integer operands are compared as integers: the constant must be whole
            _integral_constant(ep, a if ca is not None else b, node.name)
        shape = (1,) * (len(out) - len(cshape)) + tuple(cshape)
        if len(shape) != len(out) or (shape[0] != 1 and len(out) != 2):
            raise Unsupported("constant operand rank")
        if len(out) < 2:
            raise Unsupported("constant operand on a rank-1 tensor")
        # HXFG channels: dim 1 of an NCHW tensor, the last dim of a [1, C] or [B, L, C] token tensor.
        channels = out[1] if len(out) == 4 else out[-1]
        if len(out) == 4 and (len(shape) < 4 or shape[2] != 1 or shape[3] != 1):
            if not data:
                return opcode, tensor, None, None, reversed_, 2
            try:
                c_arr = np.broadcast_to(np.asarray(const, dtype=np.float32), out)
            except ValueError:
                raise Unsupported("cannot broadcast constant to output")
            hwc_arr = c_arr[0].transpose(1, 2, 0)
            stride = (channels + 31) & ~31
            if stride > channels:
                hwc_arr = np.pad(hwc_arr, ((0, 0), (0, 0), (0, stride - channels)))
            return opcode, tensor, None, np.ascontiguousarray(hwc_arr).reshape(-1), reversed_, 2
        cdim = 1 if len(out) == 4 else len(out) - 1
        if len(out) in (2, 3) and (any(d != 1 for i, d in enumerate(shape) if i not in (0, cdim))
                                   or (len(out) == 2 and shape[0] != 1)):
            # A full-size constant on a token tensor (an additive attention mask): one value per element.
            if not data:
                return opcode, tensor, None, None, reversed_, 2
            try:
                full = np.broadcast_to(np.asarray(const, dtype=np.float32), out)
            except ValueError:
                raise Unsupported("cannot broadcast constant to output")
            h, w, c = hwc_of(out, node.name)
            stride = (c + 31) & ~31
            padded = np.zeros((h * w, stride), np.float32)
            padded[:, :c] = full.reshape(h * w, c)
            return opcode, tensor, None, padded.reshape(-1), reversed_, 2
        if shape[cdim] not in (1, channels) or any(
                d != 1 for i, d in enumerate(shape) if i not in (0, cdim)):
            raise Unsupported("constant operand must be per-channel")
        if all(d == 1 for d in shape):
            # A scalar: cover every axis that may be HXFG channels (a multi-head [B, H, L, D] tensor
            # stores D as channels, which the partition-time plan cannot tell from NCHW).
            channels = max(out[1:])
        vec = (
            np.broadcast_to(const.reshape(-1), (channels,)).astype(np.float32)
            if data
            else None
        )
        return opcode, tensor, None, vec, reversed_, 1
    sa, sb = static_shape(a), static_shape(b)
    if len(sa) != len(sb):
        raise Unsupported("operands of different rank")

    def fits(small, big):  # every dim of small is big's or one; batch is one (or the shapes are equal)
        return len(small) == len(big) and (small[0] == 1 or small == big) and all(
            s in (1, o) for s, o in zip(small, big)
        )

    if sa == out and fits(sb, out):
        return opcode, a, b, None, 0
    if sb == out and fits(sa, out):
        rev = 1 if opcode in (hx.BIN_SUB, hx.BIN_DIV, hx.BIN_POW, hx.BIN_GREATER, hx.BIN_LESS, hx.BIN_GE, hx.BIN_LE) else 0
        return opcode, b, a, None, rev
    raise Unsupported("operands broadcast against each other")


def _plain_add(node, ep):
    """Same-shape tensor + tensor, which keeps the original HX_ADD with fusable activation."""
    try:
        plan = _binary_plan(node, ep)
    except Unsupported:
        return False
    opcode, x, y = plan[0], plan[1], plan[2]
    return opcode == hx.BIN_ADD and y is not None and static_shape(x) == static_shape(y)


# ---------------------------------------------------------------- classification

# ATen reductions over non-batch dims. mean.dim becomes HX_MEAN when it is the spatial mean.
REDUCE_OPS = {
    "sum.dim_IntList": hx.RED_SUM,
    "amax.default": hx.RED_MAX,
    "amin.default": hx.RED_MIN,
    "mean.dim": hx.RED_MEAN,
}


def _reduce_dims(ishape, dims, name):
    """Normalised, distinct, non-batch reduced dims of a rank-2 or rank-4 tensor, and the HWC mask."""
    if len(ishape) not in (2, 4):
        raise Unsupported(f"{name}: reductions need a rank 2 or rank 4 tensor")
    if not dims:
        raise Unsupported(f"{name}: reductions over every dimension are unsupported")
    bits = {1: 4, 2: 1, 3: 2} if len(ishape) == 4 else {1: 4}  # dim C, H, W -> HWC c, h, w
    mask, normal = 0, []
    for d in dims:
        d = int(d) + len(ishape) if int(d) < 0 else int(d)
        if d not in bits or mask & bits[d]:
            raise Unsupported(f"{name}: reduced dims must be distinct non-batch dims")
        mask |= bits[d]
        normal.append(d)
    return normal, mask


def _reduce_plan(node):
    """(HX_RED operation, HWC mask) of a reduction over batch-one rank-2 or rank-4 dims. The
    output must have the keepdim layout: each reduced dim is one, the others are kept."""
    x = node.args[0]
    ishape = static_shape(x)
    dims = node.args[1] if len(node.args) > 1 else None
    _, mask = _reduce_dims(ishape, dims, node.name)
    hin = hwc_of(ishape, x.name)
    expect = [1 if mask & 1 else hin[0], 1 if mask & 2 else hin[1], 1 if mask & 4 else hin[2]]
    if hwc_of(static_shape(node), node.name) != expect:
        raise Unsupported(f"{node.name}: output must keep each reduced dim as size one")
    return REDUCE_OPS[op_name(node.target)], mask


def kind_of(node, ep):
    """Kind of a supported node, or None. Raises Unsupported for failed static checks."""
    kind = kind_name(node)
    if kind is None:
        return None
    if kind == "constcast" and not _is_constant(ep, node):
        kind = "cast"  # a cast of a runtime tensor is an HX_CAST
    check(kind, node, ep)
    return kind


def _norm_affine(ep, arg, target, what, data, spatial):
    """Per-channel float32 vector of a normalisation weight or bias. `target` is (C,) or, for a
    rank-4 LayerNorm over (C, H, W), (C, H, W); the value must broadcast to it and, with
    `spatial`, be constant over H and W. None when absent or when only checking (data unset)."""
    if arg is None:
        return None
    _require_constant(ep, arg, what)
    try:
        ok = np.broadcast_shapes(tuple(static_shape(arg)), target) == target
    except ValueError:
        ok = False
    if not ok:
        raise Unsupported(f"{what} must have shape {target} or broadcast to it")
    if not data:
        return None
    full = np.broadcast_to(np.asarray(_const_value(ep, arg), np.float32), target)
    vec = full[:, 0, 0] if spatial else full
    if spatial and not np.array_equal(full, np.broadcast_to(vec[:, None, None], target)):
        raise Unsupported(f"{what} must be constant over H and W")
    return np.ascontiguousarray(vec, dtype=np.float32)


def _norm_plan(node, ep, data=False):
    """(HX_NORM mode, groups, epsilon, gamma, beta, has_getitem) of native_layer_norm, native_group_norm,
    instance_norm or batch_norm over a batch-one tensor."""
    name = op_name(node.target)
    if name in ("instance_norm.default", "native_instance_norm.default"):
        x = node.args[0]
        ishape = static_shape(x)
        hwc = hwc_of(ishape, x.name)
        if len(ishape) not in (3, 4):
            raise Unsupported(f"{node.name}: instance norm requires a rank-3 or rank-4 tensor")
        c = hwc[2]
        weight = node.args[1] if len(node.args) > 1 else None
        bias = node.args[2] if len(node.args) > 2 else None
        eps = float(node.args[7]) if len(node.args) > 7 and node.args[7] is not None else (float(node.args[3]) if len(node.args) > 3 and isinstance(node.args[3], float) else 1e-5)
        mode, groups, spatial, target = hx.NORM_GROUP, c, False, (c,)
        gamma = _norm_affine(ep, weight, target, "normalisation weight", data, spatial)
        beta = _norm_affine(ep, bias, target, "normalisation bias", data, spatial)
        if data:
            gamma = np.ones(c, np.float32) if gamma is None else gamma
            beta = np.zeros(c, np.float32) if beta is None else beta
        return mode, groups, eps, gamma, beta, False

    if name in ("_native_batch_norm_legit.no_stats", "_native_batch_norm_legit.default"):
        users = list(node.users)
        if len(users) != 1 or kind_name(users[0]) != "getitem" or users[0].args[1] != 0:
            raise Unsupported(f"{node.name}: only the normalised output is supported")
        x = node.args[0]
        ishape = static_shape(x)
        hwc = hwc_of(ishape, x.name)
        if len(ishape) != 4:
            raise Unsupported(f"{node.name}: batch norm on batch 1 requires rank 4")
        c = hwc[2]
        weight = node.args[1] if len(node.args) > 1 else None
        bias = node.args[2] if len(node.args) > 2 else None
        if name == "_native_batch_norm_legit.default":
            # (input, weight, bias, running_mean, running_var, training, momentum, eps)
            training = bool(node.args[5]) if len(node.args) > 5 else False
            if not training:
                raise Unsupported(f"{node.name}: batch norm with training=False is unsupported; fold running stats before export")
            eps = float(node.args[7]) if len(node.args) > 7 and node.args[7] is not None else 1e-5
        else:
            # (input, weight, bias, training, momentum, eps)
            training = bool(node.args[3]) if len(node.args) > 3 else True
            if not training:
                raise Unsupported(f"{node.name}: batch norm with training=False is unsupported")
            eps = float(node.args[5]) if len(node.args) > 5 and node.args[5] is not None else 1e-5
        mode, groups, spatial, target = hx.NORM_GROUP, c, False, (c,)
        gamma = _norm_affine(ep, weight, target, "normalisation weight", data, spatial)
        beta = _norm_affine(ep, bias, target, "normalisation bias", data, spatial)
        if data:
            gamma = np.ones(c, np.float32) if gamma is None else gamma
            beta = np.zeros(c, np.float32) if beta is None else beta
        return mode, groups, eps, gamma, beta, True

    users = list(node.users)
    if len(users) != 1 or kind_name(users[0]) != "getitem" or users[0].args[1] != 0:
        raise Unsupported(f"{node.name}: only the normalised output is supported")
    x = node.args[0]
    ishape = static_shape(x)
    hwc = hwc_of(ishape, x.name)
    if static_shape(users[0]) != ishape:
        raise Unsupported(f"{node.name}: output shape differs from the input")
    c = hwc[2]
    if name == "native_layer_norm.default":
        normalized = [int(v) for v in node.args[1]]
        weight, bias, eps = node.args[2], node.args[3], float(node.args[4])
        if len(ishape) in (2, 3):
            if normalized != [c]:
                raise Unsupported(f"{node.name}: a token layer norm must normalise over C")
            mode, groups, spatial, target = hx.NORM_CHANNEL, 0, False, (c,)
        else:
            if normalized != list(ishape[1:]):
                raise Unsupported(f"{node.name}: a rank-4 layer norm must normalise over (C, H, W)")
            mode, groups, spatial = hx.NORM_GROUP, 1, True
            target = (c, hwc[0], hwc[1])
    else:  # native_group_norm(x, weight, bias, N, C, HxW, group, eps)
        if len(ishape) != 4 or int(node.args[3]) != 1 or int(node.args[4]) != c:
            raise Unsupported(f"{node.name}: group norm needs batch one and the input's channels")
        if int(node.args[5]) != hwc[0] * hwc[1]:
            raise Unsupported(f"{node.name}: group norm HxW disagrees with the input")
        groups, eps = int(node.args[6]), float(node.args[7])
        weight, bias = node.args[1], node.args[2]
        if groups < 1 or c % groups:
            raise Unsupported(f"{node.name}: group count must divide the channels")
        mode, spatial, target = hx.NORM_GROUP, False, (c,)
    gamma = _norm_affine(ep, weight, target, "normalisation weight", data, spatial)
    beta = _norm_affine(ep, bias, target, "normalisation bias", data, spatial)
    if data:
        gamma = np.ones(c, np.float32) if gamma is None else gamma
        beta = np.zeros(c, np.float32) if beta is None else beta
    return mode, groups, eps, gamma, beta, True


def _rms_norm_plan(node, ep, data=False):
    x = node.args[0]
    ishape = static_shape(x)
    hwc = hwc_of(ishape, x.name)
    c = hwc[2]
    if len(node.args) > 1 and node.args[1] is not None:
        normalized = [int(v) for v in node.args[1]]
        if normalized != [c]:
            raise Unsupported(f"{node.name}: rms_norm normalized_shape {normalized} must match last dimension {c}")
    weight = node.args[2] if len(node.args) > 2 else None
    eps = float(node.args[3]) if len(node.args) > 3 and node.args[3] is not None else 1e-5
    target = (c,)
    gamma = _norm_affine(ep, weight, target, "rms_norm weight", data, spatial=False)
    return eps, gamma


def _perm(node):
    """The full permutation of a permute or transpose.int node."""
    if op_name(node.target) == "transpose.int":
        rank = len(static_shape(node))
        d0, d1 = (int(v) % rank for v in node.args[1:3])
        perm = list(range(rank))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return perm
    rank = len(node.args[1])
    return [int(v) % rank for v in node.args[1]]


def _is_swap12(node):
    """A rank-3 transpose of the last two axes ([1, A, B] -> [1, B, A])."""
    return (kind_name(node) == "transpose" and len(static_shape(node)) == 3
            and _perm(node) == [0, 2, 1] and static_shape(node)[0] == 1)


def token_view(node):
    """Role of `node` in the attention token views, or None.

    [1, C, H, W] -view-> [1, C, HW] -transpose(1, 2)-> [1, HW, C] and its reverse are a single
    HWC-order copy, because the HWC storage of [1, C, H, W] is already the row-major [HW, C]
    matrix. "flat" / "unflat" mark the first node of each pair (it emits nothing; its only user
    is the second); "flat_tail" / "unflat_tail" mark the second, which copies from the first's
    input. Returns (role, source node of the pair).
    """
    if node.op != "call_function":
        return None
    kind = kind_name(node)
    if kind == "reshape" and len(node.users) == 1:
        ishape, out = static_shape(node.args[0]), static_shape(node)
        user = next(iter(node.users))
        if (len(ishape) == 4 and ishape[0] == 1 and out == (1, ishape[1], ishape[2] * ishape[3])
                and _is_swap12(user)):
            return "flat", node.args[0]
    if kind == "transpose" and _is_swap12(node):
        src = node.args[0]
        if isinstance(src, torch.fx.Node) and (token_view(src) or ("",))[0] == "flat":
            return "flat_tail", src.args[0]
        if len(node.users) == 1:
            user = next(iter(node.users))
            ishape = static_shape(src)
            out = static_shape(user) if kind_name(user) == "reshape" else None
            if (out is not None and len(out) == 4 and out[0] == 1 and out[1] == ishape[2]
                    and out[2] * out[3] == ishape[1]):
                return "unflat", src
    if kind == "reshape":
        src = node.args[0]
        if isinstance(src, torch.fx.Node) and (token_view(src) or ("",))[0] == "unflat":
            return "unflat_tail", src.args[0]
    return None


def _channel_preserving(ishape, oshape):
    """A reshape between batch-one rank-4 tensors that keeps dim 1 (channels): a copy of the HWC pixel
    sequence, exact because both sides read the pixels row-major and the channel vector is unchanged."""
    return (len(ishape) == 4 and len(oshape) == 4 and ishape[0] == 1 and oshape[0] == 1
            and ishape[1] == oshape[1] and ishape[2] * ishape[3] == oshape[2] * oshape[3])


def _rowmajor_reshape(node):
    """A reshape whose operands keep PyTorch row-major order in HXFG (token or attention tensors)."""
    x = node.args[0]
    if len(static_shape(x)) <= 3 and len(static_shape(node)) <= 3:
        return True
    attention = ("sdpa", "bmm")
    feeds = node.users and all(kind_name(u) in attention for u in node.users)
    fed = isinstance(x, torch.fx.Node) and kind_name(x) in attention
    return bool(feeds or fed)


# ---------------------------------------------------------------- new operation plans
# Each plan validates one edge operation and returns what its emitter needs. Checks run on fake tensors
# (data=False); values are read only when lowering. Layout facts used below (see the module docstring):
# HWC pixels are row-major over the non-channel dims; channels are dim 1 of [1, C, H, W] and the last
# dim of [L, C] or [1, L, C]. ARGMAX/TOPK index pixels linearly, but REDUCE needs the exact keepdim HWC.

_MULTI_OUTPUT_KINDS = ("topk", "maxdim")  # their getitem users must all be delegated
_FOLDED_KINDS = ("constfactory", "constcast", "reshape")  # constants read through _const_value


def _arg(node, index, name, default=None):
    """Positional or keyword argument of an edge operation."""
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _dtype_of(node):
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return val.dtype if isinstance(val, torch.Tensor) else None


def _float_result(node):
    return _dtype_of(node) == torch.float32


def _is_int_node(node):
    """An integer tensor (argmax / argmin / topk indices), stored as int32 in HXFG."""
    return _dtype_of(node) in (torch.int32, torch.int64)


def _integral_constant(ep, arg, name):
    """Refuse a fractional constant against an integer tensor (the comparison would change meaning)."""
    values = np.asarray(arg if isinstance(arg, (int, float)) else _const_value(ep, arg), dtype=np.float64)
    if not np.all(values == np.rint(values)):
        raise Unsupported(f"{name}: integer tensor compared with a fractional constant")


def _internal_result(node):
    """A result that must stay inside the delegate: a bool or integer tensor, or a multi-output
    producer whose outputs are read through getitem (delegate boundary tensors are float32)."""
    if kind_name(node) in _MULTI_OUTPUT_KINDS:
        return True
    dtype = _dtype_of(node)
    return dtype is not None and dtype != torch.float32


def _require_runtime(ep, x, name):
    if not isinstance(x, torch.fx.Node) or _is_constant(ep, x):
        raise Unsupported(f"{name}: operand must be a runtime tensor")
    if not _float_result(x):
        raise Unsupported(f"{name}: operand must be float32")


def _channel_axis(shape):
    return 1 if len(shape) == 4 else len(shape) - 1


def _channel_dim(node, index, ishape, name, default=None):
    dim = _arg(node, index, "dim", default)
    if dim is None:
        raise Unsupported(f"{name}: reductions over every dimension are unsupported")
    dim = int(dim) + len(ishape) if int(dim) < 0 else int(dim)
    if dim != _channel_axis(ishape):
        raise Unsupported(f"{name}: only the channel axis is supported (dim {dim} of {ishape})")
    return dim


def _output_users(node, allowed):
    """Map getitem index -> user for a multi-output producer; any other user is unsupported."""
    users = {}
    for user in node.users:
        if kind_name(user) != "getitem" or int(user.args[1]) not in allowed:
            raise Unsupported(f"{node.name}: only the getitem outputs {allowed} are supported")
        users[int(user.args[1])] = user
    return users


def _argreduce_plan(node, ep):
    """ARGMAX / ARGMIN over the channel axis: one int32 index per pixel."""
    x = node.args[0]
    _require_runtime(ep, x, node.name)
    ishape = static_shape(x)
    _channel_dim(node, 1, ishape, node.name)
    hin, hout = hwc_of(ishape, x.name), hwc_of(static_shape(node), node.name)
    if hout[:2] != hin[:2] or hout[2] != 1:  # HX_ARGMAX preflight: the output is [h, w, 1] of the input
        raise Unsupported(f"{node.name}: output must be one index per pixel in the input's HWC layout")
    return ishape


def _topk_plan(node, ep):
    """(k, largest, sorted, output shape) of a TOPK over the channel axis."""
    x = node.args[0]
    _require_runtime(ep, x, node.name)
    ishape = static_shape(x)
    dim = _channel_dim(node, 2, ishape, node.name, default=-1)  # aten.topk's dim defaults to -1
    k = int(_arg(node, 1, "k"))
    if not 1 <= k <= min(hwc_of(ishape, x.name)[2], 128):
        raise Unsupported(f"{node.name}: k must be between 1 and min(C, 128)")
    expect = list(ishape)
    expect[dim] = k
    expect = tuple(expect)
    hwc_of(expect, node.name)
    for index, user in _output_users(node, (0, 1)).items():
        if static_shape(user) != expect:
            raise Unsupported(f"{node.name}: output {index} must have shape {expect}")
    return k, bool(_arg(node, 3, "largest", True)), bool(_arg(node, 4, "sorted", True)), expect


def _maxdim_plan(node, ep):
    """max.dim / min.dim with keepdim: values by REDUCE (keepdim HWC), indices by ARGMAX/ARGMIN."""
    x = node.args[0]
    _require_runtime(ep, x, node.name)
    ishape = static_shape(x)
    dim = _channel_dim(node, 1, ishape, node.name)
    if not bool(_arg(node, 2, "keepdim", False)):
        raise Unsupported(f"{node.name}: max/min over a dimension needs keepdim=True")
    expect = list(ishape)
    expect[dim] = 1
    expect = tuple(expect)
    users = _output_users(node, (0, 1))
    if 0 in users and hwc_of(static_shape(users[0]), node.name) != hwc_of(expect, node.name):
        raise Unsupported(f"{node.name}: values must keep the reduced channel as size one")
    if 1 in users and static_shape(users[1]) != expect:
        raise Unsupported(f"{node.name}: indices must have shape {expect}")
    return users


def _tile_plan(node, ep):
    """expand_copy: same rank, each dim one or equal to the output (HWC multiples), no constant input."""
    x = node.args[0]
    _require_runtime(ep, x, node.name)
    ishape, out = static_shape(x), static_shape(node)
    if len(ishape) != len(out) or any(a not in (1, o) for a, o in zip(ishape, out)):
        raise Unsupported(f"{node.name}: expand only repeats size-one dims")
    hin, hout = hwc_of(ishape, x.name), hwc_of(out, node.name)
    if any(o % i for i, o in zip(hin, hout)):
        raise Unsupported(f"{node.name}: expand is not an HWC tile")


def _clamp_plan(node, ep):
    """(lower, upper) bound of a clamp or hardtanh that is not a ReLU/ReLU6 fusion; None for a missing bound."""
    if op_name(node.target) == "hardtanh.default":
        lo, hi = _arg(node, 1, "min_val"), _arg(node, 2, "max_val")
    else:
        lo, hi = _arg(node, 1, "min"), _arg(node, 2, "max")
    for bound in (lo, hi):
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
            raise Unsupported(f"{node.name}: clamp bounds must be numbers")
    if lo is None and hi is None:
        raise Unsupported(f"{node.name}: clamp without bounds")
    _require_runtime(ep, node.args[0], node.name)
    if not _float_result(node):
        raise Unsupported(f"{node.name}: clamp is float32 only")
    hwc_of(static_shape(node), node.name)
    return (None if lo is None else float(lo)), (None if hi is None else float(hi))


def _where_plan(node, ep, data=False):
    """(condition node, [(mode, tensor node or scalar bits or constant array), x and y operands])."""
    cond, x, y = node.args[0], node.args[1], node.args[2]
    out = static_shape(node)
    if _dtype_of(node) != torch.float32:
        raise Unsupported(f"{node.name}: where is supported for float32 results only")
    hwc_out = hwc_of(out, node.name)
    if not isinstance(cond, torch.fx.Node) or _is_constant(ep, cond):
        raise Unsupported(f"{node.name}: the condition must be a runtime tensor")
    if not _fits(static_shape(cond), out):
        raise Unsupported(f"{node.name}: condition must broadcast to the output")
    operands = []
    for arg in (x, y):
        if isinstance(arg, torch.fx.Node) and not _is_constant(ep, arg):
            if not _fits(static_shape(arg), out):
                raise Unsupported(f"{node.name}: operand must broadcast to the output")
            operands.append((0, arg, None))
            continue
        const = _constant_operand(ep, arg, data)
        if const is None:
            raise Unsupported(f"{node.name}: unsupported operand")
        cshape, cvals = const
        try:
            if np.broadcast_shapes(tuple(cshape), out) != out:
                raise Unsupported(f"{node.name}: constant operand must broadcast to the output")
        except ValueError:
            raise Unsupported(f"{node.name}: constant operand must broadcast to the output") from None
        if not data:
            operands.append((0, None, None))
            continue
        full = np.broadcast_to(np.asarray(cvals, np.float32), out)
        plane = np.ascontiguousarray(full[0].transpose(1, 2, 0)) if len(out) == 4 else full.reshape(hwc_out)
        if np.all(plane == plane[0, 0, 0]):
            operands.append((1, float(plane[0, 0, 0]), None))  # one value: a scalar
        elif np.all(plane == plane[:1, :1, :]):
            operands.append((2, None, _pad32(plane[0, 0])))  # per channel
        else:
            stride = (hwc_out[2] + 31) & ~31
            padded = np.zeros(plane.shape[:2] + (stride,), np.float32)
            padded[:, :, :hwc_out[2]] = plane
            operands.append((3, None, padded.reshape(-1)))  # full size, per pixel
    return cond, operands


def _fits(small, big):
    """Every dim of `small` is one or the matching dim of `big` (same rank)."""
    return len(small) == len(big) and all(s in (1, o) for s, o in zip(small, big))


def _var_plan(node, ep):
    """(reduced HWC mask, reduced NCHW dims, correction factor N / (N - correction)) of var.correction."""
    x = node.args[0]
    _require_runtime(ep, x, node.name)
    ishape = static_shape(x)
    dims, mask = _reduce_dims(ishape, _arg(node, 1, "dim"), node.name)
    hin = hwc_of(ishape, x.name)
    expect = [1 if mask & 1 else hin[0], 1 if mask & 2 else hin[1], 1 if mask & 4 else hin[2]]
    if hwc_of(static_shape(node), node.name) != expect:
        raise Unsupported(f"{node.name}: output must keep each reduced dim as size one")
    count = int(np.prod([ishape[d] for d in dims]))
    correction = _arg(node, 2, "correction")
    correction = 1.0 if correction is None else float(correction)
    if count - correction <= 0:
        raise Unsupported(f"{node.name}: correction must be below the reduced count")
    return mask, dims, count / (count - correction)


def _const_plan(ep, node):
    _const_value(ep, node)  # rejects literal arguments that cannot be folded
    static_shape(node)


def _bool_valued(node):
    return _dtype_of(node) == torch.bool


def _conv1d_params(node):
    """(stride, padding, dilation, groups) of a rank-3 aten.convolution: one value per call argument."""
    def first(value, default):
        return int(value[0]) if value else default
    stride = first(node.args[3], 1) if len(node.args) > 3 else 1
    pad = first(node.args[4], 0) if len(node.args) > 4 else 0
    dil = first(node.args[5], 1) if len(node.args) > 5 else 1
    groups = int(node.args[8]) if len(node.args) > 8 else 1
    return stride, pad, dil, groups


def _conv1d_check(node, ep):
    """A batch-one conv1d [1, C, L] with weights [O, I, K] (dense or depthwise), lowered as a 2-D
    convolution with one row: its input and output go through HX_TRANSPOSE (see _emit_conv1d)."""
    x, w = node.args[0], node.args[1]
    ishape, oshape, wshape = static_shape(x), static_shape(node), static_shape(w)
    if _transposed(node) or ishape[0] != 1 or len(oshape) != 3 or len(wshape) != 3:
        raise Unsupported(f"{node.name}: conv1d must be batch one and not transposed")
    if len(node.args) > 7 and any(int(v) for v in node.args[7]):
        raise Unsupported(f"{node.name}: output padding")
    stride, pad, dil, groups = _conv1d_params(node)
    depthwise = groups == wshape[0] == ishape[1] and wshape[1] == 1 and groups > 1
    if groups != 1 and not depthwise:
        raise Unsupported(f"{node.name}: only dense or depthwise conv1d is supported")
    if groups == 1 and wshape[1] != ishape[1]:
        raise Unsupported(f"{node.name}: input channel mismatch")
    if stride < 1 or pad < 0 or not 1 <= dil <= 8:
        raise Unsupported(f"{node.name}: stride, padding or dilation out of range")
    length = (ishape[2] + 2 * pad - dil * (wshape[2] - 1) - 1) // stride + 1
    if oshape != (1, wshape[0], length):
        raise Unsupported(f"{node.name}: output shape disagrees with the convolution geometry")
    _require_constant(ep, w, "convolution weight")
    if len(node.args) > 2 and node.args[2] is not None:
        _require_constant(ep, node.args[2], "convolution bias")
    hwc_of(ishape, x.name)
    hwc_of(oshape, node.name)


# ---------------------------------------------------------------- casts, gathers, scatters, softmax, batch norm
# Plans for the operations that map onto HX_CAST, HX_GATHER_ELEMENTS, HX_GATHER_ND, HX_SCATTER_ND,
# HX_ATTN_SOFTMAX and per-channel batch normalisation. Parameter encodings follow lower_onnx_float.py
# (op_Cast, op_GatherElements, op_GatherND, op_ScatterND, op_Softmax).

_CAST_DTYPES = (torch.float32, torch.int32, torch.int64, torch.bool)


def _hwc_axis(shape, dim, name="tensor"):
    """HXFG axis (0 = H, 1 = W, 2 = C) of PyTorch dimension `dim` of a batch-one tensor in the layout hwc_of
    gives it: channels are dim 1 of [1, C, H, W] and the last dim of [L, C] or [1, L, C]."""
    rank = len(shape)
    dim = dim + rank if dim < 0 else dim
    if not 0 <= dim < rank:
        raise Unsupported(f"{name}: dimension {dim} is out of range")
    hwc_of(shape, name)
    if rank == 4 and shape[0] == 1 and dim != 0:
        return {1: 2, 2: 0, 3: 1}[dim]
    if rank == 3:
        if dim == 2:
            return 2
        if dim == 1 and shape[0] == 1:
            return 0
    if rank == 2:
        return 2 if dim == 1 else 0
    if rank == 1:
        return 1
    raise Unsupported(f"{name}: dimension {dim} of {tuple(shape)} has no HWC axis")


def _hwc_array(values, shape):
    """`values` (PyTorch order, `shape`) as the dense [H, W, C] array HXFG stores for that shape."""
    arr = np.asarray(values).reshape(shape)
    h, w, c = hwc_of(shape)
    return np.ascontiguousarray(arr[0].transpose(1, 2, 0)) if len(shape) == 4 else arr.reshape(h, w, c)


def _cast_plan(node, ep):
    """(source dtype, destination dtype) of a cast of a runtime tensor between float32, int32 and bool
    (int64 is stored as int32). Layout-changing dim orders and other dtypes are refused."""
    x = node.args[0]
    if not isinstance(x, torch.fx.Node) or _is_constant(ep, x):
        raise Unsupported(f"{node.name}: cast operand must be a runtime tensor")
    src, dst = _dtype_of(x), _dtype_of(node)
    if src not in _CAST_DTYPES or dst not in _CAST_DTYPES:
        raise Unsupported(f"{node.name}: only casts between float32, int32 and bool are lowered ({src} -> {dst})")
    if not _float_cast_layout(node):
        raise Unsupported(f"{node.name}: cast changes the memory layout")
    hwc_of(static_shape(x), x.name)
    hwc_of(static_shape(node), node.name)
    if static_shape(x) != static_shape(node):
        raise Unsupported(f"{node.name}: cast changes the shape")
    return src, dst


def _float_cast_layout(node):
    """The cast keeps the dense default dim order (an explicit dim_order must be the identity)."""
    allowed = {"dtype", "dim_order", "layout", "device", "pin_memory", "non_blocking", "memory_format"}
    val = node.meta.get("val")
    return (isinstance(val, torch.Tensor) and set(node.kwargs) <= allowed
            and list(node.kwargs.get("dim_order") or range(val.dim())) == list(range(val.dim())))


def _emit_cast(b, node):
    """HX_CAST modes: 0 float -> int32, 1 int32 -> float, 6 identity. Bool tensors are float 0/1 in this
    front end (comparisons write them), so float/int -> bool is `x != 0` (EQUAL then NOT) and bool -> int32
    is a float -> int32 cast; bool -> float and same-type casts are identity copies."""
    x = node.args[0]
    src, dst = _dtype_of(x), _dtype_of(node)
    ints = (torch.int32, torch.int64)
    out, tx = b.tid(node), b.tid(x)
    if dst == torch.bool and src != torch.bool:
        eq = b.temp(node.name + "_eq", static_shape(node))
        c = hwc_of(static_shape(node), node.name)[2]
        b.emit_to(hx.BINARY, tx, NONE, eq, [hx.BIN_EQUAL, 1, _pool_vec(b, np.zeros(c, np.float32), src in ints), 0])
        b.emit_to(hx.ACT, eq, NONE, out, [hx.FN_NOT, 0, 0])
        return
    if dst in ints:
        mode = 6 if src in ints else 0
        b.dtypes[out] = hx.DTYPE_INT32
    elif src in ints:  # to float32
        mode = 1
    else:  # float32 or bool to float32 (bool is already 0/1 floats), or bool to bool
        mode = 6
    b.emit_to(hx.CAST, tx, NONE, out, [mode])


def _gather_plan(node, ep):
    """(HXFG axis, normalised dim) of aten.gather(x, dim, index): `index` has x's shape except along dim."""
    x, dim, index = node.args[0], int(node.args[1]), node.args[2]
    if not isinstance(x, torch.fx.Node) or not isinstance(index, torch.fx.Node):
        raise Unsupported(f"{node.name}: gather operands must be tensors")
    if _is_constant(ep, x):
        raise Unsupported(f"{node.name}: gather from a constant table is unsupported (the data must be a runtime tensor)")
    if _dtype_of(x) != torch.float32 or _dtype_of(node) != torch.float32:
        raise Unsupported(f"{node.name}: only float32 data is gathered")
    if _dtype_of(index) not in (torch.int32, torch.int64):
        raise Unsupported(f"{node.name}: indices must be integers")
    xs, idxs = static_shape(x), static_shape(index)
    if len(xs) != len(idxs) or static_shape(node) != idxs:
        raise Unsupported(f"{node.name}: index and data ranks differ")
    dim = dim + len(xs) if dim < 0 else dim
    if any(a != c for i, (a, c) in enumerate(zip(xs, idxs)) if i != dim):
        raise Unsupported(f"{node.name}: index must match the data except along dim")
    axis = _hwc_axis(xs, dim, x.name)
    hwc_of(idxs, index.name)
    return axis, dim


def _emit_gather_elements(b, ep, node):
    x, index = node.args[0], node.args[2]
    axis, _ = _gather_plan(node, ep)
    if _is_constant(ep, index):
        h, w, c = hwc_of(static_shape(index), index.name)
        idx = _hwc_array(np.asarray(_const_value(ep, index)).astype(np.int32), static_shape(index)).reshape(-1)
        # The runtime requires the constant to span the output tensor's padded storage.
        idx = np.pad(idx, (0, (h * w * ((c + 31) & ~31) + 31) // 32 * 32 - idx.size))
        mode, offset, y = 2, b.pool.emit(idx, "<i4"), NONE
    else:
        mode, offset, y = 0, 0, b.tid(index)
    b.emit(hx.GATHER_ELEMENTS, b.tid(x), y, node, [axis, 0, mode, offset])
    b.record(node, "gather")


def _index_plan(node, ep):
    """Classify aten.index.Tensor. ("rows", idx) is x[idx] on dim 0, which lowers to HX_GATHER;
    ("axis", dim, idx) is x[:, ..., idx] with one 1-D index on dim > 0; ("pixels", iy, ix) is
    x[:, :, iy, ix] on a batch-one [1, C, H, W] map with 1-D indices, an HX_GATHER_ND with tuple length 2."""
    x, indices = node.args[0], node.args[1]
    xs = static_shape(x)
    positions = [i for i, v in enumerate(indices) if v is not None]
    if not positions or any(not isinstance(indices[i], torch.fx.Node) for i in positions):
        raise Unsupported(f"{node.name}: unsupported index list")
    for i in positions:
        if _dtype_of(indices[i]) not in (torch.int32, torch.int64):
            raise Unsupported(f"{node.name}: boolean or float index tensors are unsupported")
    if len(indices) > len(xs):
        raise Unsupported(f"{node.name}: too many indices")
    if len(positions) == 1:
        d, idx = positions[0], indices[positions[0]]
        if d == 0:
            return ("rows", idx)
        if len(static_shape(idx)) != 1:
            raise Unsupported(f"{node.name}: an index on dimension {d} must be one-dimensional")
        return ("axis", d, idx)
    if (len(xs) == 4 and xs[0] == 1 and xs[3] > 1 and positions == [2, 3] and len(indices) == 4
            and all(len(static_shape(indices[i])) == 1 for i in positions)
            and static_shape(indices[2]) == static_shape(indices[3])
            and _dtype_of(node) == torch.float32 and _dtype_of(x) == torch.float32
            and _is_constant(ep, indices[2]) == _is_constant(ep, indices[3])
            and not _is_constant(ep, x)):
        m = static_shape(indices[2])[0]
        if static_shape(node) != (1, xs[1], m):
            raise Unsupported(f"{node.name}: unexpected output shape for a pixel gather")
        hwc_of((m, xs[1]), node.name)
        return ("pixels", indices[2], indices[3])
    raise Unsupported(f"{node.name}: unsupported advanced indexing")


def _emit_index_pixels(b, ep, node, iy, ix):
    """x[:, :, iy, ix] as HX_GATHER_ND over (h, w) pairs (tuple length 2, slice = the channel vector) into
    [M, 1, C], then a TRANSPOSE to the [1, C, M] result."""
    x = node.args[0]
    c, m = static_shape(x)[1], static_shape(iy)[0]
    if _is_constant(ep, iy):
        pairs = np.stack([np.asarray(_const_value(ep, iy)), np.asarray(_const_value(ep, ix))], axis=-1)
        mode, offset, y = 2, b.pool.emit(pairs.astype(np.int32).reshape(-1), "<i4"), NONE
    else:
        y = b.temp(node.name + "_pairs", (1, 2, 1, m))  # [1, M, 2]: (iy, ix) per row
        b.emit_to(hx.CONCAT, b.tid(iy), b.tid(ix), y, [2])
        mode, offset = 0, 0
    rows = b.temp(node.name + "_rows", (m, c))  # [M, 1, C]
    b.emit_to(hx.GATHER_ND, b.tid(x), y, rows, [0, 0, mode, offset, 2, c])
    b.emit_to(hx.TRANSPOSE, rows, NONE, b.tid(node), [2, 1, 0])
    b.record(node, "index")


def _scatter_plan(node, ep, data=False):
    """Row scatter into a token tensor [L, C] or batch-one [1, L, C]: (x, indices, updates, reduction,
    constant index array or None, constant update array or None, row count M).

    index_put(x, [idx], values, accumulate) with a one-dimensional index on the token dimension,
    select_scatter(x, src, dim, index) with a constant row, and slice_scatter(x, src, dim, start, end)
    with a unit step. Each is HX_SCATTER_ND with tuple length 1 and the channel vector as the slice."""
    name = op_name(node.target)
    x = node.args[0]
    xs = static_shape(x)
    if not isinstance(x, torch.fx.Node) or _is_constant(ep, x) or _dtype_of(x) != torch.float32:
        raise Unsupported(f"{node.name}: scatter needs a runtime float32 destination")
    if len(xs) not in (2, 3) or (len(xs) == 3 and xs[0] != 1):
        raise Unsupported(f"{node.name}: scatter needs a token tensor [L, C] or [1, L, C]")
    tdim = len(xs) - 2
    length, channels = xs[tdim], xs[-1]
    hwc_of(xs, x.name)
    reduction, idx_arr = 0, None
    if name == "index_put.default":
        indices, values = node.args[1], node.args[2]
        accumulate = bool(node.args[3]) if len(node.args) > 3 else bool(node.kwargs.get("accumulate", False))
        if (len(indices) != tdim + 1 or any(i is not None for i in indices[:tdim])
                or not isinstance(indices[tdim], torch.fx.Node)):
            raise Unsupported(f"{node.name}: index_put must index the token dimension with one tensor")
        idx = indices[tdim]
        if _dtype_of(idx) not in (torch.int32, torch.int64) or len(static_shape(idx)) != 1:
            raise Unsupported(f"{node.name}: the row index must be a one-dimensional integer tensor")
        m = static_shape(idx)[0]
        want = (m, channels) if len(xs) == 2 else (1, m, channels)  # a rank-3 destination may get [M, C]
        if (not isinstance(values, torch.fx.Node) or _dtype_of(values) != torch.float32
                or static_shape(values) not in (want, (m, channels))):
            raise Unsupported(f"{node.name}: values must be a float32 tensor of the gathered shape {want}")
        reduction = 1 if accumulate else 0
        if _is_constant(ep, idx):
            idx_arr = (np.asarray(_const_value(ep, idx)).astype(np.int32).reshape(-1) if data
                       else np.zeros(m, np.int32))
        updates = values
    elif name == "select_scatter.default":
        src, dim, index = node.args[1], int(node.args[2]), int(node.args[3])
        dim = dim + len(xs) if dim < 0 else dim
        if dim != tdim:
            raise Unsupported(f"{node.name}: select_scatter must select along the token dimension")
        index = index + length if index < 0 else index
        want = (channels,) if len(xs) == 2 else (1, channels)
        if not isinstance(src, torch.fx.Node) or static_shape(src) != want or _dtype_of(src) != torch.float32:
            raise Unsupported(f"{node.name}: src must be a float32 tensor of shape {want}")
        idx_arr, m, updates = np.asarray([index], np.int32), 1, src
    elif name == "slice_scatter.default":
        src = node.args[1]
        dim = int(node.args[2]) if len(node.args) > 2 else int(node.kwargs.get("dim", 0))
        start = node.args[3] if len(node.args) > 3 else node.kwargs.get("start")
        end = node.args[4] if len(node.args) > 4 else node.kwargs.get("end")
        step = node.args[5] if len(node.args) > 5 else node.kwargs.get("step", 1)
        dim = dim + len(xs) if dim < 0 else dim
        if dim != tdim or step != 1:
            raise Unsupported(f"{node.name}: slice_scatter must slice the token dimension with step 1")
        start = 0 if start is None else int(start)
        end = length if end is None else min(int(end), length)
        start = start + length if start < 0 else start
        m = end - start
        want = (m, channels) if len(xs) == 2 else (1, m, channels)
        if m < 1 or not isinstance(src, torch.fx.Node) or static_shape(src) != want or _dtype_of(src) != torch.float32:
            raise Unsupported(f"{node.name}: src must be a float32 tensor of shape {want}")
        idx_arr, updates = np.arange(start, end, dtype=np.int32), src
    else:  # pragma: no cover - the table and this function are kept in step
        raise Unsupported(name)
    if static_shape(node) != xs or _dtype_of(node) != torch.float32:
        raise Unsupported(f"{node.name}: scatter result must match the destination")
    upd_const = None
    if _is_constant(ep, updates):
        upd_const = np.asarray(_const_value(ep, updates), np.float32).reshape(m, channels) if data else np.zeros(1)
    idx_node = idx if name == "index_put.default" and idx_arr is None else None
    return x, idx_node, updates, reduction, idx_arr, upd_const, m


def _emit_scatter_rows(b, ep, node):
    x, idx_node, updates, reduction, idx_arr, upd_const, m = _scatter_plan(node, ep, data=True)
    xs = static_shape(x)
    channels = xs[-1]
    flags, p6, p7, y, z = 0, 0, 0, NONE, NONE
    if idx_arr is not None:
        flags |= 1
        p6 = b.pool.emit(idx_arr.reshape(-1), "<i4")
    else:
        y = b.tid(idx_node)
    if upd_const is not None:
        flags |= 2
        p7 = b.pool.emit(np.asarray(upd_const, np.float32).reshape(-1))
    else:
        z = b.tid(updates)
        if hwc_of(static_shape(updates), updates.name) != [m, 1, channels]:  # a lone row [C] or [1, C]
            row = b.temp(node.name + "_rows", (m, channels))
            b.emit_to(hx.RESHAPE, z, NONE, row, [0])
            z = row
    b.emit_to(hx.SCATTER_ND, b.tid(x), y, b.tid(node), [0, 0, 1, channels, reduction, flags, p6, p7, m], z=z)
    b.record(node, op_name(node.target))


# ---- where with a constant condition (masked_fill with a constant mask)

def _const_where_plan(node, ep, data=False):
    """where(cond, a, b) with a constant bool cond, one runtime float tensor and one scalar:
    (tensor, masked where-cond-selects-scalar bool array or None, scalar value or None).

    It lowers as tensor * keep + fill with two per-element constants (and folds into a softmax as a causal
    mask when the masked elements are the strictly upper triangle and the scalar is at most -100)."""
    cond, a, b_ = node.args[0], node.args[1], node.args[2]
    out = static_shape(node)
    if _dtype_of(node) != torch.float32 or _dtype_of(cond) != torch.bool or len(out) not in (2, 3, 4):
        raise Unsupported(f"{node.name}: constant-condition where needs float32 data and a bool mask")
    hwc_of(out, node.name)
    tensor_first = isinstance(a, torch.fx.Node) and not _is_constant(ep, a)
    tensor = a if tensor_first else b_
    other = b_ if tensor_first else a
    if not isinstance(tensor, torch.fx.Node) or _is_constant(ep, tensor) or static_shape(tensor) != out:
        raise Unsupported(f"{node.name}: one branch must be a runtime tensor of the output's shape")
    scalar = _constant_operand(ep, other, data)
    if scalar is None or any(d != 1 for d in scalar[0]):
        raise Unsupported(f"{node.name}: the other branch must be a scalar constant")
    try:
        if np.broadcast_shapes(tuple(static_shape(cond)), out) != out:
            raise Unsupported(f"{node.name}: the mask must broadcast to the output")
    except ValueError:
        raise Unsupported(f"{node.name}: the mask must broadcast to the output") from None
    if not data:
        return tensor, None, None
    mask = np.broadcast_to(np.asarray(_const_value(ep, cond)).astype(bool), out)
    masked = mask if not tensor_first else ~mask  # elements that take the scalar
    return tensor, masked, float(np.asarray(scalar[1]).reshape(-1)[0])


def _hwc_plane(full, out):
    """`full` (PyTorch order, shape `out`) as the padded per-pixel constant HXFG binary operations read."""
    h, w, c = hwc_of(out)
    stride = (c + 31) & ~31
    plane = np.zeros((h, w, stride), np.float32)
    plane[:, :, :c] = _hwc_array(np.asarray(full, np.float32), out)
    return plane.reshape(-1)


def _emit_const_where(b, ep, node):
    tensor, masked, value = _const_where_plan(node, ep, data=True)
    out = static_shape(node)
    keep = np.where(masked, 0.0, 1.0).astype(np.float32)
    fill = np.where(masked, np.float32(value), np.float32(0.0)).astype(np.float32)
    scaled = b.temp(node.name + "_keep", out)
    b.emit_to(hx.BINARY, b.tid(tensor), NONE, scaled, [hx.BIN_MUL, 2, _pool_vec(b, _hwc_plane(keep, out)), 0])
    b.emit(hx.BINARY, scaled, NONE, node, [hx.BIN_ADD, 2, _pool_vec(b, _hwc_plane(fill, out)), 0])
    b.record(node, "where")


# ---- softmax over any axis, with an optional causal mask folded from a constant additive mask

def is_causal_mask(values):
    """True for an additive mask that is zero on and below the diagonal and at most -100 above it."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 2:
        return False
    h, w = arr.shape[-2], arr.shape[-1]
    if h != w or h < 2:
        return False
    upper = np.triu(np.ones((h, w), bool), 1)
    return bool(np.all(np.abs(arr[..., ~upper]) <= 1e-3) and np.all(arr[..., upper] <= -100.0))


def _softmax_dim(node):
    """Normalised softmax dimension, refusing a dtype-changing softmax."""
    x = node.args[0]
    ishape = static_shape(x)
    dim = int(node.args[1])
    dim = dim + len(ishape) if dim < 0 else dim
    cast = node.args[2] if len(node.args) > 2 else None
    cast = bool(cast) if op_name(node.target) == "_softmax.default" else cast not in (None, torch.float32)
    if cast or static_shape(node) != ishape or not 0 <= dim < len(ishape):
        raise Unsupported(f"{node.name}: only same-dtype softmax")
    return dim


def _upper_triangle(masked, shape):
    """True when `masked` (broadcast to `shape`) is exactly the strictly-upper-triangular positions."""
    h, w = shape[-2], shape[-1]
    return h == w and h >= 2 and bool(np.array_equal(masked, np.broadcast_to(np.triu(np.ones((h, w), bool), 1), shape)))


def _causal_add_scores(add, ep):
    """The scores operand when `add` (an additive causal mask, or a where that fills the strictly upper
    triangle with a value of at most -100, i.e. masked_fill) only feeds a softmax over the last dimension of
    a token tensor, else None. Reads the mask's values, so it only answers while lowering."""
    if not isinstance(add, torch.fx.Node) or add.op != "call_function" or len(add.users) != 1:
        return None
    if kind_name(next(iter(add.users))) != "softmax":
        return None
    name = op_name(add.target)
    out = static_shape(add)
    if len(out) not in (2, 3) or _dtype_of(add) != torch.float32 or out[-1] != out[-2]:
        return None
    try:
        if _softmax_dim(next(iter(add.users))) != len(out) - 1:
            return None
        if name == "where.self":
            if not _is_constant(ep, add.args[0]):
                return None
            tensor, masked, value = _const_where_plan(add, ep, data=True)
            return tensor if value <= -100.0 and _upper_triangle(masked, out) else None
        if name != "add.Tensor" or len(add.args) != 2 or add.kwargs.get("alpha", 1) != 1:
            return None
        for scores, mask in (add.args, add.args[::-1]):
            if (isinstance(scores, torch.fx.Node) and isinstance(mask, torch.fx.Node)
                    and not _is_constant(ep, scores) and _is_constant(ep, mask) and static_shape(scores) == out
                    and static_shape(mask)[-2:] == out[-2:] and all(d == 1 for d in static_shape(mask)[:-2])
                    and is_causal_mask(_const_value(ep, mask))):
                return scores
    except (Unsupported, ValueError):  # not a mask this lowering folds
        return None
    return None


def _softmax_axis(shape, dim, multihead):
    """(HXFG axis, uses HX_ATTN_SOFTMAX) of a softmax over `dim` of a tensor in its token or multi-head
    layout. HX_SOFTMAX is the per-pixel channel softmax; any other axis needs HX_ATTN_SOFTMAX."""
    rank = len(shape)
    if rank == 4 and shape[0] == 1 and dim != 0:
        axis = {1: 0, 2: 1, 3: 2}[dim] if multihead else {1: 2, 2: 0, 3: 1}[dim]
    elif rank == 3:
        if dim == 2:
            axis = 2
        elif dim == 1 and shape[0] == 1:
            axis = 1 if multihead else 0
        else:
            raise Unsupported(f"softmax over dimension {dim} of {tuple(shape)}")
    elif rank == 2:
        axis = 2 if dim == 1 else 0
    else:
        raise Unsupported(f"softmax over dimension {dim} of {tuple(shape)}")
    return axis


def _emit_softmax(b, ep, node):
    x = node.args[0]
    dim = _softmax_dim(node)
    shape = static_shape(x)
    scores = _causal_add_scores(x, ep)
    causal = scores is not None
    src = scores if causal else x
    mask = None
    if not causal:
        mask = _runtime_mask(b, ep, x, shape)
        if mask is not None:
            src, mask = mask
    multihead = b.tid(src) in b.multihead
    axis = _softmax_axis(shape, dim, multihead)
    if axis == 2 and not causal and mask is None:
        b.emit(hx.SOFTMAX, b.tid(src), NONE, node, [])
    else:
        length = shape[-2] if len(shape) >= 2 else 0
        b.emit(hx.ATTN_SOFTMAX, b.tid(src), b.tid(mask) if mask is not None else NONE, node,
               [axis, int(causal), length if causal else 0])
    b.record(node, "softmax")


def _runtime_mask(b, ep, add, shape):
    """(scores, mask) when `add` is a same-shape add of two runtime token tensors that only the softmax
    reads and the mask's layout broadcasts into the scores' (so HX_ATTN_SOFTMAX adds it), else None."""
    if (not isinstance(add, torch.fx.Node) or add.op != "call_function" or op_name(add.target) != "add.Tensor"
            or len(add.users) != 1 or len(add.args) != 2 or len(shape) not in (2, 3)
            or add.kwargs.get("alpha", 1) != 1 or _dtype_of(add) != torch.float32):
        return None
    scores, mask = add.args
    if not (isinstance(scores, torch.fx.Node) and isinstance(mask, torch.fx.Node)):
        return None
    for s, m in ((scores, mask), (mask, scores)):
        if _is_constant(ep, s) or _is_constant(ep, m) or static_shape(s) != tuple(shape):
            continue
        ms = static_shape(m)
        if len(ms) != len(shape) or any(d not in (1, e) for d, e in zip(ms, shape)):
            continue
        if b.tid(s) in b.multihead or b.tid(m) in b.multihead:
            continue
        sh, mh = hwc_of(tuple(shape)), hwc_of(ms)
        if mh[0] in (1, sh[0]) and mh[2] in (1, sh[2]):
            return s, m
    return None


def _fused_softmax_add(b, ep, add):
    """True when lowering folds `add` into its softmax (causal constant mask or runtime additive mask)."""
    if _causal_add_scores(add, ep) is not None:
        return True
    if not add.users or len(add.users) != 1:
        return False
    soft = next(iter(add.users))
    if kind_name(soft) != "softmax":
        return False
    try:
        found = _runtime_mask(b, ep, add, static_shape(add))
        return found is not None and _softmax_dim(soft) == len(static_shape(add)) - 1
    except Unsupported:
        return False


# ---- inference batch norm: scale and shift, folded into a preceding convolution when possible

def _bn_plan(node, ep, data=False):
    """(scale, shift, eps) of _native_batch_norm_legit_no_training over a batch-one [1, C, H, W] or [N, C]
    tensor; running statistics, weight and bias are constants."""
    x = node.args[0]
    ishape = static_shape(x)
    if len(ishape) not in (2, 4) or (len(ishape) == 4 and ishape[0] != 1):
        raise Unsupported(f"{node.name}: batch norm needs batch-one rank 4 or rank 2")
    c = hwc_of(ishape, x.name)[2]
    users = list(node.users)
    if not users or any(kind_name(u) != "getitem" or int(u.args[1]) != 0 for u in users) or len(users) != 1:
        raise Unsupported(f"{node.name}: only the normalised output is used")
    weight, bias, mean, var = node.args[1], node.args[2], node.args[3], node.args[4]
    eps = float(node.args[6]) if len(node.args) > 6 else float(node.kwargs.get("eps", 1e-5))
    for arg, what in ((mean, "running mean"), (var, "running variance")):
        _require_constant(ep, arg, what)
    for arg, what in ((weight, "weight"), (bias, "bias")):
        if arg is not None:
            _require_constant(ep, arg, what)
    if not data:
        return None, None, eps
    mean_v = np.asarray(_const_value(ep, mean), np.float64).reshape(-1)
    var_v = np.asarray(_const_value(ep, var), np.float64).reshape(-1)
    gamma = np.asarray(_const_value(ep, weight), np.float64).reshape(-1) if weight is not None else np.ones(c)
    beta = np.asarray(_const_value(ep, bias), np.float64).reshape(-1) if bias is not None else np.zeros(c)
    if not (mean_v.size == var_v.size == gamma.size == beta.size == c):
        raise Unsupported(f"{node.name}: batch norm statistics must have {c} channels")
    scale = gamma / np.sqrt(var_v + eps)
    return scale.astype(np.float32), (beta - mean_v * scale).astype(np.float32), eps


def _folding_bn(conv, ep):
    """The batch-norm node a convolution folds into, or None: its only user is an inference batch norm that
    normalises its output (a two-dimensional convolution, not transposed grouped-decomposed)."""
    if len(conv.users) != 1 or len(static_shape(conv.args[0])) != 4:
        return None
    bn = next(iter(conv.users))
    if kind_name(bn) != "bn" or bn.args[0] is not conv:
        return None
    try:
        _bn_plan(bn, ep)
    except Unsupported:
        return None
    return bn


def _index_select_check(x, dim, index, ep):
    ishape = static_shape(index)
    if len(ishape) != 1:
        raise Unsupported("index_select index must be 1D")
    xshape = static_shape(x)
    dim = dim + len(xshape) if dim < 0 else dim
    if (len(xshape) in (3, 4) and dim == 0) or (len(xshape) == 4 and dim not in (1, 2, 3)):
        raise Unsupported(f"index_select rank-{len(xshape)} dim={dim} unsupported")
    if len(xshape) == 4 and _is_constant(ep, x):
        raise Unsupported("index_select with constant rank-4 table unsupported")


def check(kind, node, ep):
    args = node.args
    if kind == "constcast" or (kind == "reshape" and op_name(node.target) in _CONST_PASSTHROUGH
                               and _is_constant(ep, node)):
        if not _is_constant(ep, node):
            raise Unsupported("cast of a runtime tensor")
        return  # folded into the consumer's constant operand
    if kind in ("reshape", "transpose") and token_view(node) is not None:
        hwc_of(static_shape(node.args[0]), node.args[0].name)
        hwc_of(static_shape(node), node.name)
        return
    if kind == "conv" and len(static_shape(args[0])) == 3:
        _conv1d_check(node, ep)
    elif kind == "conv":
        x, w = args[0], args[1]
        out = static_shape(node)
        ishape = static_shape(x)
        transposed = _transposed(node)
        if transposed and _dilation(node) != [1, 1]:
            raise Unsupported("dilated transposed convolution")
        if not transposed and any(not 1 <= d <= 8 for d in _dilation(node)):
            raise Unsupported("dilation must be between 1 and 8 on each axis")
        if not transposed and len(args) > 7 and any(int(v) for v in args[7]):
            raise Unsupported("output padding")
        groups = int(args[8]) if len(args) > 8 else 1
        wshape = static_shape(w)
        if len(wshape) != 4:
            raise Unsupported("convolution weight must be rank 4")
        if transposed:  # weight [in, out, kh, kw]; geometry checked by the lowering plan
            if groups != 1:
                raise Unsupported("grouped transposed convolution")
            if wshape[0] != ishape[1]:
                raise Unsupported("input channel mismatch")
            _transposed_plan(node)
        else:
            depthwise = groups == wshape[0] == ishape[1] and wshape[1] == 1 and groups > 1
            if groups != 1 and not depthwise:
                if ishape[1] % groups != 0 or wshape[0] % groups != 0 or wshape[1] != ishape[1] // groups:
                    raise Unsupported("grouped convolution channels mismatch")
            elif groups == 1 and wshape[1] != ishape[1]:
                raise Unsupported("input channel mismatch")
        _require_constant(ep, w, "convolution weight")
        if len(args) > 2 and args[2] is not None:
            _require_constant(ep, args[2], "convolution bias")
        hwc_of(ishape, x.name)
        hwc_of(out, node.name)
    elif kind == "linear":
        x, w = args[0], args[1]
        xs = static_shape(x)
        if not (len(xs) == 2 or (len(xs) == 3 and xs[0] == 1)):
            raise Unsupported("linear input must be rank 2 or batch-one rank 3")
        _require_constant(ep, w, "linear weight")
        if len(args) > 2 and args[2] is not None:
            _require_constant(ep, args[2], "linear bias")
        hwc_of(static_shape(node), node.name)
    elif kind == "addmm":
        bias, x, w = args[0], args[1], args[2]
        if len(static_shape(x)) != 2:
            raise Unsupported("addmm input must be rank 2")
        _require_constant(ep, w, "addmm weight")
        if bias is not None:
            _require_constant(ep, bias, "addmm bias")
        hwc_of(static_shape(node), node.name)
    elif kind == "slice":
        x = args[0]
        ishape = static_shape(x)
        dim = int(args[1]) if len(args) > 1 and args[1] is not None else 0
        _hwc_axis(ishape, dim, x.name)  # rank 2, batch-one rank 3 or rank 4; never the batch dimension
        step = int(args[4]) if len(args) > 4 and args[4] is not None else 1
        if step != 1:
            raise Unsupported("slice step must be 1")
        hwc_of(ishape, x.name)
        hwc_of(static_shape(node), node.name)
    elif kind == "split":
        x = args[0]
        ishape = static_shape(x)
        if len(ishape) not in (2, 4):
            raise Unsupported("split input must be rank 2 or 4")
        split_sizes = args[1]
        dim = int(args[2]) if len(args) > 2 and args[2] is not None else 0
        dim = dim + len(ishape) if dim < 0 else dim
        if dim == 0:
            raise Unsupported("splitting batch dimension is unsupported")
        if isinstance(split_sizes, int):
            num_splits = (ishape[dim] + split_sizes - 1) // split_sizes
            sizes = [min(split_sizes, ishape[dim] - i * split_sizes) for i in range(num_splits)]
        else:
            sizes = [int(s) for s in split_sizes]
        if sum(sizes) != ishape[dim]:
            raise Unsupported("split sizes must sum to dim size")
        hwc_of(ishape, x.name)
        for u in node.users:
            if kind_name(u) != "getitem":
                raise Unsupported("split user must be getitem")
            idx = int(u.args[1])
            if idx < 0 or idx >= len(sizes):
                raise Unsupported("split getitem index out of bounds")
    elif kind == "embedding":
        w, indices = args[0], args[1]
        wshape = static_shape(w)
        if len(wshape) != 2:
            raise Unsupported("embedding weight must be rank 2")
        _ = static_shape(indices)
        _ = static_shape(node)
    elif kind == "index_select":
        _index_select_check(args[0], int(args[1]), args[2], ep)
        _ = static_shape(node)
    elif kind == "index":
        x = args[0]
        _ = static_shape(x)
        _ = static_shape(node)
        plan = _index_plan(node, ep)
        if plan[0] == "axis":
            _index_select_check(x, plan[1], plan[2], ep)
    elif kind == "cast":
        _cast_plan(node, ep)
    elif kind == "gather_elements":
        _gather_plan(node, ep)
    elif kind == "scatter_rows":
        _scatter_plan(node, ep)
    elif kind == "bn":
        _bn_plan(node, ep)
    elif kind in ("add", "binary"):
        _binary_plan(node, ep)
    elif kind == "act":
        hwc_of(static_shape(args[0]), args[0].name)
        hwc_of(static_shape(node), node.name)
        if op_name(node.target) == "bitwise_not.default" and not (
                isinstance(args[0], torch.fx.Node) and _bool_valued(args[0])):
            raise Unsupported("bitwise_not is supported on bool tensors only (logical NOT)")
        if op_name(node.target) == "gelu.default" and (
            node.kwargs.get("approximate", "none") != "none"
        ):
            raise Unsupported("only exact (erf) GELU")
        ACT_FUNCTIONS[op_name(node.target)](node)  # rejects parameters HXFG cannot express
    elif kind == "avgpool":
        x = args[0]
        k = _ints(args[1], 2)
        s = _ints(args[2], 2) if len(args) > 2 and args[2] else k
        p = _ints(args[3], 2) if len(args) > 3 else [0, 0]
        ceil = len(args) > 4 and args[4]
        divisor = args[6] if len(args) > 6 else None
        if ceil or divisor is not None:
            raise Unsupported("ceil-mode or divisor-override average pooling")
        ih, iw, _ = hwc_of(static_shape(x), x.name)
        if len(static_shape(x)) != 4 or p[0] > k[0] // 2 or p[1] > k[1] // 2:
            raise Unsupported("average pooling padding or rank")
        if max(k) > 15 or max(s) > 8:
            raise Unsupported("average pooling window or stride too large")
        expect = ((ih + 2 * p[0] - k[0]) // s[0] + 1, (iw + 2 * p[1] - k[1]) // s[1] + 1)
        oshape = static_shape(node)
        if tuple(oshape[2:]) != expect or oshape[:2] != (1, static_shape(x)[1]):
            raise Unsupported("average pooling output shape")
    elif kind == "gap":
        x = args[0]
        ishape, out = static_shape(x), static_shape(node)
        if len(ishape) != 4 or [int(v) for v in args[1]] != [1, 1] or out != (1, ishape[1], 1, 1):
            raise Unsupported("only global adaptive average pooling")
        hwc_of(ishape, x.name)
    elif kind == "prelu":
        x, slope = args[0], args[1]
        c = hwc_of(static_shape(x), x.name)[2]
        _require_constant(ep, slope, "PReLU slope")
        if static_shape(slope) not in ((1,), (c,)):
            raise Unsupported("PReLU slope must be per channel or scalar")
    elif kind == "maxpool":
        x = args[0]
        users = list(node.users)
        if len(users) != 1 or kind_name(users[0]) != "getitem" or users[0].args[1] != 0:
            raise Unsupported("max pooling indices are used")
        k = _ints(args[1], 2)
        s = _ints(args[2], 2) if len(args) > 2 and args[2] else k
        p = _ints(args[3], 2) if len(args) > 3 else [0, 0]
        d = _ints(args[4], 2) if len(args) > 4 else [1, 1]
        if d != [1, 1] or (len(args) > 5 and args[5]):
            raise Unsupported("dilated or ceil-mode max pooling")
        ih, iw, _ = hwc_of(static_shape(x), x.name)
        oshape = static_shape(users[0])
        if p[0] > k[0] // 2 or p[1] > k[1] // 2:
            raise Unsupported("max pooling padding above half the kernel")
        expect = ((ih + 2 * p[0] - k[0]) // s[0] + 1, (iw + 2 * p[1] - k[1]) // s[1] + 1)
        if tuple(oshape[2:]) != expect or oshape[:2] != (1, static_shape(x)[1]):
            raise Unsupported("max pooling output shape")
    elif kind == "norm":
        _norm_plan(node, ep)
    elif kind == "rmsnorm":
        _rms_norm_plan(node, ep)
    elif kind == "getitem":
        prod = args[0]
        if not (
            isinstance(prod, torch.fx.Node)
            and (
                (op_name(prod.target) in (
                    "max_pool2d_with_indices.default", "native_layer_norm.default",
                    "native_group_norm.default", "_native_batch_norm_legit.no_stats",
                    "_native_batch_norm_legit.default"
                ) and args[1] == 0)
                or (op_name(prod.target) in (
                    "split_with_sizes_copy.default", "split_with_sizes.default",
                    "split_copy.Tensor", "split.Tensor"
                ))
                or (op_name(prod.target) in ("topk.default", "max.dim", "min.dim") and args[1] in (0, 1))
                or (op_name(prod.target) == "_native_batch_norm_legit_no_training.default" and args[1] == 0)
            )
        ):
            raise Unsupported("only the value output of max pooling, normalisation or split is supported")
    elif kind == "pad":
        x, pads = args[0], [int(v) for v in args[1]]
        value = args[2] if len(args) > 2 else 0
        if value != 0:
            raise Unsupported("non-zero pad value")
        ishape = static_shape(x)
        if len(ishape) != 4 or len(pads) != 4 or min(pads) < 0:
            raise Unsupported("only non-negative spatial pads on rank 4")
        wl, wr, ht, hb = pads
        out = static_shape(node)
        if out != (1, ishape[1], ishape[2] + ht + hb, ishape[3] + wl + wr):
            raise Unsupported("pad shape mismatch")
        hwc_of(ishape, x.name)
    elif kind == "resize":
        x = args[0]
        ishape, out = static_shape(x), static_shape(node)
        if len(ishape) != 4 or len(out) != 4 or ishape[1] != out[1]:
            raise Unsupported("resize must keep rank 4 and channels")
        if _is_nearest(node):
            # HX_RESIZE nearest maps out index to floor(out * in / out); a given scale must agree.
            if op_name(node.target).endswith(".vec"):
                scales = list(args[2]) if len(args) > 2 and args[2] is not None else []
            else:
                scales = list(args[2:4])
            for s, n_in, n_out in zip(scales, ishape[2:4], out[2:4]):
                if s is not None and abs(float(s) - n_out / n_in) > 1e-6:
                    raise Unsupported("nearest resize scale must equal the output/input ratio")
        hwc_of(out, node.name)
    elif kind == "reshape":
        x = args[0]
        ishape, out = static_shape(x), static_shape(node)
        src = hwc_of(ishape, x.name)
        hwc_of(out, node.name)
        # A spatial source is flattened in NCHW order into a temporary of N channels.
        if (src[:2] != [1, 1] and src[0] * src[1] * src[2] > MAX_CHANNELS
                and src != hwc_of(out, node.name) and not _rowmajor_reshape(node)
                and not _channel_preserving(ishape, out)):
            raise Unsupported("reshape flattens more elements than the channel limit")
    elif kind == "transpose":
        x = args[0]
        ishape, out = static_shape(x), static_shape(node)
        hwc_of(ishape, x.name)
        hwc_of(out, node.name)
        if len(ishape) in (2, 3):
            try:
                _token_transpose_p(node, False, False)  # rank 2 / batch-one rank 3: a swap of two token axes
            except Unsupported:
                # An attention tensor [B, L, C] with a batch above one (the key transpose of a bmm).
                if not (len(ishape) == 3 and node.users
                        and all(kind_name(u) == "bmm" for u in node.users)):
                    raise
                _token_transpose_p(node, True, True)
        elif _transpose_p(node) is None:
            raise Unsupported("permute must be rank 4, batch-one, with perm (0, a, b, c)")
        if tuple(ishape[i] for i in _perm(node)) != out:
            raise Unsupported("permute output shape disagrees with its perm")
    elif kind == "concat":
        tensors = args[0]
        if len(tensors) < 2 or not all(isinstance(t, torch.fx.Node) and not _is_constant(ep, t) for t in tensors):
            raise Unsupported("concatenation needs at least two runtime inputs")
        out = static_shape(node)
        dim = int(args[1]) if len(args) > 1 else 0
        dim = dim + len(out) if dim < 0 else dim
        if (len(out) not in (2, 3, 4) or (len(out) == 2 and dim != 1) or (len(out) == 4 and dim == 0)
                or (len(out) == 3 and (out[0] != 1 or dim == 0))):
            raise Unsupported("concatenation along an unsupported dimension")
        axis = _concat_axis(len(out), dim)
        shapes = [static_shape(t) for t in tensors]
        hwc = [hwc_of(s, t.name) for s, t in zip(shapes, tensors)]
        ohwc = hwc_of(out, node.name)
        for i in range(3):
            if i != axis and not all(h[i] == ohwc[i] for h in hwc):
                raise Unsupported("concatenation shape mismatch")
        if sum(h[axis] for h in hwc) != ohwc[axis]:
            raise Unsupported("concatenation size mismatch")
    elif kind in ("mean", "reduce"):
        _reduce_plan(node)
    elif kind == "logistic":
        hwc_of(static_shape(args[0]), args[0].name)
        hwc_of(static_shape(node), node.name)
    elif kind == "softmax":
        # _softmax(x, dim, half_to_float) and softmax.int(x, dim, dtype): only same-dtype softmax. Any dim of a
        # rank 2, rank 3 (batch-one for the token axis) or batch-one rank-4 tensor; the channel axis is
        # HX_SOFTMAX, the others HX_ATTN_SOFTMAX.
        x = args[0]
        ishape = static_shape(x)
        hwc_of(ishape, x.name)
        hwc_of(static_shape(node), node.name)
        dim = _softmax_dim(node)
        if len(ishape) not in (2, 3, 4) or (len(ishape) == 4 and dim == 0):
            raise Unsupported("softmax over the batch dimension or of an unsupported rank")
        _softmax_axis(ishape, dim, False)
    elif kind == "activation":
        if act_code(node) is None:  # a general clamp or hardtanh: MAX then MIN with the bounds
            _clamp_plan(node, ep)
        hwc_of(static_shape(node), node.name)
    elif kind == "where":
        if isinstance(args[0], torch.fx.Node) and _is_constant(ep, args[0]):
            _const_where_plan(node, ep)
        else:
            _where_plan(node, ep)
    elif kind == "ne":
        _binary_plan(node, ep)
    elif kind in ("argmax", "argmin"):
        _argreduce_plan(node, ep)
    elif kind == "topk":
        _topk_plan(node, ep)
    elif kind == "maxdim":
        _maxdim_plan(node, ep)
    elif kind == "tile":
        _tile_plan(node, ep)
    elif kind == "rsqrt":
        _require_runtime(ep, args[0], node.name)
        hwc_of(static_shape(node), node.name)
    elif kind == "logsigmoid":
        _require_runtime(ep, args[0], node.name)
        if static_shape(node) != static_shape(args[0]):
            raise Unsupported(f"{node.name}: output must match the input shape")
        hwc_of(static_shape(args[0]), args[0].name)
    elif kind == "var":
        _var_plan(node, ep)
    elif kind == "constfactory":
        if not _is_constant(ep, node):
            raise Unsupported(f"{node.name}: the factory's users are not all constant readers")
        _const_plan(ep, node)
    elif kind == "bmm":
        a, b_arg = args[0], args[1]
        shape_a = static_shape(a)
        shape_b = static_shape(b_arg)
        if len(shape_a) not in (3, 4) or len(shape_b) not in (3, 4):
            raise Unsupported("bmm operands must be rank 3 or 4")
        if shape_a[-1] != shape_b[-2]:
            raise Unsupported(f"bmm inner dimensions disagree: {shape_a[-1]} vs {shape_b[-2]}")
        if shape_a[0] != shape_b[0]:
            raise Unsupported(f"bmm batch dimensions disagree: {shape_a[0]} vs {shape_b[0]}")
        hwc_of(shape_a, a.name, multihead=True)
        hwc_of(shape_b, b_arg.name, multihead=True)
        hwc_of(static_shape(node), node.name, multihead=True)
    elif kind == "sdpa":
        q, k, v = args[0], args[1], args[2]
        shape_q, shape_k, shape_v = static_shape(q), static_shape(k), static_shape(v)
        if len(shape_q) != 4 or len(shape_k) != 4 or len(shape_v) != 4:
            raise Unsupported("sdpa operands must be rank 4 [B, H, L, D]")
        if shape_q[:2] != shape_k[:2] or shape_q[:2] != shape_v[:2]:
            raise Unsupported("sdpa batch/head dimensions disagree")
        if shape_q[3] != shape_k[3] or shape_k[2] != shape_v[2]:
            raise Unsupported("sdpa dimension mismatch")
        dropout_p = args[4] if len(args) > 4 and args[4] is not None else 0.0
        if float(dropout_p) != 0.0:
            raise Unsupported("sdpa dropout is unsupported at inference")
        hwc_of(shape_q, q.name, multihead=True)
        hwc_of(shape_k, k.name, multihead=True)
        hwc_of(shape_v, v.name, multihead=True)
        hwc_of(static_shape(node), node.name, multihead=True)
    elif kind in ("lfm_linear_q", "lfm_rope", "lfm_shortconv", "lfm_kv_write"):
        _check_lfm(kind, node, ep)
    else:  # pragma: no cover - table and checks are kept in step
        raise Unsupported(kind)


def _lfm_token(node, arg, what, rank4=False):
    """(L, C) of a batch-one token operand [1, L, C] (or [L, C])."""
    shape = static_shape(arg)
    if len(shape) == 3 and shape[0] == 1:
        return shape[1], shape[2]
    if len(shape) == 2:
        return shape[0], shape[1]
    raise Unsupported(f"{node.name}: {what} must be a batch-one [1, L, C] token tensor, got {shape}")


def _lfm_string(node, index, what):
    value = node.args[index] if len(node.args) > index else None
    if not isinstance(value, str) or not value:
        raise Unsupported(f"{node.name}: {what} must be a non-empty string")
    return value


def _lfm_position(node, ep, arg, length):
    """The runtime position operand of lfm_rope / lfm_kv_write: a float32 [1, 1] graph input."""
    if arg is None:
        return None
    if _is_constant(ep, arg):
        raise Unsupported(f"{node.name}: pass a constant position as `start`, not a tensor")
    if length != 1 or static_shape(arg) not in ((1, 1), (1,)):
        raise Unsupported(f"{node.name}: a runtime position is one float32 [1, 1] value for one token")
    if arg.meta["val"].dtype != torch.float32:
        raise Unsupported(f"{node.name}: a runtime position must be float32 (delegate inputs are float32)")
    return arg


def _check_lfm(kind, node, ep):
    args = node.args
    x = args[0]
    length, width = _lfm_token(node, x, "input")
    if kind == "lfm_linear_q":
        blocks, bias, fmt = args[1], args[2], int(args[3])
        if fmt not in (0, 1):
            raise Unsupported(f"{node.name}: fmt must be 0 (Q4_0) or 1 (Q8_0)")
        _require_constant(ep, blocks, "quantized weight blocks")
        shape = tuple(int(d) for d in blocks.meta["val"].shape)  # uint8: not a float tensor
        n = shape[0]
        want = (n, width // 32, 18 if fmt == 0 else 34)
        if width % 32 or n % 32 or shape != want or blocks.meta["val"].dtype != torch.uint8:
            raise Unsupported(f"{node.name}: K and N must be multiples of 32 and the blocks uint8 {want}")
        if bias is not None:
            _require_constant(ep, bias, "quantized linear bias")
            if static_shape(bias) != (n,):
                raise Unsupported(f"{node.name}: bias must have N values")
    elif kind == "lfm_rope":
        pos, cos, sin, heads, rot, interleaved, start = args[1:8]
        if len(static_shape(x)) != 3:
            raise Unsupported(f"{node.name}: rope input must be rank 3 [1, L, heads * head_dim]")
        heads, rot, start = int(heads), int(rot), int(start)
        if heads <= 0 or width % heads:
            raise Unsupported(f"{node.name}: num_heads must divide the width")
        head_dim = width // heads
        rot = rot or head_dim
        if rot > head_dim or rot < 2 or rot % 2:
            raise Unsupported(f"{node.name}: rotary_dim must be even and at most the head size")
        _require_constant(ep, cos, "cos cache")
        _require_constant(ep, sin, "sin cache")
        # The table itself is verified when the data is read (lower_program); checks see fake tensors.
        if static_shape(cos) != static_shape(sin) or static_shape(cos)[1:] != (rot // 2,):
            raise Unsupported(f"{node.name}: cos and sin must be [max_position, rotary_dim / 2]")
        _lfm_position(node, ep, pos, length)
        if pos is None and (start < 0 or start + length > static_shape(cos)[0]):
            raise Unsupported(f"{node.name}: position outside the cos/sin cache")
    elif kind == "lfm_shortconv":
        weight, bias, gate = args[1], args[2], args[3]
        _lfm_string(node, 4, "state")
        activation = _lfm_string(node, 5, "activation")
        if activation not in ("none", "relu", "silu"):
            raise Unsupported(f"{node.name}: activation must be none, relu or silu")
        if length != 1:
            raise Unsupported(f"{node.name}: one token per run")
        _require_constant(ep, weight, "shortconv weight")
        wshape = static_shape(weight)
        if len(wshape) != 3 or wshape[0] != width or wshape[1] != 1 or not 2 <= wshape[2] <= 15:
            raise Unsupported(f"{node.name}: weight must be [C, 1, k] with k in 2..15")
        if bias is not None:
            _require_constant(ep, bias, "shortconv bias")
            if static_shape(bias) != (width,):
                raise Unsupported(f"{node.name}: bias must have C values")
        if gate is not None and static_shape(gate) != static_shape(x):
            raise Unsupported(f"{node.name}: gate must have the shape of x")
    else:  # lfm_kv_write
        pos, state, max_len, start = args[1], args[2], int(args[3]), int(args[4])
        _lfm_string(node, 2, "state")
        if len(static_shape(x)) != 3:
            raise Unsupported(f"{node.name}: kv_write input must be rank 3 [1, L, C]")
        _lfm_position(node, ep, pos, length)
        if length > max_len or (pos is None and not 0 <= start <= max_len - length):
            raise Unsupported(f"{node.name}: position outside the cache")
        if static_shape(node) != (1, 1, max_len, width):
            raise Unsupported(f"{node.name}: cache output must be [1, 1, max_len, C]")
        if not node.users or any(kind_name(u) not in ("sdpa", "bmm") for u in node.users):
            raise Unsupported(f"{node.name}: a cache may only be read by attention "
                              f"(users: {[op_name(u.target) if u.op == 'call_function' else u.op for u in node.users]})")
    hwc_of(static_shape(x), x.name)


# ---------------------------------------------------------------- partition selection


def select_nodes(ep):
    """Call nodes the partitioner should delegate, in graph order.

    A node is refused when one of its inputs is an unselected node that already
    depends on a selected node: delegating both would create a partition cycle
    (the CPU op would have to run inside the delegate). Returns [] when the
    partition is smaller than MIN_OPS operations.
    """
    chosen, done, activation_of = [], set(), {}
    selected = set()
    depends_on_selected = set()  # unselected or selected nodes with a selected ancestor
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        inputs = [a for a in node.all_input_nodes if a.op == "call_function"]
        if any(a not in selected and a in depends_on_selected for a in inputs):
            # Refused: keep tracking ancestry so downstream nodes are judged correctly.
            if any(a in selected or a in depends_on_selected for a in inputs):
                depends_on_selected.add(node)
            continue
        try:
            kind = kind_of(node, ep)
        except Unsupported:
            kind = None
        take = kind is not None
        if take and kind == "activation":
            prod = fusable_producer(node, done, activation_of, ep)
            if prod is not None:
                activation_of[prod] = act_code(node)
        if take and kind == "getitem":
            take = node.args[0] in done
        if take:
            selected.add(node)
            chosen.append(node)
            done.add(node)
        if any(a in selected or a in depends_on_selected for a in inputs):
            depends_on_selected.add(node)
    chosen = _prune_boundary(ep, chosen)
    if len(chosen) < MIN_OPS:
        return []
    return chosen


# Argument positions whose emitters read a constant operand through _constant or _const_value. Any other
# constant operand would need a tensor id that no operation produces, so the consumer stays on the CPU.
_CONST_ARGS = {
    "binary": (0, 1), "add": (0, 1), "ne": (0, 1), "where": (0, 1, 2), "conv": (1, 2), "linear": (1, 2),
    "addmm": (0, 2), "norm": (1, 2, 3, 4), "rmsnorm": (2,), "prelu": (1,), "embedding": (0,),
    "index_select": (0, 2), "index": (0, 1), "lfm_linear_q": (1, 2), "lfm_rope": (2, 3),
    "lfm_shortconv": (1, 2), "bn": (1, 2, 3, 4), "gather_elements": (2,), "scatter_rows": (1, 2),
}


def _reads_constant_args(ep, node):
    """False when the node is delegated and one of its constant operands has no constant reader."""
    if _is_constant(ep, node):
        return True  # a folded constant reads its own input as a value (see _const_value)
    allowed = _CONST_ARGS.get(kind_name(node), ())
    for index, arg in enumerate(node.args):
        for value in (arg if isinstance(arg, (list, tuple)) else [arg]):
            if isinstance(value, torch.fx.Node) and _is_constant(ep, value) and index not in allowed:
                return False
    return True


def _boundary_violation(ep, node, selected):
    """True when a selected node cannot be delegated with the rest of the selection."""
    if not _reads_constant_args(ep, node):
        return True
    if kind_name(node) in _MULTI_OUTPUT_KINDS:
        # Its live getitem outputs must all be delegated; an unused getitem needs nothing.
        if any(u not in selected and u.users for u in node.users):
            return True
    elif (_is_constant(ep, node) or _internal_result(node)) and (
            not node.users or any(u not in selected for u in node.users)):
        return True  # folded constants and bool/int results have no CPU-visible form
    if kind_name(node) == "constfactory" and any(
            a.op == "call_function" and a not in selected for a in node.all_input_nodes):
        return True  # full_like reads its input's shape: the input must be delegated too
    for arg in node.all_input_nodes:
        if arg.op != "call_function" or arg in selected:
            continue
        if _internal_result(arg) or (_is_constant(ep, arg) and kind_name(arg) in _FOLDED_KINDS):
            return True  # the delegate would read a CPU bool/int or an unfolded constant
    return False


def _prune_boundary(ep, chosen):
    """Drop nodes that cannot cross the delegate boundary (see _boundary_violation).

    A dropped node takes its selected descendants with it, so the delegated set stays convex: no CPU
    operation sits between two delegated ones. Repeats until no selected node violates a rule.
    """
    selected = set(chosen)
    changed = True
    while changed:
        changed = False
        for node in chosen:
            if node not in selected or not _boundary_violation(ep, node, selected):
                continue
            doomed, stack = {node}, [node]
            while stack:
                for user in stack.pop().users:
                    if user in selected and user not in doomed:
                        doomed.add(user)
                        stack.append(user)
            selected -= doomed
            changed = True
    return [n for n in chosen if n in selected]


# ---------------------------------------------------------------- lowering


class _Builder:
    def __init__(self, ep):
        self.ep = ep
        self.ids, self.alias, self.shape = {}, {}, {}
        self.sdpa_token = set()
        self.ops, self.op_of_out = [], {}
        self.pool = hx.ConstantPool(dedupe=True)  # repeated subnetworks share their weights
        self.source_ops = []
        self.multihead = set()
        self.multihead_names = set()
        self.persistent = {}  # tensor id -> state id (conv rings and KV caches)
        self.state_names = {}  # state id -> name
        self.dtypes = {}  # tensor id -> HX_DTYPE_INT32 for integer results (argmax, topk indices)
        self.folded_bn = set()  # batch-norm nodes folded into the convolution before them

    def state_id(self, node, name):
        sid = zlib.crc32(name.encode()) | 0x80000000
        if sid in self.state_names:
            raise Unsupported(f"{node.name}: state '{name}' is already used or collides with "
                              f"'{self.state_names[sid]}'")
        self.state_names[sid] = name
        return sid

    def canon(self, node):
        name = node.name if hasattr(node, "name") else str(node)
        while name in self.alias:
            name = self.alias[name]
        return name

    def tid(self, node):
        name = self.canon(node)
        if name not in self.ids:
            self.ids[name] = len(self.ids)
            self.shape[self.ids[name]] = static_shape(node)
            if name in self.multihead_names:
                self.multihead.add(self.ids[name])
        return self.ids[name]

    def temp(self, name, shape):
        """A synthetic internal tensor with no fx node; shape is NCHW like static_shape."""
        key = f"{name}#{len(self.ids)}"
        self.ids[key] = len(self.ids)
        self.shape[self.ids[key]] = tuple(int(d) for d in shape)
        return self.ids[key]

    def emit(self, kind, x, y, out_node, p):
        return self.emit_to(kind, x, y, self.tid(out_node), p)

    def emit_to(self, kind, x, y, out, p, z=NONE):
        p = (list(p) + [0] * 11)[:11]
        op = [kind, x, y, z, out] + p
        self.ops.append(op)
        self.op_of_out[out] = op
        return out

    def record(self, node, name):
        self.source_ops.append({"op": name, "output": node.name})


def _transpose_p(node):
    """HX_TRANSPOSE p for a batch-first rank-4 permute, or None. The ONNX-style axes (1=C, 2=H,
    3=W) map to HWC axes (H=0, W=1, C=2); output h, w and c read source axes perm[2], perm[3]
    and perm[1]. Matches op_Transpose in lower_onnx_float.py."""
    perm = _perm(node)
    if len(perm) != 4 or perm[0] != 0 or sorted(perm) != [0, 1, 2, 3]:
        return None
    hwc_axis = {1: 2, 2: 0, 3: 1}
    return [hwc_axis[perm[2]], hwc_axis[perm[3]], hwc_axis[perm[1]]]


def _token_transpose_p(node, src_multihead, dst_multihead):
    """HX_TRANSPOSE p for a permute of a rank-2 or batch-one rank-3 token tensor. Tokens are [L, 1, C] (rows
    L, channels C); an attention (multi-head) rank-3 tensor [B, L, C] is [B, L, C] (batch, rows, channels).
    Output axis k carries source axis p[k]; the batch axis of a token tensor must stay put."""
    ishape, perm = static_shape(node.args[0]), _perm(node)

    def axes(shape, multihead):  # PyTorch dim -> HXFG axis
        if len(shape) == 2:
            return {0: 0, 1: 2}
        if multihead:
            return {0: 0, 1: 1, 2: 2}
        if shape[0] != 1:
            raise Unsupported(f"{node.name}: permute of a token tensor needs batch one")
        return {1: 0, 2: 2}

    src, dst = axes(ishape, src_multihead), axes(static_shape(node), dst_multihead)
    p = [None, None, None]
    for out_dim, in_dim in enumerate(perm):
        if out_dim in dst:
            if in_dim not in src:  # the batch dimension (size one) lands on whichever axis is left
                if ishape[in_dim] != 1:
                    raise Unsupported(f"{node.name}: permute moves the batch dimension")
                continue
            p[dst[out_dim]] = src[in_dim]
    spare = [a for a in range(3) if a not in p]  # size-one axes pair up in order
    for k in range(3):
        if p[k] is None:
            p[k] = spare.pop(0)
    return p


def _emit_reshape(b, node, x):
    """view_copy. Equal HWC shapes, or 1x1 on both sides, keep HWC order (flag 0). Otherwise
    the element order is NCHW: flatten to 1x1xN (flag 1) and unflatten (flag 2), each skipped
    when its side is already 1x1xN. Rank-2 and rank-3 token tensors and attention (multi-head)
    tensors keep PyTorch's row-major order, which is the logical HWC order (flag 0), so a view
    between such a side and a rank-4 NCHW side flattens or unflattens only the NCHW side."""
    src = hwc_of(static_shape(x), x.name)
    dst = hwc_of(static_shape(node), node.name)
    src_rm = len(static_shape(x)) <= 3 or b.canon(x) in b.multihead_names
    dst_rm = len(static_shape(node)) <= 3 or b.canon(node) in b.multihead_names
    if (src == dst or (src[:2] == [1, 1] and dst[:2] == [1, 1]) or (src_rm and dst_rm)
            or _channel_preserving(static_shape(x), static_shape(node))):
        b.emit(hx.RESHAPE, b.tid(x), NONE, node, [0])
        return
    n = src[0] * src[1] * src[2]
    # Source side: the element sequence as a 1x1xN tensor (None when it already is one).
    source = b.tid(x)
    if src[:2] != [1, 1]:
        source = b.temp(node.name, [1, n, 1, 1])
        b.emit_to(hx.RESHAPE, b.tid(x), NONE, source, [0 if src_rm else 1])
    if dst[:2] == [1, 1]:
        b.emit_to(hx.RESHAPE, source, NONE, b.tid(node), [0])
    elif dst_rm:  # a row-major destination reads the sequence in logical HWC order
        b.emit_to(hx.RESHAPE, source, NONE, b.tid(node), [0])
    else:
        b.emit_to(hx.RESHAPE, source, NONE, b.tid(node), [2])


def _transposed(node):
    """aten.convolution's `transposed` flag (args[6])."""
    return len(node.args) > 6 and bool(node.args[6])


def _dilation(node):
    """aten.convolution's dilation (args[5]) as [dh, dw]."""
    return _ints(node.args[5], 2) if len(node.args) > 5 else [1, 1]


def _compact_window(users):
    """A one-channel input is compact (stride 1) only for the DSP's compact-image convolution: a
    single square 3x3 or 5x5 dense convolution with equal strides of at most 2. Any other consumer
    would make the blocked FP16 plan refuse the whole graph."""
    if len(users) != 1:
        return False
    u = users[0]
    k = list(u.args[1].meta["val"].shape[2:])
    s = _ints(u.args[3], 2)
    groups = u.args[8] if len(u.args) > 8 else 1
    return groups == 1 and k[0] == k[1] and k[0] in (3, 5) and s[0] == s[1] and s[0] <= 2


def _dilated_z(dilation):
    """HXFG z for a convolution: NONE for dilation [1, 1], else (dh << 8) | dw."""
    return NONE if list(dilation) == [1, 1] else (dilation[0] << 8) | dilation[1]


def _transposed_plan(node):
    """Transposed convolution as zero insertion plus a dense stride-1 convolution (the same
    derivation as ConvTranspose in lower_onnx_float.py). Returns (stride, leading zeros,
    zero-inserted (H, W), kernel) and checks the output size."""
    x, w = node.args[0], node.args[1]
    ih, iw = static_shape(x)[2], static_shape(x)[3]
    kh, kw = static_shape(w)[2], static_shape(w)[3]
    stride, pad = _ints(node.args[3], 2), _ints(node.args[4], 2)
    op = _ints(node.args[7], 2)
    if min(stride) < 1 or max(stride) > 8 or max(kh, kw) > 15:
        raise Unsupported(f"{node.name}: transposed stride above 8 or kernel above 15")
    zt, zl = kh - 1 - pad[0], kw - 1 - pad[1]  # the padding is symmetric in ATen
    if zt < 0 or zl < 0:
        raise Unsupported(f"{node.name}: padding exceeds the kernel size minus one")
    expect = [(ih - 1) * stride[0] + kh - 2 * pad[0] + op[0],
              (iw - 1) * stride[1] + kw - 2 * pad[1] + op[1]]
    if list(static_shape(node)[2:]) != expect:
        raise Unsupported(f"{node.name}: output size disagrees with the transposed geometry")
    zh = zt + (ih - 1) * stride[0] + 1 + zt + op[0]
    zw = zl + (iw - 1) * stride[1] + 1 + zl + op[1]
    return stride, [zt, zl], [zh, zw], [kh, kw]


def _emit_conv(b, ep, node):
    if len(static_shape(node.args[0])) == 3:
        return _emit_conv1d(b, ep, node)
    x = node.args[0]
    w = _constant(ep, node.args[1], "convolution weight")
    bias_node = node.args[2] if len(node.args) > 2 else None
    groups = int(node.args[8]) if len(node.args) > 8 else 1
    ishape, oshape = static_shape(x), static_shape(node)
    co = oshape[1]
    if _transposed(node):
        stride, zpad, zhw, _ = _transposed_plan(node)
        zid = b.temp(node.name, (1, ishape[1], zhw[0], zhw[1]))
        b.emit_to(hx.ZEROINSERT, b.tid(x), NONE, zid, [stride[0], stride[1], zpad[0], zpad[1]])
        # Equivalent dense kernel: [in, out, kh, kw] -> [out, in, kh, kw], flipped in both axes.
        w = np.ascontiguousarray(w.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
        src, stride, pad = zid, [1, 1], [0, 0]
        record = "conv_transpose2d"
    else:
        src = b.tid(x)
        stride, pad = _ints(node.args[3], 2), _ints(node.args[4], 2)
        record = None
    dilation = [1, 1] if _transposed(node) else _dilation(node)
    kh, kw = w.shape[2], w.shape[3]
    ostride = (co + 31) & ~31
    if bias_node is None:
        bias = np.zeros(co, np.float32)
    else:
        bias = _constant(ep, bias_node, "convolution bias").reshape(-1)
        if bias.size != co:
            raise Unsupported("bias size mismatch")
    bn = _folding_bn(node, ep)
    if bn is not None:  # y = scale * (conv(x) + bias) + shift per output channel, folded into the weights
        scale, shift, _ = _bn_plan(bn, ep, data=True)
        if scale.size != co:
            raise Unsupported(f"{bn.name}: batch norm channels disagree with the convolution")
        w = np.ascontiguousarray(w * scale.reshape(-1, 1, 1, 1))
        bias = (bias * scale + shift).astype(np.float32)
        getitem = next(iter(bn.users))
        b.folded_bn.add(bn.name)
        b.alias[bn.name] = node.name
        b.alias[getitem.name] = node.name
    if groups > 1 and not (groups == w.shape[0] == ishape[1] and w.shape[1] == 1):
        cin = ishape[1]
        cig = cin // groups
        cog = co // groups
        ih, iw = ishape[2], ishape[3]
        oh, ow = oshape[2], oshape[3]
        acc_id, acc_c = None, 0
        for g in range(groups):
            piece = b.temp(f"{node.name}_gin", (1, cig, ih, iw))
            b.emit_to(hx.SLICE, src, NONE, piece, [0, 0, g * cig])
            w_g = w[g * cog:(g + 1) * cog]
            b_g = bias[g * cog:(g + 1) * cog]
            ostride_g = (cog + 31) & ~31
            wk_g = np.pad(w_g.transpose(2, 3, 1, 0), ((0, 0),) * 3 + ((0, ostride_g - cog),))
            packed_g, tiles_g = hx.pack_hmx_weights(wk_g)
            p_g = [kh, kw, stride[0], stride[1], pad[0], pad[1],
                   b.pool.emit(wk_g), b.pool.emit(_pad32(b_g)), hx.ACT_NONE,
                   b.pool.emit(packed_g, "<f2"), tiles_g]
            part = b.tid(node) if (g == groups - 1 and acc_id is None) else b.temp(f"{node.name}_gout", (1, cog, oh, ow))
            b.emit_to(hx.CONV, piece, NONE, part, p_g, _dilated_z(dilation))
            if acc_id is None:
                acc_id, acc_c = part, cog
                continue
            acc_c += cog
            comb = b.tid(node) if g == groups - 1 else b.temp(f"{node.name}_gcat", (1, acc_c, oh, ow))
            b.emit_to(hx.CONCAT, acc_id, part, comb, [2])
            acc_id = comb
        b.record(node, record or "grouped_conv2d")
        return
    if groups == 1:
        kind = hx.CONV
        wk = np.pad(w.transpose(2, 3, 1, 0), ((0, 0),) * 3 + ((0, ostride - co),))
    else:
        kind = hx.DEPTHWISE
        wk = np.pad(w[:, 0].transpose(1, 2, 0), ((0, 0), (0, 0), (0, ostride - co)))
    p = [kh, kw, stride[0], stride[1], pad[0], pad[1],
         b.pool.emit(wk), b.pool.emit(_pad32(bias)), hx.ACT_NONE]
    if kind == hx.CONV:
        packed, tiles = hx.pack_hmx_weights(wk)
        if not hx.hmx_fits(tiles, ostride):
            raise Unsupported(f"{node.name}: convolution exceeds the HMX budget")
        p += [b.pool.emit(packed, "<f2"), tiles]
    b.emit_to(kind, src, NONE, b.tid(node), p, _dilated_z(dilation))
    b.record(node, record or ("depthwise_conv2d" if kind == hx.DEPTHWISE else "conv2d"))


def _emit_linear(b, ep, node):
    x = node.args[0]
    w = _constant(ep, node.args[1], "linear weight")  # [out, in]
    out_features = w.shape[0]
    bias = (
        _constant(ep, node.args[2], "linear bias").reshape(-1)
        if len(node.args) > 2 and node.args[2] is not None
        else np.zeros(out_features, np.float32)
    )
    ostride = (out_features + 31) & ~31
    wk = np.pad(w.T[None, None], ((0, 0), (0, 0), (0, 0), (0, ostride - out_features)))
    packed, tiles = hx.pack_hmx_weights(wk)
    if not hx.hmx_fits(tiles, ostride):
        raise Unsupported(f"{node.name}: linear layer exceeds the HMX budget")
    p = [1, 1, 1, 1, 0, 0, b.pool.emit(wk), b.pool.emit(_pad32(bias)), hx.ACT_NONE,
         b.pool.emit(packed, "<f2"), tiles]
    b.emit(hx.CONV, b.tid(x), NONE, node, p)
    b.record(node, "linear")


def _emit_addmm(b, ep, node):
    bias_node, x, w_node = node.args[0], node.args[1], node.args[2]
    alpha = float(node.args[4]) if len(node.args) > 4 else float(node.kwargs.get("alpha", 1.0))
    beta = float(node.args[3]) if len(node.args) > 3 else float(node.kwargs.get("beta", 1.0))
    w = _constant(ep, w_node, "addmm weight") * alpha  # shape [in, out]
    out_features = w.shape[1]
    bias = (
        _constant(ep, bias_node, "addmm bias").reshape(-1) * beta
        if bias_node is not None
        else np.zeros(out_features, np.float32)
    )
    ostride = (out_features + 31) & ~31
    wk = np.pad(w[None, None], ((0, 0), (0, 0), (0, 0), (0, ostride - out_features)))
    packed, tiles = hx.pack_hmx_weights(wk)
    if not hx.hmx_fits(tiles, ostride):
        raise Unsupported(f"{node.name}: addmm exceeds the HMX budget")
    p = [1, 1, 1, 1, 0, 0, b.pool.emit(wk), b.pool.emit(_pad32(bias)), hx.ACT_NONE,
         b.pool.emit(packed, "<f2"), tiles]
    b.emit(hx.CONV, b.tid(x), NONE, node, p)
    b.record(node, "addmm")


def _emit_maxpool(b, node):
    x = node.args[0]
    k = _ints(node.args[1], 2)
    s = _ints(node.args[2], 2) if len(node.args) > 2 and node.args[2] else k
    p = _ints(node.args[3], 2) if len(node.args) > 3 else [0, 0]
    getitem = next(iter(node.users))
    # Name the HXFG output after the getitem(0) result the graph consumes.
    b.emit(hx.MAXPOOL, b.tid(x), NONE, getitem, [k[0], k[1], s[0], s[1], p[0], p[1]])
    b.alias[node.name] = getitem.name
    b.record(node, "max_pool2d")


def _token_bmm(node):
    """A bmm of rank-3 operands with a batch of one: its tensors are ordinary tokens [L, 1, K]."""
    shape = static_shape(node.args[0])
    return len(shape) == 3 and shape[0] == 1


def _emit_bmm(b, node):
    a = node.args[0]
    b_arg = node.args[1]
    shape_a = static_shape(a)
    B = shape_a[0] if len(shape_a) == 3 else shape_a[0] * shape_a[1]
    tid_a = b.tid(a)
    tid_b = b.tid(b_arg)
    tid_out = b.tid(node)
    if not _token_bmm(node):
        b.multihead.add(tid_a)
        b.multihead.add(tid_b)
        b.multihead.add(tid_out)
    b.emit(hx.MATMUL, tid_a, tid_b, node, [0, 0, B, 0])
    b.record(node, "bmm")


def _sdpa_fast(node):
    """True for the single-head unmasked SDPA that runs as MATMUL, SOFTMAX, MATMUL on token columns."""
    shape_q = static_shape(node.args[0])
    mask = node.args[3] if len(node.args) > 3 and node.args[3] is not None else None
    causal = bool(node.args[5]) if len(node.args) > 5 and node.args[5] is not None else False
    return (len(shape_q) == 4 and shape_q[0] * shape_q[1] == 1 and mask is None and not causal
            and static_shape(node.args[1])[2] <= MAX_CHANNELS)


def _token_head_view(node, to_tokens):
    """The [1, L, D] <-> [1, 1, L, D] view (unsqueeze/squeeze/view) between token tensors and a single-head SDPA."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function" or kind_name(node) != "reshape":
        return False
    src, out = static_shape(node.args[0]), static_shape(node)
    if to_tokens:
        src, out = out, src
    return len(src) == 3 and src[0] == 1 and tuple(out) == (1, 1, src[1], src[2])


def _sdpa_token_views(ep):
    """Single-head SDPAs whose q, k, v are unsqueeze(1) views of token tensors [1, L, D] and whose only user
    is the squeeze(1) back to tokens: (aliases, skipped view names, SDPA names). The token tensors are already
    the column layout [L, 1, D] the matmuls read, so the views and the multi-head repacks are dropped."""
    alias, skip, fast = {}, set(), set()
    for node in ep.graph.nodes:
        if node.op != "call_function" or kind_name(node) != "sdpa" or not _sdpa_fast(node):
            continue
        ins = node.args[:3]
        users = list(node.users)
        if not all(_token_head_view(a, False) and all(kind_name(u) == "sdpa" and _sdpa_fast(u) for u in a.users)
                   for a in ins):
            continue
        if len(users) != 1 or not _token_head_view(users[0], True):
            continue
        for a in ins:
            alias[a.name] = a.args[0].name
            skip.add(a.name)
        alias[node.name] = users[0].name
        skip.add(users[0].name)
        fast.add(node.name)
    return alias, skip, fast


# Factor applied to attention scores before they are stored (see _emit_sdpa). A power of two in (0, 1];
# 1.0 keeps the plain MATMUL/SOFTMAX pair. Set with set_attention_score_scale before lowering.
ATTENTION_SCORE_SCALE = 1.0


def set_attention_score_scale(scale):
    global ATTENTION_SCORE_SCALE
    mantissa, _ = np.frexp(scale)
    if not 0 < scale <= 1 or mantissa != 0.5:
        raise ValueError("attention score scale must be a power of two in (0, 1]")
    ATTENTION_SCORE_SCALE = float(scale)


def _emit_sdpa(b, node):
    q, k, v = node.args[0], node.args[1], node.args[2]
    mask = node.args[3] if len(node.args) > 3 and node.args[3] is not None else None
    is_causal = bool(node.args[5]) if len(node.args) > 5 and node.args[5] is not None else False
    scale = node.args[6] if len(node.args) > 6 and node.args[6] is not None else None
    scale_bits = hx.f32_bits(scale) if scale is not None else 0
    shape_q = static_shape(q)
    B = shape_q[0] * shape_q[1]
    lq, lk = shape_q[2], static_shape(k)[2]
    if B == 1 and mask is None and not is_causal and lk <= MAX_CHANNELS:
        # One unmasked head as Q.K^T (scaled), a per-row softmax over the key axis and P.V, on token
        # columns [L, 1, D]: the layout HX_MATMUL runs in HVX lanes on the FP16 plan, where HX_SDPA is
        # scalar (about 250 M cycles at 256 tokens). The multi-head operands [1, L, D] are a different
        # blocked layout (two tokens per vector), so each is repacked by an HX_RESHAPE copy.
        dk, dv = shape_q[3], static_shape(v)[3]
        token = node.name in b.sdpa_token
        if token:  # q, k, v are token tensors [1, L, D] = columns [L, 1, D]; the output is the squeeze's tensor
            cq, ck, cv = b.tid(q.args[0]), b.tid(k.args[0]), b.tid(v.args[0])
            out_node = next(iter(node.users))
        else:
            tq, tk, tv, to = b.tid(q), b.tid(k), b.tid(v), b.tid(node)
            for t in (tq, tk, tv, to):
                b.multihead.add(t)

            def column(tid, rows, cols, tag):
                dst = b.temp(node.name + tag, (rows, cols))
                b.emit_to(hx.RESHAPE, tid, NONE, dst, [0])
                return dst

            cq, ck, cv = column(tq, lq, dk, "_q"), column(tk, lk, dk, "_k"), column(tv, lk, dv, "_v")
        scores = b.temp(node.name + "_scores", (lq, lk))
        probs = b.temp(node.name + "_probs", (lq, lk))
        co = b.tid(out_node) if token else b.temp(node.name + "_o", (lq, dv))
        sc = float(scale) if scale is not None else 1.0 / float(np.sqrt(dk))
        shrink = ATTENTION_SCORE_SCALE
        # The scores tensor is FP16 on the blocked plan (max 65504); Depth Pro 2's first block reaches 2e6.
        # Store Q.K^T * shrink (a power of two, so exact) and let the softmax multiply by 1/shrink in float.
        b.emit_to(hx.MATMUL, cq, ck, scores, [1, hx.f32_bits(sc * shrink), 1, 0])
        if shrink == 1.0:
            b.emit_to(hx.SOFTMAX, scores, NONE, probs, [])
        else:
            b.emit_to(hx.ATTN_SOFTMAX, scores, NONE, probs, [2, 0, 0, hx.f32_bits(1.0 / shrink)])
        b.emit_to(hx.MATMUL, probs, cv, co, [0, 0, 1, 0])
        if not token:
            b.emit_to(hx.RESHAPE, co, NONE, to, [0])
        b.record(node, "scaled_dot_product_attention")
        return
    tid_q = b.tid(q)
    tid_k = b.tid(k)
    tid_v = b.tid(v)
    tid_out = b.tid(node)
    tid_mask = b.tid(mask) if mask is not None else NONE
    b.multihead.add(tid_q)
    b.multihead.add(tid_k)
    b.multihead.add(tid_v)
    b.multihead.add(tid_out)
    if mask is not None:
        b.multihead.add(tid_mask)
    b.emit_to(hx.SDPA, tid_q, tid_k, tid_out, [tid_mask, scale_bits, 1 if is_causal else 0, B], z=tid_v)
    b.record(node, "scaled_dot_product_attention")


def _pool_vec(b, vec, integer=False):
    """Constant-pool offset of a vector padded to 32 values (int32 for integer operands)."""
    values = np.asarray(vec, np.float32).reshape(-1)
    if integer:
        padded = np.pad(np.rint(values).astype(np.int32), (0, ((values.size + 31) & ~31) - values.size))
        return b.pool.emit(padded, "<i4")
    return b.pool.emit(_pad32(values))


def _emit_binary(b, ep, node, out, plan):
    """Emit one _binary_plan into tensor id `out`."""
    opcode, x, y, vec, rev = plan[:5]
    mode = plan[5] if len(plan) > 5 else 1
    if _plain_add(node, ep):
        b.emit_to(hx.ADD, b.tid(x), b.tid(y), out, [0])
    elif y is not None:
        b.emit_to(hx.BINARY, b.tid(x), b.tid(y), out, [opcode, 0, 0, rev])
    else:
        b.emit_to(hx.BINARY, b.tid(x), NONE, out, [opcode, mode, _pool_vec(b, vec, _is_int_node(x)), rev])


def _emit_where(b, ep, node):
    cond, operands = _where_plan(node, ep, data=True)
    fields, tensors = [], []
    for mode, value, array in operands:
        if mode == 0:
            fields.append(0)
        elif mode == 1:
            fields.append(hx.f32_bits(value))
        else:
            fields.append(b.pool.emit(array))
    x_tid = b.tid(node.args[1]) if operands[0][0] == 0 else NONE
    z_tid = b.tid(node.args[2]) if operands[1][0] == 0 else NONE
    b.emit_to(hx.WHERE, b.tid(cond), x_tid, b.tid(node),
              [operands[0][0], fields[0], operands[1][0], fields[1]], z=z_tid)


def _emit_argreduce(b, node, opcode):
    """ARGMAX / ARGMIN: first index on ties (select_last_index 0), int32 output."""
    out = b.tid(node)
    b.dtypes[out] = hx.DTYPE_INT32
    b.emit_to(opcode, b.tid(node.args[0]), NONE, out, [0, 1])


def _emit_topk(b, ep, node):
    k, largest, sorted_, expect = _topk_plan(node, ep)
    users = _output_users(node, (0, 1))
    values = b.tid(users[0]) if 0 in users else b.temp(node.name + "_values", expect)
    indices = NONE
    if 1 in users:
        indices = b.tid(users[1])
        b.dtypes[indices] = hx.DTYPE_INT32
    b.emit_to(hx.TOPK, b.tid(node.args[0]), NONE, values, [k, int(largest), int(sorted_), 1], z=indices)


def _emit_maxdim(b, ep, node):
    users = _maxdim_plan(node, ep)
    is_min = op_name(node.target) == "min.dim"
    x = b.tid(node.args[0])
    if 0 in users:  # values: REDUCE over the channel bit (mask 4)
        b.emit_to(hx.REDUCE, x, NONE, b.tid(users[0]), [hx.RED_MIN if is_min else hx.RED_MAX, 4])
    if 1 in users:
        b.dtypes[b.tid(users[1])] = hx.DTYPE_INT32
        b.emit_to(hx.ARGMIN if is_min else hx.ARGMAX, x, NONE, b.tid(users[1]), [0, 1])


def _emit_clamp(b, node, lo, hi):
    """clamp / hardtanh with bounds that are not ReLU/ReLU6: MAX with the lower bound, then MIN with the upper."""
    channels = hwc_of(static_shape(node), node.name)[2]
    steps = []
    if lo is not None:
        steps.append((hx.BIN_MAX, lo))
    if hi is not None:
        steps.append((hx.BIN_MIN, hi))
    src = b.tid(node.args[0])
    for i, (opcode, bound) in enumerate(steps):
        dst = b.tid(node) if i == len(steps) - 1 else b.temp(node.name + "_clamp", static_shape(node))
        b.emit_to(hx.BINARY, src, NONE, dst, [opcode, 1, _pool_vec(b, np.full(channels, bound, np.float32)), 0])
        src = dst


def _emit_var(b, ep, node):
    """var.correction = (mean of the squared deviations from the mean) * N / (N - correction)."""
    mask, _, factor = _var_plan(node, ep)
    x = node.args[0]
    ishape = static_shape(x)
    bits = {1: 4, 2: 1, 3: 2} if len(ishape) == 4 else {1: 4}
    keep = [1 if mask & bits.get(d, 0) else ishape[d] for d in range(len(ishape))]
    tx = b.tid(x)

    def mean_into(src, dst):
        if mask == 3 and len(ishape) == 4:
            b.emit_to(hx.MEAN, src, NONE, dst, [])
        else:
            b.emit_to(hx.REDUCE, src, NONE, dst, [hx.RED_MEAN, mask])

    mean = b.temp(node.name + "_mean", keep)
    mean_into(tx, mean)
    centred = b.temp(node.name + "_centred", ishape)
    b.emit_to(hx.BINARY, tx, mean, centred, [hx.BIN_SUB, 0, 0, 0])
    squared = b.temp(node.name + "_squared", ishape)
    b.emit_to(hx.BINARY, centred, centred, squared, [hx.BIN_MUL, 0, 0, 0])
    if factor == 1:
        mean_into(squared, b.tid(node))
        return
    summed = b.temp(node.name + "_sum", static_shape(node))
    mean_into(squared, summed)
    channels = hwc_of(static_shape(node), node.name)[2]
    b.emit_to(hx.BINARY, summed, NONE, b.tid(node),
              [hx.BIN_MUL, 1, _pool_vec(b, np.full(channels, factor, np.float32)), 0])


def _emit_conv1d(b, ep, node):
    """conv1d on [1, C, L] as a 2-D convolution with one row. HWC of [1, C, L] is [C, 1, L], so the
    input is transposed to the rank-4 layout [1, C, 1, L] (HWC [1, L, C]) and the output back."""
    x = node.args[0]
    w = _constant(ep, node.args[1], "convolution weight")  # [O, I/groups, K]
    ishape, oshape = static_shape(x), static_shape(node)
    channels, length = ishape[1], ishape[2]
    out_channels, out_length = oshape[1], oshape[2]
    stride, pad, dil, groups = _conv1d_params(node)
    kernel = w.shape[2]
    if len(node.args) > 2 and node.args[2] is not None:
        bias = _constant(ep, node.args[2], "convolution bias").reshape(-1)
    else:
        bias = np.zeros(out_channels, np.float32)
    ostride = (out_channels + 31) & ~31
    src = b.temp(node.name + "_in", (1, channels, 1, length))
    b.emit_to(hx.TRANSPOSE, b.tid(x), NONE, src, [1, 2, 0])
    out = b.temp(node.name + "_conv", (1, out_channels, 1, out_length))
    w4 = w.reshape(out_channels, -1, 1, kernel)
    dilation = [1, dil]
    if groups == 1:
        wk = np.pad(w4.transpose(2, 3, 1, 0), ((0, 0),) * 3 + ((0, ostride - out_channels),))
        packed, tiles = hx.pack_hmx_weights(wk)
        if not hx.hmx_fits(tiles, ostride):
            raise Unsupported(f"{node.name}: convolution exceeds the HMX budget")
        p = [1, kernel, 1, stride, 0, pad, b.pool.emit(wk), b.pool.emit(_pad32(bias)), hx.ACT_NONE,
             b.pool.emit(packed, "<f2"), tiles]
        b.emit_to(hx.CONV, src, NONE, out, p, _dilated_z(dilation))
    else:
        wk = np.pad(w4[:, 0].transpose(1, 2, 0), ((0, 0), (0, 0), (0, ostride - out_channels)))
        p = [1, kernel, 1, stride, 0, pad, b.pool.emit(wk), b.pool.emit(_pad32(bias)), hx.ACT_NONE]
        b.emit_to(hx.DEPTHWISE, src, NONE, out, p, _dilated_z(dilation))
    b.emit_to(hx.TRANSPOSE, out, NONE, b.tid(node), [2, 0, 1])
    b.record(node, "conv1d")


def _emit_index_select(b, ep, node, x_arg, dim_arg, idx_arg):
    """HX_GATHER along dim_arg of x_arg by a one-dimensional index tensor."""
    try:
        idx_const = np.asarray(_const_value(ep, idx_arg), dtype=np.int32) if _is_constant(ep, idx_arg) else None
    except Exception:
        idx_const = None
    try:
        x_const = _constant(ep, x_arg, "index_select table") if _is_constant(ep, x_arg) else None
    except Exception:
        x_const = None
    xshape = static_shape(x_arg)
    dim_arg = dim_arg + len(xshape) if dim_arg < 0 else dim_arg
    if len(xshape) == 4 and dim_arg == 2 and idx_const is not None and x_const is None:
        # Rows of a [1, C, H, W] map. HX_GATHER's row form reads one channel vector per output pixel, so the
        # rows go through a pixel gather over the flattened (h, w) list: pixel (r, x) reads (index[r], x).
        w_pix = int(xshape[3])
        rows = idx_const.reshape(-1).astype(np.int64)
        flat = (rows[:, None] * w_pix + np.arange(w_pix)[None, :]).reshape(-1).astype(np.int32)
        off = b.pool.emit(flat, "<i4")
        b.emit(hx.GATHER, b.tid(x_arg), NONE, node,
               [2, 0, int(xshape[2]) * w_pix, 0, 0, 0, 0, 0, off, flat.size])
        b.record(node, op_name(node.target))
        return
    if len(xshape) == 4:
        if dim_arg == 1:
            hwc_axis = 2
            vocab = int(xshape[1])
            outer_dim = int(xshape[2] * xshape[3])
            inner_dim = 1
        elif dim_arg == 2:
            hwc_axis = 0
            vocab = int(xshape[2])
            outer_dim = 1
            inner_dim = int(xshape[3] * xshape[1])
        elif dim_arg == 3:
            hwc_axis = 1
            vocab = int(xshape[3])
            outer_dim = int(xshape[2])
            inner_dim = int(xshape[1])
        else:
            raise Unsupported(f"index_select rank-4 dim={dim_arg} unsupported")
    elif len(xshape) == 3:
        if dim_arg == 2:
            hwc_axis = 2
            vocab = int(xshape[2])
            outer_dim = int(xshape[1])
            inner_dim = 1
        elif dim_arg == 1:
            hwc_axis = 0
            vocab = int(xshape[1])
            outer_dim = 1
            inner_dim = int(xshape[2])
        else:
            raise Unsupported(f"index_select rank-3 dim={dim_arg} unsupported")
    else:
        hwc_axis = 2
        vocab = int(xshape[-1])
        outer_dim = 1
        inner_dim = 1
    stride = (inner_dim + 31) & ~31
    if idx_const is not None:
        idx_arr = idx_const.reshape(-1)
        const_idx_off = b.pool.emit(idx_arr, "<i4")
        const_idx_num = idx_arr.size
        mode = 2
        x_tid = b.tid(x_arg)
        y_tid = NONE
        b.emit(hx.GATHER, x_tid, y_tid, node,
               [mode, 0, vocab, 0, hwc_axis, inner_dim, outer_dim, 0, const_idx_off, const_idx_num])
    elif x_const is not None:
        tbl = np.asarray(x_const, dtype=np.float32).reshape(outer_dim, vocab, inner_dim)
        if stride > inner_dim:
            tbl = np.pad(tbl, ((0, 0), (0, 0), (0, stride - inner_dim)))
        offset = b.pool.emit(tbl.reshape(-1))
        mode = 1
        y_tid = NONE
        table_stride = stride
        b.emit(hx.GATHER, b.tid(idx_arg), y_tid, node,
               [mode, offset, vocab, 0, hwc_axis, inner_dim, outer_dim, table_stride])
    else:
        mode = 0
        offset = 0
        y_tid = b.tid(x_arg)
        table_stride = stride
        b.emit(hx.GATHER, b.tid(idx_arg), y_tid, node,
               [mode, offset, vocab, 0, hwc_axis, inner_dim, outer_dim, table_stride])
    b.record(node, op_name(node.target))


_MH_THROUGH = ("binary", "add", "softmax", "act", "activation", "logistic")


def _concat_axis(rank, dim, multihead=False):
    """HXFG CONCAT axis for a PyTorch concat dim. Batch-one rank 3 is the token layout [L, 1, C] (dim 1 is
    the token axis H), or [1, L, C] when it carries the attention (multi-head) layout."""
    if rank == 2:
        return 2
    if rank == 3:
        return 2 if dim == 2 else (1 if multihead else 0)
    return {1: 2, 2: 0, 3: 1}[dim]


def b_canon(node, aliases):
    name = node.name
    while aliases and name in aliases:
        name = aliases[name]
    return name


def _spread_multihead(ep, names, aliases=None):
    """Extend the attention (multi-head) layout through rank-3 elementwise operations between two attention
    tensors (the score scale, the mask add, the softmax): those keep [B, L, K] rather than the token layout
    [B * L, 1, K], so they must agree with the bmm that reads or writes them. Only unary-like operations
    with constant other operands spread, and only between same-shape rank-3 tensors."""
    changed = True
    while changed:
        changed = False
        for node in ep.graph.nodes:
            if node.op != "call_function" or kind_name(node) not in _MH_THROUGH:
                continue
            x = node.args[0]
            if (not isinstance(x, torch.fx.Node) or x.op != "call_function" or len(static_shape(node)) != 3
                    or static_shape(x) != static_shape(node)
                    or any(isinstance(a, torch.fx.Node) and not _is_constant(ep, a) and a is not x
                           for a in node.args[1:])):
                continue
            a, c = x.name, node.name
            if (a in names) != (c in names):
                names.add(a if c in names else c)
                changed = True
        # Attention tensors keep [B, L, C] through a rank-3 view or a swap of L and C (the key transpose of a
        # bmm with a batch above one): rank-3 to rank-3 with a batch above one on either side.
        for node in ep.graph.nodes:
            if node.op != "call_function" or kind_name(node) not in ("reshape", "transpose"):
                continue
            x = node.args[0]
            if not isinstance(x, torch.fx.Node) or x.op != "call_function" or _is_constant(ep, x):
                continue
            sx, sn = static_shape(x), static_shape(node)
            if len(sx) != 3 or len(sn) != 3 or (sx[0] == 1 and sn[0] == 1):
                continue
            a, c = b_canon(x, aliases), b_canon(node, aliases)
            if (a in names) != (c in names):
                names.add(a if c in names else c)
                changed = True


def _by_component(ops):
    """Independent subgraphs one after another. ExecuTorch interleaves the operations of
    subnetworks that share no tensors (one module called on two hands); run in that order, both
    sets of activations are live at once and can overflow the DSP's VTCM plan. Operations are
    grouped by the connected component of the tensors they touch (constants are not tensors here)
    and the components emitted in order of their first operation, each keeping its own order, so
    every operation still follows its producers. Graphs with an operation that names tensors in
    its p[] fields (kinds from HX_RMSNORM on: RoPE tables, KV caches, attention) keep their order.
    A convolution's z (dilation code) may join two components; that only skips reordering."""
    if any(op[0] >= 22 for op in ops):
        return ops
    parent = {}

    def root(t):
        parent.setdefault(t, t)
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for op in ops:
        touched = [t for t in op[1:5] if t != NONE]
        for t in touched[1:]:
            parent[root(t)] = root(touched[0])
    order, groups = [], {}
    for op in ops:
        touched = [t for t in op[1:5] if t != NONE]
        key = root(touched[0]) if touched else id(op)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(op)
    return [op for key in order for op in groups[key]]


def _op_inputs(op):
    """Tensor ids an op reads (x, y, z; a convolution's z is a dilation code)."""
    z = NONE if op[0] in (hx.CONV, hx.DEPTHWISE, hx.MAXPOOL, hx.AVGPOOL) else op[3]
    ids = [op[1], op[2], z]
    if op[0] == hx.SDPA and len(op) > 5 and op[5] != NONE:
        ids.append(op[5])
    return [t for t in ids if t != NONE]


def _live_peak(ops, size, outputs):
    """Peak of the summed sizes of live tensors when `ops` run in order (inputs live from the start, a tensor
    dies after its last reader; graph outputs live to the end). An estimate of the planner's arena."""
    last = {}
    for j, op in enumerate(ops):
        for t in _op_inputs(op):
            last[t] = j
    for t in outputs:
        last[t] = len(ops)
    live, cur, peak = {}, 0, 0
    produced = {op[4] for op in ops}
    for t in {t for op in ops for t in _op_inputs(op)} - produced:
        live[t] = size(t)  # graph inputs
        cur += live[t]
    peak = cur
    for j, op in enumerate(ops):
        if op[4] not in live:
            live[op[4]] = size(op[4])
            cur += live[op[4]]
        peak = max(peak, cur)
        for t in [t for t in live if last.get(t, -1) <= j and t != op[4]]:
            cur -= live.pop(t)
        if last.get(op[4], -1) <= j:
            cur -= live.pop(op[4], 0)
    return peak


def _demand_order(ops, outputs, size, persistent=False):
    """Run each operation just before its first consumer instead of in the order ExecuTorch emitted it.

    A tap gather that only one convolution reads would otherwise sit materialised for the whole run of
    gathers ahead of it (sixteen 16 MB tensors for one 4x4x4 transposed convolution). Operations are emitted
    by a depth-first walk from the graph outputs, inputs in operand order. The new order is used only when
    its estimated live peak is lower; graphs with persistent state, in-place scatter or operations that name
    tensors in p[] keep their order."""
    if persistent or any(op[0] in (hx.ROPE, hx.SHORTCONV, hx.KV_WRITE, hx.SDPA, hx.SCATTER_ND) for op in ops):
        return ops
    producer = {}
    for i, op in enumerate(ops):
        if op[4] in producer:
            return ops  # a tensor written twice: order matters
        producer[op[4]] = i
    done, order = set(), []
    roots = [producer[t] for t in outputs if t in producer] + list(range(len(ops)))
    for root in roots:
        if root in done:
            continue
        stack = [(root, iter(_op_inputs(ops[root])))]
        done.add(root)
        while stack:
            i, it = stack[-1]
            for t in it:
                j = producer.get(t)
                if j is not None and j not in done:
                    done.add(j)
                    stack.append((j, iter(_op_inputs(ops[j]))))
                    break
            else:
                order.append(i)
                stack.pop()
    new = [ops[i] for i in order]
    return new if _live_peak(new, size, outputs) < _live_peak(ops, size, outputs) else ops


def lower_program(ep, destination):
    """Lower a delegated edge-dialect partition.

    Every call_function node in `ep` must be supported; the caller passes only
    delegated nodes (preprocess receives exactly the tagged subgraph).
    Returns (summary dict, HXFG image bytes, constant pool bytes).
    """
    b = _Builder(ep)
    alias, sdpa_skip, b.sdpa_token = _sdpa_token_views(ep)
    b.alias.update(alias)
    for node in ep.graph.nodes:
        if node.op == "call_function":
            k = kind_name(node)
            if k == "bmm" and _token_bmm(node):
                continue  # one batch: HX_MATMUL runs on the token layout [L, 1, K] directly
            if k == "sdpa" and node.name in b.sdpa_token:
                continue  # token columns throughout (see _sdpa_token_views)
            if k in ("bmm", "sdpa"):
                for a in node.all_input_nodes:
                    b.multihead_names.add(b.canon(a))
                b.multihead_names.add(b.canon(node))
    _spread_multihead(ep, b.multihead_names, b.alias)

    graph_inputs = [
        n for n in ep.graph.nodes if n.op == "placeholder" and not _is_constant(ep, n)
    ]
    if not graph_inputs:
        raise Unsupported("partition has no runtime inputs")
    for n in graph_inputs:
        hwc_of(static_shape(n), n.name, multihead=(b.canon(n) in b.multihead_names))
    compact = {
        n.name
        for n in graph_inputs
        if len(static_shape(n)) == 4
        and static_shape(n)[1] in (1, 3)
        and n.users
        and all(kind_name(u) == "conv" and not _transposed(u) and _dilation(u) == [1, 1]
                for u in n.users)
        and (static_shape(n)[1] == 3 or _compact_window(list(n.users)))
    }
    for n in graph_inputs:
        b.tid(n)
    input_ids = [b.tid(n) for n in graph_inputs]

    done, activation_of = set(), {}
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        if node.name in sdpa_skip:
            continue  # a head view folded into its single-head SDPA
        kind = kind_of(node, ep)
        if kind is None:
            raise Unsupported(f"{node.name}: operation {op_name(node.target)} is not supported")
        name = op_name(node.target)
        if kind in ("constcast", "constfactory") or (kind == "reshape" and name in _CONST_PASSTHROUGH
                                                     and _is_constant(ep, node)):
            continue  # constant transform; consumers read the folded value through _const_value
        if kind == "conv":
            _emit_conv(b, ep, node)
        elif kind == "linear":
            _emit_linear(b, ep, node)
        elif kind == "bmm":
            _emit_bmm(b, node)
        elif kind == "sdpa":
            _emit_sdpa(b, node)
        elif kind == "addmm":
            _emit_addmm(b, ep, node)
        elif kind in ("add", "binary"):
            if kind == "add" and _fused_softmax_add(b, ep, node):
                b.source_ops.append({"op": name, "fused_into": next(iter(node.users)).name})
                done.add(node)
                continue  # the softmax reads the scores and the mask itself
            _emit_binary(b, ep, node, b.tid(node), _binary_plan(node, ep, data=True))
            b.record(node, name)
        elif kind == "ne":  # EQUAL into a temporary, then NOT
            eq = b.temp(node.name + "_eq", static_shape(node))
            _emit_binary(b, ep, node, eq, _binary_plan(node, ep, data=True))
            b.emit_to(hx.ACT, eq, NONE, b.tid(node), [hx.FN_NOT, 0, 0])
            b.record(node, name)
        elif kind == "where":
            if _is_constant(ep, node.args[0]):
                if _fused_softmax_add(b, ep, node):
                    b.source_ops.append({"op": name, "fused_into": next(iter(node.users)).name})
                else:
                    _emit_const_where(b, ep, node)
            else:
                _emit_where(b, ep, node)
                b.record(node, name)
        elif kind in ("argmax", "argmin"):
            _emit_argreduce(b, node, hx.ARGMAX if kind == "argmax" else hx.ARGMIN)
            b.record(node, name)
        elif kind == "topk":  # the producer; its getitem users take no operation of their own
            _emit_topk(b, ep, node)
            b.record(node, name)
        elif kind == "maxdim":
            _emit_maxdim(b, ep, node)
            b.record(node, name)
        elif kind == "tile":
            b.emit_to(hx.TILE, b.tid(node.args[0]), NONE, b.tid(node), [])
            b.record(node, name)
        elif kind == "rsqrt":  # 1 / sqrt(x)
            tmp = b.temp(node.name + "_sqrt", static_shape(node))
            b.emit_to(hx.ACT, b.tid(node.args[0]), NONE, tmp, [hx.FN_SQRT, 0, 0])
            b.emit_to(hx.ACT, tmp, NONE, b.tid(node), [hx.FN_RECIPROCAL, 0, 0])
            b.record(node, name)
        elif kind == "logsigmoid":
            b.emit_to(hx.ACT, b.tid(node.args[0]), NONE, b.tid(node), [hx.FN_LOGSIGMOID, 0, 0])
            b.record(node, name)
        elif kind == "var":
            _emit_var(b, ep, node)
            b.record(node, name)
        elif kind == "slice":
            x = node.args[0]
            ishape = static_shape(x)
            dim = int(node.args[1]) if len(node.args) > 1 and node.args[1] is not None else 0
            dim = dim + len(ishape) if dim < 0 else dim
            start = int(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 0
            if start < 0:
                start += ishape[dim]
            start = max(0, min(start, ishape[dim]))
            p = [0, 0, 0]
            p[_hwc_axis(ishape, dim, x.name)] = start
            b.emit(hx.SLICE, b.tid(x), NONE, node, p)
            b.record(node, name)
        elif kind == "split":
            pass
        elif kind == "act":
            fn, alpha, beta = ACT_FUNCTIONS[name](node)
            b.emit(hx.ACT, b.tid(node.args[0]), NONE, node,
                   [fn, hx.f32_bits(alpha) if alpha else 0, hx.f32_bits(beta) if beta else 0])
            b.record(node, name)
        elif kind == "avgpool":
            k = _ints(node.args[1], 2)
            s_ = _ints(node.args[2], 2) if len(node.args) > 2 and node.args[2] else k
            pd = _ints(node.args[3], 2) if len(node.args) > 3 else [0, 0]
            include = bool(node.args[5]) if len(node.args) > 5 else True
            b.emit(hx.AVGPOOL, b.tid(node.args[0]), NONE, node,
                   [k[0], k[1], s_[0], s_[1], pd[0], pd[1], int(include)])
            b.record(node, name)
        elif kind == "gap":
            b.emit(hx.MEAN, b.tid(node.args[0]), NONE, node, [])
            b.record(node, name)
        elif kind == "prelu":
            c = hwc_of(static_shape(node.args[0]))[2]
            slope = _constant(ep, node.args[1], "PReLU slope").reshape(-1)
            if slope.size == 1:
                slope = np.full(c, slope[0], np.float32)
            b.emit(hx.PRELU, b.tid(node.args[0]), NONE, node, [b.pool.emit(_pad32(slope))])
            b.record(node, name)
        elif kind == "maxpool":
            _emit_maxpool(b, node)
        elif kind == "norm":
            mode, groups, eps, gamma, beta, has_getitem = _norm_plan(node, ep, data=True)
            if has_getitem:
                getitem = next(iter(node.users))
                b.emit(hx.NORM, b.tid(node.args[0]), NONE, getitem,
                       [mode, groups, hx.f32_bits(eps), b.pool.emit(_pad32(gamma)),
                        b.pool.emit(_pad32(beta))])
                b.alias[node.name] = getitem.name
            else:
                b.emit(hx.NORM, b.tid(node.args[0]), NONE, node,
                       [mode, groups, hx.f32_bits(eps), b.pool.emit(_pad32(gamma)),
                        b.pool.emit(_pad32(beta))])
            b.record(node, name)
        elif kind == "rmsnorm":
            eps, gamma = _rms_norm_plan(node, ep, data=True)
            gamma_off = b.pool.emit(_pad32(gamma)) if gamma is not None else NONE
            b.emit(hx.RMSNORM, b.tid(node.args[0]), NONE, node,
                   [hx.f32_bits(eps), gamma_off, NONE])
            b.record(node, name)
        elif kind == "lfm_linear_q":
            x, blocks, bias, fmt = node.args[:4]
            blob = np.ascontiguousarray(_const_value(ep, blocks), dtype=np.uint8).reshape(-1)
            n, k = int(blocks.meta["val"].shape[0]), static_shape(x)[-1]
            bias_off = b.pool.emit(_pad32(_constant(ep, bias, "bias"))) if bias is not None else NONE
            b.emit(hx.LINEAR_Q, b.tid(x), NONE, node, [int(fmt), b.pool.emit(blob, "u1"), bias_off, k, n, 0])
            b.record(node, name)
        elif kind == "lfm_rope":
            x, pos, cos, sin, heads, rot, interleaved, start = node.args[:8]
            width, length = static_shape(x)[-1], static_shape(x)[-2]
            head_dim = width // int(heads)
            rot = int(rot) or head_dim
            try:
                theta = hx.rope_theta(_const_value(ep, cos), _const_value(ep, sin), rot)
            except ValueError as error:
                raise Unsupported(f"{node.name}: {error}") from None
            b.emit(hx.ROPE, b.tid(x), b.tid(pos) if pos is not None else NONE, node,
                   [head_dim, int(heads), int(start), int(bool(interleaved)), hx.f32_bits(theta),
                    0 if rot == head_dim else rot, int(length > 1)])
            b.record(node, name)
        elif kind == "lfm_shortconv":
            x, weight, bias, gate, state, activation = node.args[:6]
            c = static_shape(x)[-1]
            w = _constant(ep, weight, "weight").reshape(c, -1)
            taps = w.shape[1]
            stride = (c + 31) & ~31
            wk = np.zeros((taps, stride), np.float32)
            wk[:, :c] = w.T
            bias_off = b.pool.emit(_pad32(_constant(ep, bias, "bias"))) if bias is not None else NONE
            sid = b.state_id(node, state)
            ring = b.temp(f"state:{state}", (1, taps - 1, c))
            b.persistent[ring] = sid
            b.emit_to(hx.SHORTCONV, b.tid(x), ring, b.tid(node),
                      [taps, b.pool.emit(wk), bias_off, {"none": 0, "relu": 1, "silu": 3}[activation]],
                      z=b.tid(gate) if gate is not None else NONE)
            b.record(node, name)
        elif kind == "lfm_kv_write":
            x, pos, state, max_len, start = node.args[:5]
            sid = b.state_id(node, state)
            out = b.tid(node)
            b.persistent[out] = sid
            b.emit_to(hx.KV_WRITE, b.tid(x), b.tid(pos) if pos is not None else NONE, out, [int(start)])
            b.record(node, name)
        elif kind == "getitem":
            prod = node.args[0]
            if isinstance(prod, torch.fx.Node) and op_name(prod.target) in (
                "split_with_sizes_copy.default", "split_with_sizes.default",
                "split_copy.Tensor", "split.Tensor"
            ):
                x = prod.args[0]
                ishape = static_shape(x)
                split_sizes = prod.args[1]
                dim = int(prod.args[2]) if len(prod.args) > 2 and prod.args[2] is not None else 0
                dim = dim + len(ishape) if dim < 0 else dim
                if isinstance(split_sizes, int):
                    num_splits = (ishape[dim] + split_sizes - 1) // split_sizes
                    sizes = [min(split_sizes, ishape[dim] - i * split_sizes) for i in range(num_splits)]
                else:
                    sizes = [int(s) for s in split_sizes]
                idx = int(node.args[1])
                offset = sum(sizes[:idx])
                if len(ishape) == 4:
                    if dim == 1:
                        p = [0, 0, offset]
                    elif dim == 2:
                        p = [offset, 0, 0]
                    elif dim == 3:
                        p = [0, offset, 0]
                else:
                    p = [0, 0, offset]
                b.emit(hx.SLICE, b.tid(x), NONE, node, p)
                b.record(node, "split")
        elif kind == "pad":
            wl, wr, ht, hb = [int(v) for v in node.args[1]]
            b.emit(hx.PAD, b.tid(node.args[0]), NONE, node, [ht, wl, 0])
            b.record(node, name)
        elif kind == "resize":
            if _is_nearest(node):
                b.emit(hx.RESIZE, b.tid(node.args[0]), NONE, node, [0, 0, 1])
            else:
                align = (bool(node.args[2]) if len(node.args) > 2 and node.args[2] is not None
                         else False)
                b.emit(hx.RESIZE, b.tid(node.args[0]), NONE, node, [int(align), int(not align)])
            b.record(node, name)
        elif kind in ("reshape", "transpose") and token_view(node) is not None:
            role, source = token_view(node)
            if role in ("flat_tail", "unflat_tail"):
                b.emit(hx.RESHAPE, b.tid(source), NONE, node, [0])  # same HWC storage, row order kept
                b.record(node, "token_view")
        elif kind == "reshape":
            _emit_reshape(b, node, node.args[0])
            b.record(node, name)
        elif kind == "transpose":
            if len(static_shape(node)) in (2, 3):
                x = node.args[0]
                p = _token_transpose_p(node, b.canon(x) in b.multihead_names, b.canon(node) in b.multihead_names)
            else:
                p = _transpose_p(node)
            b.emit(hx.TRANSPOSE, b.tid(node.args[0]), NONE, node, p)
            b.record(node, name)
        elif kind == "concat":  # a chain of two-input CONCATs; intermediate sums go through temporaries
            tensors = node.args[0]
            rank = len(static_shape(node))
            dim = int(node.args[1]) if len(node.args) > 1 else 0
            dim = dim + rank if dim < 0 else dim
            axis = _concat_axis(rank, dim, b.canon(node) in b.multihead_names)
            acc, size = b.tid(tensors[0]), static_shape(tensors[0])[dim]
            for i, tensor in enumerate(tensors[1:], 1):
                size += static_shape(tensor)[dim]
                if i == len(tensors) - 1:
                    out = b.tid(node)
                else:
                    partial = list(static_shape(node))
                    partial[dim] = size
                    out = b.temp(node.name + "_cat", partial)
                b.emit_to(hx.CONCAT, acc, b.tid(tensor), out, [axis])
                acc = out
            b.record(node, name)
        elif kind in ("mean", "reduce"):
            op, mask = _reduce_plan(node)
            if op == hx.RED_MEAN and mask == 3:  # spatial mean keeps the dedicated operation
                b.emit(hx.MEAN, b.tid(node.args[0]), NONE, node, [])
            else:
                b.emit(hx.REDUCE, b.tid(node.args[0]), NONE, node, [op, mask])
            b.record(node, name)
        elif kind == "logistic":
            b.emit(hx.LOGISTIC, b.tid(node.args[0]), NONE, node, [])
            b.record(node, name)
        elif kind == "softmax":
            _emit_softmax(b, ep, node)
        elif kind == "cast":
            _emit_cast(b, node)
            b.record(node, name)
        elif kind == "gather_elements":
            _emit_gather_elements(b, ep, node)
        elif kind == "scatter_rows":
            _emit_scatter_rows(b, ep, node)
        elif kind == "bn":
            if node.name not in b.folded_bn:
                scale, shift, _ = _bn_plan(node, ep, data=True)
                getitem = next(iter(node.users))
                x = node.args[0]
                out = b.tid(getitem)
                mid = out if not np.any(shift) else b.temp(node.name + "_scaled", static_shape(x))
                b.emit_to(hx.BINARY, b.tid(x), NONE, mid, [hx.BIN_MUL, 1, _pool_vec(b, scale), 0])
                if mid != out:
                    b.emit_to(hx.BINARY, mid, NONE, out, [hx.BIN_ADD, 1, _pool_vec(b, shift), 0])
                b.alias[node.name] = getitem.name
            b.record(node, name)
        elif kind == "embedding":
            w_arg, idx_arg = node.args[0], node.args[1]
            try:
                w_const = _constant(ep, w_arg, "embedding table")
            except Exception:
                w_const = None
            wshape = static_shape(w_arg)
            vocab = int(wshape[0])
            dim = int(wshape[1])
            stride = (dim + 31) & ~31
            if w_const is not None:
                tbl = np.asarray(w_const, dtype=np.float32).reshape(vocab, dim)
                if stride > dim:
                    tbl = np.pad(tbl, ((0, 0), (0, stride - dim)))
                offset = b.pool.emit(tbl.reshape(-1))
                mode = 1
                y_tid = NONE
                table_stride = stride
            else:
                mode = 0
                offset = 0
                y_tid = b.tid(w_arg)
                table_stride = stride
            b.emit(hx.GATHER, b.tid(idx_arg), y_tid, node,
                   [mode, offset, vocab, 0, 0, dim, 1, table_stride])
            b.record(node, name)
        elif kind == "index_select":
            _emit_index_select(b, ep, node, node.args[0], int(node.args[1]), node.args[2])
        elif kind == "index":
            plan = _index_plan(node, ep)
            if plan[0] == "axis":
                _emit_index_select(b, ep, node, node.args[0], plan[1], plan[2])
            elif plan[0] == "pixels":
                _emit_index_pixels(b, ep, node, plan[1], plan[2])
            else:
                x_arg, indices_list = node.args[0], node.args[1]
                if indices_list[0] is not None and all(v is None for v in indices_list[1:]):
                    idx_arg = indices_list[0]
                    xshape = static_shape(x_arg)
                    vocab = int(xshape[0])
                    inner_dim = int(np.prod(xshape[1:])) if len(xshape) > 1 else 1
                    stride = (inner_dim + 31) & ~31
                    try:
                        idx_const = np.asarray(_const_value(ep, idx_arg), dtype=np.int32) if _is_constant(ep, idx_arg) else None
                    except Exception:
                        idx_const = None
                    try:
                        x_const = _constant(ep, x_arg, "index table") if _is_constant(ep, x_arg) else None
                    except Exception:
                        x_const = None
                    if idx_const is not None:
                        idx_arr = idx_const.reshape(-1)
                        const_idx_off = b.pool.emit(idx_arr, "<i4")
                        const_idx_num = idx_arr.size
                        mode = 2
                        x_tid = b.tid(x_arg)
                        y_tid = NONE
                        b.emit(hx.GATHER, x_tid, y_tid, node,
                               [mode, 0, vocab, 0, 0, inner_dim, 1, 0, const_idx_off, const_idx_num])
                    elif x_const is not None:
                        tbl = np.asarray(x_const, dtype=np.float32).reshape(vocab, inner_dim)
                        if stride > inner_dim:
                            tbl = np.pad(tbl, ((0, 0), (0, stride - inner_dim)))
                        offset = b.pool.emit(tbl.reshape(-1))
                        mode = 1
                        y_tid = NONE
                        table_stride = stride
                        b.emit(hx.GATHER, b.tid(idx_arg), y_tid, node,
                               [mode, offset, vocab, 0, 0, inner_dim, 1, table_stride])
                    else:
                        mode = 0
                        offset = 0
                        y_tid = b.tid(x_arg)
                        table_stride = stride
                        b.emit(hx.GATHER, b.tid(idx_arg), y_tid, node,
                               [mode, offset, vocab, 0, 0, inner_dim, 1, table_stride])
                    b.record(node, name)
                else:
                    raise Unsupported("only single-index aten.index.Tensor supported")
        elif kind == "activation":
            if act_code(node) is None:  # general clamp or hardtanh: never fused
                _emit_clamp(b, node, *_clamp_plan(node, ep))
                b.record(node, name)
                done.add(node)
                continue
            prod = fusable_producer(node, done, activation_of, ep)
            act = act_code(node)
            if prod is None:  # not fusable: run it on its own
                b.emit(hx.ACT, b.tid(node.args[0]), NONE, node,
                       [hx.FN_RELU if act == ACT_RELU else hx.FN_RELU6])
                b.record(node, name)
                done.add(node)
                continue
            activation_of[prod] = act
            op = b.op_of_out[b.tid(prod)]
            if op[0] == hx.ADD:
                op[5] = act  # p[0]
            else:
                op[5 + 8] = act  # p[8] of convolution and depthwise
            b.alias[node.name] = prod.name
            b.source_ops.append({"op": name, "fused_into": prod.name})
        else:  # pragma: no cover
            raise Unsupported(f"{node.name}: unsupported kind {kind}")
        done.add(node)

    outputs = list(ep.graph.output_node().args[0])
    for o in outputs:
        if not (isinstance(o, torch.fx.Node) and o in done):
            raise Unsupported("delegate outputs must be delegated operation results")
    out_ids = [b.tid(o) for o in outputs]
    if len(set(out_ids)) != len(out_ids):
        raise Unsupported("duplicate delegate outputs")

    names = {v: k for k, v in b.ids.items()}
    compact_ids = {b.ids[n.name] for n in graph_inputs if n.name in compact}

    def hwc(i):
        return hwc_of(b.shape[i], names.get(i, f"tensor {i}"), multihead=(i in b.multihead))

    def stride(i):
        return hwc(i)[2] if i in compact_ids else (hwc(i)[2] + 31) & ~31

    manifest = {
        "source": "executorch-partition",
        "source_sha256": hashlib.sha256(repr(b.source_ops).encode()).hexdigest(),
        "ops": b.source_ops,
        "shapes": {i: list(s) for i, s in b.shape.items()},
    }
    manifest["extra"] = {"state_names": {str(k): v for k, v in sorted(b.state_names.items())}}
    b.ops = _by_component(b.ops)
    b.ops = _demand_order(b.ops, out_ids, lambda i: hwc(i)[0] * hwc(i)[1] * stride(i) * 4, bool(b.persistent))
    result = hx.write(b.ops, input_ids, out_ids, hwc, stride, b.pool, Path(destination), manifest,
                      dtypes=b.dtypes.get, persistent=b.persistent)
    image = Path(destination).with_suffix(".hxfg").read_bytes()
    constants = Path(destination).with_suffix(".bin").read_bytes()
    check_image(image, constants)
    summary = {
        "inputs": [list(static_shape(n)) for n in graph_inputs],
        "outputs": [list(static_shape(o)) for o in outputs],
        "operations": len(b.ops),
        "arena_bytes": int.from_bytes(image[28:32], "little"),
        "constant_bytes": len(constants),
        "state_bytes": result["state_bytes"],
        "state_names": result["state_names"],
    }
    return summary, image, constants


def check_image(image, constants):
    """Structural check of an HXFG v1 image, mirroring hexecutor/native/tensor/graph.h."""
    if len(image) < 256:
        raise Unsupported("HXFG image shorter than its header")
    (magic, version, size, tensors, ops, n_in, n_out, _arena,
     pool_bytes, to, oo, io_at) = struct.unpack_from("<12I", image, 0)
    if magic != 0x47465848 or version != 1:
        raise Unsupported("bad HXFG magic or version")
    if size != len(image) or pool_bytes != len(constants):
        raise Unsupported("HXFG size fields disagree with the files")
    if to != 256 or oo != to + 32 * tensors or io_at != oo + 64 * ops:
        raise Unsupported("HXFG section offsets are inconsistent")
    if size != io_at + 4 * (n_in + n_out):
        raise Unsupported("HXFG I/O table size mismatch")
    if hashlib.sha256(constants).digest() != image[48:80]:
        raise Unsupported("HXFG constant digest mismatch")
    if not (tensors and ops and n_in and n_out):
        raise Unsupported("HXFG graph is empty")
    return True
