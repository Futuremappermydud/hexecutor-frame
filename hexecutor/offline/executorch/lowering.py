"""Edge-dialect ExecuTorch graph -> HXFG v1 lowering for the hexecutor backend.

Shared by the partitioner (which asks `kind_of` / `select_nodes`) and the
backend (which calls `lower_program`). Every rule is fail-closed: a node that
cannot be expressed exactly in the tensor_f32 HXFG operation set raises
`Unsupported`; the partitioner leaves such nodes on the CPU.

Layout: PyTorch tensors are NCHW with batch one. HXFG stores each tensor as
HWC with channels padded to 32 floats; graph inputs whose consumers are all
convolutions may use compact 3-channel storage. A rank-2 `[1, C]` tensor is
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
from pathlib import Path

import numpy as np
import torch

from .. import hxfg_writer as hx

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
    "view_copy.default": "reshape",
    "cat.default": "concat",
    "mean.dim": "mean",
    "sigmoid.default": "logistic",
    "relu.default": "activation",
    "hardtanh.default": "activation",
    "clamp.default": "activation",
    "getitem": "getitem",
    "sub.Tensor": "binary",
    "mul.Tensor": "binary",
    "hardswish.default": "act",
    "hardsigmoid.default": "act",
    "silu.default": "act",
    "tanh.default": "act",
    "leaky_relu.default": "act",
    "gelu.default": "act",
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


# ---------------------------------------------------------------- metadata


def static_shape(node):
    """Static float32 tensor shape from the exported FakeTensor metadata."""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise Unsupported(f"{node.name}: not a tensor")
    if val.dtype != torch.float32:
        raise Unsupported(f"{node.name}: dtype {val.dtype} is not float32")
    if not all(isinstance(d, int) for d in val.shape):
        raise Unsupported(f"{node.name}: dynamic shape")
    return tuple(int(d) for d in val.shape)


def hwc_of(shape, name="tensor"):
    """HXFG [H, W, C] for a batch-one rank-4 or rank-2 PyTorch shape."""
    shape = tuple(shape)
    if len(shape) == 4 and shape[0] == 1:
        h, w, c = shape[2], shape[3], shape[1]
    elif len(shape) == 2 and shape[0] == 1:
        h, w, c = 1, 1, shape[1]
    else:
        raise Unsupported(f"{name}: shape {shape} is not batch-one rank 2 or 4")
    if min(h, w, c) < 1 or c > MAX_CHANNELS:
        raise Unsupported(f"{name}: unsupported size {shape}")
    return [h, w, c]


def _constant_fqn(ep, node):
    """Qualified name of the constant behind a placeholder, or None for runtime values."""
    if node.op != "placeholder":
        return None
    sig = ep.graph_signature
    return (
        sig.inputs_to_parameters.get(node.name)
        or sig.inputs_to_buffers.get(node.name)
        or sig.inputs_to_lifted_tensor_constants.get(node.name)
    )


def _is_constant(ep, node):
    return isinstance(node, torch.fx.Node) and _constant_fqn(ep, node) is not None


def _const_value(ep, node):
    """The constant data behind a placeholder (lowering only; checks never read data)."""
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


def _spatial_trivial(shape):
    return len(shape) == 2 or (len(shape) == 4 and shape[2] * shape[3] == 1)


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
    if prod not in done or prod in activation_of:
        return None
    if op_name(prod.target) not in FUSABLE:
        return None
    if op_name(prod.target) == "add.Tensor" and not _plain_add(prod, ep):
        return None  # lowered to HX_BINARY, which has no fused activation
    if len(prod.users) != 1 or next(iter(prod.users)) is not node:
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
}


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
    opcode = {"add.Tensor": hx.BIN_ADD, "sub.Tensor": hx.BIN_SUB, "mul.Tensor": hx.BIN_MUL}[name]
    a, b = node.args[0], node.args[1]
    alpha = node.args[2] if len(node.args) > 2 else node.kwargs.get("alpha", 1)
    if alpha != 1:
        raise Unsupported(f"{name} alpha must be one")
    out = static_shape(node)
    hwc_of(out, node.name)
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
        shape = (1,) * (len(out) - len(cshape)) + tuple(cshape)
        if len(shape) != len(out) or shape[0] != 1:
            raise Unsupported("constant operand rank")
        channels = out[1]
        if len(out) == 4 and (shape[2] != 1 or shape[3] != 1):
            raise Unsupported("constant operand varies spatially")
        if shape[1] not in (1, channels):
            raise Unsupported("constant operand channel mismatch")
        vec = (
            np.broadcast_to(const.reshape(-1), (channels,)).astype(np.float32)
            if data
            else None
        )
        return opcode, tensor, None, vec, reversed_
    sa, sb = static_shape(a), static_shape(b)
    if len(sa) != len(sb):
        raise Unsupported("operands of different rank")

    def fits(small, big):  # every dim of small is big's or one; batch is one
        return len(small) == len(big) and small[0] == 1 and all(
            s in (1, o) for s, o in zip(small, big)
        )

    if sa == out and fits(sb, out):
        return opcode, a, b, None, 0
    if sb == out and fits(sa, out):
        return opcode, b, a, None, 1 if opcode == hx.BIN_SUB else 0
    raise Unsupported("operands broadcast against each other")


def _plain_add(node, ep):
    """Same-shape tensor + tensor, which keeps the original HX_ADD with fusable activation."""
    try:
        opcode, x, y, vec, rev = _binary_plan(node, ep)
    except Unsupported:
        return False
    return opcode == hx.BIN_ADD and y is not None and static_shape(x) == static_shape(y)


# ---------------------------------------------------------------- classification


def kind_of(node, ep):
    """Kind of a supported node, or None. Raises Unsupported for failed static checks."""
    kind = kind_name(node)
    if kind is None:
        return None
    check(kind, node, ep)
    return kind


def check(kind, node, ep):
    args = node.args
    if kind == "conv":
        x, w = args[0], args[1]
        out = static_shape(node)
        ishape = static_shape(x)
        if len(args) > 6 and args[6]:
            raise Unsupported("transposed convolution")
        if _ints(args[5], 2) != [1, 1]:
            raise Unsupported("dilated convolution")
        if len(args) > 7 and any(int(v) for v in args[7]):
            raise Unsupported("output padding")
        groups = int(args[8]) if len(args) > 8 else 1
        wshape = static_shape(w)
        if len(wshape) != 4:
            raise Unsupported("convolution weight must be rank 4")
        depthwise = groups == wshape[0] == ishape[1] and wshape[1] == 1 and groups > 1
        if groups != 1 and not depthwise:
            raise Unsupported("only dense or multiplier-one depthwise convolution")
        if groups == 1 and wshape[1] != ishape[1]:
            raise Unsupported("input channel mismatch")
        _require_constant(ep, w, "convolution weight")
        if len(args) > 2 and args[2] is not None:
            _require_constant(ep, args[2], "convolution bias")
        hwc_of(ishape, x.name)
        hwc_of(out, node.name)
    elif kind == "linear":
        x, w = args[0], args[1]
        if len(static_shape(x)) != 2:
            raise Unsupported("linear input must be rank 2")
        _require_constant(ep, w, "linear weight")
        if len(args) > 2 and args[2] is not None:
            _require_constant(ep, args[2], "linear bias")
        hwc_of(static_shape(node), node.name)
    elif kind in ("add", "binary"):
        _binary_plan(node, ep)
    elif kind == "act":
        hwc_of(static_shape(args[0]), args[0].name)
        hwc_of(static_shape(node), node.name)
        if op_name(node.target) == "gelu.default" and (
            node.kwargs.get("approximate", "none") != "none"
        ):
            raise Unsupported("only exact (erf) GELU")
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
    elif kind == "getitem":
        prod = args[0]
        if not (
            isinstance(prod, torch.fx.Node)
            and op_name(prod.target) == "max_pool2d_with_indices.default"
            and args[1] == 0
        ):
            raise Unsupported("only the pooled value of max pooling is supported")
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
            raise Unsupported("bilinear resize must keep rank 4 and channels")
        hwc_of(out, node.name)
    elif kind == "reshape":
        x = args[0]
        ishape, out = static_shape(x), static_shape(node)
        hwc_of(ishape, x.name)
        hwc_of(out, node.name)
        # Only channel-only layouts keep NCHW element order under HWC storage.
        if not (_spatial_trivial(ishape) and _spatial_trivial(out)):
            raise Unsupported("reshape mixes spatial and channel order")
    elif kind == "concat":
        tensors = args[0]
        if len(tensors) != 2 or not all(isinstance(t, torch.fx.Node) for t in tensors):
            raise Unsupported("only two-input concatenation")
        out = static_shape(node)
        dim = int(args[1]) if len(args) > 1 else 0
        dim = dim + len(out) if dim < 0 else dim
        if len(out) not in (2, 4) or (len(out) == 2 and dim != 1) or (len(out) == 4 and dim == 0):
            raise Unsupported("concatenation along an unsupported dimension")
        axis = 2 if len(out) == 2 else {1: 2, 2: 0, 3: 1}[dim]
        shapes = [static_shape(t) for t in tensors]
        hwc = [hwc_of(s, t.name) for s, t in zip(shapes, tensors)]
        ohwc = hwc_of(out, node.name)
        for i in range(3):
            if i != axis and not all(h[i] == ohwc[i] for h in hwc):
                raise Unsupported("concatenation shape mismatch")
        if sum(h[axis] for h in hwc) != ohwc[axis]:
            raise Unsupported("concatenation size mismatch")
    elif kind == "mean":
        x = args[0]
        ishape, out = static_shape(x), static_shape(node)
        dims = sorted(int(v) % 4 for v in args[1])
        if dims != [2, 3] or len(ishape) != 4:
            raise Unsupported("only spatial mean on rank 4")
        if out not in ((1, ishape[1], 1, 1), (1, ishape[1])):
            raise Unsupported("unexpected mean output shape")
        hwc_of(ishape, x.name)
    elif kind == "logistic":
        hwc_of(static_shape(args[0]), args[0].name)
        hwc_of(static_shape(node), node.name)
    elif kind == "activation":
        if act_code(node) is None:
            raise Unsupported("only ReLU and ReLU6 can be fused")
        hwc_of(static_shape(node), node.name)
    else:  # pragma: no cover - table and checks are kept in step
        raise Unsupported(kind)


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
    if len(chosen) < MIN_OPS:
        return []
    return chosen


# ---------------------------------------------------------------- lowering


class _Builder:
    def __init__(self, ep):
        self.ep = ep
        self.ids, self.alias, self.shape = {}, {}, {}
        self.ops, self.op_of_out = [], {}
        self.pool = hx.ConstantPool()
        self.source_ops = []

    def canon(self, node):
        name = node.name
        while name in self.alias:
            name = self.alias[name]
        return name

    def tid(self, node):
        name = self.canon(node)
        if name not in self.ids:
            self.ids[name] = len(self.ids)
            self.shape[self.ids[name]] = static_shape(node)
        return self.ids[name]

    def emit(self, kind, x, y, out_node, p):
        out = self.tid(out_node)
        p = (list(p) + [0] * 11)[:11]
        op = [kind, x, y, NONE, out] + p
        self.ops.append(op)
        self.op_of_out[out] = op
        return out

    def record(self, node, name):
        self.source_ops.append({"op": name, "output": node.name})


def _emit_conv(b, ep, node):
    x = node.args[0]
    w = _constant(ep, node.args[1], "convolution weight")
    bias_node = node.args[2] if len(node.args) > 2 else None
    stride = _ints(node.args[3], 2)
    pad = _ints(node.args[4], 2)
    groups = int(node.args[8]) if len(node.args) > 8 else 1
    ishape, oshape = static_shape(x), static_shape(node)
    co = oshape[1]
    kh, kw = w.shape[2], w.shape[3]
    ostride = (co + 31) & ~31
    if bias_node is None:
        bias = np.zeros(co, np.float32)
    else:
        bias = _constant(ep, bias_node, "convolution bias").reshape(-1)
        if bias.size != co:
            raise Unsupported("bias size mismatch")
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
    b.emit(kind, b.tid(x), NONE, node, p)
    b.record(node, "depthwise_conv2d" if kind == hx.DEPTHWISE else "conv2d")


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


def lower_program(ep, destination):
    """Lower a delegated edge-dialect partition.

    Every call_function node in `ep` must be supported; the caller passes only
    delegated nodes (preprocess receives exactly the tagged subgraph).
    Returns (summary dict, HXFG image bytes, constant pool bytes).
    """
    b = _Builder(ep)
    graph_inputs = [
        n for n in ep.graph.nodes if n.op == "placeholder" and not _is_constant(ep, n)
    ]
    if not graph_inputs:
        raise Unsupported("partition has no runtime inputs")
    for n in graph_inputs:
        hwc_of(static_shape(n), n.name)
    compact = {
        n.name
        for n in graph_inputs
        if len(static_shape(n)) == 4
        and static_shape(n)[1] == 3
        and n.users
        and all(kind_name(u) == "conv" for u in n.users)
    }
    for n in graph_inputs:
        b.tid(n)
    input_ids = [b.tid(n) for n in graph_inputs]

    done, activation_of = set(), {}
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        kind = kind_of(node, ep)
        if kind is None:
            raise Unsupported(f"{node.name}: operation {op_name(node.target)} is not supported")
        name = op_name(node.target)
        if kind == "conv":
            _emit_conv(b, ep, node)
        elif kind == "linear":
            _emit_linear(b, ep, node)
        elif kind in ("add", "binary"):
            opcode, x, y, vec, rev = _binary_plan(node, ep, data=True)
            if _plain_add(node, ep):
                b.emit(hx.ADD, b.tid(x), b.tid(y), node, [0])
            elif y is not None:
                b.emit(hx.BINARY, b.tid(x), b.tid(y), node, [opcode, 0, 0, rev])
            else:
                b.emit(hx.BINARY, b.tid(x), NONE, node,
                       [opcode, 1, b.pool.emit(_pad32(vec)), rev])
            b.record(node, name)
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
        elif kind == "getitem":
            pass  # the maxpool already produced this value
        elif kind == "pad":
            wl, wr, ht, hb = [int(v) for v in node.args[1]]
            b.emit(hx.PAD, b.tid(node.args[0]), NONE, node, [ht, wl, 0])
            b.record(node, name)
        elif kind == "resize":
            align = bool(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else False
            b.emit(hx.RESIZE, b.tid(node.args[0]), NONE, node, [int(align), int(not align)])
            b.record(node, name)
        elif kind == "reshape":
            b.emit(hx.RESHAPE, b.tid(node.args[0]), NONE, node, [])
            b.record(node, name)
        elif kind == "concat":
            tensors = node.args[0]
            rank = len(static_shape(node))
            dim = int(node.args[1]) if len(node.args) > 1 else 0
            dim = dim + rank if dim < 0 else dim
            axis = 2 if rank == 2 else {1: 2, 2: 0, 3: 1}[dim]
            b.emit(hx.CONCAT, b.tid(tensors[0]), b.tid(tensors[1]), node, [axis])
            b.record(node, name)
        elif kind == "mean":
            b.emit(hx.MEAN, b.tid(node.args[0]), NONE, node, [])
            b.record(node, name)
        elif kind == "logistic":
            b.emit(hx.LOGISTIC, b.tid(node.args[0]), NONE, node, [])
            b.record(node, name)
        elif kind == "activation":
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
        return hwc_of(b.shape[i], names.get(i, f"tensor {i}"))

    def stride(i):
        return 3 if i in compact_ids else (hwc(i)[2] + 31) & ~31

    manifest = {
        "source": "executorch-partition",
        "source_sha256": hashlib.sha256(repr(b.source_ops).encode()).hexdigest(),
        "ops": b.source_ops,
        "shapes": {i: list(s) for i, s in b.shape.items()},
    }
    hx.write(b.ops, input_ids, out_ids, hwc, stride, b.pool, Path(destination), manifest)
    image = Path(destination).with_suffix(".hxfg").read_bytes()
    constants = Path(destination).with_suffix(".bin").read_bytes()
    check_image(image, constants)
    summary = {
        "inputs": [list(static_shape(n)) for n in graph_inputs],
        "outputs": [list(static_shape(o)) for o in outputs],
        "operations": len(b.ops),
        "arena_bytes": int.from_bytes(image[28:32], "little"),
        "constant_bytes": len(constants),
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
