# SPDX-License-Identifier: LGPL-3.0-or-later
"""Rewrite ONNX operators the HXFG operation set has no kind for into operators it does have.

`decompose(model)` edits `model.graph` in place before `lower_onnx_float.Lowerer` infers shapes.
Every rewrite is numerically equivalent (the 0/1 weight forms are exact in float32):

- ReduceL1, ReduceL2, ReduceSumSquare, ReduceLogSumExp: Abs/Mul/Exp/Log/Sqrt around ReduceSum
  (and a max-shift for LogSumExp).
- LogSoftmax: x - max - log(sum(exp(x - max))), accurate for very negative logits.
- DepthToSpace (DCR and CRD): a ConvTranspose with kernel = stride = blocksize and a 0/1 weight.
  Cost grows with C * blocksize^4 per input pixel; fine for super-resolution heads.
- SpaceToDepth: a Conv with kernel = stride = blocksize and a 0/1 weight.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _opset(model):
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    return 13


class _Builder:
    def __init__(self, model):
        self.model = model
        self.opset = _opset(model)
        self.count = 0
        self.new = []

    def name(self, hint):
        self.count += 1
        return f"{hint}__dc{self.count}"

    def const(self, value, hint="c"):
        name = self.name(hint)
        self.model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def node(self, op, inputs, hint, **attrs):
        out = self.name(hint)
        self.new.append(helper.make_node(op, list(inputs), [out], name=out + "_n", **attrs))
        return out

    def node_to(self, op, inputs, output, **attrs):
        self.new.append(helper.make_node(op, list(inputs), [output], name=output + "_n", **attrs))

    def reduce(self, op, x, axes, keepdims, hint, output=None):
        """A Reduce node with `axes` in whichever form (attribute or input) the opset uses."""
        as_input = self.opset >= (13 if op == "ReduceSum" else 18)
        inputs, attrs = [x], {"keepdims": keepdims}
        if as_input:
            inputs.append(self.const(np.asarray(axes, np.int64), "axes"))
        else:
            attrs["axes"] = list(axes)
        if output:
            self.node_to(op, inputs, output, **attrs)
            return output
        return self.node(op, inputs, hint, **attrs)


def _axes(b, node):
    for a in node.attribute:
        if a.name == "axes":
            return [int(v) for v in a.ints]
    if len(node.input) > 1 and node.input[1]:
        for init in b.model.graph.initializer:
            if init.name == node.input[1]:
                return [int(v) for v in numpy_helper.to_array(init).reshape(-1)]
    return None


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def decompose(model):
    b = _Builder(model)
    rewritten = []
    for node in list(model.graph.node):
        b.new = []
        out = node.output[0]
        if node.op_type in ("ReduceL1", "ReduceL2", "ReduceSumSquare"):
            axes = _axes(b, node)
            if axes is None:
                rewritten.append(node)
                continue
            x = node.input[0]
            keep = int(_attr(node, "keepdims", 1))
            pre = {"ReduceL1": ("Abs", [x]), "ReduceL2": ("Mul", [x, x]),
                   "ReduceSumSquare": ("Mul", [x, x])}[node.op_type]
            v = b.node(pre[0], pre[1], "pre")
            if node.op_type == "ReduceL2":
                s = b.reduce("ReduceSum", v, axes, keep, "sum")
                b.node_to("Sqrt", [s], out)
            else:
                b.reduce("ReduceSum", v, axes, keep, "sum", output=out)
        elif node.op_type == "ReduceLogSumExp":
            axes = _axes(b, node)
            if axes is None:
                rewritten.append(node)
                continue
            x, keep = node.input[0], int(_attr(node, "keepdims", 1))
            m = b.reduce("ReduceMax", x, axes, 1, "max")
            e = b.node("Exp", [b.node("Sub", [x, m], "shift")], "exp")
            lg = b.node("Log", [b.reduce("ReduceSum", e, axes, 1, "sum")], "log")
            if keep:
                b.node_to("Add", [lg, m], out)
            else:
                total = b.node("Add", [lg, m], "lse")
                if b.opset >= 13:
                    b.node_to("Squeeze", [total, b.const(np.asarray(axes, np.int64), "axes")], out)
                else:
                    b.node_to("Squeeze", [total], out, axes=list(axes))
        elif node.op_type == "LogSoftmax":
            x = node.input[0]
            axis = int(_attr(node, "axis", -1 if b.opset >= 13 else 1))
            m = b.reduce("ReduceMax", x, [axis], 1, "max")
            d = b.node("Sub", [x, m], "shift")
            lg = b.node("Log", [b.reduce("ReduceSum", b.node("Exp", [d], "exp"), [axis], 1, "sum")],
                        "log")
            b.node_to("Sub", [d, lg], out)
        elif node.op_type == "DepthToSpace":
            r = int(_attr(node, "blocksize", 0))
            mode = _attr(node, "mode", b"DCR")
            mode = mode.decode() if isinstance(mode, bytes) else mode
            shape = _static_channels(model, node.input[0])
            if r < 1 or shape is None or shape % (r * r):
                rewritten.append(node)
                continue
            c = shape // (r * r)
            w = np.zeros((shape, c, r, r), np.float32)
            for ky in range(r):
                for kx in range(r):
                    for co in range(c):
                        ci = (ky * r + kx) * c + co if mode == "DCR" else co * r * r + ky * r + kx
                        w[ci, co, ky, kx] = 1.0
            b.node_to("ConvTranspose", [node.input[0], b.const(w, "d2s_w"),
                                        b.const(np.zeros(c, np.float32), "d2s_b")], out,
                      kernel_shape=[r, r], strides=[r, r])
        elif node.op_type == "SpaceToDepth":
            r = int(_attr(node, "blocksize", 0))
            c = _static_channels(model, node.input[0])
            if r < 1 or c is None:
                rewritten.append(node)
                continue
            w = np.zeros((c * r * r, c, r, r), np.float32)
            for ky in range(r):
                for kx in range(r):
                    for ch in range(c):
                        w[(ky * r + kx) * c + ch, ch, ky, kx] = 1.0
            b.node_to("Conv", [node.input[0], b.const(w, "s2d_w"),
                               b.const(np.zeros(c * r * r, np.float32), "s2d_b")], out,
                      kernel_shape=[r, r], strides=[r, r])
        else:
            rewritten.append(node)
            continue
        rewritten.extend(b.new)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return model


def _static_channels(model, name):
    """Channel count (axis 1) of a tensor from graph inputs or value_info, if known."""
    for info in list(model.graph.input) + list(model.graph.value_info):
        if info.name == name:
            dims = info.type.tensor_type.shape.dim
            if len(dims) == 4 and dims[1].dim_value:
                return dims[1].dim_value
    inferred = onnx.shape_inference.infer_shapes(model)
    for info in list(inferred.graph.value_info) + list(inferred.graph.input):
        if info.name == name:
            dims = info.type.tensor_type.shape.dim
            if len(dims) == 4 and dims[1].dim_value:
                return dims[1].dim_value
    return None
