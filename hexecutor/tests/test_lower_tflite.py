# SPDX-License-Identifier: LGPL-3.0-or-later
"""TFLite -> HXFG lowering: single-operation models against NumPy references (synthetic models only).

Needs numpy and the offline `tflite` schema package plus `flatbuffers` (models are built with the
generated builders, no TensorFlow). The numeric tests also need the portable reference runner
(build/bin/hexecutor_tensor_run); each lowered graph runs on that CPU reference and is compared with
a NumPy implementation of the TFLite operator. Skips when a dependency is missing.
"""

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    import flatbuffers
    import tflite
    from hexecutor.offline import lower_tflite as lowering
except ImportError:  # offline tooling dependency
    tflite = None

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(os.environ.get("HEXECUTOR_TENSOR_RUN", ROOT / "build/bin/hexecutor_tensor_run"))
NUMERIC = unittest.skipUnless(tflite and RUNNER.exists(), "needs tflite, flatbuffers and the runner")
ERF = np.vectorize(math.erf)


class Model:
    """Minimal single-subgraph TFLite flatbuffer writer."""

    def __init__(self):
        self.tensors, self.buffers, self.ops, self.codes = [], [b""], [], []

    def tensor(self, shape, kind="f"):
        ttype = {"f": tflite.TensorType.FLOAT32, "i": tflite.TensorType.INT32,
                 "b": tflite.TensorType.BOOL}[kind]
        self.tensors.append((list(shape), ttype, 0))
        return len(self.tensors) - 1

    def const(self, array):
        array = np.asarray(array)
        if array.dtype.kind == "f":
            array, ttype = array.astype("<f4"), tflite.TensorType.FLOAT32
        else:
            array, ttype = array.astype("<i4"), tflite.TensorType.INT32
        self.buffers.append(array.tobytes())
        self.tensors.append((list(array.shape), ttype, len(self.buffers) - 1))
        return len(self.tensors) - 1

    def op(self, name, ins, outs, options=None, **fields):
        """options: name of the *Options table (e.g. "SoftmaxOptions"); fields its scalar fields."""
        code = getattr(tflite.BuiltinOperator, name)
        if code not in self.codes:
            self.codes.append(code)
        self.ops.append((self.codes.index(code), ins, outs, options, fields))

    def build(self, inputs, outputs):
        b = flatbuffers.Builder(1024)

        def ivec(start, values):
            start(b, len(values))
            for v in reversed(values):
                b.PrependInt32(v)
            return b.EndVector()

        def table(cls, fields):
            getattr(tflite, cls + "Start")(b)
            for key, value in fields.items():
                getattr(tflite, f"{cls}Add{key}")(b, value)
            return getattr(tflite, cls + "End")(b)

        buffers = []
        for data in self.buffers:
            vec = b.CreateNumpyVector(np.frombuffer(data, np.uint8)) if data else None
            tflite.BufferStart(b)
            if vec is not None:
                tflite.BufferAddData(b, vec)
            buffers.append(tflite.BufferEnd(b))
        tensors = []
        for shape, ttype, buf in self.tensors:
            sv = ivec(tflite.TensorStartShapeVector, shape)
            tensors.append(table("Tensor", {"Shape": sv, "Type": ttype, "Buffer": buf}))
        codes = [table("OperatorCode", {"DeprecatedBuiltinCode": min(c, 127), "BuiltinCode": c,
                                        "Version": 1}) for c in self.codes]
        ops = []
        for code, ins, outs, options, fields in self.ops:
            iv = ivec(tflite.OperatorStartInputsVector, ins)
            ov = ivec(tflite.OperatorStartOutputsVector, outs)
            opts = table(options, fields) if options else None
            tflite.OperatorStart(b)
            tflite.OperatorAddOpcodeIndex(b, code)
            tflite.OperatorAddInputs(b, iv)
            tflite.OperatorAddOutputs(b, ov)
            if opts is not None:
                tflite.OperatorAddBuiltinOptionsType(b, getattr(tflite.BuiltinOptions, options))
                tflite.OperatorAddBuiltinOptions(b, opts)
            ops.append(tflite.OperatorEnd(b))

        def vec_of(start, items):
            start(b, len(items))
            for v in reversed(items):
                b.PrependUOffsetTRelative(v)
            return b.EndVector()

        gi = ivec(tflite.SubGraphStartInputsVector, inputs)
        go = ivec(tflite.SubGraphStartOutputsVector, outputs)
        sub = table("SubGraph", {
            "Tensors": vec_of(tflite.SubGraphStartTensorsVector, tensors),
            "Inputs": gi, "Outputs": go,
            "Operators": vec_of(tflite.SubGraphStartOperatorsVector, ops)})
        model = table("Model", {
            "Version": 3,
            "OperatorCodes": vec_of(tflite.ModelStartOperatorCodesVector, codes),
            "Subgraphs": vec_of(tflite.ModelStartSubgraphsVector, [sub]),
            "Buffers": vec_of(tflite.ModelStartBuffersVector, buffers)})
        b.Finish(model)
        return bytes(b.Output())


def run(model, inputs, *args):
    """Lower `model` (bytes) and run it on the portable reference; returns the output arrays.

    Shapes are the ones the model declared: the caller reshapes. Outputs of int32/bool tensors
    come back as int32 (bool as the same 0/1 integers)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "m.tflite").write_bytes(model)
        lowering.lower(tmp / "m.tflite", tmp / "g")
        files = []
        for i, a in enumerate(inputs):
            a = np.asarray(a)
            dt = np.int32 if a.dtype.kind in "ib" else np.float32
            (tmp / f"in{i}.bin").write_bytes(np.ascontiguousarray(a, dt).tobytes())
            files.append(str(tmp / f"in{i}.bin"))
        subprocess.run([str(RUNNER), str(tmp / "g.hxfg"), str(tmp / "g.bin"), *files,
                        str(tmp / "out_")], check=True, capture_output=True, text=True)
        meta = json.loads((tmp / "g.json").read_text())
        out, i = [], 0
        while (tmp / f"out_{i}.f32").exists():
            dt = np.int32 if meta["outputs"][i]["dtype"] else np.float32
            out.append(np.fromfile(tmp / f"out_{i}.f32", dt))
            i += 1
        kinds = [o["kind"] for o in meta["ops"]]
        return out, kinds


def lower_only(model):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "m.tflite").write_bytes(model)
        return lowering.lower(Path(tmp, "m.tflite"), Path(tmp, "g"))


RNG = np.random.default_rng(7)


def rand(*shape, lo=-1.5, hi=1.5):
    return RNG.uniform(lo, hi, shape).astype(np.float32)


class Case(unittest.TestCase):
    def check(self, model, inputs, expected, tol=1e-5):
        out, _ = run(model, inputs)
        expected = expected if isinstance(expected, list) else [expected]
        self.assertEqual(len(out), len(expected))
        for got, want in zip(out, expected):
            want = np.asarray(want)
            self.assertEqual(got.size, want.size)
            if want.dtype.kind in "ib":
                np.testing.assert_array_equal(got, want.astype(np.int32).reshape(-1))
            else:
                np.testing.assert_allclose(got, want.reshape(-1), rtol=1e-4, atol=tol)

    def single(self, name, shape, expected_fn, options=None, extra_consts=(), kind="f", out_kind="f",
               x=None, tol=1e-5, **fields):
        """One float input and constant extras (in order after the input) -> one output."""
        x = rand(*shape) if x is None else x
        m = Model()
        xi = m.tensor(shape, kind)
        consts = [m.const(c) for c in extra_consts]
        want = expected_fn(x)
        out = m.tensor(np.shape(want), out_kind)
        m.op(name, [xi] + consts, [out], options, **fields)
        self.check(m.build([xi], [out]), [x], want, tol)


@NUMERIC
class ShapeOps(Case):
    def test_softmax(self):
        def ref(x):
            e = np.exp(x - x.max(-1, keepdims=True))
            return e / e.sum(-1, keepdims=True)
        self.single("SOFTMAX", (1, 3, 4, 10), ref, "SoftmaxOptions", Beta=1.0)
        self.single("SOFTMAX", (3, 7), ref, "SoftmaxOptions", Beta=1.0)

    def test_resize_nearest(self):
        def ref(size, align, half):
            def pick(i, o):
                o_idx = np.arange(o, dtype=np.float32)
                if align:
                    src = np.floor(o_idx * np.float32((i - 1) / (o - 1)) + np.float32(0.5))
                elif half:
                    src = np.floor((o_idx + np.float32(0.5)) * np.float32(i / o))
                else:
                    src = np.floor(o_idx * np.float32(i / o))
                return np.minimum(src, i - 1).astype(int)

            return lambda x: x[:, pick(4, size[0])][:, :, pick(4, size[1])]

        for size, align, half in (((6, 6), 0, 0), ((8, 5), 0, 1), ((7, 7), 1, 0), ((3, 3), 0, 0)):
            self.single("RESIZE_NEAREST_NEIGHBOR", (1, 4, 4, 4), ref(size, align, half),
                        "ResizeNearestNeighborOptions", [np.array(size, np.int32)],
                        AlignCorners=bool(align), HalfPixelCenters=bool(half))

    def test_three_channel_input(self):
        # A three-channel graph input feeding anything but a compact-eligible conv is channel-padded.
        x = rand(1, 4, 4, 3)
        self.single("RESIZE_NEAREST_NEIGHBOR", (1, 4, 4, 3), lambda v: v[:, [0, 0, 1, 2, 2, 3]][:, :, [0, 0, 1, 2, 2, 3]],
                    "ResizeNearestNeighborOptions", [np.array((6, 6), np.int32)], x=x)
        # A 3x3 VALID conv is the compact-RGB consumer and keeps the stride-3 layout.
        w, bias = rand(8, 3, 3, 3), rand(8)
        m = Model()
        xi, wi, bi, out = m.tensor(x.shape), m.const(w), m.const(bias), m.tensor((1, 2, 2, 8))
        m.op("CONV_2D", [xi, wi, bi], [out], "Conv2DOptions", Padding=1, StrideH=1, StrideW=1,
             DilationHFactor=1, DilationWFactor=1)
        want = np.zeros((1, 2, 2, 8), np.float32)
        for oy in range(2):
            for ox in range(2):
                want[0, oy, ox] = np.einsum("hwi,ohwi->o", x[0, oy:oy + 3, ox:ox + 3], w) + bias
        self.check(m.build([xi], [out]), [x], want, 1e-4)

    def test_transpose(self):
        for shape, perm in (((1, 3, 4, 5), [0, 2, 3, 1]), ((1, 3, 4, 5), [0, 3, 1, 2]),
                            ((1, 3, 4, 5), [0, 2, 1, 3]), ((3, 4, 5), [2, 0, 1]),
                            ((3, 4, 5), [1, 0, 2]), ((4, 6), [1, 0])):
            with self.subTest(shape=shape, perm=perm):
                self.single("TRANSPOSE", shape, lambda x, perm=perm: x.transpose(perm),
                            "TransposeOptions", [np.array(perm, np.int32)])

    def test_slice(self):
        self.single("SLICE", (1, 6, 5, 8), lambda x: x[:, 1:5, 2:5, 3:7], "SliceOptions",
                    [np.array([0, 1, 2, 3], np.int32), np.array([1, 4, -1, 4], np.int32)])
        self.single("SLICE", (6, 5, 8), lambda x: x[1:4, :, 2:6], "SliceOptions",
                    [np.array([1, 0, 2], np.int32), np.array([3, 5, 4], np.int32)])

    def test_strided_slice(self):
        self.single("STRIDED_SLICE", (1, 6, 5, 8), lambda x: x[:, 1:5, 0:5, 2:],
                    "StridedSliceOptions",
                    [np.array([0, 1, 0, 2], np.int32), np.array([1, 5, 5, 0], np.int32),
                     np.array([1, 1, 1, 1], np.int32)], BeginMask=0b0101, EndMask=0b1001)
        self.single("STRIDED_SLICE", (6, 5, 8), lambda x: x[2:-1, :, 1:4], "StridedSliceOptions",
                    [np.array([2, 0, 1], np.int32), np.array([-1, 5, 4], np.int32),
                     np.array([1, 1, 1], np.int32)], BeginMask=0b010, EndMask=0b010)

    def test_split(self):
        x = rand(1, 6, 5, 8)
        for axis, expected in ((3, [x[..., :4], x[..., 4:]]), (1, [x[:, :3], x[:, 3:]])):
            m = Model()
            xi, ax = m.tensor(x.shape), m.const(np.array(axis, np.int32))
            outs = [m.tensor(e.shape) for e in expected]
            m.op("SPLIT", [ax, xi], outs, "SplitOptions", NumSplits=2)
            self.check(m.build([xi], outs), [x], expected)

    def test_split_v(self):
        x = rand(1, 6, 5, 8)
        m = Model()
        xi = m.tensor(x.shape)
        sizes, ax = m.const(np.array([2, -1, 3], np.int32)), m.const(np.array(3, np.int32))
        expected = [x[..., :2], x[..., 2:5], x[..., 5:]]
        outs = [m.tensor(e.shape) for e in expected]
        m.op("SPLIT_V", [xi, sizes, ax], outs, "SplitVOptions", NumSplits=3)
        self.check(m.build([xi], outs), [x], expected)

    def test_reductions(self):
        cases = (
            ("SUM", (1, 3, 4, 5), [1], True, np.sum), ("REDUCE_MAX", (1, 3, 4, 5), [3], False, np.max),
            ("REDUCE_MIN", (1, 3, 4, 5), [1, 2], False, np.min),
            ("MEAN", (1, 3, 4, 5), [2, 3], True, np.mean), ("MEAN", (4, 5, 6), [1], True, np.mean),
            ("SUM", (1, 3, 4, 5), [1, 2, 3], True, np.sum), ("SUM", (1, 3, 4, 5), [-1], False, np.sum),
            ("MEAN", (1, 3, 4, 5), [1, 2], False, np.mean),  # the dedicated spatial mean
        )
        for name, shape, axes, keep, fn in cases:
            with self.subTest(name=name, shape=shape, axes=axes, keep=keep):
                self.single(name, shape, lambda x, a=tuple(axes), k=keep, f=fn: f(x, axis=a, keepdims=k),
                            "ReducerOptions", [np.array(axes, np.int32)], KeepDims=keep, tol=1e-4)

    def test_arg_reduce(self):
        for name, fn in (("ARG_MAX", np.argmax), ("ARG_MIN", np.argmin)):
            for shape in ((1, 3, 4, 5), (4, 6)):
                with self.subTest(name=name, shape=shape):
                    self.single(name, shape, lambda x, f=fn: f(x, axis=-1).astype(np.int32),
                                "ArgMaxOptions" if name == "ARG_MAX" else "ArgMinOptions",
                                [np.array(len(shape) - 1, np.int32)], out_kind="i")

    def test_tile(self):
        self.single("TILE", (1, 2, 3, 2), lambda x: np.tile(x, (1, 2, 1, 3)), None,
                    [np.array([1, 2, 1, 3], np.int32)])

    def test_gather_embedding(self):
        table = rand(10, 5)
        for ishape in ((1, 6), (6,)):
            idx = RNG.integers(0, 10, ishape).astype(np.int32)
            m = Model()
            ii, ti = m.tensor(ishape, "i"), m.const(table)
            out = m.tensor(ishape + (5,))
            m.op("GATHER", [ti, ii], [out], "GatherOptions", Axis=0)
            self.check(m.build([ii], [out]), [idx], table[idx])

    def test_cast(self):
        x = rand(1, 2, 3, 4, lo=-4, hi=4)
        m = Model()
        xi = m.tensor(x.shape)
        as_int, back, flag = m.tensor(x.shape, "i"), m.tensor(x.shape), m.tensor(x.shape, "b")
        out_flag = m.tensor(x.shape)
        m.op("CAST", [xi], [as_int], "CastOptions")
        m.op("CAST", [as_int], [back], "CastOptions")
        m.op("CAST", [xi], [flag], "CastOptions")
        m.op("CAST", [flag], [out_flag], "CastOptions")
        self.check(m.build([xi], [as_int, back, out_flag]), [x],
                   [x.astype(np.int32), x.astype(np.int32).astype(np.float32),
                    (x != 0).astype(np.float32)])

    def test_batch_matmul(self):
        for ash, bsh, adj_x, adj_y, ref in (
            ((2, 5, 4), (2, 4, 6), False, False, lambda a, b: a @ b),
            ((2, 4, 5), (2, 4, 6), True, False, lambda a, b: a.transpose(0, 2, 1) @ b),
            ((2, 5, 4), (2, 6, 4), False, True, lambda a, b: a @ b.transpose(0, 2, 1)),
            ((1, 3, 5, 4), (1, 3, 6, 4), False, True, lambda a, b: a @ b.transpose(0, 1, 3, 2)),
            ((3, 5, 4), (4, 6), False, False, lambda a, b: a @ b),
            ((3, 1, 4), (3, 4, 6), False, False, lambda a, b: a @ b),
            ((5, 4), (4, 6), False, False, lambda a, b: a @ b),
        ):
            with self.subTest(a=ash, b=bsh, adj=(adj_x, adj_y)):
                a, b = rand(*ash), rand(*bsh)
                m = Model()
                ai, bi = m.tensor(ash), m.tensor(bsh)
                want = ref(a, b)
                out = m.tensor(want.shape)
                m.op("BATCH_MATMUL", [ai, bi], [out], "BatchMatMulOptions", AdjX=adj_x, AdjY=adj_y)
                self.check(m.build([ai, bi], [out]), [a, b], want, 1e-4)

    def test_transpose_conv(self):
        def reference(x, w, bias, stride, same):
            _, ih, iw, ci = x.shape
            co, kh, kw, _ = w.shape
            oh, ow = (ih * stride, iw * stride) if same else ((ih - 1) * stride + kh, (iw - 1) * stride + kw)
            tot_h, tot_w = max(0, (ih - 1) * stride + kh - oh), max(0, (iw - 1) * stride + kw - ow)
            out = np.zeros((oh, ow, co), np.float32) + bias
            for iy in range(ih):
                for ix in range(iw):
                    for ky in range(kh):
                        for kx in range(kw):
                            oy, ox = iy * stride - tot_h // 2 + ky, ix * stride - tot_w // 2 + kx
                            if 0 <= oy < oh and 0 <= ox < ow:
                                out[oy, ox] += w[:, ky, kx, :] @ x[0, iy, ix]
            return out[None]

        for stride, same, k in ((2, True, 3), (2, False, 3), (1, True, 3), (2, True, 2), (3, True, 2)):
            with self.subTest(stride=stride, same=same, k=k):
                x, w, bias = rand(1, 4, 5, 6), rand(8, k, k, 6, hi=0.5), rand(8, lo=-0.3, hi=0.3)
                want = reference(x, w, bias, stride, same)
                m = Model()
                shape, wi, xi, bi = m.const(np.array(want.shape, np.int32)), m.const(w), m.tensor(x.shape), m.const(bias)
                out = m.tensor(want.shape)
                m.op("TRANSPOSE_CONV", [shape, wi, xi, bi], [out], "TransposeConvOptions",
                     Padding=tflite.Padding.SAME if same else tflite.Padding.VALID,
                     StrideW=stride, StrideH=stride)
                self.check(m.build([xi], [out]), [x], want, 1e-4)


@NUMERIC
class ElementwiseOps(Case):
    def test_unary(self):
        positive = rand(1, 3, 4, 6, lo=0.2, hi=3.0)
        cases = (
            ("EXP", np.exp, None), ("SQRT", np.sqrt, positive), ("ABS", np.abs, None),
            ("NEG", np.negative, None), ("LOG", np.log, positive), ("FLOOR", np.floor, None),
            ("CEIL", np.ceil, None), ("SIN", np.sin, None), ("COS", np.cos, None),
            ("RSQRT", lambda x: 1 / np.sqrt(x), positive),
            ("ELU", lambda x: np.where(x < 0, np.exp(x) - 1, x), None),
            ("GELU", lambda x: 0.5 * x * (1 + ERF(x / np.sqrt(2))), None),
        )
        for name, fn, x in cases:
            with self.subTest(name=name):
                options = {"GELU": ("GeluOptions", {"Approximate": False})}.get(name, (None, {}))
                self.single(name, (1, 3, 4, 6), fn, options[0], x=x, tol=1e-4, **options[1])

    def test_binary_constant_channel_vector(self):
        names = {"SUB": np.subtract, "DIV": np.divide, "MINIMUM": np.minimum, "MAXIMUM": np.maximum,
                 "POW": np.power, "MUL": np.multiply}
        vec = rand(6, lo=0.5, hi=2.0)
        for name, fn in names.items():
            for const_first in (False, True):
                with self.subTest(name=name, const_first=const_first):
                    x = rand(1, 3, 4, 6, lo=0.5, hi=2.0)
                    m = Model()
                    xi, ci = m.tensor(x.shape), m.const(vec)
                    out = m.tensor(x.shape)
                    opts = {"SUB": "SubOptions", "DIV": "DivOptions", "MUL": "MulOptions"}.get(name)
                    m.op(name, [ci, xi] if const_first else [xi, ci], [out], opts)
                    want = fn(vec, x) if const_first else fn(x, vec)
                    self.check(m.build([xi], [out]), [x], want, 1e-4)

    def test_binary_runtime_broadcast(self):
        names = {"SUB": np.subtract, "DIV": np.divide, "MINIMUM": np.minimum, "POW": np.power}
        for name, fn in names.items():
            for swap in (False, True):
                with self.subTest(name=name, swap=swap):
                    x, y = rand(1, 3, 4, 6, lo=0.5, hi=2.0), rand(1, 1, 1, 6, lo=0.5, hi=2.0)
                    m = Model()
                    xi, yi = m.tensor(x.shape), m.tensor(y.shape)
                    out = m.tensor(x.shape)
                    opts = {"SUB": "SubOptions", "DIV": "DivOptions"}.get(name)
                    m.op(name, [yi, xi] if swap else [xi, yi], [out], opts)
                    want = fn(y, x) if swap else fn(x, y)
                    self.check(m.build([xi, yi], [out]), [x, y], want, 1e-4)

    def test_compare_and_select(self):
        x, y = rand(1, 3, 4, 6), rand(1, 3, 4, 6)
        for name, fn in (("GREATER", np.greater), ("LESS", np.less), ("GREATER_EQUAL", np.greater_equal),
                         ("LESS_EQUAL", np.less_equal), ("EQUAL", np.equal)):
            with self.subTest(name=name):
                m = Model()
                xi, yi = m.tensor(x.shape), m.tensor(y.shape)
                flag = m.tensor(x.shape, "b")
                m.op(name, [xi, yi], [flag])
                self.check(m.build([xi, yi], [flag]), [x, y], fn(x, y).astype(np.int32))
        for v2 in ("SELECT", "SELECT_V2"):
            with self.subTest(select=v2):
                m = Model()
                xi, yi = m.tensor(x.shape), m.tensor(y.shape)
                flag, out = m.tensor(x.shape, "b"), m.tensor(x.shape)
                m.op("GREATER", [xi, yi], [flag])
                m.op(v2, [flag, xi, yi], [out])
                self.check(m.build([xi, yi], [out]), [x, y], np.where(x > y, x, y))
        m = Model()  # a scalar constant branch and a broadcast per-channel condition
        xi, zero = m.tensor(x.shape), m.const(np.zeros((), np.float32))
        flag, out = m.tensor(x.shape, "b"), m.tensor(x.shape)
        m.op("GREATER", [xi, zero], [flag])
        m.op("SELECT_V2", [flag, xi, zero], [out])
        self.check(m.build([xi], [out]), [x], np.maximum(x, 0))

    def test_fused_relu_after_subtract_unchanged(self):
        x, vec = rand(1, 3, 4, 6), rand(6)
        m = Model()
        xi, ci, out = m.tensor(x.shape), m.const(vec), m.tensor(x.shape)
        m.op("SUB", [xi, ci], [out], "SubOptions", FusedActivationFunction=1)
        self.check(m.build([xi], [out]), [x], np.maximum(x - vec, 0))


@unittest.skipUnless(tflite, "needs the tflite schema package and flatbuffers")
class Rejections(Case):
    def reject(self, text, name, shape, options=None, extra=(), **fields):
        m = Model()
        xi = m.tensor(shape)
        consts = [m.const(c) for c in extra]
        out = m.tensor(shape)
        m.op(name, [xi] + consts, [out], options, **fields)
        with self.assertRaisesRegex(ValueError, text):
            lower_only(m.build([xi], [out]))

    def test_unsupported_variants_are_named(self):
        self.reject("SOFTMAX: beta", "SOFTMAX", (1, 2, 2, 4), "SoftmaxOptions", Beta=2.0)
        self.reject("GELU: only the exact", "GELU", (1, 2, 2, 4), "GeluOptions", Approximate=True)
        self.reject("STRIDED_SLICE: only unit strides", "STRIDED_SLICE", (1, 4, 4, 4),
                    "StridedSliceOptions",
                    [np.zeros(4, np.int32), np.array([1, 4, 4, 4], np.int32), np.array([1, 2, 1, 1], np.int32)])
        self.reject("STRIDED_SLICE: ellipsis", "STRIDED_SLICE", (1, 4, 4, 4), "StridedSliceOptions",
                    [np.zeros(4, np.int32), np.array([1, 4, 4, 4], np.int32), np.ones(4, np.int32)],
                    ShrinkAxisMask=1)
        self.reject("TRANSPOSE: batch axis", "TRANSPOSE", (1, 3, 4, 5), "TransposeOptions",
                    [np.array([1, 0, 2, 3], np.int32)])
        self.reject("ARG_MAX: only the last", "ARG_MAX", (1, 3, 4, 5), "ArgMaxOptions",
                    [np.array(1, np.int32)])
        self.reject("SUM: batch axis", "SUM", (1, 3, 4, 5), "ReducerOptions", [np.array([0], np.int32)],
                    KeepDims=True)

    def test_batch_matmul_needs_runtime_operands(self):
        m = Model()
        ai, bi, out = m.tensor((2, 5, 4)), m.const(rand(2, 4, 6)), m.tensor((2, 5, 6))
        m.op("BATCH_MATMUL", [ai, bi], [out], "BatchMatMulOptions")
        with self.assertRaisesRegex(ValueError, "BATCH_MATMUL: both operands"):
            lower_only(m.build([ai], [out]))

    def test_unknown_operation_is_named(self):
        self.reject("unsupported operation 0: SQUARE", "SQUARE", (1, 2, 2, 4), "SquareOptions")


if __name__ == "__main__":
    unittest.main()
