"""Float ONNX -> HXFG lowering, BatchNorm folding and rejection paths (synthetic models only).

Skips when the offline `onnx` package is absent. The numeric tests also need onnxruntime
and the portable reference runner (build/bin/hexecutor_tensor_run); they compare the
lowered graph on the CPU reference against onnxruntime on the original model.
"""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from hexecutor.offline import lower_onnx_float as lowering
    from hexecutor.offline import hxfg_writer as hx
except ImportError:  # offline tooling dependency
    onnx = None

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(os.environ.get("HEXECUTOR_TENSOR_RUN", ROOT / "build/bin/hexecutor_tensor_run"))


class Builder:
    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)
        self.init, self.nodes = [], []
        self.n = 0

    def name(self, stem):
        self.n += 1
        return f"{stem}{self.n}"

    def const(self, value, stem="c"):
        name = self.name(stem)
        value = np.asarray(value)
        if value.dtype.kind != "i":
            value = value.astype(np.float32)
        self.init.append(numpy_helper.from_array(value, name))
        return name

    def rand(self, *shape, scale=0.3):
        return self.rng.normal(0, scale, shape).astype(np.float32)

    def node(self, op, inputs, **attrs):
        out = self.name(op.lower())
        self.nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def conv(self, x, ci, co, k=3, group=1, bias=True, **attrs):
        w = self.const(self.rand(co, ci // group, k, k))
        ins = [x, w] + ([self.const(self.rand(co, scale=0.1))] if bias else [])
        attrs.setdefault("kernel_shape", [k, k])
        if "auto_pad" not in attrs:
            attrs.setdefault("pads", [k // 2] * 4)
        return self.node("Conv", ins, group=group, **attrs)

    def bn(self, x, c):
        ins = [
            x,
            self.const(self.rng.uniform(0.5, 1.5, c)),
            self.const(self.rand(c, scale=0.2)),
            self.const(self.rand(c, scale=0.2)),
            self.const(self.rng.uniform(0.5, 2.0, c)),
        ]
        return self.node("BatchNormalization", ins, epsilon=1e-3)

    def model(self, inputs, outputs, opset=17, check=True):
        def vi(item):
            name, shape = item[0], item[1]
            dtype = item[2] if len(item) > 2 else TensorProto.FLOAT
            return helper.make_tensor_value_info(name, dtype, shape)

        g = helper.make_graph(
            self.nodes,
            "g",
            [vi(i) for i in inputs],
            [vi(o) for o in outputs],
            self.init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
        m.ir_version = 9
        if check:  # onnx 1.23 deprecates the opset-18 GroupNormalization schema
            onnx.checker.check_model(m)
        return m


def model_a():
    """Folded BN, residual add with fused ReLU, SE gate, SiLU, standalone BN, pools, concat,
    pad, resize, PReLU, mean, activations and a Gemm head (batch one, 3x16x16)."""
    b = Builder()
    a = b.node("Relu", [b.bn(b.conv("x", 3, 16), 16)])
    d = b.node("Clip", [b.bn(b.conv(a, 16, 16, group=16), 16), b.const(0.0), b.const(6.0)])
    e = b.bn(b.conv(d, 16, 16, k=1, bias=False), 16)
    r = b.node("Relu", [b.node("Add", [a, e])])
    g = b.node("GlobalAveragePool", [r])
    gate = b.node("HardSigmoid", [b.conv(g, 16, 16, k=1)], alpha=0.2, beta=0.5)
    s = b.node("Mul", [r, gate])
    t = b.node("Mul", [s, b.node("Sigmoid", [s])])  # SiLU
    u = b.bn(t, 16)  # nothing to fold into
    m = b.node("MaxPool", [u], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    v = b.node("AveragePool", [u], kernel_shape=[2, 2], strides=[2, 2])
    c = b.node("Concat", [m, v, m], axis=1)
    p = b.node("Pad", [c, b.const(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)), b.const(0.0)])
    p = b.node("Resize", [p, "", "", b.const(np.array([1, 48, 12, 12], np.int64))],
               mode="linear", coordinate_transformation_mode="half_pixel")
    p = b.node("PRelu", [p, b.const(b.rng.uniform(0.05, 0.3, (48, 1, 1)))])
    mean = b.node("ReduceMean", [p, b.const(np.array([2, 3], np.int64))], keepdims=0)
    h = b.node("Gelu", [b.node("Tanh", [b.node("LeakyRelu", [mean], alpha=0.1)])])
    out = b.node("Gemm", [h, b.const(b.rand(10, 48)), b.const(b.rand(10, scale=0.1))],
                 transB=1, alpha=0.5)
    return b.model([("x", [1, 3, 16, 16])], [(out, [1, 10])], opset=20)


def model_b():
    """SAME_UPPER stride-2 (asymmetric) convolution, per-channel Mul/Add folded, constant minus
    tensor, constant division, and two outputs where one also feeds the other."""
    b = Builder(seed=5)
    cv = b.conv("x", 3, 8, auto_pad="SAME_UPPER", strides=[2, 2])
    n1 = b.node("Add", [b.node("Mul", [cv, b.const(b.rand(8, 1, 1) + 1)]), b.const(b.rand(8, 1, 1))])
    n2 = b.node("Sub", [b.const(b.rand(1, 8, 1, 1)), n1])
    n3 = b.node("Div", [n2, b.const(2.0)])
    return b.model([("x", [1, 3, 16, 16])], [(n1, [1, 8, 8, 8]), (n3, [1, 8, 8, 8])])


def model_nearest():
    """Nearest resize (asymmetric, floor): an integer 2x up, then a non-integer 8x8 -> 5x3."""
    b = Builder(seed=9)
    a = b.node("Relu", [b.conv("x", 3, 4, k=1, bias=False)])
    up = b.node("Resize", [a, "", "", b.const(np.array([1, 4, 8, 8], np.int64))],
                mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    odd = b.node("Resize", [up, "", "", b.const(np.array([1, 4, 5, 3], np.int64))],
                 mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    return b.model([("x", [1, 3, 4, 4])], [(odd, [1, 4, 5, 3])])


def model_wide():
    b = Builder(seed=7)
    out = b.node("Gemm", ["x", b.const(b.rand(5000, 32, scale=0.1)), b.const(b.rand(5000))], transB=1)
    return b.model([("x", [1, 32])], [(out, [1, 5000])])


def lower_model(m, tmp, **kw):
    src = Path(tmp) / "m.onnx"
    onnx.save(m, str(src))
    return lowering.lower(src, Path(tmp) / "g", **kw)


def run_reference(tmp, x_nchw):
    tmp = Path(tmp)
    data = x_nchw[0].transpose(1, 2, 0) if x_nchw.ndim == 4 else x_nchw[0]
    out_dt = np.int32 if np.issubdtype(x_nchw.dtype, np.integer) or x_nchw.dtype == bool else np.float32
    (tmp / "in.f32").write_bytes(np.ascontiguousarray(data, out_dt).tobytes())
    subprocess.run(
        [str(RUNNER), str(tmp / "g.hxfg"), str(tmp / "g.bin"), str(tmp / "in.f32"),
         str(tmp / "out_")],
        check=True, capture_output=True,
    )
    manifest = json.loads((tmp / "g.json").read_text()) if (tmp / "g.json").exists() else {}
    outputs = manifest.get("outputs", [])
    i = 0
    while (tmp / f"out_{i}.f32").exists():
        dt_code = outputs[i].get("dtype", 0) if i < len(outputs) else 0
        if dt_code == 1:
            yield np.fromfile(tmp / f"out_{i}.f32", np.int32)
        elif dt_code == 2:
            yield np.fromfile(tmp / f"out_{i}.f32", np.int32).astype(bool)
        else:
            yield np.fromfile(tmp / f"out_{i}.f32", np.float32)
        i += 1


def kinds(tmp):
    image = (Path(tmp) / "g.hxfg").read_bytes()
    tensors, ops = struct.unpack_from("<2I", image, 12)
    return [struct.unpack_from("<I", image, 256 + 32 * tensors + 64 * i)[0] for i in range(ops)]


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class FloatOnnxTests(unittest.TestCase):
    def check_numeric(self, m, shape, tol=2e-4, x_input=None):
        if x_input is not None:
            x = x_input
        else:
            in_type = m.graph.input[0].type.tensor_type.elem_type
            if in_type in (TensorProto.INT32, TensorProto.INT64):
                x = np.random.default_rng(11).integers(0, 8, shape).astype(
                    np.int32 if in_type == TensorProto.INT32 else np.int64
                )
            else:
                x = np.random.default_rng(11).normal(0, 1, shape).astype(np.float32)
        want = onnxruntime.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(m, tmp)
            got = list(run_reference(tmp, x))
            parts = manifest["output_tensors"]
            self.assertEqual(len(got), len(parts))
            merged, i = [], 0
            for o in want:
                flat = o[0].transpose(1, 2, 0) if o.ndim == 4 else o[0]
                width = flat.shape[-1]
                chunks = []
                while sum(c.shape[-1] for c in chunks) < width:
                    c = got[i].reshape(-1, parts[i]["channels"][1] - parts[i]["channels"][0])
                    chunks.append(c)
                    i += 1
                merged.append(np.concatenate(chunks, axis=-1).reshape(flat.shape))
                np.testing.assert_allclose(merged[-1], flat, rtol=tol, atol=tol)
            return manifest, kinds(tmp)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_mixed_model_matches_onnxruntime(self):
        manifest, ops = self.check_numeric(model_a(), (1, 3, 16, 16))
        # Four BatchNorms, three of which fold into a convolution; one runs as mul + add.
        self.assertEqual(ops.count(hx.BINARY), 3, ops)  # standalone BN (2) + SE gate multiply
        self.assertEqual(ops.count(hx.DEPTHWISE), 1)
        self.assertIn(hx.AVGPOOL, ops)
        self.assertIn(hx.ACT, ops)  # SiLU, hard sigmoid, tanh, leaky ReLU
        self.assertEqual(ops.count(hx.ADD), 1)
        self.assertEqual(ops.count(hx.CONCAT), 2)  # three inputs chain through one temporary

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_nearest_resize_matches_onnxruntime(self):
        manifest, ops = self.check_numeric(model_nearest(), (1, 3, 4, 4))
        self.assertEqual(ops.count(hx.RESIZE), 2, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_asymmetric_same_pad_affine_fold_and_reversed_ops(self):
        manifest, ops = self.check_numeric(model_b(), (1, 3, 16, 16))
        # Mul and Add fold into the convolution; c - x needs negate + add; Div folds nowhere
        # (its producer has a second user), so it is one multiply.
        self.assertEqual(ops.count(hx.CONV), 1)
        self.assertEqual(ops.count(hx.BINARY), 3, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_wide_gemm_output_is_split_and_recombines(self):
        manifest, ops = self.check_numeric(model_wide(), (1, 32), tol=1e-3)
        self.assertGreater(len(manifest["output_tensors"]), 1)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_elementwise_activations_match_onnxruntime(self):
        b = Builder(seed=13)
        # Sqrt sees only positive values (|x| + 1), so every output is finite.
        pos = b.node("Add", [b.node("Abs", ["x"]), b.const(np.ones((3, 1, 1), np.float32))])
        outs = [
            b.node("Sqrt", [pos]),
            b.node("Exp", [b.node("Neg", ["x"])]),
            b.node("Erf", ["x"]),
            b.node("Abs", ["x"]),
        ]
        m = b.model([("x", [1, 3, 4, 4])], [(o, [1, 3, 4, 4]) for o in outs])
        manifest, ops = self.check_numeric(m, (1, 3, 4, 4))
        # Abs (inside the Add), Sqrt, Neg, Exp, Erf and the last Abs: one ACT each, none fused.
        self.assertEqual(ops.count(hx.ACT), 6, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_gather_embedding_lookup_matches_onnxruntime(self):
        b = Builder(seed=21)
        tbl = b.const(b.rand(10, 16))
        out = b.node("Gather", [tbl, "x"], axis=0)
        x_val = np.array([[2, 0, 7, 3]], dtype=np.int64)
        m = b.model([("x", [1, 4], TensorProto.INT64)], [(out, [1, 4, 16])])
        want = onnxruntime.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x_val})[0]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(m, tmp)
            got = list(run_reference(tmp, x_val.astype(np.int32)))[0]
            np.testing.assert_allclose(got, want.reshape(-1), atol=2e-4)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_full_size_constant_add_and_runtime_div(self):
        b = Builder(seed=23)
        c_full = b.const(b.rand(1, 4, 3, 3))
        y = b.node("Add", ["x", c_full])
        div_out = b.node("Div", [y, b.const(np.full((1, 4, 1, 1), 2.0, np.float32))])
        m = b.model([("x", [1, 4, 3, 3])], [(div_out, [1, 4, 3, 3])])
        self.check_numeric(m, (1, 4, 3, 3))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_where_select_matches_onnxruntime(self):
        b = Builder(seed=25)
        cond = b.node("Greater", ["x", b.const(np.zeros((1, 4, 2, 2), np.float32))])
        out = b.node("Where", [cond, "x", b.const(np.full((1, 4, 2, 2), -1.0, np.float32))])
        m = b.model([("x", [1, 4, 2, 2])], [(out, [1, 4, 2, 2])])
        self.check_numeric(m, (1, 4, 2, 2))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_tile_and_expand_match_onnxruntime(self):
        b = Builder(seed=27)
        t_out = b.node("Tile", ["x", b.const(np.array([1, 1, 2, 2], np.int64))])
        m = b.model([("x", [1, 4, 2, 2])], [(t_out, [1, 4, 4, 4])])
        self.check_numeric(m, (1, 4, 2, 2))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_resize_nearest_round_prefer_floor_matches_onnxruntime(self):
        b = Builder(seed=29)
        out = b.node("Resize", ["x", "", "", b.const(np.array([1, 4, 8, 8], np.int64))],
                     mode="nearest", coordinate_transformation_mode="half_pixel",
                     nearest_mode="round_prefer_floor")
        m = b.model([("x", [1, 4, 4, 4])], [(out, [1, 4, 8, 8])])
        self.check_numeric(m, (1, 4, 4, 4))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_dense_conv_split_deep_when_k_exceeds_4032(self):
        b = Builder(seed=31)
        # K = 3 * 3 * 512 = 4608 > 4032: splits along input channels into slices
        out = b.conv("x", 512, 16, k=3, pads=[1, 1, 1, 1])
        m = b.model([("x", [1, 512, 4, 4])], [(out, [1, 16, 4, 4])])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(m, tmp)
            k_list = kinds(tmp)
            self.assertIn(hx.SLICE, k_list)
            self.assertIn(hx.ADD, k_list)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_argmax_and_topk_channel_axis_matches_onnxruntime(self):
        b = Builder(seed=43)
        am_out = b.node("ArgMax", ["x"], axis=1, keepdims=1)
        vals = b.name("topk_v")
        idxs = b.name("topk_i")
        b.nodes.append(helper.make_node("TopK", ["x", b.const(np.array([3], np.int64))], [vals, idxs], axis=1))
        m = b.model([("x", [1, 8, 4, 4])],
                    [(am_out, [1, 1, 4, 4], TensorProto.INT64),
                     (vals, [1, 3, 4, 4]),
                     (idxs, [1, 3, 4, 4], TensorProto.INT64)])
        x = np.random.randn(1, 8, 4, 4).astype(np.float32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(m, tmp)
            got = list(run_reference(tmp, x))
            np.testing.assert_allclose(got[0], want[0].flatten(), atol=1e-4)
            np.testing.assert_allclose(got[1], want[1][0].transpose(1, 2, 0).reshape(-1), atol=1e-4)
            np.testing.assert_allclose(got[2], want[2][0].transpose(1, 2, 0).reshape(-1), atol=1e-4)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_resize_cubic_matches_onnxruntime(self):
        b = Builder(seed=45)
        out = b.node("Resize", ["x", "", "", b.const(np.array([1, 4, 8, 8], np.int64))],
                     mode="cubic", coordinate_transformation_mode="half_pixel")
        m = b.model([("x", [1, 4, 4, 4])], [(out, [1, 4, 8, 8])])
        self.check_numeric(m, (1, 4, 4, 4), tol=1e-3)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_elementwise_math_and_comparison_ops_match_onnxruntime(self):
        b = Builder(seed=47)
        pos = b.node("Add", [b.node("Abs", ["x"]), b.const(np.full((1, 4, 2, 2), 1.5, np.float32))])
        outs = [
            b.node("Sin", ["x"]),
            b.node("Cos", ["x"]),
            b.node("Log", [pos]),
            b.node("Reciprocal", [pos]),
            b.node("Floor", ["x"]),
            b.node("Ceil", ["x"]),
            b.node("Round", ["x"]),
            b.node("Pow", [pos, b.const(np.full((1, 4, 2, 2), 2.0, np.float32))]),
            b.node("Min", ["x", b.const(np.zeros((1, 4, 2, 2), np.float32))]),
            b.node("Max", ["x", b.const(np.zeros((1, 4, 2, 2), np.float32))]),
            b.node("Less", ["x", b.const(np.zeros((1, 4, 2, 2), np.float32))]),
            b.node("Not", [b.node("Less", ["x", b.const(np.zeros((1, 4, 2, 2), np.float32))])]),
        ]
        m = b.model([("x", [1, 4, 2, 2])], [(o, [1, 4, 2, 2]) for o in outs[:-2]] +
                    [(outs[-2], [1, 4, 2, 2], TensorProto.BOOL),
                     (outs[-1], [1, 4, 2, 2], TensorProto.BOOL)])
        x = np.random.randn(1, 4, 2, 2).astype(np.float32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            got = list(run_reference(tmp, x))
            for i, (g, w) in enumerate(zip(got, want)):
                w_hwc = w[0].transpose(1, 2, 0).reshape(-1)
                np.testing.assert_allclose(g, w_hwc, atol=1e-4, err_msg=f"mismatch in op {i}")

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_softmax_over_channels_after_conv_matches_onnxruntime(self):
        b = Builder(seed=17)
        y = b.conv("x", 3, 40)  # 40 channels: two channel blocks
        out = b.node("Softmax", [y], axis=1)
        manifest, ops = self.check_numeric(b.model([("x", [1, 3, 8, 8])], [(out, [1, 40, 8, 8])]),
                                           (1, 3, 8, 8))
        self.assertEqual(ops.count(hx.SOFTMAX), 1, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_softmax_after_gemm_with_large_logits_matches_onnxruntime(self):
        b = Builder(seed=19)
        # Weights scaled by 3 give logits of order +-15 or more; the lowered softmax subtracts the
        # per-pixel maximum, so it must still match onnxruntime.
        g = b.node("Gemm", ["x", b.const(b.rand(10, 32, scale=3.0)), b.const(b.rand(10))],
                   transB=1)
        out = b.node("Softmax", [g])  # opset 17 default axis -1 is channel axis 1 for rank 2
        manifest, ops = self.check_numeric(b.model([("x", [1, 32])], [(out, [1, 10])]), (1, 32))
        self.assertEqual(ops.count(hx.SOFTMAX), 1, ops)

    def test_bn_fold_changes_weights_not_op_count(self):
        b = Builder()
        out = b.node("Relu", [b.bn(b.conv("x", 3, 8), 8)])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(b.model([("x", [1, 3, 8, 8])], [(out, [1, 8, 8, 8])]), tmp)
            self.assertEqual(kinds(tmp), [hx.CONV])
            self.assertEqual(len(manifest["ops"]), 3)  # Conv, BatchNormalization and Relu are recorded

    def test_dense_convolutions_of_any_shape_are_fp16_eligible(self):
        b = Builder()
        y = b.conv("x", 3, 32, k=7, strides=[2, 2], pads=[3, 3, 3, 3])  # ResNet-style stem
        y = b.conv(y, 32, 32)  # dense 3x3 mid-graph
        y = b.conv(y, 32, 32, k=7, group=32, strides=[2, 2], pads=[3, 3, 3, 3])  # 7x7 depthwise
        out = b.conv(y, 32, 64, k=1, strides=[2, 2], pads=[0, 0, 0, 0])  # strided 1x1
        with tempfile.TemporaryDirectory() as tmp:
            m = b.model([("x", [1, 3, 32, 32])], [(out, [1, 64, 4, 4])])
            manifest = lower_model(m, tmp)
            self.assertTrue(manifest["fp16_plan"]["eligible"], manifest["fp16_plan"])
            # The 7x7 stem cannot use the compact RGB fast path, so the input is padded instead.
            image = (Path(tmp) / "g.hxfg").read_bytes()
            first_tensor_stride = struct.unpack_from("<8I", image, 256)[4]
            self.assertEqual(first_tensor_stride, 32)
        b = Builder()
        out = b.conv("x", 3, 32)  # one 3x3 on a 3-channel input keeps the compact layout
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(b.model([("x", [1, 3, 8, 8])], [(out, [1, 32, 8, 8])]), tmp)
            image = (Path(tmp) / "g.hxfg").read_bytes()
            self.assertEqual(struct.unpack_from("<8I", image, 256)[4], 3)

    def test_unpackable_depth_is_reported(self):
        b = Builder()
        out = b.conv("x", 512, 8, k=3, pads=[1, 1, 1, 1])  # K = 4608 > 4096
        with tempfile.TemporaryDirectory() as tmp:
            plan = lower_model(b.model([("x", [1, 512, 4, 4])], [(out, [1, 8, 4, 4])]), tmp)[
                "fp16_plan"
            ]
            self.assertFalse(plan["eligible"])
            self.assertTrue(any("no HMX packing" in r for r in plan["blockers"]))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_oversized_convolutions_are_split_by_output_channel(self):
        b = Builder(seed=9)
        c = b.conv("x", 128, 512)  # 3x3: packed weights 1.18 MB, over the HMX budget
        h = b.node("Relu", [c])
        out = h
        m = b.model([("x", [1, 128, 4, 4])], [(out, [1, 512, 4, 4])])
        manifest, ops = self.check_numeric(m, (1, 128, 4, 4), tol=1e-3)
        self.assertGreaterEqual(ops.count(hx.CONV), 2)
        self.assertGreaterEqual(ops.count(hx.CONCAT), 1)
        self.assertTrue(manifest["fp16_plan"]["eligible"], manifest["fp16_plan"])

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_wide_mid_graph_gemm_is_split_with_concat(self):
        b = Builder(seed=10)
        wide = b.node("Gemm", ["x", b.const(b.rand(2048, 256, scale=0.05)), b.const(b.rand(2048))],
                      transB=1)
        out = b.node("Gemm", [b.node("Relu", [wide]), b.const(b.rand(10, 2048, scale=0.05)),
                              b.const(b.rand(10))], transB=1)
        manifest, ops = self.check_numeric(b.model([("x", [1, 256])], [(out, [1, 10])]), (1, 256),
                                           tol=1e-3)
        self.assertGreaterEqual(ops.count(hx.CONCAT), 1)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_flatten_and_reshape_across_spatial_and_channel_order_match_onnxruntime(self):
        b = Builder(seed=23)  # conv feature map [1,4,2,2] -> Flatten (NCHW order) -> Gemm
        feat = b.conv("x", 3, 4, pads=[0, 0, 0, 0])
        flat = b.node("Flatten", [feat])
        out = b.node("Gemm", [flat, b.const(b.rand(10, 16)), b.const(b.rand(10, scale=0.1))],
                     transB=1)
        manifest, ops = self.check_numeric(b.model([("x", [1, 3, 4, 4])], [(out, [1, 10])]),
                                           (1, 3, 4, 4))
        self.assertEqual(ops.count(hx.RESHAPE), 1, ops)  # straight to 1x1x16

        b = Builder(seed=24)  # [1,4,2,2] -> [1,8,2,1] moves spatial extent into channels
        r = b.node("Reshape", ["x", b.const(np.array([1, 8, 2, 1], np.int64))])
        out = b.conv(r, 8, 6, k=1, bias=False)
        manifest, ops = self.check_numeric(b.model([("x", [1, 4, 2, 2])], [(out, [1, 6, 2, 1])]),
                                           (1, 4, 2, 2))
        self.assertEqual(ops.count(hx.RESHAPE), 2, ops)  # flatten to 1x1x16, then unflatten

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_transpose_matches_onnxruntime(self):
        b = Builder(seed=26)  # NHWC order, then a flatten (NHWC content) into a Gemm
        t = b.node("Transpose", ["x"], perm=[0, 2, 3, 1])  # [1,4,3,5] -> [1,3,5,4]
        out = b.node("Gemm", [b.node("Flatten", [t]), b.const(b.rand(6, 60)),
                              b.const(b.rand(6, scale=0.1))], transB=1)
        manifest, ops = self.check_numeric(b.model([("x", [1, 4, 3, 5])], [(out, [1, 6])]),
                                           (1, 4, 3, 5))
        self.assertEqual(ops.count(hx.TRANSPOSE), 1, ops)
        self.assertEqual(ops.count(hx.RESHAPE), 1, ops)
        for perm in ([0, 2, 3, 1], [0, 3, 2, 1], [0, 1, 3, 2], [0, 3, 1, 2], [0, 2, 1, 3],
                     [0, 1, 2, 3]):
            with self.subTest(perm=perm):  # each permutation followed by a 1x1 convolution
                b = Builder(seed=27)
                t = b.node("Transpose", ["x"], perm=perm)
                dims = [[1, 4, 3, 5][i] for i in perm]
                out = b.conv(t, dims[1], 6, k=1, bias=False)
                self.check_numeric(b.model([("x", [1, 4, 3, 5])],
                                           [(out, [1, 6, dims[2], dims[3]])]), (1, 4, 3, 5))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_slice_matches_onnxruntime(self):
        b = Builder(seed=17)
        out = b.node("Slice", ["x", b.const(np.array([1, 2, 3], np.int64)),
                               b.const(np.array([15, 8, -1], np.int64)),
                               b.const(np.array([1, 2, 3], np.int64))])  # C, H and W offsets
        out = b.node("Relu", [out])
        manifest, ops = self.check_numeric(
            b.model([("x", [1, 20, 9, 11])], [(out, [1, 14, 6, 7])]), (1, 20, 9, 11))
        self.assertEqual(ops.count(hx.SLICE), 1)
        b = Builder(seed=18)  # rank 2: channel slice of a flat tensor
        out = b.node("Slice", ["x", b.const(np.array([3], np.int64)),
                               b.const(np.array([17], np.int64)), b.const(np.array([1], np.int64))])
        self.check_numeric(b.model([("x", [1, 20])], [(out, [1, 14])]), (1, 20))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_split_matches_onnxruntime(self):
        b = Builder(seed=19)  # explicit sizes as an input
        o1, o2, o3 = b.name("split"), b.name("split"), b.name("split")
        b.nodes.append(helper.make_node("Split", ["x", b.const(np.array([8, 4, 8], np.int64))],
                                        [o1, o2, o3], axis=1))
        outs = [(b.node("Relu", [o1]), [1, 8, 6, 6]), (o2, [1, 4, 6, 6]),
                (b.conv(o3, 8, 4, k=1, bias=False), [1, 4, 6, 6])]
        manifest, ops = self.check_numeric(b.model([("x", [1, 20, 6, 6])], outs), (1, 20, 6, 6))
        self.assertEqual(ops.count(hx.SLICE), 3)
        b = Builder(seed=20)  # equal split, no sizes input
        o1, o2, o3 = (b.name("split") for _ in range(3))
        b.nodes.append(helper.make_node("Split", ["x"], [o1, o2, o3], axis=1))
        self.check_numeric(b.model([("x", [1, 12, 5, 5])],
                                   [(o, [1, 4, 5, 5]) for o in (o1, o2, o3)]), (1, 12, 5, 5))

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_grouped_convolution_matches_onnxruntime(self):
        for group, ic, co, seed in ((2, 10, 6, 21), (4, 8, 12, 22)):  # 8 -> 12 needs no padding
            with self.subTest(group=group, ic=ic, co=co):
                b = Builder(seed=seed)
                out = b.node("Relu", [b.conv("x", ic, co, group=group)])
                m = b.model([("x", [1, ic, 6, 6])], [(out, [1, co, 6, 6])])
                manifest, ops = self.check_numeric(m, (1, ic, 6, 6))
                self.assertEqual(ops.count(hx.SLICE), group, ops)
                self.assertEqual(ops.count(hx.CONV), group, ops)
                self.assertEqual(ops.count(hx.CONCAT), group - 1, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_dilated_convolution_matches_onnxruntime(self):
        # (name, build, input shape, output shape, z of each convolution or depthwise op)
        cases = [
            ("dense d2", lambda b: b.conv("x", 3, 8, dilations=[2, 2], pads=[2, 2, 2, 2]),
             (1, 3, 10, 10), (1, 8, 10, 10), [0x202]),
            ("dense d3 stride 2",
             lambda b: b.conv("x", 3, 8, dilations=[3, 3], strides=[2, 2], pads=[3, 3, 3, 3]),
             (1, 3, 10, 10), (1, 8, 5, 5), [0x303]),
            ("dense d(1,2) asymmetric pads",
             lambda b: b.conv("x", 3, 8, dilations=[1, 2], pads=[1, 2, 1, 2]),
             (1, 3, 10, 10), (1, 8, 10, 10), [0x102]),
            ("depthwise d2",
             lambda b: b.conv("x", 4, 4, group=4, dilations=[2, 2], pads=[2, 2, 2, 2]),
             (1, 4, 8, 8), (1, 4, 8, 8), [0x202]),
            ("depthwise d3 stride 2",
             lambda b: b.conv("x", 4, 4, group=4, dilations=[3, 3], strides=[2, 2],
                              pads=[3, 3, 3, 3]),
             (1, 4, 9, 9), (1, 4, 5, 5), [0x303]),
            ("grouped d2",
             lambda b: b.conv("x", 4, 6, group=2, dilations=[2, 2], pads=[2, 2, 2, 2]),
             (1, 4, 8, 8), (1, 6, 8, 8), [0x202, 0x202]),
        ]
        for name, build, in_shape, out_shape, zs in cases:
            with self.subTest(name):
                b = Builder(seed=31)
                out = build(b)
                m = b.model([("x", list(in_shape))], [(out, list(out_shape))])
                _, ops = self.check_numeric(m, in_shape)
                with tempfile.TemporaryDirectory() as tmp:
                    lower_model(m, tmp)
                    image = (Path(tmp) / "g.hxfg").read_bytes()
                    tensors, count = struct.unpack_from("<2I", image, 12)
                    fields = [struct.unpack_from("<4I", image, 256 + 32 * tensors + 64 * i)
                              for i in range(count)]  # kind, x, y, z
                got = [z for kind, _, _, z in fields if kind in (hx.CONV, hx.DEPTHWISE)]
                self.assertEqual(got, zs)
                self.assertTrue(all(z == hx.NONE for kind, _, _, z in fields
                                    if kind not in (hx.CONV, hx.DEPTHWISE)))

    @staticmethod
    def conv_transpose(b, x, ci, co, k, bias=True, **attrs):
        w = b.const(b.rand(ci, co, k, k))  # ONNX layout [in, out/group, kh, kw]
        ins = [x, w] + ([b.const(b.rand(co, scale=0.1))] if bias else [])
        return b.node("ConvTranspose", ins, **attrs)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_conv_transpose_matches_onnxruntime(self):
        cases = [  # stride 2, k3, pad 1, output_padding 1: 5x5 -> 10x10
            dict(seed=31, ih=5, ci=4, co=6, k=3, strides=[2, 2], pads=[1, 1, 1, 1],
                 output_padding=[1, 1], out=10),
            # stride 2, k4, pad 1: 6x6 -> 12x12 (output is exactly 2x)
            dict(seed=32, ih=6, ci=3, co=5, k=4, strides=[2, 2], pads=[1, 1, 1, 1], out=12),
            # stride 1, k3, no pad: 6x6 -> 8x8, then BatchNorm and ReLU that must fold/fuse
            dict(seed=33, ih=6, ci=3, co=4, k=3, strides=[1, 1], pads=[0, 0, 0, 0], out=8,
                 fold=True),
        ]
        for case in cases:
            with self.subTest(**{k: v for k, v in case.items() if k != "seed"}):
                b = Builder(seed=case["seed"])
                attrs = {k: case[k] for k in ("strides", "pads", "output_padding") if k in case}
                out = self.conv_transpose(b, "x", case["ci"], case["co"], case["k"], **attrs)
                if case.get("fold"):
                    out = b.node("Relu", [b.bn(out, case["co"])])
                else:
                    out = b.node("Relu", [out])
                n = case["out"]
                m = b.model([("x", [1, case["ci"], case["ih"], case["ih"]])],
                            [(out, [1, case["co"], n, n])])
                manifest, ops = self.check_numeric(m, (1, case["ci"], case["ih"], case["ih"]))
                self.assertEqual(ops.count(hx.ZEROINSERT), 1, ops)
                self.assertEqual(ops.count(hx.CONV), 1, ops)
                self.assertEqual(ops.count(hx.ACT), 0, ops)  # ReLU fused into the convolution
                if case.get("fold"):
                    self.assertEqual(ops.count(hx.BINARY), 0, ops)  # BatchNorm folded

    def test_conv_transpose_rejections(self):
        def build(**attrs):
            b = Builder()
            out = self.conv_transpose(b, "x", 3, 4, 3, bias=False, **attrs)
            return b, out

        def model(b, out, n):
            return b.model([("x", [1, 3, 6, 6])], [(out, [1, 4, n, n])])

        cases = []
        b = Builder()  # group 2 needs input channels divisible by the group count
        grouped = b.node("ConvTranspose", ["x", b.const(b.rand(4, 2, 3, 3))], group=2)
        cases.append((b.model([("x", [1, 4, 6, 6])], [(grouped, [1, 4, 8, 8])]),
                      "groups other than 1"))
        b, out = build(dilations=[2, 2])
        cases.append((model(b, out, 10), "dilation"))
        b, out = build(auto_pad="SAME_UPPER")
        cases.append((model(b, out, 6), "auto_pad"))
        b, out = build(output_shape=[8, 8])  # spatial sizes only
        cases.append((model(b, out, 8), "output_shape"))
        b, out = build(pads=[3, 3, 3, 3])  # k-1-p = -1: the padded input would need a negative pad
        cases.append((model(b, out, 2), "must not exceed"))
        b = Builder()  # weights from a graph input (batch one, so the input is accepted)
        wv = b.node("ConvTranspose", ["x", "w"])
        cases.append((b.model([("x", [1, 1, 6, 6]), ("w", [1, 3, 4, 3])],
                              [(wv, [1, 3, 9, 8])]), "weights must be a constant"))
        for m, text in cases:
            with self.subTest(text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, text):
                    lower_model(m, tmp)

    def test_rejections_name_the_node(self):
        def softmax():  # opset 13+ default axis -1 is W for rank 4: not the channel axis
            b = Builder()
            return b.model([("x", [1, 8, 4, 4])], [(b.node("Softmax", ["x"]), [1, 8, 4, 4])])

        def dilated():  # dilations above 8 have no HXFG encoding
            b = Builder()
            out = b.conv("x", 3, 8, dilations=[9, 9], pads=[9, 9, 9, 9])
            return b.model([("x", [1, 3, 20, 20])], [(out, [1, 8, 20, 20])])

        def grouped():  # 7 output channels cannot split into two groups
            b = Builder()
            return b.model([("x", [1, 4, 8, 8])], [(b.conv("x", 4, 7, group=2), [1, 7, 8, 8])])

        def transpose_batch():  # the batch axis may not move
            b = Builder()
            out = b.node("Transpose", ["x"], perm=[1, 0, 2, 3])
            return b.model([("x", [1, 4, 2, 2])], [(out, [4, 1, 2, 2])])

        def transpose_rank2():
            b = Builder()
            out = b.node("Transpose", ["x"], perm=[0, 1])
            return b.model([("x", [1, 4])], [(out, [1, 4])])

        def tanh_gelu():
            b = Builder()
            return b.model([("x", [1, 4, 2, 2])],
                           [(b.node("Gelu", ["x"], approximate="tanh"), [1, 4, 2, 2])], opset=20)

        def spatial_const():
            b = Builder()
            out = b.node("Add", ["x", b.const(b.rand(1, 4, 1, 1))])
            return b.model([("x", [1, 1, 2, 2])], [(out, [1, 4, 2, 2])])

        def passthrough():
            b = Builder()
            return b.model([("x", [1, 4, 2, 2])], [(b.node("Identity", ["x"]), [1, 4, 2, 2])])

        def nearest_round():
            b = Builder()
            out = b.node("Resize", ["x", "", "", b.const(np.array([1, 4, 8, 8], np.int64))],
                         mode="nearest", coordinate_transformation_mode="unsupported")
            return b.model([("x", [1, 4, 4, 4])], [(out, [1, 4, 8, 8])])

        for build, text in [
            (softmax, "channel axis"), (dilated, "between 1 and 8"), (grouped, "groups"),
            (transpose_batch, "batch axis"), (transpose_rank2, "rank-4"), (tanh_gelu, "erf"),
            (spatial_const, "broadcast the tensor"), (passthrough, "graph input"),
            (nearest_round, "coordinate transformation mode unsupported"),
        ]:
            with self.subTest(text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, text):
                    lower_model(build(), tmp)

    def test_reduction_rejections_name_the_node(self):
        def batch_axis():
            b = Builder()
            out = b.node("ReduceSum", ["x", b.const(np.array([0], np.int64))], keepdims=1)
            return b.model([("x", [1, 4, 2, 2])], [(out, [1, 4, 2, 2])], opset=20)

        def empty_axes():  # empty axes reduce everything, which HXFG does not express
            b = Builder()
            out = b.node("ReduceSum", ["x", b.const(np.array([], np.int64))], keepdims=1)
            return b.model([("x", [1, 4, 2, 2])], [(out, [1, 1, 1, 1])], opset=20)

        def channel_dropped():  # [1, H, W] has no HWC form: keepdims=1 is required
            b = Builder()
            out = b.node("ReduceMax", ["x"], axes=[1], keepdims=0)
            return b.model([("x", [1, 4, 2, 2])], [(out, [1, 2, 2])], opset=17)

        def keepdims_layout():  # keepdims=0 over (C, H) would need a [1, W, 1] layout
            b = Builder()
            out = b.node("ReduceSum", ["x", b.const(np.array([1, 2], np.int64))], keepdims=0)
            return b.model([("x", [1, 4, 2, 3])], [(out, [1, 3])], opset=20)

        for build, text in [(batch_axis, "batch axis"), (empty_axes, "empty axes"),
                            (channel_dropped, "keepdims=1"), (keepdims_layout, "keepdims layout")]:
            with self.subTest(text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, text):
                    lower_model(build(), tmp)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_channel_reductions_match_onnxruntime(self):
        b = Builder(seed=23)
        y = b.conv("x", 3, 8)
        mx = b.node("ReduceMax", [y], axes=[1], keepdims=1)  # attribute axes, opset 17
        mn = b.node("ReduceMin", [y], axes=[1], keepdims=1)
        avg = b.node("ReduceMean", [y], axes=[1], keepdims=1)  # channel mean: HX_REDUCE, not MEAN
        m = b.model([("x", [1, 3, 8, 8])],
                    [(mx, [1, 1, 8, 8]), (mn, [1, 1, 8, 8]), (avg, [1, 1, 8, 8])], opset=17)
        _, ops = self.check_numeric(m, (1, 3, 8, 8))
        self.assertEqual(ops.count(hx.REDUCE), 3, ops)
        self.assertNotIn(hx.MEAN, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_spatial_attention_pattern_matches_onnxruntime(self):
        b = Builder(seed=29)
        y = b.node("Relu", [b.conv("x", 3, 8)])  # [1, 8, 8, 8]
        avg = b.node("ReduceMean", [y, b.const(np.array([1], np.int64))], keepdims=1)
        mx = b.node("ReduceMax", [y, b.const(np.array([1], np.int64))], keepdims=1)
        att = b.node("Sigmoid", [b.conv(b.node("Concat", [avg, mx], axis=1), 2, 1, k=7)])
        out = b.node("Mul", [y, att])  # [1, 8, 8, 8] times a [1, 1, 8, 8] map
        m = b.model([("x", [1, 3, 8, 8])], [(out, [1, 8, 8, 8])], opset=20)
        _, ops = self.check_numeric(m, (1, 3, 8, 8))
        self.assertEqual(ops.count(hx.REDUCE), 2, ops)
        self.assertNotIn(hx.MEAN, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_spatial_reductions_match_onnxruntime(self):
        b = Builder(seed=31)
        y = b.node("Relu", [b.conv("x", 3, 40)])  # 40 channels: two channel blocks
        total = b.node("ReduceSum", [y, b.const(np.array([2, 3], np.int64))], keepdims=0)
        low = b.node("ReduceMin", [y, b.const(np.array([2, 3], np.int64))], keepdims=0)
        peak = b.node("GlobalMaxPool", [y])  # [1, 40, 1, 1]
        m = b.model([("x", [1, 3, 8, 8])],
                    [(total, [1, 40]), (low, [1, 40]), (peak, [1, 40, 1, 1])], opset=20)
        _, ops = self.check_numeric(m, (1, 3, 8, 8))
        self.assertEqual(ops.count(hx.REDUCE), 3, ops)
        self.assertNotIn(hx.MEAN, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_instance_norm_after_conv_matches_onnxruntime(self):
        b = Builder(seed=23)
        scale, bias = b.const(b.rng.uniform(0.5, 1.5, 8)), b.const(b.rand(8, scale=0.2))
        n = b.node("InstanceNormalization", [b.conv("x", 3, 8), scale, bias], epsilon=1e-5)
        out = b.node("Relu", [n])
        m = b.model([("x", [1, 3, 8, 8])], [(out, [1, 8, 8, 8])])
        _, ops = self.check_numeric(m, (1, 3, 8, 8))
        self.assertEqual(ops.count(hx.NORM), 1, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_instance_norm_rank3_matches_onnxruntime(self):
        c, length = 8, 16
        b = Builder(seed=46)
        scale = b.const(b.rng.uniform(0.5, 1.5, c))
        bias = b.const(b.rand(c, scale=0.2))
        n = b.node("InstanceNormalization", ["x", scale, bias], epsilon=1e-5)
        m = b.model([("x", [1, c, length])], [(n, [1, c, length])], opset=17)
        x = np.random.default_rng(11).normal(0, 1, (1, c, length)).astype(np.float32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(m, tmp)
            data = x[0].transpose(1, 0)
            (Path(tmp) / "in.f32").write_bytes(np.ascontiguousarray(data, np.float32).tobytes())
            subprocess.run([str(RUNNER), str(Path(tmp) / "g.hxfg"), str(Path(tmp) / "g.bin"),
                            str(Path(tmp) / "in.f32"), str(Path(tmp) / "out_")], check=True)
            got = np.fromfile(Path(tmp) / "out_0.f32", np.float32).reshape(length, c)
            got_ncl = got.transpose(1, 0)[None]
            np.testing.assert_allclose(got_ncl, want[0], rtol=1e-4, atol=1e-4)
            ops = kinds(tmp)
            self.assertEqual(ops.count(hx.NORM), 1, ops)

    def test_instance_norm_optional_scale_bias_lowering(self):
        b = Builder(seed=23)
        n = b.node("InstanceNormalization", [b.conv("x", 3, 8)], epsilon=1e-5)
        m = b.model([("x", [1, 3, 8, 8])], [(n, [1, 8, 8, 8])], check=False)
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            ops = kinds(tmp)
            self.assertIn(hx.NORM, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_group_norm_matches_onnxruntime_for_both_opsets(self):
        # Opset 18 gives one scale and bias per group; opset 21 gives one per channel.
        for opset, size in ((18, 2), (21, 8)):
            with self.subTest(opset=opset):
                b = Builder(seed=29 + opset)
                scale = b.const(b.rng.uniform(0.5, 1.5, size))
                bias = b.const(b.rand(size, scale=0.2))
                n = b.node("GroupNormalization", [b.conv("x", 3, 8), scale, bias],
                           num_groups=2, epsilon=1e-5)
                m = b.model([("x", [1, 3, 8, 8])], [(n, [1, 8, 8, 8])], opset=opset,
                            check=opset != 18)
                _, ops = self.check_numeric(m, (1, 3, 8, 8))
                self.assertEqual(ops.count(hx.NORM), 1, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_layer_norm_over_channels_matches_onnxruntime(self):
        b = Builder(seed=31)
        h = b.node("Gemm", ["x", b.const(b.rand(12, 16)), b.const(b.rand(12, scale=0.1))],
                   transB=1)
        scale, bias = b.const(b.rng.uniform(0.5, 1.5, 12)), b.const(b.rand(12, scale=0.2))
        out = b.node("LayerNormalization", [h, scale, bias], axis=-1, epsilon=1e-5)
        m = b.model([("x", [1, 16])], [(out, [1, 12])], opset=17)
        _, ops = self.check_numeric(m, (1, 16))
        self.assertEqual(ops.count(hx.NORM), 1, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_layer_norm_over_chw_with_spatially_constant_affine_matches_onnxruntime(self):
        b = Builder(seed=37)
        y = b.conv("x", 3, 4)
        # The scale and bias have the normalised shape [C, H, W], constant over H and W here.
        gamma = np.broadcast_to(b.rng.uniform(0.5, 1.5, (4, 1, 1)), (4, 8, 8)).astype(np.float32)
        beta = np.broadcast_to(b.rand(4, 1, 1, scale=0.2), (4, 8, 8)).astype(np.float32)
        out = b.node("LayerNormalization", [y, b.const(gamma), b.const(beta)], axis=1)
        m = b.model([("x", [1, 3, 8, 8])], [(out, [1, 4, 8, 8])], opset=17)
        _, ops = self.check_numeric(m, (1, 3, 8, 8))
        self.assertEqual(ops.count(hx.NORM), 1, ops)

    def test_normalisation_rejections_name_the_node(self):
        def layer_norm_axis():  # rank 4 normalises over (C, H, W): axis 2 is not supported
            b = Builder()
            out = b.node("LayerNormalization", ["x", b.const(np.ones((4, 2, 2), np.float32))],
                         axis=2)
            return b.model([("x", [1, 4, 2, 2])], [(out, [1, 4, 2, 2])], opset=17)

        def layer_norm_mean_used():
            b = Builder()
            y, mean = b.name("y"), b.name("mean")
            scale, bias = b.const(np.ones(8, np.float32)), b.const(np.zeros(8, np.float32))
            b.nodes.append(helper.make_node("LayerNormalization", ["x", scale, bias], [y, mean],
                                            axis=-1))
            return b.model([("x", [1, 8])], [(y, [1, 8]), (mean, [1, 1])], opset=17)

        def layer_norm_spatial_scale():  # varies over H and W: not a per-channel vector
            b = Builder()
            out = b.node("LayerNormalization", ["x", b.const(b.rand(4, 2, 2))], axis=1)
            return b.model([("x", [1, 4, 2, 2])], [(out, [1, 4, 2, 2])], opset=17)

        def group_norm_divisor():
            b = Builder()
            out = b.node("GroupNormalization", ["x", b.const(b.rand(3)), b.const(b.rand(3))],
                         num_groups=3)
            return b.model([("x", [1, 8, 2, 2])], [(out, [1, 8, 2, 2])], opset=21)

        def group_norm_size():  # 5 values are neither per group (2) nor per channel (8)
            b = Builder()
            out = b.node("GroupNormalization", ["x", b.const(b.rand(5)), b.const(b.rand(5))],
                         num_groups=2)
            return b.model([("x", [1, 8, 2, 2])], [(out, [1, 8, 2, 2])], opset=21)

        def instance_norm_rank2():
            b = Builder()
            out = b.node("InstanceNormalization", ["x", b.const(b.rand(8)), b.const(b.rand(8))])
            return b.model([("x", [1, 8])], [(out, [1, 8])])

        def norm_scale_not_constant():
            b = Builder()
            out = b.node("InstanceNormalization", ["x", "s", b.const(b.rand(4))])
            return b.model([("x", [1, 4, 2, 2]), ("s", [1, 4])], [(out, [1, 4, 2, 2])])

        for build, text in [
            (layer_norm_axis, "axis 1"), (layer_norm_mean_used, "Mean and InvStdDev"),
            (layer_norm_spatial_scale, "constant over the spatial axes"),
            (group_norm_divisor, "must divide"), (group_norm_size, "per group"),
            (instance_norm_rank2, "rank-4"), (norm_scale_not_constant, "must be constant"),
        ]:
            with self.subTest(text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, text):
                    lower_model(build(), tmp)


def model_hygiene():
    """Shapes of what torch.onnx.export emits without folding: an Identity-wrapped initializer as
    a bias, a Shape/Slice/Mul/Concat chain feeding a Resize size, a Shape/Gather/Unsqueeze/Concat
    chain feeding a Reshape, a dead Shape node, Squeeze with constant axes through a rank-3
    intermediate, a float32 Cast and a Gemm head (batch one, 3x8x8)."""
    b = Builder(seed=23)
    w = b.const(b.rand(8, 3, 3, 3))
    bias = b.node("Identity", [b.const(b.rand(8, scale=0.1))])
    y = b.node("Relu", [b.node("Conv", ["x", w, bias], pads=[1, 1, 1, 1], kernel_shape=[3, 3])])
    b.node("Shape", ["x"])  # dead: nothing reads it
    hw = b.node("Mul", [
        b.node("Slice", [b.node("Shape", [y]), b.const(np.array([2], np.int64)),
                         b.const(np.array([4], np.int64))]),
        b.const(np.array([2, 2], np.int64)),
    ])
    sizes = b.node("Concat", [b.const(np.array([1, 8], np.int64)), hw], axis=0)
    up = b.node("Resize", [y, "", "", sizes], mode="nearest",
                coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    batch = b.node("Unsqueeze", [
        b.node("Gather", [b.node("Shape", [up]), b.const(np.array(0, np.int64))]),
        b.const(np.array([0], np.int64)),
    ])
    view = b.node("Concat", [batch, b.const(np.array([8, 16, 16], np.int64))], axis=0)
    back = b.node("Reshape", [up, view])
    g = b.node("GlobalAveragePool", [back])
    s = b.node("Squeeze", [b.node("Squeeze", [g, b.const(np.array([3], np.int64))]),
                           b.const(np.array([2], np.int64))])
    c = b.node("Cast", [s], to=TensorProto.FLOAT)
    out = b.node("Gemm", [c, b.const(b.rand(4, 8)), b.const(b.rand(4, scale=0.1))], transB=1)
    return b.model([("x", [1, 3, 8, 8])], [(out, [1, 4])])


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class ExportHygieneTests(unittest.TestCase):
    """Constant folding and shape-chain resolution for exporter hygiene (hand-built, no torch)."""

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_hygiene_chains_fold_and_match_onnxruntime(self):
        m = model_hygiene()
        x = np.random.default_rng(29).normal(0, 1, (1, 3, 8, 8)).astype(np.float32)
        want = onnxruntime.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            got = list(run_reference(tmp, x))
            self.assertEqual(len(got), 1)
            np.testing.assert_allclose(got[0], want[0][0], rtol=2e-4, atol=2e-4)
            # Shape chains, the Identity bias, the Cast, the Squeezes and the dead Shape emit
            # nothing: one convolution (ReLU fused), Resize, mean and the Gemm convolution.
            self.assertEqual(kinds(tmp), [hx.CONV, hx.RESIZE, hx.MEAN, hx.CONV])

    def test_only_float32_casts_are_lowered(self):
        b = Builder()
        c = b.node("Cast", ["x"], to=TensorProto.STRING)
        m = b.model([("x", [1, 4, 2, 2])], [(b.node("Cast", [c], to=TensorProto.FLOAT),
                                             [1, 4, 2, 2])], check=False)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "casts between float32"):
            lower_model(m, tmp)

    def test_constant_pool_over_the_runtime_limit_is_rejected(self):
        b = Builder()
        out = b.conv("x", 3, 8)  # 216 weights: 864 bytes of constants
        m = b.model([("x", [1, 3, 8, 8])], [(out, [1, 8, 8, 8])])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(lowering, "CONSTANT_POOL_LIMIT", 256):
            with self.assertRaisesRegex(ValueError, "constant pool"):
                lower_model(m, tmp)


def linear_token(b, t, n_in, n_out):
    """MatMul with a constant [K, N] weight and a bias, on a token tensor [1, L, K]."""
    return b.node("Add", [b.node("MatMul", [t, b.const(b.rand(n_in, n_out))]),
                          b.const(b.rand(n_out, scale=0.1))])


def heads_attention(b, t, length, dim, heads):
    """The exporter chain for multi-head attention on a token tensor [1, L, D]: Reshape to
    [1, L, H, dh], Transpose to [1, H, L, dh] (K as [0, 2, 3, 1]), Q.K^T, the scale as a Mul,
    Softmax over the last axis, attn.V, Transpose back and Reshape to [1, L, D]."""
    dh = dim // heads
    shape = b.const(np.array([1, length, heads, dh], np.int64))
    q, k, v = (b.node("Reshape", [linear_token(b, t, dim, dim), shape]) for _ in range(3))
    qh = b.node("Transpose", [q], perm=[0, 2, 1, 3])
    kt = b.node("Transpose", [k], perm=[0, 2, 3, 1])
    vh = b.node("Transpose", [v], perm=[0, 2, 1, 3])
    scores = b.node("Mul", [b.node("MatMul", [qh, kt]), b.const(np.float32(1 / np.sqrt(dh)))])
    ctx = b.node("MatMul", [b.node("Softmax", [scores], axis=-1), vh])
    merged = b.node("Transpose", [ctx], perm=[0, 2, 1, 3])
    return b.node("Reshape", [merged, b.const(np.array([1, length, dim], np.int64))])


def model_single_head_attention():
    """Q, K and V linears (bias folded), K transposed as a [1, D, L] view (transB 1), the 1/sqrt(D)
    Div folded into the score product, row softmax, attn.V, output linear and a residual add."""
    length, dim = 5, 8
    b = Builder(seed=21)
    x = "x"
    q, k, v = (linear_token(b, x, dim, dim) for _ in range(3))
    kt = b.node("Transpose", [k], perm=[0, 2, 1])
    scores = b.node("Div", [b.node("MatMul", [q, kt]), b.const(np.float32(np.sqrt(dim)))])
    ctx = b.node("MatMul", [b.node("Softmax", [scores], axis=-1), v])
    out = b.node("Add", [x, linear_token(b, ctx, dim, dim)])
    return b.model([(x, [1, length, dim])], [(out, [1, length, dim])]), length, dim


def model_two_head_attention():
    """Two heads through the exporter's Reshape/Transpose chain, then the output linear."""
    length, dim, heads = 6, 8, 2
    b = Builder(seed=23)
    merged = heads_attention(b, "x", length, dim, heads)
    out = linear_token(b, merged, dim, dim)
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])]), length, dim


def model_vit_encoder_layer():
    """A tiny ViT encoder layer: LayerNorm, two-head attention, residual, LayerNorm, MLP with
    exact GELU, residual. Tokens are [1, 4, 8], the MLP width is 16."""
    length, dim, heads, width = 4, 8, 2, 16
    b = Builder(seed=31)
    x = "x"

    def layer_norm(t):
        return b.node("LayerNormalization",
                      [t, b.const(1 + b.rand(dim, scale=0.1)), b.const(b.rand(dim, scale=0.1))],
                      epsilon=1e-5)

    attn = linear_token(b, heads_attention(b, layer_norm(x), length, dim, heads), dim, dim)
    r1 = b.node("Add", [x, attn])
    hidden = b.node("Gelu", [linear_token(b, layer_norm(r1), dim, width)])
    out = b.node("Add", [r1, linear_token(b, hidden, width, dim)])
    m = b.model([(x, [1, length, dim])], [(out, [1, length, dim])], opset=20)
    return m, length, dim


def model_patch_embedding():
    """Conv patch embedding: Conv, Reshape to [1, C, L], Transpose [0, 2, 1] to tokens, linear."""
    b = Builder(seed=51)
    conv = b.conv("x", 3, 8, k=4, strides=[4, 4], pads=[0, 0, 0, 0])
    flat = b.node("Reshape", [conv, b.const(np.array([1, 8, 4], np.int64))])
    tokens = b.node("Transpose", [flat], perm=[0, 2, 1])
    out = linear_token(b, tokens, 8, 5)
    return b.model([("x", [1, 3, 8, 8])], [(out, [1, 4, 5])])


def model_sequence_ops():
    """Gather of a sequence row (CLS-style), Gather of a run, ReduceMean over the sequence (with
    and without keepdims) and a Concat along the sequence axis."""
    length, dim = 4, 6
    b = Builder(seed=41)
    y = linear_token(b, "x", dim, dim)
    cls = b.node("Gather", [y, b.const(np.array(0, np.int64))], axis=1)
    run = b.node("Gather", [y, b.const(np.array([1, 2], np.int64))], axis=1)
    mean = b.node("ReduceMean", [y], axes=[1], keepdims=0)
    mean_k = b.node("ReduceMean", [y], axes=[1], keepdims=1)
    cat = b.node("Concat", [y, run], axis=1)
    m = b.model([("x", [1, length, dim])],
                [(cls, [1, dim]), (run, [1, 2, dim]), (mean, [1, dim]), (mean_k, [1, 1, dim]),
                 (cat, [1, length + 2, dim])])
    return m, length, dim


def model_transposed_operand_in_add():
    """A transposed token view (K^T) used as an elementwise operand, which HXFG cannot express."""
    b = Builder(seed=61)
    t = linear_token(b, "x", 4, 4)
    tt = b.node("Transpose", [t], perm=[0, 2, 1])
    out = b.node("Add", [tt, tt])
    return b.model([("x", [1, 4, 4])], [(out, [1, 4, 4])])


def model_gather_run_with_gap():
    b = Builder(seed=62)
    y = linear_token(b, "x", 4, 4)
    out = b.node("Gather", [y, b.const(np.array([0, 2], np.int64))], axis=1)
    return b.model([("x", [1, 4, 4])], [(out, [1, 2, 4])])


def model_gather_embedding_lookup():
    b = Builder(seed=50)
    table = b.const(np.random.default_rng(1).normal(0, 1, (16, 8)).astype(np.float32))
    g = b.node("Gather", [table, "x"], axis=0)
    return b.model([("x", [1, 4], TensorProto.INT64)], [(g, [1, 4, 8])])


def model_gather_arbitrary_axis():
    b = Builder(seed=51)
    indices = b.const(np.array([1, 3, 5], dtype=np.int64))
    x = linear_token(b, "x", 6, 6)
    g = b.node("Gather", [x, indices], axis=2)
    out = b.node("Relu", [g])
    return b.model([("x", [1, 4, 6])], [(out, [1, 4, 3])])


def model_gather_elements():
    b = Builder(seed=52)
    idx_arr = np.array([[[1, 3, 0, 2, 4, 1],
                         [0, 2, 4, 1, 3, 0],
                         [2, 0, 1, 4, 2, 3],
                         [3, 1, 2, 0, 1, 4]]], dtype=np.int64)
    indices = b.const(idx_arr)
    x = linear_token(b, "x", 6, 6)
    g = b.node("GatherElements", [x, indices], axis=2)
    out = b.node("Relu", [g])
    return b.model([("x", [1, 4, 6])], [(out, [1, 4, 6])])


def model_gather_nd():
    b = Builder(seed=53)
    indices = b.const(np.array([[[0, 1], [0, 3]]], dtype=np.int64))
    x = linear_token(b, "x", 8, 8)
    g = b.node("GatherND", [x, indices])
    out = b.node("Relu", [g])
    return b.model([("x", [1, 4, 8])], [(out, [1, 2, 8])])


def model_scatter_nd():
    b = Builder(seed=54)
    indices = b.const(np.array([[[0, 1], [0, 3]]], dtype=np.int64))
    updates = b.const(np.random.default_rng(6).normal(0, 1, (1, 2, 8)).astype(np.float32))
    x = linear_token(b, "x", 8, 8)
    s = b.node("ScatterND", [x, indices, updates])
    out = b.node("Relu", [s])
    return b.model([("x", [1, 4, 8])], [(out, [1, 4, 8])])


def model_rms_normalization():
    length, dim = 4, 8
    b = Builder(seed=41)
    tok = linear_token(b, "x", dim, dim)
    gamma = b.const(1.0 + b.rand(dim, scale=0.1))
    out = b.node("RMSNormalization", [tok, gamma], epsilon=1e-5)
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])], check=False)


def model_simplified_layer_normalization():
    length, dim = 4, 8
    b = Builder(seed=41)
    tok = linear_token(b, "x", dim, dim)
    gamma = b.const(1.0 + b.rand(dim, scale=0.1))
    out = b.node("SimplifiedLayerNormalization", [tok, gamma], epsilon=1e-5)
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])], check=False), length, dim


def model_decomposed_rmsnorm(with_weight=True):
    length, dim = 4, 8
    b = Builder(seed=44)
    tok = linear_token(b, "x", dim, dim)
    sq = b.node("Pow", [tok, b.const(np.array(2.0, np.float32))])
    mean = b.node("ReduceMean", [sq], axes=[-1], keepdims=1)
    var = b.node("Add", [mean, b.const(np.array(1e-5, np.float32))])
    rstd = b.node("Sqrt", [var])
    normed = b.node("Div", [tok, rstd])
    if with_weight:
        gamma = b.const(1.0 + b.rand(dim, scale=0.1))
        out = b.node("Mul", [normed, gamma])
    else:
        out = normed
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])], opset=17), length, dim


def model_token_layer_norm():
    length, dim = 4, 8
    b = Builder(seed=45)
    tok = linear_token(b, "x", dim, dim)
    gamma = b.const(1.0 + b.rand(dim, scale=0.1))
    beta = b.const(b.rand(dim, scale=0.1))
    out = b.node("LayerNormalization", [tok, gamma, beta], axis=-1, epsilon=1e-5)
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])], opset=17), length, dim


def model_instance_norm_rank3(with_affine=True):
    c, length = 8, 16
    b = Builder(seed=46)
    if with_affine:
        scale = b.const(b.rng.uniform(0.5, 1.5, c))
        bias = b.const(b.rand(c, scale=0.2))
        inputs = ["x", scale, bias]
    else:
        inputs = ["x"]
    n = b.node("InstanceNormalization", inputs, epsilon=1e-5)
    return b.model([("x", [1, c, length])], [(n, [1, c, length])], opset=17), c, length



def model_token_slice():
    length, dim = 4, 8
    b = Builder(seed=42)
    tok = linear_token(b, "x", dim, dim)
    s1 = b.node("Slice", [tok, b.const([0]), b.const([4]), b.const([2])])
    s2 = b.node("Slice", [tok, b.const([4]), b.const([8]), b.const([2])])
    out = b.node("Concat", [s1, s2], axis=-1)
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])]), length, dim


def model_token_reshape_alias():
    length, dim = 4, 8
    b = Builder(seed=43)
    tok = linear_token(b, "x", dim, dim)
    shape = b.const(np.array([1, 2, 16], np.int64))
    r = b.node("Reshape", [tok, shape])
    out = linear_token(b, r, 16, 16)
    return b.model([("x", [1, length, dim])], [(out, [1, 2, 16])]), length, dim


def model_batched_matmul_explicit():
    """Batched MatMul of two runtime operands: Q [1, 2, 4, 8] and K [1, 2, 8, 4] -> [1, 2, 4, 4]."""
    length, dim, heads = 4, 8, 2
    dh = dim // heads
    b = Builder(seed=71)
    tok = linear_token(b, "x", dim * 2, dim * 2)
    q_tok = b.node("Slice", [tok, b.const([0]), b.const([dim]), b.const([2])])
    k_tok = b.node("Slice", [tok, b.const([dim]), b.const([dim * 2]), b.const([2])])
    shape = b.const(np.array([1, length, heads, dh], np.int64))
    q = b.node("Reshape", [q_tok, shape])
    k = b.node("Reshape", [k_tok, shape])
    qh = b.node("Transpose", [q], perm=[0, 2, 1, 3])
    kh = b.node("Transpose", [k], perm=[0, 2, 1, 3])
    kt = b.node("Transpose", [kh], perm=[0, 1, 3, 2])  # [1, H, dh, L]
    scores = b.node("MatMul", [qh, kt])  # [1, 2, 4, 4]
    merged = b.node("Transpose", [scores], perm=[0, 2, 1, 3])
    out = b.node("Reshape", [merged, b.const(np.array([1, length, heads * length], np.int64))])
    return b.model([("x", [1, length, dim * 2])], [(out, [1, length, heads * length])]), length, dim * 2


def model_einsum_attention():
    """Einsum for attention patterns: Q.K^T ("bhqd,bhkd->bhqk") and Attn.V ("bhqk,bhkd->bhqd")."""
    length, dim, heads = 4, 8, 2
    dh = dim // heads
    b = Builder(seed=72)
    x = "x"
    shape = b.const(np.array([1, length, heads, dh], np.int64))
    q, k, v = (b.node("Reshape", [linear_token(b, x, dim, dim), shape]) for _ in range(3))
    qh = b.node("Transpose", [q], perm=[0, 2, 1, 3])
    kh = b.node("Transpose", [k], perm=[0, 2, 1, 3])
    vh = b.node("Transpose", [v], perm=[0, 2, 1, 3])
    scores = b.node("Einsum", [qh, kh], equation="bhqd,bhkd->bhqk")
    probs = b.node("Softmax", [scores], axis=-1)
    ctx = b.node("Einsum", [probs, vh], equation="bhqk,bhkd->bhqd")
    merged = b.node("Transpose", [ctx], perm=[0, 2, 1, 3])
    out = b.node("Reshape", [merged, b.const(np.array([1, length, dim], np.int64))])
    return b.model([(x, [1, length, dim])], [(out, [1, length, dim])]), length, dim


def model_vit_batched_attention():
    """ViT multi-head attention block using batched MatMul lowering."""
    length, dim, heads = 4, 8, 2
    dh = dim // heads
    b = Builder(seed=73)
    x = "x"
    shape = b.const(np.array([1, length, heads, dh], np.int64))
    q, k, v = (b.node("Reshape", [linear_token(b, x, dim, dim), shape]) for _ in range(3))
    qh = b.node("Transpose", [q], perm=[0, 2, 1, 3])
    kh = b.node("Transpose", [k], perm=[0, 2, 1, 3])
    vh = b.node("Transpose", [v], perm=[0, 2, 1, 3])
    kt = b.node("Transpose", [kh], perm=[0, 1, 3, 2])
    scores = b.node("Mul", [b.node("MatMul", [qh, kt]), b.const(np.float32(1.0 / np.sqrt(dh)))])
    probs = b.node("Softmax", [scores], axis=-1)
    ctx = b.node("MatMul", [probs, vh])
    merged = b.node("Transpose", [ctx], perm=[0, 2, 1, 3])
    flat = b.node("Reshape", [merged, b.const(np.array([1, length, dim], np.int64))])
    out = b.node("Add", [x, linear_token(b, flat, dim, dim)])
    return b.model([(x, [1, length, dim])], [(out, [1, length, dim])]), length, dim


def model_causal_decoder_attention():
    """Causal decoder multi-head attention block with additive causal mask."""
    length, dim, heads = 4, 8, 2
    dh = dim // heads
    b = Builder(seed=74)
    x = "x"
    shape = b.const(np.array([1, length, heads, dh], np.int64))
    q, k, v = (b.node("Reshape", [linear_token(b, x, dim, dim), shape]) for _ in range(3))
    qh = b.node("Transpose", [q], perm=[0, 2, 1, 3])
    kh = b.node("Transpose", [k], perm=[0, 2, 1, 3])
    vh = b.node("Transpose", [v], perm=[0, 2, 1, 3])
    kt = b.node("Transpose", [kh], perm=[0, 1, 3, 2])
    scores = b.node("Mul", [b.node("MatMul", [qh, kt]), b.const(np.float32(1.0 / np.sqrt(dh)))])
    mask = np.zeros((1, heads, length, length), dtype=np.float32)
    for h in range(heads):
        for i in range(length):
            for j in range(i + 1, length):
                mask[0, h, i, j] = -10000.0
    masked_scores = b.node("Add", [scores, b.const(mask)])
    probs = b.node("Softmax", [masked_scores], axis=-1)
    ctx = b.node("MatMul", [probs, vh])
    merged = b.node("Transpose", [ctx], perm=[0, 2, 1, 3])
    flat = b.node("Reshape", [merged, b.const(np.array([1, length, dim], np.int64))])
    out = b.node("Add", [x, linear_token(b, flat, dim, dim)])
    return b.model([(x, [1, length, dim])], [(out, [1, length, dim])]), length, dim


def model_rank6_pixel_unshuffle():
    """Rank-6 reshape and transpose unshuffle (conv.merge_unshuffle) on token/image patches."""
    b = Builder(seed=75)
    tok = linear_token(b, "x", 16, 16)
    r6 = b.node("Reshape", [tok, b.const(np.array([1, 1, 1, 2, 2, 16], np.int64))])
    t6 = b.node("Transpose", [r6], perm=[0, 1, 3, 2, 4, 5])
    r_out = b.node("Reshape", [t6, b.const(np.array([1, 4, 16], np.int64))])
    out = linear_token(b, r_out, 16, 16)
    return b.model([("x", [1, 4, 16])], [(out, [1, 4, 16])]), 4, 16


def model_sdpa_op():
    """ScaledDotProductAttention op lowered to HX_SDPA."""
    length, dim, heads = 4, 8, 2
    dh = dim // heads
    b = Builder(seed=76)
    x = "x"
    shape = b.const(np.array([1, length, heads, dh], np.int64))
    q, k, v = (b.node("Reshape", [linear_token(b, x, dim, dim), shape]) for _ in range(3))
    qh = b.node("Transpose", [q], perm=[0, 2, 1, 3])
    kh = b.node("Transpose", [k], perm=[0, 2, 1, 3])
    vh = b.node("Transpose", [v], perm=[0, 2, 1, 3])
    ctx = b.node("ScaledDotProductAttention", [qh, kh, vh], scale=float(1.0 / np.sqrt(dh)))
    merged = b.node("Transpose", [ctx], perm=[0, 2, 1, 3])
    out = b.node("Reshape", [merged, b.const(np.array([1, length, dim], np.int64))])
    return b.model([(x, [1, length, dim])], [(out, [1, length, dim])], check=False), length, dim


def model_arbitrary_axis_softmax():
    """Softmax along sequence axis (axis 1) of a token tensor."""
    length, dim = 4, 8
    b = Builder(seed=77)
    tok = linear_token(b, "x", dim, dim)
    s1 = b.node("Softmax", [tok], axis=1)
    out = linear_token(b, s1, dim, dim)
    return b.model([("x", [1, length, dim])], [(out, [1, length, dim])]), length, dim


def ops_of(tmp):

    """All operation records of an HXFG image: 16 uint32 fields (kind, x, y, z, out, p[11])."""
    image = (Path(tmp) / "g.hxfg").read_bytes()
    tensors, ops = struct.unpack_from("<2I", image, 12)
    return [struct.unpack_from("<16I", image, 256 + 32 * tensors + 64 * i) for i in range(ops)]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class TransformerLoweringTests(unittest.TestCase):
    """Token layout, HX_MATMUL and the attention chain. Numeric checks reuse FloatOnnxTests'
    comparison against onnxruntime; structural checks read the lowered HXFG back."""

    def numeric(self, m, shape, tol=2e-4, x_input=None):
        return FloatOnnxTests.check_numeric(self, m, shape, tol, x_input=x_input)

    def test_single_head_attention_matches_onnxruntime(self):
        m, length, dim = model_single_head_attention()
        _, ops = self.numeric(m, (1, length, dim))
        self.assertEqual(ops.count(hx.CONV), 4, ops)  # Q, K, V and the output linear
        self.assertEqual(ops.count(hx.MATMUL), 2, ops)
        self.assertEqual(ops.count(hx.SOFTMAX), 1)
        self.assertEqual(ops.count(hx.BINARY), 0)  # the Div folded into the score product

    def test_single_head_attention_fields(self):
        m, length, dim = model_single_head_attention()
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            matmuls = [op for op in ops_of(tmp) if op[0] == hx.MATMUL]
        scores, context = matmuls
        self.assertEqual(scores[5], 1)  # transB: K is read as [N, K]
        self.assertAlmostEqual(f32(scores[6]), 1 / np.sqrt(dim), places=6)
        self.assertEqual((context[5], context[6]), (0, 0))  # attn . V, unscaled

    def test_two_head_attention_matches_onnxruntime(self):
        m, length, dim = model_two_head_attention()
        _, ops = self.numeric(m, (1, length, dim))
        # Q, K, V and the output linear are convolutions; each head adds slices, two products
        # and a softmax; the merged heads are one channel concatenation.
        self.assertEqual(ops.count(hx.CONV), 4, ops)
        self.assertEqual(ops.count(hx.SLICE), 6, ops)
        self.assertEqual(ops.count(hx.MATMUL), 4, ops)
        self.assertEqual(ops.count(hx.SOFTMAX), 2)
        self.assertEqual(ops.count(hx.CONCAT), 1)
        self.assertEqual(ops.count(hx.BINARY), 0)

    def test_two_head_scores_are_scaled_and_transposed_per_head(self):
        m, length, dim = model_two_head_attention()
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            matmuls = [op for op in ops_of(tmp) if op[0] == hx.MATMUL]
        qk, av = matmuls[:2], matmuls[2:]
        for op in qk:
            self.assertEqual(op[5], 1)  # transB
            self.assertAlmostEqual(f32(op[6]), 1 / np.sqrt(dim // 2), places=6)
        for op in av:
            self.assertEqual((op[5], op[6]), (0, 0))

    def test_vit_encoder_layer_matches_onnxruntime(self):
        m, length, dim = model_vit_encoder_layer()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertEqual(ops.count(hx.NORM), 2, ops)
        self.assertEqual(ops.count(hx.MATMUL), 4, ops)
        self.assertIn(hx.ACT, ops)  # exact GELU

    def test_patch_embedding_tokens_match_onnxruntime(self):
        _, ops = self.numeric(model_patch_embedding(), (1, 3, 8, 8))
        self.assertIn(hx.RESHAPE, ops)  # the [1, C, L] -> [1, L, C] logical reshape

    def test_sequence_gather_reduce_and_concat_match_onnxruntime(self):
        m, length, dim = model_sequence_ops()
        _, ops = self.numeric(m, (1, length, dim))
        self.assertEqual(ops.count(hx.SLICE), 2, ops)  # the row and the run
        self.assertEqual(ops.count(hx.REDUCE), 2, ops)
        self.assertEqual(ops.count(hx.CONCAT), 1, ops)

    def test_rms_normalization_lowering(self):
        m = model_rms_normalization()
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            ops = [op[0] for op in ops_of(tmp)]
        self.assertIn(hx.RMSNORM, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_simplified_layer_normalization_matches_onnxruntime(self):
        m, length, dim = model_simplified_layer_normalization()
        _, ops = self.numeric(m, (1, length, dim))
        self.assertIn(hx.RMSNORM, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_decomposed_rmsnorm_matches_onnxruntime(self):
        for with_weight in (True, False):
            with self.subTest(with_weight=with_weight):
                m, length, dim = model_decomposed_rmsnorm(with_weight=with_weight)
                _, ops = self.numeric(m, (1, length, dim))
                self.assertIn(hx.RMSNORM, ops)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_token_layer_norm_matches_onnxruntime(self):
        m, length, dim = model_token_layer_norm()
        _, ops = self.numeric(m, (1, length, dim))
        self.assertIn(hx.NORM, ops)

    def test_token_slice_matches_onnxruntime(self):
        m, length, dim = model_token_slice()
        _, ops = self.numeric(m, (1, length, dim))
        self.assertEqual(ops.count(hx.SLICE), 2, ops)
        self.assertEqual(ops.count(hx.CONCAT), 1, ops)

    def test_token_reshape_alias_matches_onnxruntime(self):
        m, length, dim = model_token_reshape_alias()
        _, ops = self.numeric(m, (1, length, dim))
        self.assertIn(hx.RESHAPE, ops)

    def test_batched_matmul_direct_matches_onnxruntime(self):
        m, length, dim = model_batched_matmul_explicit()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertIn(hx.MATMUL, ops)

    def test_einsum_attention_matches_onnxruntime(self):
        m, length, dim = model_einsum_attention()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertEqual(ops.count(hx.MATMUL), 2, ops)
        self.assertIn(hx.ATTN_SOFTMAX, ops)

    def test_vit_batched_attention_matches_onnxruntime(self):
        m, length, dim = model_vit_batched_attention()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertEqual(ops.count(hx.MATMUL), 2, ops)

    def test_causal_decoder_attention_matches_onnxruntime(self):
        m, length, dim = model_causal_decoder_attention()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertEqual(ops.count(hx.MATMUL), 2, ops)

    def test_rank6_pixel_unshuffle_matches_onnxruntime(self):
        m, length, dim = model_rank6_pixel_unshuffle()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertEqual(ops.count(hx.CONV), 2, ops)

    def test_sdpa_op_lowering_and_fields(self):
        m, length, dim = model_sdpa_op()
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            ops = ops_of(tmp)
            sdpa_ops = [op for op in ops if op[0] == hx.SDPA]
            self.assertEqual(len(sdpa_ops), 1)
            self.assertEqual(sdpa_ops[0][8], 2)

    def test_arbitrary_axis_softmax_matches_onnxruntime(self):
        m, length, dim = model_arbitrary_axis_softmax()
        _, ops = self.numeric(m, (1, length, dim), tol=5e-4)
        self.assertIn(hx.ATTN_SOFTMAX, ops)

    def test_gather_embedding_lookup_matches_onnxruntime(self):
        m = model_gather_embedding_lookup()
        _, ops = self.numeric(m, (1, 4), tol=5e-4)
        self.assertIn(hx.GATHER, ops)

    def test_gather_arbitrary_axis_matches_onnxruntime(self):
        m = model_gather_arbitrary_axis()
        _, ops = self.numeric(m, (1, 4, 6), tol=5e-4)
        self.assertIn(hx.GATHER, ops)

    def test_gather_elements_matches_onnxruntime(self):
        m = model_gather_elements()
        _, ops = self.numeric(m, (1, 4, 6), tol=5e-4)
        self.assertIn(hx.GATHER_ELEMENTS, ops)

    def test_gather_nd_matches_onnxruntime(self):
        m = model_gather_nd()
        _, ops = self.numeric(m, (1, 4, 8), tol=5e-4)
        self.assertIn(hx.GATHER_ND, ops)

    def test_scatter_nd_matches_onnxruntime(self):
        m = model_scatter_nd()
        _, ops = self.numeric(m, (1, 4, 8), tol=5e-4)
        self.assertIn(hx.SCATTER_ND, ops)

    def test_transformer_rejections_name_the_node(self):

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "Add"):
            lower_model(model_transposed_operand_in_add(), tmp)
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(ValueError, "contiguous run"):
            lower_model(model_gather_run_with_gap(), tmp)


@unittest.skipUnless(onnx and onnxruntime and RUNNER.exists(), "needs onnx, onnxruntime, and runner")
class IntegerTensorTests(unittest.TestCase):
    def test_integer_arithmetic_matches_onnxruntime(self):
        b = Builder(seed=101)
        outs = [
            b.node("Add", ["x", "y"]),
            b.node("Sub", ["x", "y"]),
            b.node("Mul", ["x", "y"]),
            b.node("Min", ["x", "y"]),
            b.node("Max", ["x", "y"]),
            b.node("Add", ["x", b.const(np.full((1, 4, 4, 8), 10, np.int32))]),
            b.node("Mul", ["x", b.const(np.full((1, 4, 4, 8), -2, np.int32))]),
        ]
        m = b.model(
            [("x", [1, 4, 4, 8], TensorProto.INT32), ("y", [1, 4, 4, 8], TensorProto.INT32)],
            [(o, [1, 4, 4, 8], TensorProto.INT32) for o in outs]
        )
        rng = np.random.RandomState(101)
        x = rng.randint(-100, 100, size=(1, 4, 4, 8)).astype(np.int32)
        y = rng.randint(-50, 50, size=(1, 4, 4, 8)).astype(np.int32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"x": x, "y": y}
        )
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            # pack both inputs
            data_x = x[0]
            data_y = y[0]
            p = Path(tmp)
            (p / "in.f32").write_bytes(np.ascontiguousarray(data_x).tobytes() + np.ascontiguousarray(data_y).tobytes())
            # runner takes: IMAGE CONSTANTS IN1 IN2 OUT_PREFIX
            (p / "in_x.i32").write_bytes(np.ascontiguousarray(data_x).tobytes())
            (p / "in_y.i32").write_bytes(np.ascontiguousarray(data_y).tobytes())
            subprocess.run(
                [str(RUNNER), str(p / "g.hxfg"), str(p / "g.bin"), str(p / "in_x.i32"), str(p / "in_y.i32"), str(p / "out_")],
                check=True, capture_output=True,
            )
            for i, w in enumerate(want):
                out_file = p / f"out_{i}.f32"
                got = np.fromfile(out_file, dtype=np.int32)
                np.testing.assert_array_equal(got, w[0].flatten(), err_msg=f"mismatch in int op {i}")

    def test_comparisons_and_logical_matches_onnxruntime(self):
        b = Builder(seed=102)
        eq = b.node("Equal", ["x", "y"])
        gt = b.node("Greater", ["x", "y"])
        lt = b.node("Less", ["x", "y"])
        ge = b.node("GreaterOrEqual", ["x", "y"])
        le = b.node("LessOrEqual", ["x", "y"])
        and_op = b.node("And", [gt, lt])
        or_op = b.node("Or", [ge, le])
        not_op = b.node("Not", [eq])
        outs = [eq, gt, lt, ge, le, and_op, or_op, not_op]
        m = b.model(
            [("x", [1, 2, 2, 4], TensorProto.INT32), ("y", [1, 2, 2, 4], TensorProto.INT32)],
            [(o, [1, 2, 2, 4], TensorProto.BOOL) for o in outs]
        )
        rng = np.random.RandomState(102)
        x = rng.randint(-10, 10, size=(1, 2, 2, 4)).astype(np.int32)
        y = rng.randint(-10, 10, size=(1, 2, 2, 4)).astype(np.int32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"x": x, "y": y}
        )
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            p = Path(tmp)
            (p / "in_x.i32").write_bytes(np.ascontiguousarray(x[0]).tobytes())
            (p / "in_y.i32").write_bytes(np.ascontiguousarray(y[0]).tobytes())
            subprocess.run(
                [str(RUNNER), str(p / "g.hxfg"), str(p / "g.bin"), str(p / "in_x.i32"), str(p / "in_y.i32"), str(p / "out_")],
                check=True, capture_output=True,
            )
            for i, w in enumerate(want):
                out_file = p / f"out_{i}.f32"
                got = np.fromfile(out_file, dtype=np.int32).astype(bool)
                np.testing.assert_array_equal(got, w[0].flatten(), err_msg=f"mismatch in comp/logical op {i}")

    def test_where_with_integer_and_bool_matches_onnxruntime(self):
        b = Builder(seed=103)
        cond = b.node("Greater", ["x", b.const(np.zeros((1, 2, 2, 4), np.int32))])
        w1 = b.node("Where", [cond, "x", "y"])
        w2 = b.node("Where", [cond, b.const(np.full((1, 2, 2, 4), 99, np.int32)), "y"])
        m = b.model(
            [("x", [1, 2, 2, 4], TensorProto.INT32), ("y", [1, 2, 2, 4], TensorProto.INT32)],
            [(w1, [1, 2, 2, 4], TensorProto.INT32), (w2, [1, 2, 2, 4], TensorProto.INT32)]
        )
        rng = np.random.RandomState(103)
        x = rng.randint(-5, 5, size=(1, 2, 2, 4)).astype(np.int32)
        y = rng.randint(10, 20, size=(1, 2, 2, 4)).astype(np.int32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"x": x, "y": y}
        )
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            p = Path(tmp)
            (p / "in_x.i32").write_bytes(np.ascontiguousarray(x[0]).tobytes())
            (p / "in_y.i32").write_bytes(np.ascontiguousarray(y[0]).tobytes())
            subprocess.run(
                [str(RUNNER), str(p / "g.hxfg"), str(p / "g.bin"), str(p / "in_x.i32"), str(p / "in_y.i32"), str(p / "out_")],
                check=True, capture_output=True,
            )
            for i, w in enumerate(want):
                got = np.fromfile(p / f"out_{i}.f32", dtype=np.int32)
                np.testing.assert_array_equal(got, w[0].flatten(), err_msg=f"mismatch in where op {i}")

    def test_argmin_and_argmax_matches_onnxruntime(self):
        b = Builder(seed=104)
        am_float = b.node("ArgMax", ["xf"], axis=1, keepdims=1)
        an_float = b.node("ArgMin", ["xf"], axis=1, keepdims=1)
        am_int = b.node("ArgMax", ["xi"], axis=1, keepdims=1)
        an_int = b.node("ArgMin", ["xi"], axis=1, keepdims=1)
        m = b.model(
            [("xf", [1, 8, 2, 2], TensorProto.FLOAT), ("xi", [1, 8, 2, 2], TensorProto.INT32)],
            [(am_float, [1, 1, 2, 2], TensorProto.INT64),
             (an_float, [1, 1, 2, 2], TensorProto.INT64),
             (am_int, [1, 1, 2, 2], TensorProto.INT64),
             (an_int, [1, 1, 2, 2], TensorProto.INT64)]
        )
        rng = np.random.RandomState(104)
        xf = rng.randn(1, 8, 2, 2).astype(np.float32)
        xi = rng.randint(-50, 50, size=(1, 8, 2, 2)).astype(np.int32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"xf": xf, "xi": xi}
        )
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            p = Path(tmp)
            (p / "in_xf.f32").write_bytes(np.ascontiguousarray(xf[0].transpose(1, 2, 0)).tobytes())
            (p / "in_xi.i32").write_bytes(np.ascontiguousarray(xi[0].transpose(1, 2, 0)).tobytes())
            subprocess.run(
                [str(RUNNER), str(p / "g.hxfg"), str(p / "g.bin"), str(p / "in_xf.f32"), str(p / "in_xi.i32"), str(p / "out_")],
                check=True, capture_output=True,
            )
            for i, w in enumerate(want):
                got = np.fromfile(p / f"out_{i}.f32", dtype=np.int32)
                np.testing.assert_array_equal(got, w.flatten(), err_msg=f"mismatch in argmax/argmin {i}")

    def test_cast_roundtrips_and_types_matches_onnxruntime(self):
        b = Builder(seed=105)
        # float -> int32
        c_f2i = b.node("Cast", ["x"], to=TensorProto.INT32)
        # int32 -> float
        c_i2f = b.node("Cast", [c_f2i], to=TensorProto.FLOAT)
        # float -> bool
        c_f2b = b.node("Cast", ["x"], to=TensorProto.BOOL)
        # bool -> float
        c_b2f = b.node("Cast", [c_f2b], to=TensorProto.FLOAT)
        # int32 -> bool
        c_i2b = b.node("Cast", [c_f2i], to=TensorProto.BOOL)
        # bool -> int32
        c_b2i = b.node("Cast", [c_i2b], to=TensorProto.INT32)

        m = b.model(
            [("x", [1, 4, 1, 2], TensorProto.FLOAT)],
            [(c_f2i, [1, 4, 1, 2], TensorProto.INT32),
             (c_i2f, [1, 4, 1, 2], TensorProto.FLOAT),
             (c_f2b, [1, 4, 1, 2], TensorProto.BOOL),
             (c_b2f, [1, 4, 1, 2], TensorProto.FLOAT),
             (c_i2b, [1, 4, 1, 2], TensorProto.BOOL),
             (c_b2i, [1, 4, 1, 2], TensorProto.INT32)]
        )
        x = np.array([[[[3.2, -4.7]], [[0.0, 1.0]], [[2.1, 0.0]], [[-0.5, 9.9]]]], dtype=np.float32)
        want = onnxruntime.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"x": x}
        )
        with tempfile.TemporaryDirectory() as tmp:
            lower_model(m, tmp)
            got = list(run_reference(tmp, x))
            for i, (g, w) in enumerate(zip(got, want)):
                np.testing.assert_allclose(g, w[0].transpose(1, 2, 0).reshape(-1), atol=1e-4, err_msg=f"mismatch in cast op {i}")

    def test_int64_narrowing_and_range_check(self):
        b = Builder(seed=106)
        # Normal int64 constant narrows to int32
        c_ok = b.const(np.array([1, 2, 3], dtype=np.int64))
        self.assertIsNotNone(c_ok)
        # Out of range int64 constant raises ValueError
        b2 = Builder()
        b2.const(np.array([2**31 + 5], dtype=np.int64))
        m = b2.model([("x", [1, 1, 1, 1])], [(b2.node("Add", ["x", b2.const(np.array([2**31 + 5], dtype=np.int64))]), [1, 1, 1, 1])], check=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "out of int32 range"):
                lower_model(m, tmp)


if __name__ == "__main__":
    unittest.main()
