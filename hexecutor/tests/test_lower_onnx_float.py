"""Float ONNX -> HXFG lowering, BatchNorm folding and rejection paths (synthetic models only).

Skips when the offline `onnx` package is absent. The numeric tests also need onnxruntime
and the portable reference runner (build/bin/hexecutor_tensor_run); they compare the
lowered graph on the CPU reference against onnxruntime on the original model.
"""

import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
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
RUNNER = ROOT / "build/bin/hexecutor_tensor_run"


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

    def model(self, inputs, outputs, opset=17):
        g = helper.make_graph(
            self.nodes,
            "g",
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
            self.init,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
        m.ir_version = 9
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
    (tmp / "in.f32").write_bytes(np.ascontiguousarray(data, np.float32).tobytes())
    subprocess.run(
        [str(RUNNER), str(tmp / "g.hxfg"), str(tmp / "g.bin"), str(tmp / "in.f32"),
         str(tmp / "out_")],
        check=True, capture_output=True,
    )
    i = 0
    while (tmp / f"out_{i}.f32").exists():
        yield np.fromfile(tmp / f"out_{i}.f32", np.float32)
        i += 1


def kinds(tmp):
    image = (Path(tmp) / "g.hxfg").read_bytes()
    tensors, ops = struct.unpack_from("<2I", image, 12)
    return [struct.unpack_from("<I", image, 256 + 32 * tensors + 64 * i)[0] for i in range(ops)]


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class FloatOnnxTests(unittest.TestCase):
    def check_numeric(self, m, shape, tol=2e-4):
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

    def test_bn_fold_changes_weights_not_op_count(self):
        b = Builder()
        out = b.node("Relu", [b.bn(b.conv("x", 3, 8), 8)])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lower_model(b.model([("x", [1, 3, 8, 8])], [(out, [1, 8, 8, 8])]), tmp)
            self.assertEqual(kinds(tmp), [hx.CONV])
            self.assertEqual(len(manifest["ops"]), 2)  # BatchNormalization and Relu are recorded

    def test_fp16_blockers_are_reported(self):
        b = Builder()
        out = b.conv(b.conv("x", 3, 32), 32, 32)  # a dense 3x3 in the middle
        with tempfile.TemporaryDirectory() as tmp:
            plan = lower_model(b.model([("x", [1, 3, 8, 8])], [(out, [1, 32, 8, 8])]), tmp)[
                "fp16_plan"
            ]
            self.assertFalse(plan["eligible"])
            self.assertTrue(any("3x3 stride-1 convolution" in r for r in plan["blockers"]))
        b = Builder()
        out = b.node("Relu", [b.bn(b.conv("x", 3, 32), 32)])
        with tempfile.TemporaryDirectory() as tmp:
            m = b.model([("x", [1, 3, 8, 8])], [(out, [1, 32, 8, 8])])
            self.assertTrue(lower_model(m, tmp)["fp16_plan"]["eligible"])

    def test_rejections_name_the_node(self):
        def softmax():
            b = Builder()
            return b.model([("x", [1, 8, 4, 4])], [(b.node("Softmax", ["x"], axis=1), [1, 8, 4, 4])])

        def dilated():
            b = Builder()
            out = b.conv("x", 3, 8, dilations=[2, 2], pads=[2, 2, 2, 2])
            return b.model([("x", [1, 3, 8, 8])], [(out, [1, 8, 8, 8])])

        def grouped():
            b = Builder()
            return b.model([("x", [1, 4, 8, 8])], [(b.conv("x", 4, 8, group=2), [1, 8, 8, 8])])

        def flatten_spatial():
            b = Builder()
            return b.model([("x", [1, 4, 2, 2])], [(b.node("Flatten", ["x"]), [1, 16])])

        def tanh_gelu():
            b = Builder()
            return b.model([("x", [1, 4, 2, 2])],
                           [(b.node("Gelu", ["x"], approximate="tanh"), [1, 4, 2, 2])], opset=20)

        def spatial_const():
            b = Builder()
            out = b.node("Add", ["x", b.const(b.rand(1, 4, 2, 2))])
            return b.model([("x", [1, 4, 2, 2])], [(out, [1, 4, 2, 2])])

        def passthrough():
            b = Builder()
            return b.model([("x", [1, 4, 2, 2])], [(b.node("Identity", ["x"]), [1, 4, 2, 2])])

        for build, text in [
            (softmax, "Softmax"), (dilated, "dilation"), (grouped, "groups"),
            (flatten_spatial, "mixes spatial"), (tanh_gelu, "exact (erf)"),
            (spatial_const, "per-channel"), (passthrough, "graph input"),
        ]:
            with self.subTest(text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, text):
                    lower_model(build(), tmp)


if __name__ == "__main__":
    unittest.main()
