# SPDX-License-Identifier: LGPL-3.0-or-later
"""QLinear ONNX -> HXFG lowering on a tiny synthetic model (no real weights).

Skips when the offline `onnx` package is absent. When the portable reference
runner exists (build/bin/hexecutor_tensor_run), the lowered graph is also
evaluated and compared with a float NumPy evaluation of the dequantized model.
"""

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # offline tooling dependency
    onnx = None

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "build/bin/hexecutor_tensor_run"
RELU6 = 6.0 / 255.0


def model(gemm_out=5000, dilation=1):
    rng = np.random.default_rng(1)
    init, nodes = [], []

    def c(name, value):
        init.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def qconv(name, x, xs, xzp, w, ys, yzp, group=1, stride=1, pad=1):
        co, k = w.shape[0], w.shape[2]
        ws = rng.uniform(0.002, 0.01, co).astype(np.float32)
        b = rng.integers(-500, 500, co).astype(np.int32)
        nodes.append(
            helper.make_node(
                "QLinearConv",
                [x, xs, xzp, c(name + "_w", w), c(name + "_ws", ws),
                 c(name + "_wzp", np.zeros(co, np.int8)), c(name + "_ys", np.float32(ys)),
                 c(name + "_yzp", np.uint8(yzp)), c(name + "_b", b)],
                [name],
                group=group, kernel_shape=[k, k], strides=[stride, stride],
                pads=[pad] * 4, dilations=[dilation, dilation],
            )
        )
        return name, name + "_ys", name + "_yzp"

    def wts(*shape):
        return rng.integers(-60, 60, shape).astype(np.int8)

    nodes.append(helper.make_node("QuantizeLinear",
                                  ["x", c("xs", np.float32(0.02)), c("xzp", np.uint8(128))], ["xq"]))
    a = qconv("c0", "xq", "xs", "xzp", wts(32, 2, 3, 3), RELU6, 0, stride=2)  # ReLU6
    b = qconv("dw", *a, wts(32, 1, 3, 3), 0.05, 0, group=32)  # ReLU (range 12.75)
    l1 = qconv("p1", *b, wts(32, 32, 1, 1), 0.04, 128, pad=0)  # linear
    l2 = qconv("p2", *l1, wts(32, 32, 1, 1), 0.04, 130, pad=0)  # linear
    nodes.append(helper.make_node(
        "QLinearAdd", [*l1, *l2, c("add_s", np.float32(0.06)), c("add_zp", np.uint8(127))],
        ["sum"], domain="com.microsoft"))
    nodes.append(helper.make_node("DequantizeLinear", ["sum", "add_s", "add_zp"], ["sumf"]))
    nodes.append(helper.make_node("ReduceMean", ["sumf"], ["mean"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node(
        "Gemm", ["mean", c("gw", rng.normal(0, 0.1, (gemm_out, 32)).astype(np.float32)),
                 c("gb", rng.normal(0, 0.1, gemm_out).astype(np.float32))],
        ["y"], transB=1))
    graph = helper.make_graph(
        nodes, "synthetic",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, gemm_out])], init)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 8
    return m


def float_reference(m, x):
    """Dequantized float evaluation with the lowering's activation rules."""
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    val = {"x": x[0], "xq": x[0]}
    for n in m.graph.node:
        if n.op_type == "QLinearConv":
            a = val[n.input[0]]
            w = init[n.input[3]].astype(np.float64) * init[n.input[4]][:, None, None, None]
            bias = init[n.input[8]] * float(init[n.input[1]]) * init[n.input[4]]
            k, s, g = w.shape[2], dict((at.name, at) for at in n.attribute)["strides"].ints[0], \
                dict((at.name, at) for at in n.attribute)["group"].i
            pad = k // 2
            ap = np.pad(a, ((0, 0), (pad, pad), (pad, pad)))
            ho = (a.shape[1] + 2 * pad - k) // s + 1
            out = np.zeros((w.shape[0], ho, ho))
            for o in range(w.shape[0]):
                chans = [o] if g > 1 else range(a.shape[0])
                for ci_i, ci in enumerate(chans):
                    for dy in range(k):
                        for dx in range(k):
                            out[o] += w[o, 0 if g > 1 else ci_i, dy, dx] * \
                                ap[ci, dy:dy + s * ho:s, dx:dx + s * ho:s]
                out[o] += bias[o]
            ys, yzp = float(init[n.input[6]]), int(init[n.input[7]])
            if yzp == 0:
                out = np.maximum(out, 0)
                if abs(255 * ys - 6) < 1e-3:
                    out = np.minimum(out, 6)
            val[n.output[0]] = out
        elif n.op_type == "QLinearAdd":
            val[n.output[0]] = val[n.input[0]] + val[n.input[3]]
        elif n.op_type == "DequantizeLinear":
            val[n.output[0]] = val[n.input[0]]
        elif n.op_type == "ReduceMean":
            val[n.output[0]] = val[n.input[0]].mean(axis=(1, 2))
        elif n.op_type == "Gemm":
            val[n.output[0]] = init[n.input[1]] @ val[n.input[0]] + init[n.input[2]]
    return val["y"]


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class LowerOnnxTests(unittest.TestCase):
    def lower(self, m, root, **kw):
        from hexecutor.offline.lower_onnx import lower
        path = Path(root) / "m.onnx"
        onnx.save(m, path)
        return lower(path, Path(root) / "g", "x", "y", **kw)

    def test_activations_split_and_layout(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self.lower(model(), root, pad_input_channels=3)
            acts = [r["activation"] for r in manifest["activation_ranges"]]
            self.assertEqual(acts, [3, 1, 0, 0, 0])  # ReLU6, ReLU, linear x2, add
            self.assertEqual(manifest["inputs"][0]["shape"], [16, 16, 3])
            parts = manifest["output_parts"]
            # K = 32 fits the HMX budget up to 8,128 outputs; tensors cap at 4,096.
            self.assertEqual([p["channels"] for p in parts], [[0, 4096], [4096, 5000]])
            image = (Path(root) / "g.hxfg").read_bytes()
            magic, version, size, tensors, ops = struct.unpack_from("<5I", image)
            self.assertEqual((magic, version, size), (0x47465848, 1, len(image)))
            self.assertEqual(ops, 8)  # conv, dw, conv, conv, add, mean, 2 head convs

    def test_rejects_dilation(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "dilated"):
                self.lower(model(dilation=2), root)

    @unittest.skipUnless(RUNNER.exists(), "portable reference runner not built")
    def test_reference_matches_float_evaluation(self):
        m = model()
        x = np.random.default_rng(2).normal(0, 1, (1, 2, 16, 16)).astype(np.float32)
        want = float_reference(m, x)
        with tempfile.TemporaryDirectory() as root:
            manifest = self.lower(m, root, pad_input_channels=3)
            nhwc = np.zeros((16, 16, 3), np.float32)
            nhwc[..., :2] = x[0].transpose(1, 2, 0)
            (Path(root) / "in.f32").write_bytes(nhwc.tobytes())
            subprocess.run([str(RUNNER), f"{root}/g.hxfg", f"{root}/g.bin", f"{root}/in.f32",
                            f"{root}/out_"], check=True, capture_output=True)
            got = np.concatenate([np.fromfile(f"{root}/out_{i}.f32", np.float32)
                                  for i in range(len(manifest["output_parts"]))])
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
