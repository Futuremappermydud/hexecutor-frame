# SPDX-License-Identifier: LGPL-3.0-or-later
"""General-model operator coverage of the float ONNX lowering (synthetic models).

Each test lowers a small model and compares the portable reference runner with onnxruntime:
Conv1d/pool1d, ceil-mode MaxPool, strided Slice, general Clip, the extra activations,
Reduce* and LogSoftmax decompositions, DepthToSpace/SpaceToDepth and channel shuffle. The
rank-5 reordering transpose must be rejected, never aliased. Needs onnx, onnxruntime and
build/bin/hexecutor_tensor_run (the tests skip without them).
"""

import tempfile
import unittest
import unittest.mock

import numpy as np

from hexecutor.tests import test_lower_onnx_float as base

Builder, RUNNER, hx, lowering = base.Builder, base.RUNNER, base.hx, base.lowering
onnx, onnxruntime = base.onnx, base.onnxruntime

needs_runner = unittest.skipUnless(onnx and onnxruntime and RUNNER.exists(),
                                   "needs onnx, onnxruntime and the runner")


class GeneralOpTests(unittest.TestCase):
    check_numeric = base.FloatOnnxTests.check_numeric

    def run_unary(self, op, shape, out_shape=None, opset=17, tol=2e-4, **attrs):
        b = Builder(seed=5)
        out = b.node(op, ["x"], **attrs)
        m = b.model([("x", list(shape))], [(out, list(out_shape or shape))], opset=opset)
        return self.check_numeric(m, shape, tol=tol)

    @needs_runner
    def test_new_activations_match_onnxruntime(self):
        for op, attrs in [("Elu", {"alpha": 0.7}), ("Celu", {"alpha": 1.3}), ("Selu", {}),
                          ("Softplus", {}), ("Mish", {}), ("Softsign", {}), ("Sinh", {}),
                          ("Cosh", {}), ("Sign", {})]:
            with self.subTest(op=op):
                _, ops = self.run_unary(op, (1, 4, 5, 5), opset=18, tol=1e-3, **attrs)
                self.assertIn(hx.ACT, ops)

    @needs_runner
    def test_general_clip_matches_onnxruntime(self):
        for lo, hi in [(-1.0, 1.0), (-0.5, None), (None, 0.25)]:
            with self.subTest(lo=lo, hi=hi):
                b = Builder(seed=2)
                ins = ["x", b.const(lo) if lo is not None else "",
                       b.const(hi) if hi is not None else ""]
                out = b.node("Clip", ins)
                m = b.model([("x", [1, 4, 5, 5])], [(out, [1, 4, 5, 5])])
                self.check_numeric(m, (1, 4, 5, 5))

    @needs_runner
    def test_conv1d_and_pool1d_match_onnxruntime(self):
        b = Builder(seed=4)
        w = b.const(b.rand(8, 4, 5))
        c = b.node("Conv", ["x", w, b.const(b.rand(8, scale=0.1))], kernel_shape=[5],
                   pads=[2, 2], strides=[1], dilations=[1])
        d = b.node("Conv", [c, b.const(b.rand(8, 1, 3)), b.const(b.rand(8, scale=0.1))],
                   kernel_shape=[3], pads=[2, 2], dilations=[2], group=8)
        p = b.node("MaxPool", [d], kernel_shape=[2], strides=[2])
        m = b.model([("x", [1, 4, 32])], [(p, [1, 8, 16])])
        # Rank-3 [1, C, L] is NCHW with H = 1: the runner's HWC layout is [L, C].
        x = np.random.default_rng(11).normal(0, 1, (1, 4, 32)).astype(np.float32)
        want = onnxruntime.InferenceSession(m.SerializeToString(),
                                            providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        with tempfile.TemporaryDirectory() as tmp:
            base.lower_model(m, tmp)
            got = list(base.run_reference(tmp, np.ascontiguousarray(x[0].T)[None]))[0]
        np.testing.assert_allclose(got.reshape(16, 8), want[0].T, rtol=2e-4, atol=2e-4)

    @needs_runner
    def test_maxpool_ceil_mode_matches_onnxruntime(self):
        for size, k, s, pad in [(16, 3, 2, 0), (15, 3, 2, 0), (10, 3, 2, 1), (9, 3, 3, 0)]:
            with self.subTest(size=size, k=k, s=s, pad=pad):
                b = Builder(seed=3)
                out = b.node("MaxPool", ["x"], kernel_shape=[k, k], strides=[s, s],
                             pads=[pad] * 4, ceil_mode=1)
                shape = (1, 3, size, size)
                m = b.model([("x", list(shape))], [(out, [1, 3, None, None])], check=False)
                self.check_numeric(m, shape)

    @needs_runner
    def test_strided_slice_matches_onnxruntime(self):
        cases = [([0, 0], [8, 8], [2, 3], [2, 3], (1, 4, 8, 8)),   # H, W
                 ([1], [7], [1], [2], (1, 8, 4, 4)),               # channels
                 ([1, 0, 1], [6, 8, 8], [1, 2, 3], [2, 2, 3], (1, 6, 8, 8))]  # all three
        for starts, ends, axes_or_steps, steps, shape in cases:
            with self.subTest(starts=starts):
                b = Builder(seed=1)
                if len(starts) == 2:
                    axes = [2, 3]
                elif len(starts) == 1:
                    axes = [1]
                else:
                    axes = [1, 2, 3]
                out = b.node("Slice", ["x", b.const(np.array(starts, np.int64)),
                                       b.const(np.array(ends, np.int64)),
                                       b.const(np.array(axes, np.int64)),
                                       b.const(np.array(steps, np.int64))])
                m = b.model([("x", list(shape))], [(out, [1, None, None, None])], check=False)
                self.check_numeric(m, shape)

    @needs_runner
    def test_reduce_decompositions_match_onnxruntime(self):
        for op in ("ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSumExp"):
            for keep in (1, 0):
                with self.subTest(op=op, keepdims=keep):
                    b = Builder(seed=6)
                    out = b.node(op, ["x"], axes=[2, 3], keepdims=keep)
                    shape = [1, 4, 1, 1] if keep else [1, 4]
                    m = b.model([("x", [1, 4, 5, 5])], [(out, shape)], opset=13)
                    self.check_numeric(m, (1, 4, 5, 5), tol=1e-3)

    @needs_runner
    def test_log_softmax_is_accurate_for_extreme_logits(self):
        b = Builder(seed=7)
        w = b.const(b.rand(10, 16, scale=3.0))
        out = b.node("LogSoftmax", [b.node("Gemm", ["x", w], transB=1)], axis=-1)
        m = b.model([("x", [1, 16])], [(out, [1, 10])])
        x = np.random.default_rng(1).normal(0, 20, (1, 16)).astype(np.float32)
        self.check_numeric(m, (1, 16), tol=1e-3, x_input=x)

    @needs_runner
    def test_depth_to_space_and_space_to_depth_match_onnxruntime(self):
        for mode in ("DCR", "CRD"):
            with self.subTest(mode=mode):
                b = Builder(seed=8)
                out = b.node("DepthToSpace", ["x"], blocksize=2, mode=mode)
                m = b.model([("x", [1, 12, 4, 4])], [(out, [1, 3, 8, 8])])
                self.check_numeric(m, (1, 12, 4, 4))
        b = Builder(seed=8)
        out = b.node("SpaceToDepth", ["x"], blocksize=2)
        m = b.model([("x", [1, 3, 8, 8])], [(out, [1, 12, 4, 4])])
        self.check_numeric(m, (1, 3, 8, 8))

    def _shuffle_model(self, perm=(0, 2, 1, 3, 4), out_shape=(1, 8, 4, 4)):
        b = Builder(seed=9)
        shp = lambda v: b.const(np.array(v, np.int64))  # noqa: E731
        a = b.node("Reshape", ["x", shp([1, 2, 4, 4, 4])])
        t = b.node("Transpose", [a], perm=list(perm))
        out = b.node("Reshape", [t, shp(list(out_shape))])
        return b.model([("x", [1, 8, 4, 4])], [(out, list(out_shape))], check=False)

    @needs_runner
    def test_channel_shuffle_matches_onnxruntime(self):
        _, ops = self.check_numeric(self._shuffle_model(), (1, 8, 4, 4))
        self.assertEqual(ops, [hx.CONV])

    @needs_runner
    def test_rank5_reordering_transpose_is_rejected_not_aliased(self):
        # Same HWC shape before and after, but the data order changes: never an alias.
        b = Builder(seed=9)
        shp = b.const(np.array([1, 2, 4, 2, 2], np.int64))
        a = b.node("Reshape", ["x", shp])
        t = b.node("Transpose", [a], perm=[0, 2, 1, 3, 4])
        out = b.node("Relu", [t])
        m = b.model([("x", [1, 8, 2, 2])], [(out, [1, 4, 2, 2, 2])], check=False)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError) as cm:
            base.lower_model(m, tmp)
        self.assertIn("reorders data", str(cm.exception))

    @needs_runner
    def test_large_weights_fall_back_to_fp32_instead_of_failing(self):
        # FP32 weights fit the pool but FP32 + packed FP16 would not: the graph must still lower.
        b = Builder(seed=10)
        x = "x"
        for _ in range(2):
            x = b.node("Relu", [b.conv(x, 512, 512, k=3)])
        m = b.model([("x", [1, 512, 4, 4])], [(x, [1, 512, 4, 4])])
        with unittest.mock.patch.object(lowering, "CONSTANT_POOL_LIMIT", 22 << 20):
            manifest, _ = self.check_numeric(m, (1, 512, 4, 4), tol=1e-3)
        self.assertFalse(manifest["fp16_plan"]["eligible"])


if __name__ == "__main__":
    unittest.main()
