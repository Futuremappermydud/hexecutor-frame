# SPDX-License-Identifier: LGPL-3.0-or-later
"""ONNX lowering of the LFM decode operators: MatMulNBits, RotaryEmbedding, ShortConv, KVCacheWrite.

Numeric checks run the lowered graph on the portable reference runner and compare with onnxruntime
(MatMulNBits, RotaryEmbedding); the stateful operators have no onnxruntime kernel, so they are checked
structurally here and numerically by the simulator test (build-scripts/test_lfm_sim.sh).
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper
    from hexecutor.offline import hxfg_writer as hx
    from hexecutor.tests import test_lower_onnx_float as base
except ImportError:  # offline tooling dependency
    onnx = None

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(os.environ.get("HEXECUTOR_TENSOR_RUN", ROOT / "build/bin/hexecutor_tensor_run"))
EXAMPLES = ROOT / "hexecutor/examples/llm/lfm"
NONE = 0xFFFFFFFF


def make_model(nodes, inputs, outputs, inits, opset=23):
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info(n, t, s) for n, t, s in outputs],
        [numpy_helper.from_array(np.asarray(v), n) for n, v in inits.items()])
    m = helper.make_model(g, opset_imports=[
        helper.make_opsetid("", opset), helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("hexecutor", 1)])
    m.ir_version = 10
    return m


def nbits_model(length, k, n, bits, bias=True, seed=1, zero_points=None, block=32):
    """MatMulNBits with fp16-representable scales; returns the model."""
    rng = np.random.default_rng(seed)
    nb = k // 32
    per = 32 * bits // 8
    data = rng.integers(0, 256, (n, nb, per), dtype=np.uint8)
    scales = (rng.uniform(0.01, 0.05, (n, nb)) * rng.choice([-1, 1], (n, nb))).astype(np.float16).astype(np.float32)
    inits = {"B": data, "S": scales.reshape(-1)}
    ins = ["x", "B", "S", "", "", ""]
    if zero_points is not None:
        inits["Z"] = zero_points
        ins[3] = "Z"
    if bias:
        inits["bias"] = rng.normal(0, 0.1, n).astype(np.float32)
        ins[5] = "bias"
    while ins and ins[-1] == "":
        ins.pop()
    node = helper.make_node("MatMulNBits", ins, ["y"], domain="com.microsoft", K=k, N=n, bits=bits,
                            block_size=block)
    return make_model([node], [("x", T.FLOAT, [1, length, k])], [("y", T.FLOAT, [1, length, n])], inits)


def rope_tables(rows, rot, theta):
    ang = np.arange(rows)[:, None] * float(theta) ** (-2.0 * np.arange(rot // 2) / rot)[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope_model(length, heads, head_dim, rot, theta, start=0, interleaved=0, rows=64, ms=False, scale_cos=1.0):
    cos, sin = rope_tables(rows, rot, theta)
    pos = (start + np.arange(length, dtype=np.int64)).reshape(1, length)
    inits = {"cos": cos * scale_cos, "sin": sin, "pos": pos}
    if ms:
        node = helper.make_node("RotaryEmbedding", ["x", "pos", "cos", "sin"], ["y"], domain="com.microsoft",
                                interleaved=interleaved, num_heads=heads, rotary_embedding_dim=rot if rot != head_dim else 0)
    else:
        node = helper.make_node("RotaryEmbedding", ["x", "cos", "sin", "pos"], ["y"], interleaved=interleaved,
                                num_heads=heads, rotary_embedding_dim=rot if rot != head_dim else 0)
    d = heads * head_dim
    return make_model([node], [("x", T.FLOAT, [1, length, d])], [("y", T.FLOAT, [1, length, d])], inits)


def state_model(conv=True, max_len=8, c=64, taps=3, names=("s",), runtime_pos=False, length=1, gate=True):
    nodes, inits, inputs = [], {}, [("x", T.FLOAT, [1, length, c])]
    if runtime_pos:
        inputs.append(("pos", T.INT64, [1, 1]))
    outputs = []
    if conv:
        inits["w"] = np.random.default_rng(3).normal(0, 0.3, (c, 1, taps)).astype(np.float32)
        inits["b"] = np.random.default_rng(4).normal(0, 0.1, c).astype(np.float32)
        ins = ["x", "w", "b"] + (["x"] if gate else [])
        nodes.append(helper.make_node("ShortConv", ins, ["y"], domain="hexecutor", state=names[0], activation="silu"))
        outputs.append(("y", T.FLOAT, [1, length, c]))
    else:
        for i, name in enumerate(names):
            ins = ["x"] + (["pos"] if runtime_pos else [])
            nodes.append(helper.make_node("KVCacheWrite", ins, [f"c{i}"], domain="hexecutor", state=name,
                                          max_len=max_len, position=2))
            outputs.append((f"c{i}", T.FLOAT, [1, max_len, c]))
    return make_model(nodes, inputs, outputs, inits)


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class QuantizedLinearTests(unittest.TestCase):
    def numeric(self, m, shape, tol):
        return base.FloatOnnxTests.check_numeric(self, m, shape, tol)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_q4_and_q8_match_onnxruntime(self):
        for bits in (4, 8):
            for length, bias in ((1, True), (3, True), (2, False)):
                with self.subTest(bits=bits, length=length, bias=bias):
                    _, ops = self.numeric(nbits_model(length, 96, 64, bits, bias), (1, length, 96), 2e-3)
                    self.assertEqual(ops, [hx.LINEAR_Q])

    def test_fields_and_packing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = base.lower_model(nbits_model(2, 64, 32, 4), tmp)
            (op,) = base.ops_of(tmp)
            blob = (Path(tmp) / "g.bin").read_bytes()
        self.assertEqual((op[0], op[5], op[8], op[9], op[10]), (hx.LINEAR_Q, 0, 64, 32, 0))
        self.assertNotEqual(op[7], NONE)  # bias
        self.assertGreaterEqual(len(blob), 32 * 2 * 18)  # 18-byte Q4_0 blocks
        self.assertEqual(manifest["state_bytes"], 0)

    def test_default_zero_points_are_accepted_and_others_rejected(self):
        nb = 2
        good = np.full((32 * nb // 2,), 0x88, np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            base.lower_model(nbits_model(1, 64, 32, 4, zero_points=good), tmp)
            bad = good.copy()
            bad[3] = 0x87
            with self.assertRaisesRegex(ValueError, "zero points"):
                base.lower_model(nbits_model(1, 64, 32, 4, zero_points=bad), tmp)

    def test_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "multiples of 32"):
                base.lower_model(nbits_model(1, 64, 48, 4), tmp)
            with self.assertRaisesRegex(ValueError, "block_size 32"):
                base.lower_model(nbits_model(1, 64, 32, 4, block=64), tmp)


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class RotaryTests(unittest.TestCase):
    def numeric(self, m, shape, tol=2e-4):
        return base.FloatOnnxTests.check_numeric(self, m, shape, tol)

    @unittest.skipUnless(onnxruntime and RUNNER.exists(), "needs onnxruntime and the runner")
    def test_matches_onnxruntime(self):
        cases = [
            dict(length=3, heads=2, head_dim=64, rot=64, theta=10000.0),
            dict(length=4, heads=2, head_dim=64, rot=32, theta=1e6, start=5),
            dict(length=1, heads=1, head_dim=32, rot=32, theta=1e6, start=9, interleaved=1),
            dict(length=3, heads=2, head_dim=32, rot=16, theta=500000.0, start=2, interleaved=1),
        ]
        for case in cases:
            with self.subTest(**case):
                kw = dict(case)
                rot = kw.pop("rot")
                m = rope_model(kw.pop("length"), kw.pop("heads"), kw.pop("head_dim"), rot, kw.pop("theta"), **kw)
                shape = tuple(d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim)
                _, ops = self.numeric(m, shape)
                self.assertEqual(ops, [hx.ROPE])

    def test_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base.lower_model(rope_model(4, 2, 64, 32, 1e6, start=5, ms=True), tmp)
            (op,) = base.ops_of(tmp)
        head_dim, heads, start, mode, theta_bits, rot, per_row = op[5:12]
        self.assertEqual((head_dim, heads, start, mode, rot, per_row), (64, 2, 5, 0, 32, 1))
        self.assertAlmostEqual(struct.unpack("<f", struct.pack("<I", theta_bits))[0] / 1e6, 1.0, places=5)
        self.assertEqual(op[2], NONE)  # constant positions need no operand

    def test_rejects_other_tables_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "rotary table"):
                base.lower_model(rope_model(2, 1, 64, 64, 1e4, scale_cos=1.05), tmp)
            m = rope_model(3, 1, 64, 64, 1e4)
            for init in m.graph.initializer:
                if init.name == "pos":
                    init.CopyFrom(numpy_helper.from_array(np.array([[0, 2, 1]], np.int64), "pos"))
            with self.assertRaisesRegex(ValueError, "consecutive"):
                base.lower_model(m, tmp)
            with self.assertRaisesRegex(ValueError, "beyond the cos/sin cache"):
                base.lower_model(rope_model(2, 1, 64, 64, 1e4, start=63, rows=64), tmp)


@unittest.skipIf(onnx is None, "offline onnx package not installed")
class StateOpTests(unittest.TestCase):
    def test_shortconv_lowers_to_a_persistent_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = base.lower_model(state_model(conv=True), tmp)
            (op,) = base.ops_of(tmp)
        self.assertEqual((op[0], op[5], op[8]), (hx.SHORTCONV, 3, 3))  # taps, silu
        self.assertEqual(manifest["state_bytes"], 2 * 64 * 4)
        (entry,) = manifest["persistent"].values()
        self.assertEqual(entry["offset"], 0)
        self.assertEqual(list(manifest["state_names"].values()), ["s"])

    def test_kv_cache_write_static_and_runtime_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = base.lower_model(state_model(conv=False, names=("k", "v")), tmp)
            ops = base.ops_of(tmp)
            self.assertEqual([o[0] for o in ops], [hx.KV_WRITE, hx.KV_WRITE])
            self.assertEqual((ops[0][2], ops[0][5]), (NONE, 2))  # static position from the attribute
            self.assertEqual(manifest["state_bytes"], 2 * 8 * 64 * 4)
            self.assertEqual(sorted(manifest["state_names"].values()), ["k", "v"])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = base.lower_model(state_model(conv=False, runtime_pos=True), tmp)
            (op,) = base.ops_of(tmp)
            self.assertNotEqual(op[2], NONE)
            dtypes = manifest["tensor_dtypes"]
            self.assertEqual(dtypes[str(op[2])], hx.DTYPE_INT32)
            self.assertIn("integer position operand", " ".join(manifest["fp16_plan"]["blockers"]))

    def test_duplicate_state_names_and_bad_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "already used"):
                base.lower_model(state_model(conv=False, names=("k", "k")), tmp)
            with self.assertRaisesRegex(ValueError, "one token per run"):
                base.lower_model(state_model(conv=True, length=2), tmp)

    def test_attention_over_a_cache_blocks_the_fp16_plan(self):
        # The full decode graph from the generator script: counted ops and the manifest.
        sys.path.insert(0, str(EXAMPLES))
        import gen_onnx_decode_golden as gen

        with tempfile.TemporaryDirectory() as tmp:
            gen.main(tmp)
            manifest = json.loads((Path(tmp) / "lfm_decode.json").read_text())
        self.assertFalse(manifest["fp16_plan"]["eligible"])
        self.assertTrue(any("persistent state" in b for b in manifest["fp16_plan"]["blockers"]))
        self.assertEqual(manifest["state_bytes"], 4608)
        self.assertEqual(sorted(manifest["state_names"].values()), ["conv0", "k_cache", "v_cache"])


if __name__ == "__main__":
    unittest.main()
