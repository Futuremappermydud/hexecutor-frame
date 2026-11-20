# SPDX-License-Identifier: LGPL-3.0-or-later
"""HXFG writer: persistent state tensors, ops 23-26 and the Q4_0/Q8_0 packers (no onnx needed)."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hexecutor.offline import hxfg_writer as hx  # noqa: E402

NONE = hx.NONE


def _unpack(path):
    data = path.with_suffix(".hxfg").read_bytes()
    magic, version, size, tensors, ops, inputs, outputs, arena, consts, to, oo, io = struct.unpack_from("<12I", data)
    tens = [struct.unpack_from("<8I", data, to + 32 * i) for i in range(tensors)]
    op = [struct.unpack_from("<16I", data, oo + 64 * i) for i in range(ops)]
    return dict(size=size, arena=arena, tensors=tens, ops=op, inputs=inputs, outputs=outputs)


def _build(tmp, persistent):
    pool = hx.ConstantPool()
    dims = {0: (1, 1, 64), 1: (2, 1, 64), 2: (1, 1, 64), 3: (8, 1, 64), 4: (1, 1, 64)}
    w = pool.emit(np.ones((3, 64), np.float32))
    ops = [
        [hx.SHORTCONV, 0, 1, NONE, 2, 3, w, NONE, 0] + [0] * 7,
        [hx.KV_WRITE, 2, NONE, NONE, 3, 0] + [0] * 10,
        [hx.RMSNORM, 2, NONE, NONE, 4, 0, NONE, NONE] + [0] * 8,
    ]
    return hx.write(
        ops, [0], [4], lambda i: dims[i], lambda i: 64, pool, Path(tmp) / "g",
        {"source": "t", "source_sha256": "0" * 64, "ops": 3, "shapes": {i: list(d) for i, d in dims.items()}},
        persistent=persistent,
    )


class PersistentStateTest(unittest.TestCase):
    def test_persistent_tensors_get_state_offsets_not_arena(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _build(tmp, {1: 0x1001, 3: 0x1002})
            g = _unpack(Path(tmp) / "g")
        flags = {i: t[5] for i, t in enumerate(g["tensors"])}
        self.assertEqual(sorted(i for i, f in flags.items() if f), [1, 3])
        ring, cache = g["tensors"][1], g["tensors"][3]
        self.assertEqual((ring[0], ring[6]), (0, 0x1001))
        self.assertEqual(cache[0], 128 * 4 and 512)  # ring: 2 rows x 64 x 4 bytes, 128-aligned
        self.assertEqual(result["state_bytes"], 512 + 2048)
        self.assertEqual(result["persistent"]["3"]["offset"], 512)
        # The arena holds only x, the shortconv output and the norm output.
        for i, t in enumerate(g["tensors"]):
            if not t[5]:
                self.assertLess(t[0] + 256, g["arena"] + 1)
        self.assertLessEqual(g["arena"], 3 * 256)

    def test_without_persistent_everything_lives_in_the_arena(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = hx.write(
                [[hx.RMSNORM, 0, NONE, NONE, 1, 0, NONE, NONE] + [0] * 8], [0], [1],
                lambda i: (1, 1, 32), lambda i: 32, hx.ConstantPool(), Path(tmp) / "g",
                {"source": "t", "source_sha256": "0" * 64, "ops": 1, "shapes": {0: [1], 1: [1]}},
            )
        self.assertEqual(result["state_bytes"], 0)
        self.assertEqual(result["persistent"], {})

    def test_state_tensor_cannot_be_a_graph_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = hx.ConstantPool()
            with self.assertRaises(ValueError):
                hx.write([[hx.RMSNORM, 0, NONE, NONE, 1, 0, NONE, NONE] + [0] * 8], [0], [1],
                         lambda i: (1, 1, 32), lambda i: 32, pool, Path(tmp) / "g",
                         {"source": "t", "source_sha256": "0", "ops": 1, "shapes": {0: [1], 1: [1]}},
                         persistent={0: 1})

    def test_op_numbers_match_the_native_enum(self):
        header = (ROOT / "hexecutor/native/tensor/graph.h").read_text()
        for name, value in (("HX_ROPE", hx.ROPE), ("HX_SHORTCONV", hx.SHORTCONV),
                            ("HX_LINEAR_Q", hx.LINEAR_Q), ("HX_KV_WRITE", hx.KV_WRITE)):
            self.assertRegex(header, rf"{name}[^\n]*/\* {value}:")


class QuantTest(unittest.TestCase):
    def test_q4_layout_and_error(self):
        rng = np.random.default_rng(1)
        w = rng.normal(0, 1, (4, 64)).astype(np.float32)
        blob, deq = hx.quantize_q4_0(w)
        self.assertEqual(len(blob), 4 * 2 * 18)
        # Decode the bytes the way the native kernel does and require it to equal the dequant.
        raw = np.frombuffer(blob, np.uint8).reshape(4, 2, 18)
        scale = raw[..., 16:].copy().view(np.float16).astype(np.float32)[..., 0]
        lo, hi = (raw[..., :16] & 15).astype(np.int32) - 8, (raw[..., :16] >> 4).astype(np.int32) - 8
        q = np.stack([lo, hi], axis=-1).reshape(4, 2, 32)
        np.testing.assert_array_equal((q * scale[..., None]).reshape(4, 64), deq)
        self.assertLess(np.abs(deq - w).max(), np.abs(w).max() / 8 * 1.01)  # one step at most: the opposite extreme clips

    def test_q8_layout_and_error(self):
        rng = np.random.default_rng(2)
        w = rng.normal(0, 1, (3, 96)).astype(np.float32)
        blob, deq = hx.quantize_q8_0(w)
        self.assertEqual(len(blob), 3 * 3 * 34)
        raw = np.frombuffer(blob, np.uint8).reshape(3, 3, 34)
        scale = raw[..., 32:].copy().view(np.float16).astype(np.float32)[..., 0]
        q = raw[..., :32].view(np.int8).astype(np.float32)
        np.testing.assert_array_equal((q * scale[..., None]).reshape(3, 96), deq)
        self.assertLess(np.abs(deq - w).max(), 0.02)

    def test_all_zero_block_and_bad_k(self):
        blob, deq = hx.quantize_q4_0(np.zeros((1, 32), np.float32))
        self.assertTrue(np.all(deq == 0))
        with self.assertRaises(ValueError):
            hx.quantize_q8_0(np.zeros((1, 33), np.float32))


@unittest.skipUnless(
    Path(os.environ.get("HEXECUTOR_DECODE_RUN", ROOT / "build/bin/hexecutor_decode_run")).exists(),
    "needs the host runner (make check builds it)")
class StatefulRunnerTest(unittest.TestCase):
    """hexecutor_decode_run keeps the persistent state region between steps (CPU reference)."""

    def run_decode(self, tmp, steps, reset_every=None):
        sys.path.insert(0, str(ROOT / "hexecutor/examples/llm/lfm"))
        import gen_hxfg_decode_golden as gen

        out = Path(tmp)
        gen.main(out)
        runner = os.environ.get("HEXECUTOR_DECODE_RUN", str(ROOT / "build/bin/hexecutor_decode_run"))
        done = subprocess.run(
            [runner, out / "lfm_decode.hxfg", out / "lfm_decode.bin", str(steps), str(out / "got_"),
             out / "inputs_x.f32", out / "inputs_pos.f32", out / "inputs_mask.f32"],
            capture_output=True, text=True)
        return out, done

    def test_matches_the_golden_over_all_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, done = self.run_decode(tmp, 6)
            self.assertEqual(done.returncode, 0, done.stderr)
            got = np.fromfile(out / "got_0.f32", "<f4")
            want = np.fromfile(out / "expected.f32", "<f4")
        self.assertEqual(got.shape, want.shape)
        self.assertLess(float(np.abs(got - want).max()), 1e-5)
        self.assertIn("state_bytes 4608", done.stderr)

    def test_state_is_what_makes_later_steps_right(self):
        # Step 1 alone, with no step 0 before it, must differ: the ring and the cache carry history.
        with tempfile.TemporaryDirectory() as tmp:
            out, done = self.run_decode(tmp, 6)
            x = np.fromfile(out / "inputs_x.f32", "<f4").reshape(6, -1)
            pos = np.fromfile(out / "inputs_pos.f32", "<f4")
            mask = np.fromfile(out / "inputs_mask.f32", "<f4").reshape(6, -1)
            for name, data in (("x", x[1:2]), ("pos", pos[1:2]), ("mask", mask[1:2])):
                data.astype("<f4").tofile(out / f"one_{name}.f32")
            alone = subprocess.run(
                [os.environ.get("HEXECUTOR_DECODE_RUN", str(ROOT / "build/bin/hexecutor_decode_run")),
                 out / "lfm_decode.hxfg", out / "lfm_decode.bin", "1", str(out / "alone_"),
                 out / "one_x.f32", out / "one_pos.f32", out / "one_mask.f32"], capture_output=True, text=True)
            self.assertEqual(alone.returncode, 0, alone.stderr)
            after_history = np.fromfile(out / "got_0.f32", "<f4").reshape(6, -1)[1]
            fresh = np.fromfile(out / "alone_0.f32", "<f4")
        self.assertGreater(float(np.abs(after_history - fresh).max()), 1e-3)

    def test_input_files_must_hold_every_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.run_decode(tmp, 6)
            done = subprocess.run(
                [os.environ.get("HEXECUTOR_DECODE_RUN", str(ROOT / "build/bin/hexecutor_decode_run")),
                 out / "lfm_decode.hxfg", out / "lfm_decode.bin", "7", str(out / "x_"),
                 out / "inputs_x.f32", out / "inputs_pos.f32", out / "inputs_mask.f32"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("must hold STEPS inputs", done.stderr)


if __name__ == "__main__":
    unittest.main()
