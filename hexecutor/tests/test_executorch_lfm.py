# SPDX-License-Identifier: LGPL-3.0-or-later
"""ExecuTorch route for the LFM decode operators (offline tooling; needs the pinned executorch venv).

    hexecutor/.work/executorch-venv/bin/python -m unittest hexecutor.tests.test_executorch_lfm -v

The hexecutor::lfm_* operators must (1) behave eagerly exactly as the independent NumPy golden does,
(2) delegate whole into one HXFG partition with persistent state, and (3) refuse what the engine
cannot express. The lowered graph itself is run against the same golden by the simulator test
(build-scripts/test_lfm_sim.sh) and, on the host, by examples/llm/lfm/test_hxfg_decode_sim.c.
"""

import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from hexecutor.offline.executorch import lowering, ops
    from hexecutor.offline import hxfg_writer as hx
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "hexecutor/examples/llm/lfm"
NONE = 0xFFFFFFFF


def decode_module():
    sys.path.insert(0, str(EXAMPLES))
    import gen_executorch_decode_golden as gen

    model = gen.base.build()
    return gen, model, gen.Decode(model["recipe"], model["ids"]).eval()


class Single(nn.Module):
    """One custom operator on a [1, 1, 64] token, with its constants as buffers."""

    def __init__(self, kind, **kw):
        super().__init__()
        self.kind, self.kw = kind, kw
        rng = np.random.default_rng(5)
        self.register_buffer("w", torch.from_numpy(rng.normal(0, 0.3, (64, 1, 3)).astype(np.float32)))
        ang = np.arange(64)[:, None] * 10000.0 ** (-2.0 * np.arange(32) / 64)[None, :]
        self.register_buffer("cos", torch.from_numpy(np.cos(ang).astype(np.float32) * kw.get("cos_scale", 1.0)))
        self.register_buffer("sin", torch.from_numpy(np.sin(ang).astype(np.float32)))
        self.register_buffer("blocks", torch.zeros(kw.get("n", 32), 2, 18, dtype=torch.uint8))

    def forward(self, x):
        k = self.kw
        if self.kind == "rope":
            return torch.ops.hexecutor.lfm_rope(x, None, self.cos, self.sin, 1, 64, False, k.get("start", 0))
        if self.kind == "conv":
            return torch.ops.hexecutor.lfm_shortconv(x, self.w, None, None, k.get("state", "s"), "none")
        if self.kind == "linear":
            return torch.ops.hexecutor.lfm_linear_q(x, self.blocks, None, 0)
        cache = torch.ops.hexecutor.lfm_kv_write(x, None, "kv", 8, 0)
        return cache.sum(dim=2) if k.get("misuse") else cache


@unittest.skipUnless(HAVE_TORCH, "needs the pinned torch/executorch venv")
class EagerTests(unittest.TestCase):
    def test_eager_decode_matches_the_numpy_golden(self):
        gen, model, module = decode_module()
        rng = np.random.default_rng(99)
        xs = rng.normal(0, 1, (gen.base.STEPS, gen.D)).astype(np.float32)
        expected, _ = gen.base.golden(model, xs)
        ops.reset_state()
        got = []
        with torch.no_grad():
            for step, x in enumerate(xs):
                mask = torch.full((1, 1, 1, gen.LMAX), -1e9)
                mask[..., : step + 1] = 0
                got.append(module(torch.from_numpy(x).reshape(1, 1, -1), torch.full((1, 1), float(step)),
                                  mask).reshape(-1).numpy())
        ops.reset_state()
        np.testing.assert_allclose(np.array(got), expected, atol=2e-4)


@unittest.skipUnless(HAVE_TORCH, "needs the pinned torch/executorch venv")
class LoweringTests(unittest.TestCase):
    def test_export_delegates_everything_with_persistent_state(self):
        gen, model, module = decode_module()
        example = (torch.zeros(1, 1, gen.D), torch.zeros(1, 1), torch.zeros(1, 1, 1, gen.LMAX))
        (summary, image, constants), program = gen.export_and_capture(module, example)
        self.assertEqual(summary["state_bytes"], 4608)
        self.assertEqual(sorted(summary["state_names"].values()), ["conv0", "k_cache", "v_cache"])
        tensors, n_ops = struct.unpack_from("<2I", image, 12)
        records = [struct.unpack_from("<8I", image, 256 + 32 * i) for i in range(tensors)]
        persistent = {r[6]: r for r in records if r[5] & 1}
        self.assertEqual(
            set(persistent), {zlib.crc32(n.encode()) | 0x80000000 for n in ("conv0", "k_cache", "v_cache")})
        kinds = [struct.unpack_from("<16I", image, 256 + 32 * tensors + 64 * i)[0] for i in range(n_ops)]
        for kind in (hx.LINEAR_Q, hx.ROPE, hx.SHORTCONV, hx.KV_WRITE, hx.SDPA, hx.RMSNORM):
            self.assertIn(kind, kinds)
        # Nothing is left on the CPU: the program has one delegate call and no kernels of its own.
        self.assertEqual(len(program.executorch_program.execution_plan[0].chains[0].instructions), 1)
        self.assertEqual(len(program.buffer) > 0, True)

    def _kinds(self, module, shape=(1, 1, 64)):
        ep = torch.export.export(module.eval(), (torch.zeros(*shape),))
        out = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                try:
                    out.append(lowering.kind_of(node, ep))
                except lowering.Unsupported as error:
                    out.append(str(error))
        return out

    def test_supported_forms_are_selected(self):
        self.assertEqual(self._kinds(Single("rope")), ["lfm_rope"])
        self.assertEqual(self._kinds(Single("conv")), ["lfm_shortconv"])
        self.assertEqual(self._kinds(Single("linear")), ["lfm_linear_q"])

    def test_refusals(self):
        self.assertIn("one token per run", self._kinds(Single("conv"), (1, 2, 64))[0])
        self.assertIn("only be read by attention", self._kinds(Single("cache", misuse=True))[0])
        self.assertIn("multiples of 32", self._kinds(Single("linear", n=48))[0])
        self.assertIn("outside the cos/sin cache", self._kinds(Single("rope", start=64))[0])

    def test_modified_rotary_table_fails_closed_at_lowering(self):
        gen, model, module = decode_module()
        module.cos.mul_(1.05)
        example = (torch.zeros(1, 1, gen.D), torch.zeros(1, 1), torch.zeros(1, 1, 1, gen.LMAX))
        with self.assertRaisesRegex(Exception, "rotary table"):
            gen.export_and_capture(module, example)

    def test_runtime_position_must_be_float32(self):
        class IntPos(nn.Module):
            def __init__(self):
                super().__init__()
                ang = np.arange(64)[:, None] * 10000.0 ** (-2.0 * np.arange(32) / 64)[None, :]
                self.register_buffer("cos", torch.from_numpy(np.cos(ang).astype(np.float32)))
                self.register_buffer("sin", torch.from_numpy(np.sin(ang).astype(np.float32)))

            def forward(self, x, pos):
                return torch.ops.hexecutor.lfm_rope(x, pos, self.cos, self.sin, 1, 64, False, 0)

        ep = torch.export.export(IntPos().eval(), (torch.zeros(1, 1, 64), torch.zeros(1, 1, dtype=torch.int64)))
        node = next(n for n in ep.graph.nodes if n.op == "call_function")
        with self.assertRaises(lowering.Unsupported):
            lowering.kind_of(node, ep)

    def test_duplicate_state_names_are_rejected(self):
        class Twice(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.zeros(64, 1, 3))

            def forward(self, x):
                y = torch.ops.hexecutor.lfm_shortconv(x, self.w, None, None, "same", "none")
                return torch.ops.hexecutor.lfm_shortconv(y, self.w, None, None, "same", "none")

        from hexecutor.offline.executorch import export as et_export

        with self.assertRaises(Exception) as caught:
            et_export.export_program(Twice(), torch.zeros(1, 1, 64))
        self.assertIn("already used", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
