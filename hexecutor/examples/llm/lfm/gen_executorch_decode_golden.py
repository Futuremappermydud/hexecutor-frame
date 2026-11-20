#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The decode graph of gen_hxfg_decode_golden.py as a PyTorch module, exported through ExecuTorch.

Same weights, same independent float64 NumPy golden, same simulator test; the HXFG file comes from the
ExecuTorch route: torch.export -> edge -> HexecutorPartitioner -> HexecutorBackend.preprocess
(offline/executorch/lowering.py), using the hexecutor::lfm_* custom operators (ops.py). The delegate
blob is captured while the program is lowered, and its image and constant pool are written out.
Run with the pinned venv (hexecutor/.work/executorch-venv).

Usage: gen_executorch_decode_golden.py OUT_DIR
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))
import gen_hxfg_decode_golden as base  # noqa: E402
from hexecutor.offline.executorch import backend, blob, export as et_export  # noqa: E402
from hexecutor.offline.executorch import ops  # noqa: E402,F401

D, DH, LMAX, ROT, THETA, K_TAPS = base.D, base.DH, base.LMAX, base.ROT, base.THETA, base.K_TAPS


def rope_tables(rows=64):
    ang = np.arange(rows)[:, None] * np.float64(np.float32(THETA)) ** (-2.0 * np.arange(ROT // 2) / ROT)[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class Decode(nn.Module):
    """Replays the recipe recorded by gen_hxfg_decode_golden.build() with torch operators."""

    def __init__(self, recipe, ids):
        super().__init__()
        self.recipe, self.ids = recipe, ids
        cos, sin = rope_tables()
        self.register_buffer("cos", torch.from_numpy(cos))
        self.register_buffer("sin", torch.from_numpy(sin))
        for n, item in enumerate(recipe):
            kind = item[0]
            if kind == "rms":
                self.register_buffer(f"gamma{n}", torch.from_numpy(np.asarray(item[3], np.float32)))
            elif kind == "linear":
                _, _, _, raw, fmt, k, nn_, bias, _act = item
                nb = k // 32
                size = 18 if fmt == 0 else 34
                self.register_buffer(f"blocks{n}", torch.from_numpy(
                    np.frombuffer(raw, np.uint8).reshape(nn_, nb, size).copy()))
                self.register_buffer(f"bias{n}", torch.from_numpy(np.asarray(bias, np.float32)))
            elif kind == "shortconv":
                self.register_buffer(f"cw{n}", torch.from_numpy(
                    np.ascontiguousarray(item[4].T.reshape(D, 1, K_TAPS), dtype=np.float32)))
                self.register_buffer(f"cb{n}", torch.from_numpy(np.asarray(item[5], np.float32)))

    def forward(self, x, pos, mask):
        v = {self.ids["x"]: x, self.ids["pos"]: pos, self.ids["mask"]: mask}
        for n, item in enumerate(self.recipe):
            kind, out = item[0], item[1]
            if kind == "rms":
                v[out] = F.rms_norm(v[item[2]], (D,), getattr(self, f"gamma{n}"), base.EPS)
            elif kind == "linear":
                y = torch.ops.hexecutor.lfm_linear_q(v[item[2]], getattr(self, f"blocks{n}"),
                                                     getattr(self, f"bias{n}"), item[4])
                v[out] = F.silu(y) if item[8] == 3 else y
            elif kind == "bin":
                a, b = v[item[2]], v[item[3]]
                v[out] = a + b if item[4] == base.hx.BIN_ADD else a * b
            elif kind == "shortconv":
                v[out] = torch.ops.hexecutor.lfm_shortconv(
                    v[item[2]], getattr(self, f"cw{n}"), getattr(self, f"cb{n}"), v[item[3]], "conv0", "none")
            elif kind == "rope":
                v[out] = torch.ops.hexecutor.lfm_rope(v[item[2]], v[item[3]], self.cos, self.sin, 1, ROT, False, 0)
            elif kind == "kvw":
                v[out] = torch.ops.hexecutor.lfm_kv_write(v[item[2]], v[item[3]], item[4], LMAX, 0)
            elif kind == "sdpa":
                q = v[item[2]].reshape(1, 1, 1, DH)
                v[out] = F.scaled_dot_product_attention(q, v[item[3]], v[item[4]], attn_mask=v[item[5]]).reshape(1, 1, DH)
            else:
                raise ValueError(kind)
        return v[self.ids["out"]]


def export_and_capture(module, example):
    """Lower through ExecuTorch; returns the (summary, image, constants) of the delegated partition."""
    captured = []
    original = backend.lower_program

    def capture(edge_program, destination):
        result = original(edge_program, destination)
        captured.append(result)
        return result

    backend.lower_program = capture
    try:
        program = et_export.export_program(module, example)
    finally:
        backend.lower_program = original
    if len(captured) != 1:
        raise RuntimeError(f"expected one delegated partition, got {len(captured)}")
    return captured[0], program


def main(out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = base.build()
    rng = np.random.default_rng(99)
    xs = rng.normal(0, 1, (base.STEPS, D)).astype(np.float32)
    expected, (ring, kc, vc) = base.golden(model, xs)
    module = Decode(model["recipe"], model["ids"]).eval()
    example = (torch.zeros(1, 1, D), torch.zeros(1, 1), torch.zeros(1, 1, 1, LMAX))
    (summary, image, constants), program = export_and_capture(module, example)
    (out_dir / "lfm_decode.hxfg").write_bytes(image)
    (out_dir / "lfm_decode.bin").write_bytes(constants)
    (out_dir / "lfm_decode.pte").write_bytes(program.buffer)
    result = {"arena_bytes": summary["arena_bytes"], "state_bytes": summary["state_bytes"],
              "constant_bytes": summary["constant_bytes"], "ops": summary["operations"]}
    base.finish(out_dir, "lfm_decode", result, xs, expected, ring, kc, vc)
    print("state names:", summary["state_names"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hexecutor/native/build/lfm/et_decode")
