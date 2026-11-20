#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""LFM3-1.7B as ExecuTorch stage modules for the hexecutor tensor backend (one token per run).

Each Stage holds consecutive decoder layers with Q4_0 weights and the hexecutor::lfm_* operators, and is
exported through ExecuTorch -> HexecutorPartitioner -> HexecutorBackend into one HXFG graph. Embedding lookup,
lm_head (tied to the embedding) and sampling stay on the host. Stage inputs, in order:

    x [1, 1, 2048]       token activation
    pos [1, 1]           position of this token (float)
    mask_full  [1, 1, 1, LMAX]   additive key mask for full attention (0 for k <= pos, -1e9 after)
    mask_slide [1, 1, 1, LMAX]   the same restricted to the last `sliding_window` keys

Checkpoint conventions that differ from the engine's, and how they are absorbed offline:

* Grouped-query attention: each KV head gets its own persistent K and V cache and its own attention call over
  its four query heads (the engine's SDPA batches equal head counts only).
* Full-attention RoPE rotates dims (i, i + 64) for i < 32 with frequencies of the full 128-wide head; the engine
  rotates (j, j + R/2) inside the first R dims. The q/k projection rows, q_norm and k_norm are permuted
  (0..31, 64..95, 32..63, 96..127) so the same rotation is the engine's R = 64 split. Sliding layers rotate all
  128 dims and need no permutation. The score q.k and the norms are invariant under that shared permutation.
* LINEAR_Q needs N % 32 == 0: the 16-wide attention gate is a float linear instead.
"""

import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from hexecutor.offline import hxfg_writer as hx  # noqa: E402
from hexecutor.offline.executorch import ops  # noqa: E402,F401

D, FF, HEADS, KV_HEADS, HD = 2048, 7680, 16, 4, 128
EPS = 1e-5
TAPS = 7
GROUP = HEADS // KV_HEADS
FF_SPLIT = 2
FULL_THETA, SLIDE_THETA, WINDOW = 1.0e6, 1.0e4, 256
PERM = np.concatenate([np.arange(0, 32), np.arange(64, 96), np.arange(32, 64), np.arange(96, 128)])


class Checkpoint:
    def __init__(self, directory):
        from safetensors import safe_open

        self.dir = pathlib.Path(directory)
        self.f = safe_open(str(self.dir / "model.safetensors"), "pt")
        self.config = json.loads((self.dir / "config.json").read_text())

    def get(self, name):
        return self.f.get_tensor(name).float().numpy()

    @property
    def layer_types(self):
        return self.config["layer_types"]


def q4_blocks(w):
    """[N, K] float -> uint8 [N, K/32, 18] Q4_0 blocks as HX_LINEAR_Q reads them, and the dequantized matrix."""
    blob, deq = hx.quantize_q4_0(np.ascontiguousarray(w, dtype=np.float32))
    n, k = w.shape
    return torch.from_numpy(np.frombuffer(blob, np.uint8).reshape(n, k // 32, 18).copy()), deq


def q4_best_blocks(w, steps=33, spread=0.25):
    """Q4_0 with a per-block search of the scale (same bytes as hx.quantize_q4_0, lower error).

    The format is code = round(w / d) + 8 clipped to [0, 15] with a float16 d; the stock packer fixes d at
    (largest-magnitude value) / -8. Here d is chosen per block from a grid around that value, minimizing the
    squared reconstruction error."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    n, k = w.shape
    blocks = w.reshape(n, k // 32, 32)
    peak = np.take_along_axis(blocks, np.abs(blocks).argmax(axis=2)[..., None], axis=2)[..., 0]
    base = peak / -8.0
    best_err = np.full(base.shape, np.inf, np.float32)
    best_d = base.astype(np.float16)
    for f in np.linspace(1.0 - spread, 1.0 + spread, steps).astype(np.float32):
        d = (base * f).astype(np.float16)
        sf = d.astype(np.float32)
        inv = np.where(sf != 0, 1.0 / np.where(sf != 0, sf, 1.0), 0.0)
        q = np.clip(np.floor(blocks * inv[..., None] + 8.5), 0, 15)
        err = (((q - 8) * sf[..., None] - blocks) ** 2).sum(axis=2)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    sf = best_d.astype(np.float32)
    inv = np.where(sf != 0, 1.0 / np.where(sf != 0, sf, 1.0), 0.0)
    q = np.clip(np.floor(blocks * inv[..., None] + 8.5), 0, 15).astype(np.int32)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    out = np.concatenate([packed, best_d.astype(np.float16).view(np.uint8).reshape(n, k // 32, 2)], axis=2)
    deq = ((q - 8).astype(np.float32) * sf[..., None]).reshape(n, k)
    return torch.from_numpy(out.copy()), deq


def q8_blocks(w):
    blob, deq = hx.quantize_q8_0(np.ascontiguousarray(w, dtype=np.float32))
    n, k = w.shape
    return torch.from_numpy(np.frombuffer(blob, np.uint8).reshape(n, k // 32, 34).copy()), deq


def rope_tables(theta, rot, rows):
    half = rot // 2
    ang = np.arange(rows)[:, None] * np.float64(theta) ** (-2.0 * np.arange(half) / HD)[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class Stage(nn.Module):
    """Layers [lo, hi) of the decoder, plus the final embedding_norm when `final`."""

    def __init__(self, ckpt, lo, hi, lmax, final=False, quant=q4_blocks, log=print, exact=False, q8=(), q4_names=(), q8_names=()):
        super().__init__()
        self.exact = exact
        self.q4_names = set(q4_names)  # full buffer names (e.g. l7_w3) forced to Q4_0 even if their kind is Q8
        self.q8_names = set(q8_names)  # full buffer names forced to Q8_0
        self.q8 = set(q8)  # projection kinds kept at Q8_0 (e.g. q k v o in_b in_c in_x out w1 w2 w3)
        self.lo, self.hi, self.lmax, self.final = lo, hi, lmax, final
        self.kinds = ckpt.layer_types[lo:hi]
        self.deq = {}  # name -> dequantized float32 matrix, for the eager reference
        self.state_names = []
        spread = np.zeros((HEADS * HD, HEADS), np.float32)
        spread[np.arange(HEADS * HD), np.arange(HEADS * HD) // HD] = 1.0
        self.register_buffer("gate_spread", torch.from_numpy(spread))
        for li in range(lo, hi):
            p = f"model.layers.{li}."
            kind = ckpt.layer_types[li]
            self._norm(f"l{li}_opn", ckpt.get(p + "operator_norm.weight"))
            self._norm(f"l{li}_pon", ckpt.get(p + "post_operator_norm.weight"))
            self._norm(f"l{li}_ffn", ckpt.get(p + "ffn_norm.weight"))
            self._norm(f"l{li}_pfn", ckpt.get(p + "post_ffn_norm.weight"))
            if kind == "conv":
                w = ckpt.get(p + "conv.in_proj.weight")
                for i, part in enumerate("bcx"):
                    self._lin(f"l{li}_in_{part}", w[i * D:(i + 1) * D], quant)
                self._lin(f"l{li}_out", ckpt.get(p + "conv.out_proj.weight"), quant)
                cw = ckpt.get(p + "conv.conv.weight")
                self.register_buffer(f"l{li}_cw", torch.from_numpy(np.ascontiguousarray(cw, dtype=np.float32)))
                self.state_names.append(f"l{li}_conv")
            else:
                full = kind == "full_attention"
                perm_rows = lambda m: m.reshape(-1, HD, m.shape[-1])[:, PERM].reshape(-1, m.shape[-1]) if full else m
                qw = perm_rows(ckpt.get(p + "self_attn.q_proj.weight"))
                kw = perm_rows(ckpt.get(p + "self_attn.k_proj.weight"))
                vw = ckpt.get(p + "self_attn.v_proj.weight")
                for j in range(KV_HEADS):  # one projection per KV head: tensors are not sliced on the device
                    self._lin(f"l{li}_q{j}", qw[j * GROUP * HD:(j + 1) * GROUP * HD], quant, base=f"l{li}_q")
                    self._lin(f"l{li}_k{j}", kw[j * HD:(j + 1) * HD], quant, base=f"l{li}_k")
                    self._lin(f"l{li}_v{j}", vw[j * HD:(j + 1) * HD], quant, base=f"l{li}_v")
                self._lin(f"l{li}_o", ckpt.get(p + "self_attn.o_proj.weight"), quant)
                qn, kn = ckpt.get(p + "self_attn.q_norm.weight"), ckpt.get(p + "self_attn.k_norm.weight")
                self._norm(f"l{li}_qn", qn[PERM] if full else qn)
                self._norm(f"l{li}_kn", kn[PERM] if full else kn)
                self.register_buffer(f"l{li}_gate", torch.from_numpy(
                    np.ascontiguousarray(ckpt.get(p + "self_attn.gate_proj.weight"), dtype=np.float32)))
                rot, theta = (64, FULL_THETA) if full else (128, SLIDE_THETA)
                cos, sin = rope_tables(theta, rot, lmax)
                self.register_buffer(f"l{li}_cos", torch.from_numpy(cos))
                self.register_buffer(f"l{li}_sin", torch.from_numpy(sin))
                for j in range(KV_HEADS):
                    self.state_names += [f"l{li}_k{j}", f"l{li}_v{j}"]
            # Tensors are limited to 4096 channels: the 7680-wide FFN runs as two 3840-wide halves.
            w1, w3, w2 = (ckpt.get(p + f"feed_forward.w{i}.weight") for i in (1, 3, 2))
            for h in range(FF_SPLIT):
                rows = slice(h * FF // FF_SPLIT, (h + 1) * FF // FF_SPLIT)
                self._lin(f"l{li}_w1_{h}", w1[rows], quant, base=f"l{li}_w1")
                self._lin(f"l{li}_w3_{h}", w3[rows], quant, base=f"l{li}_w3")
                self._lin(f"l{li}_w2_{h}", w2[:, rows], quant, base=f"l{li}_w2")
            log(f"  layer {li} ({kind}) quantized")
        if final:
            self._norm("final_norm", ckpt.get("model.embedding_norm.weight"))

    def _norm(self, name, w):
        self.register_buffer(name, torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))

    def _lin(self, name, w, quant, base=None):
        base = base or name
        if self.exact:  # float weights, to separate quantization error from structure errors
            self.register_buffer(name, torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
            return
        blocks, deq = (q8_blocks if base in self.q8_names or (base.split("_", 1)[1] in self.q8 and base not in self.q4_names) else quant)(w)
        self.register_buffer(name, blocks)
        self.deq[name] = deq

    def rms(self, x, name):
        return F.rms_norm(x, (x.shape[-1],), getattr(self, name), EPS)

    def lin(self, x, name):
        if self.exact:
            return F.linear(x, getattr(self, name))
        blocks = getattr(self, name)
        return torch.ops.hexecutor.lfm_linear_q(x, blocks, None, 1 if blocks.shape[-1] == 34 else 0)

    def forward(self, x, pos, mask_full, mask_slide):
        for li, kind in zip(range(self.lo, self.hi), self.kinds):
            n = self.rms(x, f"l{li}_opn")
            probe = os.environ.get("LFM_PROBE")
            if probe == "n":
                return n
            if kind == "conv":
                b = self.lin(n, f"l{li}_in_b")
                c = self.lin(n, f"l{li}_in_c")
                v = self.lin(n, f"l{li}_in_x")
                y = torch.ops.hexecutor.lfm_shortconv(b * v, getattr(self, f"l{li}_cw"), None, c, f"l{li}_conv", "none")
                o = self.lin(y, f"l{li}_out")
            else:
                full = kind == "full_attention"
                rot = 64 if full else 128
                cos, sin = getattr(self, f"l{li}_cos"), getattr(self, f"l{li}_sin")
                mask = mask_full if full else mask_slide
                parts = []
                probes = {"q": [], "k": [], "v": [], "qn": [], "q0": [], "rt": [], "qnorm_only": []}
                for j in range(KV_HEADS):
                    k = self.rms(self.lin(n, f"l{li}_k{j}"), f"l{li}_kn")
                    k = torch.ops.hexecutor.lfm_rope(k, pos, cos, sin, 1, rot, False, 0)
                    v = self.lin(n, f"l{li}_v{j}")
                    q0 = self.lin(n, f"l{li}_q{j}")
                    q = self.rms(q0.reshape(1, GROUP, HD), f"l{li}_qn").reshape(1, 1, GROUP * HD)
                    probes["qn"].append(q.reshape(1, GROUP * HD)); probes["rt"].append(q0.reshape(1, GROUP, HD).reshape(1, GROUP * HD)); probes["qnorm_only"].append(self.rms(q0.reshape(1, GROUP, HD), f"l{li}_qn").reshape(1, GROUP * HD)); probes["q0"].append(q0.reshape(1, GROUP * HD))
                    q = torch.ops.hexecutor.lfm_rope(q, pos, cos, sin, GROUP, rot, False, 0)
                    probes["q"].append(q.reshape(1, GROUP * HD)); probes["k"].append(k.reshape(1, HD)); probes["v"].append(v.reshape(1, HD))
                    kc = torch.ops.hexecutor.lfm_kv_write(k, pos, f"l{li}_k{j}", self.lmax, 0)
                    vc = torch.ops.hexecutor.lfm_kv_write(v, pos, f"l{li}_v{j}", self.lmax, 0)
                    a = F.scaled_dot_product_attention(q.reshape(1, 1, GROUP, HD), kc, vc, attn_mask=mask)
                    parts.append(a.reshape(1, GROUP * HD))
                # Concatenation takes two rank-2 inputs and a rank-2 linear is the float matmul: join the
                # heads pairwise and spread each head's gate over its 128 channels with a constant matrix.
                if probe in probes:
                    pp = probes[probe]
                    c = torch.cat([torch.cat(pp[:2], dim=1), torch.cat(pp[2:], dim=1)], dim=1)
                    while c.shape[1] < HEADS * HD:
                        c = torch.cat([c, c], dim=1)
                    return c.reshape(1, 1, HEADS * HD)
                att = torch.cat([torch.cat(parts[:2], dim=1), torch.cat(parts[2:], dim=1)], dim=1)
                if probe == "att":
                    return att.reshape(1, 1, HEADS * HD)
                g = torch.sigmoid(F.linear(n.reshape(1, D), getattr(self, f"l{li}_gate")))
                if not os.environ.get("LFM_NOGATE"):
                    att = att * F.linear(g, getattr(self, "gate_spread"))
                o = self.lin(att.reshape(1, 1, HEADS * HD), f"l{li}_o")
                if probe == "o":
                    return o
            x = x + self.rms(o, f"l{li}_pon")
            n = self.rms(x, f"l{li}_ffn")
            d = None
            for h in range(FF_SPLIT):
                g = F.silu(self.lin(n, f"l{li}_w1_{h}")) * self.lin(n, f"l{li}_w3_{h}")
                part = self.lin(g, f"l{li}_w2_{h}")
                d = part if d is None else d + part
            x = x + self.rms(d, f"l{li}_pfn")
        if self.final:
            x = self.rms(x, "final_norm")
        return x


def masks(pos, lmax):
    full = np.full(lmax, -1e9, np.float32)
    full[: pos + 1] = 0
    slide = np.full(lmax, -1e9, np.float32)
    slide[max(0, pos - WINDOW + 1): pos + 1] = 0
    return full.reshape(1, 1, 1, lmax), slide.reshape(1, 1, 1, lmax)


def stage_ranges(n_stages=4, layers=24):
    per = layers // n_stages
    return [(i * per, (i + 1) * per) for i in range(n_stages)]
