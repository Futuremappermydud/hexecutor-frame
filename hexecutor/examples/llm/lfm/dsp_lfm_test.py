#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deterministic numpy reference and fixture generator for tiny synthetic LFM model.

Architecture:
- 2 Conv layers + 1 Attention layer
- Dimensions: D=32, FFN=64, head_dim=16, num_heads=2, num_kv_heads=2, conv_kernel=3
- Sandwich RMSNorm (operator_norm, post_operator_norm, ffn_norm, post_ffn_norm)
- ShortConv: in_proj (D->3D: B, C, x), u = B * x, conv1d_causal(u) over state ring, y = C * conv_out, out_proj
- Attention: q_proj, k_proj, v_proj, q_norm, k_norm, RoPE, QK^T / sqrt(d), Softmax, attn*V, gate_proj, gate * attn, o_proj
- SwiGLU MLP: w2(silu(w1(x)) * w3(x))
"""

import math
import numpy as np

def rmsnorm(x, gamma, eps=1e-5):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * gamma

def silu(x):
    return x / (1.0 + np.exp(-x))

def apply_rope(x, pos, head_dim=16, num_heads=2):
    # x: [num_heads, head_dim]
    half = head_dim // 2
    out = np.zeros_like(x)
    for h in range(num_heads):
        for j in range(half):
            theta = 10000.0 ** (-2.0 * j / head_dim)
            angle = pos * theta
            cos_v = math.cos(angle)
            sin_v = math.sin(angle)
            v0 = x[h, j]
            v1 = x[h, half + j]
            out[h, j] = v0 * cos_v - v1 * sin_v
            out[h, half + j] = v0 * sin_v + v1 * cos_v
    return out

class TinyLFM:
    def __init__(self, D=32, FFN=64, head_dim=16, num_heads=2, K=3, seed=42):
        self.D = D
        self.FFN = FFN
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.K = K
        self.S = K - 1
        rng = np.random.RandomState(seed)

        self.conv_states = [np.zeros((self.S, D), dtype=np.float32) for _ in range(2)]

        # Layer 0 (Conv)
        self.l0_op_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l0_in_proj = rng.uniform(-0.1, 0.1, (3 * D, D)).astype(np.float32)
        self.l0_conv_w = rng.uniform(-0.2, 0.2, (K, D)).astype(np.float32)
        self.l0_out_proj = rng.uniform(-0.1, 0.1, (D, D)).astype(np.float32)
        self.l0_post_op_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l0_ffn_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l0_w1 = rng.uniform(-0.1, 0.1, (FFN, D)).astype(np.float32)
        self.l0_w3 = rng.uniform(-0.1, 0.1, (FFN, D)).astype(np.float32)
        self.l0_w2 = rng.uniform(-0.1, 0.1, (D, FFN)).astype(np.float32)
        self.l0_post_ffn_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)

        # Layer 1 (Conv)
        self.l1_op_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l1_in_proj = rng.uniform(-0.1, 0.1, (3 * D, D)).astype(np.float32)
        self.l1_conv_w = rng.uniform(-0.2, 0.2, (K, D)).astype(np.float32)
        self.l1_out_proj = rng.uniform(-0.1, 0.1, (D, D)).astype(np.float32)
        self.l1_post_op_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l1_ffn_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l1_w1 = rng.uniform(-0.1, 0.1, (FFN, D)).astype(np.float32)
        self.l1_w3 = rng.uniform(-0.1, 0.1, (FFN, D)).astype(np.float32)
        self.l1_w2 = rng.uniform(-0.1, 0.1, (D, FFN)).astype(np.float32)
        self.l1_post_ffn_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)

        # Layer 2 (Attn)
        self.l2_op_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l2_q_proj = rng.uniform(-0.1, 0.1, (D, D)).astype(np.float32)
        self.l2_k_proj = rng.uniform(-0.1, 0.1, (D, D)).astype(np.float32)
        self.l2_v_proj = rng.uniform(-0.1, 0.1, (D, D)).astype(np.float32)
        self.l2_q_norm = rng.uniform(0.8, 1.2, (head_dim,)).astype(np.float32)
        self.l2_k_norm = rng.uniform(0.8, 1.2, (head_dim,)).astype(np.float32)
        self.l2_gate_proj = rng.uniform(-0.1, 0.1, (num_heads, D)).astype(np.float32)
        self.l2_o_proj = rng.uniform(-0.1, 0.1, (D, D)).astype(np.float32)
        self.l2_post_op_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l2_ffn_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)
        self.l2_w1 = rng.uniform(-0.1, 0.1, (FFN, D)).astype(np.float32)
        self.l2_w3 = rng.uniform(-0.1, 0.1, (FFN, D)).astype(np.float32)
        self.l2_w2 = rng.uniform(-0.1, 0.1, (D, FFN)).astype(np.float32)
        self.l2_post_ffn_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)

        # Final norm
        self.final_norm = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)

        # KV cache: list of (k, v) per position
        self.kv_cache = []

    def conv_block(self, x, l_idx, op_norm, in_proj, conv_w, out_proj, post_op_norm,
                   ffn_norm, w1, w3, w2, post_ffn_norm):
        res = x.copy()
        # Pre-op norm
        x_norm = rmsnorm(x, op_norm)
        # in_proj -> BCx
        bcx = in_proj @ x_norm
        B = bcx[:self.D]
        C = bcx[self.D:2*self.D]
        x_val = bcx[2*self.D:]
        u = B * x_val

        # Causal conv1d over state ring
        state = self.conv_states[l_idx]
        conv_out = np.zeros(self.D, dtype=np.float32)
        for d in range(self.D):
            acc = 0.0
            for j in range(self.S):
                acc += state[j, d] * conv_w[j, d]
            acc += u[d] * conv_w[self.S, d]
            conv_out[d] = acc
            # update state
            for j in range(self.S - 1):
                state[j, d] = state[j + 1, d]
            state[self.S - 1, d] = u[d]

        # Gate and out_proj
        y = C * conv_out
        op_out = out_proj @ y
        op_out = rmsnorm(op_out, post_op_norm)
        x = res + op_out

        # FFN
        res2 = x.copy()
        x_fn = rmsnorm(x, ffn_norm)
        ffn_out = w2 @ (silu(w1 @ x_fn) * (w3 @ x_fn))
        ffn_out = rmsnorm(ffn_out, post_ffn_norm)
        return res2 + ffn_out

    def attn_block(self, x, pos, op_norm, q_proj, k_proj, v_proj, q_norm, k_norm,
                   gate_proj, o_proj, post_op_norm, ffn_norm, w1, w3, w2, post_ffn_norm):
        res = x.copy()
        x_norm = rmsnorm(x, op_norm)
        q = (q_proj @ x_norm).reshape(self.num_heads, self.head_dim)
        k = (k_proj @ x_norm).reshape(self.num_heads, self.head_dim)
        v = (v_proj @ x_norm).reshape(self.num_heads, self.head_dim)

        # QK norm per head
        for h in range(self.num_heads):
            q[h] = rmsnorm(q[h], q_norm)
            k[h] = rmsnorm(k[h], k_norm)

        # RoPE
        q = apply_rope(q, pos, self.head_dim, self.num_heads)
        k = apply_rope(k, pos, self.head_dim, self.num_heads)

        # Cache append
        self.kv_cache.append((k, v))
        ctx_len = len(self.kv_cache)

        # Attention: compute scores for each head
        head_outputs = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        scale = 1.0 / math.sqrt(self.head_dim)
        for h in range(self.num_heads):
            scores = np.zeros(ctx_len, dtype=np.float32)
            for t in range(ctx_len):
                kt = self.kv_cache[t][0][h]
                scores[t] = np.dot(q[h], kt) * scale
            # softmax
            m = np.max(scores)
            exp_s = np.exp(scores - m)
            weights = exp_s / np.sum(exp_s)
            acc = np.zeros(self.head_dim, dtype=np.float32)
            for t in range(ctx_len):
                vt = self.kv_cache[t][1][h]
                acc += weights[t] * vt
            head_outputs[h] = acc

        # Head gating: sigmoid(gate_proj(x_norm)) per head
        gate = 1.0 / (1.0 + np.exp(-(gate_proj @ x_norm))) # [num_heads]
        for h in range(self.num_heads):
            head_outputs[h] *= gate[h]

        attn_flat = head_outputs.reshape(-1)
        op_out = o_proj @ attn_flat
        op_out = rmsnorm(op_out, post_op_norm)
        x = res + op_out

        # FFN
        res2 = x.copy()
        x_fn = rmsnorm(x, ffn_norm)
        ffn_out = w2 @ (silu(w1 @ x_fn) * (w3 @ x_fn))
        ffn_out = rmsnorm(ffn_out, post_ffn_norm)
        return res2 + ffn_out

    def step(self, x, pos):
        # Layer 0: Conv
        x = self.conv_block(x, 0, self.l0_op_norm, self.l0_in_proj, self.l0_conv_w,
                            self.l0_out_proj, self.l0_post_op_norm, self.l0_ffn_norm,
                            self.l0_w1, self.l0_w3, self.l0_w2, self.l0_post_ffn_norm)
        # Layer 1: Conv
        x = self.conv_block(x, 1, self.l1_op_norm, self.l1_in_proj, self.l1_conv_w,
                            self.l1_out_proj, self.l1_post_op_norm, self.l1_ffn_norm,
                            self.l1_w1, self.l1_w3, self.l1_w2, self.l1_post_ffn_norm)
        # Layer 2: Attn
        x = self.attn_block(x, pos, self.l2_op_norm, self.l2_q_proj, self.l2_k_proj,
                            self.l2_v_proj, self.l2_q_norm, self.l2_k_norm,
                            self.l2_gate_proj, self.l2_o_proj, self.l2_post_op_norm,
                            self.l2_ffn_norm, self.l2_w1, self.l2_w3, self.l2_w2,
                            self.l2_post_ffn_norm)
        # Final norm
        return rmsnorm(x, self.final_norm)

if __name__ == "__main__":
    model = TinyLFM(seed=123)
    rng = np.random.RandomState(456)
    for pos in range(5):
        tok = rng.uniform(-1.0, 1.0, (32,)).astype(np.float32)
        out = model.step(tok, pos)
        print(f"Step {pos}: norm={np.linalg.norm(out):.6f}, first 4={out[:4]}")
