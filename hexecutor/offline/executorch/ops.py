# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch custom operators for LFM-style decode graphs, lowered by the hexecutor delegate.

Importing this module registers `hexecutor::lfm_linear_q`, `lfm_rope`, `lfm_shortconv` and
`lfm_kv_write`. Each has an eager implementation (so a model runs and can be compared on the CPU) and a
fake implementation (so `torch.export` can trace it). The delegate lowers them to HX_LINEAR_Q, HX_ROPE,
HX_SHORTCONV and HX_KV_WRITE (lowering.py).

State: `lfm_shortconv` and `lfm_kv_write` are stateful by name. On the device the ring and the cache are
persistent tensors in the graph's state region, owned by the host and kept between runs. Eagerly the same
state lives in `STATE`, a dict keyed by the `state` argument; `reset_state()` clears it. Exported programs
never touch it. A position that varies per run is a float32 `[1, 1]` tensor (the delegate's inputs are
float32); `lfm_rope` and `lfm_kv_write` take it as `pos`, or a static `start` when `pos` is None.
"""

from typing import Optional

import numpy as np
import torch

STATE = {}


def reset_state():
    STATE.clear()


def _dequant(blocks, fmt):
    """Dequantised float32 matrix [N, K] of GGML Q4_0 / Q8_0 rows ([N, K/32, 18 or 34] uint8)."""
    raw = blocks.detach().cpu().numpy()
    n, nb, size = raw.shape
    if fmt == 0:
        if size != 18:
            raise ValueError("Q4_0 blocks are 18 bytes")
        lo = (raw[..., :16] & 15).astype(np.int32) - 8
        hi = (raw[..., :16] >> 4).astype(np.int32) - 8
        q = np.stack([lo, hi], axis=-1).reshape(n, nb, 32)
    elif fmt == 1:
        if size != 34:
            raise ValueError("Q8_0 blocks are 34 bytes")
        q = raw[..., :32].copy().view(np.int8).astype(np.int32)
    else:
        raise ValueError("fmt must be 0 (Q4_0) or 1 (Q8_0)")
    scale = np.ascontiguousarray(raw[..., -2:]).view(np.float16).astype(np.float32).reshape(n, nb, 1)
    return (q * scale).reshape(n, nb * 32).astype(np.float32)


@torch.library.custom_op("hexecutor::lfm_linear_q", mutates_args=())
def lfm_linear_q(x: torch.Tensor, blocks: torch.Tensor, bias: Optional[torch.Tensor], fmt: int) -> torch.Tensor:
    w = torch.from_numpy(_dequant(blocks, fmt))
    y = x.double() @ w.double().T
    if bias is not None:
        y = y + bias.double()
    return y.to(x.dtype)


@lfm_linear_q.register_fake
def _(x, blocks, bias, fmt):
    return x.new_empty(list(x.shape[:-1]) + [blocks.shape[0]])


@torch.library.custom_op("hexecutor::lfm_rope", mutates_args=())
def lfm_rope(x: torch.Tensor, pos: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor,
             num_heads: int, rotary_dim: int, interleaved: bool, start: int) -> torch.Tensor:
    """Rotary embedding of x [1, L, num_heads * head_dim] at positions start + row (or pos + row)."""
    length, width = x.shape[1], x.shape[2]
    head_dim = width // num_heads
    rot = rotary_dim or head_dim
    base = int(pos.reshape(-1)[0].item()) if pos is not None else start
    out = x.double().clone()
    half = rot // 2
    for row in range(length):
        c, s = cos[base + row].double(), sin[base + row].double()
        for h in range(num_heads):
            v = x[0, row, h * head_dim: h * head_dim + rot].double()
            if interleaved:
                a, b = v[0::2], v[1::2]
                new = torch.empty_like(v)
                new[0::2], new[1::2] = a * c - b * s, a * s + b * c
            else:
                a, b = v[:half], v[half:]
                new = torch.cat([a * c - b * s, a * s + b * c])
            out[0, row, h * head_dim: h * head_dim + rot] = new
    return out.to(x.dtype)


@lfm_rope.register_fake
def _(x, pos, cos, sin, num_heads, rotary_dim, interleaved, start):
    return torch.empty_like(x)


_ACTIVATIONS = {"none": lambda v: v, "relu": torch.relu, "silu": torch.nn.functional.silu}


@torch.library.custom_op("hexecutor::lfm_shortconv", mutates_args=())
def lfm_shortconv(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                  gate: Optional[torch.Tensor], state: str, activation: str) -> torch.Tensor:
    """One causal depthwise-convolution step: x [1, 1, C], weight [C, 1, k]; the last k - 1 inputs
    persist under `state`. Output = act(conv([ring, x]) + bias) * gate."""
    c, taps = weight.shape[0], weight.shape[2]
    ring = STATE.setdefault(state, torch.zeros(taps - 1, c, dtype=torch.float64))
    window = torch.cat([ring, x.reshape(1, c).double()], dim=0)  # oldest first
    y = (window * weight.reshape(c, taps).T.double()).sum(0)
    if bias is not None:
        y = y + bias.double()
    STATE[state] = window[1:].clone()
    y = _ACTIVATIONS[activation](y)
    if gate is not None:
        y = y * gate.reshape(c).double()
    return y.reshape(x.shape).to(x.dtype)


@lfm_shortconv.register_fake
def _(x, weight, bias, gate, state, activation):
    return torch.empty_like(x)


@torch.library.custom_op("hexecutor::lfm_kv_write", mutates_args=())
def lfm_kv_write(x: torch.Tensor, pos: Optional[torch.Tensor], state: str, max_len: int, start: int) -> torch.Tensor:
    """Append x [1, L, C] at row start (or pos); returns the whole cache as [1, 1, max_len, C]
    (batch and head dimensions of one, the layout scaled_dot_product_attention takes)."""
    length, c = x.shape[1], x.shape[2]
    cache = STATE.setdefault(state, torch.zeros(max_len, c, dtype=torch.float64))
    at = int(pos.reshape(-1)[0].item()) if pos is not None else start
    cache[at: at + length] = x[0].double()
    return cache.clone().reshape(1, 1, max_len, c).to(x.dtype)


@lfm_kv_write.register_fake
def _(x, pos, state, max_len, start):
    return x.new_empty([1, 1, max_len, x.shape[2]])
