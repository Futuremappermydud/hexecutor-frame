#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build a tiny LFM-style decode graph through hxfg_writer and an independent NumPy golden.

The graph is two layers plus the final norm, one token per run:

  layer 0 (conv):  RMSNorm -> Q8 in-proj u, Q4 gate -> ShortConv(u) * gate -> Q4 out-proj -> +x
                   RMSNorm -> SwiGLU (Q4 silu gate, Q8 up, Mul, Q4 down) -> +x
  layer 1 (attn):  RMSNorm -> Q8 q, Q4 k, Q8 v -> RoPE(theta 1e6, half the head rotated) on q and k
                   -> KV_WRITE k, v into persistent caches -> SDPA over the caches with an additive
                   mask input -> Q4 out-proj -> +x, then the same SwiGLU block
  final RMSNorm

Inputs per run are the token embedding x, the position (float) and the additive key mask. The conv
rings and both KV caches are persistent tensors that the host keeps between runs. The golden is
float64 NumPy over the dequantized weights and shares no code with the engine.

Usage: gen_hxfg_decode_golden.py OUT_DIR   (writes OUT_DIR/lfm_decode.{hxfg,bin,json} and
OUT_DIR/hxfg_decode_data.h with the per-step inputs and expected outputs for the simulator test)
"""

import pathlib
import sys
import zlib

import numpy as np

# Import through the package: putting hexecutor/offline on the path would let its `executorch`
# subpackage shadow the real ExecuTorch.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from hexecutor.offline import hxfg_writer as hx  # noqa: E402

NONE = hx.NONE
D, F, DH, LMAX, K_TAPS, STEPS = 64, 128, 64, 8, 3, 6
ROT, THETA, EPS = 32, 1.0e6, 1e-5


def build(seed=7):
    rng = np.random.default_rng(seed)
    pool = hx.ConstantPool()
    dims, ops = {}, []
    persistent = {}
    counter = [0]

    def tensor(h, w, c, persist=False):
        i = counter[0]
        counter[0] += 1
        dims[i] = (h, w, c)
        if persist:
            persistent[i] = 0x2000 + i
        return i

    def op(kind, x, y, z, out, *p):
        assert len(p) <= 11
        ops.append([kind, x, y, z, out, *p, *([0] * (11 - len(p)))])
        return out

    def vec(n, lo=0.8, hi=1.2):
        return rng.uniform(lo, hi, n).astype(np.float32)

    def padded(a):
        a = np.asarray(a, np.float32)
        out = np.zeros(a.shape[:-1] + (((a.shape[-1] + 31) // 32) * 32,), np.float32)
        out[..., : a.shape[-1]] = a
        return out

    gold = {}
    recipe = []  # high-level nodes, replayed by gen_onnx_decode_golden.py through the ONNX lowering

    def rms(x, gamma, out_c):
        o = op(hx.RMSNORM, x, NONE, NONE, tensor(1, 1, out_c), hx.f32_bits(EPS), pool.emit(padded(gamma)), NONE)
        recipe.append(("rms", o, x, gamma))
        return o

    def linear(x, n, fmt, act=0):
        k = dims[x][2]
        w = rng.normal(0, 1.0 / np.sqrt(k), (n, k)).astype(np.float32)
        blob, deq = (hx.quantize_q4_0 if fmt == 0 else hx.quantize_q8_0)(w)
        bias = rng.normal(0, 0.05, n).astype(np.float32)
        o = op(
            hx.LINEAR_Q, x, NONE, NONE, tensor(1, 1, n), fmt,
            pool.emit(np.frombuffer(blob, np.uint8), dtype="u1"), pool.emit(padded(bias)), k, n, act,
        )
        gold[o] = (deq, bias, act)
        recipe.append(("linear", o, x, blob, fmt, k, n, bias, act))
        return o

    def binary(x, y, kind):
        o = op(hx.BINARY, x, y, NONE, tensor(1, 1, dims[x][2]), kind, 0, 0, 0)
        recipe.append(("bin", o, x, y, kind))
        return o

    x_in = tensor(1, 1, D)
    pos_in = tensor(1, 1, 1)
    mask_in = tensor(1, 1, LMAX)

    layers = []

    def swiglu(x, norm):
        n = rms(x, norm, D)
        a = linear(n, F, 0, act=3)
        b = linear(n, F, 1)
        h = binary(a, b, hx.BIN_MUL)
        d = linear(h, D, 0)
        return binary(x, d, hx.BIN_ADD), (n, a, b, h, d)

    # layer 0: gated short conv
    g_op, g_ffn = vec(D), vec(D)
    ring = tensor(K_TAPS - 1, 1, D, persist=True)
    conv_w = rng.normal(0, 0.4, (K_TAPS, D)).astype(np.float32)
    conv_b = rng.normal(0, 0.1, D).astype(np.float32)
    n = rms(x_in, g_op, D)
    u = linear(n, D, 1)
    gate = linear(n, D, 0)
    c = op(hx.SHORTCONV, u, ring, gate, tensor(1, 1, D), K_TAPS, pool.emit(padded(conv_w)), pool.emit(padded(conv_b)), 0)
    recipe.append(("shortconv", c, u, gate, conv_w, conv_b, ring))
    o = linear(c, D, 0)
    x1 = binary(x_in, o, hx.BIN_ADD)
    x2, _ = swiglu(x1, g_ffn)
    layers.append(("conv", g_op, g_ffn, conv_w, conv_b))

    # layer 1: attention with a persistent KV cache
    g_op2, g_ffn2 = vec(D), vec(D)
    kc = tensor(LMAX, 1, DH, persist=True)
    vc = tensor(LMAX, 1, DH, persist=True)
    n = rms(x2, g_op2, D)
    q = linear(n, DH, 1)
    k = linear(n, DH, 0)
    v = linear(n, DH, 1)
    qr = op(hx.ROPE, q, pos_in, NONE, tensor(1, 1, DH), DH, 1, 0, 0, hx.f32_bits(THETA), ROT, 0)
    kr = op(hx.ROPE, k, pos_in, NONE, tensor(1, 1, DH), DH, 1, 0, 0, hx.f32_bits(THETA), ROT, 0)
    op(hx.KV_WRITE, kr, pos_in, NONE, kc, 0)
    op(hx.KV_WRITE, v, pos_in, NONE, vc, 0)
    att = op(hx.SDPA, qr, kc, vc, tensor(1, 1, DH), mask_in, 0, 0, 1)
    recipe += [("rope", qr, q, pos_in), ("rope", kr, k, pos_in), ("kvw", kc, kr, pos_in, "k_cache"),
               ("kvw", vc, v, pos_in, "v_cache"), ("sdpa", att, qr, kc, vc, mask_in)]
    o = linear(att, D, 0)
    x3 = binary(x2, o, hx.BIN_ADD)
    x4, _ = swiglu(x3, g_ffn2)
    layers.append(("attn", g_op2, g_ffn2))

    g_final = vec(D)
    out = rms(x4, g_final, D)

    fixed = dict(
        ops=ops, inputs=[x_in, pos_in, mask_in], outputs=[out], dims=dims, persistent=persistent, pool=pool,
        gold=gold, recipe=recipe, layers=layers, g_final=g_final, ids=dict(x=x_in, pos=pos_in, mask=mask_in, out=out),
    )
    return fixed


def golden(model, xs):
    """Float64 decode steps over the dequantized weights; returns outputs and the state after each step."""
    ops, gold = model["ops"], model["gold"]
    l0, l1 = model["layers"]
    ring = np.zeros((K_TAPS - 1, D))
    kc = np.zeros((LMAX, DH))
    vc = np.zeros((LMAX, DH))
    outs = []

    def rmsn(x, g):
        return x / np.sqrt(np.mean(x * x) + EPS) * g

    def lin(op_out, x):
        w, b, act = gold[op_out]
        y = w.astype(np.float64) @ x + b
        if act == 3:
            y = y / (1 + np.exp(-y))
        return y

    linear_ops = [o[4] for o in ops if o[0] == hx.LINEAR_Q]
    it = iter(linear_ops)

    def rope(v, pos):
        out = v.copy()
        half = ROT // 2
        for j in range(half):
            ang = pos * THETA ** (-2.0 * j / ROT)
            c, s = np.cos(ang), np.sin(ang)
            out[j] = v[j] * c - v[half + j] * s
            out[half + j] = v[j] * s + v[half + j] * c
        return out

    for step, x in enumerate(xs):
        it = iter(linear_ops)
        x = x.astype(np.float64)
        # conv layer
        n = rmsn(x, l0[1])
        u = lin(next(it), n)
        gate = lin(next(it), n)
        acc = l0[4].astype(np.float64).copy()
        for j in range(K_TAPS - 1):
            acc += ring[j] * l0[3][j]
        acc += u * l0[3][K_TAPS - 1]
        ring = np.vstack([ring[1:], u[None]])
        x1 = x + lin(next(it), acc * gate)

        def ffn(x, g):
            n = rmsn(x, g)
            a = lin(next(it), n)
            b = lin(next(it), n)
            return x + lin(next(it), a * b)

        x2 = ffn(x1, l0[2])
        n = rmsn(x2, l1[1])
        q, k, v = lin(next(it), n), lin(next(it), n), lin(next(it), n)
        q, k = rope(q, step), rope(k, step)
        kc[step], vc[step] = k, v
        s = kc[: step + 1] @ q / np.sqrt(DH)
        p = np.exp(s - s.max())
        p /= p.sum()
        att = p @ vc[: step + 1]
        x3 = x2 + lin(next(it), att)
        x4 = ffn(x3, l1[2])
        outs.append(rmsn(x4, model["g_final"]))
    return np.array(outs), (ring, kc, vc)


def emit_header(path, arrays):
    lines = ["/* Generated by gen_hxfg_decode_golden.py; do not edit. */", "#include <stdint.h>"]
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            body = ",".join(str(int(v)) for v in arr.ravel())
            lines.append(f"static const uint8_t {name}[{arr.size}] __attribute__((aligned(128))) = {{{body}}};")
        else:
            body = ",".join(float(v).hex() if False else repr(float(np.float32(v))) + "f" for v in arr.ravel())
            lines.append(f"static const float {name}[{arr.size}] = {{{body}}};")
    path.write_text("\n".join(lines) + "\n")


def main(out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build()
    rng = np.random.default_rng(99)
    xs = rng.normal(0, 1, (STEPS, D)).astype(np.float32)
    expected, (ring, kc, vc) = golden(model, xs)

    dims = model["dims"]
    manifest = {
        "source": "gen_hxfg_decode_golden.py",
        "source_sha256": "0" * 64,
        "ops": len(model["ops"]),
        "shapes": {i: list(d) for i, d in dims.items()},
    }
    result = hx.write(
        model["ops"], model["inputs"], model["outputs"],
        lambda i: dims[i], lambda i: ((dims[i][2] + 31) // 32) * 32,
        model["pool"], out_dir / "lfm_decode", manifest, persistent=model["persistent"],
    )
    finish(out_dir, "lfm_decode", result, xs, expected, ring, kc, vc)


def finish(out_dir, stem, result, xs, expected, ring, kc, vc):
    """Write hxfg_decode_data.h: the image, constants, per-step inputs and the golden outputs."""
    out_dir = pathlib.Path(out_dir)
    masks = np.full((STEPS, LMAX), -1e9, np.float32)
    for s in range(STEPS):
        masks[s, : s + 1] = 0
    image = np.frombuffer((out_dir / f"{stem}.hxfg").read_bytes(), np.uint8)
    consts = np.frombuffer((out_dir / f"{stem}.bin").read_bytes(), np.uint8)
    # Per-step input files for hexecutor_decode_run (host or Frame), and the golden to compare with.
    (out_dir / "inputs_x.f32").write_bytes(xs.astype("<f4").tobytes())
    (out_dir / "inputs_pos.f32").write_bytes(np.arange(STEPS, dtype="<f4").tobytes())
    (out_dir / "inputs_pos.i32").write_bytes(np.arange(STEPS, dtype="<i4").tobytes())
    (out_dir / "inputs_mask.f32").write_bytes(masks.astype("<f4").tobytes())
    (out_dir / "expected.f32").write_bytes(np.asarray(expected, "<f4").tobytes())
    emit_header(
        out_dir / "hxfg_decode_data.h",
        {
            "decode_image": image, "decode_constants": consts,
            "decode_x": xs, "decode_mask": masks, "decode_expected": expected,
            "decode_ring_final": ring, "decode_k_final": kc, "decode_v_final": vc,
        },
    )
    with open(out_dir / "hxfg_decode_data.h", "a") as f:
        f.write(
            f"#define DECODE_STEPS {STEPS}\n#define DECODE_D {D}\n#define DECODE_LMAX {LMAX}\n"
            f"#define DECODE_DH {DH}\n#define DECODE_TAPS {K_TAPS}\n"
            f"#define DECODE_STATE_ID_RING {zlib.crc32(b'conv0') | 0x80000000}u\n"
            f"#define DECODE_STATE_ID_K {zlib.crc32(b'k_cache') | 0x80000000}u\n"
            f"#define DECODE_STATE_ID_V {zlib.crc32(b'v_cache') | 0x80000000}u\n"
            f"#define DECODE_ARENA_BYTES {result['arena_bytes']}u\n"
            f"#define DECODE_STATE_BYTES {result['state_bytes']}u\n"
        )
    print(f"wrote {out_dir}: {result['arena_bytes']} arena, {result['state_bytes']} state, "
          f"{result['constant_bytes']} constant bytes, {result['ops'] if isinstance(result['ops'], int) else len(result['ops'])} ops")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hexecutor/native/build/lfm/hxfg_decode")
