#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build the resident CDSP image for qwen_dsp.c from the Q4_0 GGUF.

Only re-arranges bytes (plus a constant RoPE table); the Q4_0 codes and
scales are copied unchanged. Layout, from the start of cdsp_mem:

  0       header (Image in qwen_dsp.c), 4 KB
  small   RoPE [ctx][64][cos, sin] f32, output norm, per-layer norms
  ...     matrices in the decode layout (see qwen_dsp.c), 4 KB aligned
  kv      per layer: K^T then V, fp16 (see qwen_dsp.c), left zero (not stored)
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = (PATHS.repo / 'hexecutor/examples/llm')
MODEL_DEFAULT = Path.home() / "frame-chat/models/Qwen3-0.6B-Q4_0.gguf"
GGUF_PY_DEFAULT = Path.home() / "frame-chat/vendor/llama.cpp/gguf-py"
IMAGE_MAGIC = 0x50534451
VERSION = 2
LAYERS = 28
HEADER_WORDS = 16 + LAYERS * 8


def align(n, a=4096):
    return (n + a - 1) // a * a


def q4_blocks(tensor):
    """Raw Q4_0 rows [n][k/32][18] as uint8."""
    k, n = map(int, tensor.shape)
    return np.asarray(tensor.data, dtype=np.uint8).reshape(n, k // 32, 18), k, n


def pack_matrix(raw):
    """[n][k/32][18] Q4_0 -> decode layout bytes (see qwen_dsp.c)."""
    n, kb, _ = raw.shape
    g = n // 32
    qs = raw[..., 2:]
    q = np.concatenate((qs & 15, qs >> 4), axis=-1)               # [n][kb][32]
    q = q.reshape(g, 32, kb, 4, 2, 4)                              # g r kb j half i
    codes = (q[:, :, :, :, 0, :] | (q[:, :, :, :, 1, :] << 4))    # g r kb j i
    codes = codes.transpose(0, 2, 3, 1, 4).reshape(g, kb * 512)
    d = raw[..., :2].reshape(g, 32, kb // 2, 2, 2)                 # g r p half byte
    scales = d.transpose(0, 2, 1, 3, 4).reshape(g, (kb // 2) * 128)
    return np.concatenate((codes, scales), axis=1).reshape(-1)


def rope_table(ctx, theta=1.0e6, dim=128):
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles = np.arange(ctx, dtype=np.float64)[:, None] * inv[None, :]
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype("<f4")


def fingerprint(model, ctx):
    st = Path(model).stat()
    text = f"{Path(model).resolve()}:{st.st_size}:{st.st_mtime_ns}:{ctx}:{VERSION}"
    return hashlib.sha256(text.encode()).digest()[:16]


def build(model, gguf_python, ctx, output):
    sys.path.insert(0, str(gguf_python))
    import gguf
    reader = gguf.GGUFReader(model, mode="r")
    tensors = {t.name: t for t in reader.tensors}

    parts = []          # (offset, bytes-like)
    cursor = 4096

    def place(blob, a=128):
        nonlocal cursor
        cursor = align(cursor, a)
        off = cursor
        parts.append((off, blob))
        cursor += len(blob)
        return off

    small = align(cursor)
    cursor = small
    rope = place(rope_table(ctx).tobytes())
    out_norm = place(np.asarray(tensors["output_norm.weight"].data, "<f4").tobytes())
    norms = []
    for i in range(LAYERS):
        p = f"blk.{i}."
        norms.append([place(np.asarray(tensors[p + name].data, "<f4").tobytes())
                      for name in ("attn_norm.weight", "attn_q_norm.weight",
                                   "attn_k_norm.weight", "ffn_norm.weight")])
    small_bytes = align(cursor) - small

    def matrix(*names):
        raws = [q4_blocks(tensors[name]) for name in names]
        k = raws[0][1]
        if any(r[1] != k for r in raws):
            raise ValueError(f"{names}: mismatched K")
        raw = np.concatenate([r[0] for r in raws], axis=0)
        return place(pack_matrix(raw).tobytes(), 4096)

    out = matrix("output.weight")
    layers = []
    for i in range(LAYERS):
        p = f"blk.{i}."
        qkv = matrix(p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight")
        o = matrix(p + "attn_output.weight")
        gateup = matrix(p + "ffn_gate.weight", p + "ffn_up.weight")
        down = matrix(p + "ffn_down.weight")
        an, qn, kn, fn = norms[i]
        layers.append((qkv, o, gateup, down, an, qn, kn, fn))
        print(f"layer {i} packed", flush=True)
    kv = align(cursor)
    kv_stride = ctx * 2048 * 2
    total = kv + LAYERS * kv_stride

    header = np.zeros(HEADER_WORDS, "<u4")
    header[:11] = (IMAGE_MAGIC, VERSION, LAYERS, ctx, rope, out_norm, out, kv, kv_stride, small, small_bytes)
    header[11:15] = np.frombuffer(fingerprint(model, ctx), "<u4")
    header[16:] = np.asarray(layers, "<u4").reshape(-1)
    image_bytes = align(cursor)
    with open(output, "wb") as stream:
        stream.truncate(image_bytes)
        stream.seek(0)
        stream.write(header.tobytes())
        for off, blob in parts:
            stream.seek(off)
            stream.write(blob)
    meta = {"ctx": ctx, "image_bytes": image_bytes, "total_bytes": total, "kv": kv,
            "kv_layer_stride": kv_stride, "fingerprint": fingerprint(model, ctx).hex()}
    Path(str(output) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    p.add_argument("--gguf-python", type=Path, default=GGUF_PY_DEFAULT)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--output", type=Path, default=MODEL_DEFAULT.parent / "Qwen3-0.6B.dspq4")
    args = p.parse_args()
    build(args.model, args.gguf_python, args.ctx, args.output)


if __name__ == "__main__":
    main()
