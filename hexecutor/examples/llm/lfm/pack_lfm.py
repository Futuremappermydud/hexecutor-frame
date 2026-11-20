#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build resident CDSP images and HXFG constant pools for LFM3 models.

Converts Hugging Face checkpoints (LiquidAI/LFM3-1.7B and LiquidAI/LFM3-VL-2B)
into:
1. Unified 'Packed layout v1' binary images (.dspbin) with heterogeneous layer
   descriptors (causal short-conv vs full/sliding attention, Sandwich RMSNorms,
   width-7 depthwise taps, gating projections), supporting FP16, INT8, and Q4_0.
2. HXFG-compatible partitioned constant pools (.bin) and manifest metadata
   for 4-stage CDSP broker graphs under hexecutord.

Usage:
  # Pack deterministic test/dummy model to FP16 and Q4_0:
  python3 pack_lfm.py --dummy --quant fp16 --output /path/to/lfm3_fp16.dspbin
  python3 pack_lfm.py --dummy --quant q4 --output /path/to/lfm3_q4.dspbin

  # Pack real Hugging Face checkpoint:
  python3 pack_lfm.py --model /path/to/model.safetensors --quant q4 --output lfm3.dspbin
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Header and Format Constants
LFM_MAGIC = 0x314D464C  # "LFM1"
LFM_VERSION = 1

QUANT_FP16 = 0
QUANT_INT8 = 1
QUANT_Q4_0 = 2

LAYER_CONV = 0
LAYER_FULL_ATTENTION = 1
LAYER_SLIDING_ATTENTION = 2

# LFM3-1.7B / LFM3-VL-2B Standard Dimensions
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 7680
NUM_HIDDEN_LAYERS = 24
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
VOCAB_SIZE = 128000
CONV_KERNEL_SIZE = 7
SLIDING_WINDOW = 256
NORM_EPS = 1e-5
ROPE_FULL_THETA = 1000000.0
ROPE_SLIDING_THETA = 10000.0
PARTIAL_ROTARY_DIM = 64  # 0.5 * head_dim

# 6-layer macro-block repeating pattern
MACRO_BLOCK = [
    LAYER_CONV,
    LAYER_CONV,
    LAYER_FULL_ATTENTION,
    LAYER_CONV,
    LAYER_SLIDING_ATTENTION,
    LAYER_CONV,
]
LAYER_TYPES = MACRO_BLOCK * 4  # 24 layers


def align(n: int, a: int = 4096) -> int:
    """Align byte offset to boundary a."""
    return (n + a - 1) // a * a


def quantize_ggml_q4_0(matrix: np.ndarray) -> np.ndarray:
    """Quantize float matrix [N, K] into raw GGML Q4_0 blocks [N, K/32, 18].

    Each 32-weight block has:
      - 2 bytes IEEE float16 scale factor d
      - 16 bytes containing 32 4-bit unsigned nibbles (offset by +8).
        Low nibble holds elements 0..15, high nibble holds elements 16..31.
    """
    n, k = matrix.shape
    assert k % 32 == 0, f"K ({k}) must be a multiple of 32"
    x_blk = matrix.reshape(n, k // 32, 32).astype(np.float32)

    amax = np.max(np.abs(x_blk), axis=-1, keepdims=True)
    scale = (amax / -8.0).astype(np.float16)
    scale_f32 = scale.astype(np.float32)

    inv_scale = np.where(scale_f32 != 0, 1.0 / scale_f32, 0.0)
    q = np.round(x_blk * inv_scale) + 8
    q = np.clip(q, 0, 15).astype(np.uint8)

    q_low = q[..., :16]
    q_high = q[..., 16:]
    qs = (q_low | (q_high << 4)).astype(np.uint8)

    d_bytes = scale.view(np.uint8).reshape(n, k // 32, 2)
    raw = np.concatenate((d_bytes, qs), axis=-1)
    return raw


def dequantize_ggml_q4_0(raw: np.ndarray) -> np.ndarray:
    """Dequantize raw GGML Q4_0 blocks [N, K/32, 18] back to float32 [N, K]."""
    n, kb, _ = raw.shape
    d_bytes = raw[..., :2]
    qs = raw[..., 2:]
    scale = d_bytes.view(np.float16).astype(np.float32)

    q_low = qs & 0x0F
    q_high = qs >> 4
    q = np.concatenate((q_low, q_high), axis=-1)
    x = (q.astype(np.float32) - 8.0) * scale
    return x.reshape(n, kb * 32)


class LfmModelPacker:
    """Packs LFM3 weights into Packed Layout v1 and HXFG constant pools."""

    def __init__(
        self,
        quant_type: str = "fp16",
        num_layers: int = NUM_HIDDEN_LAYERS,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_size: int = INTERMEDIATE_SIZE,
        vocab_size: int = VOCAB_SIZE,
    ):
        self.quant_type_name = quant_type.lower()
        if self.quant_type_name == "fp16":
            self.quant_type = QUANT_FP16
        elif self.quant_type_name == "int8":
            self.quant_type = QUANT_INT8
        elif self.quant_type_name in ("q4", "q4_0"):
            self.quant_type = QUANT_Q4_0
        else:
            raise ValueError(f"Unknown quant type: {quant_type}")

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.layer_types = LAYER_TYPES[:num_layers]

    def _pack_matrix_blob(self, matrix: np.ndarray) -> bytes:
        """Packs a 2D matrix into bytes according to target quantization."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if self.quant_type == QUANT_FP16:
            return matrix.astype(np.float16).tobytes()
        elif self.quant_type == QUANT_INT8:
            # INT8 per-row quantization
            n, k = matrix.shape
            scale = (np.max(np.abs(matrix), axis=-1, keepdims=True) / 127.0).astype(np.float16)
            scale_f32 = scale.astype(np.float32)
            inv_scale = np.where(scale_f32 != 0, 1.0 / scale_f32, 0.0)
            q = np.clip(np.round(matrix * inv_scale), -128, 127).astype(np.int8)
            return q.tobytes() + scale.tobytes()
        elif self.quant_type == QUANT_Q4_0:
            raw = quantize_ggml_q4_0(matrix)
            return raw.tobytes()
        raise ValueError(f"Unsupported quant type {self.quant_type}")

    def build_dummy_weights(self) -> Dict[str, np.ndarray]:
        """Build deterministic synthetic weights for offline verification."""
        rng = np.random.default_rng(2026)
        weights = {}

        # Embeddings & final norm
        weights["model.embed_tokens.weight"] = (
            rng.normal(0, 0.02, (self.vocab_size, self.hidden_size)).astype(np.float32)
        )
        weights["model.embedding_norm.weight"] = np.ones(self.hidden_size, dtype=np.float32)

        for i, ltype in enumerate(self.layer_types):
            p = f"model.layers.{i}."
            # Sandwich norms
            weights[p + "operator_norm.weight"] = np.ones(self.hidden_size, dtype=np.float32)
            weights[p + "post_operator_norm.weight"] = np.ones(self.hidden_size, dtype=np.float32)
            weights[p + "ffn_norm.weight"] = np.ones(self.hidden_size, dtype=np.float32)
            weights[p + "post_ffn_norm.weight"] = np.ones(self.hidden_size, dtype=np.float32)

            # SwiGLU FFN
            weights[p + "feed_forward.w1.weight"] = (
                rng.normal(0, 0.02, (self.intermediate_size, self.hidden_size)).astype(np.float32)
            )
            weights[p + "feed_forward.w3.weight"] = (
                rng.normal(0, 0.02, (self.intermediate_size, self.hidden_size)).astype(np.float32)
            )
            weights[p + "feed_forward.w2.weight"] = (
                rng.normal(0, 0.02, (self.hidden_size, self.intermediate_size)).astype(np.float32)
            )

            if ltype == LAYER_CONV:
                weights[p + "conv.in_proj.weight"] = (
                    rng.normal(0, 0.02, (3 * self.hidden_size, self.hidden_size)).astype(np.float32)
                )
                weights[p + "conv.conv.weight"] = (
                    rng.normal(0, 0.05, (self.hidden_size, 1, CONV_KERNEL_SIZE)).astype(np.float32)
                )
                weights[p + "conv.out_proj.weight"] = (
                    rng.normal(0, 0.02, (self.hidden_size, self.hidden_size)).astype(np.float32)
                )
            else:
                # Attention
                weights[p + "self_attn.q_proj.weight"] = (
                    rng.normal(0, 0.02, (self.hidden_size, self.hidden_size)).astype(np.float32)
                )
                weights[p + "self_attn.k_proj.weight"] = (
                    rng.normal(0, 0.02, (NUM_KV_HEADS * HEAD_DIM, self.hidden_size)).astype(np.float32)
                )
                weights[p + "self_attn.v_proj.weight"] = (
                    rng.normal(0, 0.02, (NUM_KV_HEADS * HEAD_DIM, self.hidden_size)).astype(np.float32)
                )
                weights[p + "self_attn.o_proj.weight"] = (
                    rng.normal(0, 0.02, (self.hidden_size, self.hidden_size)).astype(np.float32)
                )
                weights[p + "self_attn.gate_proj.weight"] = (
                    rng.normal(0, 0.02, (NUM_ATTENTION_HEADS, self.hidden_size)).astype(np.float32)
                )
                weights[p + "self_attn.q_norm.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
                weights[p + "self_attn.k_norm.weight"] = np.ones(HEAD_DIM, dtype=np.float32)

        return weights

    def pack(self, weights: Dict[str, np.ndarray], output_path: Union[str, Path], include_embeds: bool = True) -> Dict:
        """Pack weights dictionary into Packed Layout v1 binary file."""
        output_path = Path(output_path)
        parts = []  # list of (offset, bytes_blob)
        cursor = 256 + self.num_layers * 64  # Start after header + layer table

        def place(blob: bytes, alignment: int = 4096) -> int:
            nonlocal cursor
            cursor = align(cursor, alignment)
            off = cursor
            parts.append((off, blob))
            cursor += len(blob)
            return off

        # Standardize weight tensor name prefix: accept model.layers or model.language_model.layers
        sample_key = next(iter(weights.keys()))
        prefix = "model.language_model." if "model.language_model." in sample_key else "model."

        # Global final norm
        emb_norm_w = np.asarray(weights[prefix + "embedding_norm.weight"], dtype=np.float32)
        emb_norm_off = place(emb_norm_w.tobytes(), alignment=128)

        # Token embeddings (if stored)
        embed_tokens_off = 0
        embed_key = prefix + "embed_tokens.weight"
        if include_embeds and embed_key in weights:
            embed_tokens_w = np.asarray(weights[embed_key], dtype=np.float32)
            embed_tokens_off = place(self._pack_matrix_blob(embed_tokens_w), alignment=4096)

        # Per-layer descriptors and tensor placement
        layer_descs = []
        for i in range(self.num_layers):
            ltype = self.layer_types[i]
            lp = f"{prefix}layers.{i}."

            op_norm = place(np.asarray(weights[lp + "operator_norm.weight"], dtype=np.float32).tobytes(), 128)
            post_op_norm = place(np.asarray(weights[lp + "post_operator_norm.weight"], dtype=np.float32).tobytes(), 128)
            ffn_norm = place(np.asarray(weights[lp + "ffn_norm.weight"], dtype=np.float32).tobytes(), 128)
            post_ffn_norm = place(np.asarray(weights[lp + "post_ffn_norm.weight"], dtype=np.float32).tobytes(), 128)

            w1_off = place(self._pack_matrix_blob(weights[lp + "feed_forward.w1.weight"]), 4096)
            w3_off = place(self._pack_matrix_blob(weights[lp + "feed_forward.w3.weight"]), 4096)
            w2_off = place(self._pack_matrix_blob(weights[lp + "feed_forward.w2.weight"]), 4096)

            desc = [ltype, op_norm, post_op_norm, ffn_norm, post_ffn_norm]

            if ltype == LAYER_CONV:
                in_p = place(self._pack_matrix_blob(weights[lp + "conv.in_proj.weight"]), 4096)
                conv_t = np.asarray(weights[lp + "conv.conv.weight"], dtype=np.float32).reshape(self.hidden_size, CONV_KERNEL_SIZE)
                conv_t_off = place(conv_t.tobytes(), 128)
                out_p = place(self._pack_matrix_blob(weights[lp + "conv.out_proj.weight"]), 4096)
                desc.extend([in_p, conv_t_off, out_p, 0, 0, 0, 0, 0])
            else:
                # Attention
                qp = place(self._pack_matrix_blob(weights[lp + "self_attn.q_proj.weight"]), 4096)
                kp = place(self._pack_matrix_blob(weights[lp + "self_attn.k_proj.weight"]), 4096)
                vp = place(self._pack_matrix_blob(weights[lp + "self_attn.v_proj.weight"]), 4096)
                op = place(self._pack_matrix_blob(weights[lp + "self_attn.o_proj.weight"]), 4096)
                gp = place(self._pack_matrix_blob(weights[lp + "self_attn.gate_proj.weight"]), 4096)
                qn = place(np.asarray(weights[lp + "self_attn.q_norm.weight"], dtype=np.float32).tobytes(), 128)
                kn = place(np.asarray(weights[lp + "self_attn.k_norm.weight"], dtype=np.float32).tobytes(), 128)
                desc.extend([qp, kp, vp, op, gp, qn, kn, 0])

            desc.extend([w1_off, w3_off, w2_off])
            assert len(desc) == 16, f"Layer {i} descriptor word count {len(desc)} != 16"
            layer_descs.append(desc)

        total_image_bytes = align(cursor, 4096)

        # Construct Header (256 bytes = 64 u32 words)
        header_words = [0] * 64
        header_words[0] = LFM_MAGIC
        header_words[1] = LFM_VERSION
        header_words[2] = self.quant_type
        header_words[3] = 0  # text model
        header_words[4] = self.num_layers
        header_words[5] = self.hidden_size
        header_words[6] = self.intermediate_size
        header_words[7] = self.vocab_size
        header_words[8] = NUM_ATTENTION_HEADS
        header_words[9] = NUM_KV_HEADS
        header_words[10] = HEAD_DIM
        header_words[11] = CONV_KERNEL_SIZE
        header_words[12] = SLIDING_WINDOW
        header_words[13] = struct.unpack("<I", struct.pack("<f", float(NORM_EPS)))[0]
        header_words[14] = int(ROPE_FULL_THETA)
        header_words[15] = int(ROPE_SLIDING_THETA)
        header_words[16] = PARTIAL_ROTARY_DIM
        header_words[17] = 256 + self.num_layers * 64
        header_words[18] = 256  # layer table offset
        header_words[19] = emb_norm_off
        header_words[20] = embed_tokens_off
        header_words[21] = 0  # vision tower
        header_words[22] = total_image_bytes

        header_bytes = struct.pack("<64I", *header_words)
        layer_table_bytes = b"".join(struct.pack("<16I", *desc) for desc in layer_descs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.truncate(total_image_bytes)
            f.seek(0)
            f.write(header_bytes)
            f.seek(256)
            f.write(layer_table_bytes)
            for off, blob in parts:
                f.seek(off)
                f.write(blob)

        meta = {
            "format": "PackedLayout-v1",
            "quant_type": self.quant_type_name,
            "layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "total_bytes": total_image_bytes,
            "embedding_norm_offset": emb_norm_off,
            "embed_tokens_offset": embed_tokens_off,
        }
        output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Packed LFM3 ({self.quant_type_name}) successfully to {output_path} ({total_image_bytes / 1024 / 1024:.2f} MiB)")
        return meta

    def emit_hxfg_stage_constant_pools(self, weights: Dict[str, np.ndarray], output_dir: Union[str, Path]) -> List[Path]:
        """Emits HXFG constant pools for 4 stage graphs under hexecutord."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stages = []

        # 4 stages of 6 layers each
        layers_per_stage = max(1, self.num_layers // 4)
        sample_key = next(iter(weights.keys()))
        prefix = "model.language_model." if "model.language_model." in sample_key else "model."

        for stage_idx in range(4):
            start_l = stage_idx * layers_per_stage
            end_l = min(self.num_layers, (stage_idx + 1) * layers_per_stage)
            if start_l >= self.num_layers:
                break

            pool = bytearray()

            def emit_blob(blob: bytes, a: int = 128) -> int:
                at = (len(pool) + a - 1) // a * a
                pool.extend(bytes(at - len(pool)))
                pool.extend(blob)
                return at

            for i in range(start_l, end_l):
                lp = f"{prefix}layers.{i}."
                ltype = self.layer_types[i]
                emit_blob(np.asarray(weights[lp + "operator_norm.weight"], dtype=np.float32).tobytes())
                emit_blob(np.asarray(weights[lp + "post_operator_norm.weight"], dtype=np.float32).tobytes())
                emit_blob(np.asarray(weights[lp + "ffn_norm.weight"], dtype=np.float32).tobytes())
                emit_blob(np.asarray(weights[lp + "post_ffn_norm.weight"], dtype=np.float32).tobytes())

                emit_blob(self._pack_matrix_blob(weights[lp + "feed_forward.w1.weight"]), 128)
                emit_blob(self._pack_matrix_blob(weights[lp + "feed_forward.w3.weight"]), 128)
                emit_blob(self._pack_matrix_blob(weights[lp + "feed_forward.w2.weight"]), 128)

                if ltype == LAYER_CONV:
                    emit_blob(self._pack_matrix_blob(weights[lp + "conv.in_proj.weight"]), 128)
                    conv_t = np.asarray(weights[lp + "conv.conv.weight"], dtype=np.float32).reshape(self.hidden_size, CONV_KERNEL_SIZE)
                    emit_blob(conv_t.tobytes())
                    emit_blob(self._pack_matrix_blob(weights[lp + "conv.out_proj.weight"]), 128)
                else:
                    emit_blob(self._pack_matrix_blob(weights[lp + "self_attn.q_proj.weight"]), 128)
                    emit_blob(self._pack_matrix_blob(weights[lp + "self_attn.k_proj.weight"]), 128)
                    emit_blob(self._pack_matrix_blob(weights[lp + "self_attn.v_proj.weight"]), 128)
                    emit_blob(self._pack_matrix_blob(weights[lp + "self_attn.o_proj.weight"]), 128)
                    emit_blob(self._pack_matrix_blob(weights[lp + "self_attn.gate_proj.weight"]), 128)
                    emit_blob(np.asarray(weights[lp + "self_attn.q_norm.weight"], dtype=np.float32).tobytes())
                    emit_blob(np.asarray(weights[lp + "self_attn.k_norm.weight"], dtype=np.float32).tobytes())

            if stage_idx == 3:
                emit_blob(np.asarray(weights[prefix + "embedding_norm.weight"], dtype=np.float32).tobytes())

            pool_path = out_dir / f"lfm3_stage_{stage_idx}.bin"
            pool_path.write_bytes(pool)
            stages.append(pool_path)
            print(f"Emitted HXFG constant pool stage {stage_idx} to {pool_path} ({len(pool) / 1024 / 1024:.2f} MiB)")

        return stages


def load_safetensors_file(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Pure-Python / NumPy safetensors loader."""
    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size).decode("utf-8")
        metadata = json.loads(header_json)
        data_base_offset = 8 + header_size

        tensors = {}
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
            shape = info["shape"]
            dtype_str = info["dtype"]
            start_off, end_off = info["data_offsets"]
            f.seek(data_base_offset + start_off)
            num_bytes = end_off - start_off
            raw = f.read(num_bytes)

            if dtype_str == "BF16":
                # Convert bfloat16 to float32
                u16 = np.frombuffer(raw, dtype=np.uint16)
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32).reshape(shape)
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype=np.uint8)
            tensors[name] = arr
    return tensors


def main():
    p = argparse.ArgumentParser(description="LFM3 DSP Model Packer")
    p.add_argument("--model", type=Path, default=None, help="Path to safetensors model")
    p.add_argument("--output", type=Path, default=Path("lfm3.dspbin"), help="Output packed binary file")
    p.add_argument("--quant", type=str, default="fp16", choices=["fp16", "int8", "q4"], help="Quantization format")
    p.add_argument("--layers", type=int, default=NUM_HIDDEN_LAYERS, help="Number of hidden layers")
    p.add_argument("--dummy", action="store_true", help="Generate synthetic deterministic weights for offline testing")
    p.add_argument("--emit-hxfg-stages", type=Path, default=None, help="Directory to emit 4-stage HXFG constant pools")
    args = p.parse_args()

    packer = LfmModelPacker(quant_type=args.quant, num_layers=args.layers)

    if args.dummy or args.model is None:
        print("Generating synthetic LFM3 weights for packing verification...")
        weights = packer.build_dummy_weights()
    else:
        print(f"Loading weights from {args.model}...")
        weights = load_safetensors_file(args.model)

    packer.pack(weights, args.output)

    if args.emit_hxfg_stages is not None:
        packer.emit_hxfg_stage_constant_pools(weights, args.emit_hxfg_stages)


if __name__ == "__main__":
    main()
