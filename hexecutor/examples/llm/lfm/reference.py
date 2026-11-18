#!/usr/bin/env python3
"""CPU reference forward pass for LFM3 models.

Reads 'Packed layout v1' binary images (.dspbin, FP16 or Q4_0), dequantizes
weights on the fly, and implements:
1. Single decode step (1 token forward, with persistent conv state and KV cache)
2. Short prefill forward (multi-token prompt with KV/conv state accumulation)
3. Numeric comparison against Hugging Face reference model forward, reporting
   max absolute logit error and relative logit error.

Usage:
  python3 reference.py --packed lfm3_fp16.dspbin --compare
  python3 reference.py --packed lfm3_q4.dspbin --compare
"""

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Format constants
LFM_MAGIC = 0x314D464C
QUANT_FP16 = 0
QUANT_INT8 = 1
QUANT_Q4_0 = 2

LAYER_CONV = 0
LAYER_FULL_ATTENTION = 1
LAYER_SLIDING_ATTENTION = 2


def dequantize_ggml_q4_0(raw: np.ndarray) -> np.ndarray:
    """Dequantize raw GGML Q4_0 blocks [N, K/32, 18] to float32 [N, K]."""
    n, kb, _ = raw.shape
    d_bytes = raw[..., :2]
    qs = raw[..., 2:]
    scale = d_bytes.view(np.float16).astype(np.float32)

    q_low = qs & 0x0F
    q_high = qs >> 4
    q = np.concatenate((q_low, q_high), axis=-1)
    x = (q.astype(np.float32) - 8.0) * scale
    return x.reshape(n, kb * 32)


class LfmPackedReader:
    """Parses a Packed Layout v1 binary file and provides dequantized weights."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        with open(self.file_path, "rb") as f:
            header_bytes = f.read(256)
            hw = struct.unpack("<64I", header_bytes)
            assert hw[0] == LFM_MAGIC, f"Invalid magic: {hw[0]:#x}"
            assert hw[1] == 1, f"Unsupported version: {hw[1]}"
            self.quant_type = hw[2]
            self.model_type = hw[3]
            self.num_layers = hw[4]
            self.hidden_size = hw[5]
            self.intermediate_size = hw[6]
            self.vocab_size = hw[7]
            self.num_heads = hw[8]
            self.num_kv_heads = hw[9]
            self.head_dim = hw[10]
            self.conv_kernel_size = hw[11]
            self.sliding_window = hw[12]
            self.norm_eps = struct.unpack("<f", struct.pack("<I", hw[13]))[0]
            self.rope_full_theta = float(hw[14])
            self.rope_sliding_theta = float(hw[15])
            self.partial_rotary_dim = hw[16]
            self.embedding_norm_offset = hw[19]
            self.embed_tokens_offset = hw[20]

            # Read layer descriptors (24 * 64 bytes = 1536 bytes)
            f.seek(256)
            self.layer_descs = []
            for _ in range(self.num_layers):
                desc_bytes = f.read(64)
                self.layer_descs.append(struct.unpack("<16I", desc_bytes))

            # Memory-map the full file for fast slice access
            self.raw_data = self.file_path.read_bytes()

    def read_vector(self, offset: int, length: int) -> np.ndarray:
        """Read float32 vector of given length."""
        raw = self.raw_data[offset : offset + length * 4]
        return np.frombuffer(raw, dtype=np.float32)

    def read_matrix(self, offset: int, rows: int, cols: int) -> np.ndarray:
        """Read and dequantize a 2D matrix [rows, cols]."""
        if self.quant_type == QUANT_FP16:
            raw = self.raw_data[offset : offset + rows * cols * 2]
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(rows, cols)
        elif self.quant_type == QUANT_INT8:
            raw_q = self.raw_data[offset : offset + rows * cols]
            raw_s = self.raw_data[offset + rows * cols : offset + rows * cols + rows * 2]
            q = np.frombuffer(raw_q, dtype=np.int8).reshape(rows, cols).astype(np.float32)
            scale = np.frombuffer(raw_s, dtype=np.float16).astype(np.float32).reshape(rows, 1)
            return q * scale
        elif self.quant_type == QUANT_Q4_0:
            bytes_per_row = (cols // 32) * 18
            raw = self.raw_data[offset : offset + rows * bytes_per_row]
            raw_blk = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // 32, 18)
            return dequantize_ggml_q4_0(raw_blk)
        raise ValueError(f"Unknown quant type {self.quant_type}")

    def load_layer_weights(self, layer_idx: int) -> Dict:
        """Loads and dequantizes all tensors for a specific layer."""
        desc = self.layer_descs[layer_idx]
        ltype = desc[0]
        layer = {
            "type": ltype,
            "operator_norm": self.read_vector(desc[1], self.hidden_size),
            "post_operator_norm": self.read_vector(desc[2], self.hidden_size),
            "ffn_norm": self.read_vector(desc[3], self.hidden_size),
            "post_ffn_norm": self.read_vector(desc[4], self.hidden_size),
            "w1": self.read_matrix(desc[13], self.intermediate_size, self.hidden_size),
            "w3": self.read_matrix(desc[14], self.intermediate_size, self.hidden_size),
            "w2": self.read_matrix(desc[15], self.hidden_size, self.intermediate_size),
        }
        if ltype == LAYER_CONV:
            layer["in_proj"] = self.read_matrix(desc[5], 3 * self.hidden_size, self.hidden_size)
            raw_taps = self.read_vector(desc[6], self.hidden_size * self.conv_kernel_size)
            layer["conv_taps"] = raw_taps.reshape(self.hidden_size, self.conv_kernel_size)
            layer["out_proj"] = self.read_matrix(desc[7], self.hidden_size, self.hidden_size)
        else:
            layer["q_proj"] = self.read_matrix(desc[5], self.hidden_size, self.hidden_size)
            layer["k_proj"] = self.read_matrix(desc[6], self.num_kv_heads * self.head_dim, self.hidden_size)
            layer["v_proj"] = self.read_matrix(desc[7], self.num_kv_heads * self.head_dim, self.hidden_size)
            layer["o_proj"] = self.read_matrix(desc[8], self.hidden_size, self.hidden_size)
            layer["gate_proj"] = self.read_matrix(desc[9], self.num_heads, self.hidden_size)
            layer["q_norm"] = self.read_vector(desc[10], self.head_dim)
            layer["k_norm"] = self.read_vector(desc[11], self.head_dim)
        return layer


class Lfm3CPUReference:
    """Pure CPU reference forward pass matching LFM3 PyTorch semantics."""

    def __init__(self, reader: LfmPackedReader):
        self.reader = reader
        self.hidden_size = reader.hidden_size
        self.intermediate_size = reader.intermediate_size
        self.vocab_size = reader.vocab_size
        self.num_layers = reader.num_layers
        self.num_heads = reader.num_heads
        self.num_kv_heads = reader.num_kv_heads
        self.head_dim = reader.head_dim
        self.conv_kernel_size = reader.conv_kernel_size
        self.sliding_window = reader.sliding_window
        self.norm_eps = reader.norm_eps
        self.partial_rotary_dim = reader.partial_rotary_dim

        # Global weights
        self.embedding_norm = reader.read_vector(reader.embedding_norm_offset, self.hidden_size)
        if reader.embed_tokens_offset != 0:
            self.embed_tokens = reader.read_matrix(reader.embed_tokens_offset, self.vocab_size, self.hidden_size)
        else:
            self.embed_tokens = None

        # Cache all layer weights
        self.layers = [reader.load_layer_weights(i) for i in range(self.num_layers)]

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Compute RMSNorm without mean subtraction: y = x / sqrt(mean(x^2) + eps) * weight."""
        variance = np.mean(np.square(x), axis=-1, keepdims=True)
        return (x / np.sqrt(variance + self.norm_eps)) * weight

    def apply_rope(self, x: np.ndarray, pos: int, theta: float, partial_rotary_factor: float = 1.0) -> np.ndarray:
        """Applies partial RoPE with zero-tail progression matching Lfm3RotaryEmbedding."""
        dim = self.head_dim
        rope_angles = int(partial_rotary_factor * dim) // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / dim))
        inv_freq = np.concatenate((inv_freq, np.zeros(dim // 2 - rope_angles, dtype=np.float64)))

        freqs = float(pos) * inv_freq
        emb = np.concatenate((freqs, freqs))
        cos = np.cos(emb).astype(np.float32)
        sin = np.sin(emb).astype(np.float32)

        # rotate half: cat(-x2, x1)
        x1 = x[..., : dim // 2]
        x2 = x[..., dim // 2 :]
        rot_x = np.concatenate((-x2, x1), axis=-1)
        return (x * cos) + (rot_x * sin)

    def init_state(self) -> Dict:
        """Initializes empty conv states and KV caches."""
        state = {"conv_state": {}, "kv_cache": {}}
        for i, l in enumerate(self.layers):
            if l["type"] == LAYER_CONV:
                # 6 historical tokens per channel
                state["conv_state"][i] = np.zeros((self.hidden_size, self.conv_kernel_size - 1), dtype=np.float32)
            else:
                # K and V caches: list of head vectors
                state["kv_cache"][i] = {"k": [], "v": []}
        return state

    def decode_step(self, token_id: int, pos: int, state: Dict) -> Tuple[np.ndarray, Dict]:
        """Executes a single token decode step (batch=1, seq_len=1).

        Args:
          token_id: Integer token ID.
          pos: Sequence position integer.
          state: Mutable persistent state dict.

        Returns:
          Tuple of (logits [vocab_size], updated state).
        """
        assert self.embed_tokens is not None, "embed_tokens must be loaded"
        x = self.embed_tokens[token_id].astype(np.float32).copy()

        for i, layer in enumerate(self.layers):
            residual = x.copy()
            x_norm = self.rmsnorm(x, layer["operator_norm"])

            if layer["type"] == LAYER_CONV:
                # 1. Input projection and chunk
                bcx = x_norm @ layer["in_proj"].T  # [6144]
                b = bcx[: self.hidden_size]
                c = bcx[self.hidden_size : 2 * self.hidden_size]
                u = bcx[2 * self.hidden_size :]
                g = b * u  # elementwise gating [2048]

                # 2. Causal depthwise conv width 7
                # State holds 6 past tokens: [2048, 6]
                hist = state["conv_state"][i]
                # Full 7-tap window: [h0, h1, h2, h3, h4, h5, g]
                window = np.column_stack((hist, g))  # [2048, 7]
                conv_out = np.sum(window * layer["conv_taps"], axis=1)  # [2048]
                # Update persistent state: discard oldest h0, append g
                state["conv_state"][i] = window[:, 1:]

                # 3. Output gating and out_proj
                y = c * conv_out
                op_out = y @ layer["out_proj"].T
            else:
                # Attention Layer
                q_raw = x_norm @ layer["q_proj"].T  # [2048]
                k_raw = x_norm @ layer["k_proj"].T  # [512]
                v_raw = x_norm @ layer["v_proj"].T  # [512]
                gate_logits = x_norm @ layer["gate_proj"].T  # [16]
                gate = 1.0 / (1.0 + np.exp(-gate_logits))  # sigmoid [16]

                # Reshape to heads
                q_heads = q_raw.reshape(self.num_heads, self.head_dim)
                k_heads = k_raw.reshape(self.num_kv_heads, self.head_dim)
                v_heads = v_raw.reshape(self.num_kv_heads, self.head_dim)

                # Per-head QK-Norm
                for h in range(self.num_heads):
                    q_heads[h] = self.rmsnorm(q_heads[h], layer["q_norm"])
                for h in range(self.num_kv_heads):
                    k_heads[h] = self.rmsnorm(k_heads[h], layer["k_norm"])

                # Rotary Embeddings
                theta = self.reader.rope_full_theta if layer["type"] == LAYER_FULL_ATTENTION else self.reader.rope_sliding_theta
                factor = 0.5 if layer["type"] == LAYER_FULL_ATTENTION else 1.0
                for h in range(self.num_heads):
                    q_heads[h] = self.apply_rope(q_heads[h], pos, theta, factor)
                for h in range(self.num_kv_heads):
                    k_heads[h] = self.apply_rope(k_heads[h], pos, theta, factor)

                # Append to KV cache
                state["kv_cache"][i]["k"].append(k_heads)
                state["kv_cache"][i]["v"].append(v_heads)

                # Multi-head attention with GQA
                k_all = np.array(state["kv_cache"][i]["k"])  # [T, 4, 128]
                v_all = np.array(state["kv_cache"][i]["v"])  # [T, 4, 128]
                total_len = len(k_all)

                head_outputs = []
                kv_group_size = self.num_heads // self.num_kv_heads  # 4

                for h in range(self.num_heads):
                    kv_h = h // kv_group_size
                    k_h = k_all[:, kv_h, :]  # [T, 128]
                    v_h = v_all[:, kv_h, :]  # [T, 128]

                    # Scores
                    scores = (k_h @ q_heads[h]) / math.sqrt(self.head_dim)  # [T]

                    # Sliding window mask if applicable
                    if layer["type"] == LAYER_SLIDING_ATTENTION:
                        min_pos = pos - self.sliding_window
                        for past_p in range(total_len):
                            if past_p <= min_pos:
                                scores[past_p] = -1e9

                    # Softmax
                    exp_s = np.exp(scores - np.max(scores))
                    attn_weights = exp_s / np.sum(exp_s)
                    head_out = attn_weights @ v_h  # [128]
                    head_out = head_out * gate[h]
                    head_outputs.append(head_out)

                concat_heads = np.concatenate(head_outputs)  # [2048]
                op_out = concat_heads @ layer["o_proj"].T

            # Sandwich post-operator norm and residual add
            op_normed = self.rmsnorm(op_out, layer["post_operator_norm"])
            x = residual + op_normed

            # SwiGLU FFN
            ffn_residual = x.copy()
            x_ffn_norm = self.rmsnorm(x, layer["ffn_norm"])
            w1 = x_ffn_norm @ layer["w1"].T
            w3 = x_ffn_norm @ layer["w3"].T
            silu_w1 = w1 * (1.0 / (1.0 + np.exp(-w1)))
            ffn_hidden = silu_w1 * w3
            ffn_out = ffn_hidden @ layer["w2"].T
            ffn_normed = self.rmsnorm(ffn_out, layer["post_ffn_norm"])
            x = ffn_residual + ffn_normed

        # Final embedding norm and tied head projection
        x_final = self.rmsnorm(x, self.embedding_norm)
        logits = x_final @ self.embed_tokens.T
        return logits, state

    def prefill(self, token_ids: List[int], state: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Prefill a sequence of prompt tokens."""
        if state is None:
            state = self.init_state()
        all_logits = []
        for pos, tok in enumerate(token_ids):
            logits, state = self.decode_step(tok, pos, state)
            all_logits.append(logits)
        return np.array(all_logits), state


def compare_with_hf(
    packed_path: Union[str, Path],
    prompt_tokens: List[int] = [12, 45, 89, 102],
) -> Dict:
    """Verifies reference CPU forward against Hugging Face implementation."""
    reader = LfmPackedReader(packed_path)
    ref_model = Lfm3CPUReference(reader)

    print(f"\n--- Running Reference CPU Forward ({reader.num_layers} layers, quant_type={reader.quant_type}) ---")
    ref_logits, _ = ref_model.prefill(prompt_tokens)
    last_ref_logits = ref_logits[-1]

    # Dynamically build matching HF model configuration
    import torch
    # Check if local scratch download has configuration_lfm3
    scratch_dir = Path("/Users/owenelliott/.gemini/antigravity-cli/brain/198b6a0b-f9a2-4ed7-a582-3225b04ae3be/scratch/lfm_download")
    sys.path.insert(0, str(scratch_dir))
    import vl2b.configuration_lfm3 as cfg_mod
    import vl2b.modeling_lfm3 as model_mod

    layer_types_str = []
    for l in ref_model.layers:
        if l["type"] == LAYER_CONV:
            layer_types_str.append("conv")
        elif l["type"] == LAYER_FULL_ATTENTION:
            layer_types_str.append("full_attention")
        else:
            layer_types_str.append("sliding_attention")

    cfg = cfg_mod.Lfm3Config(
        hidden_size=ref_model.hidden_size,
        intermediate_size=ref_model.intermediate_size,
        num_hidden_layers=ref_model.num_layers,
        num_attention_heads=ref_model.num_heads,
        num_key_value_heads=ref_model.num_kv_heads,
        head_dim=ref_model.head_dim,
        vocab_size=ref_model.vocab_size,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        conv_L_cache=ref_model.conv_kernel_size,
        conv_bias=False,
        use_sandwich_norm=True,
        layer_types=layer_types_str,
        rope_parameters={
            "full_attention": {"rope_theta": ref_model.reader.rope_full_theta, "partial_rotary_factor": 0.5, "rope_type": "default"},
            "sliding_attention": {"rope_theta": ref_model.reader.rope_sliding_theta, "rope_type": "default"},
        },
    )

    hf_model = model_mod.Lfm3ForCausalLM(cfg).eval()

    # Load dequantized reference weights into HF model
    with torch.no_grad():
        hf_model.model.embed_tokens.weight.copy_(torch.from_numpy(ref_model.embed_tokens).clone())
        hf_model.lm_head.weight.copy_(torch.from_numpy(ref_model.embed_tokens).clone())
        hf_model.model.embedding_norm.weight.copy_(torch.from_numpy(ref_model.embedding_norm).clone())

        for i, l in enumerate(ref_model.layers):
            m = hf_model.model.layers[i]
            m.operator_norm.weight.copy_(torch.from_numpy(l["operator_norm"]).clone())
            m.post_operator_norm.weight.copy_(torch.from_numpy(l["post_operator_norm"]).clone())
            m.ffn_norm.weight.copy_(torch.from_numpy(l["ffn_norm"]).clone())
            m.post_ffn_norm.weight.copy_(torch.from_numpy(l["post_ffn_norm"]).clone())

            m.feed_forward.w1.weight.copy_(torch.from_numpy(l["w1"]).clone())
            m.feed_forward.w3.weight.copy_(torch.from_numpy(l["w3"]).clone())
            m.feed_forward.w2.weight.copy_(torch.from_numpy(l["w2"]).clone())

            if l["type"] == LAYER_CONV:
                m.conv.in_proj.weight.copy_(torch.from_numpy(l["in_proj"]).clone())
                m.conv.conv.weight.copy_(torch.from_numpy(l["conv_taps"][:, None, :]).clone())
                m.conv.out_proj.weight.copy_(torch.from_numpy(l["out_proj"]).clone())
            else:
                m.self_attn.q_proj.weight.copy_(torch.from_numpy(l["q_proj"]).clone())
                m.self_attn.k_proj.weight.copy_(torch.from_numpy(l["k_proj"]).clone())
                m.self_attn.v_proj.weight.copy_(torch.from_numpy(l["v_proj"]).clone())
                m.self_attn.o_proj.weight.copy_(torch.from_numpy(l["o_proj"]).clone())
                m.self_attn.gate_proj.weight.copy_(torch.from_numpy(l["gate_proj"]).clone())
                m.self_attn.q_norm.weight.copy_(torch.from_numpy(l["q_norm"]).clone())
                m.self_attn.k_norm.weight.copy_(torch.from_numpy(l["k_norm"]).clone())

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        hf_out = hf_model(input_ids, use_cache=False)
        hf_logits = hf_out.logits[0, -1].numpy()

    max_abs_err = float(np.max(np.abs(last_ref_logits - hf_logits)))
    rel_err = float(np.max(np.abs(last_ref_logits - hf_logits) / (np.abs(hf_logits) + 1e-5)))
    rms_err = float(np.sqrt(np.mean(np.square(last_ref_logits - hf_logits))))

    report = {
        "packed_file": str(packed_path),
        "quant_type": "FP16" if reader.quant_type == QUANT_FP16 else ("Q4_0" if reader.quant_type == QUANT_Q4_0 else "INT8"),
        "prompt_len": len(prompt_tokens),
        "max_abs_error": max_abs_err,
        "max_rel_error": rel_err,
        "rms_error": rms_err,
        "top5_ref": [int(x) for x in np.argsort(-last_ref_logits)[:5]],
        "top5_hf": [int(x) for x in np.argsort(-hf_logits)[:5]],
    }
    print(f"Comparison Result: Max Abs Error = {max_abs_err:.6e}, Max Rel Error = {rel_err:.6e}, RMS Error = {rms_err:.6e}")
    print(f"Top 5 Ref Token IDs: {report['top5_ref']}")
    print(f"Top 5 HF Token IDs:  {report['top5_hf']}")
    return report


def main():
    p = argparse.ArgumentParser(description="LFM3 CPU Reference Runner")
    p.add_argument("--packed", type=Path, required=True, help="Path to packed .dspbin file")
    p.add_argument("--prompt", type=int, nargs="+", default=[12, 45, 89, 102], help="Prompt token IDs")
    p.add_argument("--compare", action="store_true", help="Compare with Hugging Face modeling output")
    args = p.parse_args()

    if args.compare:
        compare_with_hf(args.packed, prompt_tokens=args.prompt)
    else:
        reader = LfmPackedReader(args.packed)
        ref_model = Lfm3CPUReference(reader)
        logits, state = ref_model.prefill(args.prompt)
        print(f"Prefill of {len(args.prompt)} tokens finished. Last logit shape: {logits[-1].shape}")


if __name__ == "__main__":
    main()
