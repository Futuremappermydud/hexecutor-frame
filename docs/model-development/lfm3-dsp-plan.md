# LFM3 DSP Architecture Delta, Memory Plan, and Packed Layout v1

**Date:** 2026-11-17  
**Status:** In Progress (Worker 1 of 3: Architecture Delta + Weight Converter)  
**Target:** Liquid AI LFM3 family on Qualcomm Hexagon CDSP via Hexecutor Broker (`hexecutord`)  
**Primary Checkpoint:** `LiquidAI/LFM3-VL-2B` (and text decoder `LiquidAI/LFM3-1.7B`)

---

## 1. Executive Summary & Checkpoint Identification

This plan specifies the execution of the Liquid "LFM3" family architecture on the Steam Frame CDSP.

Target checkpoints identified directly from Hugging Face Hub:
1. **Multimodal / Vision-Language Target:** `LiquidAI/LFM3-VL-2B` (`lfm3_vl`)
2. **Text-Decoder Base:** `LiquidAI/LFM3-1.7B` (`lfm3`)

### Key Architecture Finding:
The text decoders of `LiquidAI/LFM3-VL-2B` (`text_config`) and `LiquidAI/LFM3-1.7B` are **bit-for-bit architecturally identical**:
- Same 24 hidden layers with identical 6-layer repeating hybrid pattern: 16 causal short-conv layers, 4 full-attention layers, 4 sliding-window attention layers.
- Same dimensions: $d_{model} = 2048$, $d_{ffn} = 7680$ (SwiGLU), $n_{heads} = 16$, $n_{kv\_heads} = 4$ (GQA 4:1), $d_{head} = 128$.
- Same vocabulary: 128,000 tokens with tied input/output word embeddings (`tie_word_embeddings = true`).
- Same Sandwich RMSNorm topology with $\epsilon = 10^{-5}$.
- Same width-7 causal depthwise convolution with `conv_L_cache = 7` (6 historical tokens).

In `LiquidAI/LFM3-VL-2B`, visual tokens from the 14-layer vision encoder ($d_{vision} = 1024$) and 2-layer MLP projector are mapped to $d=2048$ and spliced into the text embedding stream; the downstream text decoder processes them identically to text tokens.

---

## 2. Packed Layout v1 Specification

> [!IMPORTANT]
> **Contract for Worker 2 (DSP Kernels):** This section defines the resident memory and binary format for LFM3 weights, normalization factors, and layer tables. Worker 2's DSP kernels consume this layout.

The Packed Layout v1 format stores model weights either as a unified binary image or as partitioned HXFG constant pools suitable for the CDSP broker.

### 2.1 File Header (256 bytes)

All fields are little-endian.

```c
typedef struct {
    uint32_t magic;               /* 0x314d464c ("LFM1") */
    uint32_t version;             /* 1 */
    uint32_t quant_type;          /* 0 = FP16, 1 = INT8, 2 = Q4_0 */
    uint32_t model_type;          /* 0 = Text (1.7B), 1 = VL (2B) */
    uint32_t layers;              /* 24 */
    uint32_t hidden_size;         /* 2048 */
    uint32_t intermediate_size;   /* 7680 */
    uint32_t vocab_size;          /* 128000 */
    uint32_t num_heads;           /* 16 */
    uint32_t num_kv_heads;        /* 4 */
    uint32_t head_dim;            /* 128 */
    uint32_t conv_kernel_size;    /* 7 */
    uint32_t sliding_window;      /* 256 */
    uint32_t norm_eps_bits;       /* Float32 bit pattern of 1e-5 */
    uint32_t rope_full_theta;     /* 1000000 */
    uint32_t rope_sliding_theta;  /* 10000 */
    uint32_t partial_rotary_dim;  /* 64 (0.5 * head_dim) */
    uint32_t header_bytes;        /* 256 + 24 * 64 = 1792 bytes */
    uint32_t layer_table_offset;  /* 256 */
    uint32_t embedding_norm_off;  /* Offset to final RMSNorm (2048 float32s) */
    uint32_t embed_tokens_off;    /* Offset to token embeddings (or 0 if host-resident) */
    uint32_t vision_tower_off;    /* Offset to vision tower weights (or 0 if text-only) */
    uint32_t total_image_bytes;   /* Total size of packed image */
    uint8_t  fingerprint[16];     /* SHA-256 slice */
    uint32_t reserved[39];        /* Padded to 256 bytes */
} LfmPackedHeader;
```

### 2.2 Per-Layer Descriptor Table (64 bytes per layer, 1536 bytes total)

Immediately follows the 256-byte header at offset 256:

```c
enum LfmLayerType {
    LFM_LAYER_CONV = 0,
    LFM_LAYER_FULL_ATTENTION = 1,
    LFM_LAYER_SLIDING_ATTENTION = 2
};

typedef struct {
    uint32_t layer_type;              /* LFM_LAYER_* */
    uint32_t operator_norm_offset;    /* Pre-operator RMSNorm (2048 f32, 8 KB) */
    uint32_t post_operator_norm_off;  /* Post-operator RMSNorm (2048 f32, 8 KB) */
    uint32_t ffn_norm_offset;         /* Pre-FFN RMSNorm (2048 f32, 8 KB) */
    uint32_t post_ffn_norm_offset;    /* Post-FFN RMSNorm (2048 f32, 8 KB) */

    /* Union based on layer_type */
    union {
        struct {
            uint32_t in_proj_offset;   /* [6144, 2048] quantized matrix */
            uint32_t conv_taps_offset; /* [2048, 7] float32 depthwise taps */
            uint32_t out_proj_offset;  /* [2048, 2048] quantized matrix */
            uint32_t reserved[5];
        } conv;
        struct {
            uint32_t q_proj_offset;    /* [2048, 2048] quantized matrix */
            uint32_t k_proj_offset;    /* [512, 2048] quantized matrix */
            uint32_t v_proj_offset;    /* [512, 2048] quantized matrix */
            uint32_t o_proj_offset;    /* [2048, 2048] quantized matrix */
            uint32_t gate_proj_offset; /* [16, 2048] quantized matrix */
            uint32_t q_norm_offset;    /* [128] float32 per-head RMSNorm */
            uint32_t k_norm_offset;    /* [128] float32 per-head RMSNorm */
            uint32_t reserved;
        } attn;
    } op;

    /* Feed-Forward (SwiGLU) */
    uint32_t w1_offset;               /* [7680, 2048] quantized matrix */
    uint32_t w3_offset;               /* [7680, 2048] quantized matrix */
    uint32_t w2_offset;               /* [2048, 7680] quantized matrix */
    uint32_t reserved[3];
} LfmLayerDesc;
```

### 2.3 Quantized Matrix Layouts
All matrix offsets are aligned to 4096 bytes (4 KB page boundary) to allow zero-copy DMA streaming and memory mapping.
- **FP16 (`quant_type = 0`):** Contiguous row-major IEEE 754 float16 `[N, K]`.
- **INT8 (`quant_type = 1`):** Row-major signed int8 `[N, K]` followed by per-row FP16 scales `[N]`.
- **Q4_0 (`quant_type = 2`):** Blocked 32-weight group layout. Each 32-weight block consists of:
  - 16 bytes of packed 4-bit nibbles (low nibble first).
  - 2 bytes of IEEE 754 float16 scale factor $d$.
  - Packed into groups of 32 output rows matching HVX vector widths ($32 \times 4 = 128$ bytes) to enable direct single-cycle HVX `vrmpy` instructions without transposition.

---

## 3. Per-Layer Block Table

The decoder comprises 24 layers organized into 4 identical 6-layer macro-blocks:

| Layer Index | Layer Type | Operator | Operator Weights | FFN (SwiGLU) Weights | Norms Present |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0, 6, 12, 18** | `conv` | Gated Depthwise Conv1d (W=7) | $W_{in} [6144, 2048]$<br>$W_{conv} [2048, 7]$<br>$W_{out} [2048, 2048]$ | $W_1, W_3 [7680, 2048]$<br>$W_2 [2048, 7680]$ | operator_norm (2048)<br>post_operator_norm (2048)<br>ffn_norm (2048)<br>post_ffn_norm (2048) |
| **1, 7, 13, 19** | `conv` | Gated Depthwise Conv1d (W=7) | (same as above) | (same as above) | (same as above) |
| **2, 8, 14, 20** | `full_attn` | Multi-Head Attention (Full context) | $W_q [2048, 2048]$<br>$W_k, W_v [512, 2048]$<br>$W_o [2048, 2048]$<br>$W_{gate} [16, 2048]$ | (same as above) | operator_norm, post_operator_norm,<br>ffn_norm, post_ffn_norm (all 2048)<br>q_norm, k_norm (128) |
| **3, 9, 15, 21** | `conv` | Gated Depthwise Conv1d (W=7) | (same as layer 0) | (same as above) | (same as layer 0) |
| **4, 10, 16, 22** | `sliding_attn` | Sliding Window Attention (W=256) | (same as layer 2) | (same as above) | (same as layer 2) |
| **5, 11, 17, 23** | `conv` | Gated Depthwise Conv1d (W=7) | (same as layer 0) | (same as above) | (same as layer 0) |

- **Final Layer Output:** Followed by `embedding_norm` (RMSNorm 2048, $\epsilon = 10^{-5}$) before `lm_head`.
- **Tied Word Embeddings:** `lm_head.weight` is tied to `model.embed_tokens.weight` ($128000 \times 2048$).

---

## 4. Architecture Delta: LFM3 vs. HXFG Tensor Backend

The Hexecutor tensor backend (`docs/hexecutor/tensor-backend.md`) executes static batch-one graphs (`tensor_f32`) through `hexecutord`.

### 4.1 What Maps to Existing HXFG Ops

| LFM3 Component | Operator Semantics | Existing HXFG Op Kind | Mapping Details |
| :--- | :--- | :--- | :--- |
| **Linear projections** | Matrix-vector product | `HX_CONV` (1x1) or `HX_MATMUL` | Convolutions with $1 \times 1$ spatial extent over token row. |
| **SwiGLU activation** | $\text{SiLU}(w_1 \cdot x)$ | `HX_ACT` (`HX_ACT_SILU`) | Native SiLU activation handler in `graph.c`. |
| **Elementwise gating** | $B \odot x$, $C \odot \text{conv\_out}$, $\text{silu}(w_1) \odot w_3$, $attn \odot \sigma(gate)$ | `HX_BINARY` (`HX_BIN_MUL`) | Broadcastable elementwise multiplication. |
| **Residual additions** | $x + \text{op\_out}$, $x + \text{ffn\_out}$ | `HX_ADD` / `HX_BINARY` (`HX_BIN_ADD`) | In-place or dual-operand float32 addition. |
| **Attention Softmax** | $\text{Softmax}(Q K^T / \sqrt{d})$ | `HX_SOFTMAX` | Per-pixel channel softmax with numerical stability subtraction. |
| **Attention MatMul** | $Q K^T$ and $A V$ | `HX_MATMUL` (Kind 21) | Runtime matrix multiplication ($p[0] = 1$ for $Q K^T$ transposed $B$, $p[1]$ = scale $1/\sqrt{d}$). |
| **LayerNorm (vision)** | Channel normalization | `HX_NORM` (`HX_NORM_CHANNEL`) | Multi-modal projector LayerNorm (computes mean and variance). |

### 4.2 What Requires New Ops or Backend Extensions

1. **`HX_RMSNORM` (RMS Normalization):**
   - *Current limitation:* Existing `HX_NORM` computes standard LayerNorm ($x - \mu / \sqrt{\sigma^2 + \epsilon}$).
   - *LFM3 requirement:* RMSNorm does not center around $\mu$: $y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$.
   - *Solution:* Add `HX_NORM_RMS` mode to `HX_NORM` ($p[0] = 2$) or a dedicated `HX_RMSNORM` op.
2. **`HX_SHORT_CONV` (Causal Depthwise 1D Conv with Persistent State):**
   - *Current limitation:* Existing `HX_DEPTHWISE` is a spatial 2D conv with stateless inputs.
   - *LFM3 requirement:* Width-7 1D causal depthwise conv maintaining a 6-token historical state per channel ($2048 \times 6$ FP16 values per layer).
   - *Decode step:* Shifts in 1 new token, computes dot-product with 7 kernel weights, updates state in-place.
3. **Persistent State Management (KV Cache & Conv State):**
   - *Current limitation:* HXFG graphs run statelessly from host input buffers to host output buffers.
   - *LFM3 requirement:* Transferring 18 MiB of KV cache across Unix sockets per token step would throttle performance. KV cache and conv state must live persistently in CDSP DDR / VTCM across broker dispatches.
4. **RoPE (Rotary Position Embeddings):**
   - *Current limitation:* HXFG has no rotary position operator.
   - *LFM3 requirement:* Complex multiplication of interleaved or halved pairs. Specifically, LFM3 `full_attention` uses **partial rotation (factor 0.5)**: only the first 64 dimensions of each head are rotated, and the remaining 64 dimensions remain unrotated (zero-tail frequency).
5. **Weight-Only Quantized MatVec (`HX_MATVEC_Q4` / `HX_CONV_Q4`):**
   - *Current limitation:* `HX_CONV` expects FP32 or FP16 weights in constant memory.
   - *LFM3 requirement:* 1.7B parameters in FP16 require 3.4 GB, which exceeds the entire physical DRAM of the CDSP! Weight-only Q4_0 (with FP16 activation scaling) reduces weights to ~800 MB, fitting into CDSP memory.

---

## 5. Execution Model under Hexecutor Broker (`hexecutord`)

The CDSP broker wraps one tensor `Session` over up to 8 graphs (`HXFG_MAX_GRAPHS = 8`).

### 5.1 Graph Partitioning Strategy (4 Stages)
Because the 24-layer model contains 1.49B layer weights, it cannot be loaded as a single monolithic HXFG graph. We exploit the natural 6-layer macro-block periodicity:

- **Graph 0 (`stage_0`):** Layers 0 to 5 (2 conv, 1 full_attn, 1 conv, 1 sliding_attn, 1 conv)
- **Graph 1 (`stage_1`):** Layers 6 to 11
- **Graph 2 (`stage_2`):** Layers 12 to 17
- **Graph 3 (`stage_3`):** Layers 18 to 23 + final `embedding_norm`

Total graphs = 4 (well within `HXFG_MAX_GRAPHS = 8`).

### 5.2 Token Step Execution Flow
```
Host (ARM)                                        CDSP (hexecutord broker)
Token ID -> Embedding lookup [1, 2048]
  │
  ├─ submit(stage_0, x) ─────────────────────────► Run Layers 0-5 (updates Conv State & KV Cache)
  │                                                   │
  ├◄─── x_stage0 [1, 2048] ───────────────────────────┘
  │
  ├─ submit(stage_1, x_stage0) ──────────────────► Run Layers 6-11
  │                                                   │
  ├◄─── x_stage1 [1, 2048] ───────────────────────────┘
  │
  ├─ submit(stage_2, x_stage1) ──────────────────► Run Layers 12-17
  │                                                   │
  ├◄─── x_stage2 [1, 2048] ───────────────────────────┘
  │
  ├─ submit(stage_3, x_stage2) ──────────────────► Run Layers 18-23 + embedding_norm
  │                                                   │
  ├◄─── x_final [1, 2048] ────────────────────────────┘
  │
Top-k / Greedy Sampling on Host (or optional lm_head stage on CDSP)
```

Keeping the $128,000 \times 2048$ embedding / `lm_head` lookup on the ARM host saves 141 MB of DSP RAM and prevents streaming giant vocabulary logit vectors through the IPC socket.

---

## 6. Weights and Memory Budget Table

### 6.1 Parameter Breakdown

| Component | Layer Count | Parameters per Layer | Total Parameters | FP16 Size | INT8 Size | Q4_0 Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conv Layers** | 16 | 63,985,664 | 1,023,770,624 | 2,047.5 MB | 1,023.8 MB | 575.9 MB |
| **Attention Layers** | 8 | 57,712,536 | 461,700,288 | 923.4 MB | 461.7 MB | 259.7 MB |
| **Final RMSNorm** | 1 | 2,048 | 2,048 | < 0.01 MB | < 0.01 MB | < 0.01 MB |
| **Text Decoder Subtotal** | **24** | — | **1,485,472,960** | **2,970.9 MB** | **1,485.5 MB** | **835.6 MB** |
| Embeddings / LM Head | 1 | 262,144,000 | 262,144,000 | 524.3 MB | 262.1 MB | 147.5 MB |
| **Total Text Model (1.7B)** | — | — | **1,747,616,960** | **3,495.2 MB** | **1,747.6 MB** | **983.0 MB** |
| Vision Tower (14 layers) | 14 | — | 164,205,312 | 328.4 MB | 164.2 MB | 92.4 MB |
| Multimodal Projector | 1 | 25,174,016 | 25,174,016 | 50.3 MB | 25.2 MB | 14.2 MB |
| **Total Vision + Projector** | **15** | — | **189,379,328** | **378.8 MB** | **189.4 MB** | **106.5 MB** |
| **Total LFM3-VL-2B** | — | — | **1,936,996,288** | **3,874.0 MB** | **1,937.0 MB** | **1,089.5 MB** |

*(Note: Q4_0 byte calculation includes 16 bytes nibbles + 2 bytes FP16 scale per 32 weights = 0.5625 bytes/weight).*

### 6.2 Dynamic Runtime Buffers

| Buffer | Calculation / Parameters | Memory Required |
| :--- | :--- | :--- |
| **KV Cache (2048 ctx)** | 4 full_attn layers ($4 \times 2 \times 4 \times 128 \times 2048 \times 2$ B) = 16.0 MiB<br>4 sliding_attn layers ($4 \times 2 \times 4 \times 128 \times 256 \times 2$ B) = 2.0 MiB | **18.0 MiB** |
| **KV Cache (4096 ctx)** | $32.0\text{ MiB} + 2.0\text{ MiB}$ | **34.0 MiB** |
| **Conv State** | 16 layers $\times 2048\text{ channels} \times 6\text{ tokens} \times 2\text{ bytes}$ | **0.39 MiB** (393.2 KB) |
| **Activation Scratch Arena** | Peak intermediate during decode step ($1 \times 7680 \times 4$ B) | **4.0 MiB** |
| **Prefill Scratch Arena** | 128-token prefill chunk ($128 \times 7680 \times 4$ B) | **16.0 MiB** |

### 6.3 CDSP Memory Lease Analysis & Recommended `size_mb`

- **Case 1: Decoder-Only (LFM3-1.7B) at Q4_0, Host-Assisted Embeddings:**
  - Layer Weights: 797 MiB
  - KV Cache (2048 ctx): 18 MiB
  - Conv State: 0.4 MiB
  - Activations + Engine Code + Descriptor tables: 32 MiB
  - **Minimum required lease:** 848 MiB
  - **Recommended `size_mb`:** **1024 MiB** (64 JTLB slots of 16 MiB).

- **Case 2: Full Multimodal (LFM3-VL-2B) at Q4_0, Host-Assisted Embeddings:**
  - Text Decoder Weights: 797 MiB
  - Vision Tower + Projector Weights: 102 MiB
  - Dynamic KV Cache + Conv State: 18.4 MiB
  - Vision Image Patch Activation Scratch: 32 MiB
  - Activations + Engine Code: 32 MiB
  - **Minimum required lease:** 982 MiB
  - **Recommended `size_mb`:** **1280 MiB** (80 JTLB slots) or **1536 MiB** (96 JTLB slots).

- **Case 3: All-on-DSP (LFM3-VL-2B + Embeddings on CDSP) at Q4_0:**
  - Total Model Weights: 1,039 MiB
  - Total Buffers: ~100 MiB
  - **Minimum required lease:** 1,140 MiB
  - **Recommended `size_mb`:** **1536 MiB** (96 JTLB slots).

---

## 7. Engine Ops (Reserved for Worker 2)

*Worker 2 (DSP kernels) will append engine op implementations, HVX scheduling, and simulator execution evidence here.*
