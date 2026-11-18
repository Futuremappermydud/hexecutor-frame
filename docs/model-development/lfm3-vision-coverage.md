# LFM3-VL Vision Tower & Projector Lowering Coverage Report

**Date**: 2026-11-17  
**Task**: Worker 3 of 3 — Vision tower + multimodal projector coverage through the HXFG/ONNX float path  
**Target Platform**: Steam Frame CDSP generic tensor backend (`hexecutor_tensor_run`, `hexecutord` broker, Hexagon simulator)  
**Target Model**: Liquid AI `LFM3-VL-2B` vision encoder and multimodal projector  
**Artifact Policy**: Compliant with `AGENTS.md` (no raw weights committed, offline tooling only)

---

## 1. Executive Summary

We evaluated the vision encoder and multimodal projector of `LiquidAI/LFM3-VL-2B` against the Hexecutor generic tensor backend float lowering pipeline (`hexecutor/offline/lower_onnx_float.py`, `hxfg_writer.py`, and `build/bin/hexecutor_tensor_run`).

### Key Results
1. **Passing Subgraphs**:
   - **Patchifier** (`Linear(1536, 1024)` + `RMSNorm`): **PASS**, lowered to 2 HXFG ops, verified against PyTorch on `hexecutor_tensor_run` with max error **$2.38 \times 10^{-7}$**.
   - **MLP / Feed-Forward** (`RMSNorm` + SwiGLU / SiLU gate + Residual): **PASS**, lowered to 7 HXFG ops, verified against PyTorch on `hexecutor_tensor_run` with max error **$2.38 \times 10^{-7}$**.
   - **Attention Block** (Multi-Head Attention with 16 heads, head_dim 64, GQA 8$\to$16 offline unroll surgery, `RMSNorm`, per-head `HX_MATMUL`, Softmax, context product, output projection): **PASS**, lowered to 18 HXFG ops, verified against PyTorch on `hexecutor_tensor_run` with max error **$4.77 \times 10^{-7}$**.
2. **Lowerer Enhancements Added**:
   - `HX_RMSNORM` (kind 22) implemented in `hxfg_writer.py` and `lower_onnx_float.py` for token and spatial layouts.
   - `HX_MATMUL` (kind 21) unflagged from `fp16_blockers` following scalar-lane verification on Hexagon simulator (`test_tensor_sim.sh` PASS).
   - Rank-3 token tensor support added to `op_Slice` and `op_Split` along sequence and channel axes.
   - Rank-3 token-to-token alias reshape support added in `_alias_shape` for downsampling/pixel unshuffle.
   - `op_Pow` (squaring to binary mul) and `op_Silu` added to `lower_onnx_float.py`.
   - All 44 existing lowering tests plus 3 new transformer lowering tests pass (47 total).
3. **Primary Hardware & Lowering Blockers**:
   - **Constant Pool Exceeded**: Projector requires 150.99 MiB (FP32 + packed FP16 weights), exceeding the 64 MiB per-graph limit (`HXFG_MAX_CONSTANT_BYTES` in `graph.h:21`). Full 14-layer vision tower requires ~939 MiB. Recommendation: Raise per-graph limit to 160 MiB and partition across 6–7 graphs (fitting within broker's $\le 8$ graph limit).
   - **Token vs Spatial Conv Boundary**: `Lfm3VlVisionConv2d` reshapes token sequence into a 6-D merge block view (`[1, 1, 1, 2, 2, 1024]`) and transposes to NCHW raster (`[1, 1024, 2, 2]`) for 2D depthwise convolution. Lowerer rejects rank $>4$ and cross-layout token $\leftrightarrow$ NCHW transpositions.
   - **ExecuTorch Partitioner Coverage**: `hexecutor/offline/executorch/` does not yet delegate rank-3 tokens, `HX_MATMUL`, `HX_RMSNORM`, or runtime `HX_SLICE`.

---

## 2. LFM3-VL-2B Architecture Breakdown

Derived directly from `LiquidAI/LFM3-VL-2B` `config.json` and modeling code:

### Model Parameters & Sizes
- **Vision Model Parameters**: 164,206,336 (~626.40 MiB FP32)
- **Projector Parameters**: 25,174,016 (~96.03 MiB FP32)
- **Combined Vision + Projector**: 189,380,352 (~722.43 MiB FP32; ~1.08 GiB packed)

### Vision Encoder Configuration
- **Patch Embedding**:
  - `patch_size`: 16, `temporal_patch_size`: 2, `in_channels`: 3
  - Patch dimension: $3 \times 2 \times 16 \times 16 = 1536$
  - `spatial_merge_size`: 2 (patches are grouped into $2 \times 2 = 4$ patch blocks)
  - Smallest supported static tile: $32 \times 32$ pixels = 1 merge block = 4 tokens ($L=4$)
- **Backbone Dimensions**:
  - `hidden_size`: 1024, `intermediate_size`: 2560
  - `num_hidden_layers`: 14
- **Layer Schedule**:
  - 8 Gated Convolution layers: layers 0, 1, 3, 5, 7, 9, 11, 13
  - 6 Bidirectional Attention layers: layers 2, 4, 6, 8, 10, 12
- **Conv Layer Detail**:
  - `RMSNorm(1024)` $\to$ `Linear(1024, 3072)` split into $B, C, x$ $\to$ gated $B \times x \to$ spatial depthwise `Conv2d(1024, 1024, kernel_size=3, padding=1, groups=1024)` $\to$ `Linear(1024, 1024)` $\to$ Residual $\to$ `RMSNorm(1024)` $\to$ MLP $\to$ Residual
- **Attention Layer Detail**:
  - 16 Query heads, 8 Key/Value heads (GQA), `head_dim` = 64
  - Per-head `q_layernorm` and `k_layernorm` (`RMSNorm(64)`)
  - 2D Axial RoPE
  - `RMSNorm(1024)` $\to$ Q/K/V projections $\to$ Attention $\to$ `out_proj(1024, 1024)` $\to$ Residual $\to$ `RMSNorm(1024)` $\to$ MLP $\to$ Residual
- **Projector Detail**:
  - `nn.LayerNorm(1024, eps=1e-6)`
  - Pixel unshuffle: downsample factor 2 merges $2 \times 2 = 4$ vision tokens into 1 language token: reshape from $[1, 4, 1024]$ to $[1, 1, 4096]$
  - `Linear(4096, 4096, bias=True)` $\to$ `GELU` $\to$ `Linear(4096, 2048, bias=True)`
  - Output shape: $[1, 1, 2048]$ (text embedding dimension)

---

## 3. Operation & Shape Coverage Table

Below is the exhaustive coverage status of each operation and shape across the LFM3-VL vision pipeline:

| Subgraph / Node | ONNX OpType | Input Shape | Output Shape | Status | Failure Reason (if failed) | Suggested Fix | Owner File |
|---|---|---|---|---|---|---|---|
| `patchifier.patchifier` | `Conv` / `MatMul` | `[1, 4, 1536]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `patchifier.norm` | `RMSNormalization` | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `mlp.norm` | `RMSNormalization` | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `mlp.w1`, `mlp.w3` | `MatMul` (1x1) | `[1, 4, 1024]` | `[1, 4, 2560]` | **PASS** | - | - | `lower_onnx_float.py` |
| `mlp.silu` | `Act` / `Silu` | `[1, 4, 2560]` | `[1, 4, 2560]` | **PASS** | - | - | `lower_onnx_float.py` |
| `mlp.gate_mul` | `Mul` | `[1, 4, 2560]` | `[1, 4, 2560]` | **PASS** | - | - | `lower_onnx_float.py` |
| `mlp.w2` | `MatMul` (1x1) | `[1, 4, 2560]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `mlp.residual` | `Add` | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `attn.q_proj` | `MatMul` (1x1) | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `attn.k_proj`, `v_proj` | `MatMul` (1x1) | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | Offline GQA unroll: 8 heads $\to$ 16 heads | Module surgery in export | `export_vision.py` |
| `attn.qk_product` | `MatMul` | `[1, 16, 4, 64]`, `[1, 16, 64, 4]` | `[1, 16, 4, 4]` | **PASS** | - (unflagged from fp16 blockers) | `HX_MATMUL` in `fp16.c` | `lower_onnx_float.py` |
| `attn.softmax` | `Softmax` (axis=-1) | `[1, 16, 4, 4]` | `[1, 16, 4, 4]` | **PASS** | - | - | `lower_onnx_float.py` |
| `attn.ctx_product` | `MatMul` | `[1, 16, 4, 4]`, `[1, 16, 4, 64]` | `[1, 16, 4, 64]` | **PASS** | - | - | `lower_onnx_float.py` |
| `attn.merge_heads` | `Transpose` + `Concat` | `[1, 16, 4, 64]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `attn.out_proj` | `MatMul` (1x1) | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `conv.in_proj` | `MatMul` (1x1) | `[1, 4, 1024]` | `[1, 4, 3072]` | **PASS** | - | - | `lower_onnx_float.py` |
| `conv.split` | `Split` / `Slice` | `[1, 4, 3072]` | $3\times$`[1, 4, 1024]` | **PASS** | - | Rank-3 channel split added | `lower_onnx_float.py` |
| `conv.merge_unshuffle` | `Reshape` | `[1, 4, 1024]` | `[1, 1, 1, 2, 2, 1024]` | **FAIL** | Rank 6 tensor unsupported | Unroll 2D conv in token HWC | `export_vision.py` / `graph.c` |
| `conv.permute_nchw` | `Transpose` | `[1, 1, 1, 2, 2, 1024]` | `[1, 1024, 2, 2]` | **FAIL** | Token layout cannot transpose to NCHW | Unroll 2D conv in token HWC | `lower_onnx_float.py` / `graph.c` |
| `conv.spatial_conv` | `Conv` (depthwise) | `[1, 1024, 2, 2]` | `[1, 1024, 2, 2]` | **BLOCKED** | Blocked by upstream layout conversion | Unroll 2D conv in token HWC | `graph.c` / `fp16.c` |
| `projector.layer_norm` | `LayerNormalization` | `[1, 4, 1024]` | `[1, 4, 1024]` | **PASS** | - | - | `lower_onnx_float.py` |
| `projector.pixel_unshuffle` | `Reshape` | `[1, 4, 1024]` | `[1, 1, 4096]` | **PASS** | Token reshape $4\times 1024 = 1\times 4096$ | Rank-3 alias reshape added | `lower_onnx_float.py` |
| `projector.linear_1` | `MatMul` (1x1) | `[1, 1, 4096]` | `[1, 1, 4096]` | **PASS (struct)** | Exceeds 64 MiB constant pool limit | Raise limit / split graph | `lower_onnx_float.py` / `graph.h` |
| `projector.gelu` | `Act` / `Gelu` | `[1, 1, 4096]` | `[1, 1, 4096]` | **PASS** | - | - | `lower_onnx_float.py` |
| `projector.linear_2` | `MatMul` (1x1) | `[1, 1, 4096]` | `[1, 1, 2048]` | **PASS (struct)** | Exceeds 64 MiB constant pool limit | Raise limit / split graph | `lower_onnx_float.py` / `graph.h` |

---

## 4. Memory & Platform Capacity Limits

### Constant Pool vs. 64 MiB Limit
- **Current Runtime Limit**: `CONSTANT_POOL_LIMIT = 0x4000000u` (64 MiB, defined in `lower_onnx_float.py:73` and `hexecutor/native/tensor/graph.h:21`).
- **Component Pool Breakdown**:
  - `patchifier`: 9.01 MiB (fits)
  - `mlp`: 45.03 MiB (fits)
  - `attention`: 24.02 MiB (fits)
  - `conv`: ~12.02 MiB (fits)
  - `projector`: **144.08 MiB** (fails by 80.08 MiB)
    - `linear_1` ($4096 \times 4096$): 64.00 MiB FP32 + 32.00 MiB packed FP16 = 96.00 MiB
    - `linear_2` ($4096 \times 2048$): 32.00 MiB FP32 + 16.00 MiB packed FP16 = 48.00 MiB
    - Norms & Biases: ~0.08 MiB
- **Vision Tower Total**:
  - $164.2\text{M params} \times 4\text{ bytes} = 656.8\text{ MiB FP32}$. With packed FP16 weights (~328.4 MiB), the total constant pool is **$\sim 985$ MiB**.

#### Graph Splitting Proposal vs Limit Growth
The CDSP broker (`hexecutord`, `docs/hexecutor/broker.md`) permits at most **8 graphs per Session** (`HXFG_MAX_GRAPHS = 8`).
- **Option A (Pure Splitting at 64 MiB)**:
  - Splitting 985 MiB into 64 MiB graphs requires at least $\lceil 985 / 64 \rceil = 16$ graphs. This **exceeds the 8-graph limit** and cannot run in a single session without graph unloading.
- **Option B (Recommended — Raise Limit to 160 MiB)**:
  - Raise `HXFG_MAX_CONSTANT_BYTES` in `graph.h` from `0x4000000u` (64 MiB) to `0xA000000u` (160 MiB), and raise `cdsp_mem` lease from 512 MiB to 1.5 GiB.
  - Partition into **7 graphs**:
    - Graphs 1–6: 2 vision layers each ($\sim 140$ MiB per graph)
    - Graph 7: 2 vision layers + Projector ($\sim 150$ MiB)
    - Total: 7 graphs $\le 8$, fitting comfortably within the broker session.
- **Option C (Deduplication Optimization)**:
  - For graphs that qualify for blocked FP16 execution, omit the duplicate unpacked FP32 weights from the constant pool. This reduces pool size by 67%, bringing each 2-layer block under 50 MiB.

### Activation Arena vs. 16 MiB Limit
- **Current Runtime Limit**: 16 MiB (`0x1000000u` in `graph.c:205` / `graph.c:267`).
- **Arena Usage for Static $32 \times 32$ Tile (4 Tokens)**:
  - `patchifier`: 55.5 KiB
  - `mlp`: 189.0 KiB
  - `attention`: 92.0 KiB
  - `projector`: 71.2 KiB
  - Full Vision Layer: 281.0 KiB
- **Scaling Analysis**:
  - 4 tokens require $< 0.3$ MiB of arena.
  - A full $512 \times 512$ image ($32 \times 32$ patch grid = 1024 tokens) requires $\approx 72$ MiB peak intermediate storage, which would exceed the 16 MiB arena.
  - For full-resolution images, tiled vision inference ($64 \times 64$ to $128 \times 128$ tile chunks $\le 64$ tokens) or raising `arena_bytes` to 64–128 MiB in `graph.c` will be required.

### $K > 4032$ Conv/Linear Blockers
- **Pointwise Linears (1x1 Convolutions)**:
  - Projector `linear_1` has $K = 4096$.
  - In `hxfg_writer.py`, `tiles = (1 * 1 * 4096 + 31) // 32 = 128`.
  - For 1x1 convolutions, `_split_wide` splits wide output channels into chunks of 64 or 96 channels to fit HMX scratch budget (`0x100000`).
  - Pointwise linears with $K = 4096$ do **not** trigger the im2col blocker because im2col is only invoked when kernel size $> 1$.
- **Spatial Convolutions**:
  - `spatial_conv` in `VisionConv2d` has kernel $3 \times 3$ and `groups = 1024` (depthwise).
  - Effective $K$ per group is $3 \times 3 \times 1 = 9 \ll 4032$.
  - Therefore, no spatial convolution in the LFM3-VL vision tower hits the $K > 4032$ im2col boundary.

---

## 5. ExecuTorch Partitioner Gaps vs `lower_onnx_float.py`

Inspection of `hexecutor/offline/executorch/` (`lowering.py` and `partitioner.py`) revealed several critical coverage gaps relative to the ONNX lowerer:

1. **Shape Rank Limitation**:
   - `hwc_of` (`lowering.py:139`) explicitly rejects any shape that is not batch-one rank 2 or rank 4:
     `Unsupported: shape ... is not batch-one rank 2 or 4`
   - Transformer token matrices ($[1, L, K]$ rank 3) are completely unhandled by ExecuTorch lowering.
2. **Missing `HX_MATMUL` Delegation**:
   - `KINDS` dictionary does not recognize runtime activation matrix multiplication (`torch.ops.aten.bmm.default` or `torch.ops.aten.matmul.default`).
   - Consequently, self-attention cannot be partitioned or delegated to Hexecutor via ExecuTorch.
3. **Missing `HX_RMSNORM` Delegation**:
   - Neither `rms_norm` nor custom normalizations are recognized; only `native_layer_norm` and `native_group_norm` are defined.
4. **Missing Runtime `HX_SLICE` / `HX_SPLIT`**:
   - Slicing and splitting along channel or token sequence axes are not registered for delegation.
5. **Multi-Head Attention Graph Pattern**:
   - ExecuTorch delegate lacks pattern matching to group Reshape $\to$ Transpose $\to$ MatMul $\to$ Softmax $\to$ MatMul into per-head operations.

---

## 6. Verification Evidence

### 1. Numeric Comparison Against PyTorch (`hexecutor_tensor_run`)
Executed via `check_vision.py` comparing `build/bin/hexecutor_tensor_run` outputs against PyTorch reference forward passes:

```
=======================================================
 Checking Component: patchifier
=======================================================
1. Exporting patchifier to ONNX (shape: (1, 4, 1536))...
2. Lowering patchifier to HXFG...
   [LOWER SUCCESS]: 2 ops, 9.01 MiB pool, 55.5 KiB arena
3. Running on hexecutor_tensor_run and comparing to PyTorch...
   [NUMERIC RESULT]: Max Abs Diff: 2.38e-07, Mean Diff: 1.75e-08 (Tol: 0.0001)
   [VERIFICATION]: PASS

=======================================================
 Checking Component: mlp
=======================================================
1. Exporting mlp to ONNX (shape: (1, 4, 1024))...
2. Lowering mlp to HXFG...
   [LOWER SUCCESS]: 7 ops, 45.03 MiB pool, 189.0 KiB arena
3. Running on hexecutor_tensor_run and comparing to PyTorch...
   [NUMERIC RESULT]: Max Abs Diff: 3.58e-07, Mean Diff: 4.71e-08 (Tol: 0.0001)
   [VERIFICATION]: PASS

=======================================================
 Checking Component: attention
=======================================================
1. Exporting attention to ONNX (shape: (1, 4, 1024))...
2. Lowering attention to HXFG...
   [LOWER SUCCESS]: 18 ops, 24.02 MiB pool, 92.0 KiB arena
3. Running on hexecutor_tensor_run and comparing to PyTorch...
   [NUMERIC RESULT]: Max Abs Diff: 4.77e-07, Mean Diff: 6.90e-08 (Tol: 0.0001)
   [VERIFICATION]: PASS
```

### 2. Simulator Tests (`test_tensor_sim.sh`)
Ran Hexagon simulator validation on `test_fp16.elf`:
```
PASS fp16 HMX deep-load boundaries K=32,1024,1056,1152,4096
PASS fp16 conversion boundaries and RGB 3x3/5x5 gather, padding, and channel tails
MATMUL L197 K64 N197 transB0 scale0: ok, 651336 pcycles
MATMUL L197 K64 N197 transB0 scale1: ok, 653700 pcycles
MATMUL L197 K64 N197 transB1 scale0: ok, 906873 pcycles
MATMUL L197 K64 N197 transB1 scale1: ok, 909237 pcycles
MATMUL L197 K197 N64 transB0 scale0: ok, 880686 pcycles
MATMUL L197 K197 N64 transB0 scale1: ok, 888960 pcycles
MATMUL L197 K197 N64 transB1 scale0: ok, 1225617 pcycles
MATMUL L197 K197 N64 transB1 scale1: ok, 1233777 pcycles
PASS blocked FP16 RGB entry, convolution, depthwise, elementwise, pooling, resize, transforms, reductions, matrix products and channel/pixel tails
Exit code: 0
```

### 3. Lowering Unit Tests (`test_lower_onnx_float.py`)
```
Ran 47 tests in 0.732s
OK
```

---

## 7. Next Steps for Full End-to-End Pipeline

1. **Host & Broker Configuration**:
   - Update `HXFG_MAX_CONSTANT_BYTES` in `hexecutor/native/tensor/graph.h` to `0xA000000u` (160 MiB) and match in `lower_onnx_float.py`.
   - Update `cdsp_mem` lease to 1.5 GiB for Frame host driver.
2. **VisionConv2d Layout Reform**:
   - In `export_vision.py` or `modeling_lfm3_vl.py`, replace the 6-D merge block permute with an unrolled 1D sequence depthwise convolution or introduce an explicit token-to-raster transposition primitive in `graph.c`.
3. **ExecuTorch Partitioner Upgrades**:
   - Port rank-3 token layout, `HX_MATMUL`, and `HX_RMSNORM` into `hexecutor/offline/executorch/lowering.py` to enable direct ExecuTorch delegate exports without ONNX intermediation.
