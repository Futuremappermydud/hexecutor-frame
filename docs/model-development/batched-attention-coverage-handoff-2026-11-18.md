# Batched MatMul, Attention and Higher-Rank Layouts in Hexecutor, Handoff 2026-11-18

Branch: `Futuremapper/hx-attention` (based on committed `HEAD`).
Durable contract and ABI details are documented in [tensor-backend](../hexecutor/tensor-backend.md). This file summarizes the changes, test results, op numbering, and handoff state for merge.

## Why

Enable ViT vision transformers, causal decoder attention blocks, and higher-rank pixel unshuffle natively in the Hexecutor HXFG tensor backend (`tensor_f32`) and ExecuTorch AOT lowering without requiring an interpreter fallback or decomposing into single-head 2D slices.

## Op Kind Allocations (Collision-Free Merge)

To avoid collisions with the coordinator's branch (which reserves kinds 22–26 for `RMSNORM`, `ROPE`, `SHORTCONV`, `LINEAR_Q`, `KV_WRITE`), op numbers were assigned as follows:
- **`HX_MATMUL = 21`** (extended): Runtime matrix multiplication extended with batch count ($p[2]$), transposed operand A ($p[3]$), transposed operand B ($p[0]$), and scalar factor folding ($p[1]$).
- **`HX_SDPA = 27`**: Fused scaled dot-product attention ($Out = \text{Softmax}(Q \cdot K^T \cdot \text{scale} + \text{mask}) \cdot V$) with operands $x=Q$, $y=K$, $z=V$, optional additive mask tensor $p[0]$, scale factor bits $p[1]$, causal mask flag $p[2]$, and batch count $p[3]$.
- **`HX_ATTN_SOFTMAX = 28`**: Attention softmax over arbitrary axis $p[0]$ (0: $H$, 1: $W$, 2: $C$/key axis), causal mask flag $p[1]$, query sequence length $p[2]$, and optional additive mask tensor $y$.
- **`HX_SOFTMAX = 15`**: Retained strictly unparameterized (channel-only, 0 params) for backward compatibility with baseline regression tests.

## What Changed

### 1. DSP & Native Portable Backend (`hexecutor/native/tensor/`)
- `graph.h`:
  - Added op kinds `HX_SDPA = 27`, `HX_ATTN_SOFTMAX = 28`.
  - Documented ABI parameters and operands for `HX_MATMUL`, `HX_SDPA`, and `HX_ATTN_SOFTMAX`.
- `graph.c`:
  - Added `matmul_dims` validator supporting multi-head $w > 1$ and token $w = 1$ layouts, $transA$, $transB$, head broadcast ($B_b = 1$ to $B_a$), and inner dimension $K \le 4096$.
  - Added `sdpa_dims` validator for fused SDPA checking $Q, K, V$ dimensions and scratch requirements.
  - Implemented `HX_MATMUL` execution with accumulation in float32, supporting all four transpose combinations ($!transA \ \& \ !transB$, $!transA \ \& \ transB$, $transA \ \& \ !transB$, $transA \ \& \ transB$) and batch dimension indexing.
  - Implemented `HX_ATTN_SOFTMAX` execution with arbitrary axis reduction (H=0, W=1, C=2), causal masking ($k > q$), query indexing, and additive broadcast mask $y$.
  - Implemented `HX_SDPA` execution using stack scratch buffer for stable max-subtracted softmax and matrix multiplication with $V$.
- `fp16.c`:
  - Added scalar-lane fallbacks `matmul_fp16_scalar`, `softmax_fp16_scalar`, and `sdpa_fp16_scalar` in `hx_fp16_op` for the FP16 plan.
- `hexecutor/host/cpp/tensor_run.cpp`:
  - Extended CLI to accept variable input count $n_{in} \ge 1$: `hexecutor_tensor_run IMAGE CONSTANTS INPUT_F32... OUTPUT_PREFIX [LOADER ENGINE BOOTSTRAP CHUNK_PROBE]`.
- `hexecutor/tests/test_tensor_graph.cpp`:
  - Added comprehensive C++ unit test suites covering unbatched MatMul, batched multi-head MatMul, masked/causal Softmax, and fused SDPA.

### 2. HXFG Serialization (`hexecutor/offline/hxfg_writer.py`)
- Defined op enums: `MATMUL = 21`, `RMSNORM = 22`, `SDPA = 27`, `ATTN_SOFTMAX = 28`.
- Updated `tensor_refs(op)` and `write(...)` to register explicit $z$ operand and mask parameter $p[0]$ for `SDPA`.

### 3. ONNX Lowering (`hexecutor/offline/lower_onnx_float.py`)
- **Batched MatMul**: `_emit_batched_matmul()` handles runtime operands in multi-head layout `[1, H, M, K]` $\times$ `[1, H, K, N]` (or transposed), emitting `HX_MATMUL` with batch count $H$.
- **Scale Folding**: Scalar Mul/Div following a MatMul is folded directly into `p[1]` of `HX_MATMUL`.
- **Einsum**: `op_Einsum()` parses 2-operand contractions and matches common attention patterns (`"bhqd,bhkd->bhqk"` for $Q \cdot K^T$ and `"bhqk,bhkd->bhqd"` for $attn \cdot V$, as well as single-head `"bqd,bkd->bqk"` / `"bqk,bkd->bqd"`).
- **Arbitrary-Axis Softmax**: `op_Softmax()` maps ONNX axis to HWC reduction axis (emitting `HX_ATTN_SOFTMAX` for key axis 3/-1 or spatial axes, and keeping `HX_SOFTMAX` for standard NCHW channel axis 1).
- **Causal Mask Detection**: Detects upper-triangular causal masks in `Add` operations preceding Softmax, fusing them into `HX_ATTN_SOFTMAX` with causal flag $p[1]=1$.
- **Higher-Rank & Pixel Unshuffle Layouts**: `hwc()` supports rank-5 and rank-6 tensors with collapsed spatial/head dimensions. `_alias_shape()` aliases rank-6 unshuffle reshapes and transposes (`conv.merge_unshuffle`) as zero-copy aliases when element counts and channels match.
- **SDPA**: Added `op_ScaledDotProductAttention` lowering to `HX_SDPA` (kind 27).

### 4. ExecuTorch Lowering (`hexecutor/offline/executorch/`)
- `partitioner.py`: Kept `torch.ops.aten.bmm.default` and `torch.ops.aten.scaled_dot_product_attention.default` from decomposition.
- `lowering.py`: Added handlers for `bmm.default` (`_emit_bmm`) and `scaled_dot_product_attention.default` (`_emit_sdpa`), mapping token/batch shapes to HWC.

## Verification & Tests Passed

1. **Host C++ unit tests (`build/test_tensor_graph`)**:
   - `test_tensor_graph`: Passed (kernels, padding, layout, aliases, ranges, batched MatMul, masked Softmax, SDPA).
   - `ctest --test-dir build --output-on-failure`: 13/13 passed (100%).
2. **ONNX Lowering & Reference Tests (`hexecutor/tests/test_lower_onnx_float.py`)**:
   - 51/51 unit tests passed against `onnxruntime` and `hexecutor_tensor_run` reference:
     - `test_batched_matmul_direct_matches_onnxruntime` (PASS)
     - `test_einsum_attention_matches_onnxruntime` (PASS)
     - `test_vit_batched_attention_matches_onnxruntime` (PASS)
     - `test_causal_decoder_attention_matches_onnxruntime` (PASS)
     - `test_rank6_pixel_unshuffle_matches_onnxruntime` (PASS)
     - `test_arbitrary_axis_softmax_matches_onnxruntime` (PASS)
     - `test_sdpa_op_lowering_and_fields` (PASS)
3. **ExecuTorch AOT Tests (`hexecutor/tests/test_executorch_aot.py`)**:
   - 26/26 tests passed, including `test_bmm_is_fully_delegated` and `test_sdpa_is_fully_delegated`.
4. **Hexagon Simulator (`hexecutor/build-scripts/test_tensor_sim.sh`)**:
   - Simulator tests compiled with Hexagon-clang `v75 -mhmx -mhvx` and ran on `hexagon-sim -mv75`:
     - `test_hmx.elf`: PASS fp16 HMX deep-load boundaries K=32,1024,1056,1152,4096.
     - `test_kernels.elf`: PASS fp16 conversion boundaries and RGB gather/pad.
     - `test_fp16.elf`: PASS blocked FP16 execution including `graph.c` and `fp16.c`.

## Known Gaps & Future Work

1. **HVX Vectorization for FP16 MatMul/SDPA**: The FP16 execution path for `HX_MATMUL`, `HX_ATTN_SOFTMAX`, and `HX_SDPA` in `fp16.c` currently executes in scalar lanes (using float32 accumulator). It is correct across all shapes, but can be vectorized with HVX vector instructions (`Vsf`/`Vqf32`) for higher throughput on the DSP.
2. **HMX Integration for Batched MatMul**: When batch count $B > 1$ and matrices fit HMX tile limits, dispatching individual head batches to HMX will provide peak compute throughput on the NPU/DSP.
