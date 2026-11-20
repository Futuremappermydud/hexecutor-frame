# HXFG tensor backend

Hexecutor accepts explicitly lowered static graphs through the native `tensor_f32` backend. A `Session` can run up to eight programs under one device lease. Use `run(buffers, program_index)` for direct execution or the [broker](broker.md) for shared scheduling. The learned-depth QDRX/NPRG backend has a separate four-input/two-output ABI. See the [native API](api.md) and [program formats](formats.md).

## Format and execution contract

- HXFG v1 uses a 256-byte header, 32-byte tensor records, 64-byte operation records and ordered I/O tensor IDs. Constants are stored separately and checked by SHA-256. The format limits are 1,024 tensors, 1,024 ops and 16 IDs per input/output list. Float32, int32 and bool tensor types have explicit storage rules. The live engine and host independently preflight shape, extent, alias, scratch, allocation and constant references. [Format and op ABI](../../hexecutor/native/tensor/graph.h)
- Shapes map to logical HWC. Channels are normally padded to 32 elements. The private FP16 plan packs channel blocks into pixel pairs in 32-pixel tiles, pads row width to an even value and keeps the extra column zero. Compact convolution inputs and packed single-channel layouts have dedicated validation and loading rules. [Graph validation](../../hexecutor/native/tensor/graph.c), [FP16 implementation](../../hexecutor/native/tensor/fp16.c)
- The activation arena is capped at 16 MiB per graph. Constants in the standard layout occupy relative private DDR 32–96 MiB. Large pools can use the high region starting at 128 MiB, provided the engine, bootstrap, probe and kernel lease cover the mapping. The native format ceiling is 512 MiB per graph. The generic float ONNX lowerer still uses a 64 MiB export ceiling. Total placement is constrained by the mapped window and the other registered graphs. [Memory layout](memory-layout.md), [Dynamic leases](dynamic-memory.md), [Float ONNX lowerer](../../hexecutor/offline/lower_onnx_float.py)
- Worker zero owns HMX, while four HVX workers split independent outputs. A matching four-worker engine and bootstrap are required. Without a blocked FP16 plan the DSP can use the FP32 path. `HEXECUTOR_TENSOR_F32=1` selects FP32 intermediates with HMX and `HEXECUTOR_TENSOR_PRECISE=1` selects the precise float path in the tensor runner. The portable CPU runner is an explicit comparison mode.

## Lowering and coverage

| Front end | Scope and checks |
| --- | --- |
| [TFLite](../../hexecutor/offline/lower_tflite.py) | Static batch-one float graphs with NHWC axis rules. Supports convolution, pooling, activation/broadcast arithmetic, resize and rearrangement, reductions, runtime batch MatMul, constant-table Gather with int32 indices, ArgMax/ArgMin and typed Cast/Select for supported signatures. [Portable cases](../../hexecutor/tests/test_lower_tflite.py) |
| [Quantized ONNX](../../hexecutor/offline/lower_onnx.py) | Static QLinear graphs. Weights are dequantized offline and output ranges approximate ReLU/ReLU6. Compare against the quantized reference on representative inputs. |
| [Float ONNX](../../hexecutor/offline/lower_onnx_float.py) | Static spatial and token graphs. Supports dense, depthwise and grouped convolution, attention/norm, reductions, and supported indexed reads/writes. Constant folding and inference BatchNorm folding reduce graph size. The decomposer covers examples including stable LogSoftmax, DepthToSpace and SpaceToDepth. The manifest reports FP16 plan blockers and can omit packed weights when the float graph fits the export ceiling. [Decompositions](../../hexecutor/offline/onnx_decompose.py), [Portable cases](../../hexecutor/tests/test_lower_onnx_float.py) |
| [ExecuTorch](../../hexecutor/offline/executorch/README.md) | Partitions supported exported ATen patterns into delegated HXFG regions. Check rank, layout, mask and index constraints per pattern. Partitioning success and whole-model delegation are separate measurements. |
| [QNN inspection](../../hexecutor/offline/inspect_qnn.py) | Offline schematic parsing and compiler configuration. QNN16 context extraction and engine support have a separate contract. [NIR hand workflow](../model-development/nir-hand-tracking.md), [Mercury](../model-development/mercury-hand-tracking.md) |

The compiler and graph validator are the source of truth for exact option coverage. When adding support, test a supported shape and close rejection cases. A shape-compatible higher-rank reshape may alias storage, but a permutation that changes element order needs a real operation. Keep exporter layout decisions consistent with HWC channel strides, especially for tokens and vision patch merging.

## Persistent state and decode

HXFG kinds 22–26 cover RMSNorm, RoPE, causal short convolution, quantized linear and KV writes. Fused SDPA is kind 27. Indexed movement and typed operations follow in the [format header](../../hexecutor/native/tensor/graph.h). The persistent-state allocator uses a dedicated DDR region. `HX_TENSOR_PERSISTENT` marks a tensor and a matching nonzero FNV-1a name hash allows compatible tensors in separate graphs to share an allocation. `HX_KV_WRITE` appends at a fixed or runtime-supplied position. `Session::reset_state()` clears the state. The broker retains state across routine idle gaps and requires a reset if the lease is lost. [Broker lifecycle](broker.md#persistent-state)

## Build and validate

Use `hexecutor/build-scripts/build_tensor.sh` with an explicit SDK and toolchain image for the engine, matching bootstrap and chunk-map probe. Use `make check` for the portable host checks and `hexecutor/build-scripts/test_tensor_sim.sh` for HMX/HVX comparisons against the portable reference. Measure host wall time, DSP cycles, warm runs, graph switches, and wake/close latency independently. `Session::profile()` exposes DSP counters and the selected blocked path. [Validation evidence](validation.md)

## Batched MatMul, attention and higher-rank layouts

The backend supports batched runtime matrix multiplication (`HX_MATMUL`, kind 21), fused scaled dot-product attention (`HX_SDPA`, kind 27), and attention softmax with arbitrary-axis and optional causal/additive mask support (`HX_ATTN_SOFTMAX`, kind 28).

- `HX_MATMUL` (kind 21): Extends runtime matrix multiplication to batched and multi-head layouts (`[B, M, K]` $\times$ `[B, K, N]` or transposed). Op parameters:
  - `p[0]`: `transB` (0: un-transposed, 1: transposed operand $B$, e.g. $Q \cdot K^T$)
  - `p[1]`: float32 bit pattern of scalar factor applied to result (0 selects 1.0f)
  - `p[2]`: batch count $B$ when tensors have $w = 1$ (if 0 or 1, inferred from tensor height $h$)
  - `p[3]`: `transA` (0: un-transposed, 1: transposed operand $A$)
  Inner dimension $K \le 4096$. Broadcast across the batch/head dimension ($B_b = 1$ to $B_a$) is supported. Accumulation is float32.

- `HX_ATTN_SOFTMAX` (kind 28): Attention softmax supporting arbitrary axis reduction and masking:
  - `x`: input tensor, `out`: output tensor (same shape)
  - `y`: optional additive mask tensor ID (or `HXFG_NONE`), supporting arbitrary broadcasting on $h, w, c$
  - `p[0]`: axis (0: $H$, 1: $W$, 2: $C$ / channels, which is the key axis for multi-head scores `[H, Lq, Lk]`)
  - `p[1]`: causal mask flag (1: causal masking, sets $k > q$ to 0.0f; 0: none)
  - `p[2]`: query sequence length $L$ for causal query indexing if $w = 1$ (0 selects default)
  Standard unparameterized channel-axis softmax remains `HX_SOFTMAX` (kind 15).

- `HX_SDPA` (kind 27): Fused scaled dot-product attention ($Out = \text{Softmax}(Q \cdot K^T \cdot \text{scale} + \text{mask}) \cdot V$):
  - `x`: $Q$ (`[B, Lq, D]`), `y`: $K$ (`[B, Lk, D]`), `z`: $V$ (`[B, Lk, Dv]`), `out`: Output (`[B, Lq, Dv]`)
  - `p[0]`: optional additive mask tensor ID (or `HXFG_NONE`)
  - `p[1]`: float32 bits of scale (0 selects $1 / \sqrt{D}$)
  - `p[2]`: causal mask flag (1: causal masking; 0: none)
  - `p[3]`: batch/head count $B$ when tensors have $w = 1$
  - `p[4]`: reserved (0)

- Higher-rank views & Pixel Unshuffle: Rank-5/6 tensors with leading unit dimensions or collapsed spatial dimensions (e.g. `[1, 1, H, L, D]`, `[1, 1, 1, 2, 2, 1024]`) map into the generic 3D HWC representation without requiring an interpreter or hybrid fallback. Rank-6 pixel unshuffle reshapes and transposes (`conv.merge_unshuffle`) alias cleanly as zero-copy views when physical memory and channel strides match.

- FP16/HVX path behavior: `HX_MATMUL` uses vector lanes for batch-one width-one matrices when a per-worker workspace slice holds its converted operands. `HX_ATTN_SOFTMAX` uses the channel Softmax vector kernel for channel-axis rows, including causal and additive-mask cases. `HX_SDPA` uses qf32 vector dot products and value accumulation for batch-one width-one Q/K/V with no additive mask when its workspace slice holds scores and a query row; causal masking is supported. Other shapes and workspace cases retain scalar lanes. The full FP16 simulator suite compares these paths with the portable reference; See the validation guide for the shapes covered by earlier device runs.

The current FP16 vector coverage also includes simple standalone activations (ReLU, ReLU6, hard-swish, hard-sigmoid, leaky ReLU, abs and negate), SiLU and logistic; channel and group normalization; spatial reductions that preserve channels; channel-aligned slices; and generic depthwise shapes. General im2col packing remains scalar except for 3×3 stride-1 pad-1 convolutions of matching even width with channels in groups of 32, a guarded 3×3 stride-1 blocked-row gather for whole 32-channel blocks and even output width (including pad 2), and unpadded 1×1 stride-2 convolutions whose input width is a multiple of four and channels are in groups of 32. These guards preserve the blocked pixel-pair layout and its zero pad column. The remaining scalar paths cover other activation functions, reductions over channels, misaligned slices, attention softmax over spatial axes, masked or wider SDPA, and general im2col shapes. The simulator compares vector output against the portable float reference with FP16 tolerance; host-only syntax checks do not establish CDSP execution.
