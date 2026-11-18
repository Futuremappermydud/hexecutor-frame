# Generic tensor execution

The `tensor_f32` backend accepts static batch-one HXFG graphs with variable input/output counts. Multiple graphs share one `Session`; `run(buffers, program_index)` switches graphs while retaining the verified device lease. Neural-model semantics belong to the calling component. The live host and kernels do not import model tooling or launch Python.

QDRX/NPRG remains the existing prepared backend with its four-input/two-output ABI. Its depth package and generated schedules are unchanged. HXFG does not imply support for arbitrary TFLite or QNN contexts.

## Contracts and preflight

`load_tensor_program(image, constants)` checks the HXFG structure, constant SHA-256 and ordered float32 external I/O. `Session` independently reads/preflights its own immutable snapshots before device access. The DSP validates the same image and private allocation contract before dispatch.

HXFG v1 has a 256-byte header, 32-byte tensor records, 64-byte operation records and an ordered tensor-ID I/O table. It stores metadata and constant offsets; weights live in a separate pool. Channels are padded to 32 float32 values. Compact RGB is permitted at convolution inputs. Convolution and depthwise ops may dilate their window through the z field, `(dh << 8) | dw` with 1 to 8 per axis (other ops require z to be NONE). Shapes, sizes, offsets, unused fields, constants, producer ordering, live aliases and per-op scratch limits are checked. Unsupported operations reject the graph; no older scratch fallback is used.

Supported families are convolution/fully connected, multiplier-one depthwise convolution, equal-shape add, channel PReLU, max pooling, spatial/channel padding, bilinear resize, logical reshape, two-input concatenation, spatial mean and logistic, plus three added 2026-11-17: standalone activations (`HX_ACT`: ReLU, ReLU6, hard-swish, hard-sigmoid with alpha/beta, SiLU, tanh, leaky ReLU, exact GELU), broadcasting binary arithmetic (`HX_BINARY`: add, sub, mul, reversible, with a tensor operand whose h/w/c are each the output's or one, or a per-channel constant) and average pooling (`HX_AVGPOOL`, padding counted or not). `HX_SLICE` (added 2026-11-17) copies a channel/spatial sub-box from offsets p[0..2]; it runs in scalar lanes on both the portable and FP16 paths. The float ONNX lowering uses it for Slice, Split and grouped convolution (one Slice, dense Conv and channel Concat per group). Same-shape tensor adds keep the original `HX_ADD` and its fused ReLU. Scalar activation semantics live in one function, `hx_activation` (graph.c), used by the portable path, the FP32 DSP path and the FP16 DSP path. The standalone `HX_ACT` functions also include exp, sqrt (NaN for negative inputs), abs, negate and erf (`HX_ACT_EXP`, `HX_ACT_SQRT`, `HX_ACT_ABS`, `HX_ACT_NEG`, `HX_ACT_ERF`). The FP16 plan runs the new operations in its scalar-lane path (the same one the logistic uses), not in HVX: they are correct but slow on large tensors, and the FP16 code is checked only by syntax on the host. It has not been built for or run on the DSP or the simulator. Softmax over the channel axis (`HX_SOFTMAX`: rank-2 `[1, C]` or rank-4 axis 1, per pixel, max-subtracted for stability) is lowered from ONNX `Softmax` and ExecuTorch `softmax`, and runs on the portable and FP16 scalar-lane paths, but the FP16 path has not been run on the simulator or device. Nearest resize (`HX_RESIZE` with `p[2]` set to 1, asymmetric floor mapping) runs in the same scalar-lane FP16 path, and both the ONNX and ExecuTorch front ends lower it. `HX_ZEROINSERT` (stride and leading pad) inserts zeros between source pixels, and ConvTranspose lowers to it followed by a dense stride-1 convolution with the flipped, channel-swapped kernel (both front ends, groups 1 and dilation 1 only). `HX_REDUCE` (`p[0]` mean, sum, max or min; `p[1]` a non-zero h/w/c bitmask) reduces any non-batch subset with keepdims layout and runs in the portable and FP16 scalar lanes; the spatial mean stays `HX_MEAN`. `HX_NORM` (channel mode, or group mode with `p[1]` groups over all pixels, float32 per-channel gamma and beta) is lowered from ONNX LayerNormalization, GroupNormalization and InstanceNormalization and from ExecuTorch `native_layer_norm` and `native_group_norm`; it runs in the portable and FP16 scalar lanes, and the FP16 path has not been run on the simulator or device. `HX_MATMUL` (kind 21) multiplies two runtime tensors, `x` = A `[L,K]` and `y` = B, into `[L,N]`: `p[0]` 0 reads B as `[K,N]`, 1 as `[N,K]` (so `Q.K^T`); `p[1]` is an optional float32 scale (0 means 1.0). Tokens are ordinary HWC rows (`h = L`, `w = 1`, `c = K`), with `K <= 4096`. It runs in the portable reference and on the FP16 path in HVX lanes (B converted once per part to float blocks, qf32 accumulate, with scalar-lane and per-element fallbacks when the workspace slice is too small); the simulator takes about 0.65 M cycles for L 197, K 64, N 197 (B as [K,N]) against 171 M for the old per-element loop, and the path is not yet on HMX or the device. ExecuTorch `instance_norm` is not lowered, because its edge graph is a batch norm over reshaped data. The offline TFLite lowering rejects unsupported dilation, broadcasting, dynamic shapes, quantized graphs and option variants. It reuses dead activation allocations and precomputes float/FP16 convolution weight layouts. See [the format header](../../hexecutor/native/tensor/graph.h) and [lowering tool](../../hexecutor/offline/lower_tflite.py). [`lower_onnx.py`](../../hexecutor/offline/lower_onnx.py) lowers static QLinear ONNX graphs (onnxruntime's static quantizer output) the same way: it dequantizes weights offline, infers ReLU/ReLU6/linear from each quantized output range, and records those ranges in the manifest. The float graph therefore approximates the u8 arithmetic rather than reproducing it. It splits a wide final Gemm into several outputs so every convolution keeps HMX weights within the per-op budget, as the FP16 plan requires. [`lower_onnx_float.py`](../../hexecutor/offline/lower_onnx_float.py) lowers ordinary static float32 ONNX (batch one, rank 2 or 4), for example `torch.onnx.export` output: `python hexecutor/offline/lower_onnx_float.py model.onnx build/model`. It folds inference BatchNormalization, and per-channel Mul/Add/Sub/Div by constants, into a preceding single-user Conv or Gemm/MatMul, so those add no operations; a BatchNorm that cannot fold becomes a per-channel multiply and add. Before lowering it folds exporter hygiene: constant subexpressions (Identity-wrapped initializers, Shape/Gather/Concat size chains, scalar arithmetic) become initializers and unread nodes are dropped; a constant pool above the runtime's 32 MiB limit is rejected at lowering time. It rewrites Sigmoid-times-input to SiLU, fuses Relu/Clip(0, 6) into single-user conv or same-shape add producers, chains N-input Concat, and aliases Flatten/Reshape/Squeeze/Unsqueeze/Identity/Dropout when their HWC shape is unchanged, and lowers other reshapes as NCHW flatten and unflatten (`HX_RESHAPE` p[0] 1 and 2). Transpose and the NCHW-order flatten and unflatten use ONNX/PyTorch element order and run in scalar lanes on the FP16 path, which the simulator has not yet run. Operators: Conv (dense, multiplier-one depthwise or grouped, `auto_pad` and asymmetric SAME), Gemm, MatMul with a constant right operand, Add, Sub, Mul, Div by a constant, Relu, Clip, LeakyRelu, PRelu, Sigmoid, Tanh, HardSigmoid, HardSwish, exact Gelu, MaxPool, AveragePool, GlobalAveragePool, ReduceMean over (2, 3), Concat, constant Pad, linear Resize. Anything else, dilation, ceil-mode pooling raises an error naming the node. A Gemm wider than the HMX budget is split into several graph outputs and must be a graph output. The manifest's `fp16_plan` lists, per operation, what keeps the graph out of the DSP's blocked FP16 plan (a mirror of `supported()` in `fp16.c`: mid-graph convolutions must be 1x1, depthwise square 3x3 or 5x5 with stride at most 2, convolution weights must fit the HMX budget); such a graph still runs, on the FP32 path. Tests compare the lowered graph on the portable reference with onnxruntime. Both front ends share [`hxfg_writer.py`](../../hexecutor/offline/hxfg_writer.py) for the constant pool, HMX packing, allocation and serialization.

## Device memory and execution

The shared bootstrap preserves the full 8 MiB VTCM lease, validates mappings and owns worker stacks/restoration. The tensor configuration is separate from depth's QDRX pool/I/O schema:

| Relative private DDR region | Use |
| --- | --- |
| 0–1 MiB | Engine |
| 1–16 MiB | Graph images |
| 16–32 MiB | Host exchange/activation arenas |
| 32–96 MiB | Immutable constants (one graph up to 64 MiB; raised from 32–64 MiB on 2026-11-17, host and simulator checked only, not yet run on the Frame) |

Allocations include guards and cannot overlap. The existing VTCM backup, protected stack and restoration regions remain reserved. Without the FP16 plan, a graph with an arena at most 7 MiB uses `0x31000000..0x31700000` for its activations and the upper MiB as kernel scratch; larger valid graphs keep DDR activations. Inputs/outputs transfer through the private host exchange arena. Small immutable graph metadata and the FP16 plan are staged in the lower L2 TCM region, away from worker stacks/control state.

Worker zero owns HMX; four HVX workers (hthreads 0, 1, 2 and 5) split independent outputs for other operators. Tensor sessions keep HT5's stack in L2. The host requires an engine and bootstrap that both use four workers. Dispatch joins every operation; idle workers use bounded pauses between polls. The HMX leaf splits deep loads at 32 K tiles, preserving its accumulator for larger channel counts. Completion stores have their own tile-aligned scratch region.

Public I/O and the HXFG file remain FP32. Eligible graphs use a private FP16 layout planned at session open: `[channel block][pixel pair][channel][2 pixels]` in whole 32-pixel tiles, with rows padded to an even physical width and the pad column kept zero. Pointwise (1x1, stride 1, unpadded) convolutions read and write this layout in place with one completion per operation. Every other dense convolution (any kernel up to 15x15, stride up to 8, padding, including strided 1x1) packs an im2col operand into the workspace one batch of 32-pixel output tiles at a time and then runs the same matrix strip; the pack is a plain-C gather in [`fp16_pack.h`](../../hexecutor/native/tensor/fp16_pack.h), scalar and not yet optimised, whose layout is checked on the host against a direct convolution (`test_tensor_pack`) but whose matrix-unit use has not been run on the simulator or device. K = kh * kw * input channels must stay at or below 4032. Depthwise shapes other than square 3x3/5x5 with stride 1 or 2 run a scalar-lane path (correct, slow). The compact RGB input remains a fast path for exactly one 3x3 or 5x5 convolution with stride at most 2; a graph whose first convolution differs reads a channel-padded blocked input instead. The float ONNX lowering splits convolutions whose packed weights exceed the 1 MiB HMX budget into output-channel chunks joined by channel concats; every deep load covers one tile-aligned tile, so none crosses the 4 MiB VTCM TLB page boundary. HMX weights, depthwise filters, bias tables and PReLU slopes stay resident in VTCM between runs of a graph and are re-staged after a graph switch or an FP32 run. The compact RGB input is staged inside zero padding so that every window is contiguous. Depthwise copies each worker's rows into a zero-padded window and uses generated, hand-scheduled loops; `gen_depthwise.py` produces `fp16_depthwise.S`.

`hexecutor/offline/compact_hxfg.py` removes channels that are exactly zero or never read, and orders the remaining internal channels by descending weight magnitude, because HMX accumulation is order-sensitive. See the [optimization checkpoint](../model-development/hand-tracking-progress.md) for evidence and remaining validation. `Artifacts::tensor_fp16=false` selects FP32 intermediates with HMX, and `Artifacts::tensor_hmx=false` selects the float DSP comparison path. The tensor runner exposes these as `HEXECUTOR_TENSOR_F32=1` and `HEXECUTOR_TENSOR_PRECISE=1`. Its separate portable CPU reference is explicitly selected; there is no automatic CPU/Python fallback.

`Session::profile()` reports DSP cycles, 19.2 MHz architectural timer ticks, HMX operation count, blocked-layout selection and per-op cycle totals. Host `run()` latency also includes packing/unpacking, publication/polling and guard verification. Warm inference, session initialization/close and application latency are distinct measurements.

## Build and validation

Use `hexecutor/build-scripts/build_tensor.sh` with explicit `HEXECUTOR_SDK_ROOT` and `HEXECUTOR_TOOLCHAIN_IMAGE`. It emits a generic engine, preserving tensor bootstrap and the correct chunk-map probe under `hexecutor/native/build/tensor/`; it never rebuilds/replaces the verified loader.

`make check` exercises portable format, arithmetic, ownership, tracker and existing worker behavior. `hexecutor/build-scripts/test_tensor_sim.sh` requires the SDK and checks actual HMX/HVX instructions, including K>1,024, FP16 conversion boundaries and RGB channel/padding tails, and (added 2026-11-17; passing on the Hexagon simulator the same day) im2col dense convolutions of several kernels, strides and paddings, including a case with a small workspace that forces several pack batches, scalar-lane depthwise shapes, and the standalone activation, broadcasting binary and average-pool operations. The [hand-tracking guide](../model-development/hand-tracking.md) describes reproducible model/device comparisons.

Offline QNN inspection lives under `hexecutor/offline/`: `compile_qnn.py` records an explicit target configuration and hashes; `inspect_qnn.py` parses literal schematic data with bounded scalar arithmetic. It ignores executable expansion loops and never executes a compiler dump. Generated contexts/schematics stay ignored. A compiled topology or host simulator duration is not a measured device benchmark.

## Transformer lowering

[`lower_onnx_float.py`](../../hexecutor/offline/lower_onnx_float.py) also lowers batch-one, static-shape transformer encoders (ViT or BERT style) from float32 ONNX. A rank-3 `[1, L, K]` tensor read or written by MatMul, last-axis LayerNormalization or last-axis Softmax is a token matrix (HWC `h = L`, `w = 1`, `c = K`); other rank-3 tensors keep the NCHW-with-H-1 meaning of convolution-style graphs. Lowered forms:

- MatMul with a constant right operand (Linear layers) and Gemm on tokens: a 1x1 convolution over the `L` rows.
- MatMul of two runtime tensors: one `HX_MATMUL` (`transB` from the operand layout, so `K^T` is a view, not a copy); a rank-3 `[1, K, N]` token operand is read as is.
- Attention from the exporter chain: Reshape to `[1, L, H, dh]`, Transpose to `[1, H, L, dh]` (per-head channel slices, one `HX_MATMUL` per head), a scalar Mul or Div folded into the score product's scale, Softmax over the last axis (`HX_SOFTMAX` per row), the value product, Transpose `[0, 2, 1, 3]` merged by a channel Concat, and Reshape to `[1, L, D]`.
- LayerNormalization over the last axis (`HX_NORM` channel mode), Softmax over the last axis, GELU (exact `Gelu`), residual Add of tokens and per-channel constant Mul/Add/Div.
- Gather of a contiguous run of sequence positions (a slice; CLS-style rows give `[1, K]`), ReduceMean or ReduceSum over the sequence axis, and Concat along the sequence or channel axis.
- Transpose `[0, 2, 1]` of a convolution-style `[1, C, L]` into tokens (patch embedding) is a logical reshape.

Not lowered, and rejected with an error naming the node: Gather or Slice with non-contiguous indices, full-size constant operands on tokens (positional embeddings), constant concatenation (a CLS token parameter), batched or 5-D attention layouts, sequence-first `[L, 1, K]` tensors (torch's `nn.MultiheadAttention`, even with `batch_first=True`; a hand-written batch-first block exports cleanly), token Transpose other than the forms above, and more than 1024 operations (each head of an attention block costs three slices, two products and a softmax, plus one concatenation to merge the heads).

## Persistent state and autoregressive decode

Autoregressive decode (such as LFM3-VL-2B or LFM3-1.7B text decode) requires per-layer state resident on the DSP between `run()` calls:
1. **Short-convolution state rings:** `HX_SHORTCONV` (op 24) maintains `y` as a persistent state tensor `[1, K-1, C]` shifted and updated in place for every decode token.
2. **KV caches:** `HX_KV_WRITE` (op 26) appends key and value slices `[1, 1, C]` (or multi-token / multi-part `[tokens, parts, channels]`) into persistent tensors `[max_tokens, 1, C]` at a sequence position given as a small input tensor `y` (or static `p[0]`).

### Tensor format and state lifecycle

- **Marking persistent tensors:** A tensor is marked persistent by setting `HX_TENSOR_PERSISTENT` (`t.reserved[0] |= 1u`). Persistent tensors are backed by the session-level persistent state region (`g->state`) rather than the per-run activation scratch arena (`g->arena`).
- **Sharing across graphs:** Multiple graphs registered in one `Session` (up to 8 graphs, e.g. a prefill graph and a decode graph) share persistent buffers by specifying a matching non-zero 32-bit state ID / name hash in `t.reserved[1]`. `Session::Impl::open()` computes total persistent state requirements, assigns consistent offsets in the shared DDR persistent region, and updates image offsets.
- **Hazard and validation rules:**
  - `hx_graph_open()` marks persistent tensors as live (`live[i] = 1`) at session start without requiring an input producer.
  - Span bounds checking enforces `t.offset + bytes <= g->state_bytes`.
  - Overlap checking is partitioned: persistent tensors cannot alias non-persistent scratch tensors, preventing activation reuse collisions while allowing in-place state mutation for `HX_SHORTCONV` and `HX_KV_WRITE`.
  - Ops mutating persistent state cannot be reordered across the persistent hazard.
- **Initialization and reset:** State is zero-initialized at session creation (`memset`). Host callers or the broker can issue an explicit reset via `Session::reset_state()` or `Client::reset_state()`, which sends `QDS_OP_RESET_STATE` (4) to the DSP resident entry to zero the persistent region.
- **Blocked FP16 layout and state contract:** Operations 22–26 (`HX_RMSNORM`, `HX_ROPE`, `HX_SHORTCONV`, `HX_LINEAR_Q`, `HX_KV_WRITE`) run natively in the blocked FP16 execution plan. Persistent tensors (`HX_TENSOR_PERSISTENT`) reside exclusively in the session persistent state region (`g->state`) in plain FP32 layout rather than VTCM arena allocation. Ops converting at the boundary (`HX_SHORTCONV` and `HX_KV_WRITE`) preserve this contract, ensuring persistent state survives across `run()` invocations identically between the FP32 reference and FP16 execution paths.

### Memory map update

Persistent state lives in the private DDR lease following constants:

| Relative private DDR region | Use |
| --- | --- |
| 0–1 MiB | Engine |
| 1–16 MiB | Graph images |
| 16–32 MiB | Host exchange / activation arenas |
| 32–96 MiB (or configurable) | Immutable constants |
| 96 MiB+ (up to lease limit) | Persistent state (KV caches + conv state rings; e.g. 32.75 MiB for 2048 ctx FP16) |

Because it resides in the mapped CDSP DDR lease, persistent state survives across inference steps and routine worker wake rotations without re-staging.

### Front-end lowering requirements

To target this path from offline lowering tools (`hxfg_writer.py`, `lower_onnx_float.py`, and ExecuTorch):
1. **Tensor attributes:** Set `reserved[0] = HX_TENSOR_PERSISTENT` and `reserved[1] = fnv1a_32(state_name)`.
2. **Buffer layout:** Allocate persistent tensor offsets within the persistent state region (`state_offset`) instead of the activation arena (`arena_offset`).
3. **KV append operation:** Emit `HX_KV_WRITE` (kind 26) with:
   - `x`: input token slice tensor index `[tokens, 1, C]`
   - `y`: runtime sequence position scalar tensor index `[1, 1, 1]` (or `HXFG_NONE` to use static `p[0]`)
   - `out`: persistent KV cache tensor index `[max_seq, 1, C]`
   - `p[0]`: static sequence position if `y == HXFG_NONE`

## Batched MatMul, attention and higher-rank layouts

Added 2026-11-18: batched runtime matrix multiplication (`HX_MATMUL`, kind 21), fused scaled dot-product attention (`HX_SDPA`, kind 27), and attention softmax with arbitrary-axis and optional causal/additive mask support (`HX_ATTN_SOFTMAX`, kind 28).

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

- FP16/HVX path behavior: `HX_MATMUL`, `HX_ATTN_SOFTMAX`, and `HX_SDPA` run in scalar lanes on the FP16 path (in `fp16.c`), maintaining bit-accurate arithmetic with float32 accumulation and half-precision storage. They are correct across all shapes and tested on host, portable reference, and simulator.
