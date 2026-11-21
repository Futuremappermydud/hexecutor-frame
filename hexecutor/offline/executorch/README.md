# ExecuTorch AOT delegate for hexecutor (tensor_f32 / HXFG)

Offline Python half of an ExecuTorch backend. `torch.export` produces an edge
graph, `HexecutorPartitioner` tags the operations the HXFG operation set can
express exactly, and `HexecutorBackend.preprocess` lowers each tagged partition
to an HXFG v1 image plus constant pool (through `hexecutor/offline/hxfg_writer.py`)
and wraps them in an HXET v1 blob. The C++ runtime that consumes the blob is
separate and must match `blob.py` exactly.

Nothing here runs on the Frame, the DSP or the simulator. Generated `.pte`
files and HXFG images belong under `hexecutor/.work/` (ignored); do not commit them.

## Pinned environment

- executorch 1.0.0, torch 2.9.1, numpy 2.5.5 (CPython 3.12.14, macOS arm64)
- See `requirements.txt`. The venv is `hexecutor/.work/executorch-venv`.

```sh
uv venv --python 3.12 hexecutor/.work/executorch-venv
uv pip install --python hexecutor/.work/executorch-venv/bin/python -r hexecutor/offline/executorch/requirements.txt
```

## Usage

From the repository root:

```sh
hexecutor/.work/executorch-venv/bin/python -m hexecutor.offline.executorch.export \
    --factory my_package.models:build --input-shape 1,3,32,32 \
    --out hexecutor/.work/executorch/model.pte
```

`--factory` names a zero-argument callable returning an `nn.Module`, or a
`(module, example_inputs)` pair. From Python, use `export(model, example_inputs, out_pte)`
or `export_program(model, example_inputs)` for the ExecuTorch program manager.
`export.py` puts the venv's `flatc` on `PATH` when the venv is not activated.

Tests (under 5 seconds; the numeric check runs only when `build/bin/hexecutor_tensor_run` exists):

```sh
hexecutor/.work/executorch-venv/bin/python -m unittest hexecutor.tests.test_executorch_aot -v
```

## Supported operations (edge dialect, batch one, static shapes, float32)

| Operation | HXFG kind | Notes |
| --- | --- | --- |
| `convolution` (dense) | CONV | Includes 1x1. Dilation, transposed and output padding rejected. Weights must fit the HMX budget. |
| `convolution` (groups = C, multiplier one) | DEPTHWISE | Depth multiplier other than one rejected. |
| `linear` | CONV 1x1 | Kept whole through `ops_to_not_decompose`; input must be `[1, K]`. |
| `add` | ADD | Equal shapes, alpha 1, both operands tensors (no constants). |
| `prelu` (`aten.prelu`) | PRELU | Per-channel or scalar slope; kept whole through `ops_to_not_decompose`. |
| `max_pool2d_with_indices` + `getitem(0)` | MAXPOOL | No dilation or ceil mode. Padding at most half the kernel. Padded positions are ignored (see assumptions). |
| `constant_pad_nd` | PAD | Zero value, spatial dimensions only (no batch or channel padding). |
| `upsample_bilinear2d` | RESIZE | `align_corners` maps to HXFG (align, half-pixel = not align). |
| `view_copy` | RESHAPE | Only when both sides have a trivial spatial extent (1x1 or rank 2). |
| `cat` | CONCAT | Exactly two inputs; the output shape must match on the non-concat axes. |
| `mean.dim` over (2, 3) | MEAN | Output `[1, C, 1, 1]` or `[1, C]`. |
| `sigmoid` | LOGISTIC | |
| `elu`, `selu`, `celu`, `mish`, `sinh`, `cosh`, `sign`, `softplus` (beta 1, threshold 20) | ACT | Kept whole through `ops_to_not_decompose`; `elu` only with unit scale and input_scale. |
| `hardswish`, `hardsigmoid`, `silu`, `tanh`, `leaky_relu`, `gelu` (exact) | ACT | Kept whole through `ops_to_not_decompose` (hardswish, hardsigmoid, silu). Tanh-approximate GELU rejected. |
| `relu`, `hardtanh(0, 6)`, `clamp(0, ...)` with no fusable producer | ACT | Standalone ReLU/ReLU6. |
| `add`, `sub`, `mul` (tensors, Python scalars, constants) | ADD or BINARY | Same-shape tensor add stays ADD. Otherwise one operand must be the output's shape and the other's h/w/c each the output's or one; a constant must be per-channel. Rank must match. alpha 1. |
| `avg_pool2d` | AVGPOOL | No ceil mode or divisor override; padding at most half the kernel; `count_include_pad` honoured. |
| `_adaptive_avg_pool2d` to (1, 1) | MEAN | |
| `relu`, `hardtanh(0, 6)`, `clamp(0, None)`, `clamp(0, 6)` | fused | Fused only into a single-user convolution, depthwise convolution or add that is not already activated. |

### Spatial and attention operators

| Operation | Lowering | Notes |
| --- | --- | --- |
| `linear` on `[1, L, K]` | CONV 1x1 | Rank-3 token input (tokens are pixels of `[L, 1, K]`). |
| `view` `[1, C, H, W] -> [1, C, HW]` + `transpose(1, 2)`, and the reverse | one RESHAPE (flag 0) | The attention token view. The pair is recognised as a whole (`lowering.token_view`); the first node emits nothing. |
| `unsqueeze`, `squeeze`, `clone`, `_clone_dim_order`, `alias_copy` | RESHAPE | Shape-only views. |
| `_to_copy` / `_to_dim_order_copy` of a constant | folded | ExecuTorch wraps scalar operands as rank-0 tensors plus a float32 dim-order copy; the consumer reads the constant. |
| `scaled_dot_product_attention`, one head, no mask, not causal | RESHAPE to token columns, MATMUL (Q.K^T, scaled), SOFTMAX, MATMUL (P.V), RESHAPE back | HX_MATMUL runs in HVX lanes on the FP16 plan; HX_SDPA is scalar (Frame: 256 tokens x 64, 178 ms -> 9.9 ms). Multi-head, masked or causal SDPA keeps HX_SDPA. |

Fixes in the same change: a constant operand of a rank-2/3 binary took its channel count from `out[1]` (L, not C), so
`x * 2.0` on `[1, L, C]` with L < 32 scaled only the first channels (now the last axis; a scalar covers every candidate
axis, which also fixes multi-head `[B, H, L, D]` scalars). A rank-1 constant binary raised `IndexError` from the
partitioner instead of `Unsupported`. `ops_to_not_decompose` now decides on a copy decomposed with the same table
(minus the kept ops) instead of the raw export, where spellings like `conv2d` and `group_norm` have no kind and
refused every node after them (so `silu`, `linear` and SDPA were decomposed and fell to the CPU).

Example: `hexecutor/examples/diffusion/unet2d_export.py` (diffusers `UNet2DModel`, fully delegated, 265 HXFG ops).
Example: `hexecutor/examples/fastfoundationstereo/ffs_export.py` (C-Fast-FoundationStereo at 128 x 192, fully delegated, 4344 HXFG ops; see its README).

### Arithmetic, comparisons and shape operators

| Operation | Lowering | Notes |
| --- | --- | --- |
| `where.self` | WHERE | The condition is a runtime tensor (bool or 0/1) that broadcasts to the output. X and Y are runtime tensors, or constants folded as one value (a scalar), per channel, or full size. Float32 results only. |
| `gt`, `lt`, `ge`, `le`, `eq` with a Python scalar; `gt`, `lt`, `ge`, `le`, `eq`, `ne` with a tensor | BINARY (0/1 result) | `ne` is EQUAL then NOT. A bool result may feed only delegated operations (see below). |
| `bitwise_not` on bool tensors | ACT (NOT) | Other integer `bitwise_not` stays on the CPU. |
| `argmax`, `argmin` over the channel axis, with `keepdim` | ARGMAX, ARGMIN | int32 indices; the first index wins ties. Channel axis: dim 1 of `[1, C, H, W]`, the last dim of `[L, C]` or `[1, L, C]`. Other axes, `dim=None` and `keepdim=False` stay on the CPU (the native preflight needs the input's H and W). |
| `topk` over the channel axis, `k` at most 128 | TOPK | Values and indices through `getitem`; `largest` and `sorted` honoured. |
| `max.dim`, `min.dim` over the channel axis, `keepdim=True` | REDUCE (values), ARGMAX or ARGMIN (indices) | `keepdim=False` stays on the CPU. |
| `expand_copy` | TILE | Same rank; each dim is one or equal to the output (HWC multiples). A constant input stays on the CPU. |
| `rsqrt` | ACT (SQRT, then RECIPROCAL) | Through a temporary. |
| `log_sigmoid` | ACT (LOGSIGMOID) | Kept whole through `ops_to_not_decompose` (otherwise decomposed into `log1p`, `minimum`, ...). |
| `clamp` and `hardtanh` with bounds other than ReLU/ReLU6 | BINARY (MAX, then MIN) | Bounds become per-channel constants. ReLU and ReLU6 keep their fusion. |
| `cat` with three or more inputs | CONCAT chain | Intermediate sums go through temporaries. |
| `var.correction` over the spatial axes or the channel axis (rank 2 or 4) | MEAN or REDUCE, BINARY SUB and MUL, scale | Computes mean, centred squares, their mean, then `N / (N - correction)`. `keepdim` output layout required. `std` is `var` followed by the `sqrt` ACT already listed. |
| `full`, `full_like`, `scalar_tensor`, `arange`, and casts or views of them | folded | Emit nothing; the consumer reads the value. A constant is folded only when every user reads it at a constant argument (for example the operand of a binary, where, conv or linear); a constant that a CPU operation also reads stays on the CPU. |
| `convolution` on rank 3 (`conv1d`), batch one | TRANSPOSE, CONV or DEPTHWISE, TRANSPOSE | Dense or depthwise, any stride, padding and dilation (no output padding). Lowered as a convolution with one row. No fused activation. |

### Indexed access and exporter patterns

| Operation | Lowering | Notes |
| --- | --- | --- |
| `softmax` over any dim of rank 2, batch-one rank 3 or batch-one rank 4 | SOFTMAX (channel axis) or ATTN_SOFTMAX (`p[0]` = HXFG axis) | The key axis of `[L, K]` and `[1, L, K]` scores is the channel axis. |
| `add(scores, causal mask constant)` or `masked_fill` / `where` with a constant bool mask, then `softmax` over the last dim of `[L, L]` or `[B, L, L]` | ATTN_SOFTMAX with the causal flag (`p[1]`, `p[2]` = L) | The mask add or where emits nothing; the lowering reads the constant and folds it only when it is exactly causal (zero on and below the diagonal, at most -100 above). Any other constant mask is a full-size per-element multiply and add. |
| `add(scores, runtime mask)` then `softmax` | ATTN_SOFTMAX with `y` = the mask tensor | Token layout only; the mask's h and c each equal the scores' or one. |
| `where` with a constant bool mask and a scalar branch | BINARY (MUL) and BINARY (ADD), per-element constants | `masked_fill` with a constant mask. Rank 2, 3 or 4. |
| `add`/`mul` with a full-size constant on rank 2 or 3 | BINARY (mode 2) | An additive attention mask on tokens. |
| `_to_copy`, `_to_dim_order_copy` of a runtime tensor between float32, int32/int64 and bool | CAST (float -> int32 mode 0, int32 -> float mode 1, same type mode 6); to bool is EQUAL then NOT | Bool tensors are float 0/1 in this front end, so bool -> float is a copy and bool -> int32 is a float -> int32 cast. Other dtypes and dim orders stay on the CPU. |
| `gather(x, dim, index)` | GATHER_ELEMENTS | Runtime float32 data, runtime or constant integer index with the data's shape except along `dim` (rank 1 to 4, batch one). A constant data table stays on the CPU. |
| `index.Tensor` `x[:, :, iy, ix]` on `[1, C, H, W]`, one-dimensional equal-length indices, constant or runtime | GATHER_ND (tuple length 2), TRANSPOSE; runtime pairs go through CONCAT | The result `[1, C, M]` is transposed from the gathered `[M, C]` rows. |
| `index.Tensor` `x[:, idx]`, `x[:, :, idx]` with one 1-D index | GATHER (as `index_select`) | `x[idx]` on dim 0 was already lowered. |
| `index_put`, `select_scatter`, `slice_scatter` on a token tensor `[L, C]` or `[1, L, C]`, along the token dim | SCATTER_ND (tuple length 1, channel vector as the slice) | `index_put(accumulate=True)` is the add reduction. Constant or runtime index; constant or runtime values. `scatter.src` is element-wise (ScatterElements) and is not lowered. |
| `_native_batch_norm_legit_no_training` (inference BatchNorm) | folded into the convolution before it (a following ReLU still fuses), else BINARY MUL and ADD | Rank 4 batch one or rank 2; weights and statistics must be constants. |
| `permute` / `transpose` of a rank-2 or batch-one rank-3 token tensor | TRANSPOSE | `K^T` of attention. |
| `slice` on rank 2 (rows or channels) and batch-one rank 3 | SLICE | |
| `bmm` with batch one | MATMUL on the token layout `[L, 1, K]` | Higher batches keep the multi-head layout `[B, L, K]`; rank-3 mask, scale and softmax between two such bmm calls follow that layout. |

A view between a rank-2 or rank-3 token tensor and `[1, C, H, W]` now flattens or unflattens only the NCHW side (it
used to treat the token side as NCHW too and scrambled `linear -> view [1, L, H, dh]`). Grouped convolution, `addmm`
after a view, `instance_norm` and a view after a delegated permute are lowered and covered by numeric tests.

Integer and bool values stay inside the delegate: delegate inputs and outputs are float32, so an
argmax, a topk index, a comparison or a `bitwise_not` result is delegated only when every user is delegated
too. The same holds for a folded constant. An integer tensor from the CPU cannot feed a delegated operation either, so
an `index_select` whose index is computed on the CPU (for example `roll`) stays on the CPU. The partitioner drops the nodes that break these
rules, together with their delegated descendants, which keeps the partition convex.

Partitions with fewer than `MIN_OPS` (2) operations are not delegated. ExecuTorch
splits the graph wherever an unsupported operation sits between delegated ones.
`select_nodes` refuses any node whose input depends on a CPU operation that
itself follows a delegated one, because delegating both would create a
partition cycle.

### Cost volumes, 3D convolutions and other rewrites (`volume.py`, `decompositions.py`)

`export_program` re-traces the decomposed program through `volume.rewrite_volumes` when it holds a rank-5
tensor, a channels-last `linear` or layer norm, a dense convolution over the HMX budget, or an operation in
`decompositions.CUSTOM`. The rewrite runs eagerly once (it creates the new constants and checks the result against
the original program, `rtol` 1e-4), then exports again, so the delegate sees ordinary rank-4 ATen. Constant
subgraphs (`arange`, `eye`, `sin`/`cos` tables, casts, clamps of constants) are folded into buffers, and dead
nodes, unused `getitem` outputs and views undone by the next view are removed. Anything the rewrite has no form for
raises `VolumeRewriteError` with the node and shape.

| Operation | Rewritten as | Notes |
| --- | --- | --- |
| rank-5 value `[1, C, D, H, W]` | the depth-row layout `[1, C, D, H*W]` | Pointwise operations, batch norm, `cat` along C or D, broadcasts, `unsqueeze`/`expand`/`view` from and to `[1, C, ...]` maps, `squeeze` of C=1 (a transposition) and the `permute` pair used by per-pixel depth attention are covered. H*W is at most 4096. |
| `convolution` 3D (dense, strided, dilated, grouped, depthwise) | per spatial tap (ky, kx): `index_select` over the H*W columns (zero column for padding), then a `(kd, 1)` convolution; the taps are summed. The output pixels run in column chunks of about `MAX_TEMP_BYTES` (6 MiB) of gathered tap, each reading only its own column range plus one zero column, and the chunks are joined by a balanced tree of two-input concatenations | `(kd, 1, 1)` kernels are one 2D convolution. Strides, padding and dilation in H and W are the gather index; in D they are the 2D convolution's. |
| `convolution` 3D transposed (4x4x4 stride 2, 3x3x3 stride 2 with output padding, ...) | the same gather over the output pixels (zero for positions between inserted zeros) plus a transposed `(kd, 1)` convolution | Dilated transposed convolutions are not rewritten. |
| convolution kernel above 15 rows (a `kd` = 17 depth convolution) | taps split into groups of at most 15, each read from the padded input and summed | HXFG kernels are at most 15 x 15. |
| dense 2D convolution over the HMX scratch budget | input-channel chunks (summed) and output-channel chunks (concatenated) | Applies to transposed convolutions too (`conv_chunks`). |
| `index.Tensor` with separable constant (D, H, W) indices | `index_select` over D and one over the pixel columns | The interpolation weights are ordinary broadcasts. |
| `index_select` of a rank-5 value along D, H or W with a constant index | `index_select` over the depth rows, or over the pixel columns with the H or W index expanded | |
| `upsample_trilinear3d` | three separable linear interpolations (W, then H, then D), each two `index_select`s and a weighted sum (`decompositions.py`) | Upstream's decomposition gathers all eight corners at the full output size at once; this keeps one axis live. |
| `[1, G, Cg, D, H, W]` views with `linalg_vector_norm`/`sum` over Cg (group-wise correlation) | 1x1 convolutions with block-ones weights (sum, replicate) | |
| channels-last `[1, H, W, C]` blocks (`permute`, layer norm, `linear`, GELU, per-channel scale, `permute` back) | token matrices `[1, H*W, C]`; the NCHW pair is one HWC copy | `linear` over `[B, L, C]` with B above one runs as `[B*L, C]`. |
| `linalg_vector_norm` (order 1 or 2), `F.normalize` | `sqrt(sum(x*x))`, `sum(abs(x))` | `decompositions.py` |
| `im2col`, `F.unfold`, batch one | a convolution with one-hot taps (`torch.eye`) | |
| `adaptive_max_pool2d` to 1x1 | `amax` over the spatial dims | |
| `select.int` | `narrow` and `squeeze` (slice and reshape) | |
| `grid_sampler_2d`, batch one, bilinear, zeros or border padding, either `align_corners` | four `index` pixel gathers with zero-outside weights | The 4096-element NCHW flatten limit applies to the `[1, C, H*W]` result. The per-pixel batch of upstream stereo lookups (B*H*W one-row images) is not batch one; `examples/fastfoundationstereo` rewrites it as gathers along the depth axis. |
| transposed 2D convolution (4x4 stride 2, ...) | ZEROINSERT and a dense convolution | Already lowered; now covered by `test_transposed_2d_convolution`. |

Lowering additions (`lowering.py`): `index_select` along the rows of `[1, C, H, W]` with a constant index is a pixel
gather over the flattened (h, w) list (the earlier row form read one channel vector per pixel and returned wrong
rows); a reshape between batch-one rank-4 tensors that keeps dim 1 is a pixel-order HWC copy, whatever its size;
`alias` has a kind before the edge conversion; the key transpose of a bmm with a batch above one and rank-3 views
between attention tensors keep the `[B, L, C]` layout (`_spread_multihead`); equal-shape adds with a batch above one.

Memory: HXFG runs operations in program order, and ExecuTorch emits every independent gather ahead of the operations that read them. `lowering._demand_order` re-schedules a graph by a depth-first walk from its outputs (each operation just before its first consumer) and keeps the result only when its estimated live peak is lower; graphs with persistent state, KV writes, RoPE, short convs, SDPA or scatter keep the emitted order. For the C-Fast stereo model at 224 x 288 this took the planned arena from 338 MiB to 133 MiB, and the column chunks to 83 MiB (see `examples/fastfoundationstereo/README.md`).

Native limits that the rewritten graphs hit (set in `hexecutor/native/tensor/graph.h`/`graph.c`, not changed here):
`HXFG_MAX_TENSORS` and `HXFG_MAX_OPS` are 1024, the arena is at most 16 MB, and a tensor dimension is at most
4096. A stereo cost-volume model lowers to thousands of operations and hundreds of MB of activations, so it passes
the AOT lowering and the numeric comparison on a host build with larger limits but not the standard preflight (the limits are compile-time settings now: see `graph.h`). The H*W of a rank-5 value is at most 4096; there is no zero-padded copy of a whole volume, so a volume of exactly 4096 pixels is accepted. Larger images need the pixel axis tiled (not implemented).

## Not supported (stay on the CPU)

Broadcasts that stretch both operands, other pooling kinds, and every other operator not in the table. Also dynamic shapes, batch greater than one,
and float results other than the bool/int intermediates above. Integer arithmetic, `dim=None` reductions, reductions and argmax/topk on a non-channel axis, and
`keepdim=False` max/min. Transposed or grouped conv1d, and conv1d with output padding. Concatenation of constants, and reshapes that mix spatial and
channel order. Linear layers whose packed weights exceed the per-op HMX budget (`hxfg_writer.hmx_fits`) are rejected
rather than split; dense 2D convolutions over the budget are split by `volume.py` when exported through
`export_program`. Cumulative sums (`cumsum`) and per-pixel-batched `grid_sample` on runtime tensors have no form;
`cumsum` of constants (positional tables) is folded into a buffer whenever the rewrite runs.

## Layout contract

- PyTorch tensors are NCHW. HXFG stores each tensor as HWC with channels padded to
  32 floats. A rank-2 `[1, C]` tensor is stored as `[1, 1, C]`.
- Graph inputs whose consumers are all convolutions use compact 3-channel storage.
- External I/O is compact float32 NHWC with no channel padding. The runtime
  transposes rank-4 tensors between NCHW (PyTorch) and NHWC (HXFG). The blob marks
  these with `layout = 1`; rank-2 tensors use `layout = 0`.

## Blob format (HXET v1, `blob.py`)

All little-endian, no padding:

```
b"HXET", u32 version=1, u32 backend=1 (tensor_f32 HXFG), u32 n_inputs, u32 n_outputs
for each input, then each output:
    u32 layout (0 as-is, 1 NCHW<->NHWC), u32 rank, u32 dims[rank]   (PyTorch dims, float32)
u64 image_size, u64 constants_size
image bytes (.hxfg), constants bytes (.bin)
```

`blob.pack` and `blob.unpack` implement this; `unpack` rejects bad magic, unknown
versions or layouts, truncation and trailing bytes.

## Validation

- `lowering.check_image` mirrors the header checks in `hexecutor/native/tensor/graph.h`:
  magic, version, size fields, section offsets, the constant SHA-256 and non-empty
  graph and I/O. It runs on every lowered image. The DSP repeats its own checks.
- The tests compare the HXFG output with the portable CPU reference
  (`build/bin/hexecutor_tensor_run`) for a conv/depthwise/add/sigmoid model and a
  model with strided convolution, max pooling (padding 1), PReLU, concatenation,
  padding, bilinear resize, mean, linear and sigmoid. Both match PyTorch within
  1e-3 (observed differences were about 1e-7). The check covers the portable CPU
  reference only, not the DSP kernels.
- The operator-coverage tests (`test_*_match_torch_on_cpu_reference`, `test_constant_factories_fold_into_consumers`,
  `test_conv1d_matches_torch_on_cpu_reference`, and the negative tests `test_argmax_outside_the_channel_axis_stays_on_cpu`
  and `test_integer_output_is_not_delegated`) each check one delegate, that the new HXFG kinds are present, and the
  CPU reference against PyTorch.

## Assumptions

- Max pooling with padding: the portable reference matches PyTorch for padding 1
  (the `WideOps` test), which is consistent with padded positions being ignored
  (equivalent to `-inf` padding, as in the TFLite SAME lowering). The DSP kernel
  has not been checked against this.
- The compact RGB input layout is used only when every consumer is a convolution.
- General clamp and hardtanh use `fminf` and `fmaxf`, so a NaN input takes the bound instead of
  propagating as `torch.clamp` does. Inputs are finite in every test; this is the one known numeric difference.

## Source references

- Format and operations: `hexecutor/native/tensor/graph.h`, `docs/hexecutor/tensor-backend.md`.
- Shared writer: `hexecutor/offline/hxfg_writer.py` (`write`, `ConstantPool`,
  `pack_hmx_weights`, `hmx_fits`). Layout and weight packing mirror
  `lower_tflite.py` and `lower_onnx.py`, which are unchanged.

## Stateful decode operators

`ops.py` registers `hexecutor::lfm_linear_q` (GGML Q4_0/Q8_0 rows, uint8 `[N, K/32, 18|34]` blocks),
`lfm_rope`, `lfm_shortconv` and `lfm_kv_write`. They run eagerly (state in `ops.STATE`, cleared with
`reset_state()`) and lower to HX_LINEAR_Q, HX_ROPE, HX_SHORTCONV and HX_KV_WRITE. The conv ring and the KV
cache are persistent tensors in the graph's state region, named by the `state` argument; the host owns that
region and keeps it between runs. Rules the lowering enforces: one token per run for the conv and runtime
positions; a runtime position is a float32 `[1, 1]` input (delegate inputs are float32), otherwise a static
`start`; the rotary cos/sin caches must be the plain `theta ** (-2j / rot)` table (checked when the data is
read, so a modified table fails the export); a cache may only feed `scaled_dot_product_attention`, whose
operands are `[B, H, L, D]` (the cache is `[1, 1, max_len, C]`) with an additive mask input. The `.pte` delegate
blob needs no new field: the image's tensor table carries the persistent flag and state id. The runtime that
opens it must supply the state region (not yet done in hexecutord / the ExecuTorch backend).

    hexecutor/.work/executorch-venv/bin/python -m unittest hexecutor.tests.test_executorch_lfm -v
    hexecutor/.work/executorch-venv/bin/python hexecutor/examples/llm/lfm/gen_executorch_decode_golden.py OUT
