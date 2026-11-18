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
| `hardswish`, `hardsigmoid`, `silu`, `tanh`, `leaky_relu`, `gelu` (exact) | ACT | Kept whole through `ops_to_not_decompose` (hardswish, hardsigmoid, silu). Tanh-approximate GELU rejected. |
| `relu`, `hardtanh(0, 6)`, `clamp(0, ...)` with no fusable producer | ACT | Standalone ReLU/ReLU6. |
| `add`, `sub`, `mul` (tensors, Python scalars, constants) | ADD or BINARY | Same-shape tensor add stays ADD. Otherwise one operand must be the output's shape and the other's h/w/c each the output's or one; a constant must be per-channel. Rank must match. alpha 1. |
| `avg_pool2d` | AVGPOOL | No ceil mode or divisor override; padding at most half the kernel; `count_include_pad` honoured. |
| `_adaptive_avg_pool2d` to (1, 1) | MEAN | |
| `relu`, `hardtanh(0, 6)`, `clamp(0, None)`, `clamp(0, 6)` | fused | Fused only into a single-user convolution, depthwise convolution or add that is not already activated. |

Partitions with fewer than `MIN_OPS` (2) operations are not delegated. ExecuTorch
splits the graph wherever an unsupported operation sits between delegated ones.
`select_nodes` refuses any node whose input depends on a CPU operation that
itself follows a delegated one, because delegating both would create a
partition cycle.

## Not supported (stay on the CPU)

Broadcasts that stretch both operands, constant operands that vary spatially, other pooling kinds, `permute`/`transpose`, `softmax`, `where`/`gt` and every other
operator not in the table, dynamic or batch greater than one, non-float32
tensors, dilated, transposed or grouped (non-depthwise) convolution, concatenation of other than
two inputs, and reshapes that mix spatial and channel order.
Convolutions or linear layers whose packed weights exceed the per-op HMX budget
(`hxfg_writer.hmx_fits`) are rejected rather than split.

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

## Assumptions

- Max pooling with padding: the portable reference matches PyTorch for padding 1
  (the `WideOps` test), which is consistent with padded positions being ignored
  (equivalent to `-inf` padding, as in the TFLite SAME lowering). The DSP kernel
  has not been checked against this.
- The compact RGB input layout is used only when every consumer is a convolution.

## Source references

- Format and operations: `hexecutor/native/tensor/graph.h`, `docs/hexecutor/tensor-backend.md`.
- Shared writer: `hexecutor/offline/hxfg_writer.py` (`write`, `ConstantPool`,
  `pack_hmx_weights`, `hmx_fits`). Layout and weight packing mirror
  `lower_tflite.py` and `lower_onnx.py`, which are unchanged.
