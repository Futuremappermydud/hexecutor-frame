# Fast-FoundationStereo (C-Fast) on the hexecutor delegate

`ffs_export.py` exports NVIDIA's C-Fast-FoundationStereo ([code](https://github.com/NVlabs/Fast-FoundationStereo),
Apache-2.0; [weights](https://huggingface.co/nvidia/c-fast-foundationstereo), NVIDIA Open Model Agreement) to one
delegate. Keep the checkpoint and every generated `.pte` or HXFG image under the ignored `hexecutor/.work/`.

## Status (2026-11-20)

- 17.65 M parameters. At 128 x 192, 4 refinement iterations, `max_disp` 192 the whole model is one delegate:
  4344 HXFG operations, 126.9 MB of constants, 438 MB of activations.
- The model rewrites in `ffs_export.py` match the unmodified upstream model (`--check`): max abs 4.8e-6 on
  disparities near 3 px; a textured pair shifted by 6 px gives 5.96 to 6.14 px.
- The HXFG image matches the PyTorch output on the portable CPU reference: max abs 5.2e-6, mean 7.3e-7. This used
  a private host build with `HXFG_MAX_TENSORS`/`HXFG_MAX_OPS` raised to 65536 and the arena limit to 1 GiB
  (`graph.h`, `graph.c`, `executor.cpp`); the standard `build/bin/hexecutor_tensor_run` rejects the image
  (`tensor preflight failed: error 1`: more than 1024 operations or tensors, arena above 16 MB).
- Not measured: DSP cycles. The image does not fit the native limits above, no DSP or simulator run was made, and
  the Frame was unreachable.

## What it exercises

The 3D cost-volume convolutions (dense, strided, depth kernels of 3, 9, 13 and 17, transposed 4x4x4), the group-wise
correlation, the trilinear resize, channels-last EdgeNeXt blocks, oversized 320-channel convolutions, vector norms,
`im2col`, the global max pool, and the iterative lookups. See the operator table in
`hexecutor/offline/executorch/README.md` for how each lowers.

## Cost at this size

This is the first pass; the memory pass below restructured it. At 480 x 736 the pixel axis (22080) is above the
4096 limit and the activations are about 15 times larger.

## Sizing and measurements (2026-11-20, later)

Per-stage MACs at 256 x 320, `max_disp` 128 (from `torch.utils.flop_counter` on the unmodified model; the volume
builds and the lookups are not counted): 57.2 GMAC at 4 iterations (cost aggregation 22.2, update block 17.0
for four calls, features 8.8, volume attention 7.6), 48.7 at 2 iterations, 72.0 with `max_disp` 192. 384 x 512 with
`max_disp` 192 is 173 GMAC.

- The pixel axis of the volume rewrite is H/4 x W/4 <= 4096, so 224 x 288 (4032) exports unchanged; 256 x 320
  (5120) needs the volume tiled along pixels.
- 224 x 288, 4 iterations, `max_disp` 128 exports as one delegate (first pass: 4584 ops, 806.8 MiB arena, 145.2 MiB
  constants; superseded by the memory pass below). A host build with the limits raised matches PyTorch to 3.8e-6.
- `HXFG_MAX_TENSORS`, `HXFG_MAX_OPS` and `HXFG_MAX_ARENA_BYTES` (graph.h) can be defined at compile time for host
  and simulator builds; the defaults and the DSP engine are unchanged. The Frame engine still has a 16 MiB arena.
- The blocked-FP16 simulator harness (`hexecutor/tests/test_tensor_model_sim.c`) keeps every activation in 8 MiB of
  VTCM. It cannot plan the whole model, or the backbone (its 4 x 4 stride-4 compact-input stem is not supported
  there). One GRU update call at 128 x 192 plans and runs, but its output disagrees with the PyTorch and portable
  references, so its cycle count is unvalidated.

## Memory pass (2026-11-20, third)

Goal: every tensor and the planned peak under the 96 MiB DSP arena, 224 x 288 and 256 x 320. Measured with the
HXFG planner (`arena_bytes` in the image header, first-fit with reuse) and the live-bytes peak of the same op order
(sum of live tensor sizes at the worst op), 4 iterations, `max_disp` 128:

| Size | Ops | Planned arena | Live peak | Largest tensor | Constants | CPU reference vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 224 x 288 (first pass) | 4584 | 806.8 MiB | n/m | 110.2 MiB | 145.2 MiB | 3.8e-6 max |
| 224 x 288 | 5593 | 83.0 MiB | 70.6 MiB | 15.8 MiB | 140.1 MiB | 4.3e-6 max, 4.8e-7 mean |
| 256 x 256 | 5785 | 84.2 MiB | 71.8 MiB | 16.0 MiB | 140.8 MiB | 3.6e-6 max, 4.6e-7 mean |
| 256 x 320 | not exported | | | | | |

`--check` (rewritten model against the unmodified upstream model): 2.9e-6 max at 224 x 288, 6.0e-6 at 256 x 256.
The CPU reference ran on the private host build with `HXFG_MAX_TENSORS`/`HXFG_MAX_OPS` 16384 and a 1 GiB arena. The
standard runner (1024 ops, 16 MB arena) still rejects these images, as does the Frame engine; nothing was run on the
simulator or the Frame.

What brought 806.8 MiB down:

1. `VolumeStem` (`ffs_export.py`) builds the group-wise correlation and the concat volume one disparity at a time:
   cosine similarity from a 1x1 convolution of ones over `ref * shift_d(tgt)` on the `W - d` columns where the shifted
   target exists, divided by the two group norms, then `[sim, left, shift_d(right)]` per disparity and a balanced
   concatenation tree. Upstream materialises `[1, 8, 28, D, H, W]` products and a `[1, 224, D, H, W]` shifted volume
   (110 MiB each) and joins the shifts with a growing concatenation chain. 806.8 -> 338.1 MiB together with the next item.
2. `upsample_trilinear3d` is three separable interpolations (decompositions.py); upstream's decomposition gathers all
   eight corners at the full 13.8 MiB output size.
3. `lowering._demand_order` runs each operation just before its first consumer. ExecuTorch emitted the 16 tap
   gathers of one 4x4x4 transposed convolution ahead of the convolutions that read them. 338.1 -> 133.2 MiB.
4. 3D convolutions run their output pixels in column chunks (`volume.MAX_TEMP_BYTES`, 6 MiB of gathered tap), each
   reading only its own source range plus one zero column. There is no whole-volume padded copy any more, which also lets a
   volume of exactly 4096 pixels through. 133.2 -> 83.0 MiB. A chunk is split further when its source range alone
   would not fit a tensor (a stride-2 convolution reads twice its width).

The remaining peak is four 15.8 MiB volume-sized tensors live at the FeatureAtt elementwise chain (28 channels are
padded to a stride of 32). The 256 x 256 image is the largest square the volume rewrite takes.

### 256 x 320 is not done

A 64 x 80 = 5120-pixel 1/4-resolution map is over the 4096 limit on every tensor dimension, and the failure is not
only in the volume rewrite (`convolution_110: 64x80 -> 64x80 pixels exceed the 4096-column HXFG limit`). In the
224 x 288 graph 848 tensors have a dimension of 3300 or more, in three families that all grow with H*W:

- the `[1, C, D, H*W]` volumes of the cost aggregation (the rewrite),
- the iterative-update lookups of `GeoEnc4`/`SplitConv`: `[1, C, K, H*W]` samples, `[1, H*W, W2]` correlation rows and
  `[H*W, 1, 1]` position columns,
- the EdgeNeXt token blocks of the feature extractor (`[1, H*W, C]` layer norms, linears and the cross-covariance
  attention's token reductions).

Tiling would be by image-row bands (each band at most 4096 pixels, with a one-row halo exchange for every 3x3
convolution and a halo of 3 for the 7x7x7 classifier), carried through all three families; the cross-covariance
attention sums over tokens, so its bands would also need a reduction across bands. None of this is implemented.
Sizes with at most 4096 pixels at 1/4 resolution (224 x 288 and 256 x 256 were exported and checked) are what the model takes today.
