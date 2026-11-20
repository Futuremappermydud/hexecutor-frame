# Depth Pro 2 (S) on the CDSP via ExecuTorch: handoff, 2026-11-20

## Why

The user asked to run `apple/DepthPro2` on the CDSP with the Arcturus passthrough camera parameters and measure
its speed. No device number exists yet: the 128 px export is lowered, checked on the host and staged on the Frame,
but has not been run on the DSP.

## Model and camera

- `apple/DepthPro2` has no weights. The sizes are separate repos (S 24M, B 97M, L 334M, H+ 872M) plus a shared
  focal estimator (`apple/DepthPro2-focal`). Only **S** is a realistic CDSP target. Licence: Apple ML Research,
  research use only. Weights are kept outside the repo.
- Weights: `apple/DepthPro2-S` `model.safetensors` (SHA-256 starts `583c4d630779fd23`). Code:
  `github.com/apple-aiml-research/ml-depth-pro-2` at `aa389cb`. Both live in the session scratchpad
  (`/private/tmp/claude-501/.../scratchpad/dp2/{S,src}`), which is temporary, so re-download them if it's gone.
  The config is ViT-S/16 DINOv3 (RoPE, 4 registers), taps at blocks 2/5/8/11, cross-frame blocks 5/7/9/11, and a
  DPT decoder with 64 channels.
- Arcturus V3 passthrough calibration (`~/Downloads/arcturus/arcturus-shareable/calibration/sensor-rig-calibration.json`):
  2464 x 2464 `kb` fisheye. Right camera fx 1476.2, fy 1478.3, cx 1391.0, cy 1238.1. Left camera fx 1477.5,
  fy 1479.1, cx 1082.4, cy 1243.4.
- Focal length only post-scales the output: `depth = f_px / (S * inv_depth)`. With f_px known, the focal estimator
  is skipped, and the camera parameters only set the input crop and the host-side metric conversion. A centre crop
  of width `crop` resized to S has `f_px = 1477 * S / crop`. Undistortion/rectification to pinhole is not done yet.

## Code (uncommitted)

- `hexecutor/examples/depthpro2/depthpro2_export.py` (new) exports a one-frame model. With one frame, upstream
  cross-frame blocks are ordinary blocks, except that they **keep the incoming cls/register tokens**. Missing that
  gave a 20% error at first. Rewrites, each exact against upstream (`--check`):
  - The host passes patch tokens `[1, N, 768]` of raw RGB in (c, ky, kx) order (`DepthPro2Frame.patches`). Patch
    embedding is a linear with the ImageNet normalisation folded in. HXFG conv stops at kernel 15 / stride 8, so the
    16x16 stride-16 conv can't run as a conv.
  - The cls+register tokens are a second input `[1, 5, 384]`, so the token concat is between two runtime tensors.
  - RoPE is folded into the weights: fused `q|k`, rotated `q|k` (`W^T M`) and `v` linears; one cos/sin multiply-add
    over all heads; six single-head SDPAs on channel slices; concat; one proj.
  - LayerScale is folded into proj/fc2. The MLP is split in two (per-op HMX weight budget). Over-budget convs (the
    3x3 s2 384->384 reassemble conv) are split by output channels.
  - Also written but **not exercised yet**: `fold_fusion_upsample` (DPT deconv and 1x1 out_conv as one
    ConvTranspose), `TailStrip` (full-res head in row strips) and `segment_partitioner` (one delegate per strip).
    `main()` doesn't pass `strips` or use the partitioner yet. These are for sizes of 256 and up.
- `hexecutor/offline/executorch/lowering.py`:
  - Rank-3 batch-one `cat` is now allowed (`_concat_axis`: the token dim maps to HXFG axis 0).
  - Single-head SDPA on `unsqueeze(1)` token tensors whose only user is `squeeze(1)` now runs MATMUL/SOFTMAX/MATMUL
    on the token columns directly (`_sdpa_token_views`), with no multi-head repack copies (four RESHAPEs per head
    saved).

## Checks

- `python -m unittest hexecutor.tests.test_executorch_aot`: 64 tests OK. This ran after the lowering changes and
  before the patch-token rewrite. The lowering hasn't changed since.
- 128 px: rewrite vs upstream max abs 1.3e-6 (output max 0.22). The host `hexecutor_tensor_run` CPU reference of the
  lowered HXFG vs PyTorch is 6.9e-7. One delegate, no CPU operators. 797 ops, 799 tensors, FP32 arena 4.73 MiB
  (under the 7 MiB VTCM/FP16 gate), constants 159.7 MiB.
- Not run: ISS. `test_tensor_model_sim` exited after 118k instructions, apparently allocating the 160 MiB pool.
  `make check` was not run either.

## Limits found (they set the plan for bigger inputs)

- HXFG allows at most 1024 ops/tensors per graph. Per-head projections gave 1962 ops; the fused form is about 800.
- The engine uses the FP16/VTCM plan only when a graph's FP32 arena is ≤ 7 MiB
  (`engine.c`, `arena_bytes <= 0x700000`). At 256 the full-res head (32 ch padded, 8 MiB FP32) can't fit in one
  graph, hence the strips plus one delegate per strip (≤ 8 graphs per session).
- At 159.7 MiB, constants exceed the [32, 96) MiB region, so they need the high-constant region: an engine that maps
  more than 128 MiB, plus a lease resize.

## Frame

- Staged in `~/hexecutor-dp2/`: `dp2s_128.pte` and `hx128/{g.hxfg,g.bin,x.f32,p.f32,want.f32}`.
- Engine: a 32-slot build from this tree, `hexecutor/native/build/tensor-32/engine.elf`. Its SHA-256 `972683ed…` is
  identical to `~/fht-et/dsp/engine-c4.elf`, so use that file. Loader, bootstrap and chunk probe are in
  `~/fht-et/dsp/`. The runner is `~/fht-et/build-nn/hexecutor-executorch/hexecutor_et_run`.
- At 11:20 the hand-tracking worker (`frame_hand_tracking.worker`, PID 9251) held `/dev/cdsp_mem`, so nothing was
  run. The user turned it off. The DSP read `[1,0,1]`, callback 0, regions zero at the end of this session.
  `cdsp_mem` `size_mb` 128.
- On sharing: one process can hold many models (one Session, up to 8 graphs), but a second process can't open a
  session while another owns the CDSP. The ExecuTorch path doesn't go through `hexecutord` yet.

## Next

1. Run the 128 export when the DSP is free. Resize the lease to 512 MiB first (ioctl `0x40044301`, see
   `hexecutor/device/kmod/dynamic_mem_stage_test.sh`; it resets after every session), then:
   ```sh
   D=~/fht-et/dsp; export HEXECUTOR_LOADER=$D/qd_loader.elf HEXECUTOR_ENGINE=$D/engine-c4.elf \
     HEXECUTOR_BOOTSTRAP=$D/bootstrap.elf HEXECUTOR_CHUNK_PROBE=$D/chunk_probe.elf \
     HEXECUTOR_CHUNK_WINDOW_MB=512 HEXECUTOR_CDSP_MEM_MB=512
   ~/fht-et/build-nn/hexecutor-executorch/hexecutor_et_run ~/hexecutor-dp2/dp2s_128.pte --seed 3 --warmup 3 --iters 20
   ```
   Compare against `HEXECUTOR_ET_REFERENCE=1` as `~/hexecutor-et-diff/run.sh` does. Check DSP state before and after.
2. Get a per-op profile on real inputs with `tensor_run` on `hx128/` (`LFM_OP_PROFILE`-style dump if available).
3. 256 px: pass `strips=4` and `segment_partitioner(model)` into the export, then check the per-graph arena is
   ≤ 7 MiB and ops are ≤ 1024.
4. Shrinking constants (FP32 masters beside the packed FP16 weights) would avoid the high region.
5. Host side: undistort the Arcturus fisheye to a pinhole crop, then convert to metric depth with
   `f_px = 1477 * S / crop`.
