# Depth Pro 2 S on the Frame CDSP

## Status on 2026-11-20

Depth Pro 2 S ran a calibrated Arcturus left-camera frame at 128 × 128 through a single ExecuTorch delegate on the Frame CDSP. The runner reported **810.793 ms median warm inference**, about **1.23 inferences/s**, on twenty measured calls after the first call and three warmups. The engine used FP32 intermediates with HMX convolutions. The benchmark starts at model inputs and excludes image preprocessing, metric conversion, camera capture and display.

| Input and runner | Method load | First call | Median warm | P90 warm |
| --- | ---: | ---: | ---: | ---: |
| Calibrated Arcturus left frame, ExecuTorch on CDSP | 4,127.142 ms | 816.515 ms | 810.793 ms | 814.510 ms |
| Seed-3 synthetic tensors, ExecuTorch on CDSP | 4,514.834 ms | 843.993 ms | 839.396 ms | 842.136 ms |

On the calibrated frame, the DSP output was byte-identical between ExecuTorch and the direct HXFG runner. Against the matched ExecuTorch CPU reference, the DSP had 0.001073 max and 0.000324 mean absolute inverse-depth error. Using the calibrated pinhole focal length, the predicted optical-axis depth ranged from 1.14 m to 10.91 m. The mean absolute DSP-to-reference difference after conversion was 0.014 m. This comparison measures implementation agreement on one saved frame.

The seeded run randomizes both model inputs. Its maximum absolute error against the matched ExecuTorch CPU reference was 0.001284 in focal-normalized inverse depth over 16,384 pixels, with mean absolute error 0.000778 and cosine similarity 0.999999960. The direct HXFG run on the staged exporter's random RGB patches and model prefix took 844.494 ms wall time and 790.759 ms DSP timer time. It agreed with PyTorch to 0.000960 max and 0.000730 mean absolute error. These are separate inputs.

The calibrated direct HXFG run measured 765–769 ms of DSP time over four calls. The final call reported `blocked_fp16=false` and 793 operator cycle counts. Attention matmuls used 57.4% of attributed cycles, normalization 12.4%, standalone activations 11.6%, and convolution and linear projections 6.7%. The profile and output dumps are local to `~/hexecutor-dp2/cdsp128-l9k9wj/` on the Frame. The kernel was `7.2.0-g9a8c8ea46950`. Before and after each device run, the firmware control words read `[1,0,1]`, the callback was zero, and the loader, payload and arena regions were zero. The CDSP lease returned to 128 MiB. No eye client was active during these runs.

## Model and camera contract

- Apple publishes four model sizes and a shared focal estimator. This handoff uses the 24M-parameter S checkpoint with SHA-256 `583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c`. Keep the research-use weights and derived artifacts local. [Model card](https://huggingface.co/apple/DepthPro2), [S checkpoint](https://huggingface.co/apple/DepthPro2-S), [Apple ML Research Model License](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/LICENSE_MODEL).
- The [Apple implementation](https://github.com/apple-aiml-research/ml-depth-pro-2/tree/aa389cbb0b74341664ed663a0966c859905cd6c0) uses ViT-S/16 DINOv3 with RoPE and four register tokens, taps blocks 2/5/8/11, uses cross-frame attention at 5/7/9/11, and has a 64-channel DPT decoder. The one-frame rewrite preserves the incoming class and register tokens at the cross-frame blocks.
- Input zero holds row-major patches shaped `[1, N, 768]` with RGB values in `[0, 1]` and channel/row/column order within each 16 × 16 patch. The linear patch embed folds ImageNet normalization into the weights and bias. Input one is the checkpoint's class and register prefix, shaped `[1, 5, 384]`.
- The Arcturus V3 calibration has 2464 × 2464 Kannala–Brandt color cameras. Right `fx, fy, cx, cy` are about `1476.2, 1478.3, 1391.0, 1238.1`. Left values are `1477.5, 1479.1, 1082.4, 1243.4`. The offline exporter samples the raw image onto an upright square pinhole through the calibrated lens model. It uses the input width to choose the virtual focal length, `f_px = fx * S / width`, and applies the right sensor's 180° mount correction. See [camera coordinates and calibration](../runtime/cameras.md).
- The output is focal-normalized inverse depth. Convert a positive inverse-depth prediction to optical-axis depth in metres with `depth_m = f_px / (S * inv_depth)`, using the focal length of the pinhole input. This skips the focal estimator. The conversion uses a camera scale and needs an independent scene reference to establish metric accuracy.

## Reproduce the calibrated export

The [offline exporter](../../hexecutor/examples/depthpro2/depthpro2_export.py) accepts a saved unrotated color frame and the local `sensor-rig-calibration.json`. Put all outputs in an ignored local directory.

```sh
PY=hexecutor/.work/executorch-venv/bin/python
OUT=hexecutor/.work/depthpro2/arcturus-left-128
"$PY" hexecutor/examples/depthpro2/depthpro2_export.py \
  --src "$APPLE_SOURCE" --weights "$DEPTHPRO2_WEIGHTS" \
  --size 128 --image "$LEFT_RAW_PNG" \
  --arcturus-calibration "$CALIBRATION_JSON" --camera left \
  --out "$OUT/dp2s.pte" --hxfg-dir "$OUT/hxfg" --check
```

The export writes `x.f32`, `p.f32`, the PyTorch `want.f32`, the graph and constants, and `pinhole.json`. If a copied virtual environment has a stale `flatc` launcher, put the actual compiler on `PATH`. In the existing ExecuTorch wheel it is under `lib/python3.12/site-packages/executorch/data/bin` in the virtual environment.

The calibrated input used `frame-1794177634228728-23773-camera-1.png` from the Frame's passthrough viewer. Its SHA-256 is `a2e6b132a55be5837dde37b00d9361de36a4ce7c51ea6d8efe78150b4d389a9d`. The calibration SHA-256 is `f8a05f3ef8012b91084286457773a8d94c6ab7f1cc606975be8027fb6a5c5f29`. The output pinhole has `fx = fy = 76.754737` px and `cx = cy = 63.5` px. The calibrated export agreed with the upstream FP32 modules within `1.33e-6` max absolute inverse-depth error. The extracted HXFG CPU reference agreed with the rewritten model within `2.09e-6`.

## Run on the Frame

Check that the CDSP is idle and that no other custom client owns `/dev/cdsp_mem`. The 159.7 MiB constant pool goes above the first 128 MiB of private DDR. Build the engine, bootstrap and probe together with `HEXECUTOR_QDO_SLOTS=32` using [build_tensor.sh](../../hexecutor/build-scripts/build_tensor.sh). Use the verified loader already on the Frame. The originally staged bootstrap and probe cover only 128 MiB. The 512 MiB preflight with that probe returned a wrong-chunk error and exited before inference. The 32-slot build passed without a reboot.

The fixed package and runs are under `~/hexecutor-dp2/cdsp128-l9k9wj/` on the Frame. The engine matches the original `engine-c4.elf`.

| Artifact | SHA-256 |
| --- | --- |
| Seeded benchmark `dp2s_128.pte` | `6e118317cc3d543e97a45f852f61e73c29cc0b58dc66b86d3d520eb58d35f029` |
| Calibrated `arcturus-left-128/dp2s.pte` | `ec70ab5ad95367e618d4d6d0be285330e514222ec13178fa00b223e9a64c27aa` |
| Calibrated input `arcturus-left-128/x.f32` | `39c99fccc3d9809ace699899cea806a4b204b47c539c7eb7887a63320d5c8727` |
| File-input runner `hexecutor_et_run` | `8ea9496b8ee457dfa3f138289067c21cdb792fa0a774bf357bd5f1159deaa840` |
| Engine | `972683eddd263e11e0f447ce8f8c865662b7779b20a58e0f9f0e41a1c92691a9` |
| Bootstrap | `270b88182448a96c5742814c0ccedc346d2030ad3a850b1875d4bdb49fc945a8` |
| Chunk probe | `d3324be5141c5bfabd94034c1d8a054ec423a9b3f325662b0cc8603898f3f8a1` |
| Loader | `828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572` |

Grow the idle driver's lease to 512 MiB before starting the direct session. The matching probe runs before the session acquires its lease. The documented driver ioctl is `CDSP_MEM_IOC_SET_SIZE` (`0x40044301`, a native uint32 MiB value). Check the driver's reported size and follow the [session state checks](../hexecutor/operations.md#session-lifecycle). Recheck the state and lease after each session. [Driver and lease details](../hexecutor/dynamic-memory.md).

```sh
RUN=~/hexecutor-dp2/cdsp128-l9k9wj
A="$RUN/arcturus-left-128"
export HEXECUTOR_LOADER=~/fht-et/dsp/qd_loader.elf
export HEXECUTOR_ENGINE="$RUN/dsp32/engine.elf"
export HEXECUTOR_BOOTSTRAP="$RUN/dsp32/bootstrap.elf"
export HEXECUTOR_CHUNK_PROBE="$RUN/dsp32/chunk_probe.elf"
export HEXECUTOR_CHUNK_WINDOW_MB=512 HEXECUTOR_CDSP_MEM_MB=512

HEXECUTOR_ET_DUMP="$RUN/arcturus-dsp.f32" \
  "$RUN/hexecutor_et_run" "$A/dp2s.pte" \
  --input "$A/x.f32" --input "$A/p.f32" --warmup 3 --iters 20

HEXECUTOR_ET_REFERENCE=1 HEXECUTOR_ET_DUMP="$RUN/arcturus-ref.f32" \
  "$RUN/hexecutor_et_run" "$A/dp2s.pte" \
  --input "$A/x.f32" --input "$A/p.f32" --iters 1
```

The [ExecuTorch runner](../../hexecutor/executorch/hexecutor_et_run.cpp) accepts one `--input` per model input in model order. It validates file sizes before opening a session. The direct `hexecutor_tensor_run` takes the same graph, input binaries, and four DSP artifacts and writes `<output-prefix>profile.json` with the DSP timer, cycle counts and blocked-plan selection.

## FP16 and larger images

- HXFG allows 1,024 ops and 1,024 tensors per graph. Per-head projections initially generated 1,962 ops. The fused Q/K and output projections, folded RoPE and LayerScale, and single-head attention on token columns reduced the staged 128 px graph to 797 ops and 799 tensors. The current calibrated export has 793 ops and 795 tensors after folding the DPT fusion transposed convolutions and following 1×1 projections.
- The blocked FP16 planner failed on the staged 128 px graph even though its FP32 arena was 4.73 MiB, below the 7 MiB engine gate. The instrumented SDK simulator ran the planner against the HXFG metadata. Greedy resident-weight placement left 256 KiB workspace, and the first non-resident 384 → 384 projection that failed needs 292 KiB. A prototype that reserves space for streamed convolution weights before placing resident weights accepted both 128 px graphs in the planner, with 872 KiB workspace for the staged graph and 868 KiB for the calibrated graph. This was a planner-only experiment. The full model simulator run from the first session exited before inference.
- **FP16 gain error (about 1.09–1.12×).** Block 0's per-head attention scores reach 1.6e6–2.1e6 (blocks 1–11 stay below 100). The blocked FP16 plan stores the `Q·Kᵀ` tensor in FP16, so it overflows to `inf` (simulator dumps `op027`–`op042`); saturating instead gives gain 1.42 on the calibrated frame in a host emulation. The exporter now stores scores scaled by a power of two (`--score-scale`, default 1/1024) and `HX_ATTN_SOFTMAX` has a temperature field `p[3]` (float bits, axis 2 only) that multiplies them back in float before the mask. Host emulation of FP16 storage gives gain 0.9966 and 0.0014 max absolute error with this, against 0.997 and 0.0011 for plain FP16 rounding. Host tests (CTest `tensor_graph`, 64 AOT tests including a new score-scale case) pass and the FP32 reference matches PyTorch to 1.1e-6. The FP16 vector and scalar softmax changes and the new `test_tensor_fp16_sim.c` cases (steps 8 and 11) are **not yet run** on the simulator or Frame. Re-lower the graph, then compare the simulator output's gain against the FP32 reference before a device run.
- The current unsegmented 256 px graph has 793 ops, 795 tensors, 160.75 MiB of constants, and a **17.93 MiB** FP32 arena. The native reader rejected it at header validation because a graph is capped at 16 MiB.
- A four-strip 256 px experiment reproduced upstream inverse depth within `1.19e-6` max absolute error after assembling the PyTorch stripes. ExecuTorch lowered the body and four head stripes to five delegates. The body needs 9.87 MiB and the strips each need 4.62–4.70 MiB, about 28.50 MiB total. The body exceeds the 7 MiB FP16 gate and the whole set exceeds the direct session's shared 16 MiB DDR arena region. The partitioner is experimental. Reduce body liveness and address the total per-session arena before device timing.
- FP32 masters and packed FP16 weights account for much of the constant pool. A smaller deployment would require a format or packing change that preserves a supported CPU reference.

## Validation

All 64 ExecuTorch AOT tests and all 20 portable CTest cases passed. The full `make check` stops at a pre-existing broken link to the temporal-fusion report in `docs/README.md` after CTest. The standalone artifact policy tests and history audit passed.
