# NIR skeleton viewer integration handoff — 2026-11-13

## Scope and deployment target

The requested target is the native NIR hand viewer on the Frame CDSP:

```sh
frame_nir_hand_viewer --qnn16-dir ...
```

ONNX is only the interchange format and CPU numerical oracle. There is no new Python or CPU fallback in the live viewer. The learned network runs through QNN16; forward kinematics and the small 3x3 palm Procrustes solve run natively on the host. Crop geometry is used only by this host post-processing, not as an additional network input.

Work stopped at the user's request before native compilation, tests, replay, or device deployment.

## Source model and verified export

Checkpoint:

```text
/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/data/runs/skeleton_partial_v1/last.pt
SHA-256 ed3f243157c3f3e1213356937508652a75e67fdf4b8591e9cb35f250ee3730e7
```

Modified `hand-tracking/tools/export_nirhands_onnx.py` to:

- Export `SkeletonHandNet` through six CDSP-friendly outputs without FK/SVD.
- Include the checkpoint's 33-value mean profile and architecture in `manifest.json`.
- Check raw ONNX Runtime outputs and the Python host-decode oracle against PyTorch.
- Replace `adaptive_avg_pool2d` with the mathematically equivalent fixed 2x2 average pool. This fixed the dynamic-batch ONNX export failure.

Verified export artifacts are in:

```text
hand-tracking/.work/nirhands-skeleton-models/
```

The exporter completed successfully. Maximum absolute differences:

- Detector ONNX vs PyTorch: `4.291534423828125e-06`.
- Skeleton raw outputs: `2.1457672119140625e-06`.
- Skeleton decoded keypoints/disparity: `2.5331974029541016e-07`.

ONNX output order:

1. `keypoints_hm` [B,21,2]
2. `disparity_hm` [B,21]
3. `visibility_logit` [B,21]
4. `palm_points` [B,18]
5. `angles` [B,30]
6. `scale` [B]

Existing QNN16 artifacts, previously lowered and simulator-checked, remain in:

```text
hand-tracking/.work/qnn-v75/nirhands-skeleton/v2/
  skel.q16p
  skel.q16c
  skel.q16.json
```

QNN output order differs from ONNX:

1. auxiliary keypoints, 42 floats
2. auxiliary disparity, 21 floats
3. visibility logits, 21 floats
4. scale, 1 float
5. palm points, 18 floats
6. angles, 30 floats

Reuse these artifacts only after checking that the final exported graph contract still matches. Their earlier simulator result was `PASS`, 2509 ops, 286 matrix ops, 215988435 pcycles. This was not rerun during this integration session.

## Native code added or changed

Added:

- `hand-tracking/nirhands/include/nirhands/skeleton.hpp`
- `hand-tracking/nirhands/src/skeleton.cpp`

The new host decoder implements:

- Mean-profile scaling by network scale.
- Five three-bone chains with two angles per bone and the training model's rest frames/local rotations.
- Absolute palm points relative to palm entry 3 (joint 9).
- Double-precision Horn quaternion Procrustes using a small Jacobi eigensolver.
- Root-relative FK followed by root joint-9 decoding from auxiliary crop keypoint/disparity.
- Re-encoding into the existing normalized keypoint/disparity contract, including mirrored crops and the 20 mm depth clamp.

Changed `hand-tracking/nirhands/include/nirhands/pipeline.hpp` and `src/pipeline.cpp`:

- `Models::Run` now receives an optional `const Geometry*`.
- `Models::keypoints` now receives the sample `Geometry`.
- Skeleton manifests enable six-output ORT/QNN handling and load the 33-value mean profile.
- Old manifests without `arch` remain three-output keypoint models.
- QNN and ORT output orders are handled separately.
- The tracking pipeline continues to call `decode_prediction`, so tracking and display remain unchanged.

Updated callback signatures in:

- `runtime/tools/frame_nir_hand_viewer.cpp`
- `hand-tracking/nirhands/tools/nirhands_replay.cpp`

Replay's `--check-cpu` now passes the same crop geometry into the ORT oracle and accepts either three or six QNN outputs for final keypoint/disparity/visibility comparison.

These native edits have **not been compiled or numerically tested**.

## Required next steps

1. Finish `hand-tracking/nirhands/tests/test_npu_layout.cpp`. It still uses the old callback and `Models::keypoints` signatures and therefore will not compile as-is. Keep the existing three-output coverage and add a skeleton-manifest case with two inputs and six outputs in QNN order.
2. Add `hand-tracking/nirhands/src/skeleton.cpp` to the explicit `test_npu_layout` source list in root `CMakeLists.txt`. The viewer's wildcard source build already picks it up.
3. Add and register a portable `test_skeleton.cpp` with a synthetic profile, zero angles, known palm rotation/translation, and mirrored geometry. Independently verify the normalized keypoint/disparity result without model artifacts.
4. Build the native host targets and run `make check`. Fix compiler or numerical errors before deployment; none have been run yet.
5. Build Mac replay with the existing ORT under `hand-tracking/.work/ort/onnxruntime-osx-arm64-1.24.1` and compare a bounded recording against Python. CPU is only an oracle.
6. Verify/reuse the QNN graph, build the Frame engine/viewer, deploy the skeleton program as `nirhands_keypoints.q16p/.q16c`, and run bounded CDSP replay with `--check-cpu`.
7. Only after CDSP numerical validation, run a bounded headset viewer session and check overlay, timing, and clean DSP shutdown.
8. Update `docs/model-development/nirhands-live-seeding.md`, `docs/runtime/hand-tracking-viewer.md`, and `docs/model-development/nirhands-npu-handoff-2026-11-13.md` with only checks that actually pass.

## Workspace caution

The cowrie working tree contains extensive unrelated uncommitted and untracked work. Do not discard, overwrite, or stage it as cleanup. No commits were made. Temporary export output is at `hand-tracking/.work/nirhands-skeleton-models/export.log`; retain or remove it as appropriate during the next verification pass.

## Continuation, 22:30: host side verified, CDSP blocked on scratch size

Steps 1–5 are done. Step 6 stopped at the headset: the skeleton program does not fit the engine's scratch region. Steps 6 (CDSP part) and 7 have not run.

**Passed**

- `make check` on the Mac: 15 of 15 ctest tests and the Python checks.
  - `npu_layout` now takes the skeleton model directory as a second argument and covers six outputs in QNN order, plus a wrong-order case that must throw.
  - `nirhands_skeleton` (`hand-tracking/nirhands/tests/test_skeleton.cpp`) is new and needs no ORT or model files: synthetic profile, five rotation/mirror/clamp cases, bone-length and palm-rigidity checks.
- `decode_skeleton` against the Python oracle (`decode_skeleton_export`) on 2,000 random cases with the trained profile: largest disparity difference 6.5e-5; keypoints agree to float32 rounding (the one row over 1e-3 has every joint depth-clamped and keypoint values near 4,500).
- QNN16 `v2` program against the final `keypoints.onnx` on the trace input, in the simulator: largest difference 0.0019 (visibility logits); QNN output order as listed above. The simulator rerun on the current tree passed with identical outputs (2509 ops, 215,988,435 pcycles).
- Headset builds from the current tree in `~/nirhands-skel/`: `libhexecutor_host.a`, `hexecutor_qnn16_run`, `run/nirhands_replay` (with `NIRHANDS_QNN16`) and `runtime/bin/frame_nir_hand_viewer`. CPU replay ran there on 20 and 60 pairs.

**Bug fixed**

- `manifest_profile` in `nirhands/src/pipeline.cpp`: the `mean_profile` regex used `[^]]*`, which libc++ rejects, so every skeleton manifest failed to load. Now `[^\]]*`.

**Blocker: scratch region**

`hexecutor_qnn16_run` refuses the program with `Q16P scratch exceeds the engine region`. It needs 4,190,976 bytes; the region is `0x3ff000` (4,190,208), 768 bytes short. The total is the 2,073,344-byte persistent arena plus 2,117,632 bytes of staging for one `CONV3X3_S2` (op 1343, input `[1,9,10,768]`, output `[1,8,8,256]`). That staging is the generic float32 fallback's real need, and the region is bounded by the 4 MiB private page and the host guard at 64 MiB − 4 KiB, so this needs a decision in Hexecutor (for example no fallback reservation for ops that always take the HMX path, or a different region). Nothing was changed there. The DSP was never entered; control state was `(1,0,1)` before and after.

**Open: closed-loop agreement**

On 150 pairs of `live-1414.raw`, native replay against `nirhands_pipeline.py` with the skeleton checkpoint: raw joint error p50 1.4 mm, p90 15.8 mm, max 153 mm; track IDs differ on 23 frames. The old keypoint model on the same pairs: p50 2.3 mm, max 18 mm, no ID differences. While a track is stable the skeleton model agrees to 0.2–1 mm; the large errors are on tracks about to be dropped. The same code on the headset CPU against the Mac (60 pairs, 74 hand-frames) also differs: p50 1.6 mm, max 135 mm, no ID differences. Since the decoder and the ONNX graph each match their oracles, this looks like the tracking loop amplifying small input differences with this checkpoint, but that has not been shown.

**Checkpoint**

The exported `last.pt` (SHA above) is from epoch index 1, the second of 8 epochs, copied at 20:39 while training was running. The QNN16 program has the same weights. Pull `best.pt` from the PC, re-export and re-lower before judging quality on the headset.

**Headset state**

- `~/nirhands-skel/`: rsync of the tree plus `run/models` (skeleton export), `run/q16` (detector, the skeleton program as `nirhands_keypoints.*`, `engine.elf` built 21:50 with `-DQ16_NO_OP_PROFILE`) and `run/in` (`crops.raw`, `valid.raw`, `pairs20.raw`; the first 20 pairs have no hands).
- `~/nirhands-npu/` was not touched.

**Remaining**

1. Resolve the scratch limit, then run the program alone on the DSP and compare with `v2/sim/out*.f32`.
2. CDSP replay with `--check-cpu` on pairs that contain hands (pairs 25–85 of `live-1414.raw` do).
3. Headset viewer session.
4. Re-export from `best.pt`.

## Update, 22:35: scratch widened, program runs on the DSP, replay crashed and left a session installed

**Scratch region.** The 60 MiB split was a software choice inside private page 3 (16 MiB, [48, 64 MiB), cached like page 2). `SCRATCH_OFFSET` is now `0x3800000` and `SCRATCH_BYTES` `0x7ff000` in `hexecutor/native/qnn16/engine.c`; `executor.cpp` keeps Q16P constants below 56 MiB and accepts scratch up to `0x7ff000`. Constants for both programs total under 13 MiB. Engine and host library must be updated together. `make check` passes; the "Blocker" section above is superseded.

**DSP, program alone.** `hexecutor_qnn16_run` on `crops.raw`/`valid.raw`, three runs: all six outputs bit-identical to `v2/sim/out*.f32`; 131.9 ms DSP time (p50), 286 matrix ops; control `(1,0,1)` afterwards. Most of the time is `REDUCE_MEAN` and `L2NORM` on the generic path.

**DSP replay: failed.** `nirhands_replay --qnn16-dir q16 --check-cpu` on 12 pairs (pairs 44–55 of `live-1414.raw`) segfaulted at 22:28:58 before printing any pair. The cause is not known; the core dump is not readable. This was the first run of the replay's two-program QNN16 path with any model, so it may not be specific to the skeleton model or to the scratch change.

**Headset DSP state after the crash: not clean.** Control `(1,1,1)`, callback `0x87606000`, loader/payload/arena regions non-zero: the host died without closing its session. Nothing further was run. Recover the DSP (reboot if there is no safer way) before any other DSP work, then debug the replay under `gdb` on the host.

## Update, 22:45: DSP replay passes

After two reboots (each followed by `runtime/scripts/setup_frame.sh --qnn`; its chunk check needs `~/frame-dsp/quest-depth/.venv/bin/python` because system Python has no numpy):

- **Replay crash, cause found.** In `nirhands_replay.cpp` the `--check-cpu` recorder lambda was declared inside the `else` block and called by the runner after the block ended. It is now at function scope. The runner also takes a mutex, as the viewer does, because the pipeline runs two hands concurrently. The second crash and reboot came from rerunning with only the mutex added.
- **`--check-cpu` now compares decoded values.** For a skeleton model it used to compare the raw auxiliary outputs with the CPU's decoded joints. It now decodes the NPU outputs through a second `Models` and compares final keypoints, disparity and visibility.
- **Two programs through `hexecutor_qnn16_run`:** detector 9.1 ms, skeleton keypoints 128 ms DSP time, clean exit.
- **Replay on the DSP, 12 pairs (44–55 of `live-1414.raw`), `--check-cpu`:** 9 pairs with hands, 21 keypoint calls. Largest NPU-vs-CPU difference: keypoints 0.0014, disparity 0.0012, visibility logit 0.0067, detector maps 0.025. Control `(1,0,1)` afterwards.

Still open: the viewer session (step 7) and speed. At 128 ms per hand the skeleton program is not usable live; `REDUCE_MEAN`, `FORMAT` and `L2NORM` are 65% of the simulated cycles.

## Update, 23:10: skeleton keypoints 25.3 ms on the DSP

Three HVX tile forms, written by subagents in isolated copies and merged here:

| Kind | Change | Simulated pcycles before | After |
| --- | --- | --- | --- |
| `REDUCE_MEAN` (17 ops) | `qnn16_tile_reduce_mean_h` in `qnn16_tile_reduce.c`: mean over H of `[1,H,W,C]` | 71.7M | 0.9M |
| `FORMAT` (66 ops) | `format_blocked` in `qnn16_tile_cast.c`: blocked to blocked with different padding (48 ops were on the generic path) | 40.8M | 2.6M |
| `L2NORM` (5 ops) | `qnn16_tile_l2norm.c`: axis 1, whole 32-channel groups | 29.0M | 0.4M |

Each kind passes `--fuzz` on the merged tree (12: 14 layouts, 42: 2, 43: 1; none failed). `REDUCE_MEAN` and `L2NORM` were added to the `--compare` list; `--compare` itself was not run.

| Check | Before | After |
| --- | --- | --- |
| Whole program, simulator | 215,988,435 pcycles | 78,289,959 pcycles |
| Skeleton keypoints, Frame DSP (p50 of 5) | 128 ms | 25.3 ms |
| Detector, Frame DSP | 9.1 ms | 9.4 ms |
| Replay, 12 pairs, no CPU check, mean per pair | 267 ms | 65.7 ms |

- DSP outputs are bit-identical to the simulator's (`v3-sim/`).
- Simulator against ONNX on the trace input: keypoints 0.0005, disparity 0.0002, visibility 0.0019, scale 0.0002, palm points 0.0002, angles 0.0020 rad. Angles were 0.0005 before; the qf32 sums in the new forms differ from the generic float32 sums by an FP16 unit in places.
- Replay with `--check-cpu`, 21 keypoint calls: keypoints 0.0010, disparity 0.0008, visibility logit 0.0056. Control `(1,0,1)` afterwards.
- `make check` passes.

The engine on the Frame is `~/nirhands-skel/run/q16/engine-v3.elf` (`-DQ16_NO_OP_PROFILE`). What is left in the simulator profile: `CONV3X3` 16.9M, `CONCAT` 9.4M, `TRANSPOSE` 9.0M, `COPY` 7.0M, `SWISH` 5.7M. Two hands plus the detector are about 60 ms, still four times the 15 ms target.

## Update, 23:05: 8-epoch model on the DSP, viewer running

- **Model.** `skeleton_partial_v1/best.pt` (epoch index 7, synthetic-val MPJPE 17.3 mm) pulled from the PC. `export_nirhands_onnx.py` needed `_trunk(...)[:2]` because the model code now returns a third value; it refuses temporal checkpoints. Export in `.work/nirhands-skeleton-models` is unchanged; the new one is `.work/nirhands-skeleton-best` (raw outputs vs PyTorch 2.9e-6).
- **Lowering.** `compile_qnn.py ... --input-dim crops=1,2,128,128 --input-dim second_valid=1`, `htp_graph.py --no-source-names`, `trace.sh`, `qnn16_lower.py --write-tensor-config`, `trace.sh tensors`, `qnn16_program.py`, `qnn16_scratch.py`. Result in `.work/qnn-v75/nirhands-skeleton-best/v2/`: 2512 ops, scratch 4,192,896, simulator PASS at 78,295,197 pcycles; against ONNX keypoints 0.0004, disparity 0.0002, visibility 0.0023, scale 0.0003.
- **Output order changed.** This compile emits angles before palm points. `Models::keypoints` now takes scale, palm points and angles by size (1, 18, 30 floats) and throws on anything else; `test_npu_layout` covers three orders and an invalid set. `make check` passes.
- **Frame.** `~/nirhands-skel/run/models-best` and `run/q16-best` (detector, new keypoints, `engine.elf` = the 23:10 engine). Replay, 12 pairs, `--check-cpu`: keypoints 0.0030, disparity 0.0021, visibility logit 0.0106; control `(1,0,1)`.
- **Viewer.** `~/nirhands-skel/run_viewer.sh [seconds]` (needs sudo for `runtime/bin/stream_nir_pairs`). A 20 s session ran and exited cleanly: 600 frames submitted, `visible=0` (session reached state 3 only, headset not worn), DSP clean afterwards. Hands on screen have not been seen by anyone yet. A 600 s session was started at 23:04.

## Update, 2026-11-14 00:20: concat form, simulator only

A second session (the one that added the four op kinds and the scratch planner change) resumed after the 23:10 work and added one more tile form. Nothing in this section has run on the Frame.

- **`CONCAT` along H in tiles** (`qnn16_tile_concat.c`, hooked through `concat_tiles` in `exec.c`): copies rows vector by vector; declines when an input shares a VTCM block with the output. `--fuzz 19 --all` on the `best` program: 11 layouts, none failed, all bit-exact. The 13-input concat before the fuse convolution (op 1016) went from 8.4M to 0.33M pcycles.
- **Whole program, simulator, `nirhands-skeleton-best/v2`:** 78,295,197 to 69,278,298 pcycles; all six outputs bit-identical to `v2/sim/` (new outputs in `v2/sim-concat/`). If the device scales as before, that is roughly 22 ms per hand; not measured.
- **Poison test:** `hexecutor/tests/test_qnn16_scratch_poison.py` passes on the `best` program (and earlier on the L2 `v2/kp` program), so the staging blocks `qnn16_scratch.py` now assigns are dead when used.
- **`PW_SWISH` tile form** (`qnn16_tile_pw_swish.c`) fails the `--fuzz 39` screen on 7 of 8 layouts: 1 to 4 elements per op, up to 0.031 absolute. `--fuzz 27` shows the existing `SWISH` form failing the same screen the same way (up to 0.019 absolute, 16% relative), so this is the open `swish_core` issue from the NPU handoff, shared through `swish_qf`, not something specific to the new form. The formula itself is accurate to 2.3e-6 when evaluated in double precision, which points at the qf32 arithmetic or the `expf` reference on the simulator. Cause still not found.
- **Harness:** `QNN16_SIM_CFLAGS=-DQ16_SIM_PER_OP` prints one cycle line per operation; `--fuzz 19` now randomises the concat inputs.
- **Removed:** a duplicate `qnn16_tile_axis_h.c` (mean and L2 norm along H) that this session wrote before seeing the 23:10 forms. It was never built into an engine.

Per-op profile of what is left in the data-movement ops (simulator, with per-op printing): the 38 correlation transposes about 6.8M in total (already the vector form, about 180k each), 68 same-grid `MUL` 3.5M, 16 `S2D` on the `[1,17,130,1]` input strips 3.6M, 48 shifted `FORMAT` 1.8M. The structural cost is that QNN evaluates the 17-shift correlation as 17 separate shift, multiply, two-transpose and mean chains.

**Next:** build the engine from this tree, deploy it beside `engine-v3.elf` and confirm the DSP outputs match `v2/sim-concat/` before using it in the viewer.

## Update, 2026-11-14 23:15: `skeleton_tight96_full_v1` lowered, staged on the Frame

- **Model.** `skeleton_tight96_full_v1/best.pt`, 96 px, `pool: 3`, `tight_crop: true`. ONNX export in `.work/nirhands-skeleton-models`. QAIRT 2.50 context and lowering in `.work/qnn-v75/nirhands-skeleton-tight96/`, program in `v2/skel.*`: 2330 ops, scratch 4,031,360 bytes.
- **Fixes needed for this compile:**
  - `qnn16_lower.py --write-tensor-config` skips nodes that occupy several worker rows. QAIRT ran `Transpose_impl` node 1325 on 4 HVX threads (trace rows 509–512, kernel `0xf5eb60`, not the usual `0x261ca5a`), and those argument structs hold no tensor handles in the standard slots. Its perm (tensor 1285) is captured at node 3137 instead. The 25-word plan (1284) is unused.
  - `htp_graph.py` now emits `zap_tables` for 2.25+ `@BlockZap`. Each table is a header word `code << 16 | copy << 15 | n` followed by n blocks. With the copy bit set, the n entries are (source, destination) pairs: QNN copies source to destination, then clears the code region in the destination. The table holds `sources` and `blocks` (destinations). All 34 zaps reproduce QNN's VTCM writes exactly.
  - `qnn16_program.py` lowers each copy pair to a `MOVE` between 2 KiB direct VTCM tensors before the `ZAP`. No engine change.
- **Accuracy (ISS, trace crop) vs ONNX:** keypoints 0.0018, disparity 0.0009, visibility logit 0.020, scale 0.0003, angles 0.0020, palm points 0.0001. QNN's own FP16 result is about 10× further from ONNX (visibility 0.175), so `test_qnn16_program.py` fails its tolerance against QNN on visibility (0.186 vs 0.128) even though the program is closer to the float model.
- **Cost (fixed 23:50):** the first lowering ran at 412M pcycles. 50 convolutions with outputs that are not whole tiles took the float32 fallback: 48 stride-1 and 2 stride-2 (4×12 and 6×6 outputs; ops 949 and 1172 alone took 293M). Some of those outputs are 4-row views of 8-row blocks.
  - The HMX 3×3 path now stages partial last tiles whole: tile counts round up in `exec.c`, `qnn16_tile.c` and both fast forms in `qnn16_tile_stage.c`. `qnn16_tile_stage3x3_ok` accepts an output whose tile grid matches its blocks.
  - `hmx_conv3` saves each partial tile's output blocks in the op scratch before the HMX store and restores every lane outside the tensor afterwards, so data owned by other views of those blocks is untouched.
  - `qnn16_scratch.py` stages these ops.
  - Result: 72.7M pcycles (241 matrix ops). `--fuzz 6` (9 layouts) and `--fuzz 29` (5 layouts) pass, with no writes outside the logical output. Against ONNX: keypoints 0.0022, disparity 0.0009, visibility 0.016, angles 0.0024. Simulator outputs are in `v2/sim3/`. `make check` passes.
  - Largest remaining ops: L2NORM op 595 (8.3M), the 146-channel `TRANSPOSE` op 840 (7.2M), and two `TRANSPOSE` ops at 1.6M.
- **Frame.** `~/nirhands-skel/run/q16-tight96` (detector from `q16-best`, new keypoints, `engine.elf` built from this tree with `-DQ16_NO_OP_PROFILE`; the slow first versions are `engine-v1.elf` and `nirhands_keypoints-v1.q16p`) and `run/models-tight96`. The viewer binary was stale (no tight-crop code). The nirhands sources are synced and `frame_nir_hand_viewer` is rebuilt; the old one is `bin/frame_nir_hand_viewer.pre-tight96`. The Frame had rebooted, so `cdsp_mem` needs `sudo insmod`. Not run on the DSP yet. Run `~/nirhands-skel/check_tight96.sh` (bit-compare with the ISS), then `~/nirhands-skel/run_viewer_tight96.sh [seconds]`.

## Update, 2026-11-16 04:55: `skeleton_tight96_full_v2` lowered, staged on the Frame

- **Model.** `FutureMapper/skeleton_tight96_full_v2` `best.pt` (16 epochs, SHA-256 `850d59ad…4c1ac4f8762daf`), local copy in cowfish `data/runs/skeleton_tight96_full_v2/`. Same architecture as full_v1 (96 px, `pool: 3`, tight crop). ONNX in `.work/nirhands-skeleton-full-v2` (decoded vs PyTorch 5.5e-7).
- **Lowering.** Same steps as 23:15 (`compile_qnn.py` in `hexecutor-qairt-tools`, `htp_graph.py --no-source-names`, `trace.sh`, `qnn16_lower.py --write-tensor-config`, `trace.sh tensors`, `qnn16_program.py`, `qnn16_scratch.py`; `--known depth-model/sim/qairt_2_50_functions.json`). Output in `.work/qnn-v75/nirhands-skeleton-full-v2/v2/skel.*`: 2330 ops, scratch 4,031,360, simulator PASS at 72.7M pcycles. No tool changes were needed.
- **The trace crop is all zeros.** `trace/crops.raw` in the tight96, ft4-fg and full-v2 directories is zero, so earlier "ISS vs ONNX" numbers for 96 px programs were measured on a degenerate input (zero variance into GroupNorm; QNN's own FP16 output is about 0.02 off ONNX on it). Real crops are now in `.work/qnn-v75/nirhands-skeleton-full-v2/real/` (captured from `live-nir-20261114-1102-tight96.frnir` through the Python pipeline). **The program input is NHWC `[1,96,96,2]`**; feeding the NCHW crop gives garbage. On two real crops (NHWC) the simulator vs ONNX: keypoints 0.0037, disparity 0.0022, visibility 0.016, scale 0.0002, angles 0.0049, palm points 0.0002.
- **Frame.** `~/nirhands-skel/run/models-full-v2` and `run/q16-full-v2` (detector and `engine.elf` copied from `q16-ft4-fg`, md5 matches the local 14-11 23:51 engine). `run/in-full-v2` holds the two real NHWC crops and their ISS outputs. Not run on the DSP yet: the Frame had rebooted and `cdsp_mem` needs `sudo insmod`. Run `~/nirhands-skel/check_full_v2.sh` (bit-compare with the ISS on both crops), then `~/nirhands-skel/run_viewer_full_v2.sh [seconds]`.

## Update, 2026-11-16 07:10: combined output (`--output combined`)

- **Why.** On `full-v2-20261116-0459.raw.zst` against a FAH-L two-view reference (298 hands), the skeleton output (template hand at a predicted scale) is 15.2 mm MPJPE / 26.0 mm fingertips. The model's own auxiliary heatmap keypoints are 11.3 / 19.1 but have no bone constraints (bone lengths vary 15% per frame). The combined hand is 11.6 / 18.8 (benchmark held-out: 11.3 → 9.8 mm): the wearer's bones (per-side median of the heatmap hands' profiles), finger angles 0.5 skeleton + 0.5 heatmap, palm placed onto the heatmap palm. It does not fix individual finger articulation, which both heads flatten.
- **Code.** `skeleton.hpp/.cpp`: `measure_profile`, `inverse_kinematics`, `forward_kinematics_profile`, `combine_skeleton`, `ProfileEstimator` (ports of `nirhands.training.skeleton`). `Models::keypoints` optionally returns `keypoints_hm`/`disparity_hm` (QNN outputs 0 and 1, already computed). `TrackerConfig::output` = Skeleton (default) | Heatmap | Combined; tracking and crops always use the skeleton output; the profile updates from stereo hands with mean visibility ≥ 0.6, window 300, needs 20 per side (skeleton output is shown until then). `HandResult::combined` flags it. Native vs the Python reference on the 298 hands: < 0.0001 mm.
- **Frame.** Viewer rebuilt with `make bin/frame_nir_hand_viewer NIR_ORT=/home/steamos/nirhands-viewer/ort` (the Makefile default ORT path does not exist on the Frame); previous binary is `bin/frame_nir_hand_viewer.pre-combined`. Launch `~/nirhands-skel/run_viewer_full_v2_combined.sh [seconds] [record.raw.zst]` (QNN16, same as `run_viewer_full_v2.sh` plus `--output combined`). Throttled DSP replay of the recording: both hands combined by step 61, no errors. The Python `nirhands_pipeline.py` does not have this yet.

## Update, 2026-11-16 10:30: keypoints 33.0 → 18.2 ms on the DSP (engine change only)

- **Why.** The live viewer's capture-to-result latency was 111 ms median, 145 ms p90 (`rec/full-v2-20261116-0459.viewer.log`). The `full_v2` keypoint program took 33.0 ms per hand on the DSP, and both hands run back to back through one session. On the ISS, 47% of its 72.7M pcycles were 7 `L2NORM` ops (c = 24 misses the tile form's whole-channel-group rule) and 4 `TRANSPOSE` ops on direct tensors that fell to the generic per-element path: op 840 `(1,146,24,24)→(1,24,24,146)` and op 2277 `(1,21,24,24)→(1,24,24,21)` (perm 0,2,3,1, no tile form), and ops 66/67 `(1,48,96,2)→(2,48,96,1)` (`transpose_direct_b` required w % 64 == 0; w = 96).
- **Code.** `qnn16_tile_l2norm.c`: channel counts that are not a multiple of 32 take `l2norm_scalar`, the generic path's float32 arithmetic in its order with direct tile addressing (bit-identical; qf32 lanes were 22% faster but not bit-exact and moved the angle outputs by a median 0.3°, and HVX IEEE float needs `-mhvx-ieee-fp`, which the engine is not built with). `qnn16_tile_transpose.c`: `transpose_direct_b` finishes a width tail with scalar copies; new `transpose_direct_hwc_whc` for perm (0,2,3,1) on unpadded direct tensors (blocked 2D transpose).
- **Checks.** `test_qnn16_program_sim.sh --fuzz 42 --all` and `--fuzz 22 --all`: 0 elements differ in all 7 L2NORM and 54 TRANSPOSE layouts. Whole program on both real crops (`real/nhwc{20,50}.raw`): all six outputs bit-identical to the original engine, 72.70M → 50.76M pcycles. The original sources rebuild the deployed `engine.elf` byte for byte (md5 `0842a1e7…`); the new build is `engine-opt.elf` (md5 `15a51a4f…`, `-DQ16_NO_OP_PROFILE`). Note `real/sim50` from 04:48 was made from different inputs and is not a reference; fresh baselines are in `real/../l2opt/base{20,50}`.
- **Frame.** `~/nirhands-skel/run/q16-full-v2/engine-opt.elf` (beside the unchanged `engine.elf`). `hexecutor_qnn16_run` on both crops: outputs bit-identical to the ISS, 18.2 ms per call (was 33.0). Headless replay of the 04:59 recording through `Pipeline::step` (two hands, `--output combined`): step p50 72.3 → 41.5 ms, p90 122 → 76 ms. Launch: `~/nirhands-skel/run_viewer_full_v2_combined_opt.sh [seconds] [record.raw.zst]`. Not yet run live in the headset.
- **Still open.** The one-second wake rotation still adds ~50 ms to about one call in 13 (step p99 129 ms). The ORT CPU path is faster again (step p50 26 ms with 3 threads, `run_viewer_full_v2_combined_cpu.sh`) but loads CPUs 2–7, which overlap vrcompositor (0–4). Next ISS costs in this program: CONV3X3 (168 ops, 11.2M), COPY weight staging (605 ops, 6.9M).
