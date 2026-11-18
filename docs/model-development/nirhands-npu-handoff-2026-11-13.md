# Handoff, 2026-11-13 (updated 17:30): NIR hand viewer on the NPU

Follows [nirhands-native-handoff-2026-11-13.md](nirhands-native-handoff-2026-11-13.md). Goal: run `frame_nir_hand_viewer` on the Frame with both NIR networks on the CDSP (QNN16 path), not ORT.

## Update, 22:30: skeleton program does not fit the scratch region

The skeleton keypoint program (`.work/qnn-v75/nirhands-skeleton/v2/skel.*`) passes the simulator and matches its ONNX, but the host refuses it: it needs 4,190,976 bytes of scratch and the engine region is `0x3ff000`, 768 bytes short. Details and the op responsible are in [the skeleton integration handoff](nirhands-skeleton-integration-handoff-2026-11-13.md). It has not run on the DSP.

## Update, 17:30: keypoints 14.3 ms, detector 8.8 ms on the DSP

This section supersedes the 17:00 section below where they disagree. The under-15 ms target is met per network; one detector call plus two hands is about 37 ms, so the whole pipeline is not under 15 ms.

| Program | 17:00 | Now (Frame DSP, p50 of 10, no per-op profiling) | Check |
| --- | --- | --- | --- |
| Keypoints | 82.86 ms | **14.28 ms** (25.7M cycles with profiling) | Frame output bit-identical to the ISS; ISS vs QNN unchanged: keypoints 0.0034, disparity 0.0013, vis logit 0.043 |
| Detector | 86.05 ms | **8.81 ms** (15.8M cycles with profiling) | Frame output bit-identical to the ISS; ISS vs QNN max 0.0078, mean 0.0005 |
| Keypoints, ISS | 174M pcycles | 52.7M pcycles | |
| Detector, ISS | 294M pcycles | 58.2M pcycles | |

Both baselines were re-measured in the same session before the change. The DSP control state was `(1,0,1)` after every run. The programs are unchanged (`v2/kp`, `v2/det`); only the engine changed.

**What changed**

- **New tile forms, one source file each** (`hexecutor/native/qnn16/qnn16_tile_<name>.c`), behind hooks declared at the end of `qnn16_tile.h`. A hook returns 0 for a layout it does not support and the generic path runs.

  | File | Op | Device cycles before -> after (all ops of the kind) |
  | --- | --- | --- |
  | `qnn16_tile_resize.c` | `RESIZE_BILINEAR`, exact 2x in x, scale 0.5 with any offset in y | 76.8M -> 0.19M |
  | `qnn16_tile_transpose.c` | W<->C transpose in 32x32 blocks for any size; direct `[1,h,w,2] -> [2,h,w,1]` | 12.8M -> 0.59M |
  | `qnn16_tile_mul.c` | tensor x tensor; per-column and per-row factors | 10.0M -> 0.32M |
  | `qnn16_tile_decimate.c` | stride (2,2) | 8.2M -> 0.19M |
  | `qnn16_tile_softmax.c` | softmax over channels (qf32 polynomial exp, as SiLU) | 7.9M -> 0.09M |
  | `qnn16_tile_reduce.c` | `REDUCE_SUM` over H,W per channel and over W,C per row | 7.0M -> 0.08M |
  | `qnn16_tile_cast.c` | `CAST_F16` direct to direct; `FORMAT` direct to 1-channel tiles | 2.8M + 1.7M -> 0.06M + 0.32M |
  | `qnn16_tile_f32.c` | `SLICE_F32`, `WELFORD_MERGE` | 1.4M -> 0.25M |
  | `qnn16_tile_stage.c` | 3x3 staging, stride 1: loads the 10x4 window once per (tile, group) | `CONV3X3` 13.3M -> 8.6M |
  | `qnn16_tile_nearest.c` | `RESIZE_NEAREST` (detector) | 18.4M -> 0.34M |

- **Detector depthwise** (`qnn16_tile.c`): `DW3X3`, `DW3X3_S2` and `DW1X1` accept a partial last channel group (48, 72, 24, 8 channels) and store it through a channel mask. 128M -> 3.9M cycles.
- **`qnn16_tile_priv.h`** (new) holds the static helpers the tile sources share; they moved out of `qnn16_tile.c` unchanged. The build scripts compile `qnn16_tile*.c`.
- **`exec.c`**: `small_op` dispatches the new kinds and refuses any tensor in the scratch arena (HVX must stay in VTCM), except a `MUL` factor of at most 64 elements, which `qnn16_tile_mul` reads as scalars. The three broadcast `MUL` factors in the keypoint head live in the arena.
- **Simulator harness**: `test_qnn16_program_sim.sh --fuzz KIND [--all] PREFIX OUT INPUT...` runs only ops of one kind, one per operand layout, on random operands: generic path, then the tile path split in four parts as on the device. It also fails an op that changes anything outside its logical output (padding lanes, padding channels, bytes around a direct tensor). About 1-2 minutes per kind. `QNN16_SIM_ELF` names the test binary so two runs can share the tree. `--compare` now covers the new kinds.

**Keypoint profile now** (profiling build, 25.7M cycles):

| Kind | Ops | Mcycles | Note |
| --- | --- | --- | --- |
| CONV3X3 | 184 | 8.6 | staging is still most of it |
| GroupNorm chain (Welford partial/merge, moments, moves, instance norm, swish, add-swish) | 1,070 | 8.5 | about 1.6 is profiling overhead; most of the rest is per-op dispatch |
| COPY | 480 | 1.8 | |
| PW_SWISH | 1 | 1.7 | still generic; one `[1,1,1,128]` op |
| S2D, CONV3X3_S2 | 77 | 2.4 | stride-2 staging still uses the plain form |
| everything else | | 2.7 | |

**Detector profile now** (15.8M cycles): `PW1X1` 7.7M (150 ops, about 51k each), `S2D` 1.9M, `DW3X3` 1.8M, `DW3X3_S2` 1.1M, `DW1X1` 1.0M.

**Next steps, in order**

1. Detector `PW1X1` (half of the detector): input channel counts such as 9, 16, 24, 40, 72 take `qnn16_tile_pw`; check whether it or the HMX leaf is the better form.
2. Keypoint `CONV3X3`: stage once per band instead of once per output-channel op, and add the stride-2 form to `qnn16_tile_stage3x3_fast`.
3. GroupNorm chain: run consecutive independent ops of one kind as one parallel job (as COPY runs already do); most of its 8.5M is dispatch.
4. `PW_SWISH` through the HMX leaf followed by the HVX swish.

My estimate after steps 1-4 is about 7 ms per hand (13M cycles: `CONV3X3` near 4M, GroupNorm chain near 3M, `COPY` 1.8M as a floor because the weights are re-staged every run) and 4.5-5 ms for the detector, so about 19-20 ms for one detector call plus two hands. None of that is measured. Getting the whole pipeline under 15 ms needs something structural: both hands in one keypoint call, skipping the detector on most frames, or fusing QNN's GroupNorm split into one op at lowering time.

**Not done or not verified**

- **`SWISH` and `ADD_SWISH` fail the per-op compare screen** (`--compare`, now that in-place ops are compared): in all 136 + 32 ops checked so far, 10-20 of 4,096 elements differ from the exact `expf` path by more than the screen, up to 0.017 absolute and 9% relative. These are the forms from the 17:00 session, not changed here. End to end the outputs still match QNN as before, so this may be the reference being stricter than QNN's own SiLU, but the cause is not found and the "within 1 FP16 ULP" claim below does not hold on the device code as tested.
- The detector `--compare` run finished: 504 ops compared, none over the screen (`hand-tracking/.work/nirhands-qnn16/v3/det1.log`).
- The keypoint `--compare` run was still going at 17:45 (`v3/kp1.log`, 527 ops compared). It started before `qnn16_tile_mul.c` and `qnn16_tile_stage.c` were merged, so rerun it on the final tree. Everything it had reached except `SWISH` and `ADD_SWISH` was inside the screen. The new forms were each fuzz-checked, and the final tree was checked end to end.
- The fuzz inputs are uniform in [-4, 4); nothing checks the new forms on large values, infinities or NaN.
- Still open from 17:00: no real-data check (`nirhands_replay --check-cpu`), host library on the Frame not rebuilt, no poison test of the staging blocks. The viewer has not been run with `--qnn16-dir`.

**Artifacts**

- Mac: `hand-tracking/.work/nirhands-qnn16/v3/` holds `kp2/`, `det2/` (ISS outputs of the final tree) and the logs.
- Frame: `~/nirhands-npu/run/v3/engine.elf` (profiling) and `engine-noprof.elf`; outputs and summaries in `~/nirhands-npu/run/out3/` (`base_*` are the re-measured 17:00 engines, `v3*` the new ones). The Frame's source tree under `~/nirhands-npu/` was not synced.

```sh
# Mac: fuzz one kind (37 = RESIZE_BILINEAR; numbers are the enum order in program.h)
sh hexecutor/build-scripts/test_qnn16_program_sim.sh --fuzz 37 $W/v2/kp $W/fz $T/crops.raw $T/valid.raw
# Frame: as at 17:00 with v3/engine-noprof.elf
HEXECUTOR_RUNS=10 ../build/bin/hexecutor_qnn16_run $D/qd_loader.elf v3/engine-noprof.elf $D/bootstrap4.elf $D/chunk_probe.elf out3/v3kp \
  --program v2/kp in/crops.raw in/valid.raw
```

## Update, 17:00: keypoints run on the DSP, 82.6 ms

Target set by the user: inference under 15 ms. Not reached yet. This section supersedes the performance section below where they disagree.

| Program | Before | Now | Check |
| --- | --- | --- | --- |
| Keypoints, Frame DSP | not run (about 1 s estimated) | **82.6 ms** (84.3 ms with per-op profiling) | Bit-identical to the ISS run; ISS vs QNN: keypoints 0.0034, disparity 0.0012, vis logit 0.043 |
| Keypoints, ISS | 3,634M pcycles | 174M pcycles | `--compare` passed on ops 0–299 (79 ops, none over the screen) |
| Detector, Frame DSP | 86 ms | 86 ms (unchanged, not worked on) | Output identical to the earlier run |

The DSP control state was `(1,0,1)` after every run.

**What changed**

- **3×3 convolutions on HMX** (`Q16_CONV3X3`, `Q16_CONV3X3_S2`). QNN splits each layer into 8-row bands of 32 output channels. Workers copy each output tile's nine tap windows per input group into scratch tiles (HVX), and the existing 1×1 HMX leaf accumulates the K = 9 × groups tiles against QNN's weight tiles, which are already in that order. Only tile-aligned 2 KiB loads are used, the form already proven on silicon.
- **Scratch for the staged tiles** comes from `hexecutor/offline/qnn16_scratch.py IN.q16p OUT.q16p CONSTANTS.q16c`. QNN's plan uses all 8 MiB of VTCM, so the pass finds blocks that are dead while each convolution runs (next access is a whole overwrite: weight COPY, ZERO or HMX 1×1 store) and appends them to the word table (`p[3]` index, `p[4]` count). When fewer blocks are free than the whole op needs, the DSP stages a few output tiles per pass. All 211 convolutions got blocks.
- **HVX tile forms** for `WELFORD_PARTIAL` (closed form; the pass sets `p[0] = 1` after checking the weights are 1/(i+1), true for all 220), `INSTANCE_NORM`, `SWISH` and `ADD_SWISH` (qf32 polynomial, within 1 FP16 ULP of exact over the whole FP16 range in a NumPy model), and `S2D` with a partial input channel group.

**Measured on the Frame**

- **Per-op profiling costs about 1.5k cycles per op** (the uncached write of each op's cycle count): 84.27 ms with it, 82.61 ms without, over 2,027 ops. Build the timing engine with `QNN16_DEFINES=-DQ16_NO_OP_PROFILE`, and subtract about 1.5k cycles per op when reading `op_cycles` from a profiling build.
- **HMX costs about 42 cycles per tile pair** plus about 2.3k per op (fit over the detector's 108 `CONV1X1` ops). The keypoint network needs about 54k pairs, so about 2.3M cycles (1.2 ms) of HMX time per hand.
- **Scalar access to VTCM costs roughly 50–70 cycles per element.** This is why every remaining generic handler is slow, and why even a 256-byte `MOVE` costs 2.4k cycles.
- The DSP runs at about 1.84M cycles per ms.

**Keypoint profile now** (profiling build, 155M cycles):

| Kind | Ops | Mcycles | Note |
| --- | --- | --- | --- |
| RESIZE_BILINEAR | 12 | 76.8 | generic scalar; all are 2× upsamples |
| CONV3X3 | 184 | 13.3 | HMX; about 5/6 of this is the staging loop (`ld()` per vector) |
| TRANSPOSE | 18 | 12.8 | direct `[1,32,128,2]→[2,32,128,1]` and W↔C shapes the vector form rejects |
| MUL | 21 | 10.0 | tensor × tensor and broadcast along H or W on `[8,32,21]` |
| DECIMATE | 11 | 8.2 | stride (2,2) |
| SOFTMAX | 6 | 7.9 | 1,024 channels |
| REDUCE_SUM | 3 | 7.0 | |
| CAST_F16, FORMAT, PW_SWISH | 31 | 6.1 | |
| GroupNorm chain (Welford, merge, moments, moves, instance norm, swish) | 1,070 | 9.0 | about 1.6 of it is profiling overhead |
| COPY | 480 | 1.5 | 8.8 MB of weights per run |
| S2D, CONV3X3_S2, rest | | 3.0 | |

**Next steps, in order**

1. HVX form for the 2× `RESIZE_BILINEAR` (weights are 0.25/0.75; about half of the remaining time).
2. Faster conv staging: load each source row once and write all nine tap tiles from it, and stage once per band instead of once per output-channel op. Expect `CONV3X3` near 3M cycles.
3. HVX forms for the tail: the transposes, the three `MUL` shapes, `DECIMATE`, `SOFTMAX`, `REDUCE_SUM`, `CAST_F16`, `FORMAT` to a 1-channel tile tensor; `PW_SWISH` through the HMX leaf followed by the HVX swish.
4. Small float32 tensors in VTCM (`MOVE`, `SLICE_F32`, `WELFORD_MERGE`): read and write them through an HVX load or store of a stack vector instead of scalar VTCM access.
5. Run consecutive independent ops of one kind as one parallel job (as COPY runs already do), and fuse `S2D` into the stride-2 staging.
6. Detector: its depthwise ops miss the tile forms because output channel counts such as 48 are not multiples of 32 (`blocked()`); allow a partial last group with masked stores, and add a tile `RESIZE_NEAREST`.

My estimate after steps 1–5 is 5–6 ms per hand, so two hands plus the detector will be close to the 15 ms target, not comfortably under it. That estimate is not measured.

**Not done or not verified**

- `SWISH` and `INSTANCE_NORM` run in place, and the ISS `--compare` skipped them. The harness now saves and restores the input for in-place ops, but compare has not been rerun since. The only check on them so far is the end-to-end match against QNN.
- No real-data check (`nirhands_replay --check-cpu`). `~/nirhands-npu/run/in/` holds only a partial temp file of `live-1414.raw`; the copy did not finish.
- The host library on the Frame was not rebuilt. Its older validator ignores the new `p[3]`/`p[4]` fields, so it accepts the patched programs.
- The free-block analysis has no poison test yet (fill the staging blocks with garbage after each op and confirm outputs do not change).

**Files changed this session (cowrie, uncommitted)**

- `hexecutor/native/qnn16/`: `exec.c` (HMX 3×3 driver, new tile hooks), `qnn16_tile.c/.h` (staging, Welford, instance norm, swish, S2D mask), `program.c/.h` (staging-block validation, op comments), `engine.c` (`Q16_NO_OP_PROFILE`).
- `hexecutor/offline/qnn16_scratch.py` (new).
- `hexecutor/tests/test_qnn16_program_sim.c` and `hexecutor/build-scripts/test_qnn16_program_sim.sh`: `--stop N`, compare for the new kinds, in-place compare.

**Artifacts**

- Mac: `hand-tracking/.work/nirhands-qnn16/v2/` holds `kp.q16p/.q16c` and `det.q16p/.q16c` (patched programs), `out/` (ISS outputs) and `simfull.log`.
- Frame: `~/nirhands-npu/run/v2/` holds the same programs, `engine.elf` (profiling) and `engine-noprof.elf`. Outputs and summaries are in `~/nirhands-npu/run/out2/`.

```sh
# Mac: patch a program, run it on the ISS (about 3.5 min for keypoints), build the engine
W=hand-tracking/.work/nirhands-qnn16; T=hand-tracking/.work/qnn-v75/nirhands-keypoints/trace
python3 hexecutor/offline/qnn16_scratch.py $W/nirhands_keypoints.q16p $W/v2/kp.q16p $W/v2/kp.q16c
sh hexecutor/build-scripts/test_qnn16_program_sim.sh [--compare] [--stop N] $W/v2/kp $W/v2/out $T/crops.raw $T/valid.raw
QNN16_DEFINES=-DQ16_NO_OP_PROFILE sh hexecutor/build-scripts/build_qnn16.sh   # then scp build/qnn16/engine.elf
# Frame
cd ~/nirhands-npu/run && D=~/hexecutor-hand-viewer/dsp
HEXECUTOR_RUNS=5 ../build/bin/hexecutor_qnn16_run $D/qd_loader.elf v2/engine-noprof.elf $D/bootstrap4.elf $D/chunk_probe.elf out2/kp \
  --program v2/kp in/crops.raw in/valid.raw
```

## State at a glance (16:15, before the update above)

| Check | Detector | Keypoints |
| --- | --- | --- |
| Lowered Q16P (`hand-tracking/.work/nirhands-qnn16/nirhands_{detector,keypoints}.q16{p,c}`, same bytes as `.work/qnn-v75/nirhands-*/program.*`) | 835 ops | 2,027 ops |
| Host interpreter vs QNN simulator (`hexecutor/tests/test_qnn16_program.py`) | PASS, max 0.0078 (values up to 7.25) | PASS: keypoints 0.0031, disparity 0.0013, vis logit 0.047 |
| QNN fp16 vs ORT fp32 on the same trace input | max 0.0125, mean 0.001 | keypoints 0.0028, disparity 0.0015, vis logit 0.050 |
| Hexagon ISS, whole program with HMX (`test_qnn16_program_sim.sh`) | PASS, 294M pcycles | PASS (same errors as host), **3,634M pcycles** |
| **Frame DSP** (`hexecutor_qnn16_run`) | **Runs. Bit-identical to the ISS, 86 ms/call** | **Not run yet** (see the risk below) |
| Real-data check (NPU vs ORT inside the pipeline) | not run | not run |

The trace inputs are synthetic: the keypoint crop is uniform noise in [0,1], so real-data validation is still needed. The new replay `--check-cpu` mode (below) does this.

Notes:
- The detector trace has no `input_list.txt`. Test with a copy holding `image:=/ctx/trace/image.raw`.
- ISS outputs are in `hand-tracking/.work/nirhands-qnn16/sim/{det,kp}/`.

## Performance: the blocker

The NPU is currently much slower than ORT on the CPU (detector 8.2 ms; keypoints about 40 ms per hand).

- **Detector on DSP:** 86 ms, 158M cycles. By op kind:
  - DW3X3_S2: 46%, 28 ops at about 5.2M cycles each
  - DW3X3: 25%
  - RESIZE_NEAREST: 12%
  - DW1X1: 10%

  These all take the float/generic path, not the HVX tile leaves. For example, the DW3X3_S2 strips have `x=[1,9,58,256] → y=[1,8,56,48]`, so `qnn16_tile_dw3x3_s2_ok` (`x.c == 4*y.c`, `hexecutor/native/qnn16/qnn16_tile.c:134`) rejects them. The 256-channel input is presumably padded or channel-sliced. Per-op breakdown: `python3 hand-tracking/.work/nirhands-qnn16/opcycles.py PROGRAM.q16p SUMMARY.json [program_index]` (the summary is the runner's `<out>summary.json`).
- **Keypoints on ISS:** 3.63G pcycles. The detector's DSP/ISS cycle ratio was about 0.54, so expect about 2G cycles, roughly 1 s per hand, on the device. By kind:
  - CONV3X3_S2: 2,150M (27 ops), the biggest target
  - INSTANCE_NORM: 375M
  - CONV3X3: 356M
  - WELFORD_PARTIAL: 204M
  - ADD_SWISH: 201M
  - SWISH: 189M
  - S2D: 48M
  - RESIZE_BILINEAR: 50M

  None of these has a tile form yet.
- **Risk:** Mercury notes say DSP wakes over about 2.5 s risk the `(2,0,8)` control-state hang, which needs a reboot. A single keypoint call should be about 1 s, but run it alone first with `HEXECUTOR_RUNS=1`, and check `depth_state.py` afterwards.

**Recommended order:**
1. Run keypoints once on the device for correctness.
2. Add tile forms or fix the eligibility for CONV3X3_S2, depthwise S2/DW3X3/DW1X1 with padded channels, INSTANCE_NORM/Welford/Swish, and the resizes.
3. Only then switch the viewer to `--qnn16-dir`.

Until then the viewer's CPU/ORT mode is the usable one: about 15–20 hand updates/s.

## Code changes this session (cowrie, uncommitted)

- **`runtime/Makefile`:** new `nir-hand-viewer-host` target that builds only `hexecutor_host`. `nir-hand-viewer` used `hand-viewer-host`, which also needs `mercury_executor_networks` and fails without Mercury assets.
- **`hand-tracking/nirhands/tools/nirhands_replay.cpp`:**
  - adds `--qnn16-dir DIR --loader --engine --bootstrap --chunk-probe` (compile with `-DNIRHANDS_QNN16` and link `build/libhexecutor_host.a`);
  - adds `--check-cpu`, which runs ORT on every NPU input and prints per-output max and mean-of-max |NPU − CPU| on stderr.

  The Mac build without the define still works.
- **This doc.** No other files were touched.

The previous agents' already-present work, unchanged:
- `runtime/tools/frame_nir_hand_viewer.cpp` (with `--qnn16-dir`);
- `Models(dir, Run)` NPU mode in `nirhands/src/pipeline.cpp`;
- `nirhands/tests/test_npu_layout.cpp`;
- the new QNN16 handlers (Welford, group moments, instance norm, swish and others) in `hexecutor/native/qnn16/`;
- `runtime/scripts/run_nir_hand_viewer.sh`.

## Frame state (`ssh frame`)

- **`~/nirhands-npu/`:** an rsync of the cowrie tree (no `.git`/`.work`), with:
  - `build/libhexecutor_host.a` and `build/bin/hexecutor_qnn16_run`;
  - `runtime/bin/frame_nir_hand_viewer`, built 15:56 with `make nir-hand-viewer NIR_ORT=/home/steamos/nirhands-viewer/ort OPENXR_INCLUDE=/home/steamos/frame-spatial-containers/third_party/openxr/include`;
  - `run/q16/`: both programs and `engine.elf`, built locally 15:40 after the last kernel edits;
  - `run/in/`: `image.raw`, `crops.raw`, `valid.raw`, and `live-1414.raw` (copy in progress at handoff time);
  - `run/nirhands_replay`, the NPU build, which was being compiled at handoff time. Check that it exists; if not, rerun the build command below.
- **DSP:** `cdsp_mem` loaded (128 MB, iova 2684354560). Control state `(1,0,1)`, regions zero after the detector run. Loader, bootstrap and chunk probe come from `~/hexecutor-hand-viewer/dsp/`.
- **Viewer status:** not launched in the headset this session.
- **Capture stream format:** `stream_nir_pairs` now defaults to compressed FRNIR002 (another session's transport change). The native viewer and replay need FRNIR001, which `run_nir_hand_viewer.sh` requests with `--raw`. The binary in `/tmp/nirhands-live/` predates that change.

## Commands

The detector on the DSP, as it was run and passed:

```sh
cd ~/nirhands-npu/run && D=~/hexecutor-hand-viewer/dsp
HEXECUTOR_RUNS=3 ../build/bin/hexecutor_qnn16_run $D/qd_loader.elf q16/engine.elf $D/bootstrap4.elf $D/chunk_probe.elf out/det \
  --program q16/nirhands_detector in/image.raw
```

Next is the keypoint program on the DSP, run once and alone, then compared with `.work/qnn-v75/nirhands-keypoints/trace/Result_0/*.raw`:

```sh
HEXECUTOR_RUNS=1 ../build/bin/hexecutor_qnn16_run $D/qd_loader.elf q16/engine.elf $D/bootstrap4.elf $D/chunk_probe.elf out/kp \
  --program q16/nirhands_keypoints in/crops.raw in/valid.raw
python3 ~/frame-dsp/quest-depth/wave1-20261101/frame/bench/depth_state.py | head
```

Build the NPU replay on the Frame:

```sh
cd ~/nirhands-npu/hand-tracking && ORT=/home/steamos/nirhands-viewer/ort
g++ -O2 -std=c++17 -DNIRHANDS_QNN16 -Inirhands/include -I../hexecutor/include -I$ORT/include nirhands/src/*.cpp \
  nirhands/tools/nirhands_replay.cpp ../build/libhexecutor_host.a -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib -lpthread -o ../run/nirhands_replay
```

The real-data check (both models), once keypoint speed allows; keep `--max-pairs` small for now:

```sh
cd ~/nirhands-npu/run && ./nirhands_replay --models ~/nirhands-viewer/models --input in/live-1414.raw --max-pairs 20 --quiet \
  --qnn16-dir q16 --loader $D/qd_loader.elf --engine q16/engine.elf --bootstrap $D/bootstrap4.elf --chunk-probe $D/chunk_probe.elf --check-cpu
```

Headset run, from an `ssh -t frame` terminal where the user types the sudo password:

```sh
cd ~/nirhands-npu/runtime && ./scripts/run_nir_hand_viewer.sh --models ~/nirhands-viewer/models    # CPU/ORT
# NPU, after the speed work:
./scripts/run_nir_hand_viewer.sh --models ~/nirhands-viewer/models --qnn16-dir ~/nirhands-npu/run/q16 \
  --loader $D/qd_loader.elf --engine ~/nirhands-npu/run/q16/engine.elf --bootstrap $D/bootstrap4.elf --chunk-probe $D/chunk_probe.elf
```

Simulator runs (Mac):
- Set `HEXECUTOR_SDK_ROOT=/Users/owenelliott/Documents/projects/frame-spatial-runtime/depth-model/.work/hexagon-sdk/6.6.0.0` and `HEXECUTOR_TOOLCHAIN_IMAGE=depth-model-hmx-tools`.
- The keypoint ISS run takes about 13 minutes.
- Don't run two simulator jobs at once: both write `hexecutor/native/build/qnn16/test_program.elf`.

## Remaining work

1. Keypoint program on the DSP: one run, check its accuracy and time.
2. Speed: tile or HVX forms for the ops listed above, with the ISS bit-exact check and a host check after each one.
3. `nirhands_replay --check-cpu` on `live-1414.raw`: NPU vs ORT per call on real crops, then tracking equivalence.
4. Headset validation of the viewer: overlay, world lock, rate, clean Ctrl-C. Do CPU mode first; NPU mode once fast.
5. Update `docs/runtime/hand-tracking-viewer.md`. Its line 11 says the keypoint Q16P is pending, which is now stale, and it should mention `nir-hand-viewer-host`.
