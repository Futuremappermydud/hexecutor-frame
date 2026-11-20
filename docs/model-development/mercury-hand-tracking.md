# Mercury stereo hand tracking

The OpenXR hand viewer can run Monado's Mercury detector, keypoint model and native stereo tracker on the rectified compositor pair. The C++17 viewer selects the QNN16 run-list backend with `--mercury-dir`. A separate ExecuTorch example exports the two networks as HXFG for portable and DSP comparisons. These are distinct model and runner contracts. [Viewer and launch options](../runtime/hand-tracking-viewer.md), [ExecuTorch Mercury exporter](../../hexecutor/executorch/mercury/export_executorch.py)

## Model and compiler inputs

1. **Sources.** Monado `3acbda6b7dacbe014749020e8543fded54ef076b` (`src/xrt/tracking/hand/mercury`, BSL-1.0) and the `monado/utilities/hand-tracking-models` files `grayscale_detection_160x160.onnx` (SHA-256 `1f1a039a…ed36`) and `grayscale_keypoint_jan18.onnx` (`40c0daa5…d1d5`), both matching their LFS pointers. All are under ignored `samples/hand-tracking/.work/external/`. **The models repository has no license file**; establish the model license before distributing converted artifacts or shipping.
2. **QNN compilation.** `hexecutor/offline/compile_qnn.py` now accepts ONNX (`--input-dim` optional). Both nets compile for V75, SM8650, 8 MB VTCM, four HVX threads, O3, in a local `hexecutor-qairt-tools` image (Ubuntu 22.04, Python 3.10 with `libpython3.10`, QAIRT 2.50's pinned converter packages, clang 14). Detection lowers to 997 nodes and keypoint to 1,138.
3. **Accuracy oracle.** `samples/hand-tracking/tools/mercury_qnn_oracle.py` builds Mercury-normalized inputs from an image, runs ONNX Runtime and the x86 HTP simulator, and compares the two. On `right_hands.jpg`: detection max error 5e-4; keypoint heatmap max 0.0020 with all 21 argmaxes equal, depth heatmap 0.0040, `scalar_extras` 0.0099 of 14.5, curls 0.0033. The QNN FP16 graph is therefore an acceptable target. The context returns `heatmap_xy` as NHWC `[1,22,22,21]`.
4. **Context decoding.** `samples/depth-model/tools/htp_graph.py` now reads QAIRT 2.50 contexts: dictionaries after extra words, the `0xf, 0xfffc0003` I/O preamble with any input count, one checkpoint word when flags are 3, and `@BlockZap` block tables. The legacy depth context still decodes byte-identically. Detection: 1,005 ops, 868 tensors. Keypoint: 939 ops, 938 tensors.
5. **Run lists.** The 2.50 trace hook (`samples/depth-model/sim/qnn_trace_hook.cpp`) runs unchanged on these contexts. Detection executes 530 main and 179 vector-list entries; keypoint executes 516 and 138. Traces are under `samples/hand-tracking/.work/qnn-v75/mercury-*/trace/`.

## Kernel census

All activations are FP16 crouton in TCM. Weights are packed FP16 (`Xpe`/`Xwe`), and bias/scale are FP32 (`Fi`). Counts are detection + keypoint.

| Kernel | Count | Notes |
| --- | --- | --- |
| `ConvLayer.fp16.s1.tcm` | 158 + 216 | HMX FP16 stride 1, including 1×1, depthwise and Gemm-as-conv; Clip fused |
| `ConvLayer.fp16.s2.tcm` | 15 + 7 | Stride 2 |
| `ConvLayer_SpaceToDepth_stride2.tcm` | 66 + 54 | Input staging for stride-2 convolutions |
| `ConvLayer.opt.inconv_im2col_cast.tcm` | 10 + 8 | FP32 single-channel image to FP16 im2col |
| `ConvLayer.opt.weights_to_vtcm` | 192 + 165 | DMA weight/bias copies, with `@DmaCheckpointSet/Wait` |
| `*InputSlicePad`, `*InputSlice`, `*OutputSlice`, `*OutputCast_fp16_to_fp32` | 14 + 21 | External I/O |
| `ForceFormat_*`, `Reshape`, `Cast_*`, `Concat`, `AlignedSlice` | 6 + 52 | Layout and format moves |
| `ChannelShuffle`, `Transpose.2D`, `x2s_opt`, `AvgPool_with_pad`, `Mul.fp16`, `ReluMinMax.fp16`, `Sigmoid.fp16` | 3 + 12 | Small elementwise and data movement |
| `@BlockZap`, `@SyncOp`, `@Dummy*`, `zero_to_vtcm` | scheduling | TCM zeroing and list synchronization |

Depth's handlers are u8/u16, and the MediaPipe tensor engine does not consume QNN programs, so these FP16 variants have their own handlers in `hexecutor/native/qnn16/`; see [QNN FP16 handlers](#qnn-fp16-handlers).

## Viewer evidence and validation

In a bounded worn-viewer run on 2026-11-11, `frame_hand_viewer --mercury-dir` tracked one hand and restored the DSP at exit. The wearer reported incorrect hand rotation and joint motion. A captured 616×616 frame exposed aliasing in the original 160×160 detector resize. The antialiased C++ reduction matched the offline resize to 0.00054 mean normalized input difference on that frame. After the fix the detector reached 0.997 confidence and ran keypoints. The keypoint adapter uses QNN output order: depth, extras, XY heatmap, curls.

The separate [HXFG / ExecuTorch simulator evidence](#hxfg--executorch-simulator-evidence) covers paired engine comparisons. Tracker articulation, two-hand geometry, target-firmware accuracy and cleanup remain device checks.

## QNN FP16 handlers

The trace-oracle fixtures check the QNN16 handlers' per-operation VTCM writes. The chosen shape determines tile, HMX or float/generic execution. The dates and timings below describe paired hardware runs.

**Oracle.** `hexecutor/offline/qnn16_oracle.py` reads the decoded context and a trace-hook run (`QD_TRACE_LOG`, `QD_WRITES`).
- **Run-list mapping.** In 2.50 FP16 graphs, list 1 holds the flags-4/6 operations. List 0 holds every other executable operation in serial order. The `InputSlicePad` stub (`0xebc680`) is the only executor-only entry, found by alignment. Every function offset maps to one kernel type.
- **Replay.** Replaying the writes reproduces every simulator output exactly.
- **Padded tensors.** `max_dimensions` is the whole padded extent, and `padding` locates `dimensions` inside it. For example, a `[10,66,32]` input with padding `[7,3]` has a 24×72 extent and 54 blocks. The oracle used to add the padding a second time, which misplaced every padded block. That bug is what made the 3×3 convolutions look wrong.
- **Fixtures.** The tool exports per-operation fixtures under the ignored `samples/hand-tracking/.work/qnn16-fixtures/`.

**Formats.** These were confirmed against the ONNX weights, so no repacking is needed:
- An FP16 crouton block is an HMX activation tile: 16 pixel pairs × 32 channels × 2. Row `y`, column `x` and channel `c` are at half `((y·2 + x/2)·32 + c)·2 + x%2`.
- `Xpe` weights are `[co/32][ci/32]` tiles of `[16 ci pairs][32 co][2]`.
- `Fi` is one 256-byte `mxmem2` table per 32 output channels. Words 0–31 hold the scale in their low half, and words 32–63 hold the bias in their high half. Clipped activations add a second table, whose words 0–31 hold the upper clamp (6.0) in their **high** half. HMX applies it with a second `cvt.hf = acc(6)` step. Parameter 6 was found by sweeping candidates against the oracle.
- **t=514 depthwise `Xpe`** (`[3,3,2,C]`) packs 576 halves per 32-channel group: `[group][tap ky·3+2−kx][16 channel pairs][even, 0, 0, odd]`.
- **t=544 pairwise 1×1** (`[1,1,4,C]`) combines channels `c` and `c+32` using two weights from the four-channel group.
- **Stride-2 `Xwe`** holds nine diagonal 32×32 `[ci pair][co][2]` tiles per channel group, group-major. The tile for tap `ky·3+kx` is `{1,4,0,7,8,6,3,5,2}`.
- **Space-to-depth staging** (`ConvLayer_SpaceToDepth_stride2`) writes channel `((c/32)·4 + q)·32 + c%32` of output `(y, x)`. The source is row `2y + q/2`, column `2x − q%2` of its 32-channel input, and is zero outside it. The stride-2 convolution then reads tap row `ky` at phase `ky%2`, row `y + (ky==2)`, and tap column `kx` at phase `(kx==1)`, column `x + (kx>0)`.
- **External I/O parameters** are the operations' `s4` shape values:
  - `InputSlicePad`: `[pad before][start][output shape][slot][source shape]`. Mercury's bands use origin `(16k − 1, −1)`.
  - `InputSlice`: the same without padding.
  - `OutputSlice`: `[output shape][origin][slot]`.
  - `OutputCast`: `[shape][shape][origin][slot]`.

  Slots follow the input and output order in `execution_metadata.yaml`.
- **`@BlockZap`** only writes zeros, and only within its tables' blocks. Each table's `code` selects a region of every block:
  - `0xC0 | k` clears the last k rows (k = 3, 4, 5 or 6 here).
  - `0x50`, `0x4C` and `0x54` clear columns 3, 2–3 and 1–3. `0x50` always shares its blocks with a `0xC5` table, so only its column-3 effect on rows 0–2 is observed.

  The lane mask `0x55555555, 1` did not change the result in any traced zap. `zap_mode` is the operation's serialized byte count, not a region selector.

**Accuracy criterion.** QAIRT's x86 simulator evaluates HMX in exact FP32 before rounding. The Hexagon ISS models the hardware accumulator, which QNN on silicon also uses, so bit equality with the x86 oracle is not the target. Each valid convolution output half must be within 2^-10·Σ|a·w·scale| + 1 ULP + 2^-14. Padding lanes are not compared. The x86 simulator's FP16→FP32 output cast is within about one float32 ULP of exact conversion (a few ULPs on FP16 subnormals). That is the only non-exact non-matrix comparison.

| Kernel (detection + keypoint count) | Handler | Check | Result |
| --- | --- | --- | --- |
| `ConvLayer.fp16.s1` 1×1, `t`=0/3, 1 or 2 tables (93 + 114) | `qnn16_conv1x1` (HMX) | ISS, `test_qnn16_sim.sh` | 225 fixtures pass; worst 0.523 of tolerance |
| `ConvLayer.fp16.s1` `t`=544 pairwise 1×1 (31 + 56) | `qnn16_depthwise1x1_f16` | `test_qnn16_depthwise_qd.py` | Pass |
| `ConvLayer.fp16.s1` `t`=514 depthwise 3×3 (34 + 41) | `qnn16_depthwise3x3_f16` | `test_qnn16_depthwise_qd.py` | Pass; worst 0.720 |
| `ConvLayer.fp16.s1` `t`=0 3×3 heads, `Xwe` 160→63 channels (0 + 5) | `qnn16_conv3x3_f16` | `test_qnn16_conv3x3_qd.py` | Pass; worst 0.161 |
| `ConvLayer_SpaceToDepth_stride2` (66 + 54) | `qnn16_space_to_depth_stride2_f16` | `test_qnn16_stride2_qd.py` | Bit exact |
| `ConvLayer.fp16.s2` `t`=514 (15 + 7) | `qnn16_depthwise3x3_stride2_f16` | `test_qnn16_stride2_qd.py` | Pass; worst 0.545 |
| `ConvLayer.opt.inconv_im2col_cast` (10 + 8) | `qnn16_tensor_im2col_cast` | `test_qnn16_im2col_qd.py` | Bit exact |
| `ForceFormat_*` (2 + 38), `Cast_*` (0 + 11) | `qnn16_tensor_copy_format`, casts | `test_qnn16_format_qd.py`, `test_qnn16_cast_qd.py` | Pass |
| `ChannelShuffle`, `AlignedSlice`, `Concat`, `x2s_opt`, `Transpose.2D`, `AvgPool_with_pad` (5 + 8) | data-movement leaves | `test_qnn16_moves_qd.py` | Pass |
| `ReluMinMax` (0 + 2), `Sigmoid` (1 + 0) | `qnn16_tensor_minmax`, `qnn16_tensor_sigmoid` | `test_qnn16_relu_qd.py`, `test_qnn16_sigmoid_qd.py` | Pass |
| `Mul.fp16` by a scalar (0 + 4), `Reshape` (1 + 1), `zero_to_vtcm` (1 + 1) | `qnn16_tensor_mul`, `qnn16_gather_f16`, zeroing | `test_qnn16_io_qd.py` | Pass |
| `*InputSlicePad` (10 + 8), `*InputSlice` (0 + 2) | `qnn16_window_f32` | `test_qnn16_io_qd.py` against the model inputs | Bit exact |
| `*OutputSlice` (0 + 9), `*OutputCast_fp16_to_fp32` (4 + 3) | `qnn16_place_f32`, `qnn16_tensor_f16_to_f32` | `test_qnn16_io_qd.py` against `Result_0` | All 8 outputs reassembled |
| `@BlockZap` (80 + 1) | `qnn16_zap_region`, `qnn16_zap_blocks` | `test_qnn16_io_qd.py` | Bit exact; no stray writes |
| `ConvLayer.opt.weights_to_vtcm` (192 + 165) | `qnn16_copy_chunks` (synchronous) | indirect | Not checked directly: the trace has no pool-2 source. The convolutions above read the copied weights correctly. |

`@DmaCheckpointSet/Wait`, `@SyncOp` and `@Dummy*` write no data of their own; DMA results land at the matching wait. In the 1×1 tolerance, weight rows past the packed input-channel extent (for example the second K tile of 44-channel `Xpe`) hold stale VTCM bytes and can be NaN. They are excluded. The fixtures were regenerated after the padding fix and now cover every 1×1 operation (the earlier set had 28 + 46 linear and 53 + 27 clipped).

The lowering and engine below use these handlers. One correction found while wiring them in: `Xpe` packs `ceil(ci/2)` input-channel pairs per 32-output group, so when `ci` is not a multiple of 32 (keypoint's 42-input `lastKeypoints` FC slices, the 12-input t=3 layers) the group stride is not K × 2 KiB. Those ops use the scalar `qnn16_pointwise_f16` instead of the HMX leaf. The earlier ISS fixtures passed only because those inputs were zero in the trace.

```sh
K=samples/depth-model/sim/qairt_2_50_functions.json
for g in detection keypoint; do
  C=samples/hand-tracking/.work/qnn-v75/mercury-$g/context T=samples/hand-tracking/.work/qnn-v75/mercury-$g/trace
  for t in cast conv3x3 depthwise format im2col moves relu sigmoid stride2; do
    python3.14 hexecutor/tests/test_qnn16_${t}_qd.py $C $T --known $K
  done
  python3.14 hexecutor/tests/test_qnn16_io_qd.py $C $T --known $K --inputs samples/hand-tracking/.work/mercury-oracle
  python3.14 hexecutor/offline/qnn16_oracle.py $C $T --known $K \
    --conv1x1 samples/hand-tracking/.work/qnn16-fixtures/$g-conv1x1.bin \
    --conv1x1-clip samples/hand-tracking/.work/qnn16-fixtures/$g-conv1x1clip.bin \
    --conv1x1-small samples/hand-tracking/.work/qnn16-fixtures/$g-conv1x1-small.bin
done
HEXECUTOR_SDK_ROOT=... HEXECUTOR_TOOLCHAIN_IMAGE=depth-model-hmx-tools sh hexecutor/build-scripts/test_qnn16_sim.sh
```

The trace needs `QD_WRITES` in addition to `QD_TRACE_LOG`. Build the hook with clang++ in `hexecutor-qairt-tools`. Keypoint's trace has no `Sigmoid`, so that one check reports nothing to compare.

## Q16P program and engine

The Q16P and Q16C artifacts carry the run list and constants for the native QNN16 session.

- **Constants.** `hexecutor/offline/qnn16_lower.py --write-tensor-config`, then a simulator run with the trace hook's `QD_TENSOR_CONFIG`/`QD_TENSORS`, dumps every pool-2 input. Sizes come from `max_dimensions`: a `[256, 8]` `Xpe` is stored as `[256, 32]`, and sizing from `dimensions` silently truncated it. All 192 + 165 weight dumps equal the VTCM bytes their consumers read. Dumps are in `samples/hand-tracking/.work/qnn-v75/mercury-*/consts/`.
- **Lowering.** `hexecutor/offline/qnn16_program.py` writes `PREFIX.q16p` (format: `hexecutor/native/qnn16/program.h`), `PREFIX.q16c` (constants) and `PREFIX.q16.json`. It keeps QNN's own VTCM offsets and runs the traced entries serially in simulator order; weight DMA becomes a synchronous copy at issue time. Detection is 602 ops and keypoint 538. Programs are in `samples/hand-tracking/.work/qnn16-programs/`.
- **Validation and interpreter.** `program.c` validates every range and op contract (host preflight and DSP). `exec.c` interprets ops. The 1×1 convolutions with whole K tiles run on HMX. Convolution-family ops and space-to-depth use the float path in `qnn16_f32.c`: gather to float32 scratch, prepare weights once with the scale folded in, compute and scatter, split by rows over four workers. Other ops use the FP16 handlers through logical scratch. `qnn16_half.h` holds the inline FP16 conversions, checked exhaustively against the compiler's on the host and ISS.
- **Engine.** `hexecutor/native/qnn16/engine.c`, built by `hexecutor/build-scripts/build_qnn16.sh`, runs on the unchanged tensor bootstrap (same `HxSessionConfig`), with feature `HXQ16E01` and result magic `Q16R`. Its scratch is private DDR `[56 MiB, 64 MiB − 4 KiB)` in L1-cached page 3. The host keeps constants below the scratch boundary. HVX bypasses L1, so the engine builds with `-fno-vectorize` and its `memcpy`/`memset` stay scalar in that range.
- **Host.** `Backend::qnn16_f16` and `load_qnn16_program()` in `hexecutor/host/cpp/executor.cpp`. `hexecutor_qnn16_run` runs several programs round-robin in one session and writes outputs and per-op cycles.

**Checks.**
- `test_qnn16_program.py` runs the host interpreter (float 1×1 reference), with per-op comparison against the trace.
- `test_qnn16_program_sim.c` runs the whole program on the ISS with HMX.
- On the Frame, outputs differ from QNN by at most 0.0005 for detection and 0.0012 for keypoint heatmaps (`scalar_extras` 0.0078 of 14.5). One `heatmap_xy` joint (12) can pick a different peak: QNN's top two values for it differ by 0.0002.

**Frame timing, 2026-11-09** (DSP timer, warm, 5 runs, both programs alternating in one session):

| Program | First run (scratch in uncached DDR) | Scratch in cached page 3 | Depthwise/S2D tile leaves | All tile leaves (`engine-tile2.elf`) | Tuned (`engine-tile11.elf`, compact programs) |
| --- | --- | --- | --- | --- | --- |
| Detection | 804 ms | 130 ms | 24.3 ms | 3.44 ms | **1.65 ms** |
| Keypoint | 2,378 ms | 288 ms | 123.5 ms | 4.18 ms | **2.45 ms** |

Every Frame run since `engine-tile2.elf` is bit-identical to the ISS run of the same code on all 8 outputs. Against QNN they are as before: detection within 0.0022; keypoint `heatmap_xy` 0.0021 with 20/21 argmax (joint 12 is the known near-tie), `heatmap_depth` 0.0036, `scalar_extras` 0.0099, curls 0.0029. The Frame stayed up after the session closed.

Commands are under `/home/steamos/hexecutor-qnn16/` on the Frame. The run used `qd_loader.elf`, `bootstrap4.elf` and `chunk_probe.elf` from `~/hexecutor-hand-viewer/dsp/`, which are byte-identical to the local builds. Keep single commands short: wakes over about 2.5 s risk the (2,0,8) hang. After a reboot, load the module first: `sudo insmod ~/frame-dsp/kmod-7.2/cdsp_mem.ko size_mb=128`, then check that `size_mb` is 128 and `iova` is 2684354560. The `wave1-20261101` path in `setup_frame.sh` now lives under `~/frame-dsp/quest-depth/` and holds the same file.

```sh
cd ~/hexecutor-qnn16/run && D=~/hexecutor-hand-viewer/dsp
HEXECUTOR_RUNS=5 ../build/bin/hexecutor_qnn16_run $D/qd_loader.elf engine-tile2.elf $D/bootstrap4.elf $D/chunk_probe.elf tile2/out \
  --program mercury_detection detection-in-inputImg.f32 \
  --program mercury_keypoint keypoint-in-inputImg.f32 keypoint-in-lastKeypoints.f32 keypoint-in-useLastKeypoints.f32
```

### HVX tile leaves

`hexecutor/native/qnn16/qnn16_tile.c` runs ops directly on QNN's FP16 crouton tiles in VTCM, without logical scratch or float32 staging. `exec.c` uses a tile form whenever the operand check passes. Otherwise, and on hosts, it uses the float or generic path. `Q16_EXEC_NO_TILE` forces the fallback.

| Ops | Form | Split |
| --- | --- | --- |
| DW3X3, DW3X3_S2, DW1X1, S2D | Per-row tile-pointer tables. DW3X3 uses a sliding window specialized for column parity: one load and one shuffle pair per row per step. | Rows over 4 workers |
| CONV3X3, PW1X1 | Dense. Each input word is broadcast with one `vdelta` (control `(i ^ 4w) & 0x7c`) against the `Xwe`/`Xpe` row holding that input pair for all 32 outputs. Four output pairs share each weight row. | Rows over 4 workers |
| MINMAX (ReLU6), MUL by a scalar, X2S, TRANSPOSE (H↔W and W↔C), AVGPOOL | Tile to tile | Serial |
| FORMAT (tiles ↔ direct), CAST_F32 | `vdeal`/`vshuff` and rotated masked stores. CAST_F32 is an exact integer hf→f32 widening; qf32→sf was 1–2 float ULPs off. | Serial |
| ZAP | HVX (masked) stores | Serial |

Also outside the tile file:
- COPY is split over the workers for copies of 32 KiB or more.
- IM2COL is split over the workers and writes pixel pairs as words.
- INPUT copies whole rows.
- CONCAT of compact direct tensors uses plain copies.
- The generic FP16 handlers (gather/scatter, FORMAT, casts, MINMAX, SIGMOID, MUL) do one block lookup per 32 channels instead of one per element.

Notes:
- **Arithmetic.** Products accumulate in qf32 (`vmpy hf`). The qf32 scale multiply and bias add come before one conversion to FP16, and ReLU6 is a signed-halfword `vmax`/`vmin`.
- **Epilogue and weight vectors** come straight from the tables with `vshuffe`/`vshuffo`, with an even-word mux for t=514 and a `vdelta` word copy for t=544.
- **Stores.** Only output lanes inside the logical tensor are stored.
- **Scalar VTCM.** The ISS charges heavily for scalar VTCM loads and stores, so scalar per-element code in VTCM is slow there.

**Checks.**
- **Compare mode.** `hexecutor/build-scripts/test_qnn16_program_sim.sh --compare PREFIX OUT_DIR INPUTS...` runs the whole program on the ISS. For each tile op it runs the float or generic path, then the tile path on the same VTCM, and diffs the outputs. The screen is |Δ| ≤ |ref|/256 + 2^-12. S2D, MINMAX, X2S, TRANSPOSE, FORMAT and CONCAT must be bit exact, and float32 outputs must match bitwise (±0 allowed). Result: 159 detection and 263 keypoint ops, none over the screen.
- **Timing.** Without `--compare`, the same script prints serial pcycles per op kind; the ISS harness has no workers.
- **Host.** `test_qnn16_program.py` with `--context/--known` still passes per op against the QNN trace.

**Tuning since `engine-tile2.elf`** (each step checked with `--compare` on the ISS and Frame outputs equal to the ISS):
- **IM2COL.** Exact vector form: integer RNE float→FP16, plus three rotated row vectors per output pair.
- **DW3X3_S2.** Weights are built with vector masks from the `Xwe` diagonals, and the loop slides at even offsets.
- **Serial ops.** SHUFFLE uses `vdelta` word broadcasts. RESHAPE to a compact tensor uses `vdeal` and masked stores. TRANSPOSE W↔C does a half swap, then a 16×16 transpose of 8-byte groups in four rounds of `vshuff`. MINMAX, MUL, X2S, TRANSPOSE and SHUFFLE now split their rows over the workers.
- **Worker split.** Workers take contiguous (channel group, row) ranges, so each worker builds weights and epilogue roughly once per group.
- **Setup.** `splat_base` uses five rotate/xor masks instead of a 64-step chain. Row pointer tables cost one block load per two pairs. S2D has a separate interior loop.
- **COPY runs.** Consecutive COPYs run as one parallel job (`q16_exec_copies`), placing each copy at a 128-byte boundary of a shared stream.
- **Compact stride-2 weights.** This changes the program format: `qnn16_program.py` copies only the diagonal of stride-2 depthwise weights, one 128-byte vector per (group, tap). DW3X3_S2 `p1 = 1` marks the layout, and the float path reads it too. Constants went from 4.74 to 4.00 MB (detection) and from 2.50 to 2.24 MB (keypoint). Regenerate the programs with `--tensors .../consts/tensors.bin`.

**Measured and rejected:**
- A serial threshold for small parallel ops (detection 1.97 → 2.24 ms).
- Shorter worker pauses: 8 or 32 instead of 255.
- Adaptive worker spinning before the long pause.
- A shorter join pause (no change).
- `-O3`: DW3X3 −6%, CONV3X3 +7%.
- Walking the block table directly in S2D (slower than pointer tables).
- `-falign-functions=64`.

**Measuring.** Wall time per binary moves by up to about 4% with code layout alone, for example when only keypoint-side code changes. The DSP clock also varies between sessions, so the same cycle total can give different milliseconds. Compare binaries alternately in one session, with 20 runs each, and read both milliseconds and `op_cycles` totals.

### Remaining optimizations

Frame profile with `engine-tile9.elf`:

| Program | Total | Largest kinds |
| --- | --- | --- |
| Detection | 3.25 Mcycles | COPY 22% (DDR-bound), S2D 16%, DW3X3 14%, CONV1X1 10%, DW3X3_S2 9%, PW1X1 9%, ZAP 6% |
| Keypoint | 4.70 Mcycles | CONV3X3 18%, DW3X3 14%, S2D 11%, COPY 11%, PW1X1 9%, DW1X1 8%, CONV1X1 7% |

1. **CONV3X3 on HMX** (18% of keypoint). Use QNN's slide-window `mxmem` mode (see the depth engine's NPRG2 slide code). In-place windows need the single 16 MB VTCM TLB page ([vtcm-single-page]); otherwise stage a contiguous window. Silicon can wedge on unsupported addressing that the ISS accepts, so do this with the user present (each hang means a reboot plus `insmod`).
2. **Hand-scheduled depthwise loops.** DW3X3 is about 200 ISS cycles per output vector, against about 70 for the tensor engine's generated two-packets-per-tap loop.
3. **S2D + DW3X3_S2 fusion in the lowering.** S2D staging buffers are shared and refilled across stride-2 ops, so this needs dataflow analysis over block addresses.
4. **COPY** is now DDR-bound at about 5 bytes per cycle. Only fewer bytes or overlap with compute would help.
5. **Lanes.** Run QNN's list-1 entries on worker threads ([qnn-lanes-frame]).

Earlier list, kept for history: Frame profile with `engine-tile2.elf`:

| Program | Total | Share by kind | Largest ops |
| --- | --- | --- | --- |
| Detection | 6.3 Mcycles | DW3X3_S2 23%, IM2COL 16%, COPY 16%, S2D 12%, DW3X3 9%, SHUFFLE 5%, CONV1X1 5% | Two SHUFFLEs (0.18 + 0.16 M) |
| Keypoint | 7.5 Mcycles | DW3X3_S2 14%, CONV3X3 12%, S2D 11%, IM2COL 10%, DW3X3 9%, PW1X1 9%, TRANSPOSE 8%, COPY 8% | TRANSPOSE W↔C (0.59 M) |

1. **DW3X3_S2.** Give it the same sliding-window, parity-specialized row loop as DW3X3. Fusing S2D into it would also remove the S2D cost.
2. **IM2COL.** It is still scalar `fh()` per element. Convert rows with HVX, or fuse it into the following PW1X1.
3. **COPY.** Use DMA issued at `weights_to_vtcm` and waited at `@DmaCheckpointWait`, overlapped with compute.
4. **Remaining serial ops.** SHUFFLE (tile `vdelta`), RESHAPE and TRANSPOSE W↔C (vector form).
5. **Lanes.** Run QNN's list-1 entries on worker threads while list 0 continues ([qnn-lanes-frame]).
6. **Cancellation checks** read uncached DDR every eighth op.
7. **Wake time and broker.** At 3–4 ms per network, running through `hexecutord` alongside the other graphs is now practical.

Housekeeping before committing:
- Add `test_qnn16_program.py`, `test_qnn16_program_sim.sh --compare` and `build_qnn16.sh` to the docs' check lists.
- Decide whether `qnn16_lower.py`'s JSON `lower()` is still needed.
- Run `make check` and `tools/model_artifact_policy.py`.

## Mercury C++ port

`samples/hand-tracking/mercury/` is a C++17 port of Mercury's per-frame pipeline. `mercury::Tracker` (in [tracker.hpp](../../samples/hand-tracking/mercury/include/mercury/tracker.hpp)) takes a rectified grayscale stereo pair, pinhole intrinsics for each view, and `left_in_right`. It returns two hands of 26 OpenXR-ordered joints, with poses and radii in left-camera axes (+X right, +Y up, −Z forward).

- **Networks.** The two networks sit behind `mercury::Networks`, which receives every detection or keypoint request of a frame stage as one batch, in the ONNX tensor layouts. The library has no CPU, ONNX Runtime or Python inference path. The DSP session from steps 1–3 implements this interface and converts QNN's NHWC `heatmap_xy`. Request fields after the tensors (view, hand, crop direction) are informational only.
- **Vendored code.** `third_party/monado/` holds unmodified Monado files: `kine_lm`, hand-size refinement, `u_box_iou`, and the vector and Eigen interop headers. The exception is `hg_hand_size_opt`, whose `std::span` parameters became `const std::array&` for C++17. tinyceres is vendored too. `compat/` supplies the small subset of `xrt_defines.h`, `m_api`, logging and time those files include. `src/` ports `hg_sync.cpp`, `hg_model.cpp` and `hg_image_distorter.cpp` without OpenCV. It reproduces `cv::warpAffine`'s 1/32-pixel fixed-point bilinear letterbox except for detector reductions below 0.5×, which use an antialiased triangle filter, and retains Mercury's truncating nearest-neighbor stereographic crops. Attribution is in [NOTICE](../../samples/hand-tracking/mercury/NOTICE).
- **Behavior.** Mercury's tuneable defaults, detection scheduling (alternating views, both while hand size is refined), IoU suppression, pose-predicted crops and inputs, reprojection and distance rejection, and the 10-frame display warm-up are kept. Monado's OpenCV image orientation, debug sinks, masks sink, `u_var` UI and worker pool are not ported; calls are synchronous.
- **Eigen** 3.3+ headers are required. CMake uses an `Eigen3` package or `MERCURY_EIGEN_INCLUDE`, which defaults to the ignored `samples/hand-tracking/.work/external/eigen`. Eigen 3.4.0 is from `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz`, SHA-256 `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72` (MPL-2.0). Without Eigen, the Mercury targets are skipped with a status message. The Frame has no system Eigen, so the ignored Eigen tree was copied there for the 2026-11-11 viewer build.

### Validation

`test_mercury` (`make check` when Eigen is found) checks:
- **Preprocessing:** identity warp, letterbox bars and inverse, normalization statistics, the pinhole round trip and axis conventions.
- **Decoding:** heatmap and depth decoding, right-hand mirroring, and the 0.97 hand-probability gate.
- **Stereo tracking:** a 640×480 stereo pair whose synthetic networks emit ideal outputs for a known 3D hand. With a hand produced by Mercury's own kinematic model:
  - joints reproject within 1.6 px in both views;
  - the centroid is within 3.6–7.4 mm, depending on the Eigen version;
  - optimizer reprojection error is 0.002;
  - the detector is called only twice once both hands are held;
  - the keypoint gate drops a rejected hand.

  Flipping the synthetic stereo extrinsic, or the right-hand mirroring, gives 560–850 mm errors, so the test catches those convention errors.
- **Not asserted: per-joint 3D error.** Even ideal rays from a 6.4 cm baseline leave each joint's depth along its ray weakly determined, and Mercury's articulation priors fill it in. The mean joint error is 15–24 mm even for the self-consistent hand.

`mercury_replay` runs PGM stereo pairs through the port. Its `--exchange` option starts the development-only `samples/hand-tracking/tools/mercury_onnx_exchange.py`, which answers network requests with ONNX Runtime over a pipe. On a synthetic planar pair made from `right_hands.jpg` (50 px disparity at the runtime's rectified f = 380, baseline 63.9 mm), the real networks behaved as follows:
- they detected and tracked the right hand in both views, active from frame 9;
- the palm stayed stable at 0.41 m;
- the hand size pinned at Mercury's 0.12 m maximum, because a photographed hand is not hand-sized at that depth.

Accuracy needs real captures:

```sh
build/bin/mercury_replay --focal 380 --cx 308 --cy 308 --baseline 0.0639016 \
  --exchange "$PWD/samples/hand-tracking/.work/venv/bin/python -I $PWD/samples/hand-tracking/tools/mercury_onnx_exchange.py --models $PWD/samples/hand-tracking/.work/external/hand-tracking-models" \
  left0.pgm right0.pgm left1.pgm right1.pgm
```

## Reproduce

```sh
# Compile each net with the configured QAIRT toolchain.
docker run --rm --platform linux/amd64 -v /absolute/qairt/2.50.0.260828:/sdk:ro -v "$PWD":/workspace -w /workspace \
  hexecutor-qairt-tools python hexecutor/offline/compile_qnn.py --sdk /sdk \
  --model samples/hand-tracking/.work/external/hand-tracking-models/grayscale_keypoint_jan18.onnx \
  --output samples/hand-tracking/.work/qnn-v75/mercury-keypoint --name mercury_keypoint

samples/hand-tracking/.work/venv/bin/python -I samples/hand-tracking/tools/mercury_qnn_oracle.py --sdk /absolute/qairt/2.50.0.260828 \
  --models samples/hand-tracking/.work/external/hand-tracking-models \
  --detection-context samples/hand-tracking/.work/qnn-v75/mercury-detection/mercury_detection-v75.bin \
  --keypoint-context samples/hand-tracking/.work/qnn-v75/mercury-keypoint/mercury_keypoint-v75.bin \
  --image samples/hand-tracking/.work/right_hands.jpg --work samples/hand-tracking/.work/mercury-oracle

python3.14 samples/depth-model/tools/htp_graph.py samples/hand-tracking/.work/qnn-v75/mercury-keypoint/mercury_keypoint-v75.bin \
  samples/hand-tracking/.work/qnn-v75/mercury-keypoint/context --no-source-names
```

`--graph-name` does not yet find 2.50's debug-name table, so use `--no-source-names`.

## HXFG / ExecuTorch simulator evidence

The 2026-11-20 comparison used two Mercury HXFG exports in `hexecutor/.work/mercury-et/`. The keypoint graph
multiplies a `16x16x32` map by the scalar runtime input `useLastKeypoints`. The FP16
binary vector path now broadcasts that scalar across every 32-channel block. `HX_ZEROINSERT`
also copies blocked channel vectors onto the sparse output grid, preserving its zeros and odd-width
pad columns. A second blocked gather packs 3x3 stride-one convolution windows with whole
32-channel input blocks, including the pad-two border expansion at keypoint op 68. Unsupported
shapes retain the generic im2col gather. Unsupported binary broadcasts and zero-insert layouts
retain the scalar path. On
the single-thread Hexagon ISS, the Mercury scalar multiply fell from 2,298,294 to 29,556
pcycles, and its `10x10x63` to `21x21x63` zero insert from 3,575,466 to 57,615 pcycles.
A pad-two `8x8x32` to `10x10x32` convolution fixture fell from 383,451 to 97,491
pcycles with the same FP16 result.
These pcycles count instructions rather than predicting Frame latency.

Ordinary FP16 external input packing now splits rows across the four DSP workers. Mercury's
one-channel image is represented with a 32-channel stride, so import has a large padded footprint.
The Frame profile placed 6.58 million detection cycles outside operator timing, which includes
import and export; the precise share due to import has not been measured on device. Parallel
packing is a candidate for reducing that interval, pending a Frame measurement. The HXFG and
`.pte` files do not need re-export for these engine changes.

The serial whole-graph ISS harness measured 23,700,072 to 16,853,346 pcycles for keypoint
and 7,522,407 to 7,475,226 for detection, including import but excluding constant staging
and output export. The changed keypoint graph produced bit-identical outputs to the prior
simulator engine on all four outputs. The simulator's absolute Mercury outputs differ from the
saved Frame outputs even on the prior engine, so this is a relative regression check, not a
Frame accuracy result. The saved prior Frame keypoint heatmap matched the ONNX Runtime argmax
for 20 of 21 joints; the remaining joint needs device validation with this engine.
