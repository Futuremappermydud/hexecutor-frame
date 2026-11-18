# Handoff, 2026-11-13: NIR hand pipeline on the headset

## Goal

The user wants the stereo NIR hand pipeline running on the Steam Frame, with the result visible in the headset. The current Mac viewer streams the slam pair over SSH, which is too unstable to use: about 5 pairs/s, with stalls of up to 12 s. Background: [nirhands-live-seeding.md](nirhands-live-seeding.md), which covers the Python pipeline, its tracker rules and the first live session.

## Plan

Everything runs on the Frame, with no network in the loop:

```
sudo stream_nir_pairs --raw  |  frame_nir_hand_viewer (steamos user)
   root capture helper          native C++ pipeline (ONNX Runtime, CPU) -> HandViewerXr (OpenXR over passthrough)
```

- **Capture:** `hand-tracking/tools/stream_nir_pairs.cpp` with `--raw` (FRNIR001, about 2.1 MB/pair) through a local pipe. It needs root to duplicate XRService's descriptors. The OpenXR app runs as the normal user.
- **Inference:** the native port in `hand-tracking/nirhands/` runs both ONNX models on the CPU.
- **Display:** reuse `runtime/src/hand_viewer_xr.cpp` (`HandViewerXr`, `HandScene`), following `runtime/tools/frame_hand_viewer.cpp`. Run inference on a worker thread and render on the main loop. Clear results older than 500 ms. World-lock with `viewer.headPoseAt(capture_ns)`.
- **Clock:** stream timestamps are `CLOCK_MONOTONIC_RAW`. Convert them to `CLOCK_MONOTONIC` for `headPoseAt`, for example with `V4l2CaptureClock::monotonicMinusRawNs()` from `hand-tracking/tools/v4l2_capture_clock.cpp`.
- **Frame:** the native pipeline's world frame is already the OpenVR head frame, so joints are head-space points. Set `HandScene.local_space` as `frame_hand_viewer` does.

## Done this session

| Piece | Where | State |
| --- | --- | --- |
| ONNX export and check | `hand-tracking/tools/export_nirhands_onnx.py` → `hand-tracking/.work/nirhands-models/` (detector.onnx, keypoints.onnx, manifest.json) | ORT vs PyTorch max abs diff 4e-6 (both models) |
| Native geometry and rig | `hand-tracking/nirhands/include/nirhands/geometry.hpp`, `src/geometry.cpp` | Loads `/persist/xrservice.json` + `/persist/device_config.json` (or a Frametop session's calibration.json/device.json) into the head frame. Has a small JSON parser |
| Native detector, seeding, crops, decode | `src/geometry_ops.cpp` | Seed centres match Python to 0.03 mm median (0.2 mm p99). For identical centres, network inputs match except 0.014% of values (one-grey-level ties), and decoded joints match to 0.006 mm median, 0.14 mm max |
| Native ORT models, tracker, One Euro | `src/pipeline.cpp`, `include/nirhands/pipeline.hpp` | A port of the final `nirhands_pipeline.py` tracker. Two keypoint sessions run the two hands concurrently |
| Replay CLI | `hand-tracking/nirhands/tools/nirhands_replay.cpp` | Raw stream in, JSON lines out. 15.5 ms/pair on the Mac CPU |
| Native vs Python comparison | `hand-tracking/tools/compare_native_nirhands.py` (`--to-raw` converts JPEG recordings) | See the open issue below |
| Device calibration copy | `hand-tracking/.work/device-calibration/` (ignored) | The repo's "generic" rig file `steam_frame_tracking.yaml` is this headset's own calibration (rotation difference 0.04°) |

Mac build (from `hand-tracking/`):

```sh
ORT=.work/ort/onnxruntime-osx-arm64-1.24.1
c++ -O2 -std=c++17 -Wall -Wextra -Inirhands/include -I$ORT/include nirhands/src/*.cpp nirhands/tools/nirhands_replay.cpp \
  -L$ORT/lib -lonnxruntime -Wl,-rpath,$PWD/$ORT/lib -o .work/nirhands-build/nirhands_replay
```

`resample` reproduces OpenCV 5.0's `cv2.remap`, the version in the synthetic project's venv. That remap uses float32 map coordinates and a float32 bilinear blend, with no 1/32-pixel fixed point. The fixed-point model gave about 6% ±1 differences.

## Measured on the Frame

ONNX Runtime 1.24.1 CPU benchmarks (`/home/steamos/nirhands-viewer/bench/`). The CPU is 1 Cortex-X4, 5 A720 and 2 A520 (cores 0–1 are the A520s).

| What | Time |
| --- | --- |
| Detector, both views (batch 2), 2 threads | 8.2 ms |
| Keypoints, one hand, 2 threads, pinned to cores 2–7 | 40.6 ms (3 threads: 34.1 ms) |
| Keypoints, two hands concurrently, 2 threads each, cores 2–7 | about 46 ms |

Expect about 15–20 hand updates/s. The keypoint network is the cost; it is GroupNorm+SiLU at width 32. Options for later:
- the `norm=batch, act=relu` variant (foldable, NPU-friendly), or a smaller width;
- the CDSP through Hexecutor's ONNX→QNN16 path, as Mercury uses.

## Open issue: native vs Python tracker divergence

On the 14:14 live recording (`hand-tracking/.work/live-1414.raw`), the full native and Python pipelines disagree in about 71 of 334 frames on track IDs. Matched hands differ by 2.5 mm median (max joint error). The first difference appears at frame 37, about 1.2 mm, three frames after track 3 starts.

Per-component checks agree to hundredths of a millimetre, so the cause is in tracker state, not crops or models. The working hypothesis is that each frame's crop centre is the previous output, which amplifies tiny seed differences. It is **not confirmed**. The confirming experiment, Python against Python with 0.03 mm seed noise, gave about 500 mm differences and different IDs in 278 of 334 frames. That result is implausible and probably a bug in the test script, not evidence (`scratchpad`-only code; rewrite it).

Next steps:
1. Make the noise test credible. For example, log per-frame track centres from both runs and check the divergence starts at the perturbation.
2. If the amplification is real, accept native ≈ Python at the tracking-behaviour level and compare summary statistics instead of frame-by-frame.
3. Otherwise, diff the two trackers step by step from frame 34 by dumping centre, velocity, hits and spans per track per frame from both sides.

Behaviour so far looks equivalent: both find hands in 267/334 frames.

## Remaining work, in order

1. Resolve or bound the divergence above.
2. Write `runtime/tools/frame_nir_hand_viewer.cpp`:
   - **Input:** FRNIR001 from stdin; a reader thread keeps the newest pair.
   - **Worker:** runs `nirhands::Pipeline` and publishes a `HandScene` with `joint_count = 21` (MediaPipe bone order).
   - **Main loop:** `HandViewerXr` poll/render.
   - **Options:** `--models DIR`, `--threads`, optional `--seconds`.
   - **CPU:** pin the inference threads to cores 2–7.
3. Add a `nir-hand-viewer` target to `runtime/Makefile` linking the `hand-tracking/nirhands` sources, `src/hand_viewer_xr.o`, OpenXR, Vulkan and ORT (`-I$ORT/include -L$ORT/lib -lonnxruntime -Wl,-rpath,...`). Build on the Frame from `runtime/`. Shaders come from `make hand-viewer`'s `HAND_SPV`. `OPENXR_INCLUDE` may be needed; see [hand-tracking-viewer.md](../runtime/hand-tracking-viewer.md).
4. Write a launcher script that runs `sudo /tmp/nirhands-live/stream_nir_pairs --raw | bin/frame_nir_hand_viewer --models ...` from an `ssh -t frame` terminal; the user types the sudo password. Consider disabling compositor camera downscaling as `run_hand_viewer.sh` does, though the slam cameras may not need it.
5. Validate on the headset: joints overlay the real hands, world-locked under head motion, update rate, clean exit (Ctrl-C closes the pipe and the helper exits). Run `nirhands_replay` on the Frame against `live-1414.raw` (copy it over) for the device timing and output check.
6. Update `nirhands-live-seeding.md` and `docs/runtime/hand-tracking-viewer.md` with the new app.

## Device state (Frame, `ssh frame`)

- **`/home/steamos/nirhands-viewer/`:** `ort/` (ORT 1.24.1 linux-aarch64 include and lib), `models/` (the exported ONNX files and manifest), `bench/` (the two benchmark programs). `/home` has only about 6 GB free.
- **`/tmp/nirhands-live/`:** `stream_nir_pairs`, rebuilt today. It is lost on reboot; rebuild per nirhands-live-seeding.md.
- **Other:** no services or settings were changed.

## Local artifacts (ignored, `hand-tracking/.work/`)

| Path | Contents |
| --- | --- |
| `ort/` | ORT releases |
| `nirhands-models/` | Exported ONNX models |
| `nirhands-build/` | Mac replay binary |
| `device-calibration/` | Contains unit serials; keep local |
| `live-nir-20261113-1414.frnir`, `live-1414.raw` | The user's live session, JPEG and raw |
| `native-1414.jsonl` | Latest native replay output |
| `pipeline/` | Videos and JSON from Frametop and live replays |

Frametop images are CC BY-NC with privacy terms; keep rendered videos local.

## Unchanged constraints

- Keep the synthetic project (`cowfish/stereo_nir_hands`) read-only from here.
- Keep the C++ port in step with `nirhands_pipeline.py`. Rerun `compare_native_nirhands.py` after changing either.
- The repo guidance prefers native C++ for live paths; Python is for offline development. Don't stage or commit unrelated working-tree changes; nothing from this session is committed.

## NPU conversion checkpoint, 2026-11-13

The exported ONNX models compile with QAIRT 2.50 for V75/SM8650 (O3, 8 MB VTCM, four HVX threads). The generated contexts and logs are ignored local artifacts at `hand-tracking/.work/qnn-v75/nirhands-detector/` and `hand-tracking/.work/qnn-v75/nirhands-keypoints/`. The existing `hexecutor-qairt-tools` Docker image was used with `hexecutor/offline/compile_qnn.py`; the keypoint conversion needs `--input-dim crops=1,2,128,128 --input-dim second_valid=1`. Detector has 1,623 lowered nodes; keypoints has 3,096. Compilation is not device execution or an accuracy check.

A direct QAIRT runtime probe on Frame used the SDK's aarch64 Ubuntu `qnn-net-run` and OE Linux HTP backend and V75 stub, copied to `/home/steamos/nirhands-qnn-test/`. The runner starts and loads the backend and stub, but reports `Device Creation failure` before reading a context. `LD_DEBUG=libs` shows unsuccessful searches for `libcdsprpc.so.1` and `libcdsprpc.so`, neither of which is installed in Frame's native library paths. A Quest Android `libcdsprpc.so` on the device is an Android/bionic library, not a validated native glibc implementation. A compatible native FastRPC transport and device integration would need to be established before this runtime path can be tested; simply supplying the context is insufficient. No inference outputs were obtained, and the test did not modify services or system libraries.

The alternative is Hexecutor's native QNN16 path. The detector context decodes with `depth-model/tools/htp_graph.py`; the keypoint context currently fails its graph I/O preamble decoder. The keypoint schematic includes InstanceNorm, Welford reductions and Swish operations beyond the existing QNN16 handlers, so decoding, implementing the missing operations, producing run lists, and numerical and device validation remain necessary. Neither model has an NPU backend integrated into the NIR pipeline yet.
