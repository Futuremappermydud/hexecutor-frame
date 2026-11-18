# OpenXR hand joint viewer

## Stereo NIR viewer (development build)

`frame_nir_hand_viewer` consumes the local `FRNIR001` raw tracking-camera stream and draws the native NIR pipeline's 21 joints per hand in the same transparent OpenXR layer. A reader keeps the newest complete pair, inference runs on a worker, and results expire 500 ms after capture, including inference time. The viewer exits when the stream ends. Capture timestamps are converted from `CLOCK_MONOTONIC_RAW` for capture-time world locking.

Build on Frame from `runtime/` with `make nir-hand-viewer NIR_ORT=/home/steamos/nirhands-viewer/ort OPENXR_INCLUDE=/path/to/openxr-sdk/include`. Then run `./scripts/run_nir_hand_viewer.sh --models /home/steamos/nirhands-viewer/models` in an SSH terminal with a TTY; the launcher asks for sudo credentials to run the read-only capture helper. `FRAME_NIR_CAPTURE` overrides its default path `/tmp/nirhands-live/stream_nir_pairs`. `--threads N` and `--seconds N` are available for bounded checks. Inference threads use available CPUs 2–7 when affinity can be set. Ctrl-C stops the viewer and closes the capture pipe.

This build still executes the NIR ONNX graphs on the CPU by default. The Frame's installed ONNX Runtime has no QNN execution provider, and the native QNN16 ONNX lowerer cannot yet handle the keypoint graph's `InstanceNormalization` and dynamic shape operators. NPU deployment requires a prepared graph plus a verified native runner; do not label this build as NPU inference.

An explicit `--qnn16-dir DIR` mode loads `nirhands_detector.{q16p,q16c}` and `nirhands_keypoints.{q16p,q16c}` through Hexecutor, with required `--loader`, `--engine`, `--bootstrap`, and `--chunk-probe` paths. It has no CPU fallback. The ONNX-to-QNN NHWC input/output conversion has host layout verification. With the specialized QNN16 tile kernels (see [NIR hand tracking NPU handoff](../model-development/nirhands-npu-handoff-2026-11-13.md)), device latency on the Frame CDSP is 8.8 ms (8.81 ms, 15.8M cycles) for the detector and 14.3 ms (14.28 ms, 25.7M cycles per hand) for keypoints, bit-identical to the ISS. Full end-to-end device tracking validation requires deploying both validated Q16P programs with their matching runtime artifacts.

`frame_hand_viewer` is a small standalone OpenXR app on Frame. It draws up to two tracked hands as 3D geometry over alpha-blended passthrough. The default MediaPipe path draws 21 joints per hand; the Mercury mode draws 26. Right hands are blue, left hands orange. It needs no browser or OpenVR overlay.

`--mercury-dir DIR` selects stereo Mercury tracking on the CDSP. The directory contains `mercury_detection.{q16p,q16c}` and `mercury_keypoint.{q16p,q16c}`; use the matching QNN16 engine with explicit `--engine`, `--bootstrap`, `--loader`, and `--chunk-probe` paths. Mercury uses both rectified luma images and the calibrated stereo baseline. Its 616→160 detector input uses an antialiased triangle reduction: point-sampled downscaling suppressed hand confidence on real Frame captures. The MediaPipe path remains the default.

The app uses the existing compositor DMA-BUF discovery and Vulkan NV12 detiling. MediaPipe uses rectified left-camera RGB; Mercury uses rectified luma from both cameras. The native C++17 tracker runs on a separate worker; OpenXR continues its wait/begin/end frame loop while inference runs. Results older than 500 ms are cleared rather than displayed as current.

**Skeleton models.** A model directory whose `manifest.json` has `"arch": "skeleton"` under `keypoints` is a `SkeletonHandNet` export: the network returns six outputs and the host runs forward kinematics and the palm fit (`nirhands/src/skeleton.cpp`). When `"tight_crop": true` is set (as in the latest 96 px `FutureMapper/skeleton_tight96_full_v1` model), the native tracker fits tight crops to previous hand joints. Both 128 px and 96 px inputs are supported. Tracking and display are unchanged. This is verified on the CPU (ONNX Runtime) path only; the skeleton program does not yet fit the DSP engine's scratch region, see [the integration handoff](../model-development/nirhands-skeleton-integration-handoff-2026-11-13.md).

## Joint geometry

The viewer supports two skeletal topologies:
- **MediaPipe (21 joints):** Wrist at joint 0, radiating 4 joints per finger (metacarpophalangeal to tip) and palm outline connections (21 bones total). Hand depth is estimated monocularly from hand size in the pinhole camera model (clamped to 0.12–1.5 m).
- **OpenXR standard / Mercury (26 joints):** Conforms to `XR_EXT_hand_tracking` (`0` = Palm, `1` = Wrist, `2..5` = Thumb metacarpal to tip, `6..10` = Index, `11..15` = Middle, `16..20` = Ring, `21..25` = Little). In standard OpenXR skeletal anatomy, metacarpal bones (`{1,2}`, `{1,6}`, `{1,11}`, `{1,16}`, `{1,21}`) and the palm (`{1,0}`) articulate from Wrist joint 1 rather than radiating from the palm center, ensuring correct finger articulation.

Each hand's model-relative metric landmarks are translated so they project onto its image landmarks (least squares in the rectified pinhole camera). Joints then go through the calibrated rectified-camera-to-head transform. In Mercury, joint positions are provided in left rectified camera coordinates and calibrated stereo baseline.

The worker locates OpenXR VIEW space in LOCAL at the capture timestamp via `XR_KHR_convert_timespec_time` and transforms joints into world space using quaternion rotations (`rotate(q, v)`). Head motion during capture, inference, and display therefore does not drag or swim the hands. If the extension or pose is unavailable, joints fall back to head space and follow the display pose. The `[Hands]` log reports `world_locked` and `capture_to_publish_ms`.

The renderer draws into one transparent per-eye projection layer at the runtime's recommended size, with depth testing. It keeps two frames in flight and does not wait on the CPU for the frame it just submitted. Precompiled SPIR-V shaders are loaded from `runtime/shaders/hand_joints.{vert,frag}.spv`.

The tracker drops a hand when its landmark bounds overlap a higher-presence hand by more than `Options::duplicate_iou` (0.5). This applies to existing tracks and to new detections. Without it, two tracks could converge on one hand and fill both slots, so the second hand was never detected again.

## Build

### Using Make (on Frame)

From `runtime/`:

```sh
# MediaPipe / Mercury hand viewer:
make hand-viewer OPENXR_INCLUDE=/path/to/openxr-sdk/include -j4

# Stereo NIR hand viewer:
make nir-hand-viewer NIR_ORT=/home/steamos/nirhands-viewer/ort OPENXR_INCLUDE=/path/to/openxr-sdk/include -j4

# Or build only the host dependency for NIR viewer:
make nir-hand-viewer-host
```

`make hand-viewer-host` builds `libhand_tracking_host.a`, `libmercury_executor_networks.a`, `libmercury_hand_tracking.a`, and `libhexecutor_host.a`. `make nir-hand-viewer-host` builds `libhexecutor_host.a`. Hand shaders are compiled with `glslc` if available, or fall back to precompiled `.spv` shaders if `glslc` is absent.

### Using CMake

From repository root:

```sh
cmake -B build -DFRAME_BUILD_TESTS=ON
cmake --build build --target frame_hand_viewer --parallel
```

When `NIR_ORT` is set (or detected at `hand-tracking/.work/ort/...`), `frame_nir_hand_viewer` is also enabled:

```sh
cmake -B build -DFRAME_BUILD_TESTS=ON -DNIR_ORT=/home/steamos/nirhands-viewer/ort
cmake --build build --target frame_nir_hand_viewer --parallel
```

OpenXR headers are auto-detected from `hand-tracking/.work/openxr-sdk/include` or `$OPENXR_INCLUDE`. The OpenXR loader is supplied by SteamVR at `/opt/steamvr/bin/linuxarm64`.

## Run

SteamVR must be awake and publishing compositor color buffers. Follow [camera prerequisites](cameras.md) and the [hand model/session prerequisites](../model-development/hand-tracking.md). Do not open claimed physical sensor devices.

### Launching locally on Frame

For MediaPipe / color camera viewer, run from `runtime/`:

```sh
./scripts/run_hand_viewer.sh \
  --models /absolute/path/to/deployment \
  --loader /absolute/path/to/qd_loader.elf \
  --engine /absolute/path/to/tensor/engine.elf \
  --bootstrap /absolute/path/to/tensor/bootstrap.elf \
  --chunk-probe /absolute/path/to/tensor/chunk_probe.elf
```

The launcher temporarily disables compositor camera downscaling and restores the previous setting on exit. This keeps the existing 2464 × 2464 DMA camera buffers producing frames when the OpenXR app takes focus. The app itself links OpenXR and Vulkan, not OpenVR.

The app starts an OpenXR session named **Frame Hand Joints** and appears in the headset. Put your hands in front of the headset to see the joints and bones on them. MediaPipe alignment depends on its monocular depth estimate. Tracking covers only the rectified left-camera field of view.

For Mercury stereo tracking on the CDSP, run from `runtime/` with the QNN16 deployment paths:

```sh
./scripts/run_hand_viewer.sh --mercury-dir /absolute/path/to/qnn16-programs \
  --loader /absolute/path/to/qd_loader.elf \
  --engine /absolute/path/to/qnn16/engine.elf \
  --bootstrap /absolute/path/to/bootstrap4.elf \
  --chunk-probe /absolute/path/to/chunk_probe.elf
```

Mercury's joint positions are in left rectified camera axes (+Y up, −Z forward). The viewer converts them to the rectifier's camera axes, then uses the existing camera-to-head and capture-time world transforms. Its depth comes from Mercury's stereo optimization, not the MediaPipe size fit. See [Mercury hand tracking](../model-development/mercury-hand-tracking.md) for model provenance and validation limits.

For the stereo NIR tracking camera viewer, run from `runtime/` in an SSH terminal with a TTY:

```sh
./scripts/run_nir_hand_viewer.sh
```

`run_nir_hand_viewer.sh` automatically discovers models at `FRAME_NIR_MODELS` or `/home/steamos/nirhands-viewer/models` if `--models` is omitted. The launcher prompts for sudo credentials to read the `FRNIR001` stream from `FRAME_NIR_CAPTURE` (default `/tmp/nirhands-live/stream_nir_pairs`).

### Launching remotely over SSH

[view_hands.sh](../../runtime/scripts/view_hands.sh) launches the installed viewer over SSH:

```sh
# Color / MediaPipe / Mercury viewer:
./scripts/view_hands.sh

# Stereo NIR tracking-camera viewer:
./scripts/view_hands.sh --nir
```

The `--nir` flag switches the remote target to `run_nir_hand_viewer.sh` and automatically directs `FRAME_REMOTE_DIR` to `/home/steamos/nirhands-npu` (or `/home/steamos/frame-spatial-runtime`). Set `FRAME_HOST` (default `frame`) as needed. Ctrl-C requests an orderly stop through the SSH terminal. SIGTERM also drains current inference and explicitly closes the DSP session.

`--fps N` caps capture/tracking at 1–60 Hz (default 60), independently of OpenXR cadence; unchanged camera frames are skipped. `--precise` selects DSP float32 comparison arithmetic. `--seconds N` bounds process lifetime and `--max-frames N` bounds processed camera frames for validation.

When the OpenXR session is no longer visible, the worker releases its DSP session and camera references. New visibility and fresh camera pixels reacquire tracking. Camera stalls close the session after 500 ms without a changed left-image hash. Compositor restart triggers descriptor rediscovery and tracker reset. A fatal inference or cleanup error exits instead of retrying automatically; use [executor recovery guidance](../hexecutor/operations.md) before restarting after failed restoration.

Capture timestamps are the midpoint of the monotonic DMA-read interval, not measured exposure time. The capture-time head pose therefore lags exposure by the read delay plus up to one 5 ms poll. A hash change is the available freshness heuristic; it does not prove sensor exposure timing.

## Checks

`make check` includes portable tests for the camera-space joint fit, bone topology (both 21-joint MediaPipe and 26-joint Mercury/OpenXR hierarchies), degenerate and nonfinite inputs, quaternion rotations, world-locking transforms, two-hand tracking, and duplicate-track suppression. Device validation additionally needs real 3D joint alignment on both hands, world locking under head motion, ongoing OpenXR submissions, visible/hidden session transitions, close/reopen, signal exit and eye-client restoration when an eye client is present. Host tests alone do not prove device inference or restoration.
