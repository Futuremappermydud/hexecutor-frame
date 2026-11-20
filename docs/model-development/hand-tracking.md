# MediaPipe hand detection and tracking

MediaPipe runs color-camera palm detection and landmark tracking through the HXFG tensor backend. The OpenXR viewer and native replay share the tracker. For stereo alternatives see [Mercury](mercury-hand-tracking.md) and [NIR hands](nir-hand-tracking.md).

The native C++ hand pipeline runs Google's Hand Landmarker float16/1 bundle through Hexecutor on the Frame CDSP. Palm detection and landmark inference use HMX/HVX; RGB cropping, decoding, weighted suppression, projection and ROI tracking run in C++. The two graphs share one device session. This is an RGB API and replay executable; the standalone [live viewer](../runtime/hand-tracking-viewer.md) also connects the compositor camera to this pipeline. The depth/render application does not consume hand results yet.

## Model and coordinate contract

| Graph | Input | Ordered outputs |
| --- | --- | --- |
| Palm | Float RGB `[1,192,192,3]`, range 0–1, square letterbox with zero border | 2,016 × 18 box/keypoint regressors; 2,016 score logits |
| Landmarks | Float RGB `[1,224,224,3]`, range 0–1, rotated square ROI with replicated border | 21 × 3 image landmarks; presence; right-hand probability; 21 × 3 model-relative world landmarks |

The pipeline uses MediaPipe's 24×24×2 and 12×12×6 anchors, score threshold 0.5, weighted IoU suppression at 0.3, wrist-to-middle-MCP palm rotation, 2.6× initial crops, and partial-hand landmark ROIs with 2× expansion. Successful ROIs skip the detector; loss triggers reacquisition in the same frame. The detector also searches for additional hands every ten frames while below the configured hand count. IDs persist while an ROI is tracked. Timestamps must increase; call `reset()` for a new stream.

Image landmarks use MediaPipe's 21-joint order: x/image-width, y/image-height, and wrist-relative z/image-width with the model's 0.4 z normalization. World landmarks retain model-relative meters and rotate into image axes. They are not calibrated camera/world positions. Handedness follows MediaPipe's mirrored/selfie convention; callers supplying unmirrored cameras must interpret labels accordingly. Camera calibration, capture timing, stereo association and rendering transforms remain application responsibilities.

## Provision and build

The download recipe pins the versioned bundle to SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`. Weights, converted images, schematics and validation tensors remain ignored local artifacts. Attribution is in [NOTICE](../../samples/hand-tracking/NOTICE) and [the MediaPipe license](../../samples/hand-tracking/LICENSE.mediapipe).

From the repository root:

```sh
python3 -m venv samples/hand-tracking/.work/venv
samples/hand-tracking/.work/venv/bin/pip install -r samples/hand-tracking/tools/requirements.txt
samples/hand-tracking/.work/venv/bin/python samples/hand-tracking/tools/prepare_mediapipe.py \
  --assets samples/hand-tracking/models/mediapipe \
  --output samples/hand-tracking/.work/deployment
make build-host
```

The converter emits `palm.hxfg`, `palm.bin`, `landmark.hxfg`, `landmark.bin`, and provenance JSON. It folds constant fp16 dequantization, pretransposes/pads weights, generates HMX tiles, and reuses dead activation storage. Unsupported operations/options reject conversion. It does not substitute the unrelated legacy hand containers.

Build the model-independent DSP engine with an explicitly provisioned SDK and installed Docker toolchain image:

```sh
HEXECUTOR_SDK_ROOT=/absolute/path/to/hexagon-sdk/6.6.0.0 \
HEXECUTOR_TOOLCHAIN_IMAGE=your-installed-toolchain-image \
  sh hexecutor/build-scripts/build_tensor.sh
```

Outputs are under `hexecutor/native/build/tensor/`. Reuse the verified existing `qd_loader.elf`; this recipe deliberately does not replace the loader. Copy the native host executable, four deployment files, and matching engine/bootstrap/chunk probe to the Frame. See [executor operations](../hexecutor/operations.md) for its firmware, memory-module and exclusive-session prerequisites.

## Run and embed

The package can be checked without device access:

```sh
build/bin/frame_hand_tracker --models /absolute/path/to/deployment --check-package
```

On the Frame, binary P6 RGB PPM replay emits one JSON result per frame:

```sh
build/bin/frame_hand_tracker \
  --models /absolute/path/to/deployment \
  --loader /absolute/path/to/qd_loader.elf \
  --engine /absolute/path/to/tensor/engine.elf \
  --bootstrap /absolute/path/to/tensor/bootstrap.elf \
  --chunk-probe /absolute/path/to/tensor/chunk_probe.elf \
  --repeat 12 frame.ppm
```

`--precise` selects the DSP float32 comparison path. `--dump-inputs /absolute/path/to/ignored/work` records exact model inputs for offline comparisons. HMX is the default; there is no automatic CPU/Python fallback.

For embedding, link `hand_tracking_host`, call `hand_tracking::load_models(directory)`, construct one `hexecutor::Session` with those programs, and pass it to `hand_tracking::Tracker`. `process(Image, timestamp_ns)` accepts RGB bytes with an explicit row stride. Explicitly call `Session::close()` to receive restoration errors. The API lives in [tracker.hpp](../../samples/hand-tracking/include/hand_tracking/tracker.hpp).

## Validation and performance

`make check` covers generic tensor contracts, portable operators, alias/range rejection, crop/tracking behavior, worker compatibility, safe QNN schematic parsing and artifact policy. The SDK-dependent simulator regression checks K tiles across 1,024 channels, binary16 rounding boundaries, and RGB 3×3/5×5 packing with padding and channel tails:

```sh
HEXECUTOR_SDK_ROOT=/absolute/path/to/hexagon-sdk/6.6.0.0 \
HEXECUTOR_TOOLCHAIN_IMAGE=your-installed-toolchain-image \
  sh hexecutor/build-scripts/test_tensor_sim.sh
```

[validate_mediapipe.py](../../samples/hand-tracking/tools/validate_mediapipe.py) compares the native reference or device runner with LiteRT. It accepts explicit SDK artifacts and an optional SSH host/remote work directory, records their hashes and timings, and keeps tensors/reports in its `--work` directory. `--landmark-input` adds a real crop saved by the replay executable. The float32 comparison budget is 0.002 absolute. HMX budgets are 1% of network input extent for regressors/landmarks, 0.02 for probabilities, and 0.002 m for model-relative world coordinates. Palm logits are compared after sigmoid. These are arithmetic comparison budgets, not task-accuracy guarantees.

The 2026-11-06 development run validated zero, random, gradient, image and real-hand-crop inputs, then replayed Google's `right_hands.jpg` for twelve frames. Both hands stayed above 0.99 presence with stable IDs and one detector invocation. The image-space overlay aligned with both hands. Repeated runs and explicit closes passed the existing guard, VTCM fingerprint, mapping, clock-marker and callback cleanup checks. A SIGTERM during replay completed the current frame, closed the session, and allowed a fresh process to reopen and infer successfully; SIGINT uses the same orderly-exit path.

The initial float DSP palm path took about 700 ms. HMX, compact RGB, VTCM activations/constants, vector packing and specialized depthwise loops reduced the first fully validated warm host runs to roughly 19 ms for palms and 13.5 ms for landmarks (DSP architectural-timer measurements: 18.57 ms and 12.98 ms). The twelve-frame two-hand replay measured about 37 ms per tracked frame before the final host guard-scan optimization; a short close/reopen replay subsequently measured 31–35 ms on its two warm frames. These are dated prototype observations, not sub-millisecond performance or a live camera benchmark. Keep cold setup, model execution, host overhead and full tracking latency separate.

### Compacted FP16 measurements

The 2026-11-06 compacted planar run on Frame measured 0.945–1.004 ms DSP for landmark and 2.21–2.26 ms for palm (host warm calls roughly 1.0–1.3 ms and 2.5–3.0 ms). Engine identity `d6b96ed223a6…`, four-worker bootstrap `b8ddcf62d8cb…`. Nine LiteRT cases passed. All 28 outputs matched the preceding silicon build bit for bit. Cold staging took about 23 ms. Use the same artifact hashes before comparing a new build.

- Exact dead-channel compaction removed 4,976 landmark channels and reduced the weights from 5.45 to 3.88 MB. Ordering remaining weights by magnitude improved gradient-case handedness error to about 0.0191 against a 0.02 budget. Keep the rounding and accumulation order in accuracy comparisons.
- Depthwise loops and row/block partitioning were the major warm-path gains. FP32 residual state cost 9.41 ms versus 6.13 ms in the paired trial with little numerical benefit. QF16 depthwise accumulation degraded palm scores and handedness. Use the existing scheduled HVX implementation as the baseline. [Kernel constraints](../hexecutor/kernels.md)
- Before claiming live performance, repeat two-hand tracking, idle close/reopen, signal exit and eye-client recovery with the chosen engine and bootstrap. Compare deadline misses and wake latency as well as warm arithmetic. [Broker measurements](../hexecutor/broker.md), [Validation](../hexecutor/validation.md)

## QNN v75 compiler comparison

The generic [offline compiler wrapper](../../hexecutor/offline/compile_qnn.py) runs the supplied QAIRT converter/library generator/HTP preparer with explicit architecture, SoC, VTCM, optimization level and thread settings. It requires QAIRT's own supported Linux Python environment, separate from the development venv above. [inspect_qnn.py](../../hexecutor/offline/inspect_qnn.py) reads the resulting schematic without executing its Python code.

QAIRT 2.50, v75/SM8650, O3, 8 MiB VTCM and four HVX threads produced the following for the current Google bundle:

| Graph | Lowered graph records | FP16 stride-1 / stride-2 convolution tiles |
| --- | --- | --- |
| Palm | 2,662 | 623 / 12 |
| Landmark | 1,442 | 292 / 16 |

Records include constants, shapes and aliased/in-place placement nodes; they are not logical layer counts or a latency measure. The compiler uses `F16Crouton_TCM`, packed FP16 weights, DMA/checkpoint scheduling, in-place slices/concats, and `conv_layer_fp16_s1_v73` / `s2_v73` kernels even when targeting v75. Standalone depthwise/add/PReLU/ReLU operators disappear into the convolution lowering. The disassembled FP16 leaf performs matrix bias/conversion before storing FP16 tiles, avoiding our repeated FP32 materialization and repacking.

Qualcomm's published 0.533 ms detector / 0.707 ms landmark TFLite figures on SM8650 use older 256×256 BlazePalm/BlazeHand exports (2,944 palm anchors and a three-output landmark model). Those exports were also prepared locally for comparison. Their compiled operation families are similar, but their application contract differs from this bundle. The [Qualcomm model card](https://huggingface.co/qualcomm/MediaPipe-Hand-Detection/blob/main/README.md) supplies the published timings; local offline preparation/simulation does not reproduce a hardware latency measurement.

## Replacement models

Use `samples/hand-tracking/training/` for replacement model definitions, export tools and evaluation. Preserve dataset/checkpoint provenance and explicit crop, joint-order, coordinate, unit, confidence and state contracts. Evaluate occlusion recovery, handedness, temporal behavior, metric accuracy and device latency together.
