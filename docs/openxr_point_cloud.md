# OpenXR World Point Cloud

`bin/frame_openxr_point_cloud` is the world-space renderer. Unlike the OpenVR
diagnostic card, it owns a stereo OpenXR projection layer and renders metric
points with the runtime-provided pose, FOV, and eye separation for each eye.

## Data flow

```text
IMX616 DMA-BUF pair
  -> calibrated 616x616 fisheye rectification
  -> Vulkan bidirectional stereo + consistency/support rejection
  -> capture-midpoint OpenXR head pose
  -> disconnected depth patch rejection
  -> calibrated rectified-camera-to-head transform
  -> 2 cm stage-space voxel fusion
  -> repeat-observed points only
  -> depth-tested OpenXR stereo projection layer
```

The app enables `XR_KHR_vulkan_enable2` so SteamVR chooses the correct Vulkan
physical device, and `XR_KHR_convert_timespec_time` so the midpoint of each DMA
capture can be converted from `CLOCK_MONOTONIC` into the OpenXR time domain.
`xrLocateSpace` then supplies the head pose for that capture rather than the
later inference-completion time. The latest render pose remains the fallback for
fusion if lookup fails, but purging requires a successfully located capture pose.

Rectification creates a rotated virtual camera frame, which must be transformed
back through the left optical frame before applying camera-to-head calibration.
`DepthPipeline::rectifiedPointInHead()` shares the exact basis used by the LUT.
It composes XRService's latest `passthroughCam0FromPassthroughCam1Extrinsics`
and `frontCam0FromPassthroughCam0Extrinsics` with this unit's SteamVR
`cv.cad_from_cal` and `head` axes. The latter rig-to-head values are stored as
unit-specific constants in `depth_pipeline.cpp`; update them for another unit.

The calibrated rectified forward vector is approximately
`(-0.0083, -0.2265, -0.9740)` in head coordinates (about 13 degrees down).
Depth is referenced to the **left optical center**, not the stereo midpoint.
Using only `(x, -y, -z)` omits this optical pitch and tilts the reconstructed
room upward. Points are then transformed by the capture-time head pose into
`STAGE` space.

## Accuracy filtering

- Only the native Vulkan matcher's unique, bidirectionally consistent, locally
  supported depths between 0.25 m and 5 m enter fusion.
- The GPU support pass requires at least three of eight adjacent pixels to
  agree within 4 cm or 4% depth.
- Eight-connected, depth-consistent patches smaller than 32 pixels are rejected
  in the worker before fusion; missing depth is never filled.
- Every third distinct stereo pair, existing world voxels are projected into
  current camera depth using the capture pose and inverse calibrated rig.
- An old point is purged after at least five checks accumulating one second of
  supported measurements behind it. Six valid samples in its 5×5 neighborhood
  must all lie farther away; any agreeing or nearer sample protects it.
- Visibility tolerance accounts for stereo uncertainty: the maximum of 6 cm,
  3% of range, and a 0.75-pixel disparity error converted to depth.
- Confirmation or occlusion resets removal evidence. Missing depth, highlights,
  out-of-view points, and unavailable capture poses do not add evidence. Long
  processing gaps count for at most 0.2 seconds per check.
- Reused DMA snapshots are skipped: both camera hashes must change before
  another pair can count toward voxel confirmation.
- Samples are fused in 2 cm world voxels.
- A voxel is invisible until observed in at least three different camera pairs
  and supported by four other confirmed voxels within 4 cm.
- Repeated observations update a capped running mean instead of adding duplicate
  points.
- Unconfirmed voxels expire after 30 processed pairs; all voxels expire after
  300 pairs without another observation.
- The renderer caps its working set at 350,000 voxels and draws at most 260,000
  stable points.

## Build and run

From the headset project directory:

```bash
make -j4 bin/frame_openxr_point_cloud
./bin/frame_openxr_point_cloud
```

The process log reports both stereo timing and pose association:

```text
[OpenXR] Eye 0: 1728x1728, 3 images
[Cloud] frame=30 stereo=10.4 ms, accepted=5691, stable=7830, voxels=21391, pose=capture-time
```

Only one scene application can have OpenXR focus. The point cloud keeps
capturing while the session is synchronized, and becomes visible when the
headset transitions to the visible/focused state.

## Current limitation

The compositor DMA ring does not expose the sensor's hardware exposure timestamp.
Reading a DMA snapshot happens after exposure, so head movement can still cause
registration error even though the cloud is already projected through each eye's
current predicted OpenXR pose on every render frame. Passthrough's own latency
and reprojection can also affect apparent alignment.

No estimated timing offset is applied. Purging changes point lifetime only;
matching, measured point positions, and render projection remain unchanged.
A sensor timestamp bridge is still needed to measure exposure timing accurately.

## Regression check

```bash
g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_cloud_geometry.cpp \
  src/depth_pipeline.o -o /tmp/test_cloud_geometry
/tmp/test_cloud_geometry
```

This checks optical pitch, left-camera origin, metric length, image-axis
orientation, inverse rig transformation, and rejection of small depth patches.

```bash
g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_cloud_visibility.cpp \
  -o /tmp/test_cloud_visibility
/tmp/test_cloud_visibility
```

Visibility tests cover an object moving away, recovery on re-observation,
occlusion, highlights/missing depth, isolated mismatches, distant depth noise,
and capture gaps. Stricter support reduces coverage on thin/poorly textured
surfaces. Purging cannot immediately remove an object when the newly exposed
background has no reliable depth; the existing 300-pair general expiration
still applies in that case.

## Passthrough occlusion

The fusion worker is shared with the opt-in [OpenXR occlusion layer](openxr_occlusion.md).
`--occlusion-demo` renders virtual panels with submitted depth to exercise
that layer; ordinary point cloud rendering remains the default.

## Guarded GPU point-cloud preview

Set `FRAME_CLOUD_STEREO=guarded` when launching the ordinary point-cloud app
(no `--occlusion-demo`) to use the experimental g3/e1/r8/t32/c120/m3 matcher:

```bash
FRAME_CLOUD_STEREO=guarded OMP_NUM_THREADS=2 ./bin/frame_openxr_point_cloud
```

This draws stage-space points with no comparison HUD. The same capture-time
pose, 2 cm voxel fusion, three distinct observations, spatial support, and
moving-object visibility checks apply. The default matcher remains baseline;
the shared occlusion worker uses its independent `FRAME_OCCLUSION_STEREO` setting. The guarded matcher has
not passed the offline quality promotion gates.

The point-cloud path submits guarded matching in eight-row batches with a
3 ms host pause between submissions, allowing graphics work between batches.
Matching stays in the depth worker. Render cadence must be measured separately
from depth update cadence; enable `FRAME_OPENXR_RENDER_PROFILE=1` to inspect
blocking draw times. The parity test compares this batched regular path against
the unsplit diagnostic path, including repeated-call determinism.


## Model HMX depth backend

The existing viewer accepts `--depth-model` (equivalent to
`FRAME_CLOUD_STEREO=model`), selecting the optimized recovered Model primary model:

```bash
cd /home/steamos/frame-spatial-runtime
make -j4 bin/frame_openxr_point_cloud
OMP_NUM_THREADS=2 ./bin/frame_openxr_point_cloud --depth-model
```

Without that flag/environment setting the existing Vulkan backend remains the
default. The Model path is for the ordinary point-cloud viewer; occlusion retains
its independently selected and validated backends.

The Frame installation requires the previously validated package at
`~/frame-dsp/depth-model/depth-model-direct-v5` and its Python environment at
`~/frame-dsp/depth-model/.venv/bin/python`. It compiles the package's small native
ARM helpers on first use if necessary. Optional overrides are
`FRAME_DEPTH_PACKAGE`, `FRAME_DEPTH_PYTHON` and `FRAME_DEPTH_MODEL=alternate`.
The installed package manifest is verified before acquiring the DSP. Run one
NPU callback client at a time: chat and Model depth cannot occupy it together.
Eye inference is paused for the session and restored after the worker exits.

`ModelDepthPipeline` runs solely in the background capture/fusion worker. It
starts one persistent `scripts/model_depth_bridge.py` process, loads the graph
once and reuses the HMX session. A private inherited Unix socket carries only
the resized, rectified luma pair and float output; raw camera DMA data stays in
the existing process. No camera file descriptors are hardcoded or video devices
opened. The render thread continues to locate/render the current head pose.

The calibrated 616×616 images are resized using pixel centers to the model's
256×320 input. Plane 0 is provisionally interpreted as left-image horizontal
disparity in **model pixels**. Metric conversion uses
`Z = FOCAL_BASELINE × (256 / 616) / disparity`; vertical scaling does not enter
the horizontal disparity formula. Nearest-neighbor expansion back to 616×616
preserves rejected holes and boundaries. The existing camera-to-head transform,
capture-midpoint pose, distinct-pair checks, speckle rejection, voxel confirmation
and visibility aging then apply as before.

The adapter rejects nonfinite/negative disparity, depths outside 0.25–5 m,
invalid correspondence borders, 5×5 patches with less than 12 luma levels of
contrast, and 5×5 census disagreement above 8 of 24 bits. At least three of eight
neighbors must agree within 4 cm or 4% depth. These checks are independent of the
uninterpreted auxiliary output. Channel 1 is not assumed to be right-image depth
or confidence. Hidden state is zero and transforms are identity on every pair;
no unverified temporal feedback or motion warp is introduced.

This remains an **experimental geometry interpretation**, not a new metric
calibration or a quality promotion. Low texture and domain differences can make
the cloud sparse. Independent stereo checks reduce unsupported predictions but
do not establish a confidence calibration. In the Oct 29 test, live fusion
produced a small confirmed cloud; room-scale coverage needs wearer evaluation.

Validation on Oct 29, 2026:

- Geometry tests cover identity resize, disparity/focal scaling, zero/NaN/range
  rejection, low texture and unsupported correspondences.
- A captured IMX616 pair ran three times through the persistent hardware bridge:
  529.6 ms cold, 403.2/401.5 ms warm inference; 419.1/411.6 ms warm end to end.
  It retained 16,901 validated image pixels before the fusion worker's speckle
  pass. The existing Vulkan matcher retained 8,290; their 4,363-pixel overlap
  had 1.73% median relative depth difference. This is comparison, not ground truth.
- The actual viewer reached 20 distinct depth frames and confirmed fused voxels,
  using capture-time poses. Fully rendered profiling windows ran near 72 Hz;
  live inference logs ranged approximately 0.44–0.57 seconds under rendering load.
  This describes separate render and depth cadences, not 72 depth updates/sec.
- Closing the viewer exited the Python context and restored `[1,0,1]` controls,
  zero callback/payload/arena and one running eye client. Missing-package startup
  failed before acquiring DSP ownership. Existing cloud geometry/visibility tests
  and Make/CMake builds were checked separately.

Reproduce the adapter checks on Frame:

```bash
make -j4 bin/test_model_depth_geometry bin/test_model_depth_pipeline bin/capture_rectified_pair
./bin/test_model_depth_geometry
./bin/capture_rectified_pair /tmp/model-pair
./bin/test_model_depth_pipeline /tmp/model-pair-left.raw /tmp/model-pair-right.raw /tmp/depth-model.raw
```

### Exact prepared-executor backend (experimental)

`--depth-model-prepared` (or `FRAME_DEPTH_BACKEND=prepared` together with
`FRAME_CLOUD_STEREO=model`) runs the same model through the NPRG v2 prepared
executor. The whole model runs as one resident DSP call, bit-exact with the
emulator:

```bash
# once per boot, after reading DSP state (see depth-model master handoff)
ssh -t frame 'cd ~/frame-dsp/depth-model/perf-20261029 && sudo insmod frame/kmod/cdsp_mem.ko size_mb=128'
OMP_NUM_THREADS=2 ./bin/frame_openxr_point_cloud --depth-model-prepared
```

- The package defaults to `~/frame-dsp/depth-model/perf-20261029`.
  `FRAME_DEPTH_PACKAGE` overrides it.
- The bridge checks the pinned sha256 of the engine, the guarded bootstrap, the
  loader, the NPRG program and the QDRX image before taking the DSP lock.
  `ResidentDSP` refuses to start unless `cdsp_mem` is loaded at 128 MB and its
  private map is clean.
- One inference takes about **15.5 s**, so depth updates are very slow; the
  render loop is unaffected. The socket timeout defaults to 60 s
  (`FRAME_DEPTH_TIMEOUT_S`). On shutdown the app waits that long for an
  in-flight inference to finish before it sends SIGTERM.
- Nov 1, 2026: `test_model_depth_pipeline` on the Oct 29 pair produced depth
  byte-identical to the hybrid backend (16,901 valid pixels).
- **Known issue:** that session's shutdown left `dsp_service` at control
  `(2,0,8)`. The DSP side had exited and restored VTCM cleanly, but only a
  reboot recovered the service. Until this is understood, check
  `frame/bench/depth_state.py` after every session. See the depth-model
  `HANDOFF_20261101_FRAME_FULL_RUN.md`.

The native bridge uses bounded socket waits, validates response sequence and
shape, and stops publishing on failure. It closes the socket on normal shutdown,
allowing model/clock/eye-client cleanup; it does not forcibly kill a worker that
has not completed DSP cleanup. Logs and captures from this integration are under
`artifacts/depth-model-20261029/` on Frame.
