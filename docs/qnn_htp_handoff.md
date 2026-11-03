# Depth Backend Handoff

## Current State (2026-10-02)

The live default is a native, confidence-filtered Vulkan stereo matcher on the
Adreno 750. It does not use QNN or a learned model.

The pipeline is:

1. Read the latest `passthroughCam0FromPassthroughCam1Extrinsics` transform from
   XRService's `colorPassthroughCalibrationHistory.json`.
2. Build full stereo-rectification LUTs from the relative rotation, physical
   baseline, and current device intrinsics. The old 180°-flip-only approximation
   is gone.
3. Precompute 7×7 census descriptors on Vulkan.
4. Search 96 disparities in both directions with sparse cross aggregation.
5. Reject ambiguous matches, enforce left/right consistency, and remove pixels
   without local depth support.
6. Convert accepted disparity to 616×616 metric depth over 0.25–5.0 m.

Verified live measurements:

* 15–24 ms per complete GPU frame.
* Typically 7–14% confidence-filtered coverage, depending on scene texture.
* Roughly 70–81 FPS in the decoupled OpenVR loop.
* One captured frame produced 40,336 metric PLY vertices.
* `dsp_service` remains active and no swap is used.

The default is selected when `FRAME_DEPTH_BACKEND` is unset or set to
`vulkan-stereo`. CPU Census remains the failure fallback.

## Point-Cloud Export

Capture a single float32 depth map and rectified luma pair with:

```bash
FRAME_DEPTH_BACKEND=vulkan-stereo \
FRAME_DUMP_GPU_DEPTH=/tmp/depth.raw \
FRAME_DUMP_RECTIFIED_PREFIX=/tmp/rectified \
./bin/frame_spatial_runtime
```

After copying the three files to the host, create a binary PLY with:

```bash
ruby scripts/depth_to_ply.rb depth.raw rectified_left.raw cloud.ply 5.0 0
```

PLY coordinates are meters in the rectified camera frame: +X right, +Y up,
and -Z forward.

## Runtime Controls

```bash
# Native Vulkan stereo (default)
FRAME_DEPTH_BACKEND=vulkan-stereo ./bin/frame_spatial_runtime

# Calibrated CPU fallback
FRAME_DEPTH_BACKEND=cpu ./bin/frame_spatial_runtime

# Historical learned experiment
FRAME_DEPTH_BACKEND=vulkan-ai \
FRAME_AI_MODEL=bin/models/stereonet-vulkan-256.mnn \
./bin/frame_spatial_runtime
```

Healthy native startup includes:

```text
[DepthPipeline] Stereo alignment source: XRService latest history
[VulkanStereo] Census + bidirectional confidence pipeline ready
[VulkanStereo] ... ms, coverage=...%, range=... m
[Render] FPS: ... | ... | VULKAN STEREO ...MS
```

## Why QNN Was Abandoned

QNN GPU is incompatible with the stock Frame OS. `libQnnGpu.so` emits
Qualcomm-proprietary OpenCL extensions while the headset exposes the Adreno
through Mesa/Turnip. Rusticl enumeration does not make those generated kernels
portable.

The V75 HTP is present, but Valve's private `dsp_service` owns the usable DSP
route and the OS lacks the matching unsigned user-PD shell for generic FastRPC.
Do not stop or replace `dsp_service`; it is required by headset services.

QNN CPU executes the trained 786×490 StereoNet graph in roughly 36 seconds per
frame. The MNN Vulkan conversion runs in roughly 450–500 ms at 256×256, but its
enlarged output is too coarse for point clouds. It remains only as an optional
experiment.

The original `bin/models/hitnet_quantized.dlc` is a randomly initialized test
graph, not trained HITNet, and must not be presented as depth.

## Camera Recovery After Reboot

If `pidfd_getfd` reports `Operation not permitted`, restore YAMA access:

```bash
ssh frame "echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0"
```

## Health Check

```bash
ssh frame 'pgrep -af "^./bin/frame_spatial_runtime$"; \
  tail -n 30 /tmp/frame_spatial_runtime.log; \
  systemctl is-active dsp_service'
```
