# Headset Overlay & Diagnostic Rendering

## 1. OpenVR Overlay Architecture

The runtime runs as an **OpenVR Overlay Application** (`VRApplication_Overlay`), not an exclusive scene app.

### Current diagnostic overlay (authoritative)

| Property | Value |
| :--- | :--- |
| **Overlay key** | `frame.spatial.imx_monitor.vulkan` |
| **SBS flag** | **Off** — both eyes see the same 2D card |
| **Size** | **768 × 444** RGBA (opaque) |
| **Upload** | `SetOverlayTexture` with three persistent Vulkan images at ~**30 Hz** |
| **Pose** | Head-relative, $Z \approx -0.72\text{ m}$, width $\approx 0.92\text{ m}$ |
| **Layout** | Left IMX616 RGB \| linear metric depth \| Right IMX616 RGB |

See [`progress_imx_monitor.md`](progress_imx_monitor.md) for why this size/rate exist.

### SteamVR `SetOverlayRaw` constraint (Frame)

On this SteamVR ARM64 build, even small CPU Raw uploads eventually return **`VROverlayError_RequestFailed` (23)**. When that happens, the headset keeps showing the last successful texture. The runtime therefore uses compositor-compatible Vulkan instance/device extensions, the OpenVR-selected output GPU, and three persistent 768×444 textures. CPU composition is rate-gated before drawing and copied through mapped staging buffers at ~30 Hz.

---

## 2. Layout & Telemetry

```text
┌─────────────────────────────────────────┐
│ IMX616 STEREO  F####  FPS ##    LIVE   │
│ CAM ##HZ  LIVE  L#/R#                   │
├─────────────┬─────────────┬─────────────┤
│ L IMX616 RGB│ DEPTH .25-5M│ R IMX616 RGB│
└─────────────┴─────────────┴─────────────┘
```

* **RGB panels**: 616×616 NV12 → RGBA previews from Vulkan-detiled Y + UV of the IMX616 stereo pair.
* **Orientation**: Left preview is corrected 90° counter-clockwise; right preview is corrected 90° clockwise.
* **Producer synchronization**: The slot whose content just changed is treated as in-flight. The preview consumes the preceding synchronized ring slot, avoiding partial frames during Arcturus gaze-triggered mode writes.
* **Current validated input**: 2464×2464 from the 9,564,160-byte UBWC camera rings. Generic 37,777,408-byte `msm` display allocations and unrelated 1,769,472-byte ICP allocations are explicitly excluded; neither has validated IMX616 NV12 geometry.
* **Orange tick bar**: Advances with upload sequence so a frozen overlay is obvious.

---

## 3. Stereo pair (not monocular)

Cameras are the **`arcimx616` IMX616 RGB stereo** sensors (`video0` / `video3`, $2464\times2464$ NV12 UBWC). Capture uses lockstep left/right DMA slot selection.

The live worker rectifies both luma images with the current XRService stereo
extrinsics and runs bidirectional census matching through native Vulkan. Only
unique, left/right-consistent, locally supported disparities are converted with
`Z = fB/d` and displayed. The diagnostic overlay itself is **not** stereoscopic
SBS.

---

## 4. Threading

* **Camera worker**: Capture + detile + RGB conversion + full-extrinsic rectification; queues only the newest stereo pair and publishes a 616×616 map.
* **Stereo worker**: Runs the native Vulkan matcher asynchronously and replaces shared depth only after all confidence passes complete.
* **Render loop**: 70–90 FPS event/pose loop; submits a Vulkan texture at ~30 Hz.

---

## 5. World-space rendering

The active 3D path is now the separate OpenXR projection-layer executable
documented in [`openxr_point_cloud.md`](openxr_point_cloud.md). It uses the
runtime's per-eye poses and asymmetric FOVs, capture-time head poses, a stage
reference space, and hardware depth testing. The OpenVR card remains a 2D
camera/depth diagnostic.

## 6. Historical note (SBS world-locked cloud)

Earlier builds used `VROverlayFlags_SideBySide_Parallel`, a $1280\times720$ SBS buffer, world-locked floor grids, and colored 3D point splats. That path is **disabled** for the IMX monitor diagnostic. Reintroduce SBS / `SetOverlayTexture` when moving past Raw IPC limits.
