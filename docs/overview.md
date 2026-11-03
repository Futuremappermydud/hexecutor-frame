# Project Overview: Frame Spatial Runtime

## 1. Executive Summary
**Frame Spatial Runtime** is a real-time, low-latency spatial depth estimation and 3D room reconstruction engine built specifically for the **Valve Steam Frame (Deckard)** standalone VR headset.

By interfacing directly with the headset's physical **IMX616 stereo RGB** cameras via shared Linux DMA-BUFs (Vulkan UBWC detile), the runtime bypasses OS-level camera lockouts, builds full epipolar rectification from XRService's current stereo extrinsics, runs confidence-filtered census matching through Vulkan, and shows a live diagnostic overlay (RGB L/R + linear depth) in the headset.

---

## 2. Motivation & Problem Statement
Modern spatial computing runtimes require real-time 3D awareness of the physical room to enable:
* Accurate collision meshes and real-time physical interaction with real-world obstacles (desks, chairs, walls, doors).
* Occlusion culling between virtual holographic objects and real hands/furniture.
* Seamless mixed reality (MR) passthrough experiences.

### Why Standard APIs Failed
During initial research and prototyping on the Steam Frame, standard APIs proved insufficient:
1. **SteamVR Chaperone (`IVRChaperone`)**: Only returns a pre-configured 2D convex polygon representing the floor play area boundary. It has zero awareness of 3D objects, tables, or dynamic room changes.
2. **OpenXR (`XR_EXT_hand_tracking`, `XR_KHR_visibility_mask`)**: The current Deckard OpenXR runtime does not expose scene meshing (`XR_MSFT_scene_understanding`) or raw camera access extensions (`XR_HTC_passthrough` / `XR_META_passthrough_color`).
3. **OpenVR Tracked Camera (`IVRTrackedCamera`)**: Always returns `VRTrackedCameraError_OperationFailed` (Error 100) or `CameraError_NoCameraFrameRequested` on the Frame because camera management is privately retained by Valve's internal `XRService`.
4. **V4L2 Loopback (`/dev/video99`)**: The `/dev/video99` device created by `v4l2cam` only outputs a monocular 1080p compositor spectator view, lacking true stereo baseline parallax required for depth triangulation.

### The Breakthrough
To achieve true spatial reconstruction, we engineered a direct hardware pathway:
* Intercepting `vrcompositor`'s direct hardware DMA-BUFs containing the raw 2464x2464 @ 90Hz camera frames using Linux `pidfd_open` and `pidfd_getfd`.
* Ingesting the factory Kannala-Brandt fisheye calibration and compensating for the hardware-level 180° optical parity flip between the left and right sensors.
* Running 96-disparity bidirectional census matching through native Vulkan on the Adreno 750 and publishing a 616×616 metric depth texture.
* Rendering a head-locked OpenVR diagnostic overlay around 70–81 FPS while confidence-filtered depth runs in roughly 15–24 ms.

---

## 3. High-Level Performance Metrics
| Metric | Measurement |
| :--- | :--- |
| **Raw Camera Input** | Dual **IMX616** 2464 × 2464 NV12 UBWC (`arcimx616`) |
| **Rectified Virtual Resolution** | 616 × 616 stereo pair |
| **Disparity Search Range** | 96 disparities |
| **Depth Matching Algorithm** | Vulkan 7×7 Census + cross aggregation + bidirectional consistency |
| **GPU Depth Time** | ~15–24 ms at 616 × 616 |
| **Metric Depth Texture** | Confidence-filtered 616 × 616 float depth, 0.25–5.0 m |
| **Render Loop** | ~70–81 FPS during native stereo |
| **Overlay upload** | 768 × 444 Vulkan texture @ ~30 Hz |
| **Display Mode** | Opaque head-locked IMX monitor (SBS off) — see [headset_rendering.md](headset_rendering.md) |

---

## 4. Key Documentation Links
* [System Architecture](architecture.md)
* [Hardware & Sensor Calibration](hardware_and_calibration.md)
* [DMA-BUF Hardware Ingestion](dma_buf_capture.md)
* [Stereo Depth Estimation Pipeline](depth_pipeline.md)
* [Headset Overlay Rendering](headset_rendering.md)
* [Building and Running Guide](build_and_deploy.md)
* [IMX Monitor Progress Log](progress_imx_monitor.md)
* [Roadmap & Next Steps](next_steps.md)

* [Guarded GPU Matcher Optimization Plan](stereo_guarded_optimization_plan.md)
* [Persistent Room Mesh Plan](room_mesh_plan.md)
