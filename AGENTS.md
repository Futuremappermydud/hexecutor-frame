# AGENTS.md — AI Agent Guidance & Technical Briefing

Welcome! If you are an AI assistant or autonomous agent working on this codebase, **read this document carefully before making changes or running commands**.

This document details the architectural invariants, critical hardware peculiarities, system permissions, and operational guidelines for the **Frame Spatial Runtime** project.

---

## 1. Project Mission & Objective

This project builds a real-time, low-latency dense metric depth estimation and 3D spatial room reconstruction runtime on the **Valve Steam Frame (Deckard)** standalone VR headset.

### Core Pipeline
```text
Hardware DMA-BUFs (vrcompositor) — IMX616 stereo RGB (arcimx616)
 ├── Right Camera (2464x2464 NV12 UBWC, 180° inverted mount)
 └── Left Camera  (2464x2464 NV12 UBWC)
 │
 ▼ (pidfd_open + pidfd_getfd; Vulkan UBWC detile Y+UV)
Kannala-Brandt Fisheye Rectification LUT (616x616, right ray flipped)
 │
 ▼
64-bit Census Transform (7x7) + WTA disparity → dense metric Z
 │
 ▼
OpenVR opaque HUD: L RGB | linear depth | R RGB  (256x96 Raw ~10 Hz)
```

> Camera note: The 9,564,160-byte DMA slots are **IMX616 stereo RGB**, not OG01A1B (those are smaller GREY tracking sensors). Details: [`docs/progress_imx_monitor.md`](docs/progress_imx_monitor.md).

---

## 2. Remote Target Environment & Access

* **Remote Host**: `frame` (reachable via `ssh frame`, user `steamos`).
* **Architecture**: `linuxarm64` (`aarch64`, 8-core AMD Zen 4 CPU + Qualcomm Adreno GPU).
* **Remote Project Path**: `/home/steamos/frame-spatial-runtime/`.
* **Sudo Password**: `frame1`.
* **SteamVR Installation Paths**:
  * OpenVR Headers: `/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers/openvr.h`
  * ARM64 OpenVR Library: `/opt/steamvr/bin/linuxarm64/libopenvr_api.so`
  * VR Compositor Binary: `/opt/steamvr/bin/linuxarm64/vrcompositor`
* **Local Workspace**: `/Users/owenelliott/Documents/projects/frame-spatial-runtime/`

---

## 3. Critical Hardware & System Invariants

### Invariant 1: Stereo Depth Uses IMX616 Luma (Not OG01A1B GREY)
* The DMA slots of size **9,564,160** are the **Sony IMX616** stereo RGB pair (`arcimx616` → `video0` / `video3`, 2464×2464 NV12 UBWC).
* The **OG01A1B** sensors are separate smaller GREY tracking cameras — do not confuse them with these buffers.
* Stereo matching runs on the **Y plane** after detile (census needs luminance). UV is for RGB HUD / future texturing.
* Do not try to open `/dev/video*` directly; use compositor DMA-BUFs (Invariant 3).

### Invariant 2: The Right Sensor has a 180° Optical Parity Inversion
* The physical Right camera sensor is mounted **180° rotated** relative to the Left sensor.
* When generating the rectification rays:
  * Left ray: $\mathbf{r}_{\text{left}} = (r_x, r_y, 1.0)$
  * Right ray: $\mathbf{r}_{\text{right}} = (-r_x, -r_y, 1.0)$
* If you omit the negative signs on the right ray, epipolar geometry will break and disparity matching will fail completely.

### Invariant 3: Never Open `/dev/video*` Directly
* The camera hardware devices are claimed exclusively by `XRService` and `vrcompositor`. Standard `open("/dev/video0")` calls return `EBUSY`.
* Never use `/dev/video99` for stereo depth — that is a monocular compositor spectator mirror created by `v4l2cam`.
* Camera frames **must** be captured via `pidfd_open` / `pidfd_getfd` from `vrcompositor`'s DMA-BUFs.

### Invariant 4: Always Auto-Discover DMA-BUF File Descriptors
* **Never hardcode file descriptor numbers** (e.g. 216, 223, 228).
* Every time SteamVR restarts, `vrcompositor` gets a new PID and allocates new file descriptors.
* Always use `DMACapture::autoDiscoverCameras()`, which scans `/proc/<pid>/fd/` for symlinks pointing to `/dmabuf:` with file size exactly equal to `9,564,160` bytes.

### Invariant 5: Kernel Ptrace Scope Must Be 0
* Linux YAMA ptrace security blocks `pidfd_getfd` across non-parent processes if `ptrace_scope > 0`.
* If `pidfd_getfd` returns `Operation not permitted` (EPERM), run:
  ```bash
  ssh frame "echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0"
  ```
  Or run `./scripts/setup_frame.sh`.

### Invariant 6: OpenVR Coordinate System
* **Tracking Universe**: `vr::TrackingUniverseStanding`.
* **Axes**:
  * $+X$ = User Right
  * $+Y$ = Up (Ceiling)
  * $-Z$ = Forward (Eye gaze direction: **Negative Z is Forward**)
* **Overlay Flag**: The **diagnostic IMX monitor** uses SBS **off** (both eyes see one card). Re-enable `vr::VROverlayFlags_SideBySide_Parallel` only for true stereo world-locked overlays; see [`docs/headset_rendering.md`](docs/headset_rendering.md).

---

## 4. Key Workflows & Commands

### Deploy and Run on Frame
To sync changes, build with OpenMP, and execute in the headset:
```bash
./scripts/deploy_and_run.sh
```

### Manual Compilation on Frame
```bash
ssh frame "cd /home/steamos/frame-spatial-runtime && make clean && make -j4"
```

### Launch Binary on Frame
```bash
ssh frame "/home/steamos/frame-spatial-runtime/bin/frame_spatial_runtime"
```

### Quick Diagnostic Dump of Stereo Pair
To capture a synchronized pair to PNG/PPM on your host:
```bash
python3 scripts/dump_stereo_pair.py
```

---

## 5. Codebase Map

| File / Directory | Purpose |
| :--- | :--- |
| [`include/calibration.hpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/include/calibration.hpp) | Factory intrinsics, extrinsics, Kannala-Brandt projection function. |
| [`include/dma_capture.hpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/include/dma_capture.hpp) | `DMACapture` class definition with auto-discovery signature. |
| [`src/dma_capture.cpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/src/dma_capture.cpp) | `pidfd_open`, `pidfd_getfd`, `/proc` scanner, `mmap` zero-copy sharing. |
| [`include/depth_pipeline.hpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/include/depth_pipeline.hpp) | `DepthPipeline` class definition. |
| [`src/depth_pipeline.cpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/src/depth_pipeline.cpp) | OpenMP bilinear rectification LUT, 64-bit Census, SGM disparity matcher. |
| [`include/overlay_renderer.hpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/include/overlay_renderer.hpp) | `OverlayRenderer` class definition. |
| [`src/overlay_renderer.cpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/src/overlay_renderer.cpp) | OpenVR Side-by-Side overlay, eye matrix projection, 2px splatting, HUD. |
| [`src/main.cpp`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/src/main.cpp) | Entry point: decouples Depth Worker (~22 Hz) from Render Loop (~70–80 Hz). |
| [`shaders/`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/shaders/) | Vulkan Compute Shaders (`rectify_ssbo.comp`, `stereo_sgm.comp`). |
| [`scripts/`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/scripts/) | Shell & Python utility scripts (`setup_frame.sh`, `deploy_and_run.sh`). |
| [`tools/`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/tools/) | Standalone diagnostic tools and prior exploratory scripts. |
| [`docs/`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/docs/) | Comprehensive documentation suite. |

---

## 6. Guidelines for Making Changes

1. **Maintain Thread Separation**:
   * Do not block the render loop (`main.cpp`) with depth calculations. The render loop must poll `GetDeviceToAbsoluteTrackingPose` and render at 70–90 FPS to prevent user motion sickness.
   * Disparity matching belongs in the background `depthWorker` thread.
2. **Preserve Zero-Copy Semantics**:
   * Never copy raw camera buffers across process boundaries. Read directly from the `mmap` pointers.
3. **Keep OpenMP Flags**:
   * Always compile with `-O3 -fopenmp`. Disparity matching across $616 \times 616 \times 64$ is computationally intensive; parallelization over the 8 CPU cores is essential for ~45ms compute times.
4. **Update Documentation**:
   * When modifying camera parameters, pipeline stages, or overlay geometry, keep the corresponding docs in [`docs/`](file:///Users/owenelliott/Documents/projects/frame-spatial-runtime/docs/) synchronized.
