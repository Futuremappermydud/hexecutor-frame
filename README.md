# Frame Spatial Runtime

[![Platform](https://img.shields.io/badge/Platform-Steam%20Frame%20%28Deckard%29-blue.svg)](#)
[![Architecture](https://img.shields.io/badge/Architecture-ARM64%20%28Linux%29-green.svg)](#)
[![Display](https://img.shields.io/badge/OpenXR-World%20Point%20Cloud-orange.svg)](#)
[![Performance](https://img.shields.io/badge/Render%20Rate-70--81%20FPS-brightgreen.svg)](#)

A real-time, low-latency spatial depth estimation and 3D room reconstruction runtime for the **Valve Steam Frame (Deckard)** standalone VR headset.

This engine bypasses user-space camera lockouts via Linux kernel DMA-BUF sharing,
rectifies the stereo fisheye pair using the headset's live factory extrinsics,
and runs bidirectional confidence-filtered stereo on the Adreno GPU through
Vulkan. A dedicated OpenXR scene app fuses repeat-observed samples into a
stage-space cloud and displays them with true per-eye projection; the separate
OpenVR diagnostic overlay still provides left RGB, metric depth, and right RGB.

---

## Key Features
* **Zero-Copy Camera Ingestion**: Captures uncompressed $2464 \times 2464$ @ 90Hz stereo streams from `vrcompositor` via `pidfd_open` and `pidfd_getfd`.
* **Auto-Discovery Engine**: Dynamically detects and binds to active camera DMA-BUFs across SteamVR restarts without hardcoded descriptors.
* **Full Stereo Calibration**: Builds epipolar rectification from the latest XRService relative-camera transform, current device intrinsics, and the physical parity inversion.
* **Native Vulkan Depth**: Precomputed census, 96-disparity bidirectional matching, uniqueness rejection, left/right consistency, and local support filtering run in roughly 15–24 ms.
* **World-Locked OpenXR Cloud**: Capture-midpoint poses, 2 cm voxel fusion, repeat-observation filtering, and depth-tested per-eye rendering keep accepted points fixed in stage space.
* **OpenXR Passthrough Occlusion**: An opt-in Vulkan API layer uses a current depth surface with bounded hole filling and triangles that stop at depth jumps to mask virtual pixels behind real surfaces, only when applications already enable passthrough and submit stereo depth. See [usage and compatibility](docs/openxr_occlusion.md).
* **Metric Point Clouds**: A supplied exporter converts confidence-filtered float depth to a camera-local binary PLY in meters.
* **Safe CPU Fallback**: OpenMP-parallelized Census/WTA remains available when the Vulkan backend cannot initialize.
* **Live Diagnostic Overlay**: An opaque head-locked card displays `L RGB | metric depth | R RGB`; GPU texture uploads yield while inference is active.

---

## Directory Structure

```text
frame-spatial-runtime/
├── include/                   # Public C++ headers
│   ├── calibration.hpp        # Factory camera intrinsics, extrinsics, KB model
│   ├── depth_pipeline.hpp     # Stereo rectification and CPU fallback
│   ├── gpu_depth_pipeline.hpp # MNN Vulkan StereoNet backend
│   ├── vulkan_stereo_pipeline.hpp # Native Vulkan matcher
│   ├── dma_capture.hpp        # Zero-copy DMA-BUF acquisition via pidfd
│   ├── overlay_renderer.hpp   # OpenVR Side-by-Side stereoscopic rendering
│   └── spatial_types.hpp      # Point3D, HmdPose, and frame structures
├── src/                       # Implementation files
│   ├── calibration.hpp
│   ├── depth_pipeline.cpp     # Rectification LUT and Census fallback
│   ├── gpu_depth_pipeline.cpp # Learned disparity and metric conversion
│   ├── vulkan_stereo_pipeline.cpp # Confidence-filtered native stereo
│   ├── dma_capture.cpp        # Auto-discovery & mmap buffer sharing
│   ├── overlay_renderer.cpp   # OpenVR overlay, eye projection, HUD
│   ├── openxr_point_cloud.cpp # World fusion + OpenXR projection layer
│   └── main.cpp               # Asynchronous dual-threaded runtime
├── shaders/                   # Vulkan compute shaders for GPU acceleration
│   ├── rectify.comp           # KB fisheye rectification compute shader
│   ├── rectify_ssbo.comp      # SSBO-based rectification compute shader
│   ├── stereo_census.comp     # Precomputed 7x7 descriptors
│   ├── stereo_match.comp      # Bidirectional 96-disparity matching
│   ├── stereo_filter.comp     # Consistency + metric conversion
│   └── stereo_support.comp    # Isolated-match rejection
├── scripts/                   # Workflow and automation scripts
│   ├── setup_frame.sh         # Lift ptrace restrictions on Steam Frame
│   ├── setup_vulkan_ai.sh     # Install pinned MNN Vulkan runtime
│   ├── deploy_and_run.sh      # Rsync, compile, and run on Frame
│   ├── depth_to_ply.rb        # Metric float depth to binary PLY
│   └── dump_stereo_pair.py    # Utility to dump live stereo frames to PNG/PPM
├── tools/                     # Standalone diagnostic and exploration tools
│   ├── standalone_spatial_mesh_runtime.cpp # Single-file runtime
│   ├── camera_probe.cpp       # V4L2 and XR camera probe
│   ├── dump_stereo_pair.c     # Standalone DMA dumper
│   ├── test_hmd_pose.cpp      # Standalone HMD tracking pose test
│   └── cpu_stereo_depth.cpp   # Offline file-based CPU depth matcher
├── docs/                      # Comprehensive technical documentation
│   ├── overview.md            # Motivation, problem statement, results
│   ├── architecture.md        # System architecture, threading, dataflow
│   ├── hardware_and_calibration.md # Sensors, OG01A1B, KB distortion
│   ├── dma_buf_capture.md     # DMA-BUF ingestion via pidfd_open
│   ├── depth_pipeline.md      # Rectification, Census, SGM disparity
│   ├── headset_rendering.md   # OpenVR SBS overlay & world-locking
│   ├── openxr_point_cloud.md  # Timestamped fusion and OpenXR renderer
│   ├── build_and_deploy.md    # Compilation, deployment, and troubleshooting
│   └── next_steps.md          # Roadmap: Meshing, TSDF, GPU compute
├── CMakeLists.txt             # CMake build configuration
├── Makefile                   # High-speed compilation makefile
└── README.md                  # This file
```

---

## Quickstart

### 1. Configure the Frame (One-time)
Lift ptrace restrictions so the runtime can duplicate file descriptors from `vrcompositor`:
```bash
./scripts/setup_frame.sh
```

### 2. Build & Launch in Headset
Deploy to the headset, compile with OpenMP, and start the holographic overlay:
```bash
./scripts/deploy_and_run.sh
```

To launch the world-locked OpenXR point cloud instead:

```bash
ssh frame "cd /home/steamos/frame-spatial-runtime && make -j4 bin/frame_openxr_point_cloud && ./bin/frame_openxr_point_cloud"
```

The optimized Model HMX model is available in the same app:

```bash
ssh frame "cd /home/steamos/frame-spatial-runtime && OMP_NUM_THREADS=2 ./bin/frame_openxr_point_cloud --depth-model"
```

This selects the experimental learned disparity backend with independent stereo
validation. See [Model backend setup and limitations](docs/openxr_point_cloud.md#model-hmx-depth-backend).
`--depth-model-prepared` runs the same model through the exact prepared DSP
executor instead (about 15.5 s per depth frame, needs `cdsp_mem`); see
[its notes](docs/openxr_point_cloud.md#exact-prepared-executor-backend-experimental).

### 3. Live Telemetry
While running, the runtime reports real-time metrics both on the console and directly on your headset HUD:
```text
[OverlayRenderer] Initialized stereo overlay (SBS 1280x720)
[DepthPipeline] Initialized rectification LUTs (616x616, MaxDisp=64)
[DMACapture] Auto-discovered camera DMA-BUFs: Right=228, Left=243 (from 8 slots)
[Main] Stereoscopic spatial mesh active in headset.
[Worker] Depth estimation thread started.
[Render] FPS: 73.34 | 3D Points: 61946
[Render] FPS: 77.63 | 3D Points: 62039
[Render] FPS: 81.12 | 3D Points: 62086
```

---

## Documentation Index
* [Project Overview](docs/overview.md)
* [System Architecture](docs/architecture.md)
* [Hardware Sensors & Calibration](docs/hardware_and_calibration.md)
* [Zero-Copy DMA-BUF Ingestion](docs/dma_buf_capture.md)
* [Stereo Depth Estimation & SGM](docs/depth_pipeline.md)
* [Headset Stereoscopic Overlay](docs/headset_rendering.md)
* [OpenXR World Point Cloud](docs/openxr_point_cloud.md)
* [OpenXR Passthrough Occlusion](docs/openxr_occlusion.md)
* [Building and Deployment Guide](docs/build_and_deploy.md)
* [Roadmap & Next Steps](docs/next_steps.md)
