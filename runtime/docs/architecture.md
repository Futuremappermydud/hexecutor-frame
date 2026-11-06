# System Architecture

## 1. Architectural Overview

The Frame Spatial Runtime is designed as a high-throughput, multi-threaded pipeline decoupled into two asynchronous loops:
1. **Camera Worker Thread**: DMA ingest + Vulkan UBWC detile (Y+UV), RGB preview conversion, and Arcturus mode-transition filtering.
2. **Main Render Thread**: 70–90 Hz OpenVR event loop; `SetOverlayTexture` (~30 Hz) using persistent Vulkan textures for an opaque stereo-camera HUD.

```mermaid
flowchart TD
    subgraph Hardware ["Steam Frame Hardware"]
        C0["Right IMX616 arcimx616"]
        C1["Left IMX616 arcimx616"]
        IMU["6-DoF Head Tracking"]
    end

    subgraph SteamVR ["SteamVR Process Tree"]
        VRC["vrcompositor"]
        SHM["/dmabuf: 8 x 9564160 UBWC NV12"]
        VRC --> SHM
    end

    subgraph Runtime ["Frame Spatial Runtime"]
        subgraph Capture ["DMA + Vulkan Detile"]
            PFD["pidfd_open / pidfd_getfd"]
            VK["Import QCOM UBWC NV12"]
            YUV["Copy Y all slots + UV live L/R"]
            AUTO["Auto-Discovery 8 slots"]
            AUTO --> PFD --> VK --> YUV
        end

        subgraph Worker ["Camera Worker"]
            PRE["NV12 to RGB preview + calibrated rectification"]
            AI["Async native Vulkan stereo"]
            GUARD["5K / 2.5K transition guard"]
            YUV --> PRE --> AI --> GUARD
        end

        subgraph Shared ["Shared Snapshot"]
            LOCK["mutex + atomic"]
            SNAP["preview L/depth/R + status"]
            GUARD --> LOCK --> SNAP
        end

        subgraph Render ["Main Thread"]
            HUD["768x444 L | depth | R card"]
            RAW["SetOverlayTexture ~30 Hz triple-buffered"]
            SNAP --> HUD --> RAW
        end
    end

    C0 & C1 --> VRC
    IMU -.-> Render
```

Authoritative HUD behavior: [`headset_rendering.md`](headset_rendering.md), [`progress_imx_monitor.md`](progress_imx_monitor.md).

---

## 2. Component Breakdown

### A. Zero-Copy Hardware DMA Ingestion (`DMACapture`)
* **Role**: Obtain compositor camera DMA-BUFs and Vulkan-detile them.
* **Mechanism**: `pidfd_open` / `pidfd_getfd`, import QCOM UBWC NV12, copy Y (all ring slots) and UV (live L/R).
* **Auto-Discovery**: Eight `/dmabuf:` FDs of size `9,564,160` → 4 right + 4 left (**IMX616 stereo RGB**).

### B. Camera Preview Pipeline
* **Role**: Produce stable 616×616 RGB views for both IMX616 cameras.
* **Mode changes**: Debounce the full-readout ↔ binned transition because Arcturus keeps the DMA allocation dimensions fixed while changing sensor mode.
* **Depth**: The worker performs full-extrinsic rectification and submits only the newest pair to native Vulkan census, bidirectional disparity, consistency, and support passes. CPU Census/WTA is the initialization/runtime failure fallback.
* **GPU arbitration**: Overlay texture uploads are deferred during stereo dispatch, preserving OpenVR polling and retaining the previous valid texture on screen.

### C. Overlay Renderer (`OverlayRenderer`)
* **Role**: Opaque head-locked diagnostic monitor (SBS off).
* **Key**: `frame.spatial.imx_monitor.vulkan`, 768×444, triple-buffered Vulkan texture upload because Frame SteamVR eventually rejects Raw uploads (`VROverlayError_RequestFailed`).
* **Content**: Left RGB | linear metric depth | Right RGB + FPS/frame telemetry.
