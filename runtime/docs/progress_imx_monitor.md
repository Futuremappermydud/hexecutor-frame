# Progress Notes — IMX616 Stereo Monitor Overlay

Working log of the changes that got a live diagnostic overlay onto the Steam Frame headset. Use this as the source of truth for the current diagnostic mode; older docs that still describe SBS point-cloud overlays are historical until fully rewritten.

---

## 1. Goal

Show something **visibly updating** in the headset so pipeline progress is obvious:

1. Overlay (including FPS) must update live.
2. Cameras must be the **stereo RGB `arcimx616` / Sony IMX616** pair — not the monochrome tracking cams.
3. Treat them as a calibrated **stereo** pair (baseline + 180° right-ray flip).
4. Depth should be a **dense linear 2D depth image**, not sparse colored 3D dots.

---

## 2. Camera identity (essential correction)

| Sensor | Driver / nodes | Size / format | Role |
| :--- | :--- | :--- | :--- |
| **Sony IMX616 (stereo RGB)** | `arcimx616` on i2c (`0-0010`, `0-001a`) → `video0` / `video3` | **2464×2464 NV12**, Qualcomm UBWC, DMA-BUF size **9,564,160** | **This is what the runtime uses** |
| OmniVision OG01A1B | Tracking / SLAM path | **1056×1024 GREY** (not the 9.5 MB slots) | Do **not** use for this stereo depth path |

Earlier project docs assumed the 2464×2464 DMA-BUFs were OG01A1B monochrome. On this Frame build they are the **IMX616 color stereo** buffers. Stereo geometry still uses:

* Factory / CAD **baseline ≈ 63.96 mm**
* **180° right optical parity** (`r_right = (-rx, -ry, 1)`) in the Kannala–Brandt rectification LUT
* Lockstep L/R ring-buffer slot selection after Vulkan detile

---

## 3. Capture & detile

**Files:** `src/dma_capture.cpp`, `src/vk_detile.cpp`, `include/vk_detile.hpp`

* Discover eight `/dmabuf:` FDs of size 9,564,160 from `vrcompositor`; split into 4 right + 4 left slots.
* Import all slots as `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` with `DRM_FORMAT_MOD_QCOM_COMPRESSED`.
* Copy **plane 0 (Y)** and **plane 1 (UV / R8G8)** for every ring slot in one Vulkan command-buffer submission.
* Hash-pick the newest non-empty left/right frame only after that submission completes, then copy matching Y+UV snapshots to the worker. This prevents a slot from being recycled between separate luma and chroma reads.
* Convert the selected NV12 snapshots to packed RGBA in the worker. The live depth path performs full-extrinsic Kannala–Brandt rectification, computes 7×7 Census descriptors on Vulkan, searches 96 disparities bidirectionally, and publishes confidence-filtered metric depth.

Do not import a 9,564,160-byte slot as a 1232×1232 image. Inspection of `libArcturusPerception.so` shows that `CameraColorFrame::halfResolutionColorFrame()` allocates and caches a separate downsampled frame; it does not change the layout of the source IMX616 DMA allocation. Runtime probes also showed that this function is not called by `vrcompositor` during the gaze-triggered transition.

The gaze-dependent resolution policy is in `vrcompositor`: Frame's driver default sets `camera/autoLowerPassthroughResolution` to `1`, and the compositor contains a `PassthroughQualityManager` that applies it when another app or overlay has focus. The runtime sets this value to `0` through `IVRSettings` at startup, preserving the native camera path while the monitor has gaze focus.

`kernel.yama.ptrace_scope=0` remains required for `pidfd_getfd`.

---

## 4. Preview orientation and geometry

**Files:** `src/overlay_renderer.cpp`, `src/main.cpp`

* The overlay is 768×444 with left RGB, linear metric depth, and right RGB panels.
* The left preview is rotated 90° counter-clockwise and the right preview 90° clockwise to correct the opposite physical camera mounts.
* The center panel visualizes 0.25–5 m metric depth linearly (near white, far black). Values outside the display range are black.

---

## 5. Overlay upload path

`IVROverlay::SetOverlayRaw` was unreliable on Frame and repeatedly returned `VROverlayError_RequestFailed` (23). The monitor now uploads triple-buffered Vulkan RGBA textures with `SetOverlayTexture`. The render/event loop remains independent of camera ingestion and runs around the headset refresh rate; texture content is rebuilt at about 30 Hz.

---

## 6. Run / keep-alive on Frame

SteamVR is up; the binary can exit if the SSH session or a prior `nohup` dies. A simple watchdog is used:

```bash
ssh frame 'cat > /tmp/run_spatial.sh << "EOF"
#!/bin/bash
cd /home/steamos/frame-spatial-runtime
while true; do
  echo "=== start $(date -Is) ==="
  ./bin/frame_spatial_runtime
  echo "=== exit $? $(date -Is) ==="
  sleep 2
done
EOF
chmod +x /tmp/run_spatial.sh
echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0
: > /tmp/spatial.log
nohup /tmp/run_spatial.sh >> /tmp/spatial.log 2>&1 &
'
```

Check health:

```bash
ssh frame 'pgrep -a frame_spatial; tail -n 20 /tmp/spatial.log'
```

Healthy log lines include:

* `[OverlayRenderer] Vulkan IMX stereo HUD 768x444`
* `[OverlayRenderer] texture ok seq=…`
* `[Render] … LIVE METRIC DEPTH` with a climbing `Frame:` counter
* `[DepthPipeline] metric depth coverage=… range=…m`

---

## 7. Files touched (summary)

| Path | Change |
| :--- | :--- |
| `include/spatial_types.hpp` | RGB `previewL/R` as `uint32_t` |
| `include/vk_detile.hpp` / `src/vk_detile.cpp` | UV plane detile buffers |
| `include/dma_capture.hpp` / `src/dma_capture.cpp` | `getLeftUV` / `getRightUV` |
| `src/main.cpp` | NV12→RGBA preview, depth publication, opposite 90° rotations, full-resolution compositor policy |
| `src/depth_pipeline.cpp` | Calibrated rectification + 7×7 Census/WTA metric depth |
| `include/overlay_renderer.hpp` / `src/overlay_renderer.cpp` | 768×444 Vulkan texture overlay |
| `docs/headset_rendering.md` | Overlay properties updated for IMX monitor |
| `docs/progress_imx_monitor.md` | This document |

---

## 8. Known remaining issues

1. **Producer synchronization:** Vulkan external-memory ownership and a single Y+UV snapshot submission are now used; the full-resolution lock still needs an in-headset longevity check.
2. **Learned depth / HTP:** QAIRT 2.50 CPU inference works but is not real-time; generic HTP execution is blocked by Frame's private DSP manager and missing firmware-matched FastRPC user-PD shell. See `qnn_htp_handoff.md`.
3. **Calibration:** Intrinsics in `calibration.hpp` are still the factory KB set used for the stereo pair; Valve’s `colorPassthroughCalibrationHistory` stereo extrinsics are not yet ingested.

---

## 9. What you should see in the headset

* Head-locked card with left RGB, center metric depth, and right RGB panels.
* FPS / `F####` counting and an orange activity bar.
* Left and right color previews upright after opposite 90° rotations.
