# Building and Deployment Guide

## 1. Prerequisites & Environment

The Steam Frame runs SteamOS (Arch Linux ARM64 / `aarch64`) with an 8-core AMD Zen 4 CPU and Qualcomm Adreno GPU.

### Development Machine Requirements
* macOS or Linux with `ssh`, `scp`, and `rsync`.
* SSH access configured to `ssh frame` (`steamos@frame`).
* Sudo password on Frame: `frame1`.

### Dependencies on Steam Frame
* GCC / G++ with C++17 support and OpenMP (`g++ -fopenmp`).
* OpenVR Headers & Runtime: Located at `/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers` and `/opt/steamvr/bin/linuxarm64/libopenvr_api.so`.
* `glslc` for the native stereo compute shaders.
* MNN is optional and needed only for the historical `vulkan-ai` experiment.

---

## 2. One-Time Frame Configuration

To enable the runtime to duplicate camera DMA-BUFs from `vrcompositor`, ptrace restrictions must be lifted:
```bash
./scripts/setup_frame.sh
```
Or manually via SSH:
```bash
ssh frame "echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0"
```

---

## 3. Automated One-Command Build & Run

From your local machine, execute:
```bash
./scripts/deploy_and_run.sh
```
This script will:
1. Rsync the repository to `frame:/home/steamos/frame-spatial-runtime/`.
2. Compile the native Vulkan shaders and C++ binary with `-O3 -fopenmp`.
3. Launch the binary in the headset with active telemetry logged to your terminal.

To prepare the optional learned-depth dependency, run:

```bash
./scripts/setup_vulkan_ai.sh
```

---

## 4. Manual Compilation on Frame

If you prefer working directly inside an SSH terminal on the Frame:
```bash
ssh frame
cd /home/steamos/frame-spatial-runtime
make clean
make -j4
./bin/frame_spatial_runtime
```

### Command-Line Arguments
By default, the runtime automatically discovers the active camera file descriptors. You can optionally specify manual DMA-BUF file descriptors if testing specific slots:
```bash
./bin/frame_spatial_runtime <right_camera_fd> <left_camera_fd>
# Example:
./bin/frame_spatial_runtime 228 243
```

---

## 5. Troubleshooting & Diagnostics

### Problem: `pidfd_getfd failed: Operation not permitted`
* **Cause**: `kernel.yama.ptrace_scope` is set to `1` (usually after a reboot).
* **Fix**: Run `echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0`.

### Problem: `Failed to find enough 9564160-byte dmabufs`
* **Cause**: `vrcompositor` is not currently running or SteamVR is sleeping.
* **Fix**: Wake up the headset or restart SteamVR. Check running processes with `pgrep vrcompositor`.

### Problem: Overlay frozen (FPS text not changing) but logs show LIVE
* **Cause**: `SetOverlayRaw` returns `VROverlayError_RequestFailed` (23) for large/fast uploads; headset keeps last good texture.
* **Fix**: Keep HUD at 256×96 and ~10 Hz Raw (current code). Prefer `SetOverlayTexture` for larger content. See [`progress_imx_monitor.md`](progress_imx_monitor.md).

### Problem: Process not running / overlay gone
* **Cause**: Binary exited (SSH hangup, SteamVR restart, etc.).
* **Fix**:
```bash
ssh frame 'echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0
nohup bash -c "cd /home/steamos/frame-spatial-runtime && while true; do ./bin/frame_spatial_runtime; sleep 2; done" >> /tmp/spatial.log 2>&1 &
pgrep -a frame_spatial; tail -n 20 /tmp/spatial.log'
```

### Problem: `VR_Init failed`
* **Cause**: SteamVR compositor is not fully initialized.
* **Fix**: Put on the headset or place a piece of tape over the proximity sensor between the lenses to keep the displays and tracking active.
