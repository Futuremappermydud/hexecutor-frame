# Zero-Copy Hardware DMA Ingestion

## 1. The Access Problem

On the Steam Frame, the camera hardware drivers (`/dev/video*`) are exclusively claimed by `XRService` and `vrcompositor`. Any secondary process attempting a standard `open("/dev/video0", O_RDWR)` receives `EBUSY` (Device or resource busy).

Furthermore, high-level APIs like `IVRTrackedCamera` are stubbed or disabled on the Frame's Linux ARM64 build.

---

## 2. Linux `pidfd_open` and `pidfd_getfd` Architecture

Linux kernel 5.6+ introduces process file descriptor replication via:
1. `pidfd_open(pid, 0)`: Obtains a file descriptor referring to target process `pid`.
2. `pidfd_getfd(pfd, target_fd, 0)`: Duplicates file descriptor `target_fd` from the target process into the calling process.

### Permission Requirement
Under default Linux security policies, `ptrace` restrictions prevent non-parent processes from cloning file descriptors. To enable replication:
```bash
echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0
```
This is automated via [`scripts/setup_frame.sh`](../scripts/setup_frame.sh).

---

## 3. Auto-Discovery of Camera DMA-BUFs

When `vrcompositor` initializes, it allocates 8 rotating DMA-BUFs for the dual camera pipelines:
* **Buffer Size**: Exactly `9,564,160` bytes.
  * Raw NV12 would be $2464 \times 2464 \times 1.5$, but the compositor allocates **Qualcomm UBWC-compressed** (`DRM_FORMAT_MOD_QCOM_COMPRESSED`, modifier `0x500000000000001`) 2-plane NV12. Linear `mmap` of that buffer is compressed tile payload and looks like noise.
  * The runtime therefore imports each DMA-BUF into Vulkan with that modifier and snapshots **plane 0 (Y)** plus **plane 1 (UV)** for every slot in one GPU submission. It selects the newest slot only after completion, keeping luma and chroma from the same producer state.
  * These 9.5 MB slots are the **IMX616 (`arcimx616`) stereo RGB** cameras, not the smaller OG01A1B GREY tracking sensors.
* **Target Link**: In `/proc/<pid>/fd/`, these appear as symlinks pointing to `/dmabuf:`.

### Discovery Algorithm
Implemented in [`src/dma_capture.cpp`](../src/dma_capture.cpp):
1. Locate `vrcompositor` PID using `pgrep`.
2. Scan `/proc/<pid>/fd/` for entries where `readlink()` contains `"dmabuf"`.
3. Check `stat().st_size == 9564160`.
4. Sort matching file descriptors numerically:
   * First cluster of 4 FDs: Rotating double/quad buffers for **Camera A (Right Camera)**.
   * Second cluster of 4 FDs: Rotating double/quad buffers for **Camera B (Left Camera)**.
6. Clone the active slot for each camera using `pidfd_getfd`.
7. Import the dma-buf into Vulkan (`VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` + QCOM compressed modifier) and copy decompressed Y+UV planes to CPU-visible staging memory in one command buffer.

```cpp
// Example: Duplicating DMA-BUF from vrcompositor
int pfd = syscall(SYS_pidfd_open, compositorPid, 0);
int localFd = syscall(SYS_pidfd_getfd, pfd, remoteFd, 0);
uint8_t *ptr = (uint8_t*)mmap(NULL, 9564160, PROT_READ, MAP_SHARED, localFd, 0);
```

### Advantages of Zero-Copy DMA
* **Zero IPC Serialization**: No socket overhead, no pipe copying, and no shared memory serialization.
* **Instant Frame Ingestion**: The CPU reads directly from the physical memory page mapped by the camera driver at bus bandwidth speeds (>10 GB/s).
* **Self-Healing**: If SteamVR restarts and allocates new file descriptors, the runtime automatically re-scans and binds to the newly created buffers without crashing.
