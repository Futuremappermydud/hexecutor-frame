# Bridging XRService StereoFeatureNN and Gazenet to Hexecutor Broker

This guide describes how XRService's controller tracking (**StereoFeatureNN**) and eye tracking (**Gazenet**) are bridged into our CDSP broker daemon (`hexecutord`), eliminating firmware contention, avoiding VTCM memory hazards, and allowing XRService to be patched transparently.

---

## 1. Background & Motivation

On the Steam Frame headset (Qualcomm SM8650), the CDSP (Compute DSP) is shared across multiple concurrent perception workloads:
- Frame learned depth engine (QDRX prepared executor, ~9.7 ms per 30 Hz frame)
- MediaPipe / Mercury hand tracking (palm detection + landmark tracking, 30–60 Hz)
- XRService controller tracking (**StereoFeatureNN**, 60 Hz per controller)
- Stock eye tracking (**Gazenet** image evaluation and vector evaluation)

### Historical Contention Hazards
Previously, custom inferences held separate exclusive `hexecutor::Session` instances while XRService dispatched its own tasks directly to the stock `dsp_service` daemon:
1. **Firmware Stalling at `(2,0,8)`**: Waking or exiting the CDSP while an uncoordinated stock job was in-flight caused firmware completion queues to desynchronize, leaving the DSP stuck at control state `(2,0,8)` and requiring a full device reboot.
2. **Firmware Thread-3 VTCM Hazard**: Disassembly showed that firmware hardware thread 3 independently polls DMA mailbox `0x1522000` and uses VTCM `0x31700000–~0x317d4600` for eye-tracking weights and scratch tables. When our session held VTCM pool 1, concurrent stock requests would corrupt both activations and model weights.

### The Broker Architecture
The CDSP broker (`hexecutord`, C++17) acts as the **single process owning the CDSP lease**. All workloads submit jobs through local Unix domain socket IPC (`hexecutor::broker::Client`).

```
                    ┌──────────────────────────┐
                    │     XRService / OpenVR   │
                    │  (Controller & Eye Track)│
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ (Option A: Direct Patch/Shim) │ (Option B: Passive DMA Mailbox)
                 │  libxrservice_broker_shim.so  │  xrservice::Adapter in hexecutord
                 │  (LD_PRELOAD / C ABI calls)   │  (maps dsp_service DMA buffer)
                 └───────────────┬───────────────┘
                                 │
                   Unix Domain Socket IPC (/tmp/hexecutord.sock)
                                 │
 ┌───────────────────────────────▼──────────────────────────────────────────────┐
 │                              hexecutord                                      │
 │  ┌────────────────────────────────────────────────────────────────────────┐  │
 │  │ Priority Scheduler (Class::tracking, EDF, co-scheduled tensor graphs)  │  │
 │  └────────────────────────────────────┬───────────────────────────────────┘  │
 │                                       │                                      │
 │  ┌────────────────────────────────────▼───────────────────────────────────┐  │
 │  │                     One Persistent hexecutor::Session                  │  │
 │  │        (Shared VTCM resident records, unified FP16 tensor engine)      │  │
 │  └────────────────────────────────────┬───────────────────────────────────┘  │
 └───────────────────────────────────────┼──────────────────────────────────────┘
                                         ▼
                                Qualcomm SM8650 CDSP
```

---

## 2. Bridging Mechanisms

Two complementary bridging mechanisms are supported:

### A. Direct Patch / Preload Shim (`libxrservice_broker_shim.so`)
The cleanest, production-ready path. Rather than having XRService touch the stock CDSP DMA buffer at all, XRService calls the broker directly over the Unix domain socket:
- Can be activated via `LD_PRELOAD` without modifying XRService binaries on disk.
- Or can be called directly by patched XRService code via standard C API functions in `xrservice_broker_shim.h`.
- **Completely bypasses firmware thread 3 and DMA mailboxes**, eliminating the VTCM hazard and yield stalls.

### B. In-Daemon Mailbox Adapter (`xrservice::Adapter`)
Built into `hexecutord`, this adapter connects to `dsp_service`, maps the stock DMA buffer (`0x2000000` bytes), and watches the mailbox words at `0x1522000` (StereoFeatureNN) and `0x445000` (Gazenet). When active jobs arrive, the adapter translates buffers, runs inference through `broker::Client`, writes results back, and signals completion.

---

## 3. Workload Contracts & Data Formats

### 1. StereoFeatureNN (Controller Keypoint Network)
- **Model**: Lowered MobileNetV2 with split Gemm head (68 ops, FP16 resident weights).
- **Input**:
  - Two $128 \times 128$ u8 eye planes per controller (`views = 2` for 1 controller, `views = 4` for 2 controllers).
  - Normalization: $x_{\text{norm}} = (x / 255.0 - 0.449) / 0.226$.
  - Packed into compact 3-channel NHWC float format ($128 \times 128 \times 3 \times 4 = 196,608$ bytes per controller) with channel 2 padded with $0.0$.
- **Graph Output**:
  - 552 output floats (split across 352 and 200 channels on the lowered model; the shim/adapter joins them).
- **Output Record Formatting**:
  - Reshaped into $[2, 92, 3]$ per controller: 4 heads of 23 floats per eye (center, uncertainty, plus-x, plus-z).
  - Uncertainty head applies $2.0 \cdot \min(\text{raw}, 10.0)$ (XRService applies final exponential).
  - Written into standard $0\times450$-byte ($1,104$ bytes) record per eye ($2,208$ bytes per controller).

### 2. Gazenet Image Evaluation
- **Input**:
  - Two planar $128 \times 128$ float images (left + right eye).
  - Converted to compact 3-channel NHWC with channel 2 zero-padded ($196,608$ bytes).
- **Output**:
  - Exactly 64 output floats copied into result buffer.

### 3. Gazenet Vector Evaluation
- **Input**:
  - Float vector of up to 4,800 floats (typically exactly 4,800 floats, $19,200$ bytes).
- **Output**:
  - In-place float vector with updated float count.

---

## 4. Components & Implementations

### C API & Shim Library
- **Header**: [xrservice_broker_shim.h](file:///Users/owenelliott/orca/workspaces/frame-spatial-runtime/cowrie/hexecutor/include/hexecutor/xrservice_broker_shim.h)
- **Source**: [xrservice_broker_shim.cpp](file:///Users/owenelliott/orca/workspaces/frame-spatial-runtime/cowrie/hexecutor/host/cpp/xrservice_broker_shim.cpp)
- **C Functions Exported**:
  ```c
  void broker_shim_set_socket(const char *socket_path);
  int broker_shim_is_connected(void);
  int broker_shim_dispatch_stereofeaturenn(const void *input_u8, void *output_records,
                                          uint32_t views, uint32_t width, uint32_t height);
  int broker_shim_dispatch_gazenet(const float *input_planar_f32, float *output_64_f32);
  int broker_shim_dispatch_gazenet_vector(const float *input_f32, uint32_t in_floats,
                                         float *output_f32, uint32_t max_out_floats,
                                         uint32_t *out_floats);
  int dispatchStereoFeatureNNOnCDSP(const void *input_u8, void *output_records,
                                    uint32_t views, uint32_t width, uint32_t height);
  ```

### Daemon CLI & Launcher Aliases
- **`hexecutord` CLI**:
  - `--xrservice-featurenn GRAPH` or `--xrservice-stereofeaturenn GRAPH` (alias)
  - `--xrservice-gazenet GRAPH`
  - `--xrservice-gazenet-vector GRAPH`
- **`run_hexecutord.sh`**:
  - `FRAME_STEREOFEATURENN` or `FRAME_FEATURENN` environment variables.
  - `FRAME_GAZENET` and `FRAME_GAZENET_VECTOR` environment variables.

### Packaging & Bundling Tool
- **Script**: [install_hexecutord.sh](file:///Users/owenelliott/orca/workspaces/frame-spatial-runtime/cowrie/runtime/scripts/install_hexecutord.sh)
- **CMake Install Rules**: `cmake --install build --prefix <dir>` installs:
  - `<prefix>/bin/hexecutord`
  - `<prefix>/bin/run_hexecutord.sh`
  - `<prefix>/lib/libxrservice_broker_shim.so` (or `.dylib`)
  - `<prefix>/include/xrservice_broker_shim.h`
  - `<prefix>/systemd/10-xrservice-broker.conf` (drop-in configuration)

---

## 5. Deployment & Usage Instructions

### Step 1: Start `hexecutord` with Registered Graphs
```bash
# Register hand models + StereoFeatureNN + Gazenet
FRAME_STEREOFEATURENN=/home/steamos/hexecutor-broker/models/featurenn \
FRAME_GAZENET=/home/steamos/hexecutor-broker/models/gazenet \
    /home/steamos/hexecutor-broker/bin/run_hexecutord.sh --long-wake 30
```

### Step 2: Patch XRService via Systemd `LD_PRELOAD` Drop-in
No binary modification to XRService is needed:
```bash
# 1. Install bundle to headset
runtime/scripts/install_hexecutord.sh /home/steamos/hexecutor-broker

# 2. Apply systemd drop-in override
sudo mkdir -p /etc/systemd/system/xrservice.service.d
sudo cp /home/steamos/hexecutor-broker/systemd/10-xrservice-broker.conf /etc/systemd/system/xrservice.service.d/

# 3. Reload systemd and restart XRService
sudo systemctl daemon-reload
sudo systemctl restart xrservice
```

### Step 3: Verify Operation
Check XRService and hexecutord logs:
```bash
journalctl -u xrservice -f
journalctl -b | grep hexecutord
```
Stereo FeatureNN and Gazenet calls will dispatch through `hexecutord` with ~1.2 ms inference latency and zero firmware desynchronization.

---

## 6. Test Suite & Verification

The implementation is verified by two standalone test suites in CTest:

1. **`test_xrservice_featurenn` ([test_xrservice_featurenn.cpp](file:///Users/owenelliott/orca/workspaces/frame-spatial-runtime/cowrie/hexecutor/tests/test_xrservice_featurenn.cpp))**:
   - Tests raw ABI conversions (input normalization, head decoding, uncertainty clamping, vector handling).
   - Tests end-to-end `xrservice::Adapter` against real Unix domain socket with mock broker server and simulated 32 MiB DMA memory.
2. **`test_xrservice_broker_shim` ([test_xrservice_broker_shim.cpp](file:///Users/owenelliott/orca/workspaces/frame-spatial-runtime/cowrie/hexecutor/tests/test_xrservice_broker_shim.cpp))**:
   - Tests the C ABI functions and `LD_PRELOAD` symbol interception against a real Unix domain socket broker server.
   - Verifies 1-controller (2 views) and 2-controller (4 views) StereoFeatureNN dispatch, Gazenet image eval, and Gazenet vector eval.

Run all tests via CTest:
```bash
cmake --build build --target test_xrservice_featurenn test_xrservice_broker_shim
ctest --test-dir build --output-on-failure
```
Result: 19/19 tests passed.
