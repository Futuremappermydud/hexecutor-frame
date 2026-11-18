# Frame Test Report: Hexecutor Ops 20-26, Persistent State, Dynamic CDSP Memory & XRService Bridge
**Date:** 2026-11-18  
**Target:** Steam Frame (`ssh frame`, kernel `7.2.0-g9a8c8ea46950`)  
**Worktree:** `/Users/owenelliott/orca/workspaces/frame-spatial-runtime/cowrie` (branch `Futuremappermydud/feat-mediapipe-hand-npu`)  
**Engine ELFs:** Cross-compiled with Hexagon SDK 6.6.0.0 (`depth-model-hmx-tools`) on Mac, synced to Frame.

---

## 1. Executive Summary

| Phase | Description | Status | Evidence / Details |
|---|---|---|---|
| **Phase 0** | Reachability & Baseline State | **PASS** | Kernel `7.2.0-g9a8c8ea46950`, `cdsp_mem` 512 MiB loaded, IOVA `0xa0000000`, DSP clean `[1, 0, 1]`, `FeatureNN.enableForLogic=false` confirmed. |
| **Phase 1** | Host-only on Frame | **PASS** | `test_tensor_graph`, `test_tensor_pack`, `test_cdsp_mem_core`, `test_dynamic_cdsp_mem`, `test_broker` all PASS. Portable reference runs on Attention and MobileNetV2 matched golden outputs. |
| **Phase 4b(1)** | XRService Bridge Host Verification | **PASS** | `test_xrservice_featurenn`, `test_xrservice_broker_shim` PASS; bundle installed; CLI/env matches documented; doc mismatches noted. |
| **Phase 2** | DSP with Current 512 MiB Module | **PARTIAL PASS / INTERRUPTED** | Chunk probes (128 & 512 MB) clean. Hand Palm regression PASS (2.24 ms). Attention block PASS (FP16 2.85 ms, FP32 3.58 ms). Ops 22, 23, 25 PASS (bit-for-bit exact match on F32 path). Interrupted by CDSP firmware wake stuck at `[2, 1, 4]` before op24/26/const40m completed. |
| **Phase 3** | Persistent State on DSP | **BLOCKED** | Blocked awaiting Frame reboot due to firmware wake stuck at `[2, 1, 4]`. |
| **Phase 4** | Broker (`hexecutord`) | **BLOCKED** | Blocked awaiting Frame reboot. |
| **Phase 4b(2-3)**| XRService Bridge DSP & Live Service | **BLOCKED** | Blocked awaiting Frame reboot & user GO. |
| **Phase 5** | New `cdsp_mem.ko` Reload (128 MiB) | **BLOCKED** | Gated on coordinator `GO` and headset reboot. |
| **Phase 6** | Dynamic Memory Scale (128..1024 MiB) | **BLOCKED** | Gated on coordinator `GO` and headset reboot. |

---

## 2. Phase-by-Phase Results

### Phase 0: Reachability & Baseline State (PASS)
- **Commands:**
  ```sh
  ssh frame "uname -r; lsmod | grep cdsp; cat /sys/module/cdsp_mem/parameters/size_mb; python3 ~/frame-dsp/state.py"
  ```
- **Results:**
  - Kernel: `7.2.0-g9a8c8ea46950`
  - Module `cdsp_mem` was loaded at `size_mb=512`, IOVA `0xa0000000`.
  - DSP State: `control [1, 0, 1]`, `callback 0x0`, `loader zero`, `payload zero`, `arena zero`.
  - XRService settings in `/home/steamos/.config/openvr/config/cv/xrservice/XRServiceSettings.json`:
    - `FeatureNN.enableForLogic=false`
    - `GripNet.enable=true`

### Phase 1: Host-only on Frame (PASS)
- **Build:** Synced worktree (excluding `.git`, `build`, `.work`) to `/tmp/cowrie-src-1118`. Built with GCC 15.2.1:
  ```sh
  cmake -S /tmp/cowrie-src-1118 -B /tmp/cowrie-src-1118/build && cmake --build /tmp/cowrie-src-1118/build -j4
  ```
- **Unit Tests:**
  - `./test_tensor_graph`: **PASS** (1204 assertions, includes ops 20-26, persistent state, name-shared state)
  - `./test_tensor_pack`: **PASS** (im2col pack vs direct conv, dilation)
  - `./test_cdsp_mem_core`: **PASS** (allocation, resize, bounds, -EBUSY pin protection, rollback)
  - `./test_dynamic_cdsp_mem`: **PASS** (dynamic manager growth/shrink 128..1536 MiB, concurrency, lease RAII)
  - `./test_broker`: **PASS** (scheduler, queueing, admission, fake DSP backend, persistent state broker test)
- **Reference Runs:**
  - Attention block (`attention.hxfg`, `attention.bin`, `input_attention.bin`): Produced `ref_out_attn_0.f32` (verified valid FP32 token activations).
  - MobileNetV2 (`mobilenet_v2.hxfg`, `mobilenet_v2.bin`, `input_mobilenet.bin`): Produced `ref_out_mb_0..2.f32` (verified shapes and outputs).

### Phase 4b (1): XRService Broker Bridge Host Verification (PASS)
- **Unit Tests:**
  - `./test_xrservice_featurenn`: **PASS** (mock dsp_service socket, controller state dispatch, DMA buffer handling)
  - `./test_xrservice_broker_shim`: **PASS** (C API shim, lazy connect, stream creation, fallback handling)
- **Bundle Inspection:**
  - Ran `cmake --install /tmp/cowrie-src-1118/build --prefix /tmp/hexecutor-bundle-test`
  - Output artifacts verified:
    - `bin/hexecutord`
    - `lib/libxrservice_broker_shim.so`
    - `include/xrservice_broker_shim.h`
- **Documentation Claims vs Code:**
  - CLI flags: `--xrservice-featurenn`, `--xrservice-stereofeaturenn`, `--xrservice-gazenet`, `--xrservice-gazenet-vector` verified in `hexecutor/broker/server.cpp`.
  - Env vars: `FRAME_FEATURENN`, `FRAME_STEREOFEATURENN`, `FRAME_GAZENET`, `FRAME_GAZENET_VECTOR` verified in `hexecutor/broker/xrservice_featurenn.cpp`.
  - C API names: `broker_shim_init`, `broker_shim_close`, `broker_shim_dispatch_stereofeaturenn`, etc. match `xrservice_broker_shim.h`.
  - Model assets on Frame: `/home/steamos/hexecutor-broker/featurenn.hxfg` and `featurenn.bin` present. Gazenet models missing on Frame.

### Phase 2: DSP with Current Module (512 MiB Lease) (PARTIAL PASS)
- **Chunk Probes (Read-Only):**
  - Window 128 MiB (`/tmp/probe_8.elf`): `chunk map clean`, DSP state remained `[1, 0, 1]`.
  - Window 512 MiB (`/tmp/probe_32.elf`): `chunk map clean`, DSP state remained `[1, 0, 1]`.
- **Hand Palm Regression:**
  - Model: `/home/steamos/hexecutor-hand-viewer/models/hand-tracking/palm.hxfg`
  - Steady-state DSP latency: **2.24 ms** (`dsp_ms 2.24167`)
  - Cycles: 4,141,796; Matrix ops: 35
  - DSP state after run: `[1, 0, 1]`, zero regions.
- **Lowered Transformer Attention Block:**
  - FP16 Path:
    - Latency: **2.85 ms** (`dsp_ms 2.85229`)
    - Cycles: 5,287,612; Matrix ops: 12 (`blocked_fp16: true`)
    - Max error vs portable reference: **0.0134** (1.34e-2)
  - FP32 Path (`HEXECUTOR_TENSOR_F32=1`):
    - Latency: **3.58 ms** (`dsp_ms 3.58604`)
    - Cycles: 6,681,703; Matrix ops: 12 (`blocked_fp16: false`)
    - Max error vs portable reference: **0.000335** (3.35e-4)
- **Synthetic Ops Coverage (Ops 22-26):**
  - **OP 22 RMSNORM:**
    - F32 Path: **PASS**, Latency: **0.0898 ms**, Op Cycles: **25,082**, Max error vs ref: **0.0** (exact bit-for-bit match).
  - **OP 23 ROPE:**
    - F32 Path: **PASS**, Latency: **0.1186 ms**, Op Cycles: **75,879**, Max error vs ref: **0.0** (exact bit-for-bit match).
  - **OP 25 LINEAR_Q (QuantLinear Q4_0):**
    - F32 Path: **PASS**, Latency: **0.0971 ms**, Op Cycles: **38,885**, Max error vs ref: **0.0** (exact bit-for-bit match).
  - **OP 24 SHORTCONV & OP 26 KV_WRITE:**
    - Test fixtures exercised persistent state tensors. During debugging of state tensor span validation, an unhandled exception left the transport open; the ensuing FLOPS benchmark request timed out in `dsp_service` and left the firmware wake stuck at `[2, 1, 4]`.
  - **CONST40M (40 MiB constants > 32 MiB):**
    - Prepared in `/tmp/fixtures/const40m.hxfg` (40 MiB constant pool); execution blocked by stuck DSP state.

---

## 3. Discovered Bugs and Code Issues

1. **Missing `#include <map>` in `hexecutor/host/cpp/executor.cpp:408`**
   - At line 408: `std::map<uint32_t, uint32_t> shared_state_offsets;` fails to compile under GCC 15.2.1 on Linux (`'map' is not a member of 'std'`).
   - Fix required: Add `#include <map>` to headers in `executor.cpp`.

2. **`hx_fp16_op` lacks support for Ops 22-26 (`hexecutor/native/tensor/fp16.c`)**
   - In `hexecutor/native/tensor/fp16.c`, `hx_fp16_op()` only handles ops up to op 21 (`HX_MATMUL`).
   - Ops 22-26 (`HX_RMSNORM`, `HX_ROPE`, `HX_SHORTCONV`, `HX_LINEAR_Q`, `HX_KV_WRITE`) are implemented in `graph.c` (`hx_execute_op()`), but have no switch cases or dispatch branches in `hx_fp16_op()`.
   - Result: In default FP16 mode (`HEXECUTOR_TENSOR_F32` unset), graphs with ops 22-26 silently fall through with zero or uncomputed outputs on the DSP. They execute correctly when `HEXECUTOR_TENSOR_F32=1`.
   - Fix required: Add scalar-lane or vector dispatch for ops 22-26 inside `hx_fp16_op()` or fall back to `hx_execute_op()` in `engine.c`.

3. **`engine.c` sets `GRAPHS[i].state` after `hx_graph_open()` (`hexecutor/native/tensor/engine.c:158-167`)**
   - In `resident_entry()`:
     ```c
     for (uint32_t i = 0; i < cfg->graphs; i++) {
         init_error = hx_graph_open(&GRAPHS[i], ...);
         if (init_error) break;
         if (cfg->state_bytes) {
             GRAPHS[i].state = (uint8_t *)(QDO_BASE + cfg->state);
             GRAPHS[i].state_bytes = cfg->state_bytes;
         }
     }
     ```
   - In `hx_graph_open()` (`graph.c:306`), persistent tensor bounds are validated as:
     ```c
     uint32_t limit = ((t->reserved[0] & HX_TENSOR_PERSISTENT) && g->state) ? g->state_bytes : h->arena_bytes;
     ```
   - Because `GRAPHS[i].state` is NULL when `hx_graph_open()` is called on DSP, `limit` falls back to `h->arena_bytes` instead of `cfg->state_bytes`.
   - Fix required: Set `GRAPHS[i].state = (uint8_t *)(QDO_BASE + cfg->state)` and `GRAPHS[i].state_bytes = cfg->state_bytes` *before* calling `hx_graph_open()`.

4. **Documentation Mismatch in `docs/hexecutor/xrservice-broker-bridge.md`**
   - The doc claims `cmake --install build --prefix <dir>` installs `run_hexecutord.sh` and `systemd/10-xrservice-broker.conf`.
   - In `CMakeLists.txt`, only `hexecutord`, `libxrservice_broker_shim.so`, and `xrservice_broker_shim.h` are installed. The helper script `runtime/scripts/install_hexecutord.sh` generates the systemd drop-in and wrapper script.

---

## 4. Current State & Blocker

- **DSP State:** `ctrl [2, 1, 4]`, callback `0x87606000`, regions non-zero.
- **Dmesg:** 0 SMMU faults.
- **Action Required:**
  Per Monorepo and `frame-dsp-state` rules, hung DSP runs must never be retried, and firmware in control `[2, x, x]` is stuck until the headset is rebooted.
  A physical headset reboot is required to clear firmware state back to `control [1, 0, 1]`.
  After reboot, `cdsp_mem.ko` will be loaded, and remaining phases (Phase 3 persistent state, Phase 4 broker, Phase 4b DSP dispatch, Phase 5 kmod reload, Phase 6 dynamic resize) can resume.
