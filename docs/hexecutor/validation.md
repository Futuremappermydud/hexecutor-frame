# Hexecutor validation and device evidence

The portable CTest suite checks graph validation and arithmetic, the broker socket and scheduler, the XRService conversion and shim ABIs, and memory growth and rollback with injected backends. The Hexagon simulator exercises actual HVX/HMX instructions and compares selected kernels and whole graphs with portable or recorded oracles. Device tests additionally cover firmware ownership, actual clock and DMA behavior, and cleanup. Use the [development commands](../development.md#practical-local-checks) and [measurement guide](performance.md#measuring-changes).

| Check | Source |
| --- | --- |
| HXFG structure, allocation and numerical cases | [Graph tests](../../hexecutor/tests/test_tensor_graph.cpp), [tensor simulator](../../hexecutor/build-scripts/test_tensor_sim.sh) |
| RoPE, quantized matvec and staged decode | [LFM simulator](../../hexecutor/build-scripts/test_lfm_sim.sh) |
| Broker admission, expiry, replacement, persistent state and release | [Broker tests](../../hexecutor/tests/test_broker.cpp) |
| FeatureNN/Gazenet mailbox conversion and shim dispatch | [Mailbox test](../../hexecutor/tests/test_xrservice_featurenn.cpp), [shim test](../../hexecutor/tests/test_xrservice_broker_shim.cpp) |
| Allocation rollback and dynamic leases | [Kernel core test](../../hexecutor/tests/test_cdsp_mem_core.cpp), [host manager test](../../hexecutor/tests/test_dynamic_cdsp_mem.cpp), [device checklist](dynamic-memory.md#verification) |

## Recorded Frame results

- **2026-11-17.** CPU readback across the 512 MiB lease had zero mismatches. Read-only chunk-map probes covered 128, 256 and 512 MiB on the recorded firmware. This established visibility for those probe windows. [Memory validation](dynamic-memory.md#verification)
- **2026-11-18.** Portable tests on a Frame passed for graph arithmetic, the memory core, dynamic host sizing, the broker and XRService shim/adapter. On the CDSP the palm regression ran at 2.24 ms and the tested attention block took 2.85 ms FP16 and 3.58 ms FP32. Standalone RMSNorm, RoPE and quantized linear matched the reference bit for bit on the recorded F32 path. A failure during persistent-state testing left the firmware at `(2,1,4)` and ended that run before the remaining ops, large-constant fixture, module reload and live XRService checks completed. Treat that as a past incident with a specific artifact set.
- **2026-11-19.** Staged tensor decode exercised a 1,280 MiB lease with a matched 80-slot engine. This provides large-window evidence for that firmware and workload. Verify the configured maximum separately. [Dynamic memory checks](dynamic-memory.md#recorded-evidence-and-release-checks)

## Coverage to establish on Frame

- Confirm the exact kernel, firmware identity, window, model artifacts, client state and matching engine/bootstrap/probe before each run.
- Recheck operand layout and output agreement on representative captures. For broker workloads measure mixed jobs, queue deadlines, idle release/reset and device recovery. For stock XRService jobs verify real tracking quality and the firmware/VTCM handoff. [XRService bridge](xrservice-broker-bridge.md)
- Check state and ownership after normal close, signal exit and failure. For dynamic leases confirm rollback and shrink to the baseline on the target kernel. For stateful graphs check reset and loss/recovery separately from first-token accuracy. [Session recovery](operations.md#failure-handling)
