# Runtime contributor guidance

Read [runtime documentation](../docs/runtime/README.md) and the relevant component guide before changing behavior. Run application build/deploy commands from `depth-runtime/`. The target is ARM64 Linux on the Frame; host checks are not a full device build.

- Keep capture/depth/fusion off the render loop. Continue pose location and frame submission while inference runs.
- Preserve camera ownership: discover and duplicate compositor DMA-BUFs; do not open claimed physical `/dev/video*` sensors or hardcode descriptors. The 9,564,160-byte slots belong to the IMX616 color pair, not the separate tracking/IR sensors.
- Preserve zero-copy raw-buffer ownership and calibrated geometry. The right color sensor has 180-degree parity inversion; camera coordinates and presentation coordinates differ. Record calibration selection and capture-time pose.
- Keep the optimized/OpenMP build flags where required by the CPU path. Do not infer GPU/DSP performance from a host timing result.
- Learned-depth graph execution uses native prepared backends. The native C++ worker is the host IPC/session layer; do not restore the removed hybrid scheduler or an implicit fallback.
- Treat VTCM/L2 mappings, callbacks, clocks, eye-client state, and failure cleanup as the executor's ownership contract. Read the executor operations/memory guides before changing device interaction.
- Update the canonical `docs/` guide when changing a public flag, model contract, camera transform, publication rule, or presentation path. Keep the temporal-fusion working report/notes untouched.
