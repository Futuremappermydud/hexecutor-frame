# Current roadmap and next steps

Updated 2026-10-07. Detailed plans:

- [Guarded GPU matcher optimization](stereo_guarded_optimization_plan.md)
- [Persistent room mesh and mesh-based occlusion](room_mesh_plan.md)

## Implemented and observed

The runtime auto-discovers the IMX616 compositor DMA-BUF ring, detiles luma,
rectifies with current calibration, and computes bidirectional Vulkan depth.
The baseline matcher and experimental guarded g3/e1 matcher are separate paths.
The guarded port passed fourteen native CPU/GPU parity cases, but the underlying
algorithm still fails the complete offline quality promotion gates.

The OpenXR point-cloud viewer already uses capture-time pose association,
calibrated camera-to-head transforms, stage-space voxel fusion and independent
observation confirmation. World registration is implemented; exact sensor
exposure timestamps remain unavailable.

The panel occlusion wrapper now selects guarded GPU matching by default, with
`FRAME_OCCLUSION_STEREO=baseline` available for comparison. Eight-row compute
batches let rendering run at approximately 72 FPS in short live observations;
fresh guarded depth still takes roughly 0.8 seconds per pair. Current occlusion
uses temporally confirmed camera depth surfaces and plane completion. It is
not a persistent volumetric room mesh.

Brief empty alpha submissions retain capture/history for up to 250 ms, and
temporary capture failures retain published surfaces to their existing expiry.
Guarded legacy history has a bounded 2.5-second age budget; that supports slower
updates but can leave moving-object masks. See [operation and limitations](openxr_occlusion.md).

## Next implementation sequence

1. Measure GPU pass/batch durations, full publication age and render tails.
   Benchmark manifests/replay must record the guarded matcher and its age policy.
2. Prototype exact per-view support-arm limits and winner-neighbor cost caching.
   Preserve output parity before tuning submission size and host yields.
3. Define fresh depth observations with measured provenance, capture poses,
   calibration and tracking-space epochs; add deterministic map replay.
4. Build measured-only sparse CPU TSDF fusion and incremental world-mesh preview.
   Keep geometry across views; confirm it before allowing occlusion.
5. Add local contradiction/removal logic, support deadlines and map-epoch handling.
6. Connect immutable confirmed mesh snapshots to the panel occlusion renderer.
   Validate still/head-turn/revisit/moving-object behavior and sustained cadence.

The mesh plan and matcher instrumentation can proceed together. Quality-changing
search shortcuts and mesh-guided matching remain separate experiments to avoid
reinforcing incorrect geometry.

## Follow-on work

GPU rectification, shared-memory matching, indexed mesh buffers and GPU fusion
follow measurements rather than fixed speed claims. Accurate exposure timing,
gaze-mode camera metadata, room export, physics queries and persistent storage
remain additional projects. Hand occlusion needs lower fresh-depth latency;
increasing map persistence alone does not provide it.

The plans define proposed budgets and acceptance gates, not achieved timings
or implemented meshing APIs. Preserve automatic DMA discovery, right-camera
parity, direct mapped-buffer access, `-O3 -fopenmp`, and separate depth/mesh
workers from headset rendering throughout.
