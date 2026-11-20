# Hexecutor documentation

## Start here

- [System architecture](architecture.md) maps the executor, Frame depth runtime and model samples.
- [Build and run](getting-started.md) covers host and headset entry points.
- [Development and validation](development.md) describes the portable checks and evidence required for changes.
- [Training and artifacts](model-development/training-and-artifacts.md) covers local model data, provenance and repository policy.

## Hexecutor

| Topic | Guides |
| --- | --- |
| Architecture and integration | [Executor architecture](hexecutor/README.md), [C++ API](hexecutor/api.md), [program formats](hexecutor/formats.md), [HXFG tensor backend](hexecutor/tensor-backend.md), [ExecuTorch delegate](../hexecutor/executorch/README.md) |
| Broker and deployment | [Broker lifecycle and scheduling](hexecutor/broker.md), [XRService integration](hexecutor/xrservice-broker-bridge.md), [broker installation](hexecutor/install.md), [Frame operations and recovery](hexecutor/operations.md) |
| Memory and kernels | [Private DDR, VTCM and L2](hexecutor/memory-layout.md), [dynamic leases](hexecutor/dynamic-memory.md), [kernels and numerics](hexecutor/kernels.md), [adapter acceptance](hexecutor/kernel-adapters.md), [run-list scheduling](hexecutor/scheduling.md) |
| Measurements | [Validation and device evidence](hexecutor/validation.md), [prepared-depth performance baselines](hexecutor/performance.md), [AXI/HMX clocks](hexecutor/clocks.md) |

## Depth runtime

| Topic | Guides |
| --- | --- |
| Capture and geometry | [Runtime overview](runtime/README.md), [camera ownership, calibration and timing](runtime/cameras.md), [stereo depth and confidence](runtime/stereo.md) |
| Inference and fusion | [Learned depth integration](runtime/model-depth.md), [point fusion, persistent depth and TSDF](runtime/reconstruction.md), [datasets and replay](runtime/datasets.md) |
| Headset presentation | [Rendering and occlusion](runtime/rendering.md), [OpenXR hand viewers](runtime/hand-tracking-viewer.md) |

## Model samples and experiments

| Workload | Guides |
| --- | --- |
| Depth | [Depth model and tooling](model-development/depth.md), [CPU reference](model-development/cpu-reference.md), [DepthPro2 DSP handoff](model-development/depthpro2-dsp-handoff-2026-11-20.md) |
| Hands | [MediaPipe RGB](model-development/hand-tracking.md), [Mercury stereo](model-development/mercury-hand-tracking.md), [stereo NIR](model-development/nir-hand-tracking.md) |
| Controllers | [XRService FeatureNN](model-development/controller-featurenn.md) |

Keep active working notes in the ignored root `research/` and `research_notes/` directories. The tracked [temporal-fusion report](../reports/Temporal%20depth%20fusion%20for%20Frame.md) and [executor fusion handoff](../hexecutor/frame/handoffs/HANDOFF_FUSION.md) are preserved project records.
