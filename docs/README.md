# Frame Spatial Runtime documentation

This is the maintained handbook for the spatial application, Hexecutor DSP execution, and perception-model development. It describes the current source tree and carries forward durable engineering findings from the earlier investigation notes. Recorded timings are dated observations, not promises for a fresh installation.

## Start here

- [System architecture](architecture.md): how capture, depth, reconstruction, rendering, and the DSP connect.
- [Build and run](getting-started.md): host versus headset commands and local model prerequisites.
- [Development and validation](development.md): where to change code and what evidence a change needs.

## Spatial runtime

| Guide | What it explains |
| --- | --- |
| [Runtime overview](runtime/README.md) | Processes, threads, backends, and source ownership |
| [Cameras and calibration](runtime/cameras.md) | DMA-BUF capture, sensor identity, coordinate conventions, timing, and calibration drift |
| [Stereo matching](runtime/stereo.md) | Vulkan and CPU matchers, confidence gates, and experimental alternatives |
| [Learned depth integration](runtime/model-depth.md) | Model preprocessing, native bridge, state, and deployment selection |
| [Reconstruction](runtime/reconstruction.md) | Point fusion, persistent depth, sparse TSDF, and mesh generation |
| [Rendering and occlusion](runtime/rendering.md) | OpenVR diagnostics, OpenXR scenes, and the Vulkan API layer |
| [Datasets and evaluation](runtime/datasets.md) | Capture sessions, replay, controller marks, IR refinement, and provenance |

## Hexecutor

| Guide | What it explains |
| --- | --- |
| [Generic tensor backend](hexecutor/tensor-backend.md) | Static tensor graphs, multi-model sessions, FP16 HMX and model-independent tooling |
| [Native host API](hexecutor/api.md) | C++ tensors/programs, explicit packages, and worker lifecycle |
| [Executor architecture](hexecutor/README.md) | Host/DSP responsibilities, loader, transport, and execution modes |
| [Memory layout](hexecutor/memory-layout.md) | Private DDR, VTCM, L2 leases, aliasing, and workspace limits |
| [Dynamic CDSP memory](hexecutor/dynamic-memory.md) | Dynamic memory growth and shrinking from 128 MiB to 1.5 GiB based on workloads |
| [CDSP broker (proposed)](hexecutor/broker.md) | One owner for the CDSP: scheduling depth, hands, controller and eye work |
| [XRService broker bridge](hexecutor/xrservice-broker-bridge.md) | Bridging and patching XRService StereoFeatureNN and Gazenet via shim library |
| [Clocks and bandwidth](hexecutor/clocks.md) | AXI and HMX clock findings, controlled measurements, and restoration |
| [Kernels and arithmetic](hexecutor/kernels.md) | HVX/HMX, blocked layouts, exact conversion, and adapters |
| [Scheduling and optimizations](hexecutor/scheduling.md) | Dependencies, lane scheduling, plan replay, DMA, and rejected changes |
| [Formats and host API](hexecutor/formats.md) | QDRP, QDRX, NPRG v2, pool binding, and preflight |
| [Build and operations](hexecutor/operations.md) | Toolchains, package identity, session lifecycle, and recovery |
| [Performance interpretation](hexecutor/performance.md) | The pinned deployment versus the latest measured primary recipe |
| [Kernel adapters](hexecutor/kernel-adapters.md) | Shape-specific support boundaries and fixture generation |
| [Language-model example](hexecutor/language-model.md) | The Qwen DSP client and its shared infrastructure |

## Models and repository policy

- [Depth model](model-development/depth.md), [CPU reference](model-development/cpu-reference.md), and [hand tracking](model-development/hand-tracking.md), including its [optimization checkpoint](model-development/hand-tracking-progress.md) and the [Mercury QNN run-list plan](model-development/mercury-hand-tracking.md); [controller FeatureNN](model-development/controller-featurenn.md) on the tensor backend.
- [Training and local artifacts](model-development/training-and-artifacts.md): future model replacement, ignored assets, and history policy.

The temporal-fusion report and related working notes remain in their original locations and are outside this rewrite. Both fusion handoffs are also preserved unchanged. Superseded session diaries are available in Git history and the external documentation recovery copy; they are not the primary navigation system.
