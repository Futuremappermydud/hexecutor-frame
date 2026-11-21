# Hexecutor and Frame spatial runtime

This repository contains **Hexecutor**, a reusable native execution engine for Qualcomm Hexagon DSPs, alongside a Steam Frame depth application and model development samples. Hexecutor provides host transport, DSP programs, kernels, memory management, and scheduling. Projects with supported Hexagon workloads can use its C++ API with their own model contracts, artifacts, inputs, and outputs.

## Components

| Component | Role |
| --- | --- |
| [`hexecutor/`](hexecutor/) | Generic C++17 host API, Hexagon DSP code, broker, ExecuTorch delegate, and offline lowering tools. Start with the [executor architecture](docs/hexecutor/README.md) and [C++ API](docs/hexecutor/api.md). |
| [`depth-runtime/`](depth-runtime/) | Steam Frame C++17 application for camera capture, stereo depth, fusion, and OpenXR presentation. Learned depth can use Hexecutor through a separate model adapter and native worker. See the [runtime guide](docs/runtime/README.md). |
| [`samples/depth-model/`](samples/depth-model/) | Depth model contracts, reference implementations, deployment packages, and offline tooling. See the [depth model guide](docs/model-development/depth.md). |
| [`samples/hand-tracking/`](samples/hand-tracking/) | RGB, stereo, and NIR hand tracking experiments and model tooling. See the [hand tracking guide](docs/model-development/hand-tracking.md). |
| [`samples/controller-tracking/`](samples/controller-tracking/) | Controller tracking reference models and export tooling. See the [FeatureNN guide](docs/model-development/controller-featurenn.md). |

The [system architecture](docs/architecture.md) shows how the Frame application uses these components. The [documentation index](docs/README.md) lists the maintained guides by topic.

## Build and check

To build Hexecutor's host targets independently of the depth application:

```sh
cmake -S hexecutor -B build-hexecutor
cmake --build build-hexecutor --parallel
ctest --test-dir build-hexecutor --output-on-failure
```

From the repository root, `make build-host` builds the native host, depth worker, and portable tests; `make check` runs those tests and repository checks. `make build-runtime` builds the Frame application on a configured headset, and `make build-dsp` requires a Hexagon toolchain. Run `make help` for the full command list and see [build and run](docs/getting-started.md) for device setup.

The live depth host and worker use C++17. Python tools are for offline development and validation.

## Models and artifacts

Weights, generated model headers, execution packages, captures, and datasets are provisioned separately and are excluded from Git history. Use explicit paths for model resources; see [training and artifacts](docs/model-development/training-and-artifacts.md). The Frame depth application selects its model through `--depth-model` and `FRAME_DEPTH_*` settings. These settings belong to the Frame application.

## License

Hexecutor is `LGPL-3.0-or-later`; its kernel modules have separate `GPL-2.0` terms. See the [Hexecutor license and distribution terms](hexecutor/LICENSE.md). Model weights, datasets, and third-party components retain their own upstream licenses and attribution.
