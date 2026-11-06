# Build and run

The application targets the Frame's ARM64 Linux environment, SteamVR libraries, Vulkan, and OpenXR. A host syntax check is useful, but a normal macOS compiler is not a complete build environment for the headset application or Hexagon payloads.

## Host tools and native worker

From the repository root, `make build-host` builds the C++ host, worker, and portable tests; `make check` runs those tests plus policy, layout, documentation, and geometry checks. `make check-model` adds checks requiring local model resources. `make help` lists the available tasks. Python is used by offline development tools, not live depth inference.

## Spatial application

Work from `runtime/`. Its Makefile uses C++17, optimization, OpenMP, and the installed SteamVR/OpenXR headers and libraries. `glslc` builds the compute shaders. MNN is an optional historical Vulkan-AI dependency; the native Vulkan matcher does not require it.

```sh
cd runtime
./scripts/setup_frame.sh
./scripts/deploy_and_run.sh
```

The deploy script synchronizes the monorepo source layout to `/home/steamos/frame-spatial-runtime`, honoring artifact exclusions, builds the native host/worker and application, then launches from its `runtime/` subdirectory. It does not provision private model packages. `FRAME_HOST` and `FRAME_REMOTE_DIR` override the remote target.

To build directly on the headset:

```sh
cd /home/steamos/frame-spatial-runtime
make build-runtime
cd runtime
./bin/frame_openxr_point_cloud
```

SteamVR must be awake and its camera buffers present. The setup script configures ptrace access so the application can duplicate compositor DMA-BUF descriptors. Do not replace this with opening the claimed sensor devices.

## Learned depth

Provision the model package, matching DSP engine/bootstrap, generated program tables, separately. A fresh clone does not contain weights, model dumps, generated model headers, or capture fixtures.

```sh
# From runtime/ on the development host, after installing the matching package:
./scripts/setup_frame.sh --qnn
```

On the headset:

```sh
cd /home/steamos/frame-spatial-runtime/runtime
OMP_NUM_THREADS=2 ./bin/frame_openxr_point_cloud --depth-model-qnn
```

`--depth-model` also selects learned depth and defaults to `prepared-qnn`. `--depth-model-prepared` selects the much slower exact prepared reference engine. These are different engine/package identities, not interchangeable labels. The QNN run list supports the primary model only.

`FRAME_DEPTH_PACKAGE`, `FRAME_DEPTH_WORKER`, `FRAME_DEPTH_BACKEND`, and `FRAME_DEPTH_MODEL` select the installed resources. The generic default paths use `~/frame-dsp/depth-model/`; existing device installations need that layout or explicit overrides. The setup script also expects its configured package path on the device. Nothing in the documentation reorganization migrates a remote installation.

## DSP development

Start with [Hexecutor operations](hexecutor/operations.md), then its [memory map](hexecutor/memory-layout.md). `make -C hexecutor/native` builds the basic callback targets with a Hexagon-capable compiler; it is not the optimized prepared-engine recipe. `depth-model/deployment/lanes/build_best.sh` records that recipe and requires local generated headers and model resources.

For failures, first identify the failing boundary: camera access, GPU initialization, model package verification, DSP startup, inference, or cleanup. The runtime and executor guides explain their diagnostics. A recorded benchmark does not prove a newly installed package is healthy.
