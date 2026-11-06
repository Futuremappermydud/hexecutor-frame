# Frame Spatial Runtime

Spatial capture, depth, reconstruction, and rendering for Steam Frame, with the Hexecutor native DSP engine and depth/hand-model workspaces.

**[Read the documentation](docs/README.md)** — architecture, build/run, runtime components, DSP memory/clocks/optimizations, and model development.

| Directory | Purpose |
| --- | --- |
| `runtime/` | C++ application, camera/geometry, GPU depth, rendering, recording/replay |
| `hexecutor/` | Hexagon transport, native programs, HVX/HMX kernels, and scheduling |
| `depth-model/`, `hand-tracking/` | Model contracts, reference/training tools, local artifact workspaces |
| `docs/` | Maintained program documentation |
| `tools/` | Repository/artifact policy |

Weights, model dumps, generated model headers, and datasets remain local and ignored. The history was sanitized; do not reintroduce unfiltered refs. See [artifact policy](docs/model-development/training-and-artifacts.md).

The temporal-fusion report and related working notes remain in their original locations and were not rewritten.

Root development commands: `make build-host`, `make check`, `make check-model`, `make build-runtime`, and `make build-dsp`. `make help` lists the tasks. The live depth pipeline uses the C++ worker; Python remains available through `make run SCRIPT=...` for offline tools.
