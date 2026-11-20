# Hexecutor for Steam Frame

Hexecutor is a native Hexagon DSP engine for Steam Frame. This repo includes its C++ host and DSP kernels, the Frame depth application, and model samples.

**[Read the documentation](docs/README.md)** — architecture, build/run, runtime components, DSP memory/clocks/optimizations, and model development.

| Directory | Purpose |
| --- | --- |
| `hexecutor/` | Hexagon transport, native programs, HVX/HMX kernels, and scheduling |
| `depth-runtime/` | C++ application, camera/geometry, GPU depth, rendering, recording/replay |
| `samples/depth-model/`, `samples/hand-tracking/` | Model contracts, reference/training tools, local artifact workspaces |
| `samples/controller-tracking/` | StereoFeatureNN reference and export tools |
| `docs/` | Maintained program documentation |
| `tools/` | Repository/artifact policy |

Weights, model dumps, generated model headers, and datasets remain local and ignored. The history was sanitized; do not reintroduce unfiltered refs. See [artifact policy](docs/model-development/training-and-artifacts.md).

The top-level `research/` and `research_notes/` directories are local and ignored. The [temporal-fusion report](reports/Temporal%20depth%20fusion%20for%20Frame.md) and [executor fusion handoff](hexecutor/frame/handoffs/HANDOFF_FUSION.md) are tracked.

Root development commands: `make build-host`, `make check`, `make check-model`, `make build-runtime`, and `make build-dsp`. `make help` lists the tasks. The live depth pipeline uses the C++ worker; Python remains available through `make run SCRIPT=...` for offline tools.

To build only Hexecutor's native host tools, run `cmake -S hexecutor -B build-hexecutor` and `cmake --build build-hexecutor`.
