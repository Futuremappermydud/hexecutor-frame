# Frame Spatial Runtime

Monorepo for the Steam Frame spatial runtime, Hexagon DSP execution, and recovered perception models.

| Directory | Contents |
| --- | --- |
| [runtime/](runtime/README.md) | C++ spatial app, camera capture, Vulkan/OpenXR rendering, depth integration, and room reconstruction. |
| [hexecutor/](hexecutor/README.md) | Hexecutor: Hexagon HVX/HMX DSP executor, kernels, transport, and bring-up tools. |
| [depth-model/](depth-model/README.md) | Stereo depth models, recovered graphs, CPU reference, simulator, and model tools. |
| [hand-tracking/](hand-tracking/README.md) | Local hand-model assets and executor compatibility research. |
| research/ | Original investigation and handoff documents. |

## Build and run

The app's existing commands run from `runtime/`:

```sh
cd runtime
make -j4
# On the development host, deploy with ./scripts/deploy_and_run.sh
```

DSP builds run from `hexecutor/frame/`; see its existing build and validation documentation.
Model tools run from `depth-model/`. Compatibility links preserve the historical
`depth-model/frame` layout and provide model resources under `hexecutor/` for
scripts that locate resources relative to their resolved file path. Keep these
links when packaging or syncing these components. The app deployment script
deploys the app from `runtime/`; the existing remote DSP package paths remain
valid and can still be selected with `FRAME_DEPTH_PACKAGE`.

## Model development

Depth and hand tracking will have independent distillation/fine-tuning workflows:
[depth training](depth-model/training/README.md) and
[hand-tracking training](hand-tracking/training/README.md).
Keep new training code, model definitions, and evaluation recipes in those
component directories. Keep checkpoints and datasets out of Git. Existing
recovered models and research form the legacy reference; new models should have
explicit input/output contracts and their own provenance, configuration, and
validation results. Existing depth-specific DSP code may need adaptation for a
replacement model.

The intended migration is to replace legacy model dependencies and remove
obsolete assets and references from the current tree once replacements are
validated. Preserving the original Git history retains historical material in
past commits. Fine-tuning or distillation alone does not establish that a
replacement has independent provenance; record its actual dependencies.

## Git history and migration

The `monorepo/hexecutor` branch joins the original `model-tsdf` runtime tip and
the `frame-speed-20261105` spatial-stuff tip through relocation commits followed
by a two-parent merge commit.
Original commits, authors, timestamps, and commit IDs are unchanged. The old
paths exist in those commits; use `git log --follow -- <new-path>` to trace a
file across the move. The other spatial-stuff branches are retained locally
under `refs/remotes/spatial-stuff/`. The original spatial-stuff checkout and
its linked worktrees remain available.

Working edits from both checkouts are preserved as uncommitted changes at
their new paths. Local datasets, build products, and hand-model binaries remain
on disk and are ignored. They are not automatically included in a fresh clone.
The hand-tracking directory currently contains research and local assets;
it does not yet implement a live hand-tracking pipeline.
