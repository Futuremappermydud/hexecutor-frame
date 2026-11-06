# Frame Spatial Runtime

Monorepo for the Steam Frame spatial runtime, Hexagon DSP execution, and perception model development.

| Directory | Contents |
| --- | --- |
| [runtime/](runtime/README.md) | C++ spatial app, camera capture, Vulkan/OpenXR rendering, depth integration, and room reconstruction. |
| [hexecutor/](hexecutor/README.md) | Hexecutor: Hexagon HVX/HMX DSP executor, kernels, transport, and bring-up tools. |
| [depth-model/](depth-model/README.md) | Depth training, CPU reference, simulator, and model tooling. Model artifacts stay local. |
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
deploys the app from `runtime/`. Configure the installed DSP package location
with `FRAME_DEPTH_PACKAGE`, and the Python executable with `FRAME_DEPTH_PYTHON`.
The default package paths now use the generic `depth-model` directory name;
existing installations must use that layout or set the overrides.
The learned inference backend defaults to `prepared-qnn`; `prepared` remains
available. The hybrid Python inference path has been removed.

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

Keep the provenance and dependencies of future models explicit in their training
recipes. Renaming code and removing distributed weights does not change model
provenance. External source URLs and required license notices retain their
original attribution.

## Git history and migration

The `monorepo/hexecutor` branch combines the spatial app and DSP histories.
History was subsequently rewritten with `git-filter-repo` to use generic
component/API names and remove model weights, recovered graphs, compiled model
contexts, captured pools, embedded model arrays, and distribution archives
across every local branch and imported branch ref. Commit IDs changed; authors,
timestamps, and the surviving source history remain. Imported branches remain
under `refs/remotes/spatial-stuff/` with sanitized names and history.
The original spatial-stuff checkout and its linked worktrees are separate and
were not rewritten. A pre-cleanup recovery snapshot is stored outside this repo.

Working edits from both checkouts are preserved as uncommitted changes at
their new paths. Local datasets, build products, and hand-model binaries remain
on disk and are ignored. They are not automatically included in a fresh clone.
The hand-tracking directory currently contains research and local assets;
it does not yet implement a live hand-tracking pipeline.

Run `python3 tools/model_artifact_policy.py` to audit the current index, or add
`--history` to audit paths across all refs. The shared policy also detects large
C array initializers during history filtering, so captured weights stored in
headers are excluded alongside binary files. See [artifact setup](depth-model/ARTIFACTS.md).
