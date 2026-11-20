# Training, local artifacts, and source history

Depth and hand tracking have independent development workspaces. Keep reusable inference code in Hexecutor and camera/application integration in `depth-runtime/`. Place model definitions, reproducible training configuration, export/lowering code, and evaluation recipes with the corresponding model component.

## What belongs in Git

Track source, configuration, model contracts, documented provenance, and aggregate evaluation methodology/results that contain no raw dumps. Keep datasets, checkpoints, compiled contexts, recovered graphs/constants, captured tensor pools, generated model headers, and distribution archives local.

The artifact policy excludes common weight/container formats, tensor arrays, memory captures, model-specific generated/array headers, binary build products, portable validation packages, and address-named captures. Header source is not an acceptable way to check captured weights back in: the audit also detects large numeric C initializers.

| Local location | Purpose |
| --- | --- |
| `samples/depth-model/models/`, `recovered/`, `metadata/` | Legacy local model resources and evidence |
| `samples/hand-tracking/models/`, `model-assets/` | Private hand/model collections |
| Model `training/checkpoints/`, `datasets/`, `runs/` | New training artifacts |
| `hexecutor/native/build/`, fixture directories | Compiled engines, generated packages, numerical captures |
| `depth-runtime/experiments/` | Recorded sessions, replay products, and evaluation data |

Model data, engine artifacts, and generated programs have explicit paths. `FRAME_DEPTH_DATA` configures offline model resources; the native model adapter accepts package/artifact/program locations without filesystem aliases. Full-model builds/validation require the corresponding local assets; generic geometry, protocol, and repository-policy checks do not.

## Replacement model record

For each trained export, record the base model and teacher if used, dataset identity/splits, training configuration, preprocessing, tensor/state contract, export/compiler versions, generated schedule identity, and output hashes. Evaluate accuracy, temporal stability, and on-device performance against the same defined inputs.

Distillation or fine-tuning does not by itself establish independent provenance. Generic API names and removal of distributed legacy assets do not change the origin of remaining dependencies. Retain accurate source URLs and required license notices.

## History cleanup

The monorepo combines the application and DSP histories. An explicitly authorized `git-filter-repo` cleanup then removed raw artifacts and applied generic component/API names across every local/imported ref. Commit IDs changed; original authors/timestamps and surviving source history were retained. The original spatial-stuff checkout is separate and was not rewritten.

Pre-cleanup source/Git and documentation recovery copies are outside the monorepo. Do not merge their unfiltered refs into the sanitized repository. The root `research/` and `research_notes/` directories were filtered from local history and remain ignored working directories. The temporal-fusion report remains tracked.

Run from the monorepo root:

```sh
python3 tools/model_artifact_policy.py
python3 tools/model_artifact_policy.py --history
```

The first audits the index; the second checks paths and blob contents across all refs. Add new artifact formats to both the shared policy and `.gitignore` before using them. Ignoring an already tracked file does not remove it from history.
