# Local model artifacts

This repository tracks runtime code, kernel implementations, reference tools,
training recipes, and documentation. It does not distribute raw weights,
compiled model contexts, recovered graphs, tensor/memory captures, or model
archives. These were removed from every local Git ref with `git-filter-repo`.

Local copies remain available in ignored directories:

- `models/`: model weights and compiled contexts.
- `recovered/`: exported graphs, constants, and execution plans.
- `metadata/`: exported binary/model metadata and captured evidence.
- `frame/build/`: local compiled executors and packaged runtime artifacts.
- `frame/native_program/fixtures/` and other fixture directories: captured tensors and memory pools.
- `training/checkpoints/`, `training/datasets/`, and `training/runs/`: model-development artifacts.

Compatibility links under `../hexecutor/` expose these resources to existing
tools. A fresh clone needs separately provisioned artifacts before running
full-model inference or model-dependent validation. Generic geometry, temporal,
bridge protocol, and artifact-policy checks do not require raw weights.

For the native bridge, set `FRAME_DEPTH_PACKAGE` to the installed package root
and `FRAME_DEPTH_PYTHON` to its Python interpreter when overriding the defaults.
`FRAME_DEPTH_BACKEND=prepared-qnn` is the default and supports the primary
run list; `prepared` also supports the alternate model. Device installations
must be updated separately; this cleanup does not deploy to a headset.

Model-specific generated headers (including run-list headers), captured array
headers, and fixture files also stay local. Generate or supply them alongside
the corresponding local model; build recipes that use them require those assets.

Audit the tracked tree with `python3 tools/model_artifact_policy.py` from the
repository root. Use `--history` to audit paths across all refs. New binary
formats or source-encoded captures must be added to both the shared policy and
`.gitignore` before they are introduced.
