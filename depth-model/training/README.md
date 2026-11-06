# Depth model development

Place depth distillation and fine-tuning code, model definitions, reproducible
configuration, export tools, and evaluation recipes here as they are developed.
The existing `../models/`, `../recovered/`, and `../tools/` are the legacy
recovery/reference lane. Keep new training workflows separate from it.

Use `datasets/`, `checkpoints/`, and `runs/` for local artifacts; these are
ignored by Git. Record dataset and teacher provenance, base-checkpoint identity,
training configuration, and export hashes with each model's tracked recipe.

Define preprocessing, eye order, tensor shapes, disparity/depth units, temporal
state, and confidence semantics explicitly. Compare accuracy, temporal
stability, and on-device latency before replacing the existing runtime backend.
Put shared DSP execution changes in `../../hexecutor/`, and application
integration changes in `../../runtime/`.

Legacy raw model artifacts have been removed from Git history and remain local
and ignored. Keep future checkpoints and dataset captures out of source control.
This directory is a development boundary, not an implemented training pipeline.
