# Hand-tracking model development

Place hand-model distillation and fine-tuning code, model definitions,
reproducible configuration, export tools, and evaluation recipes here as they
are developed. `../models/` and `../COMPATIBILITY.md` are the existing legacy
asset and research lane.

Use `datasets/`, `checkpoints/`, and `runs/` for local artifacts; these are
ignored by Git. Record dataset and teacher provenance, base-checkpoint identity,
training configuration, and export hashes with each model's tracked recipe.

Define camera preprocessing, handedness, joint ordering, coordinate frames,
units, temporal state, and confidence semantics explicitly. Evaluate accuracy,
occlusion recovery, temporal stability, and latency before integrating a model.
Put shared DSP execution changes in `../../hexecutor/`, and application
integration changes in `../../runtime/`.

As validated replacements become available, remove unused legacy assets and
references from the current tree. Original references remain in Git history.
This directory is a development boundary, not an implemented training pipeline.
