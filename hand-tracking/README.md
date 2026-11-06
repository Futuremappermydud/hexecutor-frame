# Hand tracking

Workspace for recovered hand-tracking models and future Frame integration.

- `models/`: local Model hand-tracking assets, copied from `model-assets/handtracking/`; ignored by Git, as in the original repository.
- `COMPATIBILITY.md`: existing native-executor compatibility research, retained as an uncommitted file.

The DSP implementation lives in [Hexecutor](../hexecutor/README.md). This
directory does not yet contain a working hand-tracking inference pipeline.
Model assets must be supplied separately in a fresh clone.
