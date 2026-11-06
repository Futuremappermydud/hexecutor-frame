# Monorepo guidance

- `runtime/` contains the spatial application. Read `runtime/AGENTS.md` before changing it; its hardware and execution invariants still apply. Run its build/deploy commands from `runtime/`.
- `hexecutor/frame/` contains DSP code; `hexecutor/frame-dsp-followup/` contains early bring-up tools.
- `depth-model/` owns depth models, recovered graphs, and reference/model tooling.
- `hand-tracking/` owns hand-model assets and compatibility research.
- Preserve compatibility symlinks between `depth-model/` and `hexecutor/`; existing tools use both lexical paths and resolved Python paths.
- History has been sanitized with explicit user authorization. Do not reintroduce pre-cleanup branches, raw model artifacts, captured arrays, or embedded weights. Preserve the sanitized source history.
- Model artifacts remain local and ignored. Run `python3 tools/model_artifact_policy.py --history` after changes to the tracked artifact policy or refs.
- Generic depth APIs use `ModelDepthPipeline`, `--depth-model`, and `FRAME_DEPTH_*`. Keep external source citations and required license attribution accurate.
- Existing working changes and local artifacts were retained during migration; do not stage or discard them as part of unrelated work.
