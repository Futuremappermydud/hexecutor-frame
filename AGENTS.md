# Monorepo guidance

- `runtime/` contains the spatial application. Read `runtime/AGENTS.md` before changing it; its hardware and execution invariants still apply. Run its build/deploy commands from `runtime/`.
- `hexecutor/frame/` contains DSP code; `hexecutor/frame-dsp-followup/` contains early bring-up tools.
- `depth-model/` owns depth models, recovered graphs, and reference/model tooling.
- `hand-tracking/` owns hand-model assets and compatibility research.
- Preserve compatibility symlinks between `depth-model/` and `hexecutor/`; existing tools use both lexical paths and resolved Python paths.
- The migration preserves original Git commits. Do not rewrite either imported history.
- Existing working changes and local artifacts were retained during migration; do not stage or discard them as part of unrelated work.
