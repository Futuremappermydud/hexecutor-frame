# Monorepo guidance

- Start with [the documentation index](docs/README.md) and [development guide](docs/development.md).
- Application changes are under `runtime/`; read `runtime/AGENTS.md` before touching them. DSP code is under `hexecutor/native/`, native host code under `hexecutor/host/cpp/`; model tooling belongs to its model component.
- Use explicit model/artifact paths; code/model resource symlinks have been retired. Component README links and `runtime/docs` point to canonical content under `docs/`.
- Maintain generic interfaces: `ModelDepthPipeline`, `--depth-model`, and `FRAME_DEPTH_*`. Preserve source URLs and required license attribution.
- History was sanitized with explicit user authorization. Do not reintroduce unfiltered refs, raw model artifacts, captured arrays, or embedded weights. Run `python3 tools/model_artifact_policy.py --history` after changing artifact policy or refs.
- Existing working edits and local assets must not be staged/discarded as unrelated cleanup.
- Do not modify or move `reports/Temporal depth fusion for Frame.md`, anything under `research_notes/Temporal depth fusion for Frame/`, or the preserved fusion handoffs (`research/HANDOFF_FUSION.md` and `hexecutor/frame/handoffs/HANDOFF_FUSION.md`) as part of documentation maintenance.

- The live depth host/worker is C++17. Keep Python for offline development only; do not reintroduce an interpreter or hybrid fallback into the live depth path.
