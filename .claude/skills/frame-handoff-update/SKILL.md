---
name: frame-handoff-update
description: Record a work session in this repo's dated handoff docs (docs/model-development/*-handoff-YYYY-MM-DD.md and topic guides) as a timestamped "## Update" section, with measured numbers, artifact paths on the Mac and Frame, DSP state and next steps, then update memory. Use when the user says "write it up", "update the handoff", "document where we are", or when stopping mid-task in frame-spatial-runtime. (For cowfish/stereo_nir_hands handoffs use that repo's session-handoff skill.)
---

# Update a handoff in frame-spatial-runtime

## Where it goes

- Ongoing line of work with an existing dated handoff: **append** a section to it, newest last, headed
  `## Update, YYYY-MM-DD HH:MM: <result in a few words>` (see
  `docs/model-development/nirhands-skeleton-integration-handoff-2026-11-13.md`). Don't rewrite earlier sections;
  if a later result supersedes one, say so in the new section ("the Blocker section above is superseded").
- New topic: `docs/model-development/<topic>-handoff-<today>.md`, and link it from `docs/README.md` or the topic guide
  only if it's meant to be maintained.
- Durable findings (how an op/format/tool works, rules that keep holding) belong in the maintained guides
  (`docs/hexecutor/*.md`, `docs/model-development/*.md`, `docs/runtime/*.md`), not only in a diary. For QNN16
  lowering/engine/deploy rules that's `docs/model-development/nirhands-qnn16-lowering-notes.md`.
- **Never edit**: `reports/Temporal depth fusion for Frame.md`, `research_notes/Temporal depth fusion for Frame/`,
  `research/HANDOFF_FUSION.md`, `hexecutor/frame/handoffs/HANDOFF_FUSION.md`.

## Gather facts first

- `git status --short` for the files you touched; say they're uncommitted (most of this work is).
- Exact numbers from logs, not memory: ISS `PASS ... pcycles` lines, `dsp ms p50`, replay step p50/p90/p99, ONNX diffs,
  md5s of engines/programs.
- DSP state after the last run (`frame-dsp-state`).
- What's on the Frame (`ssh frame ls ~/nirhands-skel/run ...`) and which launch scripts exist.

## Section contents (only what was actually run)

1. **Why**: the problem/measurement that motivated it.
2. **Model/Code**: checkpoint + SHA, files changed, one line each.
3. **Checks**: each gate with its result (fuzz kinds and layouts, bit-identity, `make check`), and what was *not* run.
4. **Frame**: paths of deployed files (new names, old ones kept), measured DSP/step times, control state afterwards,
   the exact launch command for the user.
5. **Still open / next**: numbered, with commands; flag decisions that belong to the user.

Write plainly and briefly; use absolute dates. Run `make check-docs` if you added links.

## Memory

Update the existing memory file for that topic (e.g. `hand-npu-perf.md`, `mercury-qnn-runlist.md`) with the new
handoff path, key number and next step; add a new file + `MEMORY.md` line only for a new topic. Commit only if the
user asks.
