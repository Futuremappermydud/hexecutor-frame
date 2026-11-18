---
name: host-checks
description: Run the repo's portable gates on the Mac - make check (CMake host build, ctest incl. qnn16/nirhands/broker tests, layout/docs/policy checks) and the model-artifact history policy - before reporting native/host changes as done or before any commit. Use after editing hexecutor/host, hand-tracking/nirhands, runtime host code, docs links, .gitignore or artifact policy.
---

# Host checks

From the repo root (`docs/development.md`, `docs/getting-started.md`):

```sh
make check          # cmake -S . -B build; cmake --build build; ctest --output-on-failure; python3 tools/dev.py check
make check-docs     # documentation links only (fast)
make check-layout   # layout, live native entry, Python syntax
```

- `make check` needs no weights. Some tests take local artifacts when present: `npu_layout` uses
  `hand-tracking/.work/nirhands-models` (and the skeleton dir), so its coverage depends on what's in `.work`.
- `make check-model` needs private local assets (program/image readers, saved oracle hashes); run it when touching
  depth model loading.
- After changing `.gitignore`, `tools/model_artifact_policy.py` or any refs (AGENTS.md requirement):
  ```sh
  python3 tools/test_model_artifact_policy.py
  python3 tools/model_artifact_policy.py --history     # same as `make audit`
  ```
- Report the ctest count (e.g. "15 of 15") and any failures with their output; a macOS pass doesn't prove the
  Linux device-transport branch or headset inference (`docs/development.md`).

## Gotchas

- The worktree carries many uncommitted edits from other work. A failure may be someone else's change: check
  `git status`/`git diff` for the failing file before "fixing" it, and don't stage or revert unrelated files.
- Never add model artifacts, captured arrays or weights to git (`.work/` dirs are ignored; keep them there).
- Live depth path is C++17 only; don't add Python to it.
