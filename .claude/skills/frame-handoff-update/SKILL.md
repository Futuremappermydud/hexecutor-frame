---
name: frame-handoff-update
description: Update the maintained Frame topic guide with verified results, coverage and remaining work. Use for requests to document progress or leave a handoff. Keep scratch notes and experiment-only results under ignored research.
---

# Update the Frame documentation

- Start with `docs/README.md` and update the maintained page for the component. Replace superseded status there when evidence changes. Put reusable interfaces, checks and constraints in the guide. Use a dated evidence entry when a measurement needs its own context.
- Keep worktree state, device paths, detailed run logs and one-off experiments under ignored `research/` or `research_notes/`. Prefer one document per experiment or topic. Git history holds the superseded tracked notes.
- Preserve `docs/model-development/depthpro2-dsp-handoff-2026-11-20.md`, the temporal-fusion report, its working notes, and both fusion handoffs per the repository guidance.

## Record evidence

Check the current implementation and named build/test entry points before revising a status. Record the artifact identity, target, input, measured outcome and what still needs verification. Keep host, simulator and device runs distinct. For hardware runs record ownership and firmware state before and after, including restoration. Inspect the target device only when the task includes device access.

Finish with `make check-docs`. Check that moved pages have no remaining references from the maintained guides or workflows.
