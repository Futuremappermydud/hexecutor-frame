# Handoff: prepared command support — 2026-10-31

The user wants continued implementation of prepared executor command support, using Luna subagents. They specifically rejected stopping after three narrowly pinned command sites. Continue expanding real family coverage and testing, not merely editing documentation. The worktree contains extensive pre-existing uncommitted work; do not reset or discard it.

## Current work

- `frame/native_program/prepared_sign.{c,h}` was broadened to FH and blocked CH Sign. The agent reported host preflight acceptance of 13/13 commands per model and exact full-pool CPU-prefix comparisons for one FH and one CH per model. A V75 run of four cases was underway; inspect its outcome and source before relying on it. Review its page alias proof and stack use (the agent described two 2,560-entry local arrays) against the DSP stack budget.
- `prepared_addsub_flat.{c,h}` was broadened from one fixed command to the six actual flat Add/Sub commands (three per model). The agent reported all six host CPU-prefix full-pool comparisons passing. Three arithmetic profiles exist, mirrored across models: one subtraction and two additions. V75 had passed the first two primary profiles; the final pool-role guard was compiled strictly for V75 after the simulator binary was built. Check whether the remaining simulator run completed and verify the final source.
- `prepared_mul_flat.{c,h}` was being broadened to the four flat Mul sites (two per model) and two quantization/parameter profiles. Four CPU-prefix fixtures were captured; the agent reported V75 in progress. Its final outcome was not received before this handoff. Inspect and run the host harness; avoid claiming coverage until all four pass. CH Mul (10 commands per model) remains a target.
- `prepared_reshape_cb.{c,h}`, `capture_reshape_cb_cpu_prefix.py`, and `test_prepared_reshape_cb.c` are new. The handler supports `Reshape.hw@CB*2` variant 49, source `[1,1,64,128]` with row padding 0–7 to output `[1,8,8,128]`, validates pool-1 page bounds, uniqueness, and source/output nonaliasing. Strict host test just passed preflight for **80/80 per model**, full-pool CPU-prefix comparisons for primary command 1967 and alternate command 2187, and malformed page rejection. The fixtures only exercise their selected source rows; capture and compare additional row-padding profiles (especially 7), add a V75 build/run script, and test on V75 before claiming simulator proof. Review the 32-byte copy mapping and padding assumptions.

All new handlers require explicit registration; they are not integrated into the Frame runtime. Full-model execution and device timing remain unsupported. `STATUS.md` and `frame/native_program/EXECUTOR.md` currently describe earlier one-site coverage and must be updated after tests finish; do not overstate numerical or hardware proof. The untracked `nprg2_plan_cache` is an unsafe unintegrated prototype: cached preflight skips live handler validation.

## Next steps

1. Inspect the current source, `git status`, and any still-running Docker/Hexagon simulator processes. Agents are no longer listed as active; do not assume pending simulations completed.
2. Run the three family host harnesses and inspect failures. Finish representative V75 tests with exact final sources. Full-pool simulator tests are slow under Rosetta; choose representative distinct profiles and retain host census across all actual commands.
3. Complete Reshape coverage tests for source row padding and V75, then consider integrating registered handlers into a bounded runner only after correctness and stack/scratch checks.
4. Expand other high-frequency uncovered families, e.g. ResizeNear (107/model), Transpose (82/model), and CH Mul (10/model), with fail-closed preflight and CPU-prefix physical-pool comparisons. Prefer meaningful family coverage over hard-coded command indices.
5. Update status/executor documentation with measured counts, test scope, and remaining gaps. Do not commit unless requested.

Useful context: QDRX command refs returned by `resident_image.unpack` are tensor **IDs**, so map with `{t['id']: t for t in decoded['tensors']}`; they are not list indices. `compact_v2.verify_equivalence` passed for the current alternate QDRX and NPRG v2. The `depth-model` directory is the working directory for simulator scripts and fixtures. The project uses `depth-model/.venv-cpu` and Docker image `depth-model-hmx-tools:latest`.

## Update — 2026-10-31 13:40 (follow-up session)

All four families are now verified on the current sources, host and V75. Results are recorded in `STATUS.md` and `frame/native_program/EXECUTOR.md`:
- Sign: host 26/26 fixtures with `-DSIGN_ALL_FIXTURES`. V75 4/4.
- AddSubFlat: host 6/6. V75 6/6 (re-run, because the earlier container predated the final source edit).
- MulFlat: host 4/4. V75 4/4.
- Reshape CB: `capture_reshape_cb_cpu_prefix.py --all-pads` added (16 new `*-reshape-cb-pad{0..7}.epcpu` fixtures, ~18 MB each). Host 16/16. New `run_prepared_reshape_cb_sim.sh` runs pads 0, 3 and 7 on V75: 6/6. In the smoke data, source rows 0–5 are identical, so only pads 6–7 actually test row selection (shown with a mutation check).
- Open: Sign CH stack/O(n^2) validation (20 KiB stack, repeated in run). CH Mul, ResizeNear and Transpose are still uncovered. Runtime integration has not started.
