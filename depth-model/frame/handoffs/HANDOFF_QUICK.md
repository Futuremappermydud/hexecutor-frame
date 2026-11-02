# HANDOFF quick worktree (items A, B, C, D), 2026-11-01

Worktree: spatial-stuff-wt/quick/depth-model. Nothing committed. Work stopped on request; the V75 focused sim run was still going (see "Not finished"). EXECUTOR.md section NOT written yet.

## A. SIGN 30/39 - code done, host-verified
- frame/native_program/prepared_elementwise.c: sign_validate/sign_run (glue sign_blocked.c + sign_levels.c), registered in register_elementwise_region (scratch region as other ew handlers). QdBlockedU16 name clash with elementwise_u16.h handled by #define to QdBlockedU16Glue + copy.
- Coverage (host cov tool .work/q/cov): 12/12 commands, 491520/491520 elements, both models. Variant 40 (FH flat, 1 cmd) stays on the existing prepared_sign handler.
## B. LINEARCLIP 39/67 and 69 - code done, host-verified
- 67: lc16_validate/run via linearclip_u16 apply_params; 30/30 cmds both models. 69: new frame/blocked_glue/linearclip_flat_u8.h (HVX flat loop, exact int expression), 9/9 cmds, 184320 elements. 68 unchanged (40/40).
- New: blocked_glue/linearclip_flat_u8_sim.c: PASSED in V75 sim (all 256 values, 17 multipliers, alignments, in place, guards).
## C. SPILL_FILL / CONCAT / slices - code done, host-verified
- prepared_spill_fill.c and k_spill_fill (prepared_reference.c): memmove -> qd_vcopy. k_concat chunk copy and k_output_slice (merged adjacent rows into runs) -> qd_vcopy. Direct-tensor xfer_copy rows merged into runs + qd_vcopy (this touches the shared xfer path; layout agent edits nearby in prepared_reference.c, expect a merge there).
- INPUT_SLICE (3/1) goes through t_write -> xfer_copy (now vcopy for direct dst). Coverage by construction: all 62+26 spill/fill, 3 concat, 2 input_slice, 6 output_slice cmds go through qd_vcopy (it falls back to memmove below 512 B). Sim pcycle gain is expected to be small (the sim hides the DDR cost).
## D. Preflight reuse + NOP strip - code done, host-verified
- prepared_executor.{h,c}: signature now includes program/bound identity; content digest (hash of geometry/tensor/command/refs/vectors/calls + expanded blocks) recorded only when reuse_preflight set; preflight itself returns early on a cache hit (no validation) across inferences; run consumes digest_fresh. no_effect flag is in the signature. Stats gained preflight_digest, digest_fresh, strip_valid, effect_bits[8192/32], stripped_by_slot[128] (Nprg2ExecStats is ~1 KB bigger; check QdPreparedState size on Frame). NOPs are skipped via bitmap without decoding.
- Safety caveat: reuse assumes validators read nothing mutable (true for reference chain, ew handlers: they re-derive in run). prepared_run in qd_graph_prepared.h was NOT changed: it still calls nprg2_runner_preflight then execute with reuse=1, so it benefits automatically, but the pcycle/ms effect is unmeasured on Frame. Host bench steady state, 5068 stub cmds: preflight(cached) 57 us + run 34 us (was full preflight each inference; sim pcycle numbers not taken).
- bench_prepared_overhead.c gained a "steady-state" line.

## Tests run (cwd depth-model), all PASS
- .work/q/build_host.sh (copy of datamove build_host.sh) then `.work/q/test_full {primary,alternate} {fast,mixedfast} both`: 8/8 PASS (lockstep + whole, emulator sha256). (Registered handlers there do not include elementwise; host cannot run HVX.)
- `sh frame/native_program/run_prepared_elementwise_host_sanitized.sh`: PASS incl. new sign/lc67/lc69 all-command acceptance + mutation rejection (run_all_new in test_prepared_elementwise_host.c; stubs file extended).
- test_prepared_executor (new cross-call reuse/digest/strip tests) and test_nprg2_plan_cache with args `frame/build/resident-primary.qdrx frame/native_program/examples/primary-v2.nprg frame/build/resident-alternate.qdrx frame/native_program/examples/alternate-v2.nprg`: PASS. test_prepared_io PASS. test_prepared_runner not built (needs Frame-only symbols).
## Baseline (pre-change, copy in .work/q/base, gate PASS 12/12): sim pcycles primary / alternate: 30/39 126.1M/126.1M; 39/67 32.7M; 39/69 7.83M; 26/8 1.76M / 1.44M; 26/9 1.24M / 0.95M; 19/15 2.73M; 3/1 0.48M; 27/4 1.0M. Parsed in .work/q/base.json.

## Not finished
- New script frame/native_program/run_prepared_ew_new_sim.sh (focused sim windows through the new handlers via new `tail <first> <dir> [window]` mode in test_prepared_reference_sim.c): flat-u8 kernel PASS; 29 window runs were still in flight, results unread (logs in .work/q/ewnew/*/*.log, outputs .work/q/ewnew.log). Check for PASS lines; a missing PASS = failure.
- Link lists in run_prepared_reference_sim.sh and run_prepared_elementwise_sim.sh edited to add linearclip_u16.c, sign_levels.c, sign_blocked.c; never compiled in Docker. build_resident_dsp.sh already lists them. Hexagon-side compile of prepared_elementwise.c (sign/lc16 glue names) is untested.
- Engine not built (scratch copy prepared at .work/q/engine/frame with line 5 patched; flags BUILD_FLAGS.json). No ELF/sha256.

## Remaining steps, in order
1. Inspect .work/q/ewnew.log for PASS on all windows; fix any fail.
2. Run gate: `SEGDIR=.work/seg SDK_MOUNT=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk sh frame/native_program/run_prepared_reference_sim.sh both segments 6 fast ew cycles` (expect 12 PASS); aggregate with `python3 .work/q/agg.py .work/seg after.json` and compare to base.json.
3. Build engine: cd .work/q/engine, `QD_SDK_ROOT=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk/6.6.0.0` plus BUILD_FLAGS.json env, sh frame/build_resident_dsp.sh; report ELF path + sha256. Check Frame stack/heap for the larger Nprg2ExecStats.
4. Append dated section to EXECUTOR.md with the numbers; rerun host tests after any merge with threads/fusion/layout worktrees.
