# HANDOFF layout worktree (perf-layout), 2026-11-01

Worktree: /Users/owenelliott/Documents/projects/spatial-stuff-wt/layout/depth-model. Nothing committed. Stopped on coordinator request.
Note: tree already had many uncommitted modifications from before I started (git status showed ~20 files); my edits are only the files below.

## Files changed by me (frame/native_program/)
- prepared_datamove.inc: new HVX helpers (dv_shuffk/dv_dealk/dv_ladder, dv_ldu/dv_stun/dv_ldn/dv_stup/dv_keep/dv_ror/dv_dealh/dv_shuffh), bk_blk, dm_visit/dm_extent/dm_clear (exact slot-bitmap overlap test, replaces dm_touch for tr_hw_fast), and new paths:
  tr_wc16_fast, tr_wc8_fast (blocked W<->C, perm 0132), tr_hw8_fast (u8 blocked H<->W), tr_flat_t8 (flat u8 0132), tr_flat_il (flat 0231 plane interleave), tr_fast_dispatch, copy_c2_fast (16/43 and 16/45), im2col_u8_fast (5/24), im2col_u16_fast (5/25). DM_COUNT ids 7..15.
- prepared_reference.c: hooks in k_transpose (tr_fast_dispatch), k_copy_values, k_im2col; DM_SHAPES diagnostic macro (SHP, host-only, harmless; can be removed).
- test_prepared_datamove_sim.c: FLAT_BASE ext region, want_id (DM_STATS counter based taken check), trb_case/copyc2_case/im2col_case/ftr_case2, cycles_new(). Not edited: run_prepared_datamove_sim.sh.
- Scratch (not tracked): .work/dm/build_host.sh (host build, OUT/EXTRA env), .work/dm/host_dm.sh (host datamove test).

## Status per item
- TRANSPOSE 18/56: coded. W<->C (perm 0132 u16) via tr_wc16_fast, H<->W via existing tr_hw_fast. Host DM stats: all 18 cmds primary on fast path (id 2 + id 7 = 1.47M elems of 1.47M). Coverage ~100% primary and alternate.
- 18/55 (u8): wc8 + hw8 coded; 100% by stats. 18/57 (flat u8 0132): tr_flat_t8, 18/58 (flat u16 0231): tr_flat_il (stat 14: 1.02M, 15: 0.33M).
- COPY_VALUES 16/43 and 16/45: copy_c2_fast, stats 10/11 = 163840 elems each = 100% of those cmds.
- IM2COL 5/24 (stat 12, 1.47M) and 5/25 (stat 13, 0.33M): 100%.
(Stat counts printed twice-doubled in the last run because lockstep+whole both ran; coverage = every command.)

## Tested (all PASS)
- Host full model: `cd depth-model; sh .work/dm/build_host.sh` then `.work/full/test_full {primary,alternate} {fast,mixedfast} both` : lockstep + whole PASS all 4 combos (run before the last flat_t8/flat_il additions: fast both models re-run PASS after them for `fast`; mixedfast NOT re-run after flat_t8/flat_il).
- Host datamove test: `sh .work/dm/host_dm.sh` : PASS 313 cases, 138 fast, 0 mismatches.
- V75 sim datamove test: `SDK_MOUNT=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk sh frame/native_program/run_prepared_datamove_sim.sh` : PASS 313 cases, 0 mismatches (~2m13s).
- Mutation check: 20 mutants (ladders, masks, overlap checks, tails, origins, stride) all killed (script was in the scratchpad, now gone; mutants were single string replacements in prepared_datamove.inc).

## Not done / half-done
- V75 sim cycles run (`... run_prepared_datamove_sim.sh cycles > .work/dm/cyc1.log`) was started in the background and may still be running; result not read. Host "cycles" tick numbers are meaningless (host clock).
- Gate NOT run: `SEGDIR=.work/seg SDK_MOUNT=<sdk> sh frame/native_program/run_prepared_reference_sim.sh both segments 6 fast ew cycles` (need 12 PASS lines + before/after per-op pcycles; baseline logs exist in /Users/owenelliott/Documents/projects/spatial-stuff-wt/survey/qdsim/segP and segA).
- Hexagon -Werror compile of prepared_reference.c/prepared_executor.c with and without -DNPRG2_SCALAR_REQUANT not run (datamove sim script builds with -Werror and passed, which covers prepared_reference.c once; scalar-requant variant untested). Reference: /Users/owenelliott/Documents/projects/spatial-stuff-wt/datamove/depth-model/.work/dm/engine.sh.
- Engine build (scratch copy of frame/, BUILD_FLAGS.json, QD_SDK_ROOT patch line 5) not done; no ELF/sha256.
- EXECUTOR.md section not appended.
- Hardware-ms expectations unmeasured. Concern: first host-only tick numbers are uninformative; sim pcycles for the new paths unknown, so a path may be slower than generic in the sim (the scalar generic path is underestimated by sim 3-6x, so judge by HW).
- Known risks: per-tile dm_clear bitmap (memset 1KB stack) cost per command; 5/24 and copy paths assume pool bases 128-aligned (checked via dm_pools_aligned).

## Remaining steps, in order
1. Read .work/dm/cyc1.log (sim pcycles; check no new path is slower than reference).
2. Re-run host `mixedfast` both models; remove DM_SHAPES block if desired.
3. Hexagon -Werror compile with and without -DNPRG2_SCALAR_REQUANT.
4. Run the 7 min gate with SEGDIR=.work/seg; compare to survey segP/segA per-op pcycles.
5. Build engine from scratch copy of frame/ (flags JSON, QD_SDK_ROOT), record ELF path + sha256.
6. Append dated section to EXECUTOR.md; do not commit.

## Addendum: V75 sim cycles finished (reference -> fast pcycles, per-call, real shapes; log .work/dm/cyc1.log)
wc16 u16 80x64x32 (18/56) 12.96M -> 0.62M; 80x32x64 13.0M -> 0.62M; 32x256x2/32 2.22M -> 0.78M; hw16 1.26M -> 0.06M
wc8 u8 1x64x128 (18/55) 0.45M -> 0.056M; 32x256x2/32 1.42M -> 0.32M; hw8 0.92M -> 0.043M
flat u8 80x4096 (18/57) 17.3M -> 1.11M; 256x80 1.10M -> 0.077M; flat u16 2x320x256 (18/58) 17.2M -> 0.07M
copy_values 16/43 5.11M -> 0.16M; 16/45 1.86M -> 0.19M; im2col 5/24 1.19M -> 0.031M; 5/25 1.56M -> 0.032M
No new path slower than reference in the sim. Step 1 of remaining steps is therefore done.
