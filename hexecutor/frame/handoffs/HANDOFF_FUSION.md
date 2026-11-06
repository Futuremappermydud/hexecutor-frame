# Handoff: conv epilogue fusion (item 7), worktree fusion, 2026-11-01
Work stopped on coordinator request. Nothing committed. A background sim job (.work/final_sim.sh) may still be running; check .work/final_sim.done.

## Pattern table (schedule analysis, .work/an*.py, .work/plan)
Fusable = conv (u8 CH 1x8xWxC full tile) -> REQUANT_U8/70 or ADD_U8/13, identical block tables, no unmodelled gap hazards.
| model | pattern | pairs | elements | raw conv output |
| primary | conv->REQUANT_U8/70 | 121 | 7.29M | 2 dead, 119 live |
| primary | conv->ADD_U8/13 | 89 | 4.83M | 72 dead, 17 live |
| alternate | conv->REQUANT_U8/70 | 122 | 7.36M | 2 dead, 120 live |
| alternate | conv->ADD_U8/13 | 94 | 5.15M | 76 dead, 18 live |
Live pairs still write the raw output (saving is only the readback). Other epilogues (9/71, 14/74, 15/73, 41/72, 39/68, 12/60) are u16 or not directly after conv: not done. ADD other operand must be computed before the conv (checked).

## Files changed
- native_program/prepared_fuse.{h,c} (new): block-level pool-1 liveness plan (pairs, dead flag, role[]).
- prepared_executor.{h,c}: Nprg2ExecFuse hook, call->epilogue/epilogue_dead, request->fuse; preflight activates pairs, run skips active epilogue commands; signature includes fuse.
- prepared_runner.{h,c}: nprg2_runner_build_fuse / enable_fuse; plan freed on close.
- prepared_reference.{h,c}: Fuse struct, fuse_decode (re-validates epilogue), rq_emit_fx (HVX), fuse_scalar, patches in fast_conv/near_compute/k_hmx_conv/tile_block, nprg2_exec_reference_fuse hook registered with conv keys; preflighted+epilogue skips dry pass; NPRG2_FUSE_POISON host option.
- qd_graph_prepared.h, build_resident_dsp.sh: QDS_PREPARED_FUSE flag (default 0, requires REFERENCE), trace[12]=pairs.
- test_prepared_full_model.c (QD_FUSE=1, spans pairs in lockstep, masks dead blocks), test_prepared_reference_sim.c (`fuse` option, tail takes optional count), run_prepared_reference_sim.sh, test_fuse_lanes_host.c (new).

## Built / tested (all from fusion/depth-model)
- Host build: `sh .work/build_host.sh` (adds prepared_fuse.c); `QD_FUSE=1 .work/full/test_full {primary,alternate} mixedfast both`: PASS lockstep+whole both models, 210/216 pairs active (74/78 dead). Also with -DNPRG2_FUSE_POISON (EXTRA=) and ASan/UBSan (primary): PASS. Mutants (rd+1, all-dead, swapped unpack) all fail as expected.
- Lanes: `.work/fuse_lanes {primary,alternate}`: all 256 q (requant, all m 0..65535, s 1..31) + all 65536 (a,b) pairs for every real pair and 3000 random ADD sets: 0 mismatches.
- test_prepared_executor, test_nprg2_plan_cache (ASan): PASS.
- V75 sim: `SDK_MOUNT=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk sh frame/native_program/run_prepared_reference_sim.sh both prefix 6 fast ew fuse`: PASS both (21/10 pairs, all live). Segment 3500..3999 (`sh .work/seg_ab.sh`), primary/alternate handler pcycles base->fuse: 140.69M->135.64M and 178.87M->172.77M (preflight 8.18M->6.17M, 9.42M->7.00M; fused spans counted under op0). Before the last small edit (skip dry pass when epilogue); rerun pending.

## Verification and completion status
- Final sim (.work/final_sim.sh): completed (.work/final_sim.done). Prefix 0..131 and segment 3500..3999 PASS on both primary and alternate; all seven pools identical to reference.
- Engine ELF: built from .work/engine-build with QDS_PREPARED_FUSE=1; sha256: 87f75636a13cfafd6ad358b16ff40f0c09c1387d7694e1ba0119bf09886d437e.
- Host ASan/UBSan: PASS on both primary and alternate models (lockstep and whole schedule).
- Lane oracle (test_fuse_lanes_host): PASS >722M cases, 0 mismatches.
- Documentation: EXECUTOR.md updated with dated "Conv epilogue fusion" section.
- Hardware run: pending on Frame CDSP (CDSP is currently in use).
