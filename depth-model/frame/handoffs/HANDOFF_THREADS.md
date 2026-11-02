# HANDOFF: 3-thread prepared executor (worktree threads, depth-model/)
Stopped on coordinator request. Nothing committed. EXECUTOR.md section NOT yet written.

## Files changed (all under depth-model/frame unless noted)
- native_program/prepared_executor.h/.c: Nprg2Parallel hook (max_parts, ready, run, order 0/1/2 serial permutations, counters), nprg2_parallel_parts/for, Nprg2ExecCall.parallel, Nprg2ExecRequest.parallel.
- native_program/prepared_runner.h/.c: nprg2_runner_set_parallel, request carries r->parallel.
- native_program/prepared_reference.h/.c: Nprg2ReferenceConv.par_near[2]/par_near_bytes; near_compute split into setup + near_rows(job, part) (rows split ys..ye); PrivPlan/priv_layout/priv_plan_slot (per-thread acc+row ring); G.par_slots validated in setup (disjoint from pools/workspace/near/scratch); PAR_CONV_MIN_MACS=2M. Only splits when output is direct-to-blocks (or host non-HVX).
- native_program/prepared_elementwise.c: ADD/REQUANT_U8, LinearClip, u16 ops split by pixel rows (dims[1]); PAR_EW_MIN_ELEMS=131072.
- native_program/prepared_datamove.inc: S2D, RESIZE, TRANSPOSE(HW u16) split by flattened outer loop; PAR_DM_MIN_ELEMS=262144. (flat transpose/copy/slice NOT threaded.)
- qd_graph_dsp.h/.c: QdsWork gets job_fn/ctx/parts/started/finished; follower() handles op 4; _Static_assert sizeof(QdsWork)<=0x200; export of thread record.
- qd_graph_prepared.h: QDS_PREPARED_THREADS flag, hook (prepared_par_run/ready, par_fault, 400 ms join deadline), helper slots, prepared_export_par at arena QDS_PROFILE2+4160 (=24640), counters cleared per run.
- build_resident_dsp.sh: QDS_PREPARED_THREADS 1|3 (3 requires REFERENCE=1).
- resident_dsp.py, bench/verify_prepared_full_model.py: parallel_profile decode/print.
- native_program/test_prepared_full_model.c: QD_PAR=0|1|2|3 (asc/desc/shuffled serial, 3 = real pthreads), QD_PAR_SLOT.
- native_program/test_prepared_reference_sim.c: args par/parrev/parshuf/thr (thr = real simulator hw threads 1,2, acquire_vector_unit each), SJ_MODE macro, stack high-water print, pool-diff detail.
- .work/ (scratch): build_host.sh, build_eng.sh, tail_sim.sh, eng1/ eng3/ copies, thr/ probes, logs/.

## Memory / sync design
- Part 0 = master, uses command's conv near 0x30078000 (32 KiB). Parts 1,2 = followers (tid1,2, parked in follower()).
- Helper conv slots: 0x300f0200..0x300f4000 (part1), 0x300f4000..0x300f7e00 (part2), 0x3e00 bytes each, inside work slot whose only occupant is QdsWork (<0x200). Elementwise region starts 0x300f8000. Convs whose per-thread plan does not fit 0x3e00 run single-thread (3x3 ci>=128 likely does; 6/35 and 6/33 coverage NOT yet measured - check how many fit).
- Shared read-only tables (taps, requant params) live in part 0's near; parts read them, write own acc/ring and distinct output block rows.
- Sync: master sets work->op=4, job fn/ctx/parts, bumps generation, runs part 0, waits done[1],done[2]==gen (400 ms). Timeout sets par_fault -> cancel; started!=finished in record shows a stuck worker. ready() requires workers_idle + lease valid; else serial. Part order independent (verified).
- Overlap: conv only splits when direct (blocked_overlap already false => output blocks don't overwrite input).

## Tested
- Host: `sh .work/build_host.sh` (EXTRA="-DPAR_DM_MIN_ELEMS=1 -DPAR_CONV_MIN_MACS=1" to force); `QD_NEAR=1 [QD_PAR=0..3] .work/full/test_full {primary,alternate} mixedfast whole`: all PASS, incl. real pthreads (QD_PAR=3), small slot 6000, alternate lockstep order 1. Default thresholds: 785 conv jobs split per schedule; dm/ew none above threshold on host path.
- Sim gate `SDK_MOUNT=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk sh run_prepared_reference_sim.sh both prefix 6 fast ew`: PASS both models (run before final edits to sim test only).
- Sim parrev (serial, forced thresholds) primary prefix: PASS.
- Engines built (threads=1, threads=3, flags BUILD_FLAGS_A + ELEMENTWISE=1 PROFILE=1) BEFORE the last one-line fix in qd_graph_prepared.h (conv init `{NULL,NULL},0` to silence missing-initializer warning): .work/eng1/engine_threads1.elf sha256 8132a85e...e663f3 ; .work/eng3/engine_threads3.elf 19ee01e9...ddd371 (full: 8132a85ee95ab39702548463195c5d21d5e7ee523ece0ff26bd88a6018e663f3 / 19ee01e90f6710213e35f2ddc246d63f7a71e8f388cd79d2cb71ce2343ddd371). Bootstrap identical both: 2e5fe2fb...417d6. MUST REBUILD after the fix (`sh .work/build_eng.sh 1|3`, copy ELF).

## Known issues / untested
- Sim `thr` (real simulator threads) first failed with pool mismatches: cause found = each sim thread needs acquire_vector_unit(HEXAGON_VECTOR_WAIT) (shared HVX context otherwise). Fix added in test; the last runs (.work/logs/thr_all2.log both prefix; .work/logs/tail_thr.log tail 4500 primary) were still running/unfinished: NO thr PASS yet. SJ_MODE 1/2 earlier runs were pre-fix (mismatch status unknown, mismatch lines absent in logs, rc=0 but no PASS line seen).
- IMPORTANT hardware question: on the Frame, do followers need acquire/own HVX context per thread? Existing legacy workers already run HVX, so likely fine, but unverified for these handlers.
- Real wins unmeasured; 6/35 and 6/33 per-thread plan fit not checked. Transpose 18/56 flat path, copy 16/43, sign 30/39, linearclip 39/67 (u16) not threaded.
- ASan elementwise host test, test_prepared_executor etc. not rerun after hook changes.

## Remaining steps
1. Rebuild host + rerun `thr_all2`/tail thr in sim; confirm PASS (not silent; docker stop incident 19:00-19:30 may have killed earlier runs).
2. Rerun sim gate (both prefix 6 fast ew) and test_prepared_executor / elementwise host tests.
3. Census conv fit in helper slots (6/35, 6/33); enlarge via smaller cols if needed.
4. Rebuild both engines, record sha256.
5. Append dated EXECUTOR.md section (design above + hardware watch list: arena 24640 record: fault==0, started==finished, not_ready/declined 0, jobs_parallel>0 means threaded; serial jobs>0 should be 0; timeouts; worker exit markers).
