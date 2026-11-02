# Handoff: 2026-11-02 session F (42.7 -> 29.0 ms)

Continues `HANDOFF_SESSION_20261102E.md`. Goal is still Model-level latency (~10.5 ms for the same 5,068-command
schedule); we are at **29.0 ms steady** (random runs 28.6), ~2.8x off. Exactness policy unchanged: bit-identical to
the hw oracle first; a QNN-like non-exact mode later, behind a flag. All gating this session was **primary only**
(`ab.sh -p l2hw`). **The alternate model has not been run since session D**: do that first (section 0).

## 0. First things to do

1. **Alternate gate on HEAD.** `sh frame/bench/ab.sh -t alt l2hw` (full: both models). Nothing this session was
   checked on the alternate model on silicon (host gates pass on both). `QDS_POOL1_NOCOPY` was also only checked on
   primary (session E).
2. **Concurrency bisect finished: exact, no gain** (section 3). Results:
   report dir `../spatial-stuff-wt/gate/depth-model/frame/build/ab/ovlp/`, log in the session scratchpad
   (`gate_ovlp.log`). Variants `base` (flag off), `o1` (`QDS_OVERLAP_LIMIT=1`), `o64`, `ofull`. If the Frame hung,
   recover it (session E section 6) before anything else.
3. `l2hw_short` "passed": false with 16 pool-1 mismatches is **expected** under `QDS_POOL1_NOCOPY` (the 2-command
   check doesn't produce pool 1's contents). Only the full-model reports are the gate.

## 1. Commits this session (each `ab.sh -p l2hw` exact on the Frame, gated on top of the previous one)

| Commit | Change | Steady ms |
|---|---|---:|
| `d52319b` | (start; session E's estimate verified on the Frame) | 42.7 |
| `bdd59de` | Dynamic-weight chain: dense in-place reads/writes for the tiny scale/bias tensors, direct convert_weights output. (Deeper/threaded `weights_to_vtcm` copy: no gain on silicon, DDR-bound.) | 41.6 |
| `9e1d60c` | Per-command memo of `xfer_check`/`dm_clear` verdicts; whole-block reshape/ForceFormat mover. Also sped up S2D and transposes. | 38.4 |
| `4f00611` | Executor run list built at preflight; `setup()` no longer memsets all of G; cached overlap checks (dispatch 1.0 -> 0.8) | 38.1 |
| `4085021` | Whole-block HVX Decimate (1.10 -> 0.07 ms); per-block S2D and resize work items | 36.7 |
| `51f1b7b` | Vector Fi build, once-per-call memo lookup. **Its Fi cache never hit** (keyed on pool 5; filter rows are in pool 1); see 18f4a46 | 35.6 |
| `18f4a46` | Per-command fused-epilogue cache: the 256-entry direct-mapped `fz_cache` made 120 of 210 fused convs evict each other and re-run `fuse_decode` every inference. Dead Fi arena removed. | 34.9 |
| `070aa94` | u16 conv 6/35: input pointer table + unrolled 3x3 staging (`N16_PT`), HMX writes straight to aligned output blocks (`N16_DIRECT_OUT`). u16 staging 2.8 -> 0.7 ms | 32.8 |
| `3682d9a` | 1x1 conv (6/32): direct per-block A staging, no row table, serial staging under 16 tiles (`NAT_1X1`) | 32.7 |
| `f5cae8e` | Stride-2 conv 6/36 reads raw 3x3 weights in place as B with K reordered (no per-call B copy / weight_tile scan): 2.71 -> 1.64 ms | 31.7 |
| `1556dc0` | Cheaper conv setup/decode (halfword geometry loads, one memo lookup per call, `is_class` without libc), HVX Abs (0.50 -> 0.05), direct Concat (0.30 -> 0.11), deeper Fill copy | 30.3 |
| `252e279` | Resize 24/51 row precompute, two-step flat transpose 18/57 (0.54 -> 0.11), whole-vector CB->FB movers (16/44, 16/50), flat->flat single move. **Memos moved to the heap** (see section 5) | **29.0** |

Tried and not committed:
- **`SLIDE_OVERLAP`** (6/33: stage chunk n+1 while HMX runs chunk n, new `qd_hmx_slide3_cont`). Exact on silicon,
  no hang, but slower: 6/33 8.82 -> 9.31 ms; even with the flag off, its helper-function refactor cost ~0.35 ms.
  6/33 is HMX-throughput bound. Patch: `../spatial-stuff-wt/a-slide/depth-model/.work/agent.patch`.
  Useful fact proven: the HMX accumulator survives HVX stores + barrier + HVX read-back between slide calls.
- **Heap-allocated Fi cache** (correct pool-1 key, all 709 static convs cached): exact, **zero gain**. Building Fi
  was never the cost. Patch kept in `a-conv/.work/agent-prev.patch`.

## 2. Where the time goes now (Frame, `l2hw`, HEAD build)

~29 ms = handler ~26.6 + dispatch 0.8 + fixed ~1.6.

| op/var | kernel | calls | ms |
|---|---|---:|---:|
| 6/33 | `ConvLayer_s1.opt@CB*2.Xwb.Fi.t` (u8 3x3 slide) | 629 | **8.40** |
| 6/32 | `ConvLayer_s1.opt@CB*2.Xpb.Fi.t` (**1x1**, dynamic weights) | 217 | 1.85 |
| 6/36 | `ConvLayer_s2.opt@CB*2.Xwb.Fi` (stride 2) | 80 | 1.59 |
| 6/35 | `ConvLayer_s1.opt@CH*2.Xwb.Fi.t` (u16 3x3) | 70 | 1.58 |
| 4/30 | `weights_to_vtcm` (DDR -> VTCM, DDR-bound) | 64 | 1.35 |
| 7/31 | `SpaceToDepth_stride2` | 174 | 0.86 |
| 24/51 | `ResizeNear_slice` | 107 | 0.69 |
| 35/23 | `convert_weights_to_signed.packed` | 80 | 0.56 |
| 18/55 | `Transpose.2D@CB*2.t` | 82 | 0.52 |
| 10/13 | `Add.tcm@CB*3` (ADD_U8) | 170 | 0.49 |
| 5/24 | `inconv_im2col` | 80 | 0.47 |
| 18/56, 16/45, 33/66, 6/34, 16/43, 26/8, 8/70, 32/21, 39/68, 14/74, 16/49 | | | 0.25-0.40 each |

u8 conv phases (all u8 convs): MAC/HMX wait **5.26**, setup 1.52, output 1.32, staging 1.25. u16: staging 0.68.
Regenerate with the snippet in session E section 2 (report: `frame/build/ab/TAG/l2hw_primary.json`).

**Conv overhead not in any phase bucket** (measured with `HMX_PROF=5` and `=9` builds, before 1556dc0): ~1.85 ms from
`nprg2_exec_reference_run` entry to `k_hmx_conv`, and ~1.7 ms from `k_hmx_conv` to `hmx_conv`. 18f4a46 and 1556dc0
removed ~1.2 ms of that. Re-profile before more work here: `HMX_PROF=k` moves sub-step k into the output slot
(prepared_hmx_conv.inc line ~35 lists the steps; 5 and 9 add the normally-untimed entry segments). Build all steps in
one `ab.sh -q` call: `p1:<l2hw set>+-DHMX_PROF=1 p2:... ` (session scratch has the loop).

## 3. Command-level concurrency (`-DNPRG2_OVERLAP=1`, off by default): exact on silicon, NO gain yet

Patch: `../spatial-stuff-wt/a-exec/depth-model/.work/agent.patch` (merges cleanly on 252e279; host gates incl. the
permuted-order tests pass on both models). Not committed.

**Frame bisect result (tag `ovlp`, all exact, no hang):**

| Variant | Steady | Random | First run |
|---|---:|---:|---:|
| `base` (flag off) | 28.7 | 28.5 / 28.3 | 123 ms |
| `o1` (LIMIT=1) | 28.8 | 28.9 / 28.9 | 398 ms |
| `o64` | 29.0 | 29.0 / 29.0 | 391 ms |
| `ofull` | 28.9 | 29.2 / 28.7 | 399 ms |

Conv u8 MAC (HMX wait) stayed 5.3 ms in every variant, so nothing was hidden under it. **Unknown whether any
command actually ran early**: `Nprg2ExecStats` has `nested_cmds, nested_cycles, polls, hosts_with_work`
(prepared_executor.h ~228) but they are not exported to the report. Next step: export them through the result/trace
words in `qd_graph_prepared.h` into the JSON, re-run `ofull`. If `polls` is ~0, the wait loops find HMX already done
on first check (poll placement is wrong: it must run before/independently of the first completion test); if polls
and nested_cmds are high but MAC didn't shrink, the nested work is serialising with HMX on worker 0 (then it has to
go to the follower threads instead). The first-run cost (+275 ms plan build) must also come down before committing.
- `nprg2_overlap_plan` (prepared_fuse.c): per conv, later commands with no RAW/WAR/WAW conflict with the conv, its
  fused epilogue, or anything in between (pool-1 2 KiB blocks + whole-pool masks). Built at preflight (~180 ms
  one-time on first run), stored in stats.
- At run time `NPRG2_OVL_POLL` in the four HMX wait loops runs the next listed command on worker 0 while HMX is busy.
  Nested commands get their own 256 KiB heap scratch; the host conv loses its near region while it has overlap work.
- Eligible: ~1,615 distinct commands (~4.8 ms) against ~5.3 ms of HMX wait. Agent's estimate: 2-4.5 ms saved.
- Host test modes: `test_prepared_reference_sim ... fast fuse ovl1|ovl2|ovl3 [parshuf|parrev]` run the listed
  commands early/late in permuted order. This caught two real dependency bugs. Keep using it.
- Knobs (compile-time, `QDS_EXTRA_CFLAGS`): `QDS_OVERLAP_LIMIT`, `QDS_OVERLAP_HOST_VARIANT`, `QDS_OVERLAP_WINDOW`,
  `QDS_OVERLAP_PER_HOST`, `QDS_OVERLAP_BLOCKS`. Bisect order: LIMIT 1 -> 64 -> unlimited, then per host variant.
- Risk: HVX work interleaved with in-flight HMX on one thread is unproven on silicon.
- `QDS_PREPARED_THREADS=4` is not functional: boot starts `htid<3` only, `QdsWork` has `[3]` arrays, two helper near
  slots, kernels ask for at most 3 parts.

## 4. What to do next, by expected gain

Kernel-by-kernel patching is now worth ~0.2-1 ms per item. The remaining 2.8x needs structural changes.

1. **Concurrency** (section 3). Exact on silicon but no gain yet; first find out whether anything ran early. If it can be made faster, commit
   with the flag on in `l2hw`, then tune WINDOW/PER_HOST/BLOCKS.
2. **6/33 HMX throughput (8.4 ms, 5.3 ms of it HMX wait).** Staging, setup, rolling windows and chunk overlap all
   gave nothing. Model runs these convs much faster, so QNN's issue pattern differs (larger ops per call,
   accumulator usage, fewer passes per output). Decode QNN's 6/33 kernel and test on silicon. This is main-session
   work: the ISS does not model HMX time and accepts things that hang silicon.
3. **Fixed per-command cost.** ~5,000 commands: 1 us each is 5 ms. Fold data movers into their neighbours or
   delete them from the schedule at load time (e.g. reshape/ForceFormat pairs that cancel, S2D feeding only a
   stride-2 conv, `weights_to_vtcm` for weights that could stay resident). Nothing has attacked this yet.
4. **4/30 `weights_to_vtcm` (1.35 ms).** DDR-bound; deeper/threaded copy didn't help. Options: DMA prefetch during
   earlier commands, or keep static weights resident in VTCM if space allows.
5. **Non-exact QNN-like mode (roadmap E).** Possibly now the main lever: exact requant/rounding costs everywhere.

Do not redo:
- 6/32 is **1x1** (all 217 calls; H=W=1), never 3x3. Roadmap B4 (6/32 on the slide path) is void.
- S2D folding into the stride-2 conv: S2D is already whole-vector HVX at ~5 us/call; folding repeats the
  deinterleave per tap.

## 5. Hazards learned this session

- **The resident engine's 1 MiB private code region holds its static data too.** Link fails with
  `resident engine exceeds its private code region`. Per-command tables sized `MEMO_COMMANDS` (8192) x two Memos
  grew the engine from 505 to 950 KB. Fixed in 252e279: the Memos are heap-allocated on first use (engine now
  558 KB; the heap is 32 MiB with ~5 MB used). Put any new per-command or large table on the heap, never static.
- **`setup()` no longer memsets all of G (4f00611).** Every new G field after `in` must be reset explicitly in
  `setup_ex`. A merge broke on this (wild pointer) and the host gate caught it.
- **`varbench.c`/`vb.sh` register no HMX conv hooks**: convs run the scalar reference there (3000+ cycles/output).
  Several agents' conv numbers were invalid because of this. Use harnesses that reach HMX and check `conv_path`:
  `frame/bench/hmx_slide/slide_cmd_sim.{c,sh}` (replays weight-prep commands so pool 1 holds real weights),
  `frame/bench/hmx_slide/s2_cmd_sim.{c,sh}` (stride 2), `frame/bench/hmx_u16/n16_test.c` (u16).
- **Hash only the output region** in sim tests. Hashing all 7.5 MB of pool 1 per call made one run take 25+ minutes.
- **Direct-mapped caches keyed on `command_index & (N-1)` collide** (the fz_cache bug). Key per command via Memo.
- **Profile buckets can hide cost**: ~3.5 ms of per-call conv work sat outside every phase bucket. Use `HMX_PROF`.
- Sim speedups for memory-bound code overstate silicon (decode/setup: ~1.2 ms predicted, ~0.45 ms measured); for
  some HVX kernels they understate it (Abs 3.9x sim, 10x Frame; Decimate 6x sim, 16x Frame).

## 6. How this session worked

- Sonnet subagents, each in its own detached worktree at main's HEAD, with ownership boundaries per function to keep
  patches mergeable. Rules file the agents read: the session scratchpad's `common.md` (copy it forward; key points
  are in section 5 above). Host gate inner loop, one short focused sim run at the end, no segment runs, no
  `docker kill`, no Frame access, no commits.
- Docker: worktrees' `.work/hexagon-sdk` is an absolute symlink the container can't see; add
  `-v /Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk:/repo/.work/hexagon-sdk`.
- The main session merges each patch into the gate worktree (`git -C ../spatial-stuff-wt/gate checkout -f --detach
  $(git rev-parse HEAD)`, then `git apply --3way`), runs the host gate (both roundings, both models, `fast` and
  `fast fuse`), runs `ab.sh -p` on the Frame, and commits only exact passes. Gate one patch at a time on top of
  the previous commit.
- Host gate:
  ```sh
  N=frame/native_program; cc -O2 -w -DNPRG2_HW_ROUNDING=1 -Iframe -I$N $N/test_prepared_reference_sim.c \
    $N/prepared_reference.c $N/prepared_executor.c $N/prepared_fuse.c $N/nprg2_runtime.c -lm -o /tmp/hg && \
    /tmp/hg primary split 1 /tmp/hgd fast fuse | tail -1
  ```
- Custom Frame variant (must list the whole `l2hw` set):
  `NAME:QDS_PREPARED_THREADS=3,QDS_PREPARED_FUSE=1,QDS_PREPARED_HMX=1,QDS_L2_CACHE=1,QDS_EXTRA_CFLAGS=-DNPRG2_HW_ROUNDING=1+-DNPRG2_SLIDE3=1+-DQDS_POOL1_NOCOPY=1+-DX`.
  Several variants in one `ab.sh` call bisect in order and stop at the first failure.

Worktrees: `gate` (test copy), `a-s2d`, `a-fmt`, `a-dyn`, `a-exec`, `a-conv` (agent worktrees, all patches merged
except a-exec's concurrency patch), `a-slide` (SLIDE_OVERLAP, not committed), `l2` (branch `exp-l2-cache`, not in
main). Old worktrees removed this session; their uncommitted diffs are in `depth-model/.work/wt-backup/`.
