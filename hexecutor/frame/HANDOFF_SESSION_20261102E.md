# Handoff: 2026-11-02 session E (68.1 -> ~46 ms, est. ~42 ms at HEAD)

Continues `HANDOFF_SESSION_20261102D.md`. The goal is Model-level latency (~10.5 ms). Exactness policy is
unchanged: bit-identical to the hw oracle (`--expect hw`) first, with a QNN-like non-exact mode later.
**The user asked for primary-model-only gating** (`ab.sh -p`). The alternate model was not run this session.

## 0. First thing to do: verify HEAD on the Frame

HEAD `d52319b` has **not run on the Frame as a whole**. The Frame went offline (ssh timeout) just
after its engine was built. Each commit below passed `ab.sh -p` on its own. However, the last two (`02322f5`
MUL, `d52319b` transpose) were each gated **on top of `a6201bd` separately**; my gate worktree had not been
moved to the MUL commit before the transpose run. They touch disjoint files and the host gate passes on
HEAD, but run this first:

```sh
cd depth-model
sh frame/bench/ab.sh -p -t head l2hw    # primary smoke + random, repeat 2; expect "passed": true
```

Expected: about **42 ms** steady (46.4 ms measured without the MUL commit, minus about 4.2 ms the MUL commit
saved when gated on its own). If it fails, gate `02322f5` alone, then `d52319b` alone, to find the culprit.
If the Frame was rebooted, recover it first (section 6).

## 1. Commits this session (all `ab.sh -p l2hw` exact on the Frame)

| Commit | Change | Steady ms (primary) |
|---|---|---:|
| `9891f4b` | (start: session D) | 68.1 |
| `4a3f4eb` | Preflight digest on HVX lane sums. The reuse check re-hashed ~716 KB of tables with latency-bound scalar DDR loads: preflight 2.35 -> 0.08 ms | ~66.3 |
| `f9adb0d` | Fused u16 HMX conv staging (`n16_tile_all`: each input row loaded once, all taps built from it): u16 staging 4.2 -> 2.5 ms. `ab.sh -p` added | ~65.8 |
| `43f6801` | `QDS_POOL1_NOCOPY` (on in `l2hw`): no 7.5 MB pool-1 VTCM copy in/out (1.9 + 0.2 ms) | ~64 |
| `aff2009` | HVX column min/max for `conv_scale_from_weights` (32/21) | ~62.4 |
| `da74f8f` | Whole-block HVX linearclip u8->u16, linearclip u16, sign (agent) | ~59.4 |
| `a6201bd` | New `prepared_datamove2.inc`: D2S, dequantize, gather, slice-pad block kernels (agent) | ~54.4 |
| `02322f5` | Whole-block 16-bit u16 elementwise (REQUANT_U16, ADD1111, ADD_U16, ADDSUB, MUL, MUL1111), MUL_INLINE/72 HVX handler, no divides in `valid4` (agent) | ~50.3 (on a6201bd) |
| `d52319b` | Transpose (W<->C, H<->W), S2D and resize: unrolled ladders, no 64-bit divides, `dm_clear` instead of quadratic touch checks (agent) | ~46.4 (on a6201bd, without MUL) |

Tried and reverted: a rolling staged window for the 3x3 slide conv (6/33). It stays exact on silicon but
is not faster (9.3 -> 9.8 ms, noise). 6/33 is HMX-bound, not copy-bound.

`QDS_POOL1_NOCOPY` relies on pool 1's starting contents not mattering. Every byte the schedule reads from
pool 1 is written earlier in the same run. The host check filled pool 1 with 0, 0x37, 0xff and random bytes
before a primary run, and all matched. **Re-check on the alternate model before relying on it there.**

## 2. Where the time goes now (Frame, `l2hw`)

The data is the transpose gate run (`a6201bd` + `d52319b`) with the MUL-commit rows substituted from its
own gate run, so it is an estimate of HEAD. The fixed costs are measured: VTCM in/out 0, preflight
0.09 ms, dispatch 1.0 ms. Variant number = command.variant (1-based index into `resident_schema.VARIANTS`).

| Bucket | ms |
|---|---:|
| Conv (6/33, 6/35, 6/36, 6/32, 6/34) | 19.3 |
| Non-conv (~70 variants, ~2,700 effective commands) | ~20.4 |
| Fixed (dispatch 1.0, preflight 0.1, lease/other ~1.5) | ~2.5 |
| **Total** | **~42** |

### Convs (19.3 ms)

| op/var | kernel | calls | ms | us/call | notes |
|---|---|---:|---:|---:|---|
| 6/33 | `ConvLayer_s1.opt@CB*2.Xwb.Fi.t` u8 3x3 slide | 629 | 9.89 | 15.7 | HMX-bound; setup share large |
| 6/35 | `ConvLayer_s1.opt@CH*2.Xwb.Fi.t` u16 3x3 | 70 | 3.12 | 44.6 | 2.58 ms is CPU staging (im2col of 9 taps) |
| 6/36 | `ConvLayer_s2.opt@CB*2.Xwb.Fi` stride 2 | 80 | 3.04 | 38.0 | old native path |
| 6/32 | `ConvLayer_s1.opt@CB*2.Xpb.Fi.t` dynamic weights | 217 | 2.85 | 13.1 | not on the slide path (`bcopy`) |
| 6/34 | `ConvLayer_s1.opt@CH*2.Xpb.Fi.t` | 40 | 0.37 | 9.3 | |

u8 conv phases (summed over u8 convs): setup **3.79**, MAC/HMX wait **5.27**, staging 1.27, output 1.24 ms.
u16: staging **2.58**, MAC 0.22, output 0.22 ms.

### Non-conv, top rows (~20.4 ms total)

| op/var | kernel | calls | ms | us/call |
|---|---|---:|---:|---:|
| 7/31 | `ConvLayer_SpaceToDepth_stride2.tcm@CB*2.s4` | 174 | 2.16 | 12.4 |
| 4/30 | `ConvLayer.opt.weights_to_vtcm@Xwb.xwb.` (DDR constants -> VTCM) | 64 | 1.35 | 21.1 |
| 24/51 | `ResizeNear_slice.tcm_special@CB*2.fi*2` | 107 | 1.19 | 11.1 |
| 23/37 | `Decimate.tcm@CB*2.s4` | 13 | 1.10 | 84.2 |
| 18/55 | `Transpose.2D@CB*2.t` | 82 | 1.06 | 12.9 |
| 18/56 | `Transpose.2D@CH*2.t` | 18 | 0.93 | 51.9 |
| 16/49 | `Reshape.hw@CB*2` (`copy_runs`, untouched) | 80 | 0.82 | 10.3 |
| 16/41 | `ForceFormat_Crouton@CB*2` | 70 | 0.69 | 9.9 |
| 35/23 | `convert_weights_to_signed.packed` | 80 | 0.64 | 7.9 |
| 5/24 | `inconv_im2col.tcm@CB.FB.s4.t` | 80 | 0.57 | 7.1 |
| 18/57 | `Transpose_impl@FB*2.fi.t` | 10 | 0.55 | 55.3 |
| 40/11 | `Abs@CH*2` | 4 | 0.50 | 124.6 |
| 10/13 | `Add.tcm@CB*3` (ADD_U8) | 170 | 0.50 | 2.9 |
| 16/45, 16/44, 16/43, 16/42 | `ForceFormat_Flat_c2f` / `Crouton_f2c` | 111 | 1.75 | 5-49 |
| 33/66, 38/19, 37/22, 34/20, 36/26 | dynamic-weight scale/bias chain | 400 | ~1.4 | 3-5 |
| 16/50 | `Reshape@FB.CB` | 80 | 0.37 | 4.6 |
| 8/70 | REQUANT_U8 | 160 | 0.31 | 1.9 |
| MUL/ADDSUB family after `02322f5` (41/72, 14/74, 9/71, 13/59, 12/60, 11/14) | | 216 | ~0.9 | |

The remaining ~6 ms is spread over ~40 variants, each under 0.35 ms, and is mostly per-call overhead.

Regenerate this table from any report:
```python
import json,sys; sys.path.insert(0,'frame'); import resident_schema as rs
x=json.load(open('frame/build/ab/TAG/l2hw_primary.json'))['runs'][-1]
for r in sorted(x['handler_profile'],key=lambda r:-r['run_ms'])[:40]:
    print(r['opcode'],r['variant'],rs.VARIANTS[r['variant']-1],r['count'],round(r['run_ms'],2))
```

## 3. What to do next, by expected gain

Model runs the same 5,068-command schedule in ~10.5 ms. We are at ~42: ~11 us per effective command
against Model's ~2.8 us.

**1. Conv per-call setup (u8 setup 3.8 ms over ~1,000 calls), ~3 ms.**
- Covers Fi building, `near_copy`, ktab, memo lookups and per-call checks in `hmx_conv`/`hmx_slide`/`hmx_native`
  (`prepared_hmx_conv.inc`).
- Most of it is constant per command: Fi words, ktab, staged/aligned verdicts. Precompute once per command
  into a per-command cache (the `Memo` already exists) and only copy into near/workspace per run.
- Profile with `HMX_MARK` slots first.

**2. 6/33 slide conv (9.9 ms), ~3-4 ms.**
- The MAC phase is HMX wait. Options:
  - Overlap per-strip staging and setup with HMX execution: issue strip n+1 while strip n runs. HMX ops
    are async until `finish`.
  - Raise the work per HMX call (more output columns per `slide3_n`).
  - Fuse the epilogue (fused calls go through C blocks plus an HVX pass; the output phase is 1.24 ms).
- Silicon only: bisect any HMX mode change with `SLIDE_LIMIT=1` (session D, section 2).

**3. u16 conv 6/35 staging (2.58 ms), ~2 ms.**
- Each call is split into ~3 workspace chunks, and each chunk pays a parallel dispatch plus an HMX round trip.
- Options:
  - Double-buffer A so staging of chunk i+1 overlaps HMX on chunk i.
  - Or decode QNN's `hmx_convhbh` window addressing (roadmap B3) so the 9-tap im2col disappears, as the u8
    slide did.

**4. Stride-2 conv 6/36 (3.0 ms) and space-to-depth 7/31 (2.2 ms), together ~3-4 ms.**
- S2D feeds the stride-2 convs. Decode `hmx_v73_convbbb_stride2` (session D, roadmap B2) and read S2D's
  source directly.
- Short of that, 7/31 is still 12 us/call for ~133K outputs per call.

**5. Dynamic-weight chain and 6/32 (2.85 + ~3.5 ms of prep: 4/30, 35/23, 32/21, 33/66, 38/19, 37/22, 34/20, 36/26), ~3 ms.**
- Fuse the per-bmm prep into one handler (`QNN_RESEARCH.md` win 3). Each op is 3-8 us of mostly
  `t_read`/`anew`/`t_write` overhead on tiny tensors.
- Let `hmx_slide` take copied weights (B4) so 6/32 gets the slide speed.
- 4/30 is DDR-bandwidth-bound (static weights from pool 5 to VTCM, up to 147 KB each). Prefetch it with DMA
  during the previous commands, or keep it resident if VTCM space allows.

**6. Format/reshape movers (~4 ms: 16/41-16/50 ForceFormat/Reshape, 23/37 Decimate, 18/57, 24/51, 40/11 Abs), ~3 ms.**
- Same treatment the agents gave D2S and transpose: whole-block HVX, hoisted lookups, no 64-bit divides,
  `dm_clear` not `dm_touch`.
- Remaining per-call overhead in data movers: `dm_clear` (~33K sim cycles on big calls) and `xfer_check`
  (~16K) in `prepared_reference.c`. Memoise them per command, as the conv path memoises `conv_checked`.
  This is safe because shapes and block tables are fixed after bind.

**7. Dispatch (1.0 ms) and handler overhead.**
- 3,769 effective commands cost ~0.27 us each in the executor loop: decode, `fill_call`, `select_scratch`, fuse
  lookup, profile clocks.
- Precompute a compact run list at bind time. Also measure with `QDS_PREPARED_PROFILE=0`: the profile clocks
  themselves cost.

**8. Concurrency (roadmap D), the last ~2x.**
- Run independent HVX commands on the follower threads while worker 0 waits on HMX.
- Try a 4th HVX thread (`QDS_PREPARED_THREADS=4`, unverified on Valve's DSP stack).

**9. Non-exact QNN-like mode (roadmap E), after the above.**

## 4. How this session worked (keep doing this)

- **Gate on hardware, primary only:** `sh frame/bench/ab.sh -p -t TAG l2hw` (primary smoke + random, ~4 min). `-q` is
  smoke only (~2 min). Commit only what passes exactly.
- **Clean gate worktree:** `/Users/owenelliott/Documents/projects/spatial-stuff-wt/gate`, a detached worktree used to build exactly
  HEAD plus one patch, while agents edit the main tree.
  - **Before each gate:** `git -C ../spatial-stuff-wt/gate checkout -f --detach $(git rev-parse HEAD)`.
    Use main's HEAD, not the worktree's own `HEAD`; that slip is why section 0 exists.
  - Its `.work/hexagon-sdk` is a symlink to the main checkout's SDK, and `frame/build/resident-*.qdrx` and
    `native-smoke` are copied in.
- **Committing one agent's hunks while others edit the same file:** build the patch in the gate worktree,
  then `git apply --cached patch && git commit`. This leaves the other agents' working-tree edits alone.
- **Subagents (Sonnet):**
  - Kernel work with the ~3 s host gate as the inner loop.
  - At most one small, focused simulator container for the exactness test and cycle numbers.
  - **No** `run_prepared_reference_sim.sh ... segments` runs and no whole-program sweeps. The user's CPU
    was saturated by these.
  - Primary model only. The main session merges and does all hardware testing.
  - Do not let agents `docker kill` everything: one did, and it killed other builds.
- **Disk:** ~57 GB free after the user cleaned up. An agent worktree costs ~1.5 GB, and the Agent tool's
  worktree base was a stale `origin/main`, 32 commits behind. Don't use `isolation: worktree` without
  checking the base commit.

Useful tools added under `.work/` (not committed):
- `.work/setupbench/vb.sh "CFLAGS" var...`: single-call sim pcycles per variant. varbench now loads pool 5 constants
  and has `VB_SEEN`/`VB_REP`.
- `.work/setupbench/vbprof.sh`: hexagon-sim `--profile` + gprof (only the flat profile works).
- `.work/setupbench/seq`: dumps the command sequence with tensor shapes for a command range.
- `.work/setupbench/shapes`: shapes per variant.
- `.work/p1/t.c`: host harness with `P1FILL=0|1|255|n` to pre-fill pool 1 (the NOCOPY check).
- Agents' tests: `.work/agent-lc/lc_test.c`, `.work/agent-mul/{fx_test,mulfast_host,bench}.c`,
  `.work/agent-tr/`, `.work/agent-d2s/`.

## 5. Hazards learned (add to session D, section 8)

- **Scalar loads from the Frame's DDR are latency-bound (~190 cycles per line, nothing caches them outside the
  lease).** Any per-run scalar pass over DDR tables costs milliseconds. Use HVX with several loads in
  flight (`prepared_vcopy.h` pattern).
- **A gate only covers what was built.** Always reset the gate worktree to main's HEAD (section 4).
- Data-mover "fast paths" can be slow: the old transpose fast path cost ~300 cycles per vector, from 64-bit
  divides, quadratic `dm_touch`/`dm_disjoint` checks and runtime ladder loops. `DMPATH` showing FAST says
  nothing about speed.
- The profile names in session D's tables were partly off (e.g. 23/37 is `Decimate`, not convert-weights).
  Use `resident_schema.VARIANTS[v-1]`.

## 6. Frame recovery (unchanged)

After a reboot or CDSP hang:
1. `ssh frame 'echo frame1 | sudo -S systemctl reboot'`, then wait for ssh.
2. `cd ~/frame-dsp/depth-model/wave1-20261101 && echo frame1 | sudo -S insmod frame/kmod/cdsp_mem.ko size_mb=128`
3. Run `/home/steamos/frame-dsp/depth-model/.venv/bin/python frame/bench/chunk_map.py` **as steamos**; expect `n_bad 0`.

The Frame drops off Wi-Fi when the user is away. Then do host and sim work only, and queue silicon tests.
