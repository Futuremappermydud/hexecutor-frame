# Current executor checkpoint, Oct 30 2026

## Latest live checkpoint after authorized reboot and startup fix

User approved reboot with “you can reboot”; root rebooted Frame at 19:03 EDT.
Do not restore the old retained snapshot over the fresh DSP. The pre-reboot
startup race left command header/seq allzero, QDO active, VTCM phase1, clocks/HMX
off, and missing follower exit markers. Root saved stable backup SHA
59ea4939ca49afe95d6d1756879473783f41c2d762f9aef9925b3a0e19a1756e,
verified its host fingerprint, then waited for explicit user permission.
Evidence: `retained-startup-20261030/` in the evidence directory and Frame build.

Startup fix now LIVE-VALIDATED: transport defers callback registration until
prepare+fence inside the blocked-signal dispatch; distinguishes loaded vs
registered pages, and retains on unexpected active CTRL without a host job.
The guarded bootstrap validates magic/version/RUN, nonzero stable sequence,
range/direct flag and full config BEFORE mapping/saving; all three worker
arrivals must match the nonzero startup sequence before ownership changes.
An allzero start and invalidconfig hardware rejection test pass with NO mapping,
NO VTCM lease, NO clock/HMX activation. Host transport lifecycle11, retention2,
eye replacement4, C header/arrival checks pass. Loader remains unchanged.

Use the new preserving bootstrap for ALL future live tests:
`frame/build/qd_graph_boot-c0-preserve-vtcm-guarded-entry.elf`, SHA
628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147.
The earlier `...-native-entry.elf` is an identical-SHA alias. Old DA343f boot
is retained as historical artifact, not the new live selection. The builder
now writes the guarded separate bootstrap whenever preserveVTCM=1.

Native1x1 HVX-guard/full-tile implementation is LIVE-VALIDATED in primary seq1195
random and alternate seq1166 smoke/random: nativecall1, all outside bytes EXACT,
target [0,+1] only, guards/exiterrors0/fullVTrestore+hostfingerprint pass. Current
Native1-only engine SHA b1f641c35d3394330acde1d4394c2d5ad996343521b740138455fb6be0e5a7d4;
coreC SHA1f5fadce…; bridge4df158ce… . Prepare~32us, issue~4.5us, drain/scatter~23us
(previously277us then121us). Frozen65-file checkpoint
`native-1x1-hvx-guards-validated/sha256.json` contains the new live reports.

FULL-GRAPH Native1-only characterization COMPLETED both models smoke/random,
2calls/case: all140 native1x1 calls, finite/deterministic outputs, memory/restore
clean. Warm primary844.7–847.1ms, alternate839.7–844.7ms. Follow-up exactcontrol
random warm862.57/860.65ms; last random native846.30/842.73 => measured16.27/17.92ms
gain under this run order. This is a modest gain, not a near20ms runtime. Native
outputs are INEXACT: randomU8 MAE0.920/max15, floatMAE~0.063/max5.30 vsinteger;
smokeU8 MAE1.238/max14, floatMAE~0.318/max2.98. No QNN-equivalence claim. Keep exact
backend/defaults untouched. Reports `native-1x1-full-{first,control}.json`.

Native3x3 is NOW INTEGRATING under `QDS_NATIVE_3X3=1` (requiresNative1). CoreAPI
1MiB workspaceVT+6MiB and H8-family inclCO2max32 has5actualSIMfixture proofs,
no-write rejection tests, bulk/HVXguard optimizations. Frozen core7f6040e6…,
headerc7e01788…, bridge wasaf2f5a3b… before root's FI-byte fix. Entire metadata
census accepts510primary/524alternate, plus73/59 odd Y-slice cases are now proved
possible by new seq3627 test in separate candidate (don't merge before review).
First graph prefix247 safely DECLINED allnativecalls0: bridge Fi row shape had
been corrected but bytecount still used onlylastdim64. ROOT FIXED it to
Fi.m[1]*Fi.m[3]*esize (CO64 requires512B); rebuild session82950 is in progress.
Initial Native3 engine21d80352… is NOT a successful native proof; all bytes
matched because legacy ran. Next: upload rebuiltengine, rerun primary247 strict
native3count1/exactoutside/[0,+1]inside using prefixvalidator --native-family3x3.
Then characterization smoke/random if clean. Native3 gdiag8 atresult+592 and
markerresult[14]=0x33584e51; Native1 remains+520/result13.

Current specialists: spatial kernel expert (odd Yslice proof + actual Cbridge
eligibility test), layout expert (native stride2, exactCPUlowering seq224, HMX
ABI/orientation unresolved), u16 expert (exact providerhelper now runsSIM and
emitsrealCHdescs/cfg; flags3 selectalignedV75leaf, mode/FI semantics unresolved).
Native1 expert completed separate paddedCI12/20→CI32 proofs (seq115,3416), no
bridge expansion yet. Native-program expert completed C NPRG2open/mockbind;
root optimized bound tensor rows to resolvedblockindex so command access reads
hotmetadata only, added alias checks; bothactualmodels strict+ASan/UBSan pass.
No real L2/VT program binding or fullnativecoverage yet. Pool1 remainsDDR.

CURRENT HARDWARE: reboot recovered clean; cdsp_mem is LOADED128MiB, latest gate
`native-entry-chunk-map.log` n_bad0. Subsequent rejection/prefix/fullgraph/control
sessions exit clean; check state before nextlive call. No active job known now.
Root sole hardwareowner. Unload only after safe finalclose; don'treapply old
backup or rebuild loader. No firmware/service/installedmodel changes.


## Native-runtime continuation (current, supersedes older native status)

User explicitly prioritized reproducing Qualcomm's native execution behavior and
speed, and authorized `gpt-6-luna` specialists. Root alone owns shared integration
and all live devices. Three specialist slots rotate among spatial kernels, native
1x1 integration, quantized u16, native-program metadata/runtime, memory and glue.
Seven domain specialists have worked so far. Do not request fresh authorization
for this work. The 20 ms target and a complete native runtime remain unproven.

**Keep the validated exact artifact:** c82fd120… (`im2col-validated/sha256.json`),
866.35/867.03 ms warm on the two models. Recent same-artifact runs with active
VR/Godot varied to about 903/946 ms; use paired stage measurements. Default
engine/boot/probe/loader and installed paths remain unchanged. Current sources
also contain opt-in experiments; use frozen sources to reproduce c82 exactly.

**Live 1x1 graph integration passed:** `QDS_NATIVE_1X1=1` requires the preserving
boot and uses the original QNN leaf, raw packed Xpb/Xwb and Fi, upper-VTCM
workspace, worker quiescence and HMX ownership gates. Shared worker keys are
invalidated around native calls. Initial experimental ELF SHA
`0a666078cfacb8973f2fcbd85f45ed44aa3e1be82ca129f34be32a8c492c1c74`.
Primary random prefix through sequence1195/index983 passes: all five pools exact
outside the native output; output 64,040 exact and 1,496 +1, no other deltas across
65,536 bytes. One native call, no failures, all guards/exit errors and full VTCM
restore plus host fingerprint pass. Exact legacy prefix control also passes all
pool bytes. Prep32.22us, leaf issue4.56us, drain+scatter277.16us. The prefix remains
~281ms because 355 earlier calls still use the old backend; this is not a whole-
graph speed claim. Frozen 51-file build/proof set:
`.work/frame-direct-kernels-20261030/native-1x1-graph-first/sha256.json`.

Adapter full-tile scatter optimization is now offline verified (nine poisoned
CPU-prefix fixtures plus masked H2); bulk 2048-byte writes replace tiny per-lane
DDR writes for fully active tiles. Core SHA `1ccd5f96…`; bridge SHA `4df158ce…`.
Root's rebuild is in progress, paired live prefix testing is NEXT. Adapter/core
are frozen for that test. Primary first native sequence1195; alternate1166.
`frame/bench/verify_native_resident_prefix.py` validates exact control or one
native call, zeros mutable pools to the CPU initial state, and accepts 0/+1 only
inside the exact target write footprint. Its three host tests pass; combined
prefix/transport/probe host checks total15 pass.

**Actual spatial 3x3 hardware proof passed:** sequence1022 CI32/CO32, all65,536
physical bytes exactly match SIM SHA865cee388e…; integer oracle63,095 exact /
2,441 +1, guards/echo/staged hashes/TLB/full512KiB VTCM restore pass. Later vector-
staged probe: staging260.55us, leaf+drain25.42us; diagnostic hashing5332.23us and
publish/config28.03us are separate probe overhead. First naive-copy leaf54.35us;
other vector-staged leaf24.97us. Reports `xwb1022-native-frame-*.json`, latest
frozen manifest `spatial-1022-vector-staging/sha256.json` in the same evidence dir.
A completion timeout was fixed: `qd_hmx_finish` reads all64 Fi int32 words; bias
words32..63 must be initialized to zero. One-hot SDK SIM independently proves
those words affect completion. Reserve a full2048B completion crouton.

Spatial CI64/CO32 and CI64/CO64 all-nine-tap, both-depth/cross-output-channel SIM
basis proofs pass. Descriptor+4 is channels/32, not constant1; Fi is interleaved
per CO32 `[mult32,bias32]`, not global multCO/biasCO. Actual CPU-prefix SIM proofs
also pass seq247 CI64/CO64 (64,863 exact,673+1), seq874 CI128/CO64 (64,024/1512),
seq3527 CI256/CO128 (64,026/1510), seq3589 CI128/CO128 (64,154/1382), all65,536B.
See `frame/native_kernels/actual_xwb_spatial_ci_coverage.md`. CO2/max32 padded
physical proof and a generic bounded 3x3 adapter are being developed offline.
Only CI32/CO32 has this actual spatial hardware proof so far.

**U16 remains unresolved:** captured actual CH5099 (input zero point0), rawXwb/
Fi and independent int64 oracle match CPU4096 logical words; need native proof
for every paddedCO32 output and guard. Generic hbh leaf confirmed, calling args
and packed control stream still under investigation. Hexagon register combine
HI:LO and postincrement semantics matter. SDK Shape<4> has 32-bit size_t fields:
dims at HEX0x4/8/c/10, max at0x14/18/1c/20, pad bytes0x24..27. Earlier guesses
of `{2,0}` from dtype enums, 0x702 tuple and context/cfg interpretation were
withdrawn. Leaf mode word0 is count then selectors; actual helper path suggests
0x70b but numeric tuple/live object still not recovered. Do not run CH on device.

**NPRG v2:** offline metadata shrank802KB→397,136/395,048B; hot/cold split hot
235,470B for primary. Three v2 and four v1 tests pass. L2 placement is still a
plan: old block cache/graph/tensor metadata regions and full-L2 convolution
worker must retire; stack high water/preservation and ownership need proof.
Native open/bind C foundation now delegated, offline only. Pool1 is still DDR
in the live engine. Binding pool1 to full8MiB VTCM requires all-command native
coverage and must never fall through to legacy VTCM scratch.

**Space-to-depth is parked:** first exact rewrite regressed44→101ms; disabled
behind `QDS_BLOCKED_S2D=0` default. Cached prepared profiled candidate passed8
smoke/random runs, stage~35ms, warm zero proof misses/mallocs; not full54 validated
or enabled in the accepted exact artifact. Native runtime takes priority.

**Hardware at this checkpoint:** cdsp_mem is LOADED128MiB, mapping gate
`native-prefix-chunk-map.log` n_bad0. Last native prefix session cleaned controls
[1,0,1], callback/loader/payload/arena zero and restored eye client. Recheck live
state before another job; root serializes every invocation. No reboot, firmware,
service config or installed model changes. Unload module after final clean
session before ending autonomous work. Do not rebuild loader SHA828ca063….




## Latest checkpoint: exact resident 866–867 ms, Luna team active

User approved **gpt-6-luna** (the available Luna model) and autonomous work.
Root owns all shared integration and all Frame/Model hardware. At most three
specialists run alongside root; completed specialists are rotated onto new
performance domains. Five Luna specialists have worked in rotation, with root handling integration:

- Native kernel expert: original QNN HMX leaves and spatial ABI recovery.
- VTCM/memory expert: exclusive lease, traffic/liveness census, copy/cache audit.
- Blocked glue expert: affine, conversion, sign and addition kernels.
- Native 1x1 adapter expert: actual QDRX geometry, raw weights/FI, staged adapter.
- Layout pipeline expert: input padding, fused im2col, then space-to-depth.
- Root: integration review, failure handling, reproducible builds and live tests.

The current active lanes are spatial kernels, the native 1x1 adapter, and
blocked space-to-depth. New files have separate ownership; specialists do not
use SSH/ADB or live DSP. Root found several concrete defects during review and
kept those prototypes off hardware until corrected.

### Performance and exact validation

The 9.7 ms figure is **Model**, not Frame. Frame's current exact resident graph
is still far from 20 ms. Its existing hybrid remains faster (~115 ms exact,
~101–103 ms inexact native). Baseline resident ~1.69 s agrees with DSP time;
Python/RPC is not the missing second. Layout, scalar memory access and staging
are the dominant port costs. Native hybrid HMX calculate time is ~6.2 ms,
separate from copy/sync/preparation; it is not isolated peak throughput.

| Validated resident checkpoint | Primary warm ms | Alternate warm ms |
| --- | ---: | ---: |
| Dynamic weights + L2 fallback | 1126.04 | 1103.07 |
| Direct affine/conversion | 1097.94 | 1068.23 |
| Direct sign | 1056.38 | 1030.92 |
| Vector spill/fill | 964.86 | 954.77 |
| Direct blocked add | 943.08 | 939.76 |
| Direct row padding | 896.07 | 897.26 |
| Fused u8 im2col | **866.35** | **867.03** |

Each integrated checkpoint independently passes **all 54 cases + two warm
runs**. The latest also passes cold random + three warm runs/model (medians
866.32/863.30 ms), and random full-prefix pool parity on both models:
primary 5068, alternate 5040; pools 1/3/4/5/6 all match CPU bytes. Guards,
exit_errors=[0,0,0], VTCM phase=2/error=0 and independent host backup
fingerprints pass. Default binaries and installed paths remain unchanged.

Measured improvements since the earlier dynamic/L2 checkpoint:

- Direct affine/conversion: 40 u8→u16 commands/model go ~34→7 ms; another
  14/19 u16→u16 commands qualify. Cache geometry only; use current tables and
  coefficients. Static touched-region alias proofs preserve padding and holes.
- Sign: evaluate the old double/floor expression's three outputs once, then
  compare lanes directly. Exact levels are **1,32768,65535**. Twelve large
  calls/model go ~49.6→1.2–1.4 ms; scalar case uses the dense selector.
- Spill/fill: 3.408 MB took ~93.5 ms through libc memmove. Disjoint, aligned
  spans entirely in uncached mutable DDR +16..32 MiB use HVX memcpy:
  **~0.79 ms** primary, ~0.62 ms alternate. All other moves retain memmove.
- Add: 170 direct blocked u8 calls/model avoid dense transfers. Stage
  ~35–38→25 ms. Four W pixels share each aligned 128-byte vector; masks
  preserve crops. Whole-model differences include ordinary timing variation.
- Input padding: 80 hot calls/model copy contiguous valid rows directly to
  output. Avoid one-byte memcpy calls and the dense output temporary;
  InputSlicePad profile ~46.5→0.75 ms. Other cases retain the old path.
- Im2col: 80 u8 calls/model keep the small input in L2, emit nine taps directly
  into cached blocked output, and zero the 23 padded depth lanes. Stage
  ~50→15.4 ms; 20 u16 cases remain ~5.6 ms. Explicit slice offsets, seven-pool
  bounds/physical alias checks, and resident HMX-key invalidation are required.

Host/ISA tests cover exhaustive sign values and add byte pairs, independent
metadata/origins/slices, tails, missing blocks, aliases and untouched bytes.
Input padding passes 5,308,416 crop/pad cases. Im2col passes normal and ASan
bounds/alias/zero-padding tests. Root fixed empty-intersection pointer formation,
unaligned predicated stores and missing pool/partial-overlap checks before
final builds. Independent read-only integration reviews found no remaining
concrete defect. Counters are nested: do not add them as disjoint wall buckets.

### Full VTCM lease and native kernel evidence

All **8 MiB** of VTCM can be temporarily borrowed under an exclusive paused-eye
lease with live snapshot/restore. Upper 0x530000 bytes passed page-tag,
independence, full restoration and eye recovery in three sessions. Static
firmware bytes do not match live contents, and live contents change between
eye sessions: always preserve the current bytes.

The preserving bootstrap snapshots to private DDR 96–104 MiB before followers
start, restores after all workers quiesce, checks TLB identity and a
position-dependent backup fingerprint, then checks every restored byte.
Host independently checks the fingerprint and records SHA256. Save ~0.227 ms;
restore/verify ~5.2–5.3 ms **once per session**, not per inference. Failure
explicitly retains mapping/backup/paused eye even after a job stops; transport
retention regression tests pass. This proves bounded borrowing, not permanent
ownership or concurrent eye operation. **Pool 1 remains DDR.** A native binding
must replace overlapping fixed VTCM cache/staging and track async lifetimes.

The original QNN 1x1 leaf passes 128 hardware fixtures ×100 poisoned, drained,
full-byte-checked calls at VTCM offsets 0, 3 MiB and 7 MiB, restoring borrowed
bytes. Hot ~0.34–0.73 us, cold ~1.85–2.87 us for these small fixtures; these are
not model times. Native arithmetic is simulator 0/+1, not exact. Offline
expanded suite has 512 fixtures and graph geometry/crop checks.

The newer actual-QDRX adapter is **unintegrated and not hardware-validated**.
Command 1195 with actual filter/FI passes 65,536 outputs (native-oracle delta
0/+1); all 140 candidate metadata records pass C preparation. Fixed workspace
reservation is 354,304 bytes. Representative cropped/sliced/runtime-FI proofs
and an isolated device probe are still required. Root caught slice-index,
preflight ordering, zero input-tail and crop-external output-padding defects.
Files: `frame/native_adapter/`. Spatial expert has source-proven Shape fields,
actual wrapper descriptor order and separate aligned/generic leaf probes;
**complete 3x3 still fails its oracle**. Do not run that leaf on hardware yet.

Memory census: required primary DDR/VTCM traffic 11,249,796 B; alternate
10,480,772 B. At optimistic bulk 10 GB/s, primary transfer floor is 1.125 ms,
excluding compute/setup/latency/contention and separate graph I/O. The 67 MB
possible dense-window staging estimate is separate local work once pool 1 is
VTCM; do not count it as required DDR traffic. Both schedules have a 512 KiB
explicitly unreferenced tail [7.5 MiB,8 MiB), but liveness analysis remains
planning evidence until hidden accesses and in-flight worker lifetimes are
proved. See `frame/vtcm_lease/CENSUS.md` and `frame/perf_audit/`.

### Reproduce and current hardware state

Build the latest exact experimental checkpoint:

```sh
QDS_DYNAMIC_TCM=1 QDS_FALLBACK_TCM=2 QDS_BLOCKED_GLUE=1 QDS_FAST_SPILL=1 QDS_PRESERVE_VTCM=1 sh frame/build_resident_dsp.sh
```

Engine `frame/build/qd_graph_dsp-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-vtcm-lease.elf`
SHA `c82fd1203a2c6172164ee78428e8f9886cc13c102c16ccd5817c4c9008dbfade`.
Select `--bootstrap frame/build/qd_graph_boot-c0-preserve-vtcm.elf`, SHA
`da343f4b6c1aeeca249c4e726b923ba0b6260ec35c3c4ba96d9913f55e2af68f`.
Diagnostic magic10 (`0x3a504643`) adds direct im2col count; reader supports all
older formats. Experimental builds never replace default engine/boot/probe.
Never rebuild loader SHA `828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572`.

Evidence and frozen sources/ELFs: `.work/frame-direct-kernels-20261030/`;
latest `im2col-validated/sha256.json`. Reports `resident-im2col-*` and earlier
`resident-input-pad-*`, `resident-blocked-add-*`, `resident-fast-spill-*`,
`resident-blocked-sign-*`, `resident-blocked-glue-*` (local and Frame build).
Probe builds: `sh frame/build_qnn_hmx_probe.sh` and
`sh frame/build_vtcm_upper_probe.sh`.

**Hardware currently clean:** cdsp_mem unloaded; controls [1,0,1], callback 0,
loader/payload/arena zero; eye client 81667 S. This supersedes older active
lease notes. No reboot, firmware/service configuration changes, or installed
v5/Qwen changes. Keep root-only hardware ownership. Future live tests must
recheck state and the mapping gate before private-DDR writes.

## Frame gap investigation: two memory fixes, 33–35% faster (Oct 30, ~15:10)

The 9.7 ms measurement is **Model**, not Frame. Frame's slow resident engine
is not evidence of a matrix-throughput limit. Baseline wall time and DSP time
agree (~1.69 s); Python/RPC is not hiding the missing second. Workers take only
~55 ms, while serial tensor access, conversions and scalar scratch work dominate.
The existing faster hybrid path is still ~115 ms exact / ~101–103 ms native
(native is inexact). Its native HMX calculate phase is ~6.2 ms, separately from
~8 ms copy/sync, ~6.7 ms A preparation and ~2.4 ms quantize/collect. Those phases
include per-tile synchronization and are not an isolated peak-throughput test.
Evidence for that hybrid breakdown remains `native-hmx-cm-full-ab.json`; it was
inspected, not rerun, in this experiment.

Added detailed timers to distinguish FI/metadata, input transfer, parameter
preparation, output processing, and two-plane fallback preparation/merge.
`resident_dsp.py` recognizes versioned counters and still reads older engines.
**Counters are nested:** do not add read/write to convolution phases, or malloc
to conv_weights, as if they were disjoint wall-time buckets.

Paired cold-random-first sessions, three warm smoke repeats per model:

| Engine phase | Primary warm ms | Alternate warm ms |
| --- | ---: | ---: |
| A: instrumented baseline | 1686.83 | 1698.99 |
| B: dynamic weights in local memory / reuse HMX packet | 1504.83 | 1473.76 |
| C: B + 16-bit fallback arrays in VTCM | 1169.73 | 1140.63 |
| D: B again, with new fallback timers but old DDR arrays | 1506.98 | 1483.05 |
| E: B + fallback arrays in L2 when they fit | **1126.04** | **1103.07** |
| A2: original baseline return | 1686.71 | 1697.05 |

E saves 560.79/595.93 ms (**33.25/35.08%**). Cold DSP execution improves from
~2.32 s to ~1.58 s; session/model construction is additional. A2 reproduces A,
so residual state or a clock drift does not explain the improvement.

1. **80 runtime weight tiles were unpacked from Xpb in DDR, copied into a
   second dense allocation, and packed back into the HMX format each run.**
   For the bounded 1x1, CI/CO-aligned geometry, the original Xpb bytes already
   have exactly the HMX B order. Reuse those bytes, compute dense/channel sums
   in lower L2 scratch, and retain a private stack copy of sums across worker
   execution. Dynamic contents are recomputed every time; no stale weight
   caching. Warm conv_weights drops from 183/213 ms to ~14 ms, packing from
   17/24 ms to <1 ms; **400 malloc calls per warm run become zero**.
2. **Ten 16-bit tiles fail the fused epilogue proof because postshift=16
   exceeds its supported maximum 15.** Their exact fallback extracts high/low
   bytes, executes two integer convolutions, and combines them in int64.
   The input planes and accumulator were in uncached DDR. In D, preparation
   takes ~176 ms and merging ~131–133 ms. VTCM reduces those to ~29/~19 ms;
   L2 reduces them to **~9.4/~4.3 ms**. Output correction/requantization also
   benefits. E uses L2 for all ten; each command's arrays fit below +512 KB,
   above BOTH the worker's rounded weight copy and raw-output staging. If they
   do not fit, use bounded VTCM, then the original DDR fallback. No arithmetic
   or rounding rules changed and no additional VTCM ownership was assumed.

Validation: B, C and E each pass **all 54 cases + two warm repeats**, including
both output hashes and allocation guards. All paired cold-random/warm runs and
A2 also pass. E records 80 dynamic TCM tiles and 10 L2 fallback tiles every run.
Build script shell syntax and Python reader compilation pass. Earlier CPU
random instrumentation check passes; changes to numerical paths are DSP-only.

These are **opt-in engines**, not a replacement for the faster hybrid or the
installed v5. Default engine, loader, bootstrap and ownership probe SHAs stay
unchanged. Reproduce the final engine on the Mac:

```sh
QDS_DYNAMIC_TCM=1 QDS_FALLBACK_TCM=2 sh frame/build_resident_dsp.sh
# QDS_FALLBACK_TCM=0: DDR baseline; 1: VTCM; 2: prefer L2, then VTCM/DDR.
# Experimental builds never overwrite default engine/bootstrap/probe files.
```

On a fresh, checked 128 MB module lease, run on Frame:

```sh
OPENBLAS_NUM_THREADS=1 ~/frame-dsp/depth-model/.venv/bin/python \
  frame/bench/verify_resident_graph.py \
  --engine frame/build/qd_graph_dsp-dynamic-tcm-fallback-l2.elf \
  --case random --repeat 3 --report frame/build/resident-gap-E.json
# Omit --case for the complete suite. Follow the normal module/map procedures.
```

Final engine SHA: `2c513b1df06eb53572bf9d10f474164071702761475f149a5d2e7abfbef9e0d5`.
C/VTCM engine SHA `617b7cec…`; the script reproduces the manually built
`qd_graph_dsp-local-work.elf` byte-for-byte. Reports on Mac/Frame:
`frame/build/resident-gap-{A,B,C,D,E,A2,dynamic-full,local-full,l2-full}.json`.
Local `.work/frame-gap-20261030/` contains logs, summary, source/ELF manifests,
frozen source snapshots and final-state evidence.

**Remaining architectural problem:** pool 1's 8 MB QNN VTCM schedule still runs
against DDR backing. Dense tensor conversion and HMX staging repeatedly move
the same data, and the resident executor does not overlap graph glue/DMA with
HMX. E still spends ~255–262 ms in tensor reads, ~105–106 ms in writes and
~114 ms in input packing (overlapping scope with kernel timings). Remaining
convolution variants take ~0.45 s; other graph work accounts for most of the
rest. The route to millisecond-scale execution remains native kernels consuming
the graph's blocked tensors directly, with a proven VTCM allocation plan and
DMA/worker overlap. These fixes demonstrate avoidable overhead; they do not
establish an apples-to-apples Frame-vs-Model hardware speed ranking.

End state: module unloaded, [1,0,1], callback and loader/payload/arena zero,
eye client 36699 state S. No reboot, firmware/service changes or installation.

## Paired C0/C7 ablation: exact, no useful gain (Oct 30, ~14:45)

- Fixed the experiment harness: `ResidentDSP(..., bootstrap=...)` and
  `verify_resident_graph.py --bootstrap` select and report the actual trampoline.
  `build_resident_dsp.sh` now writes `qd_graph_boot-c0.elf` / `-c7.elf` and matching
  named ownership probes. Only C0 updates the ordinary default filenames.
  Building C7 cannot silently replace the default C0 bootstrap anymore.
- C7 engine SHA equals the current C0 engine (`5fa45a82…`): the cache attribute
  is selected by the bootstrap, not live engine code. C7 bootstrap SHA
  `2d15bfa9c1ed5a08b86317ea9b05fd1f062fb99eaf282a04c60c4b4e846342ad`;
  C0 remains `51fef8d4…`. Code/constants/heap only; mutable pools remain C6.
- On one fresh flushed-module 128 MB lease, chunk-map gate clean; C0 / C7 / C0
  phases each ran cold random plus four warm smoke repeats per model.
  **30/30 exact**, 1046 calls / 916 HMX each. Warm primary/alternate medians:
  C0 A0 **1690.40/1699.14 ms**, C7 B **1692.85/1700.22 ms**,
  C0 A1 **1692.89/1700.00 ms**. Cold random ~2.32 s throughout.
  No C7 performance benefit is established; keep C0 default. This was a bounded
  random/smoke ablation, not a new full 54-case C7 acceptance run.
- Reports on both Mac and Frame: `frame/build/resident-c7-paired-{A0,B,A1}.json`.
  Local evidence/summary: `.work/frame-c7-paired-20261030/`.
- Loader, ordinary engine/bootstrap and ownership probe SHAs unchanged.
  `cdsp_mem` unloaded after clean exit. Final [1,0,1], callback 0, all three
  regions zero, eye client 27259 sleeping/running normally (state S).
  Installed v5 and Qwen unchanged; no reboot or firmware/service changes.

## Model QNN profiling transport checkpoint (Oct 30, follow-up)

**Superseded by the successful signed ExecuTorch route described below.** The
stock SDK and raw `nnOssQnnCreate` routes remain blocked; both original DSP-side
ExecuTorch programs now execute on the Model's vendor runtime.

### Successful vendor execution and profiling (Oct 30, ~14:40)

- New client `bench/model_metadata_qnn_probe.c`, build script
  `build_model_metadata_probe.sh`, runner `bench/verify_model_metadata.py`, decoder
  `bench/model_metadata_profile.py`. The runner verifies the exact vendor SHA
  fingerprints before using the recovered ABI. It stages only temporary files.
- `hexagon_open` opens the existing signed `libhexagon_skel.so` in a private
  user PD. `hexagon_nnExecutorchCreateModel2` accepts the preserved
  `models/{primary,alternate}/dsp.pte` **unchanged**. It uses the signed vendor
  QNN 2.25 runtime and Bolt package. No SDK skel or unsigned shell is needed.
  The raw `hexagon_nnOssQnnCreate(context.bin)` route instead fails because
  firmware lacks a DSP `libQnnSystem.so`; that is not a blocker for ET12.
- Full validation: **54 cases, 162 inferences**, all finite and repeatable.
  Warm RPC medians primary **9.7075 ms**, alternate **9.6549 ms**. Model
  creation is excluded (~0.25 s). The call includes RPC/host marshalling,
  input/output copies, DSP-side ExecuTorch and QNN execution; it is not isolated
  HMX time and does not measure a camera/outer ARM pipeline.
- **Hardware is not simulator-exact.** U8 is exact on 5/27 cases per model
  (smoke and the four flat cases), float on 0/27. Random U8 MAE 0.727, max 13;
  float MAE 0.0331/0.0330, max 5.709/5.712. Worst suite float max 22.312/22.321
  on w-2. These are raw model output units, not calibrated depth-quality metrics.
  Existing Frame native-v8 report hashes match neither head in any of the 54
  cases either. Preserve the exact backend; do not claim Model parity for it or v8.
- `hexagon_nnExecutorchRun3` with vendor profiling enum **1** yields 201/205
  logical QNN op records plus two ET wrapper records. It includes cycles,
  wait cycles, background overlap cycles, wait overlap, resource masks and
  up to ten critical background IDs. Decoder offsets were verified against
  the vendor's `parseNodeQnnProfileEvent` and `log_profiling_delegate` and SDK
  event IDs 5001..5005. These are logical op aggregates, not a tile-level
  timestamped HMX/HVX/DMA timeline. Enum 2 returned status 1 and is rejected
  by the final client. The earlier Run4 attempts also failed; use Run3.
- Smoke/random, both models, three profiled calls each: 12 more inferences,
  outputs identical to unprofiled runs. Profiling adds heavy overhead
  (~173–178 ms warm RPC); do not use that as ordinary inference latency.
  137/137/138/140 logical ops report overlap. Primary random: 4,816,903
  summed op cycles, 2,681,120 summed overlap cycles. Overlap sums are not
  additive wall time; do not convert them to milliseconds without a clock.
- Evidence: `.work/model-qnn-meta-20261030/{full,profile}-validation.json`,
  with per-case inputs hashes, raw Model and hash-checked CPU outputs,
  commands/logs, and decoded profiles. Earlier exploratory disassemblies,
  create-only and failure logs are kept separately in that directory.
- No firmware, properties, service settings, Frame module or executor binaries
  changed during Model work. `hexagon_getDspVersion` returned enum 6; this is
  not an independent physical-ISA decode. Runtime perf configuration remains
  unmeasured. Frame remained clean [1,0,1], callback/regions zero, eye client S.

Reproduce (from the Mac `depth-model` root):

```sh
OPENBLAS_NUM_THREADS=1 .venv-cpu/bin/python frame/bench/verify_model_metadata.py \
  --repeat 3 --report .work/model-qnn-meta-20261030/full-validation.json
OPENBLAS_NUM_THREADS=1 .venv-cpu/bin/python frame/bench/verify_model_metadata.py \
  --case smoke --case random --repeat 3 --profile \
  --report .work/model-qnn-meta-20261030/profile-validation.json
```

### Earlier stock-SDK transport attempts (historical)

- Frame state checked before work: controls [1,0,1], callback zero, loader/payload/arena
  zero, eye client sleeping. No Frame DSP work or build was performed.
- Root ADB is available on Legacy headset 2G0YC1ZF860CBG; SoC ID 549, machine ANORAK.
  Staged SDK 2.50 tools, both contexts, and synthetic smoke inputs in the private
  `/data/local/tmp/qd-qnn-20261030` directory. All attempts stopped at device creation,
  before loading the primary context. Alternate was staged only. **Zero inferences;
  no hardware timings or numerical parity results.**
- The live SDK backend automatically chooses V69, not V73. Initial attempt lacked
  the V69 stub; after adding it, unsigned-PD creation fails opening its shell.
  Inventory finds only signed `fastrpc_shell_3` for CDSP. Repeating with both
  DSP_LIBRARY_PATH and ADSP_LIBRARY_PATH including the private and vendor/CDSP
  directories does not fix unsigned creation.
- Explicit signed PD with the existing vendor shell succeeds in creating the PD.
  DSP_LIBRARY_PATH overrides the load search used here: include the private skel
  directory in it, not only ADSP_LIBRARY_PATH. Once found, the SDK V69 skel is
  rejected: `signature verify start failed for libQnnHtpV69Skel.so`, RPC 0x80000406.
- Actual vendor QNN DSP library `/vendor/lib/rfsa/adsp/libQnnHtpV69.so` is
  `v2.25.0.240728104910_97711`, SHA
  `9b687661f8a2b5c1511a21b324803aa96692a2c4f4a0f73ad18151562eb94d22`.
  Pulled this, its Bolt op package, libhexagon.so/shim/skel, libcdsprpc.so and
  the signed shell. MrSystemService maps libhexagon.so/shim and depth_engine;
  there is no separate host libQnnHtp mapping in the captured process map.
- Local evidence: `.work/model-qnn-20261030/manifest.json`, commands, logs/logcat,
  shell inventory, process maps, vendor binaries and SHA-256 provenance.
  Runtime perf/VTCM settings and hardware overlap remain unmeasured. The context's
  compile-time vtcm_mb=8 is not a recovered runtime configuration.
- Correction to the previous handoff: `hmx_v73_*` names alone do not prove the
  context was built for V73. Vendor naming and live backend auto-selection are
  V69; the actual ISA still needs independent confirmation.
- Next: the existing signed Meta `hexagon_nnOssQnnCreate/Run` interface and its
  QNN 2.25 runtime, or a compatible unsigned shell/runtime setup. No reboot,
  system property, service config, firmware, or executor binary changes were made.

## Heap scan pointer, chunk-map gate, Qwen module (Oct 30, ~15:00)

- **Warm conv_weights was NOT the linear cache search** (that costs ~5 ms). It is 80
  per-run conversions of genuinely dynamic weights: 1x1x128x64 `xpb` tensors produced
  each run by `convert_weights_to_signed.packed` from a uint8 activation tensor in
  pool 1. They cannot be cached. Staging their source with HVX before unpacking made cold
  much worse (432→1859 ms) and did not change warm, so it was reverted.
- The DSP first-fit `malloc` walked ~1,160 headers per call (465k per warm run,
  ~102 ms). It now has `heap_scan`: every block before it is in use, so choices are
  identical to a full walk. A host ASan/UBSan harness ran 200k random ops against the
  reference, with identical addresses and lists. malloc went from 102 to 38/73 ms warm
  primary/alternate, conv_weights from 234 to 183/212 ms, prepare_weights from 30 to 17 ms.
- Engine `5fa45a82f4c6a905e621db56f2432225aa4baf4057fd26faa792c10d10e62e2e`:
  cold random + full 27 cases + warm, both models, all exact, 1046/916. First
  inference 2.32/2.31 s, warm median 1.695/1.699 s (was ~1.76/1.73).
  Host CPU suite all 54 PASS. Reports build/resident-heapscan-{cold,full}-*.json.
  New counters: conv_weights_miss_ms/misses and malloc_ms/calls/steps (RESULT+184..+224).
- Remaining: ~170k malloc steps are free holes too small to use near the front;
  an address-ordered free list would keep first-fit exactly. About 150 ms of
  conv_weights is the dynamic unpack/convert loop itself.
- `cdsp_mem_check.py`: `ResidentDSP.__enter__` now refuses to start unless the
  read-only chunk map is clean. bench/chunk_map.py wraps it and accepts any
  lease of at least 128 MB.
- The Qwen copy `~/frame-dsp/depth-model/dev/frame/kmod/cdsp_mem.{c,ko}` is now the
  flushed version (`.orig` kept; the old dev source also lacked two range checks).
  It loads at 768 MB and the chunk map is clean there.

## quant_params fix and cdsp_mem flush fix (Oct 30, ~14:00)

**Cold inference 14.0 s → 2.45 s.** `quant_params` (qd_graph.c) no longer walks
`w->w` by column. `conv_weights` now also accumulates `Weights.wpos` (sum of
positive weights) in its existing row-major pass. `quant_params` uses
`wsum`/`wpos`, with neg = sum − pos, which is arithmetically identical.
quant_params_ms is now 11.5 s → 4.9 ms cold.
- Host CPU suite (verify_native_graph.py --backend cpu): all 54 PASS.
- Resident engine `cc1497a9257b97f6dd5dae6c5c10ea1d5a1991bcecd5d4e09da48e9ccd9bd0f0`
  (bootstrap unchanged `51fef8d4…`). Cold random-first then 2 warm, and the full
  27 cases + 4 warm, per model, all exact, 1046/916 calls/HMX. First inference
  2.46/2.44 s; warm median ~1.76/1.73 s. Reports:
  build/resident-quantfix-{cold,full}-{primary,alternate}.json.
- Remaining cold excess is conv_weights ~430 ms. It is also ~230 ms on every
  warm run, probably the linear weight-cache search in conv_weights. Next target.

**The bad mapping recurred after the reboot** on a reload that followed a lease with
real DSP work (chunks 3→56, 4→55). Reloads with only chunk-map reads had looked
fine, because the allocator returned the same chunks in the same order. The fix is in
kmod/cdsp_mem.c: `iommu_flush_iotlb_all(domain)` after the init mapping and after
the unmap in cdsp_mem_free. Strict-mode by-VA invalidation alone was not reaching the
translations the CDSP uses. Loading the flushed module over the stale state gave a
clean map. Then 3 cycles of (unload, 512 MB allocation churn, reload, chunk map,
probe, cold resident run) were all clean/pass/exact. Frame
perf-20261029/frame/kmod/cdsp_mem.ko is now the flushed build `962c1ada…`
(old build kept as cdsp_mem.ko.orig `0f4cd9d1…`; Mac copy of old source:
build/cdsp_mem.c.orig). Frame `~/frame-dsp/depth-model/dev/frame/kmod/cdsp_mem.ko`
is a separate, unpatched copy. Check which one the Qwen chat's load.sh uses. Keep
running bench/chunk_map.py after each insmod anyway.

## After reboot (Oct 30, ~13:40): resident backend correct; bad DSP mapping state cleared

Frame was rebooted. On a clean boot, with the instrumented engine `498aba51…` and bootstrap `51fef8d4…`:
- bench/chunk_map.py (DSP-side map of every 2 MB chunk) is clean on the first load
  after boot, after a plain unload/reload, after the probe and after resident runs.
- The owned-memory probe passes both sessions (build/owned-memory-post-reboot.json).
- Cold random + warm pass on both models, 1046 calls / 916 HMX, exact hashes
  (build/resident-post-reboot-cold-{primary,alternate}.json). Cold 14.0/13.9 s,
  warm 1.8 s.
- **Cold cost confirmed:** quant_params_ms 11548/11534 of the ~14 s cold run
  (4 ms warm). conv_weights_ms is ~387 cold, and still ~229 ms on warm runs:
  probably the linear weight-cache lookup, worth a look.

So the mis-mapped chunks were a bad DSP/SMMU state that a reboot clears. A
plain reload does not cause it, and neither does a resident run. The real
trigger is unknown. Candidates: the 768 MB Qwen loads at the same IOVA, the AXI
clock experiments, or the earlier hang recoveries. Earlier resident "PASSED"
reports were probably on a good mapping and may be fine, but none of them
checked the mapping. From now on, run bench/chunk_map.py after every insmod and
before any DSP writes; reboot if any chunk is bad.

## Before reboot: resident backend failed on a fresh lease (Oct 30, 06:30–11:15)

After a fresh `insmod kmod/cdsp_mem.ko size_mb=128` (no reboot):

- The unmodified final engine `fea5f7c7…` with bootstrap `51fef8d4…` fails
  cold random on primary: 1146 calls / 80 HMX (validated runs: 1046 / 916),
  output mismatch, 21.4 s. The instrumented engine gives identical results.
  Reports: build/resident-cold-orig-recheck.json, build/resident-cold-instrumented.json.
- Prefix bisection (fresh session per prefix): the first divergence is command index 2,
  `weights_to_vtcm@Xpb` copying tensor 67 (pool 5 offset 7133184) to tensor 68
  (pool 1 offset 3758080). The DSP destination is all zero; the ARM view of the
  source is correct. This is the same "command 14 / stayed zero" failure the
  earlier fix claimed to solve.
- The owned-memory probe `qd_owned_mem_probe.elf` (224f64ac…) **fails** on the
  same lease: error 5 at offset 0x600000 (page 1536), DSP read 0xb2994ea7. The DSP
  also reported nonce 0xb4b8aaa7 while the host wrote 0xb4b89fa7. That nonce
  is read from the dsp_service arena, not cdsp_mem, so the bad reads may not be
  confined to the private buffer. No SMMU/remoteproc errors in dmesg.
  Report: build/owned-memory-fresh-lease-FAIL.json. The probe restored the JTLB itself.
- Every runtime file on Frame was byte-identical to the Mac (images, constants,
  loader, bootstrap, engine, Python). Rebuilding the bootstrap and probe reproduced their SHAs.

**Update 11:15: the DSP's view of the IOVA window disagrees with the kernel mapping.**
Six more probe runs on three fresh leases (no reboot) all fail at exactly 0x400000.
The DSP nonce matched the host in all six, so the earlier nonce mismatch was not
reproduced: do not claim arena corruption. Each bad value decodes to another host
offset of the same lease (36 MB, 8 MB, 38 MB), constant within a lease.
A read-only chunk-map probe (bench/qd_chunk_map_probe.c, word 0 of every 2 MB
chunk; host script in Frame ~/frame-dsp/claude-scratch/chunk_map.py) gives:
- chunks 3, 4, 21, 22, 23, 24, 40 read the SAME foreign values on every lease
  (0x314e5341 "ASN1", 0xfb7ebf4d, 0, 0x14, 0xb85efce1): not our memory;
- chunks 2, 17–20 resolve to other chunks of the current lease, varying per lease
  (e.g. 2→3 on one lease, 2→20 on another).
This fits stale CDSP SMMU translations from earlier cdsp_mem loads/unloads (the
group 17 domain is type DMA; the kernel page table looks correct). **DSP writes into
this window may land in memory the module does not own.** The resident heap
(32 MB+, chunks 16–25) is affected, so earlier resident runs probably wrote foreign
physical memory. Reboot before any further DSP work, then run chunk_map.py on the
first load after boot and again after an unload/reload, to test the
stale-translation hypothesis. Until the reload path is proven safe, the module
should be loaded at most once per boot.

Earlier explanation: "cold" validations reused a module lease that
still held correct data from earlier runs at the same deterministic addresses,
which masked ARM→DSP visibility failures. The earlier "prior 4 MB probe
failure" was probably this same bug, not a probe artifact. Root cause unknown:
candidates are a DSP-side cache/system-cache coherence problem, or a state left by
the AXI/clock experiments. A reboot has not been tried. Do not build on the
resident backend, and do not run the cached-pools/C7 experiments, until a fresh
lease passes the probe repeatedly.

Separately, the instrumented engine `498aba51…` (now on Frame and Mac as
build/qd_graph_dsp.elf; original kept as build/pre-cold-instrument/ and
Frame build/qd_graph_dsp-orig.elf) adds `conv_weights_ms` and `quant_params_ms`
(RESULT+168/+176). On the failing run quant_params took 10.6 s and conv_weights
0.36 s. That still points to quant_params' column-strided scalar walk
(`w->w[i*co+c]` over L1-only DDR) as the cold cost. Confirm on a correct run.
cdsp_mem was unloaded afterwards; final state [1,0,1], callback 0, regions zero.

## Validated hybrid: about 100 ms

Separate build_native_hmx.sh / qd_conv_native.elf supports exact v6, native
integer v7 and native crouton load/store v8. Native modes affect HMX only;
non-HMX and EXIT remain v6. Default NativeGraph mode stays exact (0), block
output off. set_native_hmx(1|2) and set_block_output(True) explicitly opt in.

Payload SHA a21ed93fb980450c636f05adc09c2d2bce8e4e1af8a5d248b218f46712ec8041.
New files qd_hmx_native_kernel.h, qd_hmx_native_ops.S; native-aware VTCM layout
and weight-residency keys retain 421/389 reused uploads primary/alternate.

All 54 cases ran in phases A/B/C/A2. Exact phases A and A2 match the reference
hashes in every case. Native phases B and C are **not** equal to the reference
(`equal: False` in all 54); "pass" for them means they completed and native CM
(v8) equals native plain (v7) byte for byte. Report native-hmx-cm-block-full-ab.json:
- Exact/block output medians 114.50/114.89 ms primary/alternate.
- Native v7/block 103.73/104.67 ms.
- Native v8/block 99.62/100.38 ms.
- Smoke/random with 5 repetitions v8/block 98.4–99.4 ms; do not substitute these
  selected cases for the full-suite medians.
Ordinary output v8 medians 101.38/102.97 ms (native-hmx-cm-full-ab.json).
Main CPU 7, helpers 6/5/4, OPENBLAS_NUM_THREADS=1; exclude setup.

Attribution caveat: every A/B phase, including exact, uses the SDK-built
qd_conv_native.elf. Exact mode on that payload (~115 ms) is ~7 ms faster than
the earlier exact validation on qd_conv.elf (~122 ms), unexplained. Native
arithmetic itself is worth ~15 ms (115 → 100); do not credit the whole
122 → 100 ms change to it. The on-disk build/qd_conv.elf is now SHA
b57c41a8a091717b10085fe442e331a9717e963eb50a6ca8d14a181fb68d349c (modified
02:43); its source/provenance was not recorded.

Native arithmetic differs from the recovered simulator. Independent per-conv
ISA gates bound delta to 0/+1, but it propagates. U8 head smoke MAE 2.0703, p95 6,
max 15; random MAE 1.1010, p95 3, max 17. Synthetic w-2 reaches max 98 (MAE 2.2132).
Float random MAE ~0.085, max ~5.55; worst suite max ~25.25 in raw head units. No
nonfinite outputs. This does not establish actual depth quality or parity with
Model's native outputs. Keep exact mode and do not install the experiment.
Untested lead: the per-conv delta is one-sided (never −1), which suggests a
systematic rounding/bias offset rather than noise. A bias correction might make
native exact or near-exact; check this before accepting the error as inherent.

## Resident correctness recovered; performance is still poor

Whole graph, glue and convolution run on DSP through resident_dsp.py,
qd_graph_dsp.c, qd_graph_boot.c/qd_graph_stack.S, qd_resident_conv.c and a private
128 MB cdsp_mem module at IOVA a0000000. No firmware memory borrowed for DDR.

Earlier smoke-first suites concealed cold-random failure at command 14:
weights_to_vtcm destination pool 1 offset 3762176 stayed zero. Current fix combines
C6 shared-uncached mutable pools/I/O/staging, scalar-only C0 code/constants/heap,
proper dccleaninva publication (previous dcinva was invalidate-only), and heap
publication before HVX metadata reads. Do not claim an isolated causal ablation.
C0 is L1-writeback/non-L2-cacheable per SDK qurt_types.h, not uncached memory.

**Ablation not yet run:** cached mutable pools with only the dccleaninva and
heap-publication fixes. The uncached C6 pools are a likely contributor to the
1.7 s warm time; this is the first experiment to run next.

Optional C7 scalar-page mapping (QDS_SCALAR_CACHE=7) showed no timing change
(warm 1705 vs 1708 ms). That result is **unproven**: the engine SHA is identical
in both modes, and resident-cold-random-cached-validation.json and both
resident-cold-prefix-*.json do not record a bootstrap SHA, so they cannot show
which mapping actually ran. Rerun with bootstrap SHA and the readback mapping
recorded before concluding C7 has no benefit.

The owned-memory probe now verifies three fresh-nonce sentinels on EVERY 4 KB
page read-only before ANY data write, records failures, checks DSP writes, and
restores all saved JTLB entries 64..71. Two sessions each on two allocation
leases pass. Final readback is C0/C6/C0/C0/C6/C6/C6/C6. qdo_map invalidates stale
leases; qdo_unmap cleans scalar data before restoring mappings. Reports:
owned-memory-nonce-lease1.json, owned-memory-nonce-lease2.json,
owned-memory-nonce-cached-validation.json, owned-memory-final-validation.json.
The prior 4 MB probe failure was not reproduced by these stricter rebuilt probes.

Final exact resident validation PASSED 54 inputs plus 4 warm repetitions, and
fresh-session random-first both models. Reports resident-final-full-validation
and resident-final-cold-validation.json, copied to Mac. Warm medians 1708.49/
1682.63 ms primary/alternate; first cold random 12920.67/12894.89 ms.
Workers 52–55 ms; scalar/layout and DDR traffic dominate warm time. Earlier
~1.07 second smoke-first results were not cold-safe.

**Cold cost is inside the conv kernels, not weight preparation.** Per-variant
profile (resident-final-cold-validation.json vs warm resident-final-full-validation.json),
both models nearly identical:

| Variant (1-based) | Kernel | Calls | Cold ms | Warm ms |
| ---: | --- | ---: | ---: | ---: |
| 33 | ConvLayer_s1.opt@CB*2.Xwb.Fi.t | 629 | ~9208 | ~213 |
| 36 | ConvLayer_s2.opt@CB*2.Xwb.Fi | 80 | ~2035 | ~33 |
| 35 | ConvLayer_s1.opt@CH*2.Xpb.Fi.t | 70 | ~486 | ~479 |

prepare_weights_ms is only ~195/206 ms and prepare_ms ~324/336 ms cold. About
11 s of first-run time sits inside those two kernels and is not covered by any
phase counter. Earlier text blaming "uncached metadata/weight preparation" is not
supported by the counters. Instrument those kernels before optimizing anything else.

Old fused direct staging now also passes cold random and 2 warm runs per model
(resident-direct-clean-validation.json), 709 calls using graph croutons directly
for software HMX-A staging. It remains slower (~1.8 seconds warm): pack_x ~343 ms
versus ~114 ms default, despite reducing worker preparation. Keep direct_hmx off.
This is NOT the new QNN native leaf: it still builds expanded A in software.

Current final engine SHA
fea5f7c706f3b5b0057d546cdd47553775a90aade694ba0676d02137d5a8895a;
bootstrap 51fef8d4cbcb28770f8fb9c83062154a2c0993090a30d92470a85bf6367a83da;
probe 224f64ac2b662cc02161365813daa96b159b3806db94cf5a8a16d48a55b1e41a.
QDS_SCALAR_CACHE=0|7 changes mappings in bootstrap/probe; the engine SHA can
be identical, so every graph report must also record the bootstrap SHA.

Private cdsp_mem module is now UNLOADED and /dev/cdsp_mem absent. Locked final
snapshot native-resident-final-state.json: controls [1,0,1], callback 0, loader,
payload and arena zero. Transport restored clocks/power and released workers.
Do not autoload this experimental module in a packaged backend.

## Next route: native kernels consuming actual VTCM croutons

The original aligned QNN 1x1 leaf passes 128 ISA fixtures / 1,966,080 outputs
against independent int64 matrix arithmetic (identity exact, random delta 0/+1).
CI/CO 32/64/96/128, H 8/16, W 8/24, scattered block tables, signed random weights,
bias and guards. qnn-hmx-1x1-sim.json and log. Files extract_qnn_hmx.py,
qd_qnn_hmx_1x1.S, bench/qnn_hmx_1x1_sim.c and check_qnn_hmx_1x1_sim.py.
It is SIM-only, not integrated or benchmarked on hardware. Source provider
metadata/code hash is qnn-hmx-leaf.json.

Important ABI: bias starts at the supplied pointer (same-packet increment reads
old r3), not pointer+256. Flatten spatial croutons into descriptor block count
with width 64. The leaf has external branches to unextracted variants: ONLY the
proved aligned config is admissible. Native load/store avoids software A prep.
Spatial/u16 leaf ABIs, actual graph eligibility and boundary cases remain unproved.

The working hybrid's HMX arithmetic takes ~6 ms within ~100 ms total. Native
kernels with actual VTCM graph tensors and DSP glue in the same layout are the
selected path. Full residency over DDR does not solve the access-pattern cost.
Only the first 0x2d0000 VTCM bytes are verified scratch today, despite the 8 MB
mapping. Prove ownership/preserve higher VTCM, then plan live tensors/weights/scratch;
do not blindly overlap expanded HMX staging with live graph data.

750 MHz is DSP AXI master, NOT DDR. Bounded 750/1000/750 measured vector 10.09/
10.33/10.15 GB/s, scalar-dependent 200.8/196.1/201.6 ns: ~2–3% benefit from
1000 over 750. Default 750, no DDR-controller clock change. Report
axi-750-1000-validation.json.

Existing ISA gates: glue u8 1,572,864 values; glue u16 1,310,720; blocked u8/u16
12 layouts each; pack-x32 342, pitch 33/34 included; u16 gather 160; direct HMX-A 96;
native convolution 8 geometries / 14 input formats plain+CM, collectors and guards.
Host ASan/UBSan: collection 68, elementwise 31, publication 2, packets 1, conv 6.
The current crouton-load exploratory log includes an unsupported input-control 0
termination; the separate complete convolution/native 1x1 gates are the evidence.

See NATIVE_EXECUTOR_HANDOFF.md for commands and earlier history.
