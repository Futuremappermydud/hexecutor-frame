# Handoff: resident DSP executor, Oct 30 2026 (afternoon)

> **Superseded current state:** read [HANDOFF_20261030_2145.md](HANDOFF_20261030_2145.md) first. User requested handoff; all agents stopped, device clean, cdsp_mem unloaded. This file retains experiment history.

## Newest: prepared 3x3 backend validated across both model families

QDS_PREPARED_COMPACT_3X3=1 plus prepared, elementwise and preprocess flags,
engine SHA74d31b4b2986e3b776c0b5f967dacdc46aee82a1926864b8736ecc1a1c57ed0c.
Built FROM frozen transitive source bundle native-prepared-compact3-first-source/;
BUILD_FLAGS/source hashes captured before compile, all reports included.
Native3 stages only A into reservedtail; W/Fi/Y remain actual model VTCM.
L2 native-program metadata/expanded blocks retained; no legacy decoder fallback.

LIVE primary7 + alternate7 representative CB3x3 families, smoke/random each:
28 calls, CI32/64/128/256 and CO32/64/128 including logicalCO2/max32,
odd output slices69/85, all5pools outside target exact, target0/+1 only,
all guards, worker exiterrors0, full8MiBrestore/hostfingerprint pass.
Alternate direct1x1 seq1166/2467 smoke/random also pass withzero staging.
Primary247 unsupported suffix fails preflight before graphwrites/calls/upload.
Primary247 compact prep20.84us, leaf2.96us, drain5.38us; 2.08ms whole RANGE
mostlyboundary7.5MiBcopy. This is NOT new fullgraph timing. Fullgraph still
unsupported; accepted wholegraph remains647–652ms inexact, exacthybrid~115ms.

Alias-preserving compact-input diagnostic mode2 also passed Frame247:
engine SHAae314f9d458afad7c7878d3c4ce689b37347dd76b740f18baa134f4a84a7f53e.
Same repeated inputrefs/zero-dY retained, same exactoutside/output0+1 and
fullrestore. Thus aliases alone do not explain original direct3 stall;
large address span/cross-bank pattern remains unresolved. No blind retries.

Prepared preprocessing strict source a8b4bad... now frozen and rebuilt byte-
identically to d20733c045bc27f467e92e280b9aa24acddcdb40af715c336f001660c0aaf120.
Both models first InputSliceFf and Im2col smoke/random LIVE all5pools exact,
fullrestore. Checkpoint native-prepared-preprocess-final-source/ with reports.

CURRENT next work: root extending direct1x1 paddedXpb rawCI12/20→native32
weight-only expansion atreservedtail+2000 (A/Y/Fi direct), NOT yet numerical
or hardware validated. New binding fields tail_publish_bytes/staged_weight_bytes/
packed_weight_guard must be initialized/published/checked. Native-program
specialist owns feature-gated bridge +actualCPU-prefix SDK proof fixtures.

IMPORTANT S2 correction: legacy V69 leaf3c8a80 is not actual V73 QDRX leaf389980;
prior legacy lane/tap diagnostics do NOT prove production S2. Specialist is
building exact V73 worker/helper capture. Same-packet call companions are
callee argument writes, never post-call. U16 specialist rechecking actual
route too; RTTI input-count matching alone was insufficient. No U16 proof.

## Breakthrough: direct3 succeeds with input-only compaction on Frame

Primary247 random numerical SDK and actual Frame PASS with QDS_DIRECT_COMPACT_INPUT=1,
separate engine SHA114f9bafd9928fa648d8a4e5541b6626545df535fab13f3d108471829d233ca4.
Only108 A pages (221184B) are copied to reservedtail+0x2000; original W/Fi and
Y bindings remain untouched. All outside-output bytes in all5pools exact,
output64863exact/673+1; native3 call1, failures0; full8MiB restore/fingerprint
and all worker exiterrors0. Leafissue3.286us, compact/bind/metadata prep58.906us,
drain3.382us; 2.29ms range includes1.94ms boundaryupload+0.218ms download.
This is input STAGING DIAGNOSTIC, not zero-copy or fullgraph performance.
Frozen95file checkpoint native-compact-input-validated/ includes binary,
transitive source and actual SDK/Frame reports. No new reboot was needed.

This narrows prior Frame fault to Aaddress pattern; it does NOT yet distinguish
large/negative row deltas from duplicated zero-page aliases. Root now testing
mode2 (compact-input-aliases) which preserves11duplicate references/zero-dY
while retaining bounded~221KiB span. Numerical SDK proof PASS withsame staged
outputSHA cfec6dbed3eb2e88544d5c0371c8c50915ce90e6b14f44be7c4411c19bf5e15f;
Frame build pending, not invoked yet. No blind original-direct3 retries.
Prepared3 family remains disabled; latest accepted wholegraph still647–652ms.

Preprocess specialist actualCPU-prefix V75 proofs nowPASS bothmodels for
InputSliceFf command0 and Im2col41/47 (alternate active slice35 isunique;
off-slice aliases do not authorize rejection). Root linked new opt-in
QDS_PREPARED_PREPROCESS=1 in separate engine; build/hardware validation pending.
Elementwise f84 frozen source now rebuilds byte-identically (PROVENANCE updated).

## Newest: prepared ADD and LinearClip hardware-validated on both models

Fixed opt-in prepared-elementwise engine SHA
f84bcfbaeb3e86a20afea63bac2e21137b687c46938c4c55983fbca0572b68eb.
Both models, smoke/random, first ADD13 and LinearClip68: all five pools byte
exact versus actual CPU-prefix snapshots, zero legacy/native-conv calls,
no per-conv tensor staging, exit_errors[0,0,0], full8MiB lease restoration and
independent host fingerprint pass. ADD/LinearClip per-range2.2–2.7ms is mostly
7.5MiB boundary upload/download; NOT full-model inference. Metadata/expanded
blocks stay in L2, caller-owned32KiB scratch at0x30078000..0x30080000.
Corrected numerical V75SIM and ASan/UBSan actual-metadata proofs both pass.
Reports native-prepared-elementwise-fixed-{primary,alternate}.json. Frozen
87-file transitive source+binary checkpoint native-prepared-elementwise-validated/;
byte-for-byte frozen-source rebuild PASS (same f84bcf SHA). Failure c5e75a remains preserved and banned.

Next prepared coverage: InputSliceFf calibration-copy and audited Im2col helper,
offline specialist-owned preprocessing module. Full program remains unsupported;
3x3 prepared family is still disabled. Repeated preflight/alias-sort costs are
not yet optimized; static plan caching and dynamic-Fi validation separation
are needed for complete runtime behavior/performance.

S2 helper harness corrected object-field rotation: provider wrapper expects
[reserved,input,weights,Fi,output]. Earlier Fi-as-address fault/1xN selection
was a test bug. Correct-role generic rawS2 now drains but numerically mismatches;
no new nativeS2 proof. U16 onehot isolated real channel-lane offset; still inexact.
Root alone owns hardware. Last elementwise close clean; cdsp_mem128MiB loaded.

## 21:00 checkpoint: elementwise integration, null-preflight recovery

Prepared ADD v13 passed primary3424 smoke/random on Frame: all five pools exact,
zero per-conv staging. Initial LinearClip v68 engine c5e75a faulted in preflight
before boundary upload: alias_proof dereferenced absent CB output o==NULL.
SDK mapped-zero memory hid the UB. Agent fixed it and added actual-metadata
ASan/UBSan host tests (both models pass); corrected numerical SDK rerun pending.
Do NOT invoke c5e75a again. Binary/log/partial proof preserved under
native-prepared-elementwise-null-failed/. Full stable retained capture saved
under retained-linearclip-preflight-20261030/, backup SHA
0b840f255c13f19897b51dd82b8bd8a63d6fb7dbe52f57559f306cd66d4484f8;
independent fingerprint verified. Authorized recovery reboot21:00:50 EDT.
Post-reboot CTRL[1,0,1], callback0 and all service regions zero; cdsp_mem unloaded.
Never restore old retained backup on the fresh DSP.

Safe direct3 metadata-only Frame probe passed, and frozen815a cold staged247
control now PASSED without any earlier DSP conv: target64863exact/673+1,
all outside bytes exact, full lease restore. This rules out previous-HMX-op
priming; direct3 hardware address/layout contract remains unresolved.
Do not repeat blind direct3 calls. Accepted wholegraph remains647–652ms,
INEXACT; exact hybrid ~115ms still faster. 20ms not established.

Current Luna lanes: native_program_expert finishes corrected elementwise SDK
proof; vtcm_memory_expert recovers original V69 stride2 1xN raw leaf/caller;
native_u16_expert isolated +4 channel shift in onehot CH basis, actual legacy
fixture still numerically wrong. Root alone integrates and operates hardware.

## Latest: prepared L2/VT runtime now hardware-validated for bounded ranges

Root integrated a REAL prepared executor in opt-in QDS_NATIVE_PREPARED=1 engine:
`build/qd_graph_dsp-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-native-1x1-native-3x3-direct-range-prepared-vtcm-lease.elf`,
SHA1cdaca901a08b8224f606a4c44ef838d29afb1e7949a0f9eb968753d4322eca9.
Flags same directrangebase plus NATIVE_PREPARED=1; familymask1 permits ONLY1x1.
Do not enable3x3 in thisengine yet. SpatialFramecause remains unresolved.

qd_graph_prepared.h keeps oldg in privateheap (nevercopyB2000), places expanded
blocks atL2+0x2000 (481320/482368B), hotmetadataB0000..EF000 (235208/234072B),
L2tables0/ABI+800, actualVTpool1 with 7.5MiB active and512KiBcontroltail.
Host ResidentDSP(...,prepared_program=path) uploads NPRG plus optional extension
atconfig+132. Header/source/context/constant/prepared hashes validate before
hotplacement. Every requestedrange preflight precedes boundaryVTupload; no
legacy fallback; zero per-call tensorstaging. Full model remains unsupported.

LIVE PASS primary1195 and2380 / alternate1166 and2467, smoke/random, actual
CPU-prefix seeds. CO128/CO64, dynamicFi and nonzeroYslices covered. Allfivepools
outside target EXACT; outputonly0/+1; all8calls finite/guards/exiterrors0;
full8MiBrestore andhostfingerprint pass forbothsessions. Most per-range2.0–2.6ms
is7.5MiB boundaryupload/download (~1.7–2.8ms +0.2–0.25ms), NOT full inference.
Metadata path readsL2 ratherthanoldDDR; reports native-prepared-{primary,alternate}-matrix.json.
Additional LIVE exact InputSlicePad ranges: primary[19,41)22commands,
alternate[30,47)17commands; smoke/random twice, entirefivepools CPU-prefixbyte
exact, suffixpreflightrejectno graphwrites, fullrestore+hostfingerprint passes.
Reports native-prepared-input-{primary,alternate}.json. Frozen source/artifact
checkpoint `native-prepared-1x1-first-validated/sha256.json` includes theseproofs.

Direct3x3 publishedcaller stillstalled onFrame, tracephase4 (insideleaf), despite
correctcachepublication/SSR. Rootsavedverifiedbackup7b1037cdc0ddd36d6e456718b4f2cf4d552204ad8f0257dd15eb1124452755a0
under retained-direct3-publish-20261030/, rebooted20:13:03 with existinguser
recoveryauthorization; neverrestoreoldbackupoverfreshDSP. No more direct3calls
untilnarrowed. **Metadata-only Frame probe PASS** (no convolutioninstructions):
marked QDS_DIRECT_BIND_ONLY=1 separateengineSHA62d770da1e8c5adbade23de8b27868cccb39a0e24773904e2843116dcf933ea1,
seq247 tablesunchangedbefore/afterpublication, descriptors/cfg12=0x0c correct,
all108inputpages+rawcurrentW/Fi exactCPUbytes, allmodelpoolsunchanged; fulllease
restorepassed. Report native-direct-bind-only247.json. This rulesoutpointer/
operandprep corruption, not hardwareHMXaddress-span semantics.

Actual graph_direct_run SDK caller now passes too with nativeSDKL2TCM mappings
D8000000, nativeVTCMD9000000, compacted/permutedseq247scatter (11aliases,57jumps).
Earlier SDKfaultwas Cglobal-vsfixedVAcache-synonymcorruption; no Frameanalogue
(tablepublicationFrameprobeisexact). Runners/reports in native_runtime/.
Root is running a **frozen815a staged247 cold control**, CPUseededjustonecmd,
to distinguish HMXstartupstate from direct-modelVTaddress/layout. Check its
pending log/report before anynew hardwarejob. Root sole hardwareowner.

Current Luna lanes: vtcm_memory_expert recovering actualrawS2 leaf/callerABI
(0x389980, setter0x2f3ce0); native_u16_expert recovered exactlegacyfour-input
helpercfg/descs butlegacyalignedleaf stillallzeroevenidentity (no proof);
native_program_expert validating ADD13+LinearClip68 using actualCPU-prefix
fullpoolfixtures inSDK (syntheticuniformtests werecorrectlyreplaced). Elementwise
is NOT in acceptedpreparedengine yet; demands30,748B DSPscratch. LowerL2gap
[0x78000,0x80000)32KiB fits if expandedblockcapacityshrinksfrom0x7e000to0x76000
(start+0x2000). Do not overlapexpandedblocks orstacks; actualaltblocks end0x77c40.

Accepted wholegraph remains **647–652ms be36…**, 823native calls, INEXACT. Frozen
expanded95-file source nowrebuilds acceptedELF byte-for-byte (PROVENANCE updated).
20ms fullgraph stillunproven. Defaults/loader/firmware/installedmodel untouched.
Hardware cdsp_mem128MiB LOADED; lastmetadata-onlysessionclean, coldcontrolcurrent.

## Newest checkpoint: direct1x1 hardware proof; direct3x3 failure recovered

This section supersedes pending/direct/hardware status below.

**Direct1x1 LIVE PASS** CPU-seeded primary1195 random, experimental direct-range
engine SHA01af4c9e96558502406fca33fa7e078eea746d8afd8b132a6c5a66e85b38cda9.
Pool1=actualVT, raw current W/Fi alreadyVT, zero per-call tensor staging; all5
pools outside output EXACT; target64,040 exact/1496+1, guards/exiterrors0,
full8MiB restore+hostfingerprint pass. Suffix1195..end rejects nextcmdindex984
before any calls or graphwrites. Boundary7.5MiB upload1.702ms/download0.201ms;
bind15.19us, leafissue2.15us, drain+scalarchecks25.98us. This is one-command
proof, not full native/20ms inference. Frozen79-file
`native-direct-1x1-validated/sha256.json` and `native-direct-range1195-first.json`.

**Direct3x3 LIVE STALLED**, sameengine primary247, after successful preflight
suffix rejection at index172/seq251 (all pools unchanged/no calls). Single
native247 did not return; hostEXIT could not obtain worker-exit markers.
Root saved stable8MiB backup SHA c2d969286901365a4397be2fa3a7a43ec389f327762436f7bc3719ad241c91b7,
independently verified host fingerprint, copied all artifacts locally under
`retained-direct3-20261030/`, then used existing user reboot authorization.
Reboot20:00:49 EDT, back20:01:55; clean CTRL[1,0,1], callback/regions0, eye2962S,
**cdsp_mem UNLOADED**. Do not reapply retained snapshot over fresh DSP.

Root's direct caller omitted metadata cache publication that the accepted
`qd_native_3x3_issue()` and passing directSDKdriver both do. Root now added
cleaning L2tables2048B, ABI120B and tail8KiB, barrier, explicit SSRbit26
preserve/restore; phase trace atSERVICE+1024 and HVXguard reduction. This is
built in shared experimental direct-range ELF but NOT hardware-validated.
Original01af ELF is frozen; don't confuse with new build at same experimental
filename. No further live3 until actual scattered/aliased QDRpage SDK test
passes: prior direct3SIM normalized addresses to contiguous staged pages,
so that proof did not cover real scatter. vtcm_memory_expert is doing that.

**Prepared runtime offline complete:** executor/native/IO/runner modules in
native_program/, host ASAN/UBSan and strict Hexagon object compiles pass.
`nprg2_bind_metadata` separates hot placement from real execution preflight.
Runner migrates expandedblocks once into callerL2 while retaining original
setupallocation forclose. L2 now feasible for native-only scratch: tables0,
ABI+0x800, expandedblocks+0x2000..~0x77xxx (<512KiB), stacks80000..B0000,
hotB0000..EF000 (~235KiB), workF0000..100000. Oldg must stayprivateheap in
preparedengine; oldgB2000/tensorcacheB4000/legacyworkers cannot coexist with
thishotplacement. Observed primary hot235208/blocks481320B, alt234072/482368B.
Strict NOP+InputSlicePad handlers actual CPUprefix byte/bounds proofs pass,
W2V variants27..30 copy CPU semantics, writablepool1cap0x780000 mandatory.
Agent adding separate read-only runner/executor preflight so boundarypoolcopy
can happen AFTER whole requestedrange validation. No physical NPRGengine yet.

**U16 correction/breakthrough:** original Model V69 provider already exists at
`.work/model-qnn-20261030/libQnnHtpV69.so` (no section headers). It has legacy
`conv_layer_quant` CH four-input class matching QDR AND newer five-input
`conv_layer_quant_v73` class. Prior pinned2.50 investigation used wrongfive-input
variant; don't assume model has hiddenfifthcontrolstream. Matchinglegacy
vtable0x9222d4, execute0x21de24, compiler0x203320, calls worker0x2079b0 withr5=0.
U16expert now recovering truelegacyworker/leaf, no numericnativeu16proof yet.

Latest accepted wholegraph remains be36… **647–652ms** (823native), INEXACT;
full20ms unproven. Frozen expandedgraphmanifest is being checked for full
source reproducibility; add missing transitive top-levelC/H/S/ld dependencies
before claiming reproduciblebuild. Originalacceptedbe36 ELF restoredfromFrame;
sharedsourcewithdirect additions rebuilds9bb…; don'treplaceacceptedartifact.

## Current checkpoint: native coverage 647–652 ms; direct-memory runtime in progress

This section supersedes older timing/agent status below. User explicitly asked
for Luna specialists to implement Qualcomm-style memory-resident execution.
Root owns shared integration and every live hardware invocation; specialists
own native_program/, native_runtime/, and native_u16/ offline work.

**Latest LIVE whole graph:** native 1x1 + padded CI12/20 + native 3x3 + odd
output-slice support, 240 native1 + 583 native3 = 823 native calls per model.
Both models smoke/random, two calls each, completed clean with deterministic,
finite outputs, exit errors [0,0,0], full8MiB VTCM restore and host fingerprint.
Warm primary648.98/650.42/651.55 ms; alternate647.00/649.26/648.35 ms. First calls
893/896 ms. This is INEXACT vs integer; no Qualcomm-equivalence or20ms claim.
Previous native1+native3 was~722–725ms; exact frozen control~861–867ms. Existing
hybrid~115ms exact remains faster than the resident prototype.

ELF `build/qd_graph_dsp-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-native-1x1-native-3x3-native-slices-native-padded-vtcm-lease.elf`
SHA be36f84636495d2f1b5e62392388e6ef98d5b5f12967e9d01890a491d1cad25a.
Flags DYNAMIC_TCM=1 FALLBACK_TCM=2 BLOCKED_GLUE=1 FAST_SPILL=1 PRESERVE_VTCM=1
NATIVE_1X1=1 NATIVE_PADDED_1X1=1 NATIVE_3X3=1 NATIVE_3X3_SLICES=1 (all QDS_).
Reports `.work/frame-direct-kernels-20261030/native-expanded-{prefix115,full-first}.json`.
Primary first native115 strict prefix: 64,451 exact/1,085 +1 target bytes;
ALL other pool bytes exact, all recovery checks pass. Actual production3x3
core with SLICES=1 passed seq3627 CPU-prefix SDK simulation, including output
slice67 (64,457 exact/1,079 +1; guards pass). Padded1x1 actual raw CI12/20
simulations and poisoned-tail comparisons pass. Original validated native815a…
and exactc82f… artifacts/manifests are retained.

**Memory-resident path is being implemented separately:**
- native_runtime/qd_native_direct.{c,h}: direct pool1 A/Y/W/Fi binding, L2 tables
  and ABI, 7.5MiB active VTCM plus reserved512KiB tail, no per-call tensor copies;
  side-effect-free validate API for full-range preflight. Host binding/alias
  checks pass; numerical direct-leaf SDK proof is still being debugged.
- native_program/prepared_executor.{c,h}: fixed80-entry variant/op registry,
  full requested range validation before execution, scratch declaration,
  cancellation/stats, no fallback. W2V27–30 actualmetadata CPU copy semantics
  tests pass. prepared_native bridges bound hot metadata/currentpool1 W/Fi to
  directbind and backend submit/drain. Metadata-only census132native1+583native3
  permodel,131rejected; ASAN/UBSan host tests and Hexagon syntax compile pass.
  Mock coverage/callback tests are NOT numerical or physical placement proof.
  Active pool1 writable cap is being added to protect the reserved tail.
- Root added QDS_NATIVE_DIRECT_RANGE=1 separated engine/harness
  qd_graph_native_direct.h + bench/verify_native_direct_range.py. It requires
  entire range direct-native-compatible before VTCM writes; one7.5MiB boundary
  upload/download, pool1=VT inside, current W/Fi, no legacy fallback. This
  compiled but has NOT run on hardware; await numerical SIM proof first.
- U16 expert found actual provider worker consumes a fifth generated Int32
  input (mode/control stream), absent from four logical QDR refs. Explicitni=0
  is generic zero-point input, not the stream. Streamproducer remains unknown;
  all u16 native probes remain diagnostic-only. Trace compiler/vtable producer.
- Stride2 native leaf SDK proofs already pass all three actual CI families in
  native_stride2/ using zero-expanded3x3 transform; no graph adapter yet.

CURRENT HARDWARE: loaded cdsp_mem128MiB, last direct-expanded graph close clean,
CTRL[1,0,1], callback0, loader/payload/arena zero, one S eyeclient19329. Recheck
before next invocation. Use guarded preserving bootstrap SHA628e362d… ONLY.
User's reboot permission was used at19:03 for startup recovery; no further
reboot needed. No installed models/default payloads/loader/firmware changed.

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

## Latest continuation: Frame gap isolated and reduced (~15:10)

The 9.7 ms figure below is **Model**. Frame's original resident backend is
~1.70 s; the existing hybrid is ~115 ms exact / ~101–103 ms inexact native.
The gap is dominated by the port's layouts/memory/scheduling, not Python/RPC
or a demonstrated HMX throughput limit. The native hybrid's HMX calculate
phase is only ~6.2 ms. See the new section at the top of WIP for scoped counters.

Two opt-in, exact memory improvements now bring the **resident** backend to
**1126/1103 ms warm** primary/alternate (33–35% less time), ~1.58 s cold DSP
execution. 80 dynamic Xpb weight tiles already match HMX B order: reuse the
packet, compute channel sums in local memory, remove 400 warm malloc calls.
Ten 16-bit fallback tiles were extracting/merging planes in uncached DDR;
pin their live arrays above worker scratch in L2, with bounded VTCM/DDR fallbacks.
Both changes preserve the arithmetic. Full suites for dynamic-only, VTCM and
L2 versions each pass all 54 cases plus two warm repeats. Cold random-first
and original-baseline return also pass (return 1687/1697 ms).

Build on Mac: `QDS_DYNAMIC_TCM=1 QDS_FALLBACK_TCM=2 sh frame/build_resident_dsp.sh`.
Use `--engine frame/build/qd_graph_dsp-dynamic-tcm-fallback-l2.elf`; SHA
`2c513b1df06eb53572bf9d10f474164071702761475f149a5d2e7abfbef9e0d5`.
Separate engine filenames; experimental builds leave default bootstrap/probe
alone. Original default engine/bootstrap/loader/probe SHAs unchanged.
Reports `frame/build/resident-gap-*.json`, local evidence/source snapshots
`.work/frame-gap-20261030/`. Current source and reports synced to Frame.

Remaining: pool 1 still lives in DDR, repeated dense/blocked conversions,
serial execution. Native blocked-tensor kernels plus proven VTCM allocation
and DMA/worker overlap remain the large architectural step. Only the already
verified scratch region was used in this continuation. Module now unloaded;
DSP clean [1,0,1], callback/regions zero, eye client 36699 S. Installed v5/Qwen
unchanged; no reboot or service/firmware changes.

## Continuation update: Model execution now works (~14:40)

The stock SDK remains blocked, but the preserved **dsp.pte** programs now run
through the existing signed Meta `hexagon_nnExecutorch*` path and vendor QNN
2.25. Full 54-case suite / 162 calls: repeatable, finite; warm RPC medians
9.71/9.65 ms. **Not simulator-exact:** U8 exact 5/27 per model, float 0/27;
worst float difference 22.32 raw units. Frame v8 hashes also do not match Model.
Mode-1 `hexagon_nnExecutorchRun3` profiling yields 201/205 logical op records
with wait/background-overlap cycles; 12 profiled calls preserve ordinary outputs.
Profiling inflates call latency to ~173–178 ms, so keep separate timing runs.
Evidence: `.work/model-qnn-meta-20261030/{full,profile}-validation.json`.
Tools and reproduction: the new successful-route section of
`frame/RESIDENT_EXECUTOR_WIP.md`, `frame/bench/verify_model_metadata.py`.
No system configuration changes. This supersedes the blocked-only Model
follow-up and next-step 1 below. Runtime perf settings and a tile-level DMA
timeline remain unmeasured. Frame VTCM ownership remains the main porting task.

**Paired C7 follow-up is also complete (~14:45):** C0/C7/C0, cold random plus
four warm smoke repeats per model, 30/30 exact. Warm primary/alternate:
C0 1690/1699 ms, C7 1693/1700 ms, C0 return 1693/1700 ms. No useful gain.
Separate bootstraps now prevent build-mode confusion; select experiments with
`--bootstrap`. Default engine/bootstrap/loader/probe SHAs unchanged. Reports:
`frame/build/resident-c7-paired-{A0,B,A1}.json`; detailed log in WIP. Module
unloaded, final Frame clean [1,0,1], callback/regions zero, eye client S.

Mac source (no git): /Users/owenelliott/Documents/projects/spatial-stuff/depth-model
Frame dev: ~/frame-dsp/depth-model/perf-20261029 (in sync with the Mac for every file below)
Frame python: ~/frame-dsp/depth-model/.venv/bin/python. Use `ssh frame`; sudo password `frame1`.
Detailed log: frame/RESIDENT_EXECUTOR_WIP.md (newest sections at the top). Older: frame/NATIVE_EXECUTOR_HANDOFF.md.
Never rebuild qd_loader.elf (828ca063…). Installed v5 and Qwen chat are unchanged.

## Current state (verified)
- The resident engine (whole graph on the DSP) is exact on both models: cold random-first, all 27 cases
  and warm repeats, 1046 calls / 916 HMX. First inference ~2.3 s (was 12.9–14 s), warm ~1.70 s.
  Engine build/qd_graph_dsp.elf SHA 5fa45a82f4c6a905e621db56f2432225aa4baf4057fd26faa792c10d10e62e2e,
  bootstrap 51fef8d4…, probe 224f64ac…. Reports: build/resident-heapscan-{cold,full}-{primary,alternate}.json.
- The host CPU suite (`.venv-cpu/bin/python frame/bench/verify_native_graph.py --backend cpu`) passes all 54.
- The hybrid ARM+DSP path is unchanged: ~115 ms exact; native HMX ~100 ms but inexact (per-conv delta 0/+1).
- End state on the Frame: cdsp_mem unloaded, DSP clean (controls [1,0,1], callback 0, regions zero), eye client running.

## What was fixed today
1. **Stale CDSP SMMU translations on cdsp_mem reload.** After a lease with real DSP work, a reload
   resolved some IOVA chunks to other chunks or to foreign memory. DSP writes there corrupted memory
   the module doesn't own. That is why the old "PASSED" resident runs were masked by stale data and
   the "command 14 stayed zero" bug kept coming back. The fix is in kmod/cdsp_mem.c:
   `iommu_flush_iotlb_all(domain)` after the init mapping and after unmap. Verified over 3 churned
   reload cycles with probe and resident runs. Installed in both Frame copies
   (perf-20261029/frame/kmod and the Qwen copy dev/frame/kmod, with `.orig` kept). The Frame was
   rebooted once today to clear the bad state.
2. **Chunk-map gate.** cdsp_mem_check.py plus build/qd_chunk_map_probe.elf (read-only DSP probe).
   `ResidentDSP.__enter__` refuses to run unless the map is clean. Manual check:
   `frame/bench/chunk_map.py` (any lease of at least 128 MB). If any chunk is bad: reboot.
3. **Cold cost: quant_params**, a column-strided walk over L1-only DDR taking 11.5 s. conv_weights
   now also computes `Weights.wpos` in its row-major pass; quant_params uses wsum/wpos. Exact.
4. **The DSP malloc walked ~1,160 headers per call** (~100 ms per warm run). There is now a
   `heap_scan` pointer (every block before it is used), so choices are identical to full first-fit.
   Host ASan harness: 200k ops, identical addresses. malloc 102→38/73 ms, conv_weights 234→183/212 ms.
5. Docs: corrected the earlier agent's mistakes (stale SHAs, "pass" wording, the 122→100 ms
   attribution, the cold-cost attribution, the unproven C7 claim, stripped spaces).

Counters now in resident results: conv_weights_ms/miss_ms/misses, quant_params_ms, malloc_ms/calls/steps.

## Key findings about QNN (what the Model does differently)
- **QNN's pool 1 (8 MB) IS VTCM.** QNN acquires exactly 0x800000 of VTCM (nn_os_vtcm_hmx_acquire),
  and tools/sim_trace.py decodes VTCM as pool 1 plus the crouton block tables. Only pool 5 (consts),
  pools 3/4 and the 1.77 MB spill area are DDR. Our resident port puts pool 1 and every crouton in
  uncached DDR, which is the main reason for 1.7 s vs Model's ~10–15 ms.
- The context embeds QNN's compile options (`strings models/primary/context.bin | grep vtcm_mb`):
  vtcm_mb=8, hvx_threads=4, hmx_type=fg, enable_dma_move=true, dma_prefetch_distance=4,
  delay_dma=true, spill_mode=spill_furthest, autothread=1, tile_height=8.
- QNN runs two run lists at the same time: list 0 (HMX convs, DMA, sync) and list 1 (HVX worker),
  synced by @DmaCheckpointSet. Ops like weights_to_vtcm, activations_to_vtcm_dma, spill/fill are
  its DMA schedule. We run them as synchronous memmoves, serially.
- The extracted leaf is `hmx_v73_*`, but this does not establish the context's target ISA.
  The live Model backend auto-selects V69; its vendor DSP runtime is also labeled V69 (see follow-up).

## Model profiling follow-up (Oct 30, after this handoff)
- Rooted Legacy headset is connected over ADB: serial 2G0YC1ZF860CBG, SoC ID 549 / ANORAK.
- Stock SDK 2.50 `qnn-net-run --profiling_level detailed` is **blocked before context loading**.
  Zero inferences completed; no per-op timing, overlap, or output parity was measured.
  The alternate context was staged but not executed because backend creation already fails.
- SDK auto-detection selects `libQnnHtpV69Stub.so`, not V73. After staging the matching V69
  stub/skel in a private temporary directory, the default unsigned session fails opening
  its shell. The device has `/vendor/dsp/cdsp/fastrpc_shell_3`, but no unsigned shell was
  found in the vendor/odm/system_ext inventory. Both DSP search variables were tested.
- Explicit `pd_session=signed` creates a user PD using the existing vendor shell. With
  `DSP_LIBRARY_PATH` including the private skel directory, it then rejects the SDK skel:
  `signature verify start failed for libQnnHtpV69Skel.so` (RPC 0x80000406).
- Pulled the actual `/vendor/lib/rfsa/adsp/libQnnHtpV69.so`: embedded build
  `v2.25.0.240728104910_97711`, SHA `9b687661f8a2b5c1511a21b324803aa96692a2c4f4a0f73ad18151562eb94d22`.
  Also pulled libQnnBoltnnOpPackageV69.so, libhexagon.so, libhexagon_shim.so,
  libhexagon_skel.so, libcdsprpc.so and the signed shell. MrSystemService maps the Meta
  libhexagon RPC path; no separate host libQnnHtp mapping was present in the snapshot.
- Evidence and SHA manifest: `.work/model-qnn-20261030/`. SDK binaries, contexts and smoke
  inputs remain staged only under `/data/local/tmp/qd-qnn-20261030` on the Model.
  Runtime perf/VTCM configuration was not recovered; the compile-time 8 MB request remains known.
- Next profiling route: inspect/use the existing signed Meta `hexagon_nnOssQnn*` path with
  its QNN 2.25 runtime, or obtain a compatible unsigned shell/runtime setup. Do not infer
  real ISA solely from V69 naming/auto-detection, and do not change properties/reboot casually.
  No firmware, service configuration, Frame module, or executor binaries changed in this follow-up.

## Next steps (in priority order)
1. **Model baseline and logical-op profiling completed** through the signed Meta
   ExecuTorch path (see continuation above). Continue with runtime perf/VTCM
   configuration and arithmetic divergence localization if needed. The new
   ~9.7 ms measurements cover the inner DSP program plus RPC, not the outer
   ARM/camera path; they do not retroactively identify the old 10.5/14.3 ms scope.
2. **VTCM ownership on the Frame.** Only 0x2d0000 of the 8 MB VTCM is proven safe. Inventory what
   else uses VTCM (dsp_service; the eye client is paused during sessions). Adopting QNN's model
   (pool 1 in VTCM, DMA for to/from/spill/fill) needs all 8 MB.
3. Dynamic weight / 16-bit fallback overheads are now reduced in an opt-in
   engine (see latest continuation). Do not spend more time on the original
   warm malloc walk: the new weight path removes all 400 warm allocations.
4. Cached-pools ablation (mutable pools C6 to cached + dccleaninva). HVX bypasses L1, so clean before vector reads.
5. Native HMX bias correction (one-sided error suggests an offset; could make native exact, ~100 ms hybrid).
6. C7 paired rerun complete: exact on random/smoke, no useful speed gain.

## Procedures
- Before DSP work: check for other clients (`pgrep -af "qwen|chat|verify_"`), then
  `~/frame-dsp/depth-model/.venv/bin/python frame/bench/depth_state.py`, which should show [1,0,1], callback 0, regions zero.
- Load: `echo frame1 | sudo -S insmod frame/kmod/cdsp_mem.ko size_mb=128`, then `frame/bench/chunk_map.py`.
- Validate: `OPENBLAS_NUM_THREADS=1 .../python frame/bench/verify_resident_graph.py --model primary --case random --repeat 2 --report frame/build/X.json`
  (omit --case for all 27). Prefix bisection per fresh session: frame/bench/bisect_resident.py MODEL CASE 3,4,8.
- Build engine/bootstrap/probe (Mac): `sh frame/build_resident_dsp.sh` (docker, SDK 6.6).
  It saves mode-specific bootstraps/probes; only C0 updates ordinary default filenames.
  For C7, pass `--engine frame/build/qd_graph_dsp-cached.elf --bootstrap frame/build/qd_graph_boot-c7.elf`.
- Build the module on the Frame: use a scratch dir with a Kbuild containing only cdsp_mem
  (~/frame-dsp/claude-scratch/kmod); the Frame kmod dir lacks the other module sources.
- Unload when done: `sudo rmmod cdsp_mem`. Reboot only if the chunk map is bad or the DSP is stuck:
  `ssh frame "echo frame1 | sudo -S reboot"` (interrupts headset/camera; ask the user first).
