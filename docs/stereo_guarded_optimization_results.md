# Guarded optimization implementation and native results (2026-10-07)

The first four stages of the [plan](stereo_guarded_optimization_plan.md) are
implemented: timestamp/host profiling, independent arm-limit and winner-neighbor
variants, and configurable row/workgroup/yield sweeps. Both optimizations remain
**opt-in**. Live defaults retain the original matcher, 8×8 workgroups, eight-row
batches and 3000 µs yields. No quality-changing search or capture changes were made.

## Implemented contract

`GuardedStereoOptions` selects independent Vulkan specialization constants for
arm precomputation and neighbor caching. Specialization also selects local X/Y;
initialization rejects invalid row divisibility and device-limit violations.
There is at most one compute submission outstanding: each batch uses the existing
fence before the worker yields. The render thread never runs stereo work.

The new arm pass packs four extents and anchor-stop bits into one R32_UINT pixel
per view. Unknown centers inherit the first known label independently along each
arm. Matching still stops on the first insufficient joint descriptor mask and
computes texture extrema from selected samples only. A luma/bounds rejection at
the next sample takes precedence over either view's anchor rejection, preserving
selected-evidence diagnostics as well as cost. The center rejection is unchanged.
Neighbor caching retains ascending search, strict winner comparison, top-four
competitor insertion and the original parabola arithmetic.

GPU query intervals cover upload, census, seeds, anchors, masks, optional arms,
each match batch, filter/support and download. Queue timestamp-valid bits and
period are checked; unsigned subtraction is masked to handle counter wrap.
Queries use bottom-of-pipe boundaries and are read after the normal final fence,
without WAIT_BIT or extra profiling submissions. Unsupported queries retain host
profiling. GPU intervals include scheduling/preemption inside their boundaries;
they are not hardware occupancy counters. Wrap handling assumes an individual
interval is shorter than one counter period.

`lastProfile()` distinguishes GPU intervals from host upload, exposure histogram,
command recording/submission, fence wait, actual host yield and readback.
`pairMs` covers the process call (before logging), **not** capture/rectification,
speckles, meshing or publication. `lastComputeMs()` preserves its prior
recording-through-final-fence meaning. Allocation bytes count pipeline-owned
Vulkan memory; they exclude driver allocations and host vector capacity.
The opt-in worker event report records attempts, duplicates, unavailable/failed
capture, failed matching, cancelled publication, history resets by reason,
publication count, capture generation, and process RSS high-water. Render and
worker timing windows now include p99; mesh expiry is logged separately.

## Native evidence

Isolated build: `/home/steamos/stereo-guarded-opt-20261007` on Frame's Adreno 750.
Local evidence: `artifacts/stereo_guarded_optimization/20261007/`.
`parity-optimized.log` covers fourteen cases (seven scenes including recorded
frame 70, each with integer and parabolic refinement), CPU stage parity,
optimized/control exact GPU diagnostics, batched/unsplit depth equality, and two
deterministic repeats per case. All passed. CPU float/support comparisons retain
the previously documented tolerance and explicitly checked threshold-only flips;
**optimized versus original GPU comparisons require exact equality**, including
support rejection. No new threshold flips were permitted between GPU variants.

The saved-pair sweep uses frame 70, one warmup and ten measured pairs per variant
in each of two rounds, reversing variant order on round two. All 140 measured
outputs equaled the unsplit original GPU output (83,704 valid depth pixels).
Manifest hashes identify inputs, shaders, source, calibration and executable.
These are repeated saved-input process times, not fresh capture/publication ages.
The existing panel process was running but reported zero rendered frames during
inspection: these trials **do not establish focused graphics contention parity**.

| Variant | Pair p50 / p95 / p99 (ms) | Match-batch p50 / p95 / p99 (ms) |
| --- | --- | --- |
| Original control | 705.787 / 709.434 / 709.434 | 5.253 / 5.870 / 5.894 |
| Arms only | 586.778 / 591.009 / 591.009 | 3.790 / 4.432 / 4.467 |
| Neighbors only | 695.563 / 698.029 / 698.029 | see sweep CSV |
| Arms + neighbors | 585.177 / 587.013 / 587.013 | 3.784 / 4.424 / 4.465 |
| Arms + neighbors, 1000 µs yield | 435.658 / 438.334 / 438.334 | 3.800 / 4.449 / 4.510 |
| Arms + neighbors, 16-row batch | 465.767 / 472.997 / 472.997 | 7.748 / 9.728 / 9.885 |
| Arms + neighbors, 16×4 local shape | 575.068 / 576.998 / 576.998 | 3.693 / 4.334 / 4.386 |

Percentiles use floor index `q*(n-1)`; twenty pair samples are insufficient for a
strong p99 claim. Combined optimization saves about **17.1%** at the original
yield schedule. The 1 ms-yield experiment saves about **38.3%**, but its render
tails have not been measured. The arm pass costs about 1.16 ms; census ~0.28 ms,
seeds ~10.7 ms, upload ~0.03 ms and normal depth download ~0.21 ms. Transfer and
histogram work are not a reason to move rectification/histograms to the GPU yet.

Normal pipeline Vulkan allocation is 32,054,656 bytes in these builds. The two
packed arm images add 3,219,456 allocated bytes including image padding; they are
currently allocated in control mode too, keeping descriptor layouts consistent.
Normal operation does not allocate diagnostic images. No disparity cost volume
or shared-memory tile was added.

Build/check evidence:

- Native `make -j4 all occlusion`, guarded benchmark/parity binaries, and relevant
  benchmark/history/capture-generation/guarded CPU tests passed. Layer-contract
  and GPU expiry regression tests also passed.
- Native Release CMake configure/build passed for runtime, point-cloud and layer.
  CMake now includes the guarded source and shaders, previously missing there.
- Guarded workflow Python tests passed (five tests); three native-analyzer replay
  regressions passed for long retention, pose loss, and invalid/mixed policies.
  Benchmark scripts compile.

## Reproduce

Build and run from the project directory on Frame:

```sh
make -j4 all occlusion bin/test_vulkan_guarded bin/benchmark_stereo_guarded
OMP_NUM_THREADS=8 FRAME_GUARDED_PROFILE=1 FRAME_GUARDED_ARMS=1 FRAME_GUARDED_CACHE=1 \
  ./bin/test_vulkan_guarded shaders frame70_left.pgm frame70_right.pgm
python3 scripts/benchmark_stereo_guarded.py frame70_left.pgm frame70_right.pgm \
  artifacts/guarded-sweep --pairs 10 --rounds 2 --scheduling
```

The sweep writes per-pair/per-pass CSV, logs, hashes and p50/p95/p99 summaries.
Failed output comparisons are recorded and fail the run. Output directories must
be new. Environment controls (also available through `GuardedStereoOptions`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FRAME_GUARDED_PROFILE` | 0 | GPU queries and per-pair profiling log |
| `FRAME_GUARDED_ARMS` | 0 | Precomputed bilateral arm extents |
| `FRAME_GUARDED_CACHE` | 0 | Cached adjacent winner costs |
| `FRAME_GUARDED_BATCH_ROWS` | 8 | Rows per interleaved submission |
| `FRAME_GUARDED_YIELD_US` | 3000 | Host pause after prerequisites/each batch |
| `FRAME_GUARDED_LOCAL_X`, `FRAME_GUARDED_LOCAL_Y` | 8, 8 | Match workgroup shape |

For a focused, recording-disabled panel run, use these together with
`FRAME_OCCLUSION_PROFILE=1 FRAME_OPENXR_RENDER_PROFILE=1`, the existing panel
launcher and `FRAME_OCCLUSION_STEREO=guarded`. Do not run competing test workers
while collecting this evidence. Keep the headset focused for ten minutes, then
repeat control/variant order and the still/head-turn/hand-motion/recovery protocol.
The implementation does not infer headset focus or successful visual behavior
from the application's wait-frame frequency.

## Recording and remaining gates

New v2 recordings use a previously unused flag bit to identify guarded matching;
old baseline files remain readable. The analyzer uses the guarded 2500 ms legacy
age for that flag, rejects mixed matcher/invalid confidence flags, and labels the
confidence column as a counterfactual. The baseline stage replayer rejects guarded
recordings rather than silently replaying baseline shaders/ages. The recording
script explicitly selects the matcher and records retention, batch/yield/workgroup
settings, calibration hashes and where to find capture-generation/reset counters.
Old untagged guarded recordings cannot be identified from the binary and must not
be treated as live parity. Generation logs are diagnostic counters, not a new
per-frame generation field in the recording format.

Promotion still requires focused recording-disabled render p95/p99, profiling
on/off cadence checks, sustained ten-minute thermal/memory observations, and human
motion/removal-latency evaluation. The ≥25% fresh-pair improvement and ≤200 ms p95
capture-to-publication goals are **not achieved/validated** by this replay.
Shared-memory tiling/register occupancy investigation remains the next optional
prototype; transfer/rectification and quality-changing searches remain deferred.
Reserved quality seeds were not used and the failed offline quality gate is unchanged.
