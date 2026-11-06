# Guarded GPU matcher optimization plan

Status: instrumentation, arm-limit and neighbor-cache prototypes, and scheduling
sweeps implemented and built, 2026-10-07. See
[implementation/results](stereo_guarded_optimization_results.md) for native parity,
measurements, controls, and remaining promotion gates. Optimize the experimental guarded matcher
without losing its current correspondence behavior or headset render cadence.
Quality-changing experiments remain separate from performance work.

## Measured starting point

| Path | Observed result | Evidence / limits |
| --- | --- | --- |
| Unsplit guarded GPU | 423–785 ms per pair | Fourteen native parity cases; individual host submission-to-fence measurements, including transfers |
| Guarded point cloud, eight-row batches | 758–763 ms per pair, approximately 72 FPS rendering | Short live window; timing includes intentional host yields |
| Guarded panel occlusion | 770–784 ms matcher, 72–87 ms surface construction, approximately 72 FPS rendering | Short live window; scene-dependent coverage |
| Initial unsplit point cloud | Approximately half-second graphics stalls | Separate worker threads did not prevent contention on the same GPU |

See [port validation](stereo_guarded_gpu_preview.md),
[worker profiling](occlusion_performance.md), and logs in
`artifacts/stereo_guarded_gpu/20261007/`. These observations are not sustained
thermal benchmarks, GPU timestamp measurements, or proof of absolute accuracy.
The guarded matcher still fails the complete
[offline quality gates](stereo_guarded_support_experiment.md).

The implementation has seven passes: census, independent baseline integer
seeds, LR-consistent anchors, descriptor boundary masks, guarded matching,
LR/range filtering, and neighborhood support. Exposure percentiles and speckle
cleanup remain on the host. The costly-looking section is `costAt()` in
`shaders/stereo_guarded_match.comp`: each pixel searches up to 95 disparities
in both directions and walks four support arms of up to eight samples per
candidate. That is a source-level hypothesis; measure pass times before
assigning the bottleneck.

## First: make timing and comparisons reproducible

Add optional Vulkan timestamp queries around upload, census, seeds, anchors,
masks, guarded batches, filter/support, and download. Check timestamp support,
valid bits and timestamp period on the selected queue; handle counter wrap.
Read results after the existing fence, without inserting profiling-only waits.
Keep query time separate from CPU recording, fence wait, exposure histogram,
readback, intentional 3 ms pauses, and full capture-to-publication age.

Report p50/p95/p99 for complete pairs, batch durations, publication age and
application frame periods. Track rejected duplicate pairs, capture failures,
history resets by reason, mesh expiry, valid depth area, and memory high-water
marks. Count failed/cancelled frames separately rather than dropping them from
throughput statistics. Use `FRAME_OPENXR_RENDER_PROFILE=1` alongside worker
profiling. Render FPS and fresh depth updates per second are different metrics.

Use three comparison modes:

1. Replay the same saved rectified pairs to isolate matching and exact outputs.
2. Run the panel scene with recording disabled to measure GPU contention.
3. Repeat still/head-turn/hand-motion/recovery sequences to assess visual
   stability and removal latency. Use the same lighting, shader hashes,
   OpenMP setting, scene, and warmed headset; alternate variant order.

Extend the benchmark manifest/replayer before using it for guarded history:
the existing analyzer models baseline legacy/confidence ages, while guarded
live mode uses legacy confirmation with a 2.5-second budget. Record matcher,
batch/yield settings, retention policy, calibration and map generation.
Do not interpret replay under different ages as live parity.

## Ranked optimization work

| Order | Change | Expected benefit to investigate | Contract and risk |
| --- | --- | --- | --- |
| 1 | Timestamp and contention instrumentation | Identify compute versus scheduling versus transfer cost | No output changes; verify profiling does not materially alter cadence |
| 2 | Precompute per-view arm limits | Avoid repeating luma/anchor walks for every disparity | Preserve unknown-anchor label inheritance, borders and bilateral stops exactly |
| 3 | Cache neighboring winner costs | Remove the two full support walks used for parabola refinement | Preserve candidate order, tie handling, competitor exclusion and rounding |
| 4 | Tune batching and local workgroup shape | Reduce submission/yield overhead while protecting graphics | Equal outputs; optimize render tails as well as average pair time |
| 5 | Tile reused fields in shared memory | Reduce repeated image loads | Query device limits; register/shared-memory pressure may reduce occupancy |
| 6 | Improve resource reuse and transfer path | Avoid unnecessary allocation, readback or host copies | Capture pairing, synchronization and external-memory ownership remain correct |
| 7 | GPU rectification / histogram | Remove CPU stages and round trips if profiling justifies it | Rectification coordinates, right-camera parity and integer exposure thresholds require separate parity checks |

### Precomputed arm limits

For each view/pixel/arm, compute the maximal contiguous extent permitted by
that view's luma gates, image bounds and anchor labels. A candidate at disparity
`d` starts from the minimum extent of its reference and target arms. An unknown
center inherits the first known anchor on each arm exactly as today.

This does **not** make aggregation independent of disparity: joint descriptor
masks can have fewer than 24 valid bits at a sample, and that still terminates
the arm. The selected support count, texture extrema, normalized disagreements,
and minimum five-sample requirement must use the actual selected samples.
Retain the center rejection before walking arms. Four byte lengths per view
would cost about 2.9 MiB at 616²; measure whether saved walks repay the new
pass and memory loads. Do not replace bilateral support with a reference-only
window or precompute extrema over samples that were never selected.

### Winner-neighbor caching and scheduling

Store costs adjacent to the eventual winner during the ascending search,
rather than recomputing `bestD-1` and `bestD+1` afterward. Account for a winner
changing on the very next candidate, rejected winners, and boundary winners.
The top-four search implements a competitor outside the winner's ±1 band;
changing rank insertion or early exit can change uniqueness rejection.

Current rendering uses 77 eight-row batches and a 3 ms host pause after
submissions. The pauses alone account for roughly 0.23 seconds per pair,
but removing them can bring back graphics stalls. Test batch heights and yield
durations with identical input/output checks, including individual GPU batch
p95/p99. Bound outstanding compute submissions so graphics gets opportunities
to run. A separate queue/device is not evidence of hardware independence.
Keep the current batched path as a fallback.

### Memory reuse and aggregation alternatives

Cache luma/anchors first, then evaluate census/mask caching. Target-view halos
must cover the disparity range and support radius; a naive shared-memory tile
can exceed the device limit. Inspect occupancy/register use instead of assuming
larger workgroups are faster. Avoid enabling full diagnostic images in normal
operation; introduce a small evidence output only if fusion needs it.

Per-disparity row/column prefix sums could reduce arm summation work, but
joint-mask invalid samples and texture extrema require additional bounded
queries. Treat this as a later prototype. A 616² × 96 × two-view float cost
volume alone is approximately 278 MiB, before evidence, minima or scratch.
Prefer streaming disparity tiles if this approach is attempted.

Preserve direct DMA-BUF access, automatic discovery, Y-only worker capture,
right sensor inversion and `-O3 -fopenmp`. Do not optimize capture by fixing
slot FDs or by assuming only one ring slot contains the newest stereo pair.

## Quality-changing experiments: separate branches and gates

| Experiment | Benefit to test | Failure to guard against |
| --- | --- | --- |
| Coarse-to-fine disparity search | Fewer expensive fine candidates | Missed thin objects/disocclusions; rescale disparity, support and intrinsics correctly |
| Anchor/mesh-guided search ranges with full-search fallback | Reduce search in stable room regions | Self-reinforcing bad geometry; retain independent discovery pixels and contradiction-triggered fallback |
| Smaller/adaptive support or adaptive radius | Reduce arm cost | Lost weak-texture coverage or altered boundary leakage |
| Reduced resolution or keyframe matching | More frequent depth at lower cost | Lost near edges and hand response; temporal duplicates cannot count as new evidence |

Use the current four development seeds and recorded pairs first. Preserve
the reserved evaluation seeds until a candidate passes the frozen development
protocol. Report weak-texture coverage/error, exposure clipping, unrelated-pair
false support, silhouette leakage, and both integer and parabolic refinement.
Recorded room coverage alone is not an accuracy gate. Meshing must not turn
a performance shortcut's guesses into confirmed measured evidence.

## Exit criteria and implementation order

For behavior-preserving work, keep CPU reference and unsplit GPU controls:
exact integer winners/costs/gaps/rejections, anchors, masks and selected
evidence; equal acceptance masks; documented float ULP tolerance; explain any
threshold flips individually. Require batched/unsplit output equality and
repeat determinism on all fourteen native cases.

Proposed performance goals, not measured guarantees: first improve median
fresh-pair time by at least 25% on matched inputs without worsening render
p95/p99 or accepted quality; then target ≤200 ms p95 capture-to-depth
publication to make a short-lived dynamic occlusion path practical. Maintain
at least 70 application FPS in sustained focused runs. Report thermal behavior
over ten minutes, missed-frame fraction and memory bounds. If that latency
target is not attainable, keep dynamic/hand occlusion explicitly limited and
use the persistent room mesh for stable structure.

Implement instrumentation → arm-limit prototype → neighbor-cost cache →
scheduling sweep → shared-memory/resource work. Keep each change independently
revertible with source/shader hashes and a result report. GPU rectification and
quality-changing search experiments follow only when the profile supports them.

The [persistent meshing plan](room_mesh_plan.md) can start after the observation
contract is stable; it does not need to wait for every optimization.
