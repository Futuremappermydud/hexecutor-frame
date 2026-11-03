# Occlusion issues: investigation and resolution plan

This investigation covers the failures, tradeoffs, and measurement limitations
in [occlusion_benchmark.md](occlusion_benchmark.md). It examines the baseline
implementation and replays the complete candidate recording `20261007-183447`
on Frame. The numbered findings below describe the baseline investigation.
The implementation update records subsequent code corrections and engineering
validation; the original acceptance criteria still require controlled scene and
wearer measurements before claiming the visible occlusion problems are solved.

## Live candidate rejected; previous build restored

The wearer reported that the combined candidate looked much worse, specifically
more holes/missing occlusion and more flicker/trails. It was stopped and the
original binaries/shaders restarted with legacy temporal settings. The active
baseline binaries were hash-verified against
`/tmp/frame-occlusion-integrated-20261007`; corresponding sources were restored
from that directory too. Local experimental sources and the isolated candidate
build remain available for diagnosis. Passing engineering tests did not establish
visual acceptance, and the combined candidate is not the accepted live version.

Two identical recorded inputs now isolate the spatial regression. Original
completion reproduced every supported/unsupported pixel in the recording:

| Frame | Original pre-history depth support | Changed plane only | Changed hole pass only | Combined candidate | Original coarse mesh area | Combined subdivided mesh area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 70 | 28.69% | 26.24% | 27.63% | 23.19% | 23.23% | 16.84% |
| 210 | 31.23% | 28.74% | 29.98% | 25.56% | 26.70% | 19.46% |

Mesh columns use pre-history depth and camera-image area; they omit confirmation
and display-time expiry. Combined spatial changes therefore lose about 27% of
mesh area even after subdivision in these two frames. The changed-hole-only
variant uses original planes, smoothed measured brackets and guide barriers,
with no new plane veto mask; it isolates bracket/guide restrictions and is not a
complete proposed replacement. The combined changes lose more coverage than
changing either pass alone. This is a real support regression, not depth ground
truth or a count of correct/incorrect masks.

The display-time expiry simulations show additional losses while waiting for the
next worker publication. With roughly 100 ms between observations and roughly
100 ms of publication/display age, a 200 ms support budget leaves little time
for held patches; their expiry can plausibly add flicker. That mechanism is not
an independently timed measurement of the wearer's observed trails.

Candidate production logs reported roughly 7–9 Hz depth updates and publication
age medians near 100–111 ms. Initial restored windows reported roughly 11 Hz and
75–76 ms medians. These sequential scenes differ, so the numbers cannot establish
the causal cost of an individual stage. Render pacing stayed near 36 FPS. On
identical inputs on the local Mac, warmed completion+hole medians were 8.4/8.6 ms
for the original and 10.5/10.5 ms for the candidate. Those local timings are not
Frame production timings.

Evidence: [spatial ablation](../artifacts/occlusion_investigation/20261007/regression/ablation.csv),
[reproduction instructions](../artifacts/occlusion_investigation/20261007/regression/README.md),
[candidate live log](../artifacts/occlusion_investigation/20261007/live-runtime.log),
and [restored live log](../artifacts/occlusion_investigation/20261007/restored-runtime.log).
The next live comparison must isolate one change and measure support/stability
before combining restrictions or selecting expiry policy at this worker rate.

## Matcher-only follow-up

The subsequent [stereo matcher experiment](stereo_matcher_experiments.md) compares
the current GPU behavior, edge-aware aggregation and true four-path SGM on nine
recorded pairs and controlled synthetic scenes. Guarded SGM improves mean
post-speckle depth support from 9.697% to 10.483%, but synthetic weak-texture
errors remain substantial and clean-plane subpixel MAE can worsen. Its CPU
reference costs about 675 ms on Frame and is not a live replacement. No matcher
change has been deployed; the restored baseline remains active. The new tools
and measurements are independent of the rejected completion/expiry candidate.

## Implementation update

Three parallel implementation tracks corrected completion provenance, support
expiry, and stereo diagnostics. Mesh subdivision and recording/pacing tools
were integrated alongside them. The wearer clarified that virtual objects
reappear through irregular streaks across surfaces, without identifying a
specific camera-image location. Exported frame 70 already has widespread
irregular missing regions in stereo support; cleanup, confirmation, and meshing
remove further support. Its luma looks continuous on visual inspection. This
supports investigating stage losses, but does not identify the wearer's exact
streak or validate all detile/rectification rows.

Implemented corrections:

- Plane completion returns overlap-conflict, contradictory-support, and boundary
  vetoes plus measured/plane/hole provenance. Hole filling preserves existing
  completed pixels while using only smoothed measured brackets, with guide and
  veto barriers. Connected reach now follows actual allowed edges. Synthetic
  curved input formerly reopened 130 of 209 conflicting pixels; the corrected
  hole pass reopens zero. Depth, veto, and provenance agree under one/eight
  OpenMP threads.
- Rejected coarse mesh cells subdivide into independently supported one-pixel
  cells. All covered samples must agree; isolated holes, shared-edge gaps,
  silhouettes, depth steps, and thin objects are not bridged. Mixed-cell area
  and coverage masks are shared by analyzers and exports. Worst-case vertex
  capacity accounts for all four subcells.
- History transports prediction age, source-anchor support age, direct-measurement
  age, and evidence. Inferred support refresh requires declared measured-anchor
  age. Each mesh cell carries its oldest covered sample's remaining support
  lifetime to both triangles. The fragment shader rejects expired triangles at
  predicted display time. The mesh-level deadline uses the latest surviving
  triangle, with the independent 500 ms capture-freshness guard. Tracking loss,
  capture disable, or worker failure clears publication; generation tokens reject
  computations spanning disable/re-enable and restart confirmation.
- An initial whole-mesh earliest deadline was rejected during integration:
  replay p05/p50/p95 remaining life was 1.66/30.15/48.53 ms from capture end,
  often shorter than processing/publication age. Per-triangle expiry preserves
  fresh regions while retiring old ones. GPU readback tests verify mixed-age
  regions and continued rendering through worker stalls.
- Compatible stereo eyes share one command submission and fence wait. Views,
  framebuffers, and immutable per-target depth descriptors are cached until image
  destruction or render-pass/format change. GPU tests verify distinct eye images,
  shared array layers, cache reuse/invalidation, and two submissions/waits becoming
  one. The demo also avoids copying the camera mesh before replacing it with its
  fixed virtual panel geometry.
- Opt-in shader variants export disparity, winning cost, uniqueness margin,
  stage-local match/LR/range/support rejections, downstream masks, provenance,
  and history events. Full luma replay exactly reproduces speckle-cleaned measured
  float depths across all 294 frames. Marked optical-Z rectangles report missing
  fraction against all marked pixels and accepted-depth error separately.
- Scalar worker/layer/application render profiles separate blocking stages.
  The lossless v2 recorder has a configurable frame budget and the guided script
  checks storage before launch. Exhaustion, drops, gaps, and incomplete footers
  remain explicit failures. Legacy remains the default temporal mode.

Representation-only replay on **unchanged recorded pre-history inputs**, candidate
settings `2,2,3,200,250,250`:

| Phase | Original coarse mesh area | Subdivided mesh area |
| --- | ---: | ---: |
| Still | 21.31% | 23.81% |
| Head turn | 19.38% | 21.96% |
| Hand motion | 16.94% | 18.90% |
| Recovery | 15.40% | 17.42% |

These are supported camera-image areas, not independently correct display-eye
masks. They isolate meshing; they do not include the stricter completion change.
That change has a measurable coverage cost: still pre-history input decreases
from the original recording's 31.69% to 25.71%, and confirmed depth is 22.07%
under the stated candidate settings. The new-plane/old-bracket comparison yields
29.21% still pre-history support; it is not the original complete algorithm.
Removing unsafe inferred chaining and protected-boundary growth must not be
presented as a free coverage improvement.

Full corrected luma/completion/history replay, including conservative subdivision:

| Phase | Geometric mesh | Mesh surviving simulated 75 ms display age | Mesh surviving simulated 100 ms display age |
| --- | ---: | ---: | ---: |
| Still | 15.52% | 14.49% | 13.12% |
| Head turn | 13.56% | 12.52% | 11.35% |
| Hand motion | 12.41% | 11.21% | 10.26% |
| Recovery | 12.87% | 11.52% | 10.86% |

The lower geometric area reflects the completion corrections, not a failure of
subdivision. Simulated ages are explicit scenarios, not new measured production
publication/display latency. These are camera-view proxies; there is no
independent depth or submitted-eye mask ground truth. The stricter pipeline has
not established a wearer-visible coverage improvement over the original.


Artifacts: [stage masks, events, manifests and summaries](../artifacts/occlusion_investigation/20261007/stages/)
and [representation replay](../artifacts/occlusion_investigation/20261007/mesh-replay.txt).
See [stage definitions and region format](occlusion_stage_diagnostics.md),
[completion contract](occlusion_completion.md),
[renderer lifecycle/timings](occlusion_render_submission.md), and
[application pacing diagnostics](occlusion_performance.md#application-render-pacing-diagnostics).

Build and verify on Frame (no camera devices are opened by these tests):

```sh
make -j4 occlusion bin/frame_openxr_point_cloud \
  bin/test_occlusion_surface bin/test_occlusion_completion \
  bin/test_occlusion_benchmark bin/test_occlusion_budget \
  bin/test_occlusion_expiry bin/test_occlusion_history_events \
  bin/test_occlusion_capture_generation bin/test_occlusion_renderer \
  bin/test_occlusion_layer bin/test_occlusion_expiry_gpu bin/test_vulkan_stereo \
  bin/analyze_occlusion_mesh bin/replay_occlusion_stages
OMP_NUM_THREADS=8 bin/test_occlusion_surface
OMP_NUM_THREADS=8 bin/test_occlusion_completion
OMP_NUM_THREADS=8 bin/test_occlusion_benchmark
bin/test_occlusion_budget
bin/test_occlusion_expiry
bin/test_occlusion_history_events
bin/test_occlusion_capture_generation
bin/test_occlusion_renderer
bin/test_occlusion_layer
bin/test_occlusion_expiry_gpu
bin/test_vulkan_stereo
```

All eleven regression suites passed on Frame with `-O3 -fopenmp`; the final
worker update also passed capture-generation/fallback, layer-contract, and GPU
expiry readback checks. See [validation output](../artifacts/occlusion_investigation/20261007/validation.txt).

The initial integrated build and replay used an isolated Frame directory. After
the user requested a live run, the validated binaries/shaders and current sources
were deployed and the demo started with legacy temporal settings, completion
enabled, eight OpenMP threads, luma-only capture, and scalar profiling without
image recording. Startup confirmed auto-discovered cameras, capture-time poses,
and active masking. The prior deployment is archived on Frame at
`/tmp/frame-occlusion-pre-investigation-run-20261007.tar.gz`;
[startup/runtime log](../artifacts/occlusion_investigation/20261007/live-runtime.log)
is saved locally. This launch is not a matched wearer quality comparison.

Remaining work is reliable distant stereo/uncertainty
experiments, actual production timing windows, exposure/pairing metadata and
both-eye calibration, marked distances, actual submitted-eye masks, and
independently timed hand response. No controlled surface ground truth or wearer
comparison was invented. Default-policy release remains gated by the original
acceptance criteria.

## Findings established by code and replay

The candidate improves measured contradiction substantially, but most of the
camera view still has no supported mesh. A better contradiction score alone
cannot establish correct occlusion or explain the reported streaks.

The additional replay used the production codec, smoothing, completion,
confidence history (`2,2,3,200,250,250`), and mesh predicates. Local and Frame
SHA-256 hashes matched for the three occlusion headers and the match/filter/
support shader sources. The recording has 294 frames, zero drops, zero sequence
gaps, and zero capture-pose fallbacks according to its original full analyzer.

| Phase | Cleaned stereo pixels | Completed input pixels | Confirmed depth pixels | Mesh area | Cells rejected for missing samples | Cells rejected for depth spread only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Still | 11.70% | 31.69% | 27.59% | 21.31% | 78.47% | 0.08% |
| Head turn | 11.86% | 31.31% | 26.01% | 19.38% | 80.42% | 0.08% |
| Hand motion | 8.35% | 24.82% | 21.67% | 16.94% | 82.90% | 0.05% |
| Recovery | 9.04% | 23.95% | 20.43% | 15.40% | 84.43% | 0.07% |

Pixel columns use the whole 616×616 image as denominator. Rejection columns
use the 307×307 candidate mesh cells; missing-sample rejection takes precedence
over spread rejection. These columns are not interchangeable. In particular,
the 6.28 percentage-point still difference between confirmed pixels and mesh
area is not a count of erroneous pixel deletions: vertices, borders, and covered
cell area differ. It nevertheless establishes a substantial representation
loss. The overwhelming missing-sample rejection suggests prioritizing support
and gap handling before loosening the mesh depth-spread threshold.

For frames 70 and 210, recomputing smoothing → plane completion → hole fill
produced float arrays exactly equal to the recorded pre-history input:

| Frame | Smoothed anchors | Plane additions | Plane overlap conflicts | Hole-fill additions | Hole additions unavailable with measured-only brackets |
| --- | ---: | ---: | ---: | ---: | ---: |
| 70 | 41,589 | 62,166 | 48 | 5,099 | 2,893 |
| 210 | 40,559 | 72,364 | 262 | 5,579 | 3,289 |

The final column compares hole filling on completed depth with hole filling on
smoothed measured depth, counting only newly filled holes. It establishes that
the live hole pass can depend on inferred brackets. It does not establish that
those depths are wrong or count how many conflict pixels it reopens.

Raw results and the diagnostic source are saved under
[`artifacts/occlusion_benchmark/20261007-183447/investigation/`](../artifacts/occlusion_benchmark/20261007-183447/investigation/).
Reproduce this recording-specific diagnostic on Frame without restarting the demo:

```sh
g++ -O3 -fopenmp -std=c++17 -Iinclude \
  artifacts/occlusion_benchmark/20261007-183447/investigation/replay_support.cpp \
  -o /tmp/replay_occlusion_support
OMP_NUM_THREADS=8 /tmp/replay_occlusion_support \
  captures/occlusion/20261007-183447/recording.bin
```

The diagnostic assumes this verified complete, gap-free, capture-pose recording.
Use the regular analyzer first to validate other recordings. It reports aggregate
support; neither this replay nor the original report identifies the wearer's
specific streak or provides independent scene distances.

## 1. Depth disappears beyond a few feet

**Established mechanism.** There is no few-feet cutoff. The active Vulkan
`stereo_match.comp` searches disparities 1–95 with subpixel refinement;
`stereo_filter.comp` accepts metric depth from 0.25–5 m after bidirectional
consistency. `stereo_sgm.comp` is not the active worker matcher: the production
Vulkan path uses census, sparse five-position aggregation, winner selection,
consistency, and neighboring support. This distinction matters when choosing
the next algorithm.

For `Z = K/d`, with `K = 24.282608`, the local sensitivity is
`|delta Z| ≈ Z²/K × |delta d|`. At 3 m, one disparity pixel corresponds locally
to about 0.37 m; at 5 m, about 1.03 m. The 4% depth agreement tolerance corresponds
approximately to 0.32 disparity pixels at 3 m and 0.19 at 5 m. Noisy small
disparities can therefore survive the 1.25-pixel left/right test but fail later
spatial, plane, or temporal agreement. This is a plausible range-dependent
failure mechanism, not proof of the cause on a particular wall.

**Resolution.** Add opt-in intermediate GPU readback and rejection maps to
`VulkanStereoPipeline`: left/right disparity, best cost, uniqueness margin,
left/right failure, range failure, and support failure. Replay recorded luma
through the current shaders, then export speckle, smoothed, completed,
pre-history, confirmed, and mesh masks. The existing evaluator exports only
cleaned and final depths; the temporal analyzer does not rerun stereo.

Measure whole marked surface regions at independently known 1, 2, 3, 4, and
5 m distances, with textured furniture and plain walls evaluated separately.
Use camera-axis depth or convert the measured target position into optical Z;
a tape's slant distance is not automatically stereo Z. Report missing-pixel
fraction using the marked region as denominator, and depth error on accepted
pixels separately. Missing surfaces must not disappear from the statistics.

Compare edge-guided aggregation and a bounded multi-path SGM implementation
on identical inputs, with improved subpixel fitting and match uncertainty.
Choose the least expensive variant that improves distant measured support
without adding foreground/background leakage. If uncertainty dominates cleanup,
use disparity/inverse-depth uncertainty for neighborhood agreement while retaining
independent discontinuity tests. Keep the final occlusion decision conservative;
uncertainty is not permission to use a confidently wrong foreground depth.

**Acceptance.** Marked distant regions gain reliable measured and final mesh
coverage, accepted depth error does not worsen, and false foreground masking
does not increase on silhouettes, repetitive textures, or blank images. Re-run
GPU fractional-disparity and ambiguity tests over the small-disparity range.
Increasing maximum range, expiry, or blindly relaxing all tolerances is not a
solution to absent reliable measurements.

## 2. Long streaks and temporary gaps, including nearby surfaces

**Established mechanism.** `occlusionSurfaceIndices()` validates all nine samples
of a 2×2 cell. One missing interior or shared-edge sample suppresses an entire
cell, sometimes adjacent cells. Completion intentionally leaves unsupported
regions and conflicting overlaps unknown. Nearest-pixel forward reprojection
can introduce additional holes and collisions. None of these facts proves
where the reported streaks originate.

**Resolution.** Export aligned left/right luma and stage masks for a short
stationary sequence and a turning sequence. Have the wearer identify the
streak location and orientation. Track its first appearance across capture,
matching, cleanup, completion, confirmation, and meshing. Compare bands with
luma discontinuities, rectification validity, tile boundaries, and reprojection
holes; count invalid-input versus spread mesh rejections per marked region.

If bands are already present in either luma image, fix the detile/import or
rectification path first. Verify UBWC rows and strides, LUT validity, the full
stereo extrinsic transform including right-sensor parity, and epipolar residuals.
If bands first occur in disparity or cleanup, use the matcher work above and
test whether component rejection is removing real thin structures.

For gaps introduced by meshing, retain valid 2×2 cells but subdivide rejected
cells into one-pixel cells with independent support/boundary validation. Emit
only subcells whose complete covered samples agree; do not connect triangles
across an invalid sample or depth discontinuity. This can recover valid portions
of a rejected coarse cell without filling the actual unknown strip. It will
not solve a wide upstream gap. Implement in `occlusion_surface.hpp`, mirror
the area calculation in the analyzer, and measure the additional GPU load.

**Acceptance.** Stage images identify the earliest cause of each marked streak.
Nearby valid subregions recover mesh coverage while synthetic isolated holes,
open silhouettes, depth steps, and thin-object tests show no bridging. A wearer
confirms that stationary and moving-head streaks improve on the same scene.

## 3. Completion can leave holes or create false occlusion

**Established mechanism.** Plane completion requires at least 12 distributed
anchors, 90% agreement, a bounded residual, a convex hull, 24-pixel connected
reach, and separation from contradictory anchors. Luma edges split components;
same-luma depth boundaries can remain indistinguishable where measurements are
absent. Wrong stereo anchors can fit a wrong plane.

There is also a concrete contract mismatch: `cloud_worker.cpp` passes completed
depth to `fillOcclusionHoles()`. That function has no guide or rejection mask and
can use inferred brackets, or refill a gap that completion intentionally rejected.
Documentation describing measured-only brackets overstates the protection.

**Resolution.** Separate the hole-fill output from its bracket source: preserve
completed pixels, but search only original smoothed measured anchors. Return
explicit completion veto masks for overlap conflict, contradictory support,
and protected boundaries, and prevent the hole pass from overriding them.
Carry inferred provenance through both completion passes. Compare the current
path, measured-bracket path, and completion-off path on the same recordings;
some current coverage will be lost when unsupported inferred chaining is removed.

Improve under-completion only after rejection maps show which restriction is
responsible. Prefer robust plane fitting and better anchors over increasing
reach or permitting cross-boundary growth. Evaluate color-guided boundaries
only as a measured tradeoff against the cost of restoring UV capture. Neither
color nor a longer reach resolves a completely unobserved same-color object.

**Acceptance.** Add cases with inferred brackets but no measured enclosure,
overlapping contradictory planes, and luma/depth barriers. No vetoed gap is
reopened; existing measured depths remain intact; deterministic output remains
independent of OpenMP thread count. Scene coverage and false masking are both
reported, including curved surfaces and same-color moving foregrounds.

## 4. Confirmation delays new hands and clears leave visible holes

**Established mechanism.** At 9.79 Hz the mean update interval is about 102 ms.
The candidate needs two measured observations or three inferred observations.
From the first qualifying sample, uninterrupted confirmation therefore adds
roughly 102 or 204 ms, plus processing/publication and display delay. A hand
entering just after a sample can wait another interval. Baseline four-observation
inferred confirmation adds roughly 306 ms. Dropouts and reprojection loss can
extend these delays; these are estimates, not measured hand response times.

Confidence clears an old contradicted occluder immediately when measured evidence
is available, but confirmation can leave the replacement invisible temporarily.
Legacy can retain contradictory geometry. Longer expiry also permits another
50 ms of unsupported retention. Missing reliable stereo is invisible to the
contradiction proxy.

**Resolution.** Improve worker update rate before selecting new observation
counts. Record first reliable observation, confirmation, contradiction clear,
and expiry events with timestamps and evidence class. Preserve immediate
measured contradiction clearing and keep inferred depth out of the fast path.
Evaluate a fast foreground path only for independent, strong bidirectional
stereo support with tested dynamic-object rejection. Spatial agreement alone
does not make `StrongDepth` a calibrated confidence probability.

Do not simply switch to `strong-first` or `two-confirm`: the sweep already shows
their transient-rejection or stability cost. Tune counts and support durations
at the new update rate, testing interrupted observations and changes in evidence
class. Reprojection must never refresh support age.

**Acceptance.** A repeatable hand crossing/withdrawal test measures first visible
occlusion, last stale mask, and false masking. Include low-texture hands, wrong
measurements, missing measurements, and nearby moving objects. Report latency
distributions alongside supported area and still stability.

## 5. A stopped worker can leave stale geometry visible beyond history expiry

**Established mechanism.** History ages advance only when the worker processes
a frame. `openxr_occlusion_layer.cpp` permits the published GPU mesh until its
capture-end age reaches 500 ms. Thus a worker stall can keep a mesh visible
beyond the 200–250 ms history expiry; even during normal updates, the renderer
does not know each triangle's last-support deadline. A 200 ms history reset on
the next frame does not clear a mesh while no frames arrive.

**Resolution.** Carry absolute last-support/expiry times from history into the
published mesh. Expire triangles conservatively using the earliest supporting
vertex deadline, evaluated at display time. A practical first step is a whole-
mesh deadline based on its earliest expiry, with later per-triangle expiry to
avoid throwing away fresh regions. Clear on tracking/capture disable and worker
failure. Keep a global freshness cap as a secondary guard.

Separate actual stereo support time from agreeing inferred completion time;
inferred agreement currently refreshes age too. Bound how long repeatedly
completed pixels can preserve unsupported geometry using their anchor support
and provenance, rather than treating each plane prediction as a new measurement.

**Acceptance.** Stall the worker after publishing a mesh and verify mask readback
clears at the declared deadline while rendering continues. Test mixed-age
vertices, tracking fallback, repeated inferred support, and restart. No
reprojection or repeated rendering extends the deadline.

## 6. Still instability and recovery are worse for some confidence settings

**Established evidence.** In the candidate recording, still coverage change is
15.15% versus legacy's 14.81%, hand motion 15.80% versus 14.68%, and recovery
19.47% versus 17.32%. Recovery mesh is slightly lower too: 15.40% versus 15.45%.
The candidate is not an improvement on every metric. These are depth-support
changes, not eye-mask flicker.

**Resolution.** Attribute each support change to newly observed depth, pending
confirmation, measured clearing, expiry, projection rounding, or collision.
Repeated nearest-pixel forward transport may fragment support, so evaluate
surface-aware reprojection or backward sampling with explicit visibility,
discontinuity, age, and provenance checks. Do not dilate across silhouettes to
hide gaps. Optimize static and dynamic regions separately only when reliable
motion evidence supports the distinction.

**Acceptance.** Repeated still and recovery recordings show stable supported
area without increasing stale occlusion. Actual per-eye masks confirm that any
proxy improvement corresponds to less visible flicker. Keep legacy as default
until a matched live comparison establishes the intended tradeoff.

## 7. Head-turn alignment, stereo pairing, and absolute calibration are unproven

**Established limitation.** The worker locates the head at the DMA-read midpoint,
not exposure. All slots are transferred in one submission, but left/right slot
selection independently chooses the last changed slot in scan order. Hash
changes prove new content, not newest exposure or synchronized stereo. Some
aligned table edges and zero pose fallbacks do not validate camera-to-eye
calibration or stereo synchronization.

**Resolution.** Seek camera sequence/exposure metadata through the existing
XRService/compositor path and log selected slot identity, frame sequence,
exposure time, and alignment source. Pair corresponding exposures and locate
the head at that time. Keep auto-discovery and DMA-BUF ingestion; do not open
camera devices or hardcode descriptors. If authoritative timestamps are
unavailable, estimate and report time-offset uncertainty using a controlled
moving target and repeated head turns; label that calibration as an estimate.

Validate both-eye reprojection against marked static targets over image
position and distance. Check rectified vertical correspondence first, then
metric scale, camera-to-head, and eye transforms. Preserve the full right-sensor
rotation already used by production rectification. If static error remains,
correct calibrated transforms; if error scales with angular velocity, correct
timing rather than applying an arbitrary spatial offset.

**Acceptance.** Report static pixel residuals and motion-dependent residuals
for both eyes, with independent distances and known tracking space. Explicitly
state exposure synchronization status and remaining uncertainty. This requires
new controlled captures; the old recordings cannot recover missing timestamps.

## 8. Worker throughput, publication age, and render pacing

**Established evidence.** Candidate still-stage p50/p95: capture 25.34/32.28 ms,
stereo 20.75/24.23 ms, completion 21.96/25.65 ms, history 9.11/16.72 ms,
holes 5.15/9.04 ms, mesh 4.24/7.35 ms. Publication age is 90.58/101.31 ms
from read midpoint, and overall throughput 9.79 Hz. Stage percentiles must not
be summed to produce an end-to-end percentile. Completion is substantial;
meshing is not the dominant worker stage in this run.

Luma-only capture and cached completion connectivity are already implemented.
The documented gains are partial; repeating those changes is not a new fix.
The roughly 36 FPS demo rate remains unexplained. Masking submits and waits on
a fence separately for each eye; it also creates/destroys image views and
framebuffers per call. This is a concrete cost to investigate, not a proven
explanation of the frame rate.

**Resolution.** Profile non-recording worker windows and separately time
`xrWaitFrame`, application rendering, mesh upload, each-eye mask CPU/GPU work,
fence wait, and downstream `xrEndFrame`. Optimize worker allocation churn using
reusable frame and per-thread tile scratch; preserve deterministic reduction.
Optimize bounded hole searches and history transport after profiling their
substeps. Profile capture transfer/hash/selection costs before changing slots;
fewer slots without freshness metadata may select stale images.

For render submission, cache swapchain views/framebuffers for their lifecycle,
record eligible eyes together, and wait once before releasing runtime images.
Consider indexed mesh uploads to avoid repeating transformed vertices. Any
asynchronous change must preserve queue synchronization, image lifetime, and
the existing deferred-release contract. Keep all depth/completion work in the
background worker.

**Acceptance.** Compare independent production p50/p95 windows with recording
disabled, then rerun the full recorded quality benchmark. Show actual render
frame-time distributions and the cost of each layer step; preserve GPU mask and
layer contract tests. The 70–90 FPS guidance is a target, not current evidence
or a guaranteed result of these optimizations.

## 9. Benchmark blind spots and incomplete recordings

**Established limitation.** The benchmark has no depth ground truth, display-eye
mask, sensor exposure time, or independently timed hand motion. Reliable-pixel
contradictions omit unsupported regions and are not mask false positives.
Same-input replay isolates temporal policy, but does not time its variants or
compare stereo algorithms. Recording copies happen partly outside individually
timed stages and disk activity affects subsequent throughput.

**Resolution.** Extend offline replay with stage rejection maps and marked-region
ground truth. Add opt-in per-eye mask capture with the exact submitted view/FOV,
virtual depth, display time, and mesh version. Evaluate false masked virtual
pixels and real surfaces that failed to mask separately. For visible hand timing
and motion alignment, use an independently timed observation of the rendered
result; software publication timestamps alone cannot establish display latency.
Time replay variants separately after warmup and measure production runs without
recording. Use matched live legacy/candidate runs in the same lighting and panel
arrangement, repeated and with order alternated.

Keep bounded asynchronous recording and the existing error on drops, frame gaps,
truncation, or incomplete footer. At more than roughly 17 Hz a 512-frame budget
cannot cover 30 seconds, so higher throughput requires a configurable duration/
frame/storage budget before rerunning the protocol. For substantial rate gains,
support chunked recording or lossless compression off the worker and check disk
space in advance. Do not silently shorten phases or accept lossy data. Preserve
v2 reading if optional diagnostics require a new recording version.

**Acceptance.** Full phase coverage, zero drops/gaps, reproducible manifests with
source/binary/shader/calibration hashes and live settings, independently measured
depth and mask errors, and actual dynamic response measurements. Maintain the
clear distinction between engineering smoke tests and wearer validation.

## Implementation order and release decision

1. Add stage rejection/support diagnostics and controlled region annotations;
   locate the nearby streak and measure range-dependent loss. Correct the
   completion/hole-fill contract and renderer expiry semantics with focused tests.
2. Improve reliable stereo support and recover valid portions of rejected mesh
   cells. Evaluate each change independently on identical recorded inputs.
3. Reduce measured worker/render costs, then retune temporal confirmation at
   the resulting update rate. Keep the render loop independent of depth work.
4. Validate exposure/pairing and both-eye alignment; add mask and dynamic-response
   measurements. Repeat full matched legacy/candidate headset evaluations.
5. Change the default only after coverage, depth/mask correctness, still/recovery
   stability, motion alignment, hand response, and timing all have explicit
   measured acceptance results. Agree numerical application targets before that
   decision; the current report does not define them.

The replay establishes where aggregate support is lost and identifies two
implementation contracts needing correction. It cannot promise that one matcher
change will solve an unlocalized visual streak, nor that unsupported walls will
become measurable without additional evidence.
