# Image-guided occlusion surface completion

## Purpose

Local census stereo leaves low-texture interiors sparse. The occlusion mesh
requires all nine samples in each two-pixel cell, so missing depth becomes even
sparser masking. This experiment improves measured stereo acceptance and fills
supported plane interiors without adding a learned model or changing capture,
rectification, or render-thread responsibilities.

The implementation is in `include/occlusion_completion.hpp`, invoked only by the
occlusion branch of `src/cloud_worker.cpp`. Ordinary point-cloud fusion continues
to use measured depth. Raw DMA buffers stay in the production ingestion path.

## Algorithm and limits

* The Vulkan matcher keeps four ranked cost hypotheses. Uniqueness is measured
  against disparities outside the winning `d±1` basin. Four suffice because at
  most three disparities belong to that basin. Cost thresholds, bidirectional
  consistency, and support rejection still apply. This is not full SGM.
* After speckle rejection and spatial smoothing, split overlapping 32/64-pixel
  tiles into four-connected image regions. Adjacent luma differences above
  16/255 or incompatible adjacent measured depths stop a connection.
* Fit affine inverse depth, which represents a slanted 3D plane. Require at least
  12 anchors, a conditioned non-collinear fit, and at least eight pixels of support
  extent in both axes. At least 90% of anchors must agree within 4 cm or 4% of
  range; normalized RMS residual must be at most half that tolerance.
* Fill only inside the agreeing anchors' convex hull, within 24 connected pixels
  of an anchor, and more than eight connected pixels from a contradictory anchor.
  Completed pixels never become fit anchors. Existing valid depths are preserved.
* Average agreeing overlap predictions in inverse depth. Reject incompatible
  overlaps rather than choosing one side. Reduction also retains explicit veto
  bits for overlap conflict, proximity to contradictory support, and protected
  luma/depth boundaries. A proposal from another tile cannot override a veto.
  Measured samples remain intact. Tile reduction is deterministic across OpenMP
  thread counts.
* Run the small enclosed-hole pass once, preserving completed pixels but searching
  only the original smoothed measured depth for its four agreeing brackets.
  Searches stop at vetoes or guide/depth boundaries; vetoed gaps cannot reopen.
  Newly filled holes and plane predictions never become bracket sources. The
  contract records measured, plane, and hole provenance separately.
* Temporal confirmation, age expiry, capture-pose reprojection, and mesh boundary
  checks remain in force. This does not remove the three-observation delay or fix
  the missing sensor exposure timestamp. It does not add a persistent room map.

Completion adds inferred depth, not new measurements. Luma guidance cannot
reliably distinguish color-only boundaries; same-color foreground objects,
curved surfaces, and wrong stereo anchors can defeat the plane assumptions.
Completely unsupported surfaces remain unknown. Large unsupported gaps cannot
be fixed by this method alone.

## Controls and logging

Completion is enabled by default for the occlusion worker:

```bash
./scripts/run_with_occlusion.sh ./bin/frame_openxr_point_cloud --occlusion-demo
# Disable completion while retaining the corrected stereo matcher:
FRAME_OCCLUSION_COMPLETION=0 ./scripts/run_with_occlusion.sh ./bin/frame_openxr_point_cloud --occlusion-demo
```

Worker logs report `measured` smoothed anchors, `completed` plane-filled pixels,
accepted `planes`, conflicting overlap pixels, final triangles, and total surface
time. The demo also reports wall-clock render FPS every 300 rendered frames.
Existing `FRAME_OCCLUSION_RECORD` records the completed pre-history depth;
the recording file format is unchanged. The sequence replay remains a temporal
coverage check, not a comparison of the old/new stereo matcher or ground truth.

## Reproducible comparison on Frame

```bash
make -j4 occlusion bin/frame_openxr_point_cloud bin/test_occlusion_completion \
  bin/test_vulkan_stereo bin/capture_rectified_pair bin/evaluate_occlusion_completion
OMP_NUM_THREADS=8 ./bin/test_occlusion_completion
./bin/test_vulkan_stereo shaders
OMP_NUM_THREADS=8 ./bin/capture_rectified_pair /tmp/room
OMP_NUM_THREADS=8 ./bin/evaluate_occlusion_completion \
  /tmp/room-left.raw /tmp/room-right.raw shaders /tmp/room-new
```

The capture utility discovers compositor descriptors and rectifies through the
production IMX616/UBWC/KB4 path. It exports processed luma, never raw DMA slots.
As with live capture, left/right sensor exposure synchronization is not proven.
The evaluator runs six passes, excludes the first from timing, reports coverage
after rejection/smoothing/completion, and writes measured, old-surface, and
completed float32 files. To isolate the matcher change, run the same pair with
a directory containing the previous four stereo SPIR-V shaders and another
output prefix. Preserve those shaders before rebuilding.

Optional host visualization (numpy and matplotlib required):

```bash
python3 scripts/plot_occlusion_completion.py room-left.raw room-baseline room-new comparison.png
```

## Initial measurements (2026-10-07)

On a freshly captured room pair (chairs, carpet, windows), old and new shaders
ran on identical 616×616 inputs through Adreno 750/Turnip. The previously running
demo remained active during the offline measurement; these timings include GPU
contention and are not standalone performance guarantees.

| Metric | Previous matcher + hole fill | Corrected matcher + plane completion + hole fill |
| --- | ---: | ---: |
| Measured stereo pixels | 46,573 (12.27%) | 54,805 (14.44%) |
| Surface pixels before history | 46,330 (12.21%) | 142,163 (37.47%) |
| Triangles before history | 10,506 | 61,722 |
| Supported camera-view mesh coverage | 5.54% | 32.53% |
| Warm stereo pass | 16.20 ms | 18.05 ms |

Plane completion alone added 24.03 ms averaged over five warm passes on this
input with eight OpenMP threads. A stored pair improved from 9,304 to 76,204
surface pixels and from 1,654 to 33,742 triangles; its completion pass took
18.12 ms. These are coverage observations without ground-truth depth. Mask
precision, dynamic-object response, and display-eye alignment need headset
evaluation. The diagnostic image is in `artifacts/occlusion_completion/comparison.png`.

Live demo surface construction ran at approximately 31–58 ms after warmup,
including completion, history, and mesh building. An A/B run using the same
instrumented demo with the original layer/shaders versus the enhanced layer
reported approximately 36 FPS in both cases. This is the observed OpenXR demo
rate, not a claim of 70–90 FPS or a benchmark across other applications. Completion
does slow depth-worker updates; temporal confirmation and moving-object latency
remain limitations. The enhanced demo is installed and running from the normal
Frame project. The original binaries and sources are backed up at
`/tmp/frame-occlusion-pre-completion-20261007.tar.gz` on Frame.

Validation includes blank/repeating-pattern rejection and a textured fractional
disparity plane on the real GPU (99.94% valid interior, 0.159-pixel mean disparity
error, no accepted errors above one pixel); sparse/slanted completion, guide
edges, same-color depth steps, silhouettes, empty/collinear support, bounded
large holes, malformed input, and deterministic overlap tests; and the existing
surface, GPU mask, and API-layer contract tests.

For a guided still/head-turn/hand-motion recording and identical-input temporal
filter comparison, see [the occlusion benchmark](occlusion_benchmark.md).

## Completion optimization (2026-10-07)

Completion now computes measured validity and undirected luma/depth connections
once per frame, then caches global indices and neighbor masks inside each tile.
Overlapping region traversals reuse those connections instead of repeating the
same boundary checks and integer divisions. The bounded distance fields use
bytes (unknown = 255); their propagation limits remain 24 and eight pixels.
Plane fitting, traversal order, support restrictions, and deterministic overlap
reduction are unchanged. The connection cache adds three bytes per input pixel
(about 1.1 MiB at 616×616).

The evaluator can time completion on an existing measured stereo dump and its
matching rectified left luma, without executing Vulkan. Spatial smoothing runs
once before timing; it performs one warmup and 30 timed completion passes, writes
`<prefix>-planes.raw`, and reports floor-index median/p95 statistics:

```bash
OMP_NUM_THREADS=8 ./bin/evaluate_occlusion_completion --completion-only \
  room-new-measured.raw room-left.raw room-completion
```

A standalone CPU comparison on Frame used identical processed inputs extracted
from frames 70 and 210 of the completed `20261007-182413` recording. Both builds
used `g++ -O3 -fopenmp -std=c++17` and eight OpenMP threads, with one warmup and
30 timed passes per input. The original live demo remained running; these are
small observed improvements under contention, not isolated throughput guarantees.

| Recording input | Baseline median / p95 | Optimized median / p95 | Median reduction |
| --- | ---: | ---: | ---: |
| Frame 70 | 24.85 / 36.15 ms | 23.52 / 34.36 ms | 5.3% |
| Frame 210 | 23.70 / 41.21 ms | 23.01 / 34.25 ms | 2.9% |

Completed float32 output files were byte-identical between baseline and optimized
builds on both inputs; measured, filled, plane, and conflict counts also matched.
Frame 70 produced 34,519 anchors, 65,302 filled pixels, 707 planes, and 429 overlap
conflicts; frame 210 produced 30,906 anchors, 59,361 filled pixels, 646 planes, and
201 conflicts. One-thread and eight-thread outputs for frame 70 were also
byte-identical. The existing completion tests pass, including added exact
16-versus-17-level luma-boundary checks. The new evaluator mode independently
returned matching counts and output for frame 70. This change recovers only a
small part of worker time; it does not alter temporal confirmation latency.

## Completion and hole-fill contract correction (2026-10-07)

The former production call passed plane-completed depth to the legacy hole
helper. Its search could therefore use inferred brackets and refill intentional
plane conflicts. The current worker passes completed output and the original
smoothed measured bracket source separately, along with `OcclusionCompletionContract`.
The evaluator uses the same corrected call. The legacy helper remains available
for baseline comparisons and callers whose input contains measurements only.

`contract.veto` is a per-pixel bit mask: `CompletionOverlapConflict` (1),
`CompletionContradictorySupport` (2), and `CompletionProtectedBoundary` (4).
Contradictory support protects unknown pixels within the accepted plane hull and
eight connected pixels of a disagreeing anchor. Unknown endpoints of blocked
luma/depth connections are protected boundaries; hole searches additionally
check each crossed guide edge. Both plane distance fields now traverse the
actual cached connections, including internal edges in a connected component.
The deterministic reduction combines vetoes before accepting overlap votes.

`contract.provenance` distinguishes unknown (0), smoothed measured (1), plane
prediction (2), and enclosed-hole inference (3). It describes the spatial source,
not an independent confidence probability. All inferred output is supported by
this frame's measured anchors; inferred output never seeds another completion
or hole pass. Temporal evidence retains its own measured/inferred rules.

Validation on Frame used an isolated `/tmp/frame-completion-contract` build with
`g++ -O3 -fopenmp -std=c++17`. The completion test compares one and eight threads
and checks depths, veto masks, and provenance. Added cases cover inferred-only
brackets, explicit vetoes at target and bracket paths, the exact luma threshold,
depth barriers, contradictory anchors, and competing local planes on a curved
surface. In that deterministic curved case, the legacy pass reopened 130 of 209
overlap-conflict pixels; the corrected pass reopens zero. Existing measured and
completed depths remain unchanged by the hole pass. These are synthetic contract
tests; they do not establish scene coverage, false masking, or wearer acceptance.

Removing unsupported inferred chaining can reduce coverage. Existing historical
measurements above describe the former pipeline. Compare corrected, former, and
completion-off paths on identical recordings before assessing the quality tradeoff;
curved scenes and same-color moving foreground still require controlled captures.
No completion reach, matching tolerance, or temporal policy default was relaxed.
