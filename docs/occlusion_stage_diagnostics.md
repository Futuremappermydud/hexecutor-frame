# Occlusion stage diagnostics

The diagnostic path reruns the active census/cross-aggregation matcher on
rectified IMX616 luma. It changes no matcher thresholds. Production initializes
`VulkanStereoPipeline` with diagnostics disabled and allocates no diagnostic
images or additional staging space. Diagnostic initialization selects separate
shader variants; passing a `StereoDiagnostics*` opts into seven intermediate
GPU image copies on that call. Diagnostic timing includes those copies and must
not be reported as production stereo timing.

Build on Frame without restarting the demo:

```sh
make -j4 bin/test_vulkan_stereo bin/evaluate_occlusion_completion bin/replay_occlusion_stages
OMP_NUM_THREADS=8 bin/test_vulkan_stereo
OMP_NUM_THREADS=8 bin/replay_occlusion_stages \
  captures/occlusion/20261007-183447/recording.bin shaders \
  /tmp/occlusion-stages 70,210
python3 scripts/hash_occlusion_stage_replay.py /tmp/occlusion-stages
```

The hash helper records SHA-256 of the recording, executable, active and diagnostic
shader binaries/sources, calibration, replay and contract sources, live replay
settings, and any supplied region annotations. Run it immediately after replay
before rebuilding those inputs.

The replay first validates the complete v2 recording and rejects incomplete
footers, drops, frame gaps, unordered timestamps, malformed pixel data, and
missing requested frames. It then reruns every frame, saving detailed images
only for the requested frame IDs. This preserves temporal context without
exporting gigabytes for every frame. `stages.csv` reports counts for the whole
616×616 image, not marked-surface coverage. The temporal variant uses explicit
candidate settings `2,2,3,200,250,250`; recorded capture poses and transforms
supply alignment. Pose fallback is reported, resets history, and suppresses confirmed geometry. Luma replay
cannot verify the correctness of exposure pairing, rectification, or scene
calibration upstream of the recorded image.

For already extracted raw rectified pairs:

```sh
bin/evaluate_occlusion_completion left-u8.raw right-u8.raw shaders /tmp/pair
```

This exports spatial stages and mesh coverage of pre-history depth. It has no
sequence context and therefore exports no confirmed mask. Its stereo timing
uses warmed calls without intermediate readback.

## Map definitions

Files use row-major pixels at 616×616. `*.pgm` are binary P5 images: stage masks
use 255 for supported pixels and rejection masks use 255 for failures. Luma PGM
files contain the original pixel values. `*-f32.raw` are little-endian IEEE
float32 on Frame; disparity is in rectified pixels, depth is optical-axis meters.
`*-u32.raw` and `*-u8.raw` contain their stated unsigned integer types.

- Left/right disparity contains the shader's accepted subpixel disparity;
  rejected matches have zero disparity.
- Left/right match RGBA float arrays contain integer winning cost, uniqueness
  margin, integer winning disparity, and a rejection bitmask. Cost is the sum
  of five 48-bit census Hamming distances (0–240), not a probability. Margin is
  the lowest competitor cost outside winning disparity ±1 minus the best cost.
  Margin -1 means no such competitor was found; cost/margin -1 at borders or
  insufficient search means no match was evaluated. Rejection bits are border=1,
  insufficient search=2, winning disparity below 2=4, cost ≥120=8, margin ≤3=16.
  Multiple match failures may be set at once.
- Filter rejection bits are left/right inconsistency=1 and metric range=2, accepted match with subpixel disparity ≤1.5=4.
  Range is tested only after consistency succeeds. Upstream rejected left
  matches are not counted again as filter failures. A missing accepted right
  correspondence is an LR failure. The accepted range remains 0.25–5 m.
- Consistency depth precedes neighbor support. Support rejection is 1 only
  when a positive consistency depth is removed by boundary/support checks;
  upstream missing samples remain zero. Neighbor support requires three of
  eight samples within max(0.04 m, 4% of depth).
- Spatial stages are supported stereo, speckle-cleaned, smoothed, planes,
  pre-history, and confirmed. Legacy pre-history permits inferred hole brackets;
  corrected pre-history uses smoothed measured brackets and completion vetoes;
  completion-off fills from smoothed measured brackets with the luma guide.
  These spatial variants do not compare temporal policies.
- Completion veto/provenance arrays follow `OcclusionCompletionContract`.
- History events are opt-in uint16 bitmaps. `history-events.csv` reports the
  event frame's capture-end time, interval, event counts, the former whole-mesh
  minimum lifetime/deadline, current latest cell lifetime/deadline, and geometric
  and surviving mesh areas at display ages 0, 75, and 100 ms. Pending=1, confirmed=2, measured
  clear=4, prediction expiry=8, source expiry=16, reprojection loss=32,
  collision=64, measured agreement=128, inferred agreement=256, evidence
  change=512, held=1024. Reprojection loss uses the previous image's source
  pixel coordinate; collision uses the current destination. Flags can coexist;
  counts are events, not a disjoint partition of all support changes. These
  timestamps describe software observation, not independently timed display.
- Mesh PGM rasterizes the returned camera-image triangles at pixel centers;
  it includes the coarse-cell subdivision rule. Mesh indices are uint32.
  `*-support-mesh.pgm` shows geometric support; `*-mesh.pgm` includes cells
  with positive minimum covered-sample lifetime. `*-display-age75ms-mesh.pgm`
  and `*-display-age100ms-mesh.pgm` simulate expiry at those capture-end ages
  without changing geometry/viewpoint. Cell lifetimes correspond one-to-one
  with each six-index group of the support mesh. Positive snapshot lifetime
  is no guarantee a cell remains visible at display time; the renderer evaluates
  its deadline then. These are camera-image support proxies, not display-eye
  occlusion masks.
- Recorded measured/input maps are exported separately so upstream replay
  differences can be checked against the recording rather than assumed away.
  `replay_measured_mismatch` counts exact float differences after speckle cleanup.
  Recorded pre-history counts use the original old completion implementation;
  legacy-brackets maps use the current plane implementation with the old hole
  pass, so they isolate the hole contract rather than recreate the old runtime.

## Independently measured marked regions

An optional last argument is a CSV of rectangular targets:

```text
# name,x0,y0,x1,y1,optical_z_m
wall_3m,120,90,220,190,3.0
```

Coordinates are rectified-image pixels; x1/y1 are exclusive. The example is a
format illustration, not an annotation or measurement of the saved recording.
Use a rectangle wholly inside a surface with independently known constant
optical Z. Slant tape distance is not optical Z. Avoid silhouettes and sloped
surfaces; a constant-Z rectangle does not model their ground truth.

`regions.csv` reports all marked pixels as the denominator, accepted count,
missing fraction, and accepted-only depth MAE/bias separately for each stage
and selected frame. Error fields remain blank when no pixels are accepted.
`mesh-regions.csv` separately reports marked pixel area supported by geometric
mesh cells, per-cell snapshot eligibility, and simulated display ages 75/100 ms.
It uses the same
marked region denominator and does not turn absent geometry into omitted error
samples.
Regions without actual measurements must not be presented as ground truth.
Polygon targets, dense varying-depth ground truth, actual display-eye mask
errors, and independently timed hand response remain future measurements.

Frame GPU smoke tests verify blank/repetitive texture rejection, fractional
disparities 3.35, 5.2, 8.1, and 18.35 pixels, metric-range rejection, stage-local
rejection consistency, and exact production/diagnostic output parity. These
synthetic checks cannot establish wearer-visible streak improvement or distant
wall accuracy.

For the 294-frame candidate recording, replayed stereo followed by speckle
cleanup exactly matched every recorded measured float. Corrected pre-history
coverage was 25.71% still, 24.87% turning, 19.92% hand motion, and 20.18% recovery,
versus recorded old-input 31.69%, 31.31%, 24.82%, and 23.95%. Those losses quantify
the conservative completion contract; they do not measure depth accuracy.

The same replay's earliest supporting-sample lifetime was only 1.66/30.15/48.53
ms at p05/p50/p95 from capture end. A whole-mesh deadline can therefore discard
fresh geometry alongside its oldest sample before a normal publication/display
cycle completes. The implementation therefore uses per-cell deadlines;
replay now reports remaining per-cell mesh area at 75/100 ms as well as the
former whole-mesh minimum. Those ages are explicit simulations, not measured
current display latency. Snapshot maps do not establish actual visible coverage;
publication-time and display-time behavior need separate measurements.

Per-cell replay mesh area (% of the whole rectified image) after both completion
contract corrections and explicit candidate history is:

| Phase | Geometric / snapshot eligible | Simulated age 75 ms | Simulated age 100 ms |
| --- | ---: | ---: | ---: |
| Still | 15.52% | 14.49% | 13.12% |
| Head turn | 13.56% | 12.52% | 11.35% |
| Hand motion | 12.41% | 11.21% | 10.26% |
| Recovery | 12.87% | 11.52% | 10.86% |

Geometric area and snapshot eligibility matched in this recording. These area
columns use pixels as area, while confirmed-depth counts use supported samples;
they must not be subtracted to count erroneous deletions. The 75/100 ms columns
hold the camera viewpoint and mesh fixed and only expire cell deadlines. They
are neither measured live throughput nor independently timed visible masking.
The full stage manifest, masks, raw maps, counts, events, and summary are in
[`artifacts/occlusion_investigation/20261007/stages/`](../artifacts/occlusion_investigation/20261007/stages/).
