# Stereo matcher comparison on Frame

The first matcher investigation is implemented and evaluated offline. The live
restored runtime remains unchanged. The comparison uses nine identical recorded
rectified IMX616 luma pairs (frames 20, 40, 70, 100, 140, 180, 210, 240, 270)
from the complete 294-frame recording `20261007-183447`, two independent synthetic
texture seeds, and separate correctness tests on the Frame GPU and CPU.

## Findings

Four-path SGM with independent matching evidence gives a small, repeatable gain
in recorded depth support. The tested edge-aware local matcher loses support and
is not a promising replacement in this form. No variant establishes accurate
real-room depth, solves the wearer's streaks, or meets the release criteria.

| Method | Mean supported image area, nine frames | After unchanged speckle removal | Median of per-frame CPU p50 |
| --- | ---: | ---: | ---: |
| Production GPU reference | 11.900% | 9.697% recorded measured depth | See GPU timing below |
| CPU baseline reference | 11.900% | 9.697% | 196.5 ms |
| Edge-aware local matcher | 9.297% | 6.954% | 214.2 ms |
| Guarded SGM, P1=8/P2=48 | 12.351% | 10.483% | 675.0 ms |

The SGM gain after cleanup is **0.786 percentage points**, about **8.1% relative**
to baseline support. These are depth pixels in camera-image space, before
completion, history and meshing. They are neither eye-mask coverage nor counts
of correct occlusions. The recording has no independent depth ground truth.
A denser wrong depth map can be worse than missing depth.

The unchanged production GPU matcher measured 17.5–21.1 ms per-frame process-time
medians on these inputs (one warmup, five measured calls). CPU figures are
unoptimized algorithm references with different execution/memory structure;
they do not predict SGM GPU cost. The runtime was co-running, and some control
tests overlapped the reference timing run. Timing is not an isolated hardware
benchmark or display latency. This CPU SGM must not be inserted into the worker.

Synthetic weak-texture pairs improve with guarded SGM:

| Texture seed | Method | Supported image area | Accepted bad depth |
| --- | --- | ---: | ---: |
| 1709 | Baseline | 37.50% | 15.67% |
| 1709 | Guarded SGM, 8/48 | 45.24% | 13.00% |
| 991 | Baseline | 39.51% | 14.89% |
| 991 | Guarded SGM, 8/48 | 46.82% | 11.79% |

Support uses the whole image denominator; error is on accepted pixels in the
visible ground-truth region. Bad depth means error greater than
`max(0.04 m, 5% true depth)`. Missing pixels remain in the reported ground-truth
missing fraction. These still-substantial error rates prevent treating increased
coverage as reliable occlusion. The synthetic scenes do not reproduce headset
noise, camera synchronization, lens distortion or calibration errors.

There is also a tradeoff on clean textured fractional planes: accepted disparity
MAE rises from about 0.115 to 0.129 pixels for seed 1709, while the rare large-error
rate falls. On its silhouette case, background-boundary leakage falls from
0.138% to 0.046%, but foreground bad depth increases slightly from 0.071% to
0.107% of all foreground pixels. Both missing fractions are exported separately.
The candidate therefore does not satisfy every accuracy acceptance criterion.

Bounded penalty comparisons 4/24 and 16/96 were also run on frames 70/210 and
both synthetic seeds, keeping all confidence guards unchanged. Stronger penalties
reduce synthetic weak-texture errors further but worsen clean-plane subpixel
MAE. Their room support is not uniformly better. This does not select a new
default or justify choosing parameters by coverage alone.

## Algorithms and validation

`include/stereo_matcher_experiment.hpp` is an offline-only header and is not
included by either production runtime. It implements:

- Baseline: the active GPU's clamped 7x7 census, five-position cross cost,
  ranked uniqueness excluding the winning disparity's adjacent hypotheses,
  parabola refinement, bidirectional consistency, metric range and neighbor
  support. Right-reference search direction and border policy are preserved.
- Edge-aware: only the cross cost changes. Offset samples crossing a reference
  luma difference greater than 16 are dropped, with the remaining cost normalized
  back to the same five-sample scale. This leaves the census window itself
  unchanged and is a limited edge-aware experiment.
- SGM: the same cross cost is accumulated by the actual dynamic-programming
  recurrence along left/right/up/down paths, with gradient-adaptive P2. P1/P2
  use the baseline 0–240 Hamming-sum scale; selection uniqueness uses the average
  of four paths. It keeps one unary volume and one accumulated volume at a time,
  about 109 MB at 616x616x96, in addition to census/output buffers.

The recurrence follows [Hirschmueller's SGM work](https://www.robotic.dlr.de/fileadmin/robotic/hirschmu/cvpr05hh.pdf).
This is a four-path census experiment, not the paper's mutual-information unary
or its larger path set. The separate old `stereo_sgm.comp` remains inactive.

An initial regularizer-only uniqueness check fabricated confident matches on
periodic and unrelated images. SGM now also requires the selected **raw unary**
cost to beat every nonadjacent competitor by the unchanged three-cost margin;
flat unary curves are rejected. Smoothness alone cannot establish a match.
This guard limits coverage gains: accepted winners cannot jump to a different
raw matching basin. Improvements come from local regularization, refinement,
LR consistency and spatial support rather than filling wholly ambiguous areas.

Final controlled tests pass for blank/periodic rejection, known fractional
disparity, range cutoff, exposure offsets, rendered foreground/background steps
and true occlusion strips. Four unrelated-image seeds still show false acceptance:
baseline 4.82–6.83%, edge-aware 1.58–2.90%, guarded SGM 3.90–6.28%.
These tests demonstrate relative behavior, not safe absolute false-mask rates.
CPU results are identical with one and eight OpenMP threads.

GPU baseline checks pass on all nine recorded pairs and the synthetic controls.
Left/right disparity masks, pre-support depth masks and LR flags match exactly;
float differences stay below 2e-5. A few final support decisions differ at the
4% neighbor-agreement threshold because of CPU/GPU floating arithmetic. Each
exception is checked individually: observed input roundoff must explain the
neighbor compatibility changes and the crossing of the three-neighbor count.
There are no unexplained differences. Frame 70 has three such pixels and frame
210 has two. The retained preliminary parity failure preceded the explicit
Turnip ties-even rounding and boundary checks; it is not the final validation.

## Reproduce

Build and run the optional tools on Frame in an isolated checkout or the prepared
`/tmp/frame-stereo-matcher-experiment` directory. Do not sync the old rejected
completion/expiry candidate to the running project when testing these tools.

```sh
make -j4 bin/export_stereo_matcher_inputs bin/compare_stereo_matchers \
  bin/test_stereo_matcher_experiment_gpu
OMP_NUM_THREADS=8 bin/export_stereo_matcher_inputs \
  /home/steamos/frame-spatial-runtime/captures/occlusion/20261007-183447/recording.bin \
  shaders results/inputs 20,40,70,100,140,180,210,240,270
OMP_NUM_THREADS=8 bin/test_stereo_matcher_experiment_gpu
OMP_NUM_THREADS=8 bin/test_stereo_matcher_experiment_gpu --gpu shaders \
  --left results/inputs/frame-70-left.pgm --right results/inputs/frame-70-right.pgm
OMP_NUM_THREADS=8 bin/compare_stereo_matchers \
  --left results/inputs/frame-70-left.pgm --right results/inputs/frame-70-right.pgm \
  --baseline-depth results/inputs/frame-70-gpu-supported-f32.raw \
  --baseline-depth-stage supported --output results/final/default/frame-70 \
  --runs 3 --warmup 1
OMP_NUM_THREADS=8 bin/compare_stereo_matchers --synthetic --synthetic-seed 1709 \
  --output results/final/default/synthetic-1709 --runs 3 --warmup 1
```

Repeat recorded comparisons for the nine IDs and synthetic seed 991. Penalty
variants use `--methods sgm --p1 4 --p2 24` or `--p1 16 --p2 96` and separate
output directories. The exporter validates the complete recording before saving
inputs, requires diagnostic/production parity and reproduces recorded measured
depth exactly after unchanged speckle removal.

The [saved results](../artifacts/stereo_matcher_comparison/20261007/results.md),
[combined metrics](../artifacts/stereo_matcher_comparison/20261007/combined-metrics.csv)
and per-frame masks/raw stages preserve the measurements. Regenerate the summary
with `python3 scripts/summarize_stereo_matchers.py artifacts/stereo_matcher_comparison/20261007`.
Source/build/input hashes are recorded in that directory's manifest.

## Next implementation gate

The [follow-up investigation](stereo_matcher_followup.md) analyzes saved stage
losses, tests raw-cost subpixel refinement on fixed accepted masks, and reviews
better matching methods. Raw refinement restores clean-plane average precision
but worsens bad-depth rates, especially on weak texture; simple blending and
fit-agreement gating do not remove the tradeoff. The next quality experiment
should improve unary evidence/adaptive support and independently evaluate
continuous refinement. GPU regularization remains a performance candidate,
with its quality ceiling and independent matching-evidence guards explicit.
Compare any complete candidate against this reference before measuring Frame timing.
Release requires controlled real surfaces at known optical depths, checked
silhouettes and moving objects, acceptable processing age, and a single-change
headset trial. Those real-scene measurements remain outstanding; this experiment
has not solved the original occlusion report.
