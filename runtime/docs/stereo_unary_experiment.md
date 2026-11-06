# Noise-aware unary trial

Experiment date: 2026-10-07. This implements the first offline step in
[the matcher follow-up](stereo_matcher_followup.md): change matching evidence
while retaining the existing five-position aggregation and downstream stages.

**The tested masked-census candidate is not a production replacement.** It
improves fractional precision on textured noisy inputs, but rejects all held-out
weak-texture depth, reduces recorded support, increases unrelated-pair false
support, and slightly worsens silhouette error tails. The live matcher, camera
calibration, completion, history, mesh policy and headset processes were unchanged.

## Candidate and cost scale

The `noise` method retains the 7×7 census ordering and five-position cross.
A comparison participates only when `abs(neighbor-center) > tau` in **both**
corresponding descriptors. Let `N` be the joint participating comparison count
and `M` its ordering disagreements: cost is `round(240*M/N)`. Hypotheses with
`N < 120` receive cost 240 and cannot be accepted. Suppressing every comparison
therefore cannot create a confident zero-cost match.

Integer selection, parabola refinement, bidirectional matching, LR/range
filtering, neighbor support and speckle cleanup are unchanged. Invalid neighbor
costs can nevertheless affect the existing cost parabola. This is a masked
binary-census ablation, not published Quantized Census or AD-Census. Comparison
counts include overlapping windows and are not independent bits or entropy.

The 0–240 range is explicit, but normalization does **not** preserve the original
confidence distribution. Three cost units represent 1.25 percentage points of
normalized disagreement; cutoff 120 represents 50%. These are exploratory
operating points, not calibrated confidence probabilities.

Development used seeds 1709/991 with `tau=0,1,2,4`. Zero excludes tied comparisons;
`tau=-1` retains all bits and is the exact baseline parity control. A separate
`tau=2` grid used cutoffs 60/90/120 and margins 0/3/6/12 on development seed 1709,
rerunning both directions and every acceptance stage for baseline and candidate.
Higher margins reduce false support, but the tested grid does not remedy the
weak-texture information loss or establish reliable confidence.

The primary candidate (`tau=2`, information floor 120, cutoff 120, margin 3) was
fixed before held-out evaluation. It was evaluated despite developmental
failures to characterize the tradeoff; no setting was selected for deployment.
The information floor was held fixed. These results do not rule out every
masked/quantized descriptor or a separately calibrated information guard.

## Results on Frame

All nine recorded rectified IMX616 pairs were replayed in
`/tmp/frame-unary-experiment`, outside the running project, with eight OpenMP
threads, one warmup and three measured calls per method/frame.

| Stage | Baseline | Masked census, tau=2 |
| --- | ---: | ---: |
| Mean supported image area | 11.900% | 9.876% |
| After unchanged speckle cleanup | 9.697% | 8.105% |
| Median of per-frame CPU p50 | 189.3 ms | 239.9 ms |

Support after cleanup falls by 1.592 percentage points, about 16.4% relative.
Recorded coverage has no independent depth GT and cannot establish accuracy.
CPU timings include allocations and all matching stages, excluding speckle
cleanup. The live runtime co-ran; these timings do not predict Vulkan performance
or publication latency.

Held-out seeds were 4201, 7877, 12011 and 32003, with seventeen scenes per seed:
integer/fractional planes, weak texture, broad/thin silhouettes, flat/periodic
ambiguity, independent noise in both views, gain/bias, clipping, half/one-pixel
vertical misregistration, and unrelated pairing. Bounded uniform noise is a
stress control, not a measured IMX616 noise model. Sloped planes, raw exposure
pairing and real moving objects remain untested.

These errors pool accepted visible GT pixels across the four seeds. Bad depth
means `error > max(0.04 m, 5% true depth)`. Stages stop after neighbor support.

| Held-out scene | Baseline MAE | Candidate MAE | Baseline accepted bad | Candidate accepted bad |
| --- | ---: | ---: | ---: | ---: |
| Clean fractional plane | 0.1142 px | 0.1006 px | 0.0567% | 0.0490% |
| Independent noise ±3 | 0.1223 px | 0.1118 px | 0.3202% | 0.1857% |
| Independent noise ±6 | 0.1446 px | 0.1372 px | 2.1049% | 1.6314% |
| Gain/bias with ±3 noise | 0.1208 px | 0.1114 px | 0.2668% | 0.1711% |
| Clipped gain with ±3 noise | 0.2670 px | 0.1299 px | 5.5749% | 1.9686% |
| Broad silhouette | 0.1242 px | 0.1102 px | 0.1792% | 0.2012% |
| Thin silhouette | 0.1232 px | 0.1098 px | 0.1856% | 0.1917% |
| Integer plane | 0.0601 px | 0.0794 px | 0% | 0.0015% |

The ±3-noise gain survives identical common GT masks: MAE 0.1219→0.1109 px and
bad depth 0.2823→0.1573%. Conversely, common-mask silhouette tails worsen:
broad-step bad depth 0.1466→0.1593%, thin-step bad depth 0.1429→0.1650%.

Critical failures and missing-area checks:

- Weak-texture image support falls from 37.112% to **0%**. Visible GT missing
  increases from 57.366% to 100%. With no accepted candidate pixels, accuracy
  is undefined rather than perfect.
- Unrelated-pair false support rises from 11.799% to **13.367%** of the image.
  Every supported pixel is false correspondence. These smooth analytic textures
  differ from the older unrelated-image correctness fixtures.
- Broad-step background leakage rises from 0.265% to 0.369% of evaluated boundary
  pixels; missing boundary increases from 12.085% to 12.661%. Thin-step leakage
  decreases (1.185→1.078%), but thin-foreground bad depth increases
  (0.994→1.101% of all foreground GT) and missing increases (31.428→31.889%).
- At one-pixel vertical misregistration, bad depth per all GT rises from 8.484%
  to 9.415%, despite slightly lower accepted bad-depth percentage.

Accepted textured ±3-noise hypotheses retain about 207 of 240 comparisons on
average. Weak texture has no accepted left hypotheses at `tau=2`. Suppressing
uncertain comparisons helps textured evidence but does not solve low-texture
support with this guard. Even zero-deadband development retains substantial
weak-texture errors; suppressing ties alone is not a solution.

## Validation and saved evidence

Local and Frame correctness tests passed. Frame OMP=1/8 logs match, and all 22
saved candidate stage files for frame 70 are byte-identical across thread counts.
The all-bits candidate reproduces baseline disparities, depth, costs, gaps and
rejection exactly. All-bits-suppressed, flat and periodic support controls pass.
Saved accepted candidate hypotheses meet the information floor.

Twenty baseline stage files for frame 70 are byte-identical to the original
investigation; original developmental synthetic baseline metrics reproduce
within 2e-6. The live occlusion library hash remains unchanged. Production
shaders were unchanged; this experiment uses the previously validated CPU
baseline with archived reproduction checks, rather than claiming new GPU parity.

[Full results](../artifacts/stereo_unary_experiment/20261007/results.md),
[all metrics](../artifacts/stereo_unary_experiment/20261007/combined-metrics.csv),
[common-mask comparisons](../artifacts/stereo_unary_experiment/20261007/combined-comparisons.csv),
[confidence operating points](../artifacts/stereo_unary_experiment/20261007/confidence-operating-points.csv),
and [summary/checks](../artifacts/stereo_unary_experiment/20261007/summary.json)
retain the measurements. Commands, compiler/build/input hashes, tested source
snapshots and validation logs are in the same directory. Full stages are saved
for frame 70 and held-out seed 4201; other cases save metrics. Subsequent header
edits only clarify comments.

## Reproduce and next step

The tools remain opt-in and are not linked into either live runtime. In an
isolated Frame checkout, the Makefile retains `-O3 -fopenmp`:

```sh
make -j4 bin/compare_stereo_matchers bin/test_stereo_matcher_experiment
OMP_NUM_THREADS=8 bin/test_stereo_matcher_experiment
python3 scripts/run_stereo_unary_experiment.py \
  --binary bin/compare_stereo_matchers --inputs PATH_TO_ORIGINAL_INPUTS \
  --output results/unary --phase development
python3 scripts/run_stereo_unary_experiment.py \
  --binary bin/compare_stereo_matchers --inputs PATH_TO_ORIGINAL_INPUTS \
  --output results/unary --phase evaluation
python3 scripts/analyze_stereo_unary_experiment.py results/unary \
  --archive PATH_TO_ORIGINAL_MATCHER_ARTIFACTS
```

The runner refuses case overwrites. Saved command lists reproduce the actual
isolated execution. Implementation lives in `include/stereo_matcher_experiment.hpp`
and `tools/compare_stereo_matchers.cpp`.

Do not port this candidate. The next evidence comparison should add a bounded
luma/gradient residual and evaluate raw and exposure-normalized variants
separately. Hold aggregation fixed, then test bounded adaptive support from
both correspondence neighborhoods independently. Real known-depth surfaces and
motion controls remain necessary before claiming reliable headset occlusion.
