# Composite unary implementation and development results

Experiment date: 2026-10-07. The [composite-unary protocol](stereo_composite_unary_plan.md)
is implemented in the opt-in offline matcher, comparison CLI, runner and analyzer.
**No tested composite configuration passed the quality gates.** There is no
finalist for clipping sensitivity, GPU porting or a headset quality trial.

The complete development sweep ran on the local Mac: 767 verified jobs covering
158 configurations (twelve baseline operating points, two zero-weight controls,
and twelve composite costs at twelve confidence points). Each configuration
used all seventeen synthetic scenes at seeds 1709, 991, 56039 and 65537. All
fifteen initial costs at cutoff 120/margin 3 also replayed the nine recorded
IMX616 pairs. The four reserved evaluation seeds were not generated or opened.
These are development findings, not held-out accuracy claims.

## Measured tradeoffs

The following rows use the existing parabola arm, `alpha=0.25`, `beta=0.5`,
`Tl=16`, `Tg=8`, cutoff 120 and margin 3. Recorded values are means over all
nine images; synthetic values pool the four development seeds. Recorded area
uses the entire image and has no independent room-depth GT. Unrelated-pair
support is entirely false correspondence. Clipping error uses each method's
own accepted visible-GT mask before speckle cleanup.

| Method | Recorded supported area | After cleanup | Unrelated false support | Clipped-gain accepted bad depth |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 11.900% | 9.697% | 12.152% | 5.783% |
| Composite raw | 15.303% | 12.828% | 14.426% | 5.951% |
| Composite normalized | 14.872% | 12.524% | 14.082% | 6.746% |

On identical common GT masks, the normalized candidate improves ±3-noise
disparity MAE from 0.12113 to 0.11894 px and ±6-noise MAE from 0.14167 to
0.13859 px. Those gains coexist with increased false support, clipped-exposure
tails, and boundary leakage. Its broad-silhouette background leakage rises
from 0.127% to 0.219% of evaluated boundary pixels; thin-silhouette leakage
rises from 0.718% to 0.880%. Raw and normalized variants also increase bad
depth per all weak-texture GT despite slightly lower accepted bad-depth rates
at this operating point. Additional accepted area is not necessarily correct.

The selector checks integer-only and existing-parabola arms independently,
before and after unchanged speckle cleanup. None of the 144 nonzero-weight
composite operating points passed either complete arm. Only the default
baseline operating point passed the baseline non-regression checks; baseline
controls are not improvements or finalists. `development-selection.json`
records `failed-quality-gates` and an empty finalist list. Confidence thresholds
were not relaxed after inspecting the results.

## Implementation and validation

The offline header adds raw/normalized photometric channels, bounded component
costs, selected-winner/competitor diagnostics, and stored parabola fits. Both
refinement arms share integer evidence and rerun LR/range, neighbor support
and cleanup. Reused selections have unavailable full matcher timing, rather
than counting partial work as a complete call.

The runner separates development and evaluation, hashes sources and inputs,
saves exact commands and completion manifests, and refuses partial-job
overwrites. The analyzer requires a complete verified sweep, uses GT/region
denominators when pooling, rejects empty-mask precision claims, checks cleanup
as well as support, and keeps fixed-mask fitting diagnostics separate from
candidate coverage. Evaluation requires unchanged frozen configurations and
development evidence; this failed sweep did not produce an evaluation manifest.

Local and isolated Frame builds passed the controlled C++ tests and seven
Python workflow tests. Frame used the Makefile's `-O3 -fopenmp`; OMP=1/8 test
logs match. Tests cover per-hypothesis and full-stage zero-weight parity,
signed gradients, normalization under gain/bias and constant inputs, cost
bounds, deterministic diagnostics, and equality between reused acceptance and
independently matched refinement arms. Frame's saved zero-weight outputs match
across 68 stage files; all twenty corresponding original frame-70 baseline
stages are byte-identical. The local sweep additionally validates 96 saved
zero-weight stage comparisons and sixty archived baseline comparisons across
three saved frame-70 controls.

After the sweep, the component-export loop was changed to an explicitly typed
array to remove GCC's dangling-reference warning about its temporary pair
initializer list. This does not change costs, selection, fitting or stages;
the final build and export parity were checked separately. The sweep retains
its original tested source snapshots and hashes. No production matcher,
camera calibration, completion, history, mesh policy or headset process was changed.

## Saved evidence and next step

[Development results](../artifacts/stereo_composite_implementation/20261007/development-trial/development/results.md),
[full per-scene gates](../artifacts/stereo_composite_implementation/20261007/development-trial/development/summary.json),
[combined metrics](../artifacts/stereo_composite_implementation/20261007/development-trial/development/combined-metrics.csv),
[common-mask comparisons](../artifacts/stereo_composite_implementation/20261007/development-trial/development/combined-comparison.csv),
[fixed-mask fitting diagnostics](../artifacts/stereo_composite_implementation/20261007/development-trial/development/combined-fixed-mask.csv),
and [confidence points](../artifacts/stereo_composite_implementation/20261007/development-trial/development/confidence-curves.svg)
retain the local sweep. [Native Frame parity checks](../artifacts/stereo_composite_implementation/20261007/frame/frame70-checks.json)
and [Frame smoke measurements](../artifacts/stereo_composite_implementation/20261007/frame/smoke/summary.json)
provide separate device validation; local CPU timings do not establish Frame or
Vulkan performance. The native smoke trial is one development seed, not a
second full sweep.
[Final export parity](../artifacts/stereo_composite_implementation/20261007/frame/release-parity.json)
checks 274 saved stage and diagnostic files after the compiler-warning cleanup.

Keep the production baseline. Any revised confidence or exposure-handling
proposal needs a separately documented development experiment; the current
grid offers no passing unary to carry into the protocol's next stages. Retain
the fresh evaluation split for a candidate fixed before those outcomes are
opened. Real known-depth surfaces and motion/pairing controls remain necessary
before claiming reliable headset occlusion.

**Follow-up completed:** the [bounded adaptive-support screen](stereo_adaptive_support_experiment.md)
kept census fixed and tested dual-view support. It improved noise/weak-texture
tradeoffs, but no configuration passed the silhouette/exposure non-regression
gates. Production remains on baseline; evaluation seeds remain unused.
