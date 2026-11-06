# Composite census/luma/gradient unary protocol

Protocol date: 2026-10-07. This specifies the next offline experiment from
[the matcher follow-up](stereo_matcher_followup.md). The opt-in offline harness
implements this protocol; a completed quality trial is separate from correctness
validation. The parameter values below are experimental choices, not calibrated
IMX616 noise estimates or production defaults.

The hypothesis is that bounded photometric evidence can improve correspondence
selection without the weak-texture information loss of the
[masked-census trial](stereo_unary_experiment.md). Test raw and locally normalized
photometric channels independently. Keep the original unmasked 7×7 census,
five-position cross, disparity range, LR/range filter, neighbor support and
speckle cleanup. Adaptive support, SGM and continuous refinement are later trials.

## Cost definition

For a reference pixel `p=(x,y)` and integer disparity `d`, use cross offsets
`S={(0,0),(-2,0),(2,0),(0,-2),(0,2)}`. At each offset, `q=p+s` and the target
coordinate is `t=q-(d,0)` for left matching or `t=q+(d,0)` for right matching.
Use the existing interior and available-disparity rules in both directions.

Let `I(q)` be rectified 8-bit luma, promoted before arithmetic. Define signed
central-difference gradients in luma units per pixel:

```text
Gx(q) = (I(q+(1,0)) - I(q-(1,0))) / 2
Gy(q) = (I(q+(0,1)) - I(q-(0,1))) / 2
```

For the **raw** variant, set `J(q)=I(q)` and `K(q)=(Gx(q),Gy(q))`.
For the **normalized** variant, compute population mean `mu(q)` and standard
deviation `sigma(q)` over the complete 7×7 luma window centered at `q`:

```text
scale(q) = 16 / max(sigma(q), 4)
J(q)     = scale(q) * (I(q) - mu(q))
K(q)     = scale(q) * (Gx(q), Gy(q))
```

Compute statistics independently in each view, without GT, disparity-dependent
fitting or information from the other image. The standard-deviation floor caps
amplification at four. Clamp a slightly negative variance from floating-point
roundoff to zero before taking its square root. Do not quantize or clip `J` or
`K`; bound their residuals below. `K` is a scaled raw gradient, not the derivative
of the spatially varying normalized image. Both variants retain the baseline's
maximum five-pixel support radius; the normalization window can still straddle
a silhouette and must be evaluated for leakage.

For each correspondence `(q,t)`, compute:

```text
h(q,t) = popcount(census(q) XOR census(t)) / 48
l(q,t) = min(abs(Jref(q) - Jtarget(t)) / Tl, 1)
g(q,t) = (min(abs(Kxref(q) - Kxtarget(t)) / Tg, 1)
        + min(abs(Kyref(q) - Kytarget(t)) / Tg, 1)) / 2
C(p,d) = floor(48 * sum_s((1-alpha)*h(q,t)
                         + alpha*((1-beta)*l(q,t) + beta*g(q,t))) + 0.5)
```

Sum before rounding; retain a 0–240 integer cost. `alpha=0` must reproduce the
baseline cost exactly for every valid hypothesis. The same range does not imply
the same cost-gap distribution or confidence. Keep census computed from the
original luma for both variants. Save the unweighted census, luma and gradient
components at the selected winner and nearest nonadjacent competitor so cost
separation can be explained without treating overlapping samples as entropy.

## Development and confidence calibration

Use development seeds `1709, 991, 56039, 65537`. The previous evaluation seeds
`4201, 7877, 12011, 32003` are now regression inputs, not held-out evidence for
this proposal. Use all seventeen existing synthetic scenes, including gain/bias,
clipping, independent noise, vertical misregistration, thin silhouettes and
unrelated pairs. Replay all nine archived frames for regression and coverage;
they have no independent room-depth GT and must not select accuracy parameters.

The first grid has `Tl=16`, `Tg=8`, `alpha in {0.25,0.5}` and
`beta in {0,0.5,1}` separately for raw and normalized variants: twelve composite
costs, plus the `alpha=0` parity control and baseline. `beta=0` isolates luma;
`beta=1` isolates gradient. For each cost and baseline, sweep cutoff
`{60,90,120}` and uniqueness margin `{0,3,6,12}`. Retain the existing strict
acceptance inequalities, nonadjacent competitor definition and tie order.
Every operating point reruns both directions and all downstream acceptance
stages. Plot error versus retained GT coverage; also record false support on
no-correspondence controls. These thresholds are not confidence probabilities.
Replay all nine recorded pairs at cutoff 120/margin 3 for each initial cost;
the confidence sweep uses synthetic GT. Evaluation replays each frozen finalist
on all nine pairs at its selected thresholds.

Rank costs using the integer-only arm and quality gates below. Retain at most
one cost/operating point per normalization variant. For those finalists only,
test clipping pairs `(Tl,Tg)={(8,4),(16,8),(32,16)}` with the same cutoff/margin
grid on development inputs. Select at most one configuration per variant and
freeze weights, clipping, thresholds and refinement mode before evaluation.
If no configuration passes, record a failed experiment rather than relaxing
the gates after inspecting evaluation results.

## Separate selection from fractional fitting

Changing a unary also changes the current parabola's input samples. Keeping its
implementation unchanged does not isolate integer selection. Run these arms
for baseline and every candidate operating point:

1. **Integer-only:** retain accepted integer winners with zero fractional
   offset, then rerun LR, metric range, neighbor support and speckle cleanup.
   Compare nearest-integer selection, accepted/missing GT, error tails and
   boundary leakage. Half-pixel truth has two equally close integer winners;
   score against the closest integer set, not ties-to-even rounding alone.
2. **Existing parabola:** fit each method's own cost using the current formula,
   neighbor-availability checks and ±1-pixel clamp. Rerun both directions and
   every acceptance stage. This measures an executable candidate's combined
   selection/refinement behavior, separately from the integer-only result.

For each frozen finalist, also hold its integer-only supported/visible-GT mask
and integer winners fixed, then compare zero offsets with offsets from its own
cost parabola. Report the same diagnostic for baseline and on the common mask.
Do not rerun acceptance for these diagnostic rows; they have no candidate
coverage claim. A fixed-mask improvement cannot override a failed executable
quality gate. The existing-parabola baseline must separately reproduce archived
baseline stages; integer-only baseline is a new control, not an archive match.

## Fresh evaluation split and decision gates

Reserve seeds `104729, 130363, 15485863, 32452843` for final evaluation. Do not
generate or inspect their outcomes until the configuration manifest is frozen.
Changing a setting after viewing these results requires another new split.
Evaluate both finalists and baseline on every scene; include per-seed results
and equal-scene summaries so textured scenes cannot hide weak-texture failures.

Use the unchanged bad-depth definition `error > max(0.04 m, 5% true depth)`.
Report accepted MAE, accepted bad-depth fraction, bad depth per all visible GT,
missing GT, foreground errors and background-boundary leakage. Use each
method's own accepted mask and identical common masks. Undefined accuracy at
zero accepted pixels is a failure to supply evidence, not zero error.

For a conservative offline shortlist, require no pooled per-scene increase
relative to the corresponding baseline arm in accepted bad-depth percentage,
bad depth per all GT, missing GT, or boundary leakage. Require no increase in
unrelated-pair false support; blank and periodic ambiguity controls must still
reject all support. Require lower common-mask disparity MAE on the independent
±3 and ±6 noise scenes. For each arm, compare against both the baseline at its
default cutoff 120/margin 3 and the best development-selected baseline point
from the same grid that satisfies these non-regression gates relative to that
default. Freeze a baseline point separately for each arm. Rank eligible points
by mean common-mask MAE against the default baseline over the two noise scenes;
break ties by greater mean retained GT coverage over those scenes, then lower
cutoff, then higher margin. Apply the same ranking to eligible composite
settings; remaining ties use smaller alpha, beta, Tl, then Tg. If a required
common mask is empty, the MAE gate is unmet. Report any individual-seed regression
explicitly even when pooled gates pass. Apply the gates independently to the
integer-only and executable existing-parabola arms; retain only passing modes.
Check these gates both before and after unchanged speckle cleanup; report
common masks for both stages. Rank eligible settings using the pre-cleanup
noise-scene common-mask MAE and coverage defined above.

These strict gates choose an offline research candidate; they do not establish
safe headset occlusion. Add held-out recorded motion phases and real surfaces
at independently known optical depth before a live quality claim. Sloped-plane
GT, measured sensor-noise controls and raw exposure-pairing tests still need
fixture/capture work; keep those limitations explicit.

## Implementation and reproducibility requirements

Implementation is confined to the opt-in offline header, comparison tool,
runner and analyzer. The CLI provides `composite-raw`, `composite-normalized`,
and `--refinement integer|parabola|both`. The separate composite runner and
analyzer preserve the older masked-census workflow. The existing follow-up
reproduction command only analyzes archived stages.

Before measurement, verify zero-weight cost and full-stage parity, correct
left/right residual addressing, signed gradient arithmetic, cost bounds,
normalization on constant images, and OMP=1/8 deterministic outputs. Build on
Frame in an isolated directory with `-O3 -fopenmp`; record compiler/binary/source
and input hashes. Save the frozen configuration manifest before evaluation,
exact commands, component diagnostics and stage outputs for at least one
recorded frame and one evaluation seed. Refuse output overwrites and keep the
existing archives intact. CPU replay timing does not establish Vulkan speed;
GPU porting and complete worker/publication timing follow only a quality pass.

## Run the implementation

In an isolated Frame checkout containing the tools, headers, scripts, Makefile
and this protocol, build and validate before generating measurement outputs:

```sh
make -j4 bin/compare_stereo_matchers bin/test_stereo_matcher_experiment
OMP_NUM_THREADS=8 bin/test_stereo_matcher_experiment
python3 scripts/run_stereo_composite_experiment.py \
  --binary bin/compare_stereo_matchers --output results/composite --phase smoke
python3 scripts/analyze_stereo_composite_experiment.py results/composite --phase smoke
python3 scripts/run_stereo_composite_experiment.py \
  --binary bin/compare_stereo_matchers --inputs PATH_TO_ORIGINAL_INPUTS \
  --output results/composite --phase development
python3 scripts/analyze_stereo_composite_experiment.py results/composite --phase development \
  --archive PATH_TO_ORIGINAL_MATCHER_ARTIFACTS
```

If `development-selection.json` contains finalists, continue with the bounded
clipping ablation and freeze a manifest. With no finalist, stop at the documented
quality failure; do not open evaluation inputs.

```sh
python3 scripts/run_stereo_composite_experiment.py \
  --binary bin/compare_stereo_matchers --output results/composite --phase sensitivity
python3 scripts/analyze_stereo_composite_experiment.py results/composite --phase sensitivity --freeze
python3 scripts/run_stereo_composite_experiment.py \
  --binary bin/compare_stereo_matchers --inputs PATH_TO_ORIGINAL_INPUTS \
  --output results/composite --phase evaluation
python3 scripts/analyze_stereo_composite_experiment.py results/composite --phase evaluation
```

The runner saves exact commands, source snapshots, compiler identity and hashes.
It uses `--reuse-selection` to share identical matching evidence between arms
and duplicate baseline controls, while rerunning LR/range, neighbor support and
cleanup. Reused rows have unavailable CPU timing rather than a misleading full
matcher time; independently matched rows retain complete CPU-call timing.
`--plan-only` writes a plan without executing; `--resume` skips only hash-verified
complete jobs under the identical plan. Partial jobs require explicit archival
before retrying. Evaluation refuses an absent/changed frozen manifest or changed
development evidence. Both arms run for attribution, but only frozen passing
modes count as evaluation candidates. The analyzer refuses incomplete sweeps,
reports per-scene gates and per-seed regressions, writes confidence points to a
standalone SVG, and keeps fixed-mask refinement rows separate from coverage.
