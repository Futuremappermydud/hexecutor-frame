# Bounded adaptive census support: development results

Experiment date: 2026-10-07. The next aggregation ablation after the failed
[composite unary](stereo_composite_unary_experiment.md) is complete under a
[separately frozen protocol](stereo_adaptive_support_plan.md). **None of the six
configurations passed either complete refinement arm. Keep the production baseline.**

The screen ran 78 jobs: radii 2/4/8 crossed with dual-view luma thresholds 16/32,
each on all seventeen synthetic scenes at four development seeds and all nine
recorded IMX616 pairs. Confidence stayed at cutoff 120/margin 3. Integer and
existing-parabola arms independently reran LR, range, neighbor support and
unchanged speckle cleanup. Reserved evaluation seeds were never generated.
This is a fixed-confidence development screen, not an exhaustive adaptive
aggregation search or held-out accuracy claim.

## What changed

The opt-in `adaptive` matcher retains the full 7x7 census descriptor and replaces
the five isolated cross samples with contiguous bounded four-arm support.
Every arm stops at an edge in either correspondence neighborhood. Costs are
normalized by actual sample count into the original 0..240 units. Hypotheses
with fewer than five samples or luma range below four in either view receive
cost 240. Selected sample counts are exported in information files; they are
sample counts for this method, rather than the masked descriptor's bit counts.
The implementation tests a cross skeleton, not full AD-Census regions. Similar
intensities do not establish a shared surface, and census itself can still
cross depth boundaries. No confident-anchor propagation was implemented.

## Measured tradeoffs

Radius 8/threshold 32 illustrates both the gains and the remaining failures;
it is **not a selected finalist**. Synthetic rows pool the four development
seeds. Supported area uses the whole image. Bad depth uses each method's own
accepted visible-GT mask. Common-mask MAE uses exactly the same accepted pixels
for both methods. These rows use the executable parabola arm before cleanup.

| Metric | Baseline | Radius 8 / threshold 32 |
| --- | ---: | ---: |
| Recorded supported area, mean over nine pairs | 11.900% | 19.547% |
| Recorded area after cleanup | 9.697% | 17.124% |
| Weak-texture supported area | 37.926% | 83.549% |
| Weak-texture accepted bad depth | 15.279% | 2.821% |
| Noise ±3 common-mask disparity MAE | 0.12127 px | 0.10228 px |
| Noise ±6 common-mask disparity MAE | 0.14313 px | 0.11408 px |
| Unrelated-pair false support | 12.152% | 8.229% |
| Broad-silhouette background boundary leakage | 0.127% | 0.300% |
| Clipped-gain missing visible GT | 54.085% | 63.471% |

The larger cross collects useful evidence in weak texture and independent
noise. That is a more promising direction than the tested composite residuals,
but its gains do not satisfy the conservative non-regression gates. For this
configuration the integer arm fails seven pre-cleanup gates, including broad
silhouette leakage and coverage loss. The parabola arm fails eleven, including
small clean-scene error-tail regressions and exposure losses. Cleanup does not
remove those failures. Smaller radii and tighter edge thresholds also fail;
the tighter threshold particularly fragments support in noisy scenes.
Blank and both periodic ambiguity controls retain zero supported depth.
Recorded captures have no independent room-depth truth; greater area is not
an accuracy result.

## Validation and evidence

Local and isolated native Frame builds used `-O3 -fopenmp`. Existing controlled,
masked-census and composite tests passed, along with new controls for blank
rejection, target-only edge stopping, normalized cost bounds, bilateral address
symmetry and invalid-radius rejection. Native OMP=1/8 replays match across
2,718 saved float/PGM files for seed 1709 at radius 4/threshold 16. This native
replay is determinism validation, not a second full development sweep. Across
all six recorded frame-70 controls, 120 saved baseline stages are byte-identical
to the original archive. An audit verifies all 78 job manifests, source snapshots,
scene completeness and 17,568 output hashes. It separately reports individual
seed regressions and recorded coverage. CPU replay timing establishes no live
or Vulkan performance claim.

[Development summary](../artifacts/stereo_adaptive_support/20261007/development/summary.json),
[verified jobs and per-seed regressions](../artifacts/stereo_adaptive_support/20261007/development/audit.json),
[frozen commands and hashes](../artifacts/stereo_adaptive_support/20261007/development/plan.json),
[baseline parity](../artifacts/stereo_adaptive_support/20261007/local/baseline-parity.json),
[native determinism](../artifacts/stereo_adaptive_support/20261007/frame/determinism.json)
and [final native tests](../artifacts/stereo_adaptive_support/20261007/frame/final-build-test.log)
retain the evidence. The archived sources precede a test-name switch fix and
addition of the Makefile test target; matching code was unchanged afterward.

Reproduce into a new directory:

```sh
make -j4 bin/compare_stereo_matchers bin/test_stereo_matcher_experiment bin/test_stereo_adaptive
OMP_NUM_THREADS=8 bin/test_stereo_adaptive
python3 scripts/run_stereo_adaptive_experiment.py --binary bin/compare_stereo_matchers \
  --inputs artifacts/stereo_matcher_comparison/20261007/inputs --output NEW_DIRECTORY
python3 scripts/analyze_stereo_adaptive_experiment.py NEW_DIRECTORY
```

The next bounded research proposal should isolate descriptor boundary support
and confident-anchor consistency, then independently test exposure-sensitive
support construction. It should preserve these gains while addressing the
specific silhouette and coverage regressions. Confidence calibration alone
cannot establish same-surface support. Freeze a new development protocol before
measurement; no GPU port or live quality trial follows from this failed screen.

**Follow-up completed:** [guarded support and contrast-scaled thresholds](stereo_guarded_support_experiment.md)
retained most noise/weak-texture gains and reduced broad-silhouette leakage
from 0.300% to 0.161%, but no full refinement arm passed the clean-scene,
boundary and clipped-exposure coverage gates. No production replacement
or evaluation run was selected.
