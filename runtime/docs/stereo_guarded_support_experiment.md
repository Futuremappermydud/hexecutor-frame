# Guarded larger-support census: development results

Experiment date: 2026-10-07. The [frozen boundary/exposure experiment](stereo_guarded_support_plan.md)
is complete. **No configuration passed either complete refinement arm.**
The guards improve the specific broad-silhouette problem while retaining most
weak-texture and noise gains, but clipped-exposure coverage still regresses.
Keep the production baseline and the reserved evaluation seeds untouched.

## Experiment and implementation

Eight configurations cross guard modes 0/1/2/3 with raw or contrast-scaled
support thresholds. All keep radius 8, raw threshold 32, cutoff 120/margin 3.
The 104 jobs replay all seventeen synthetic scenes at four development seeds
and all nine recorded pairs. Both integer and executable parabola arms rerun
both directions, LR, range, neighbor support and unchanged speckle cleanup.
The same conservative per-scene gates apply before and after cleanup. This
is a development screen, not held-out accuracy or a production replacement.

The opt-in `guarded-adaptive` method constructs integer LR-consistent anchors
from an independent baseline match at cutoff 90/margin 6. Mode 1 stops cross
arms at anchor disparity changes; mode 2 masks census comparisons between
known conflicting anchors and normalizes the remaining joint bits; mode 3
combines them. Unknown anchors do not imply a surface label. Anchors shape
support without forcing selected disparities or publishing inferred depth.
The minimum 24 bits per sample, five support samples and texture floor remain
explicit information guards. Anchor generation is included in CPU timing.

The exposure ablation scales each view's luma support threshold and texture
floor by its global 10th/90th-percentile contrast span. It does not modify the
census descriptor, add photometric costs or restore saturated observations.
This simple global control reduces some fragmentation but is not an exposure
estimator. It also changes support in the other scenes, which are retained in
the gates. The unguarded/raw control reproduces the previous screen exactly.

## What improved and what still fails

The combined-guard/contrast-scaled row is an illustrative ablation, **not a
selected finalist**. Synthetic metrics pool the four development seeds. Area
uses the full image; accepted bad depth uses each method's own visible-GT mask.
These rows use the executable parabola arm before cleanup.

| Metric | Production baseline | Previous larger cross | Combined guards + contrast scaling |
| --- | ---: | ---: | ---: |
| Weak-texture supported area | 37.926% | 83.549% | 82.191% |
| Weak-texture accepted bad depth | 15.279% | 2.821% | 3.830% |
| Noise ±3 accepted bad depth | 0.294% | 0.118% | 0.073% |
| Noise ±6 accepted bad depth | 1.896% | 0.445% | 0.338% |
| Broad-silhouette background leakage | 0.127% | 0.300% | 0.161% |
| Thin-silhouette background leakage | 0.718% | 0.126% | 0.413% |
| Unrelated-pair false support | 12.152% | 8.229% | 10.057% |
| Clipped-gain missing visible GT | 54.085% | 63.471% | 61.791% |
| Mean recorded supported area | 11.900% | 19.547% | 20.089% |
| Mean recorded area after cleanup | 9.697% | 17.124% | 17.846% |

On the combined candidate's identical common masks with baseline, noise ±3
MAE improves from 0.12125 to 0.10046 px and ±6 from 0.14310 to 0.11031 px.
These are paired comparisons for this candidate; common masks differ across
configurations, so they must not be used as identical three-method masks.
Recorded captures have no independent room-depth GT; their area cannot choose
accuracy settings.

The separate ablations explain the direction of the effects:

| Guard mode | Raw broad-silhouette leakage | Contrast-scaled leakage |
| --- | ---: | ---: |
| None | 0.2998% | 0.2422% |
| Anchor arm stops | 0.2191% | 0.1845% |
| Descriptor boundary masks | 0.2537% | 0.2076% |
| Both | 0.1845% | 0.1614% |

The combined candidate reduces pooled broad-silhouette leaks from 26 to 14
of 8,672 evaluated boundary pixels; baseline has 11. It also reduces total
accepted bad depth in that scene from 219 baseline pixels to 124, while
accepting more GT. That broad improvement coexists with the specific leakage
regression that the protocol disallows. Guards worsen thin-boundary leakage
and unrelated support relative to the previous larger cross, even though both
remain below baseline. They also give back some weak-texture precision.

The parabola arm's remaining pre-cleanup failures are clean integer-scene
missing GT and error tails, broad-silhouette leakage, and clipped-gain missing
GT. The clean integer scene has three accepted bad-depth pixels among 133,420
accepted GT, versus baseline zero among 133,584. Clipped gain is a larger
coverage loss: it accepts 50,840 of 133,056 GT versus baseline 61,092.
Cleanup does not produce a pass. The combined integer arm also fails clean
and exposure coverage gates; after cleanup it additionally loses noise MAE
gains and regresses broad-silhouette error tails. No arm is promoted by a
fixed-mask or integer-only diagnostic.

## Why the remaining boundary errors matter

A post-screen diagnostic on saved **seed-1709 only** reproduces the archived
leakage counts and attributes remaining errors to anchor availability. Of the
combined candidate's six broad-silhouette leakage pixels, five have no left
anchor and one has an incorrect left anchor. All six use some censored
census bits; two have an anchor-stopped arm. There are valid left anchors at
1,929 of 2,168 evaluated boundary pixels, but their presence on average does
not cover the particular failure locations. This is diagnostic attribution,
not proof of causality, a new selection criterion or held-out evidence.

These findings support boundary guarding as a useful direction, while showing
why independent baseline anchors are not sufficient boundary truth. Missing
and wrong anchors remain concentrated among the residual failures. The
contrast control cannot recover information lost to clipping, and the support
sample/texture gates still remove many exposed pixels.

## Validation and saved evidence

Local and isolated native Frame builds used `-O3 -fopenmp`. Existing matcher,
masked-census, composite and adaptive controls pass, along with new controls
for full-stage zero-guard parity, bilateral LR anchor addressing, unknown-anchor
handling, descriptor bit geometry, insufficient information rejection,
affine contrast scaling, blank rejection, invalid guard settings and OMP
parity. Five new manifest/grid workflow tests and the seven existing composite
workflow tests pass.

The unguarded control and unchanged adaptive method match 2,016 saved stage
files against the prior r8/t32 archive. All eight frame-70 baseline controls
match the original archive across 160 stages. Native OMP=1/8 replays of combined
guards with contrast scaling match 3,942 saved float/PGM files. Native source
hashes match the frozen local sources. The native replay uses one development
seed; it is a determinism check, not a second full accuracy sweep. No GPU or
live performance claim follows from these CPU timings.

The analyzer verifies the complete 104-job plan, binary and fourteen source
snapshots, exact file inventories, scene/common-mask completeness and 33,792
output hashes. It retains per-seed regressions, fixed-mask fitting CSVs,
comparisons against both baseline and the larger-cross control, selected support
samples/bits/stopped-arm diagnostics and anchor counts. Selected support
statistics include rejected integer winners; they are not accepted-only counts.
The later boundary-attribution script consumes saved stages and records its
own source/input hashes without changing the frozen experiment or matcher.

[Full pooled gates and per-seed results](../artifacts/stereo_guarded_support/20261007/development/summary.json),
[frozen commands and input/source hashes](../artifacts/stereo_guarded_support/20261007/development/plan.json),
[boundary attribution](../artifacts/stereo_guarded_support/20261007/development/boundary-diagnostics.json),
[unguarded archive parity](../artifacts/stereo_guarded_support/20261007/local/unguarded-parity.json),
[baseline archive parity](../artifacts/stereo_guarded_support/20261007/local/baseline-parity.json),
[native OMP parity](../artifacts/stereo_guarded_support/20261007/frame/determinism.json)
and [native validation](../artifacts/stereo_guarded_support/20261007/frame/build-test.log)
retain the evidence. Frozen sources are stored alongside the results.

Reproduce into a new directory:

```sh
make -j4 bin/compare_stereo_matchers bin/test_stereo_guarded
OMP_NUM_THREADS=8 bin/test_stereo_guarded
python3 tools/test_stereo_guarded_workflow.py
python3 scripts/run_stereo_guarded_experiment.py --binary bin/compare_stereo_matchers \
  --inputs artifacts/stereo_matcher_comparison/20261007/inputs --output NEW_DIRECTORY
python3 scripts/analyze_stereo_guarded_experiment.py NEW_DIRECTORY
python3 scripts/diagnose_stereo_guarded_boundaries.py NEW_DIRECTORY
```

The runner refuses overwrites and permits `--resume` only for an identical
frozen plan and hash-verified completed jobs. It has no evaluation mode.

## Next decision

Retain this as evidence that larger support and boundary guards are productive
research directions, rather than a passing replacement. A new proposal should
isolate a conservative treatment of boundary regions with missing/inconsistent
anchors, including how much useful GT that treatment rejects. Exposure handling
needs a separate sample-information proposal for clipped support, rather than
another globally relaxed threshold. Clean-scene fractional fitting also needs
attribution: its few tails persist despite improved average precision. Freeze
any such experiment before measurements, preserve the current quality gates,
and continue to leave the evaluation split unused.
