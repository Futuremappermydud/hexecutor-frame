# Guarded larger-support census: development protocol

Frozen before measurement, 2026-10-07. Follow-up to the bounded adaptive-support
screen. Test whether independent census anchors can stop aggregation and censor
descriptor samples across likely depth boundaries, while a separate exposure
control reduces luma-support fragmentation. This is an offline development
experiment, with no production or camera changes and no evaluation seeds.

Hold radius 8, raw luma threshold 32, cutoff 120/margin 3 and the existing
texture floor/sample minimum fixed. Cross four guard modes {0,1,2,3} with two
exposure modes {raw,contrast-scaled}: eight configurations, all seventeen scenes
at seeds 1709/991/56039/65537 and all nine archived recorded pairs (104 jobs).
Mode 0/raw must reproduce the previous r8/t32 stages exactly.

Construct anchors from an independent baseline five-sample census match with
cutoff 90/margin 6 and integer refinement. An anchor needs an accepted match in
both directions with integer LR disagreement <=1. Use anchors solely to shape
support; do not propagate depth, force winners or remove disparity hypotheses.
Missing anchors provide no boundary claim. Anchor generation is identical for
integer/parabola candidate arms and included in full matcher CPU timing.

Guard mode 1: stop an aggregation arm before a sample with an anchor more than
one disparity pixel from that arm's center label. Use the center's anchor when
available; otherwise use the first anchor encountered on the arm as its label.
Check both correspondence neighborhoods independently. Missing anchors do not
stop an arm. Descriptor bits remain unchanged.

Guard mode 2: when a descriptor center and its 7x7 neighbor both have anchors,
exclude that comparison if their integer disparities differ by more than one.
Keep unknown-anchor comparisons. At each support sample intersect the two
masks, require at least 24/48 bits and normalize its disagreements to 48 units.
Stop an arm before a sample with insufficient joint bits. Require five samples
and sum the normalized samples into round(5*sum/sampleCount), 0..240 units.
Guard mode 3 combines modes 1 and 2. Mode 0 uses neither guard. Anchors are
estimates, not ground truth, and the guards can miss boundaries or reject good
support. Descriptor masks use only their own center's anchor, with no inferred
labels in unknown regions. No census noise-bit suppression is introduced.

Exposure mode contrast-scaled uses each image's global 10th/90th histogram
percentile span, with scale=max(0.5,span/64). Multiply its center/adjacent luma
threshold and texture floor by this scale independently in each view. Compute
histograms over the offline input image; do not normalize census or mix luma
into the cost. The raw mode uses scale 1 exactly. This isolates a bounded
contrast-based threshold proposal, not an exposure estimator or correction for
saturated pixels. Clipping destroys evidence that this control cannot recover.

Run both executable integer/parabola arms, all downstream decisions and cleanup.
Apply the unchanged conservative per-scene pooled gates against the default
baseline before and after cleanup, requiring common-mask MAE gains on both noise
scenes. Also report comparisons against the unguarded r8/t32 control, per-seed
regressions, support samples/bits, stopped arms and anchor counts. Undefined
accepted precision is not a pass. Recorded coverage is descriptive and cannot
select accuracy parameters. Do not relax gates or choose thresholds afterward.
If none passes, record the specific gains/losses and stop this screen. If any
passes, it still needs separately frozen confidence calibration before evaluation
or GPU/live trials. Reserved seeds remain untouched regardless of this result.

Save immutable commands/source/binary/input hashes before execution, complete
job manifests and representative synthetic/recorded stage diagnostics. Refuse
partial overwrites; resume only verified jobs under the identical plan. Verify
zero-guard parity, bilateral anchor consistency, mask addressing and information
bounds, exposure-scale behavior, OMP=1/8 determinism and baseline archive parity.
Build and run controls in an isolated native Frame directory with -O3 -fopenmp.
Native smoke timing and local CPU timing make no Vulkan or live speed claim.
