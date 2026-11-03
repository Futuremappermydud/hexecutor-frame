# Bounded dual-view census support: development screen

Date: 2026-10-07. Follow-up to the failed composite-unary experiment. Hypothesis:
collecting more unchanged census observations within a bounded region improves
noise precision without the composite residual's exposure sensitivity.

This first screen tests a contiguous four-arm skeleton, not the full 2D regions
or AD-Census cost of [Mei et al.](https://xing-mei.github.io/research/assets/adcensus.pdf).
At each disparity, arms stop at the first center/adjacent luma difference above
the threshold in **either** view, or an incomplete 7x7 census neighborhood.
Use radii {2,4,8} and luma thresholds {16,32}; retain every census bit and round
5*sum(Hamming)/sampleCount into 0..240 units. Require at least five samples and
luma range at least four in each view; otherwise cost is 240. Similar luma and
this texture floor do not establish same-surface support. In particular, the
7x7 descriptor can still cross silhouettes. No anchor propagation is tested.
Export selected sample count through the existing information stage; for this
method it counts samples, not masked-census bits.

Freeze this six-point screen before measurement. Use all seventeen scenes at
development seeds 1709,991,56039,65537, plus all nine archived recorded pairs.
Confidence stays cutoff 120/margin 3; this screen changes aggregation only and
does not calibrate confidence. Run both integer and existing-parabola arms,
rerunning both directions, LR, range, neighbor support and unchanged cleanup.
Apply the composite protocol's existing per-scene pooled non-regression gates
and common-mask noise MAE gains before and after cleanup against the default
baseline. Undefined accepted accuracy cannot pass. Recorded area is descriptive
and must not select accuracy settings. Save per-seed metrics and common masks.

If every configuration fails, stop and record failure; do not interpret this as
an exhaustive adaptive-support or threshold search. If one passes, design a
separate confidence calibration and anchor-consistency experiment before any
finalist or GPU port. Reserved evaluation seeds remain untouched. No production
matcher, headset process, camera calibration or runtime deployment is involved.
Record sources, binaries, inputs, commands and output hashes; refuse overwrites.
Validate bounds, bilateral stopping, blank rejection and OMP determinism; build
in an isolated Frame directory with -O3 -fopenmp for native validation. Local CPU
timing provides no Vulkan or live performance claim.
