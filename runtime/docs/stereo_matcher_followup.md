# Better stereo methods after the matcher experiments

Investigation date: 2026-10-07. This extends
[the matcher comparison](stereo_matcher_experiments.md) with saved-stage analysis,
an offline refinement diagnostic, and primary-source research. No production
matcher, headset process, completion, history, or mesh policy was changed.

The best next quality experiment is **better matching evidence and adaptive
support**, followed by guarded regularization. Porting the existing SGM reference
to Vulkan remains useful for performance research, but its small support gain
does not by itself justify a live replacement. Simply restoring raw-cost
subpixel refinement improves average precision while worsening error tails.

**Next-step experiment completed:** the [offline masked-census unary trial](stereo_unary_experiment.md)
improves textured noisy precision but loses all held-out weak-texture support,
increases unrelated-pair false support, and reduces recorded supported area from
11.900% to 9.876%. It is not a production candidate. The next unary ablation is
a bounded luma/gradient residual with separate exposure-normalization controls.
The [composite-unary protocol](stereo_composite_unary_plan.md) defines the costs,
development grid, fresh evaluation seeds, refinement controls and quality gates.
The opt-in implementation is available; a quality result requires the complete
development and, for a passing finalist, frozen evaluation workflow.

**Composite development experiment completed:** the
[implemented composite-unary trial](stereo_composite_unary_experiment.md)
completed 767 local development jobs with separate native Frame validation.
No tested nonzero-weight configuration passed the quality gates. The default
raw and normalized variants increase recorded support while increasing
unrelated-pair false support and clipped-exposure errors. No finalist was
selected, and the reserved evaluation seeds remain unused. Revise the evidence
or confidence proposal in a new development protocol before advancing stages.

**Adaptive-support screen completed:** the [bounded dual-view census trial](stereo_adaptive_support_experiment.md)
ran six configurations across four development seeds and nine recorded pairs.
Larger support improved noisy and weak-texture matching and reduced unrelated
false support, but every arm failed non-regression gates, including broad
silhouette leakage and exposure coverage loss. No finalist or live replacement
was selected. Investigate boundary-safe descriptor support and anchor consistency
in a new protocol; preserve the unused evaluation split.

**Boundary/exposure follow-up completed:** the [guarded larger-support screen](stereo_guarded_support_experiment.md)
ran 104 jobs covering anchor arm stops, descriptor boundary masks and an
independent contrast-scaling control. Combined guards retain most gains and
reduce broad-silhouette leakage, but no arm passes the unchanged clean-scene,
boundary and exposure gates. Saved-stage attribution points to missing/wrong
anchors at residual leaks; exposure clipping remains a coverage limitation.
Production and the reserved evaluation split remain unchanged.

## What the existing results actually constrain

The recordings contain no independent room-depth ground truth. Nine recorded
pairs can establish stage losses and reproducibility, but cannot choose an
accurate dense matcher by coverage. The synthetic tests establish a particular
noise/texture tradeoff; they omit real optics, residual rectification errors,
camera timing, and realistic sensor noise.

The unary confidence guard is intentionally restrictive: a selected SGM winner
must have independent cost separation from every nonadjacent unary hypothesis.
Consequently, extra paths or larger penalties cannot accept a completely
ambiguous patch without changing the evidence policy. Improving the unary or
collecting useful evidence over a larger, appropriate support region is the
direct route to more measured support. A truly uniform surface still has no
unique stereo correspondence; depth extended into it from anchors is inferred.

### Where support disappears

The new analysis counts saved float stages and checks that filter rejections
partition the accepted left matches. These are means over the same nine frames,
with **the entire 616×616 image as denominator**. Loss columns are percentage
points of that image, rather than conditional rejection percentages.

| Stage or loss | Baseline | Tested edge matcher | Guarded SGM 8/48 |
| --- | ---: | ---: | ---: |
| Accepted left hypotheses | 61.513% | 60.578% | 44.189% |
| Lost to LR consistency | 31.517% | 34.202% | 19.952% |
| Lost to metric range | 3.127% | 2.811% | 2.634% |
| Depth before neighbor support | 26.869% | 23.565% | 21.604% |
| Lost to neighbor support | 14.968% | 14.269% | 9.253% |
| Supported depth | 11.900% | 9.297% | 12.351% |
| Lost to unchanged speckle cleanup | 2.203% | 2.342% | 1.868% |
| Depth after cleanup | 9.697% | 6.954% | 10.483% |

Baseline LR rejection removes 51.2% of its accepted left hypotheses. Neighbor
support then removes 55.7% of its in-range consistent depth; SGM removes 42.8%
at that stage. SGM starts with fewer accepted matches and finishes with slightly
more coherent support. These counts explain the gain without demonstrating that
the surviving depths are correct. LR losses include missing accepted right
matches as well as numerical disagreement; they do not prove bad calibration.

Evidence: [per-frame stage losses](../artifacts/stereo_matcher_followup/20261007/stage-losses.csv)
and [summary with input hashes](../artifacts/stereo_matcher_followup/20261007/summary.json).

### Separating integer selection from subpixel refinement

The reference fits a parabola to the four-path averaged SGM cost. I recomputed
the original five-position census unary at each saved accepted integer winner
and its adjacent disparities. Every recomputed center cost matches the saved
unary exactly. A raw fit is used only at a local unary minimum with positive
curvature and an available neighbor on both sides; the offset is bounded to
±0.5 pixels. Otherwise it retains the original SGM fit.

The following diagnostics **hold the original SGM supported/GT mask fixed**.
They retain the integer winner and do not rerun right matching, LR, range,
neighbor support, or speckle decisions. They measure the effect of the fit on
already selected pixels, not the quality or coverage of an executable candidate.
The unchanged fit reproduces all 18 archived SGM scene/seed/penalty error rows
within 2e-6; accepted GT counts match exactly.

| Scene, seed; SGM 8/48 mask | SGM disparity MAE | Guarded raw-fit MAE | SGM accepted bad depth | Raw-fit bad depth |
| --- | ---: | ---: | ---: | ---: |
| Clean fractional plane, 1709 | 0.1286 px | 0.1155 px | 0.015% | 0.072% |
| Clean fractional plane, 991 | 0.1271 px | 0.1142 px | 0.015% | 0.132% |
| Weak texture, 1709 | 0.2223 px | 0.2466 px | 12.999% | 19.146% |
| Weak texture, 991 | 0.2139 px | 0.2375 px | 11.790% | 17.999% |
| Silhouette step, 1709 | 0.1348 px | 0.1227 px | 0.130% | 0.234% |
| Silhouette step, 991 | 0.1330 px | 0.1207 px | 0.098% | 0.224% |

Bad depth retains the original threshold: error greater than
`max(0.04 m, 5% true depth)`. On the common baseline/SGM clean-plane pixels,
raw-fit MAE is 0.1150/0.1137 px versus baseline 0.1151/0.1139 px for seeds
1709/991. Thus different accepted masks do not explain away the average-precision
finding. The same raw-fit behavior persists across the 4/24 and 16/96 variants.

Two bounded follow-ups also fail to provide a free improvement: a 50% blend of
raw/SGM fits and switching to raw only when the fits agree within 0.125 px both
reduce clean-plane MAE but worsen its bad-depth rate and weak-texture errors.
Those choices are exploratory parameters, not selected defaults. A simple
raw-versus-regularized agreement check is insufficient confidence.

Evidence: [all fixed-mask refinement rows](../artifacts/stereo_matcher_followup/20261007/fixed-mask-refinement.csv).
The nearest-integer disagreement field is descriptive; exact half-pixel truth
has two equally close integer hypotheses and must not be counted as a wrong
match solely from ties-to-even rounding.

## Prioritized methods

The ordering below is an engineering recommendation from these results. None
of the unimplemented methods has measured Frame accuracy or timing.

| Priority | Candidate | Why investigate it | Main constraint |
| --- | --- | --- | --- |
| 1 | Noise-aware/composite unary with bounded adaptive aggregation | Addresses correspondence ambiguity before confidence rejection | Exposure robustness and support crossing silhouettes |
| 2 | Continuous photometric refinement of guarded integer hypotheses | Targets the subpixel tradeoff directly | Noise, low gradient, and false minima require rejection |
| 3 | MGM versus four/eight-path SGM using the same unary | Tests stronger 2D regularization and directional artifacts | GPU dependencies, evidence guards, and memory traffic |
| 4 | Bounded slanted-plane/PatchMatch refinement | Useful for sloped floors, tables, and walls | Plane leakage, thin objects, and propagation cost |
| Separate backend experiment | Trained HITNet or MobileStereoNet | Learned descriptors/context may improve difficult texture | Actual portable backend cost on Turnip |
| Offline comparator | Fast-FoundationStereo | Useful external learned estimate on recorded pairs | NVIDIA-specific timing and no independent truth |

### 1. Better evidence before stronger smoothness

The current descriptor makes a hard comparison against one center sample.
Small independent intensity changes can flip many comparisons in low texture.
[Quantized Census](https://openaccess.city.ac.uk/4760/1/QCensus3DVWorkshop2014.pdf)
explicitly studies quantizing neighbor-center differences to tolerate ordering
changes from radiometric distortions. A noise-aware or quantized descriptor is
a plausible unary ablation; its benefit on IMX616 remains a hypothesis.

[AD-Census](https://xing-mei.github.io/research/assets/adcensus.pdf) combines
bounded intensity and census costs and aggregates over dynamically constructed
cross regions. This differs substantially from dropping a few samples from the
existing five-position cross. The failed edge experiment did not test larger
adaptive support or change the 7×7 descriptor that can itself cross a boundary.

Proposed offline sequence:

1. Keep the existing aggregation and compare the current census against a
   quantized/noise-aware unary. Report effective descriptor information as well
   as cost separation; suppressing noisy bits must not make flat patches confident.
2. Compare census plus a bounded luma/gradient residual. Evaluate raw luma and
   exposure-normalized variants separately using gain, bias, clipping, and
   independent noise tests. The current pipeline consumes IMX616 Y; an RGB
   adaptation would incur chroma capture and should be a separate experiment.
   Use the [concrete protocol](stereo_composite_unary_plan.md), including an
   integer-only arm: retaining the current parabola code while changing its
   input cost does not hold the resulting subpixel fit fixed.
3. Holding the unary fixed, test bounded cross-based aggregation using support
   from both correspondence neighborhoods. Record support count and normalize
   the cost. Require useful texture and consistent anchors; luma similarity
   alone is insufficient evidence that samples share a surface.
4. Apply guarded SGM to a promising unary/aggregation combination only after
   measuring each change independently.

The three-cost margin, 120-cost cutoff and P1/P2 are defined in the existing
0–240 cost units. A changed cost/descriptor needs an explicit scale and
confidence calibration; carrying those numbers across unchanged would be an
uncontrolled threshold experiment.

### 2. Refine against continuous image evidence

A better next refinement trial is a narrow continuous correspondence search
around the guarded integer basin, using interpolated luma or gradients with
local brightness compensation. Require curvature/information, retain the
original fit when refinement is underconstrained, and prevent support from
crossing established disparity boundaries. This is a proposed method, not a
validated improvement.

[Birchfield–Tomasi](https://ai.stanford.edu/~birch/publications/dissimilarity_pami1998.pdf)
provides a sampling-insensitive dissimilarity built from interpolated intensity
neighborhoods. It motivates examining resampling effects; that cost is not
itself a complete continuous subpixel solver or a substitute for exposure
handling. A continuous luma/gradient residual should be tested separately from
integer unary selection to preserve attribution.

Rerun both directions and all acceptance stages for any complete candidate.
Compare MAE and error tails on identical common masks as well as each method's
own accepted mask. A lower MAE cannot compensate for more false foreground.

### 3. MGM and additional SGM paths

[More Global Matching](https://www.bmva-archive.org.uk/bmvc/2015/papers/paper090/paper090.pdf)
couples each scan to an additional predecessor from a neighboring scanline,
addressing the weak coupling behind SGM's characteristic disparity streaks.
The paper reports about 20% overhead in its CPU examples. GPU scheduling must
respect the resulting two-dimensional dependencies, typically with wavefront
processing; independent-row SGM kernels cannot simply substitute this recurrence.

Compare MGM with SGM under identical unaries, evidence guards and refinement.
Account for the paper's data-term overcounting correction as its own ablation.
Eight-path SGM is also a reasonable orientation-bias control. Neither method
can create independent matching evidence in blank regions. The wearer's current
streaks occur under a local matcher, so they have not been identified as the
specific SGM artifact studied by MGM.

For a Vulkan SGM trial, one 616×616×96 volume has 36,427,776 entries. A uint8
unary plus uint16 summed volume occupies 109.3 MB (104.2 MiB) for one reference,
before images, path state and outputs. Packed storage, subgroup reductions and
sequential direction processing are implementation options. Query device
subgroup and storage features; use supported packed representations. Do not
retain every path volume or assume a CUDA kernel maps directly to Turnip.
[libSGM](https://github.com/fixstars/libSGM) is a CUDA implementation reference,
not an available Frame backend. The inactive `stereo_sgm.comp` is local WTA,
not a completed SGM implementation.

### 4. Plane hypotheses as matching proposals

[PatchMatch Stereo](https://www.bmva-archive.org.uk/bmvc/2011/proceedings/paper14/paper14.pdf)
matches continuous slanted support planes rather than restricting patches to
one integer fronto-parallel disparity. That model is relevant to sloped room
surfaces, which ordinary fixed-shift support does not represent well.

A bounded experiment could seed planes from strong current stereo anchors and
test their warped image correspondence over small support regions. This differs
from accepting a downstream plane solely because depth anchors fit it. Retain
independent evidence checks and explicit inferred provenance for unsupported
extensions. Full PatchMatch adds propagation/random-search work and is a later
trial, after the unary and fractional-refinement ablations.

### Learned alternatives and actual backend feasibility

[HITNet](https://arxiv.org/abs/2007.12140) uses hierarchical initialization,
geometric propagation and slanted hypotheses without a full 3D-convolution cost
volume. It is an attractive architecture to evaluate. However, Google's
[released-model documentation](https://github.com/google-research/google-research/tree/master/hitnet)
states that default TensorFlow operators run three times slower than their
custom CUDA version. Published CUDA speed is not a portable deployment result.
Test a trained model, its preprocessing, disparity units/range and complete
exported graph before optimizing it. The project's dummy HITNet graph supplies
no quality evidence.

[MobileStereoNet](https://arxiv.org/abs/2108.09770) offers both 2D and 3D designs;
the 2D variant is a useful backend comparison for avoiding expensive 3D
operators. Reduced operations or parameter counts do not establish Frame speed.
For both models, luma-channel handling and input resizing must preserve metric
disparity scale and be evaluated against the recorded camera distribution.

[Fast-FoundationStereo's official repository](https://github.com/NVlabs/Fast-FoundationStereo)
reports representative 640×480 runtimes on an RTX 3090: 14.0–23.4 ms with
TensorRT, 29.3–49.4 ms with PyTorch, and roughly 646–653 MB peak memory.
Those measurements do not establish Adreno feasibility. It can provide an
offline learned comparison on another suitable machine; disagreement with it
does not label current depth as wrong, and agreement does not establish truth.

The project's [backend investigation](qnn_htp_handoff.md) takes precedence over
the older AI-model exploration: QNN GPU kernels are incompatible with the
stock Turnip route, generic HTP access is unavailable, and the existing MNN
StereoNet experiment takes 450–500 ms at 256×256. Do not assume QNN/AI Hub solves
deployment or disturb `dsp_service`. Any learned trial needs a measured portable
runtime, including transfers, bidirectional verification and GPU contention.

## Measurement and implementation gates

Before attributing room LR losses entirely to an algorithm, inspect strong
correspondences for residual vertical disparity across the field of view,
exposure/blur asymmetry and differences between static and moving sequences.
Preserve the full factory-extrinsic rectification and right-camera parity.
Recorded rectified luma alone cannot validate raw camera exposure pairing.

Use all nine existing frames for comparisons, then add held-out frames from
every motion phase. Split parameter-development inputs from final evaluation;
two fixed synthetic seeds are insufficient for choosing a confidence rule.
Extend controls with realistic independent noise, gain, clipping, small vertical
misregistration, sloped planes, thin silhouettes, and motion/pairing failures.
Track missing area, accepted error, bad depth per all GT pixels, and boundary
leakage together. Evaluate confidence using error-versus-retained-coverage
curves rather than an uncalibrated single uniqueness margin. This evaluation
principle is discussed by
[Poggi et al.](https://openaccess.thecvf.com/content_ICCV_2017/papers/Poggi_Quantitative_Evaluation_of_ICCV_2017_paper.pdf).

Require controlled real surfaces at independently known optical depth,
silhouettes, and moving hands before claiming reliable occlusion. Keep inferred
coverage distinct from measured depth. Measure complete production p50/p95
processing and publication age, thermal behavior, and render pacing on Frame.
The observed 17.5–21.1 ms GPU reference time is a comparison point, not unused
budget: the full worker also captures, completes and builds surfaces. No timing
for a new GPU method has been established here.

The first **offline unary comparison using the existing confidence and GT
harness** is now [implemented and measured](stereo_unary_experiment.md), with
explicit cost normalization, confidence operating points and held-out seeds.
The tested masked-census guard does not satisfy the quality gate. Continue with
the composite unary ablation, holding aggregation and all other stages fixed,
before testing adaptive support. Follow the
[composite-unary protocol](stereo_composite_unary_plan.md) and freeze its selected
configuration before opening the new evaluation split. A separate refinement
task should compare continuous residuals and the existing SGM fit. Port only a quality
candidate that survives those controls, then run a single-change headset trial.
The completed [composite development sweep](stereo_composite_unary_experiment.md)
selected no finalist; clipping sensitivity and final evaluation are therefore
not triggered for this trial.

## Reproduce this investigation's additional measurements

```sh
python3 scripts/analyze_stereo_refinement.py \
  artifacts/stereo_matcher_comparison/20261007 \
  --output artifacts/stereo_matcher_followup/20261007
```

The script uses the Python standard library and reads saved outputs only. It
checks stage partitioning, support subsets, identical stereo inputs across
penalty variants, selected raw unary parity, and original synthetic metric
reproduction. Output CSVs and input/source hashes are separate from the original
experiment. These checks validate the attribution analysis; they do not validate
a new live matcher.
