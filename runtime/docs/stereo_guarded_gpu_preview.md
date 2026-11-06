# Guarded GPU preview (2026-10-07)

The combined adaptive-support research matcher (g3/e1/r8/t32/c120/m3)
has an opt-in Vulkan implementation in `VulkanGuardedStereoPipeline`.
It is a preview, not a promotion of the failed offline quality trial.

Seven GPU passes (eight with opt-in precomputed arm limits) compute census, baseline integer seeds, bidirectional
anchors, boundary masks, guarded correspondence, LR/range filtering, and
neighborhood support. Host work computes exposure percentiles and performs
the existing speckle cleanup. Camera capture and rectification are unchanged.

Native validation on Adreno 750 passed fourteen cases: seven scenes each
with integer and parabolic refinement, including recorded frame 70. Winners,
confidence, anchors, masks, and support evidence matched the CPU exactly.
Accepted disparities agreed within one float ULP; five support-threshold
flips in parabolic cases were individually explained by float rounding.
Repeated calls and regular/diagnostic output controls passed. The initial
unsplit run took 423–785 ms per pair (host submission through fence,
including transfers; single trials, not GPU timestamp benchmarks).
Evidence is in `artifacts/stereo_guarded_gpu/20261007/parity-final.log`.

The comparison HUD tool `bin/preview_stereo_guarded` uses GPU matching by
default (`--cpu` retains the reference path). The live point-cloud viewer
instead uses `FRAME_CLOUD_STEREO=guarded bin/frame_openxr_point_cloud` and
has no comparison HUD. See [point-cloud operation](openxr_point_cloud.md).
The panel occlusion wrapper now selects the guarded matcher by default;
`FRAME_OCCLUSION_STEREO=baseline` restores the original matcher.

An initial point-cloud run confirmed over 31,000 stable voxels but unsplit
matching blocked graphics submissions for roughly half a second. The
point-cloud backend now selects eight-row submissions with a 3 ms host
yield between batches. The native parity test compares batched regular
output against unsplit diagnostic output. Render cadence is measured
separately with `FRAME_OPENXR_RENDER_PROFILE=1`.

The batched native rerun passed all fourteen CPU parity cases, with regular
batched output equal to unsplit diagnostic output and two deterministic
repeats per case. Cloud geometry and visibility regression checks passed.
The restarted live application rendered approximately 72 FPS while depth
pairs took 758–763 ms including batch yields; it accumulated 16,492 stable
points by pair 20. Evidence: `parity-interleaved.log` and
`point-cloud-interleaved.log` in the same artifact directory. These are
short live observations, not a long-duration performance guarantee.

The validated isolated deployment is
`/home/steamos/stereo-guarded-cloud-20261007`, with `point-cloud.pid` and
`point-cloud-interleaved.log`. The comparison HUD process is stopped.

Next work: [optimization plan](stereo_guarded_optimization_plan.md) and
[persistent room mesh plan](room_mesh_plan.md). Deployment logs describe the
tested runs; consult the headset process state for the currently running mode.

Optimization controls and saved-pair native measurements are documented in
[optimization results](stereo_guarded_optimization_results.md). The original live
batching/matcher remains the default pending focused render-tail validation.
