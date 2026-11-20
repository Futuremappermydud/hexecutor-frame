# Stereo depth and confidence

The native Vulkan matcher is the baseline real-time depth path. It operates on calibrated rectified luma and returns metric depth with validity checks. The CPU path is a fallback and numerical/development tool. The optional learned backend is documented separately in [model integration](model-depth.md).

## GPU pipeline

The shader pipeline constructs local census descriptors, searches disparity in both directions, and filters matches using uniqueness, left/right consistency, and local support. The documented default search is 96 disparities at 616 × 616. Repeated device measurements placed that native path around 15–24 ms in its recorded configuration; that is depth compute, not a universal capture-to-display latency.

Reject textureless, ambiguous, out-of-range, and inconsistent samples before converting disparity into points. More filled pixels are not necessarily more usable geometry. Cross-edge aggregation and subpixel refinement can make a surface look smooth while introducing silhouette leakage or range bias.

The CPU rectifier/matcher shares geometry with the GPU path. It should remain a practical fallback without blocking the render loop. OpenMP and optimized builds matter for its full-resolution workload.

## Experimental matchers

The tree also contains SGBM and guarded/adaptive/composite matching experiments. Their offline runners freeze scenes, input identities, and source snapshots to make comparisons meaningful. They are not automatically promoted when one scene improves. The three concise [composite](stereo_composite_unary_plan.md), [adaptive](stereo_adaptive_support_plan.md), and [guarded](stereo_guarded_support_plan.md) protocols are retained under their stable filenames because the experiment runners hash them.

Backend choices in the cloud worker use `FRAME_CLOUD_STEREO`; the occlusion worker uses `FRAME_OCCLUSION_STEREO`. The learned model is selected only for the ordinary cloud path in the current worker. Keep the occlusion backend's independent acceptance boundary intact.

## Evaluation that affects implementation

Compare metric bias, invalid/false-valid rates, edge leakage, temporal stability, and execution time together. Include untextured regions, repeated patterns, gain/exposure changes, vertical misalignment, occlusions, and thin silhouettes. Scene-disjoint evaluation matters more than adding more nearly identical frames to a training split.

Changing support radius, exposure compensation, uniqueness thresholds, or refinement changes the estimator. Keep baseline controls and evaluate the same raw pixels; do not let automatic selection consume held-out evaluation data. GPU speed improvements also need headset/frame-pacing evidence, because the GPU is shared with the compositor.

Source: [Vulkan matcher](../../depth-runtime/src/vulkan_stereo_pipeline.cpp), [guarded matcher](../../depth-runtime/src/vulkan_guarded_stereo_pipeline.cpp), [CPU/reference geometry](../../depth-runtime/src/depth_pipeline.cpp), and [experiment implementation](../../depth-runtime/include/stereo_matcher_experiment.hpp).
