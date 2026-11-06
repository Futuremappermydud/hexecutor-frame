# Occlusion render submission and resource lifetime

Stereo eyes with the same color format now record in one command buffer and
share one synchronous queue submission and fence wait. Each target owns a cached
depth descriptor, so recording the second eye cannot replace the first eye's
sampled image. Depth layout transitions still bracket each draw, and the batch
finishes before the layer flushes deferred OpenXR swapchain releases. Stereo
views with different color formats retain synchronous per-eye submissions.

The cache keys include color/depth images and formats, both array layers,
framebuffer extent, and render pass. The layer releases entries using any image
of a destroyed swapchain before calling downstream `xrDestroySwapchain`. A color
format change clears framebuffers, descriptors, and views before replacing the
render pass. Renderer teardown clears the cache before destroying its pass and
descriptor pool. No queue work remains pending after a successful mask call.
The existing supported vertex capacity for coarse/subdivided mesh cells is
preserved.

Frame's real Vulkan GPU regression tests compare a two-submit/two-wait per-eye
reference with a one-submit/one-wait stereo batch. Tests assert two cache hits
and zero misses for a warmed repeated batch, corresponding to zero new image
views/framebuffers. The former uncached implementation created four image views
and two framebuffers per stereo pair; that count follows from the former source.
Readback tests give the two eyes opposite occlusion outcomes and exercise both
separate images and layers of a shared stereo image, catching descriptor aliasing.
Additional cases check color/depth cache invalidation, framebuffer extent, pipeline
format replacement, empty/malformed meshes, reverse/infinite depth ranges,
slanted surfaces, and silhouettes.

The vertex stream also carries each triangle's earliest supporting sample
lifetime relative to the published capture end. All vertices of a triangle have
the same lifetime. The shader receives the submitted display age and discards
expired triangles at equality; the layer's whole-mesh freshness guard uses the
last surviving triangle deadline. This avoids an old region prematurely hiding
fresh regions while keeping each stale triangle expired. Display age advances
without republishing/uploading the mesh, and repeated rendering never refreshes
its support. Missing/NaN/infinite lifetimes fail closed on the timed runtime
path. Untimed standalone diagnostic draws explicitly use a negative display age.

GPU regression tests render two regions with 100 and 300 ms lifetimes from one
upload: both are visible at 50 ms, the older region disappears at 100 ms while
the younger remains at 200 ms, and both disappear at 300 ms. Both eyes are
checked through readback, with one submission/wait per batch throughout. Zero,
negative, NaN, and infinite lifetimes produce no timed masking.

These are offscreen resource/operation counts and GPU correctness checks. The
change has not measured application frame time, production FPS, or a wearer
comparison. It does not establish that mask submission caused the previously
observed approximately 36 FPS, or promise a new render rate.

To collect production CPU submission timings without image recording, enable:

```sh
FRAME_OCCLUSION_RENDER_PROFILE=1 ./scripts/run_with_occlusion.sh \
  ./bin/frame_openxr_point_cloud --occlusion-demo
```

Every 300 successful mask batches, `[OcclusionRenderProfile]` reports p50/p95
milliseconds for resource/pipeline preparation, command recording, queue submit,
and CPU fence waiting, plus cumulative submission/wait counts and cache hits/
misses. These measure the layer's mask work; fence wait includes driver/scheduling
and GPU execution, and is not a GPU timestamp measurement. Use the independent
worker profiler with recording disabled for depth-worker windows. The demo's `FRAME_OPENXR_RENDER_PROFILE=1` supplies application drawing/wait,
`xrWaitFrame`, image acquisition, upload, frame-period, and `xrEndFrame` including
the layer timings; see [application pacing diagnostics](occlusion_performance.md#application-render-pacing-diagnostics).
Actual downstream runtime/GPU attribution still needs live measurement.
