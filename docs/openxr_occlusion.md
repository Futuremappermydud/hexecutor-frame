# OpenXR passthrough occlusion

The opt-in `XR_APILAYER_FRAME_spatial_occlusion` API layer adds measured room
occlusion to compatible applications using SteamVR's OpenXR runtime. It shares
the working point cloud viewer's IMX616 capture, calibrated rectification,
confidence filtering and capture-time poses. Occlusion completes supported plane
interiors using the rectified camera image, then builds a spatially and
temporally smoothed surface from each new depth frame; the ordinary point cloud
viewer retains voxel fusion and point aging.
It does not patch SteamVR binaries or register itself globally.

## Eligibility

A frame is masked only when the application already uses
`XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND` for passthrough and supplies valid
`XrCompositionLayerDepthInfoKHR` for **both** eyes of a stereo projection layer.
The application must enable `XR_KHR_composition_layer_depth` itself. The camera
worker starts on the first passthrough frame with stereo depth and pauses
when such frames are no longer submitted. Opaque VR,
additive frames, missing depth, invalid poses, expired support, and clouds older
than 500 ms at predicted display time pass through without masking. Passthrough is never enabled on the application's
behalf. The headset was probed with SteamVR/OpenXR 2.18.2: it exposes alpha blend
and composition depth, but no environment-depth extension.

The current implementation supports one Vulkan graphics session per process,
single-sample RGBA/BGRA 8-bit color, sampled D16/D32/D24S8/D32S8 depth, and array
layers or atlas rectangles whose color/depth rectangles have matching extents.
Static/protected swapchains, multiple outstanding acquisitions of one swapchain,
OpenGL, missing sampled-depth usage support, and other formats are skipped.
Applications must preserve their submitted depth attachment (`STORE`), including
when they would otherwise discard it after rendering.

## Build and use on Frame

From `/home/steamos/frame-spatial-runtime`:

```bash
make -j4 occlusion bin/frame_openxr_point_cloud
./scripts/run_with_occlusion.sh /path/to/openxr_application [arguments...]
```

The wrapper sets `XR_API_LAYER_PATH`, `XR_ENABLE_API_LAYERS`, and an absolute
shader directory for that process. It retains existing layer configuration.
Launch an application normally to omit this layer. A Steam launch command can
use the same wrapper before the application's launch command.

The point cloud viewer has a separate test scene:

```bash
./scripts/run_with_occlusion.sh ./bin/frame_openxr_point_cloud --occlusion-demo
```

This scene displays red, green, and blue stage-locked panels, initially 1, 2,
and 3 metres from the left eye. It submits independently acquired color and
D32 depth swapchains with matching near/far metadata. The layer supplies the
camera worker; the demo does not run a second stereo worker. New surface coverage
appears after three agreeing camera observations. The ordinary viewer remains
available by launching without `--occlusion-demo`.

## Frame processing

1. Add sampled usage to supported depth swapchains during creation. If the
   runtime rejects the additional usage, retry the application's original
   request and leave that swapchain unmodified.
2. Defer successful releases of eligible waited images until `xrEndFrame`, where
   the application supplies each projection's pose, FOV, space, and depth range.
   Re-acquisition, destruction, and session termination flush pending releases.
3. In the capture worker, remove disconnected depth speckles, then smooth
   measured inverse depth with a weighted 5×5 kernel. Only neighbors within
   4 cm or 4% of the nearer depth participate; require three supporting neighbors
   and keep missing measurements invalid at this stage. Fit inverse-depth planes
   inside image-connected regions in overlapping 32/64-pixel tiles. Require at
   least 12 distributed measured anchors, 90% agreeing support, and bounded fit
   residuals. Complete only inside the agreeing anchors' convex hull and within
   24 image-connected pixels of an anchor. Strong luma edges, nearby conflicting
   depth, and conflicting tile predictions prevent completion. Estimates never
   seed further fits. Fill remaining small enclosed holes in one pass.
   Search up to eight rectified pixels in each of four directions, requiring
   smoothed measurements on all four sides and
   agreement within 4 cm or 4% of the nearer depth. Interpolate inverse depth;
   do not extrapolate open edges or recursively grow filled regions. Set
   `FRAME_OCCLUSION_COMPLETION=0` to disable plane completion for comparison.
   See [implementation, diagnostics, and measured comparison](occlusion_completion.md).
4. Maintain separate confirmed and candidate surfaces, reprojecting depth,
   observation counts, and age using capture-time head poses and calibrated
   optical-to-head transforms. Keep the nearest sample at collisions. A candidate
   needs three distinct agreeing camera observations within the same 4 cm/4%
   tolerance before it becomes visible or replaces confirmed geometry. Agreeing
   confirmed depths blend inverse depth with 15% current and 85% previous weight.
   Legacy confirmed geometry survives missing or contradictory measurements for
   at most 200 ms since its last agreeing observation. Confidence mode uses its
   configured measured/inferred expiry. Prediction age and source stereo support
   age are separate: agreeing inferred depth never renews direct-measurement age.
   Completion may renew source support only from explicitly identified current
   measured anchors; generic inferred inputs without anchor provenance cannot
   renew source age. Reprojection and rendering never refresh either clock.
   Candidates also expire after 200 ms without support in legacy mode.
   Discard history on capture pauses, invalid tracking, fallback render poses,
   or capture gaps above 200 ms. A fallback render pose preserves pre-history input
   for diagnostics but suppresses final occlusion depth and mesh until capture-time
   tracking returns. Build triangles from two-pixel depth cells,
   requiring every sample in each 3×3 cell to be valid and within the same depth
   tolerance. Transform vertices
   into world space using the capture pose. Locate this surface in the
   application's projection space at display time and draw it through each
   eye. Give both triangles of each cell the earliest remaining support lifetime
   of every covered sample, including coarse-cell interiors and shared edges.
   Skip cells with no remaining support. At each `xrEndFrame`, convert predicted
   display time to monotonic time and pass the age since this mesh's capture-end
   timestamp to the GPU; the fragment shader discards expired triangles even if
   the worker has stopped delivering frames. Fresh regions remain visible when
   another region expires. The publication's absolute monotonic deadline is its
   latest cell deadline, so the layer clears the GPU mesh when every cell has
   expired. Missing time conversion fails closed. The 500 ms global freshness cap
   remains a secondary guard. Capture disable,
   invalid tracking, capture failure, and worker failure clear publication. A capture
   generation token rejects work begun before disable, including rapid disable and
   re-enable during a computation; the next generation restarts confirmation.
   Capture-pose fallback never publishes trusted support deadlines or GPU geometry.
   Upload only when a new camera frame arrives. Sample the application's
   depth at each covered pixel and interpolate measured surface depth.
5. Decode depth to metres, supporting reversed Z and infinite near/far planes.
   When the measured surface plus a 4 cm bias lies in front of virtual depth,
   replace the color with premultiplied transparent black. Foreground virtual
   pixels and pixels outside splats retain their original color.
6. Restore the runtime-required depth attachment layout, wait for mask GPU
   completion, release the images, and forward the frame with the application's
   original blend mode. Masked projections enable texture alpha blending.

Raw camera buffers remain in the existing DMA-BUF ingestion path. Stereo and
surface construction run in a dedicated worker; the frame submission path runs GPU masking
and synchronization only. The layer requires the application's bound Vulkan
queue to be synchronized during OpenXR submission, as the runtime also uses it.

`XR_KHR_composition_layer_depth` itself is a reprojection facility and does not
change composition order. This layer performs the occlusion comparison and
alpha masking explicitly. See the [Khronos depth specification](https://registry.khronos.org/OpenXR/specs/1.0-khr/html/xrspec.html#XR_KHR_composition_layer_depth)
and [API layer loader documentation](https://registry.khronos.org/OpenXR/specs/1.1/loader.html).

## Verification

From the headset project directory:

```bash
g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_renderer.cpp \
  src/occlusion_renderer.pic.o -lvulkan -o /tmp/test_occlusion_renderer
/tmp/test_occlusion_renderer

g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_layer.cpp \
  src/occlusion_renderer.pic.o src/cloud_worker.pic.o src/dma_capture.pic.o \
  src/vk_detile.pic.o src/depth_pipeline.pic.o src/vulkan_stereo_pipeline.pic.o \
  -lvulkan -o /tmp/test_occlusion_layer
/tmp/test_occlusion_layer

g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_surface.cpp \
  -o /tmp/test_occlusion_surface
/tmp/test_occlusion_surface

g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_expiry.cpp \
  -o /tmp/test_occlusion_expiry
/tmp/test_occlusion_expiry

g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_history_events.cpp \
  -o /tmp/test_occlusion_history_events
/tmp/test_occlusion_history_events

g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_capture_generation.cpp \
  src/cloud_worker.pic.o src/dma_capture.pic.o src/vk_detile.pic.o \
  src/depth_pipeline.pic.o src/vulkan_stereo_pipeline.pic.o -lvulkan \
  -o /tmp/test_occlusion_capture_generation
/tmp/test_occlusion_capture_generation

g++ -O3 -fopenmp -std=c++17 -Iinclude tools/test_occlusion_expiry_gpu.cpp \
  src/occlusion_renderer.pic.o src/cloud_worker.pic.o src/dma_capture.pic.o \
  src/vk_detile.pic.o src/depth_pipeline.pic.o src/vulkan_stereo_pipeline.pic.o \
  -lvulkan -o /tmp/test_occlusion_expiry_gpu
/tmp/test_occlusion_expiry_gpu
```

The GPU test reads back actual mask pixels and checks foreground/background,
the depth bias, reversed Z, infinite planes, empty clouds, and untouched pixels
outside the surface and directly beside a sharp silhouette. It also rejects
malformed triangle lists. The surface test checks spatial and temporal noise
reduction, slanted planes, immediate foreground/background changes, translated
and rotated reprojection, nearest-depth collisions, one-frame hole filling,
depth discontinuities, open silhouettes, invalid interior samples, and mesh
capacity. Coverage tests additionally verify that confirmed triangle lists stay
constant when a different quarter of depth samples disappears each frame, that
transient patches remain invisible, that sustained depth changes replace the
surface after three observations, and that missing geometry expires without age
being refreshed by reprojection. The layer contract test checks calibrated capture-pose alignment,
loader negotiation, opaque and
additive frame preservation, stale/absent cloud handling, and deferred releases.
History event diagnostics are opt-in (`enableDiagnostics(true)`), with no event
image allocation in normal processing. They distinguish pending confirmation,
confirmation, measured contradiction clearing, prediction/source expiry, held
support, evidence changes, inferred/measured agreement, transport loss, and
collisions. Transport loss uses the previous image's source coordinates; collisions
and all other event flags use current image coordinates. An offline event timestamp
is the processed capture time, so these logs locate software transitions rather
than the physical time a hand became visible in the display. Source expiry can
leave raw legacy history depth nonzero while its support lifetime is zero; the
published mesh is gated by its support deadline.

The expiry tests cover independent cell and mixed-age interior support, reprojection, unanchored
inference, fresh and old measured anchors, legacy confirmation, and reset. The GPU
expiry test drives the actual layer with a stalled worker and reads mask pixels
before and after predicted display deadlines, including mixed-age regions, repeated rendering,
restart, and capture disable. The generation test deterministically disables and
re-enables capture while an old computation is in flight, rejects its publication,
and checks history reset, confirmation restart, capture-pose fallback suppression, and worker failure.
These deterministic checks do not establish wearer
latency or dynamic scene quality. The existing cloud geometry and visibility tests
continue to apply.

To compare coverage flicker on the same live camera sequence, set
`FRAME_OCCLUSION_RECORD=/tmp/occlusion-sequence.bin` when launching the demo.
The worker records the first 48 processed depth frames and their relative capture
transforms, then closes the file. Compile and replay with:

```bash
g++ -O3 -fopenmp -std=c++17 -Iinclude tools/analyze_occlusion_sequence.cpp \
  -o /tmp/analyze_occlusion_sequence
/tmp/analyze_occlusion_sequence /tmp/occlusion-sequence.bin
```

The tool compares the earlier depth-averaging filter with coverage hysteresis,
reporting mask coverage changes after pose alignment and average covered pixels.
It excludes eight startup frames. Scene changes, head motion, and reprojection
sampling still contribute to this metric; it is not a perceptual headset score.

## Limits

Occlusion coverage follows a confirmed surface aligned to the latest capture
pose. Brief stereo dropouts preserve known geometry for up to 200 ms; sustained
missing depth expires. New or changed surfaces require three observations,
which adds response delay and can briefly leave old geometry visible when an
object moves. Plane completion improves supported interiors; large unsupported
gaps and uncertain boundaries stay open. Inferred surfaces are estimates, not
additional sensor measurements. Luma edges can miss color-only silhouettes,
and same-color objects or incorrect stereo anchors can still lead to false
completion. Coverage
outside the current camera view is not retained. Two-pixel cells can trim
edges by up to two rectified pixels; there is no splat radius extending the
silhouette. Textureless surfaces, thin objects, and movement can still produce
incomplete masks, and stereo matching can misestimate a boundary. The sensor
exposure timestamp is still unavailable, as in the
point cloud sample. Head motion can therefore misregister real and virtual
edges. A reference-space recenter requires restarting the application.

This has been built and exercised on Frame with the included demo and GPU
readback tests. Other applications need individual compatibility checks. No
Vulkan validation layer was installed on the test headset. Mask GPU work waits
before releasing runtime images; render-rate impact depends on cloud coverage,
application workload, and driver behavior and has not been benchmarked across
games.

The experimental confidence-aware history and guided recording/replay workflow
are described in [occlusion_benchmark.md](occlusion_benchmark.md). The default
history remains unchanged until the headset benchmark supports switching it.

## Guarded GPU panel preview (2026-10-07)

`run_with_occlusion.sh` now defaults `FRAME_OCCLUSION_STEREO` to `guarded`.
The shared worker uses the g3/e1/r8/t32/c120/m3 guarded Vulkan matcher with
interleaved eight-row submissions. Override with
`FRAME_OCCLUSION_STEREO=baseline` to compare the original matcher.
The point-cloud selection `FRAME_CLOUD_STEREO` remains independent.

```bash
FRAME_OCCLUSION_STEREO=guarded OMP_NUM_THREADS=2 \
  ./scripts/run_with_occlusion.sh ./bin/frame_openxr_point_cloud --occlusion-demo
```

The guarded preview uses legacy three-observation surface confirmation and
a bounded 2.5-second capture-relative history/publication age, because its
roughly 0.8-second matching time exceeds the old 200 ms lifetime. GPU masks
still expire individual cells against their source deadlines; repeated draws
do not refresh observations. Baseline timing and confidence settings retain
their existing defaults. Guarded mode uses legacy temporal confirmation even
if `FRAME_OCCLUSION_TEMPORAL=confidence` is set. This increases stale-mask
latency for moving objects and is an experimental viewing tradeoff.

This step changes the depth backend and its retention budget. It still uses
the existing temporally confirmed depth surface and plane completion; a
persistent, continually refined room mesh is future work. The guarded
matcher has not passed the offline quality promotion gates.

Native surface and expiry checks pass, including three guarded observations
800 ms apart followed by missing-data expiry. The earlier fourteen-case
matcher test verified batched/unsplit output parity and repeat determinism.
The isolated preview deployment is
`/home/steamos/stereo-guarded-cloud-20261007`; its log is
`panel-occlusion-guarded.log`.

Live validation confirmed `guarded GPU g3/e1`, nonzero confirmed triangles,
and successful stereo projection masking. Rendering remained approximately
72 FPS while guarded matching took 770–784 ms and surface construction
72–87 ms in the observed interval. Log snapshot:
`artifacts/stereo_guarded_gpu/20261007/panel-occlusion-guarded.log`.

### Avoiding unnecessary history resets

Empty alpha-blend submissions now keep capture/history alive for up to 250 ms
after the last depth-bearing projection. They receive no mask, and they do not
renew the grace period or any surface deadline. Longer interruptions,
nonempty incompatible submissions, session changes, and invalid reference
space tracking still disable capture and invalidate history. Brief failed
camera reads retain the last published surface until its existing expiry.

Native regression checks cover one-frame gaps, cumulative gap expiry,
incompatible submissions, explicit reset, and existing surface/missing-data
expiry. The restarted guarded panel preview still renders about 72 FPS and
successfully masks stereo projection. Log:
`artifacts/stereo_guarded_gpu/20261007/panel-occlusion-reset-fix.log`.

The [persistent room mesh plan](room_mesh_plan.md) describes the next geometry
backend and its staged validation. The [matcher optimization plan](stereo_guarded_optimization_plan.md)
separates exact-output performance work from quality-changing experiments.
