# Occlusion worker performance

`FRAME_OCCLUSION_PROFILE=1` prints `[OcclusionProfile]` after each 120 published
occlusion frames. It collects only scalar timings, with no camera/depth recording
or disk writer. Each line reports throughput, recording-frame count, pose
fallback count, and p50/p95 timings for the same stages as the benchmark.
Percentiles use the analyzer's floor-index convention. Publication age starts at
the DMA-read midpoint, not sensor exposure, and ends when the mesh is published.
Throughput includes waits, rejected duplicate captures, and stage work within
the window. Individual stage distributions include accepted frames only.

For representative production timings, leave `FRAME_OCCLUSION_BENCHMARK` and
`FRAME_OCCLUSION_RECORD` unset and check `recording_frames=0`. Compare windows
under similar lighting, view, filter, headset temperature, and OpenMP settings;
completion work depends on how much depth is visible. The existing demo logs do
not report capture or individual completion/history percentiles.

## Luma-only capture

The cloud/depth worker requests Y only by default. It never consumes UV. Set
`FRAME_CAPTURE_UV=1` to restore Y+UV transfers for an otherwise identical A/B run.
The diagnostic RGB monitor and other callers still request Y+UV by default.
After a luma-only capture, UV getters return null until a successful RGB capture.

This retains all ring-slot Y reads, hashes, pair rejection, automatic DMA-BUF
discovery, and Vulkan external image ownership barriers. It removes only UV GPU
transfers, UV host invalidation, and copies of selected UV planes. It does not
skip ring slots or assume a fixed active camera FD. Both modes import the same
buffers and allocate the same initial storage. Initialization still performs one
RGB capture, so startup cost is unchanged. Rectification and stereo use exactly
the same selected luma planes in both modes.

The capture stage includes recording/submitting transfers, GPU waiting,
invalidation, Y hashing, slot selection, and selected Y copies. Substage GPU
profiling has not been added, so a capture improvement cannot be assigned solely
to bandwidth or CPU memcpy. Optimizing slot count without reliable camera
sequence metadata could change pairing/freshness and requires separate work.

## Verification

The changed worker, DMA wrapper, and detiler compile on Frame with `-O3
-fopenmp -std=c++17`. The integrated demo and occlusion layer also built successfully on Frame.
Integrated completion, temporal surface, benchmark codec, Vulkan renderer, and
occlusion-layer regression suites passed.
A hardware smoke check switched RGB → luma → RGB with auto-discovered DMA
slots and verified non-null Y, nonzero hashes, and UV getter validity in each
mode. A headset visual check remains necessary for camera
alignment and hand trails; scalar timings cannot establish those properties.

## Confidence configuration

With `FRAME_OCCLUSION_TEMPORAL=confidence`,
`FRAME_OCCLUSION_CONFIDENCE=2,2,3,150,200,200` selects the three observation
counts (strong measured, other measured, inferred), followed by inferred,
measured, and pending expiry in milliseconds. Counts must be integers in
`[1,255]`; expiry must be finite and in `(0,1000]` milliseconds. All six comma
separated fields are required. Invalid values log a diagnostic and retain
baseline confidence settings. Legacy ignores this configuration.

The parser was checked with valid input, fractional counts, extra/missing
fields, wrong separators, zero count/expiry, out-of-range expiry, and the legacy
mode. This control does not change which temporal filter runs by default.

## Live comparison, 2026-10-07

Two sequential modes used the same isolated binaries, `OMP_NUM_THREADS=8`,
completion enabled, legacy history, and no image/depth recorder. Each ran two
120-frame windows; only the second window is compared to exclude startup and
compilation contention. Both compared windows had zero pose fallbacks.

| Metric | Y+UV | Y only |
| --- | ---: | ---: |
| Depth throughput | 9.132 Hz | 9.769 Hz |
| Capture p50 / p95 | 33.606 / 40.875 ms | 25.413 / 32.955 ms |
| Completion p50 / p95 | 23.960 / 30.685 ms | 24.090 / 28.757 ms |
| Publication age p50 / p95 | 92.608 / 105.384 ms | 89.532 / 99.401 ms |

Luma-only capture reduced capture median by 8.19 ms (24.4%) and p95 by
7.92 ms (19.4%); observed throughput increased 7.0%. Full runtime logs are
`artifacts/occlusion_performance/20261007/capture_rgb.log` and
`capture_luma.log`. These short sequential windows are engineering evidence,
not a controlled room benchmark: the scene and GPU scheduling can change, and
there is no captured sensor timestamp or visual ground truth. Completion
medians were similar, but other stage variations mean publication-age and
throughput changes cannot be attributed entirely to chroma transfer.

The demo's own render log reported approximately 36 FPS in both modes. These
changes improve the depth worker; they do not establish the desired 70–90 Hz
render-loop target or explain the runtime's render pacing.

## Deployment and candidate smoke check

The optimized sources, demo, layer, analyzer, and evaluator were built in the
normal `/home/steamos/frame-spatial-runtime` project. The previous sources and
binaries are backed up at `/tmp/frame-occlusion-pre-optimization-20261007.tar.gz`.
The original full benchmark recording remains unchanged.

The guided script's candidate configuration was checked end to end with
`--smoke --temporal confidence --confidence 2 2 3 200 250 250`. It recorded and
replayed 27 complete frames with no drops, gaps, or pose fallbacks. Live startup,
the manifest, and replay all used the requested configuration. Reports are in
`artifacts/occlusion_benchmark/20261007-183319/`. This was an engineering smoke
check, without a wearer performing the motion protocol; it is not visual
validation. The demo was then returned to legacy history with luma-only capture,
completion enabled, and recording disabled. The full candidate headset
evaluation was subsequently run as `20261007-183447`; see
[the benchmark results and wearer feedback](occlusion_benchmark.md#candidate-headset-evaluation-2026-10-07).
The wearer reported improved overall occlusion and some aligned table edges,
but weak depth beyond a few feet. Distance-dependent support loss is the next
quality investigation; the runtime default remains legacy.

## Application render pacing diagnostics

`FRAME_OPENXR_RENDER_PROFILE=1` enables scalar CPU wall-time windows in the demo,
with no image recording. Every 120 completed frames it reports p50/p95 of
`xrWaitFrame`, tracking/view location, application mesh upload, swapchain image
acquire/wait, each eye's draw plus fence wait, `xrEndFrame` including the API
layer, complete frame work, and frame-start period. The window includes frames
with `shouldRender=false`; their skipped stage durations are zero and the
`rendered` count makes this visible. Frame-start period includes event polling
between frames; the first sample uses complete frame duration. Failed frames
are not included. These are CPU elapsed times, not GPU timestamp queries.

Combine this with `FRAME_OCCLUSION_RENDER_PROFILE=1` for the layer's prepare,
record, submit, and CPU fence-wait timing, and `FRAME_OCCLUSION_PROFILE=1` for
worker timings. Disable both recorders before comparing independent production
windows. Application `xrEndFrame` includes mask work and downstream runtime
submission; compare the layer timings to attribute its cost. Never add
individual p95 values into an end-to-end p95.

The demo also skips copying the published camera mesh into its application
vertex list: it renders its fixed virtual panels, while the API layer consumes
the camera mesh. The point-cloud viewer retains its existing cloud-copy path.
This removes a redundant full-mesh copy under the render-loop cloud mutex. The
updated application builds on Frame; production frame-rate distributions and
GPU execution timing remain to be measured on matched live windows.
