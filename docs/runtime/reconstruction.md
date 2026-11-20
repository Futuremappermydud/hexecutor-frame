# World reconstruction

Reconstruction consumes depth observations with a camera-to-world pose. It cannot recover metric truth from an inconsistent projection, biased calibration, or a pose taken at the wrong time. Preserve validity, confidence, timing, and calibration provenance before choosing a fusion method.

## Representations

The cloud worker can accumulate accepted points for a world-locked cloud and can feed a sparse TSDF for a mesh. `persistent_depth` provides a separate persistent-depth representation. These serve different publication and rendering paths; enabling one does not automatically make another the occlusion source.

`TsdfVolume` uses a hash of 4 × 4 × 4 voxel leaves. A voxel stores half-precision SDF and weight, with luma and contradiction bookkeeping. The baseline integrates projective signed distance with inverse-square range weighting and an incidence factor. Optional confidence multiplies that weight. Weight is capped so an old observation does not make a voxel permanently immutable.

The baseline configuration in `tsdf_volume.hpp` uses 3 cm voxels, a 12 cm truncation half-band, maximum weight 32, and accepted depth 0.25–4 m. Callers can override these values; record the effective configuration rather than assuming every experiment used the defaults.

## Update and mesh behavior

Observations behind the truncation band are left untouched. Free-space updates in front of a surface can carve stale geometry. `TsdfIntegrateOptions` exposes robust weighting, per-frame weight caps, optional normal-based incidence, and carving controls. Defaults retain the baseline behavior.

Surface extraction uses surface nets and requires sufficient corner weight. Leaf allocation and triangle caps bound memory/work; extraction prioritizes geometry near the supplied focus when capped. Low latency and a large filled mesh are not evidence that previously observed room surfaces remain correctly placed.

## Downstream CPU processing

The learned-depth path validates disparity and reprojects it into rectified metric depth before eight-connected speckle rejection. Live mesh modes then apply the separable inverse-depth power-mean filter and triangulate accepted depth cells. Persistent mesh mode integrates the accepted depth into the TSDF and periodically extracts surface nets.

Spherical conversion computes left-ray tangents per column and elevation cosines per row. Confidence expansion uses a 256-entry uncertainty lookup with the existing weight formula. Speckle rejection and filtering retain worker-owned scratch capacity; every frame resets validity and output values so rejected samples cannot survive from an earlier frame. The filter supports radii 0 through 2. Live triangulation lazily projects a shared grid point once using a two-row cache, retaining the original tear gates, triangle ordering, face normals and shading.

TSDF integration also retains its touched-leaf/work buffers. Surface extraction skips a leaf neighborhood when its sufficiently weighted voxels have only one sign, since it cannot emit any crossing. Mixed neighborhoods still use the same surface-net construction and focus ordering. Mesh publication reuses buffers through the existing mutex-protected swap, with all processing on the background depth worker.

`[Mesh]` logs separate `postprocess`, `speckles`, `filter` and `mesh` milliseconds. `[TSDF]` logs `postprocess`, `speckles`, `integrate` and `mesh`; the mesh time remains the most recent scheduled extraction/publication time. The model's `postprocess` covers CPU metric conversion and confidence expansion, including spherical-map construction on first use. `stereo` remains the backend's inference time.

From `depth-runtime/`, build and run the CPU-only checks and synthetic benchmark:

```sh
make bin/test_depth_postprocess bin/test_meshing_pipeline bin/test_tsdf_volume bin/benchmark_depth_processing
bin/test_depth_postprocess
bin/test_meshing_pipeline
bin/test_tsdf_volume
OMP_NUM_THREADS=4 bin/benchmark_depth_processing 60
```

The benchmark opens no cameras or NPU session. It reports warm stage p50/p95, full depth/filter/mesh checksums, triangle counts and projection calls for a synthetic wall with a foreground step and an uncertainty hole. Its geometry conversion timing excludes worker IPC, temporal state and confidence expansion. The live mesh uses a test pinhole/pose fixture. Match compiler options, OpenMP thread counts and iteration counts for comparisons; the TSDF accumulates observations throughout the run. `FRAME_PROCESSING_BASELINE` selects the original allocation-returning APIs when compiling the same benchmark against a saved pre-change source/header snapshot.

A 2026-11-06 host comparison on an Apple M4 Pro used Apple Clang 21, `-O3 -std=c++17`, and a single thread (OpenMP disabled). Each process warmed up for three frames and measured 60 frames; the table is the median stage p50 across five alternating before/after pairs, against a snapshot of the working tree before these optimizations.

| CPU stage | Before, ms | After, ms |
| --- | ---: | ---: |
| Spherical metric conversion | 0.873 | 0.708 |
| Speckle rejection | 2.398 | 1.955 |
| Power-mean filter | 3.510 | 3.089 |
| Live triangulation | 4.481 | 3.865 |
| TSDF integration | 5.450 | 5.146 |
| TSDF extraction | 3.680 | 2.353 |

The full depth, filter, live geometry/color and TSDF geometry/luma checksums matched in every pair. Both versions emitted 178,435 live triangles and 31,640 TSDF triangles from 3,370 leaves. Live projection calls fell from 535,305 to 90,154. The integration difference was small and varied across runs. These observations establish host implementation savings; headset/OpenMP performance, capture-to-publication latency and real-room quality still require separate device measurements. Neither the benchmark nor these changes establish an algorithmic performance limit.

## Boundaries that need separate evaluation

Distinguish per-frame depth error from fused-surface error. Repeated frames can reinforce a systematic calibration bias. A pose error often produces multiple displaced surfaces, while disparity uncertainty can produce local thickness or silhouette leakage. Measure these separately before tuning weight caps or carving to hide the symptom.

The replay tools allow the same recorded observations to be integrated under different configurations. Compare configuration, source hashes, pose timing, spatial coverage, and retained versus rejected samples. Saved outputs must identify whether geometry came from RGB depth, IR refinement, or a combined diagnostic run.

This guide describes the implemented representations. It does not replace or modify the ongoing temporal-fusion report or its related notes.

Source: [TSDF interface](../../depth-runtime/include/tsdf_volume.hpp), [integration and meshing](../../depth-runtime/src/tsdf_volume.cpp), [persistent depth](../../depth-runtime/src/persistent_depth.cpp), and [worker](../../depth-runtime/src/cloud_worker.cpp).
