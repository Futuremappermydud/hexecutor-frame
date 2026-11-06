# World reconstruction

Reconstruction consumes depth observations with a camera-to-world pose. It cannot recover metric truth from an inconsistent projection, biased calibration, or a pose taken at the wrong time. Preserve validity, confidence, timing, and calibration provenance before choosing a fusion method.

## Representations

The cloud worker can accumulate accepted points for a world-locked cloud and can feed a sparse TSDF for a mesh. `persistent_depth` provides a separate persistent-depth representation. These serve different publication and rendering paths; enabling one does not automatically make another the occlusion source.

`TsdfVolume` uses a hash of 4 × 4 × 4 voxel leaves. A voxel stores half-precision SDF and weight, with luma and contradiction bookkeeping. The baseline integrates projective signed distance with inverse-square range weighting and an incidence factor. Optional confidence multiplies that weight. Weight is capped so an old observation does not make a voxel permanently immutable.

The baseline configuration in `tsdf_volume.hpp` uses 3 cm voxels, a 12 cm truncation half-band, maximum weight 32, and accepted depth 0.25–4 m. Callers can override these values; record the effective configuration rather than assuming every experiment used the defaults.

## Update and mesh behavior

Observations behind the truncation band are left untouched. Free-space updates in front of a surface can carve stale geometry. `TsdfIntegrateOptions` exposes robust weighting, per-frame weight caps, optional normal-based incidence, and carving controls. Defaults retain the baseline behavior.

Surface extraction uses surface nets and requires sufficient corner weight. Leaf allocation and triangle caps bound memory/work; extraction prioritizes geometry near the supplied focus when capped. Low latency and a large filled mesh are not evidence that previously observed room surfaces remain correctly placed.

## Boundaries that need separate evaluation

Distinguish per-frame depth error from fused-surface error. Repeated frames can reinforce a systematic calibration bias. A pose error often produces multiple displaced surfaces, while disparity uncertainty can produce local thickness or silhouette leakage. Measure these separately before tuning weight caps or carving to hide the symptom.

The replay tools allow the same recorded observations to be integrated under different configurations. Compare configuration, source hashes, pose timing, spatial coverage, and retained versus rejected samples. Saved outputs must identify whether geometry came from RGB depth, IR refinement, or a combined diagnostic run.

This guide describes the implemented representations. It does not replace or modify the ongoing temporal-fusion report or its related notes.

Source: [TSDF interface](../../runtime/include/tsdf_volume.hpp), [integration and meshing](../../runtime/src/tsdf_volume.cpp), [persistent depth](../../runtime/src/persistent_depth.cpp), and [worker](../../runtime/src/cloud_worker.cpp).
