#pragma once
// Sparse TSDF modelled on Model's Vogon SdfFusion (depth_engine.so, see
// depth-model/metadata/model-runtime/HIDDEN_STATE.md):
//  - 4x4x4 leaves, each voxel an fp16 SDF + fp16 weight (MaxSdfWeightBuffer),
//  - projective integration with w = 1/z^2 * max(2cos^2(theta) - 1, 0.1),
//    sdf <- (sdf*W + s*w)/(W + w), W <- min(W + w, maxWeight),
//  - voxels behind the band (sdf < -mu) are left untouched; in front they
//    are pushed towards +1, which carves stale surfaces.
// Differences from Vogon: a flat hash of leaves instead of the 4-level tree,
// surface nets instead of marching cubes, no confidence image (the Model
// bridge only returns validated depth), and our own voxel/band/weight values
// because the Model's were not recoverable.
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace FrameSpatial {

struct TsdfConfig {
    float voxel = 0.03f;        // metres
    float truncation = 0.12f;   // band half-width mu, metres
    float maxWeight = 32.0f;    // in 1/m^2 units (one observation at 1 m = 1)
    float minDepth = 0.25f, maxDepth = 4.0f;
    int allocationStride = 2;   // depth pixels sampled when allocating leaves
    float minMeshWeight = 0.5f; // every cell corner must reach this weight
    size_t maxLeaves = 300000;  // ~20 MB of voxel data
};

// world = rotation * camera + translation (rectified camera: x right, y down, z forward)
struct TsdfPose {
    float rotation[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float translation[3]{};
};

struct TsdfVertex {
    float x, y, z;
    float luma;
};

struct TsdfStats {
    size_t touchedLeaves = 0, allocatedLeaves = 0, updatedVoxels = 0, evictedLeaves = 0;
};

class TsdfVolume {
public:
    explicit TsdfVolume(const TsdfConfig& config = TsdfConfig{});
    // depth: metric z per pixel (0 = invalid); luma and confidence optional (may be null).
    // confidence multiplies the sample weight (Vogon fuseDepthMapConfidence).
    TsdfStats integrate(const float* depth, const uint8_t* luma, int width, int height,
                        float focal, float cx, float cy, const TsdfPose& cameraToWorld,
                        const float* confidence = nullptr);
    // Triangle list (3 vertices per triangle), at most maxTriangles. Leaves
    // nearest `focus` (world metres) are meshed first, so a cap trims far geometry.
    void extractMesh(std::vector<TsdfVertex>& triangles, size_t maxTriangles,
                     const float* focus = nullptr) const;
    size_t leafCount() const { return leaves_.size(); }
    const TsdfConfig& config() const { return config_; }
    void clear() { leaves_.clear(); frame_ = 0; }

    // Exposed for tests: SDF in metres and weight at a voxel index (weight 0 = unknown).
    bool voxelAt(int ix, int iy, int iz, float& sdfMetres, float& weight) const;

private:
    struct Leaf {
        __fp16 sdf[64];
        __fp16 weight[64];
        uint8_t luma[64];
        uint32_t lastFrame = 0;
    };
    static uint64_t key(int lx, int ly, int lz);
    const Leaf* find(int lx, int ly, int lz) const;
    void evict();

    TsdfConfig config_;
    std::unordered_map<uint64_t, Leaf> leaves_;
    uint32_t frame_ = 0;
};

}  // namespace FrameSpatial
