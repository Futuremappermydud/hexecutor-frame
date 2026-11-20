#pragma once
// Sparse projective TSDF with 4x4x4 leaves and half-precision SDF and weight.
// Near-surface weights depend on range and incidence. Observations in front
// of the truncation band can carve stale surfaces.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
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

// Optional departures from the baseline update. Defaults reproduce it.
// normal is camera-frame xyz, 3 floats per pixel, used only when normalIncidence
// is set. It is not the E04 estimator.
struct TsdfIntegrateOptions {
    bool robust = false;             // w *= (1 - min(|sdf|/mu, 1))^2
    bool frameWeightCap = false;     // w = min(w, frameWeightCapValue)
    float frameWeightCapValue = 1.f;
    const float* normal = nullptr;
    bool normalIncidence = false;
    int carveAfter = 1;              // full free-space updates of an established surface
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
    // leafList_ holds pointers into leaves_, so a copy must rebuild it.
    TsdfVolume(const TsdfVolume& other);
    TsdfVolume& operator=(const TsdfVolume& other);
    TsdfVolume(TsdfVolume&&) = default;
    TsdfVolume& operator=(TsdfVolume&&) = default;
    // depth: metric z per pixel (0 = invalid); luma and confidence optional (may be null).
    // confidence multiplies the sample weight (Vogon fuseDepthMapConfidence).
    TsdfStats integrate(const float* depth, const uint8_t* luma, int width, int height,
                        float focal, float cx, float cy, const TsdfPose& cameraToWorld,
                        const float* confidence = nullptr,
                        const TsdfIntegrateOptions& options = {});
    // Triangle list (3 vertices per triangle), at most maxTriangles. Leaves
    // nearest `focus` (world metres) are meshed first, so a cap trims far geometry.
    void extractMesh(std::vector<TsdfVertex>& triangles, size_t maxTriangles,
                     const float* focus = nullptr) const;
    // Same surface-nets output, caching each leaf and rebuilding its dirty one-leaf halo.
    void extractMeshIncremental(std::vector<TsdfVertex>& triangles, size_t maxTriangles,
                                const float* focus = nullptr);
    // Fast incremental extraction for publication: one call from voxels to the final vertex
    // buffer. Only leaves whose voxels (or whose neighbours' voxels) changed since the last
    // call are re-meshed, in parallel; the rest come from a compact per-leaf cache.
    // convert(const TsdfVertex&) -> V is applied while the buffer is written. The same
    // triangles as extractMesh, but in cache order unless the cap (maxTriangles) is hit; then
    // they are exactly extractMesh's selection and order.
    template <class V, class Convert>
    void extractMeshAs(std::vector<V>& out, size_t maxTriangles, const float* focus, Convert convert) {
        writeMesh(out, planMesh(maxTriangles, focus, false), convert);
    }
    size_t leafCount() const { return leaves_.size(); }
    size_t lastMeshRebuiltLeaves() const { return lastMeshRebuiltLeaves_; }
    size_t cachedMeshLeafCount() const { return mesh_.list.size(); }
    size_t cachedMeshVertexBytes() const;
    const TsdfConfig& config() const { return config_; }
    void clear() {
        leaves_.clear(); leafList_.clear(); touched_.clear(); work_.clear(); resetMesh(); frame_ = 0;
    }

    // Exposed for tests: SDF in metres and weight at a voxel index (weight 0 = unknown).
    bool voxelAt(int ix, int iy, int iz, float& sdfMetres, float& weight) const;

private:
    static constexpr uint32_t kNoMesh = ~0u;
    // Cached surface-nets triangles of a leaf (count vertices); index is its position in
    // mesh_.list while it has any. changed: queued in mesh_.changed; dirty: claimed for rebuild.
    // A copy starts empty: a copied volume rebuilds its own cache.
    struct MeshSlot {
        std::unique_ptr<TsdfVertex[]> vertices;
        uint32_t count = 0, index = kNoMesh;
        uint8_t changed = 0, dirty = 0;
        MeshSlot() = default;
        MeshSlot(const MeshSlot&) {}
        MeshSlot& operator=(const MeshSlot&) { *this = MeshSlot(); return *this; }
        MeshSlot(MeshSlot&&) = default;
        MeshSlot& operator=(MeshSlot&&) = default;
    };
    struct Leaf {
        __fp16 sdf[64];
        __fp16 weight[64];
        uint8_t luma[64];
        uint8_t contradict[64];
        uint32_t lastFrame = 0;
        uint32_t touchFrame = 0;  // frame_ when last selected by the frustum scan
        MeshSlot mesh;
    };
    struct MeshRef { Leaf* leaf; const TsdfVertex* vertices; uint32_t count; };
    struct MeshPlan { const MeshRef* refs; size_t count, vertices; };
    using LeafEntry = std::pair<const uint64_t, Leaf>;
    static uint64_t key(int lx, int ly, int lz);
    const Leaf* find(int lx, int ly, int lz) const;
    // Flat index of leaves_ for the per-frame frustum scan; rebuilt by evict().
    struct LeafRef { uint64_t key; Leaf* leaf; float c[3]; };
    void addToList(uint64_t k, Leaf* leaf);
    void evict();
    void evictLeaves();
    std::vector<std::pair<float, const LeafEntry*>> meshOrder(const float* focus) const;
    void extractLeaf(uint64_t key, const Leaf& leaf, std::vector<TsdfVertex>& out) const;
    void refreshMesh();
    MeshPlan planMesh(size_t maxTriangles, const float* focus, bool exactOrder);
    void resetMesh() { mesh_ = MeshCache(); }
    template <class V, class Convert>
    void writeMesh(std::vector<V>& out, const MeshPlan& plan, Convert convert) const {
        out.resize(plan.vertices);
        V* dst = out.data();
        const size_t* offset = mesh_.offset.data();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < plan.count; ++i) {
            const MeshRef& ref = plan.refs[i];
            V* d = dst + offset[i];
            for (uint32_t j = 0; j < ref.count; ++j) d[j] = convert(ref.vertices[j]);
        }
    }

    TsdfConfig config_;
    std::unordered_map<uint64_t, Leaf> leaves_;
    // Integration scratch retains capacity between observations. Pointers are
    // rebuilt before use; unordered_map rehash preserves references to leaves.
    std::vector<uint64_t> touched_;
    std::vector<std::pair<uint64_t, Leaf*>> work_;
    std::vector<LeafRef> leafList_;
    std::vector<uint8_t> inView_;
    std::vector<Leaf*> found_;
    std::vector<std::vector<uint64_t>> allocParts_;
    // list: the leaves that have triangles; changed: the leaves integrated since the last
    // extraction (tracked only while valid). Holds pointers into leaves_, so a copy is empty.
    struct MeshCache {
        std::vector<MeshRef> list, capped;
        std::vector<std::pair<uint64_t, Leaf*>> changed, dirty;
        std::vector<size_t> offset;
        size_t vertices = 0;
        bool valid = false;
        MeshCache() = default;
        MeshCache(const MeshCache&) {}
        MeshCache& operator=(const MeshCache&) { *this = MeshCache(); return *this; }
        MeshCache(MeshCache&&) = default;
        MeshCache& operator=(MeshCache&&) = default;
    } mesh_;
    size_t lastMeshRebuiltLeaves_ = 0;
    uint32_t frame_ = 0;
};

}  // namespace FrameSpatial
