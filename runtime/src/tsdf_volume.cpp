#include "tsdf_volume.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace FrameSpatial {
namespace {
constexpr int kLeaf = 4;   // voxels per leaf edge (Vogon LeafNode<..., 2, 2, 2>)
constexpr int kBias = 1 << 20;

inline int floorDiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
inline int voxelIndex(int x, int y, int z) { return (z * kLeaf + y) * kLeaf + x; }
}  // namespace

TsdfVolume::TsdfVolume(const TsdfConfig& config) : config_(config) {
    leaves_.reserve(config_.maxLeaves / 2);
}

uint64_t TsdfVolume::key(int lx, int ly, int lz) {
    return (uint64_t(uint32_t(lx + kBias)) << 42) | (uint64_t(uint32_t(ly + kBias)) << 21) |
           uint64_t(uint32_t(lz + kBias));
}

const TsdfVolume::Leaf* TsdfVolume::find(int lx, int ly, int lz) const {
    const auto it = leaves_.find(key(lx, ly, lz));
    return it == leaves_.end() ? nullptr : &it->second;
}

bool TsdfVolume::voxelAt(int ix, int iy, int iz, float& sdf, float& weight) const {
    const int lx = floorDiv(ix, kLeaf), ly = floorDiv(iy, kLeaf), lz = floorDiv(iz, kLeaf);
    const Leaf* leaf = find(lx, ly, lz);
    if (!leaf) { sdf = config_.truncation; weight = 0; return false; }
    const int v = voxelIndex(ix - lx * kLeaf, iy - ly * kLeaf, iz - lz * kLeaf);
    sdf = float(leaf->sdf[v]) * config_.truncation;
    weight = float(leaf->weight[v]);
    return weight > 0;
}

TsdfStats TsdfVolume::integrate(const float* depth, const uint8_t* luma, int width, int height,
                                float focal, float cx, float cy, const TsdfPose& pose,
                                const float* confidence) {
    TsdfStats stats;
    ++frame_;
    const float mu = config_.truncation, voxel = config_.voxel;
    const auto& R = pose.rotation;
    const auto& T = pose.translation;

    // 1. Leaves within +-mu of every sampled measurement, along its ray.
    std::vector<uint64_t> touched;
    touched.reserve(size_t(width) * height / (config_.allocationStride * config_.allocationStride));
    const int stride = std::max(1, config_.allocationStride);
    const float leafSize = voxel * kLeaf;
    uint64_t recent[4]{~0ull, ~0ull, ~0ull, ~0ull};
    unsigned next = 0;
    for (int y = 0; y < height; y += stride) {
        for (int x = 0; x < width; x += stride) {
            const float z = depth[size_t(y) * width + x];
            if (!(z >= config_.minDepth && z <= config_.maxDepth)) continue;
            const float rx = (x - cx) / focal, ry = (y - cy) / focal;
            // Sample the band at half-leaf spacing so no crossed leaf is skipped.
            const int steps = int(std::ceil(2 * mu / (0.5f * leafSize)));
            for (int s = 0; s <= steps; ++s) {
                const float zz = z - mu + 2 * mu * s / steps;
                if (zz <= 0) continue;
                const float px = rx * zz, py = ry * zz, pz = zz;
                const float wx = R[0][0] * px + R[0][1] * py + R[0][2] * pz + T[0];
                const float wy = R[1][0] * px + R[1][1] * py + R[1][2] * pz + T[1];
                const float wz = R[2][0] * px + R[2][1] * py + R[2][2] * pz + T[2];
                const uint64_t k = key(int(std::floor(wx / leafSize)), int(std::floor(wy / leafSize)),
                                       int(std::floor(wz / leafSize)));
                // Neighbouring samples mostly share leaves; drop repeats of the
                // last few before the global sort.
                if (k != recent[0] && k != recent[1] && k != recent[2] && k != recent[3]) {
                    touched.push_back(k);
                    recent[next++ & 3] = k;
                }
            }
        }
    }
    // Existing leaves inside the frustum are updated too (Vogon's frustum test),
    // so free space seen through them carves surfaces that are gone.
    const float margin = leafSize * 0.87f;  // half leaf diagonal
    for (const auto& entry : leaves_) {
        const uint64_t k = entry.first;
        const float wx = ((int((k >> 42) & 0x1fffff) - kBias) + 0.5f) * leafSize - T[0];
        const float wy = ((int((k >> 21) & 0x1fffff) - kBias) + 0.5f) * leafSize - T[1];
        const float wz = ((int(k & 0x1fffff) - kBias) + 0.5f) * leafSize - T[2];
        const float pz = R[0][2] * wx + R[1][2] * wy + R[2][2] * wz;
        if (pz < config_.minDepth || pz > config_.maxDepth + mu) continue;
        const float px = R[0][0] * wx + R[1][0] * wy + R[2][0] * wz;
        const float py = R[0][1] * wx + R[1][1] * wy + R[2][1] * wz;
        const float slack = focal * margin / pz;
        const float u = focal * px / pz + cx, v = focal * py / pz + cy;
        if (u < -slack || v < -slack || u > width - 1 + slack || v > height - 1 + slack) continue;
        touched.push_back(k);
    }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    stats.touchedLeaves = touched.size();

    std::vector<std::pair<uint64_t, Leaf*>> work;
    work.reserve(touched.size());
    for (const uint64_t k : touched) {
        auto [it, inserted] = leaves_.try_emplace(k);
        if (inserted) {
            std::fill(std::begin(it->second.sdf), std::end(it->second.sdf), __fp16(1.0f));
            std::fill(std::begin(it->second.weight), std::end(it->second.weight), __fp16(0.0f));
            std::fill(std::begin(it->second.luma), std::end(it->second.luma), uint8_t(0));
            ++stats.allocatedLeaves;
        }
        work.push_back({k, &it->second});
    }

    // 2. Projective update of every voxel in those leaves.
    // world -> camera: R^T (p - T)
    size_t updated = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : updated)
    for (size_t i = 0; i < work.size(); ++i) {
        const uint64_t k = work[i].first;
        Leaf& leaf = *work[i].second;
        const int lx = int((k >> 42) & 0x1fffff) - kBias;
        const int ly = int((k >> 21) & 0x1fffff) - kBias;
        const int lz = int(k & 0x1fffff) - kBias;
        bool any = false;
        for (int vz = 0; vz < kLeaf; ++vz)
            for (int vy = 0; vy < kLeaf; ++vy)
                for (int vx = 0; vx < kLeaf; ++vx) {
                    const float wx = ((lx * kLeaf + vx) + 0.5f) * voxel - T[0];
                    const float wy = ((ly * kLeaf + vy) + 0.5f) * voxel - T[1];
                    const float wz = ((lz * kLeaf + vz) + 0.5f) * voxel - T[2];
                    const float px = R[0][0] * wx + R[1][0] * wy + R[2][0] * wz;
                    const float py = R[0][1] * wx + R[1][1] * wy + R[2][1] * wz;
                    const float pz = R[0][2] * wx + R[1][2] * wy + R[2][2] * wz;
                    if (pz < config_.minDepth) continue;
                    const int u = int(std::lround(focal * px / pz + cx));
                    const int v = int(std::lround(focal * py / pz + cy));
                    if (u < 0 || v < 0 || u >= width || v >= height) continue;
                    const size_t pixel = size_t(v) * width + u;
                    const float d = depth[pixel];
                    if (!(d >= config_.minDepth && d <= config_.maxDepth)) continue;
                    const float sdf = d - pz;
                    if (sdf < -mu) continue;  // occluded: leave unknown/behind untouched
                    const float s = std::min(1.0f, sdf / mu);
                    const float cos2 = pz * pz / (px * px + py * py + pz * pz);
                    float w = std::max(2.0f * cos2 - 1.0f, 0.1f) / (d * d);
                    if (confidence) w *= confidence[pixel];
                    const int idx = voxelIndex(vx, vy, vz);
                    const float W = float(leaf.weight[idx]);
                    const float total = W + w;
                    leaf.sdf[idx] = __fp16((float(leaf.sdf[idx]) * W + s * w) / total);
                    if (luma && s < 1.0f)
                        leaf.luma[idx] = uint8_t(std::lround((leaf.luma[idx] * W + luma[pixel] * w) / total));
                    leaf.weight[idx] = __fp16(std::min(total, config_.maxWeight));
                    any = true;
                    ++updated;
                }
        if (any) leaf.lastFrame = frame_;
    }
    stats.updatedVoxels = updated;
    if (leaves_.size() > config_.maxLeaves) {
        const size_t before = leaves_.size();
        evict();
        stats.evictedLeaves = before - leaves_.size();
    }
    return stats;
}

void TsdfVolume::evict() {
    // Drop leaves that never saw a surface first, then the least recently updated.
    const size_t target = config_.maxLeaves * 9 / 10;
    for (auto it = leaves_.begin(); it != leaves_.end() && leaves_.size() > target;) {
        bool surface = false;
        for (int i = 0; i < 64 && !surface; ++i)
            surface = float(it->second.weight[i]) >= config_.minMeshWeight && float(it->second.sdf[i]) < 0.999f;
        it = surface ? std::next(it) : leaves_.erase(it);
    }
    if (leaves_.size() <= target) return;
    std::vector<uint32_t> ages;
    ages.reserve(leaves_.size());
    for (const auto& entry : leaves_) ages.push_back(entry.second.lastFrame);
    std::nth_element(ages.begin(), ages.begin() + (leaves_.size() - target), ages.end());
    const uint32_t cutoff = ages[leaves_.size() - target];
    for (auto it = leaves_.begin(); it != leaves_.end() && leaves_.size() > target;)
        it = it->second.lastFrame < cutoff ? leaves_.erase(it) : std::next(it);
}

void TsdfVolume::extractMesh(std::vector<TsdfVertex>& out, size_t maxTriangles,
                             const float* focus) const {
    out.clear();
    std::vector<std::pair<float, const std::pair<const uint64_t, Leaf>*>> order;
    order.reserve(leaves_.size());
    const float leafSize = config_.voxel * kLeaf;
    for (const auto& entry : leaves_) {
        float distance = 0;
        if (focus) {
            const uint64_t k = entry.first;
            const float dx = ((int((k >> 42) & 0x1fffff) - kBias) + 0.5f) * leafSize - focus[0];
            const float dy = ((int((k >> 21) & 0x1fffff) - kBias) + 0.5f) * leafSize - focus[1];
            const float dz = ((int(k & 0x1fffff) - kBias) + 0.5f) * leafSize - focus[2];
            distance = dx * dx + dy * dy + dz * dz;
        }
        order.push_back({distance, &entry});
    }
    if (focus) std::sort(order.begin(), order.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
    const float voxel = config_.voxel, minW = config_.minMeshWeight;
    // Local window: voxels -1..4 around a leaf (6^3), cells -1..3 (5^3).
    constexpr int G = 6, C = 5;
    auto g = [](int x, int y, int z) { return ((z + 1) * G + (y + 1)) * G + (x + 1); };
    auto c = [](int x, int y, int z) { return ((z + 1) * C + (y + 1)) * C + (x + 1); };
    float s[G * G * G], w[G * G * G], l[G * G * G];
    float cv[C * C * C][4];
    bool has[C * C * C];
    for (const auto& ordered : order) {
        const auto& entry = *ordered.second;
        const Leaf& self = entry.second;
        bool surface = false;
        for (int i = 0; i < 64 && !surface; ++i)
            surface = float(self.weight[i]) >= minW && float(self.sdf[i]) < 0.999f;
        if (!surface) continue;
        const uint64_t k = entry.first;
        const int lx = int((k >> 42) & 0x1fffff) - kBias;
        const int ly = int((k >> 21) & 0x1fffff) - kBias;
        const int lz = int(k & 0x1fffff) - kBias;
        const Leaf* neighbor[27];
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    neighbor[(dz + 1) * 9 + (dy + 1) * 3 + dx + 1] =
                        (dx | dy | dz) ? find(lx + dx, ly + dy, lz + dz) : &self;
        for (int z = -1; z < G - 1; ++z)
            for (int y = -1; y < G - 1; ++y)
                for (int x = -1; x < G - 1; ++x) {
                    const int ox = floorDiv(x, kLeaf), oy = floorDiv(y, kLeaf), oz = floorDiv(z, kLeaf);
                    const Leaf* leaf = neighbor[(oz + 1) * 9 + (oy + 1) * 3 + ox + 1];
                    const int i = g(x, y, z);
                    if (!leaf) { s[i] = 1; w[i] = 0; l[i] = 0; continue; }
                    const int v = voxelIndex(x - ox * kLeaf, y - oy * kLeaf, z - oz * kLeaf);
                    s[i] = float(leaf->sdf[v]); w[i] = float(leaf->weight[v]); l[i] = leaf->luma[v];
                }
        // Cell vertices: mean of the edge crossings of each mixed-sign cell.
        static const int corner[8][3]{{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
        static const int edge[12][2]{{0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}};
        for (int z = -1; z < C - 1; ++z)
            for (int y = -1; y < C - 1; ++y)
                for (int x = -1; x < C - 1; ++x) {
                    const int ci = c(x, y, z);
                    has[ci] = false;
                    float cs[8], cl = 0;
                    bool ok = true, neg = false, pos = false;
                    for (int q = 0; q < 8 && ok; ++q) {
                        const int i = g(x + corner[q][0], y + corner[q][1], z + corner[q][2]);
                        ok = w[i] >= minW;
                        cs[q] = s[i]; cl += l[i];
                        (cs[q] < 0 ? neg : pos) = true;
                    }
                    if (!ok || !neg || !pos) continue;
                    float sum[3]{}; int n = 0;
                    for (const auto& e : edge) {
                        const float a = cs[e[0]], b = cs[e[1]];
                        if ((a < 0) == (b < 0)) continue;
                        const float t = a / (a - b);
                        for (int axis = 0; axis < 3; ++axis)
                            sum[axis] += corner[e[0]][axis] + t * (corner[e[1]][axis] - corner[e[0]][axis]);
                        ++n;
                    }
                    cv[ci][0] = ((lx * kLeaf + x) + sum[0] / n + 0.5f) * voxel;
                    cv[ci][1] = ((ly * kLeaf + y) + sum[1] / n + 0.5f) * voxel;
                    cv[ci][2] = ((lz * kLeaf + z) + sum[2] / n + 0.5f) * voxel;
                    cv[ci][3] = cl / 8;
                    has[ci] = true;
                }
        // Quads for sign-changing edges whose first voxel belongs to this leaf.
        for (int z = 0; z < kLeaf; ++z)
            for (int y = 0; y < kLeaf; ++y)
                for (int x = 0; x < kLeaf; ++x) {
                    const int i0 = g(x, y, z);
                    if (w[i0] < minW) continue;
                    for (int axis = 0; axis < 3; ++axis) {
                        const int i1 = g(x + (axis == 0), y + (axis == 1), z + (axis == 2));
                        if (w[i1] < minW || (s[i0] < 0) == (s[i1] < 0)) continue;
                        // The four cells sharing this edge, around the other two axes.
                        int quad[4];
                        bool ok = true;
                        for (int q = 0; q < 4 && ok; ++q) {
                            const int du = (q == 1 || q == 2) ? 0 : -1, dv = (q >= 2) ? 0 : -1;
                            int p[3]{x, y, z};
                            p[(axis + 1) % 3] += du;
                            p[(axis + 2) % 3] += dv;
                            quad[q] = c(p[0], p[1], p[2]);
                            ok = has[quad[q]];
                        }
                        if (!ok) continue;
                        if (s[i0] < 0) std::swap(quad[1], quad[3]);
                        if (out.size() / 3 + 2 > maxTriangles) return;
                        for (const int q : {quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]})
                            out.push_back({cv[q][0], cv[q][1], cv[q][2], cv[q][3]});
                    }
                }
    }
}

}  // namespace FrameSpatial
