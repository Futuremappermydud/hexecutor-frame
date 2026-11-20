#include "tsdf_volume.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iterator>

namespace FrameSpatial {
namespace {
constexpr int kLeaf = 4;   // voxels per leaf edge (Vogon LeafNode<..., 2, 2, 2>)
constexpr int kBias = 1 << 20;

inline int floorDiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
inline int voxelIndex(int x, int y, int z) { return (z * kLeaf + y) * kLeaf + x; }
bool appendLeafMesh(std::vector<TsdfVertex>& out, const std::vector<TsdfVertex>& leaf, size_t maxTriangles) {
    for (size_t first = 0; first < leaf.size(); first += 6) {
        if (out.size() / 3 + 2 > maxTriangles) return false;
        out.insert(out.end(), leaf.begin() + first, leaf.begin() + first + 6);
    }
    return true;
}
}  // namespace

TsdfVolume::TsdfVolume(const TsdfConfig& config) : config_(config) {
    leaves_.reserve(config_.maxLeaves / 2);
}

TsdfVolume::TsdfVolume(const TsdfVolume& other)
    : config_(other.config_), leaves_(other.leaves_), frame_(other.frame_) {
    for (auto& entry : leaves_) addToList(entry.first, &entry.second);
}

TsdfVolume& TsdfVolume::operator=(const TsdfVolume& other) {
    if (this != &other) *this = TsdfVolume(other);
    return *this;
}

size_t TsdfVolume::cachedMeshVertexBytes() const {
    return mesh_.vertices * sizeof(TsdfVertex);
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
                                const float* confidence, const TsdfIntegrateOptions& options) {
    TsdfStats stats;
    ++frame_;
    const float mu = config_.truncation, voxel = config_.voxel;
    const auto& R = pose.rotation;
    const auto& T = pose.translation;

    // 1. Leaves within +-mu of every sampled measurement, along its ray.
    const int stride = std::max(1, config_.allocationStride);
    const float leafSize = voxel * kLeaf;
    const int steps = int(std::ceil(2 * mu / (0.5f * leafSize)));
    // Row bands are sampled in parallel; each band sorts its own keys.
    constexpr int kBands = 16;
    allocParts_.resize(kBands);
    const int nx = (width + stride - 1) / stride;
    std::vector<float> rxs(nx), offs(steps + 1);
    for (int i = 0; i < nx; ++i) rxs[i] = (i * stride - cx) / focal;
    for (int s = 0; s <= steps; ++s) offs[s] = 2 * mu * s / steps;
#pragma omp parallel for schedule(dynamic, 1)
    for (int band = 0; band < kBands; ++band) {
        auto& part = allocParts_[band];
        part.clear();
        std::vector<uint64_t> keys(size_t(steps + 1) * nx);
        std::vector<int> cellX(nx), cellY(nx), cellZ(nx);
        uint64_t recent[4]{~0ull, ~0ull, ~0ull, ~0ull};
        unsigned next = 0;
        const int y0 = int((int64_t(height) * band / kBands + stride - 1) / stride) * stride;
        const int y1 = int((int64_t(height) * (band + 1) / kBands + stride - 1) / stride) * stride;
        for (int y = y0; y < y1 && y < height; y += stride) {
            const float* row = depth + size_t(y) * width;
            const float ry = (y - cy) / focal;
            // Sample the band at half-leaf spacing so no crossed leaf is skipped.
            // The projection is branch-free so it vectorises; ~0 marks a skipped
            // sample (real keys never set bit 63).
            for (int s = 0; s <= steps; ++s) {
                const float off = offs[s];
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const float z = row[size_t(i) * stride];
                    const float zz = z - mu + off;
                    const bool valid = z >= config_.minDepth && z <= config_.maxDepth && zz > 0;
                    const float zc = valid ? zz : 1.0f;
                    const float px = rxs[i] * zc, py = ry * zc, pz = zc;
                    const float wx = R[0][0] * px + R[0][1] * py + R[0][2] * pz + T[0];
                    const float wy = R[1][0] * px + R[1][1] * py + R[1][2] * pz + T[1];
                    const float wz = R[2][0] * px + R[2][1] * py + R[2][2] * pz + T[2];
                    cellX[i] = valid ? int(std::floor(wx / leafSize)) : INT_MIN;
                    cellY[i] = int(std::floor(wy / leafSize));
                    cellZ[i] = int(std::floor(wz / leafSize));
                }
                uint64_t* out = keys.data() + size_t(s) * nx;
                for (int i = 0; i < nx; ++i)
                    out[i] = cellX[i] == INT_MIN ? ~0ull : key(cellX[i], cellY[i], cellZ[i]);
            }
            for (int i = 0; i < nx; ++i) {
                for (int s = 0; s <= steps; ++s) {
                    const uint64_t k = keys[size_t(s) * nx + i];
                    // Neighbouring samples mostly share leaves; drop repeats of the
                    // last few before sorting.
                    if (k != ~0ull && k != recent[0] && k != recent[1] && k != recent[2] && k != recent[3]) {
                        part.push_back(k);
                        recent[next++ & 3] = k;
                    }
                }
            }
        }
        std::sort(part.begin(), part.end());
        part.erase(std::unique(part.begin(), part.end()), part.end());
    }
    auto& touched = touched_;
    touched.clear();
    for (const auto& part : allocParts_) touched.insert(touched.end(), part.begin(), part.end());
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

    // Existing leaves inside the frustum are updated too (Vogon's frustum test),
    // so free space seen through them carves surfaces that are gone.
    const float margin = leafSize * 0.87f;  // half leaf diagonal
    const size_t existing = leafList_.size();
    inView_.resize(existing);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < existing; ++i) {
        const LeafRef& ref = leafList_[i];
        const float wx = ref.c[0] * leafSize - T[0];
        const float wy = ref.c[1] * leafSize - T[1];
        const float wz = ref.c[2] * leafSize - T[2];
        const float pz = R[0][2] * wx + R[1][2] * wy + R[2][2] * wz;
        bool in = false;
        if (!(pz < config_.minDepth || pz > config_.maxDepth + mu)) {
            const float px = R[0][0] * wx + R[1][0] * wy + R[2][0] * wz;
            const float py = R[0][1] * wx + R[1][1] * wy + R[2][1] * wz;
            const float slack = focal * margin / pz;
            const float u = focal * px / pz + cx, v = focal * py / pz + cy;
            in = !(u < -slack || v < -slack || u > width - 1 + slack || v > height - 1 + slack);
        }
        inView_[i] = in;
        if (in) ref.leaf->touchFrame = frame_;
    }

    auto& work = work_;
    work.clear();
    for (size_t i = 0; i < existing; ++i)
        if (inView_[i]) work.push_back({leafList_[i].key, leafList_[i].leaf});
    stats.touchedLeaves = work.size();
    // Sampled keys that are existing leaves already in view are duplicates;
    // the rest are looked up in parallel (read-only) and new ones emplaced in key order.
    found_.resize(touched.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < touched.size(); ++i) {
        const auto it = leaves_.find(touched[i]);
        found_[i] = it == leaves_.end() ? nullptr : &it->second;
    }
    for (size_t i = 0; i < touched.size(); ++i) {
        Leaf* leaf = found_[i];
        if (leaf && leaf->touchFrame == frame_) continue;
        ++stats.touchedLeaves;
        if (!leaf) {
            auto it = leaves_.try_emplace(touched[i]).first;
            leaf = &it->second;
            std::fill(std::begin(leaf->sdf), std::end(leaf->sdf), __fp16(1.0f));
            std::fill(std::begin(leaf->weight), std::end(leaf->weight), __fp16(0.0f));
            std::fill(std::begin(leaf->luma), std::end(leaf->luma), uint8_t(0));
            std::fill(std::begin(leaf->contradict), std::end(leaf->contradict), uint8_t(0));
            ++stats.allocatedLeaves;
            addToList(touched[i], leaf);
        }
        work.push_back({touched[i], leaf});
    }

    // 2. Projective update of every voxel in those leaves.
    // world -> camera: R^T (p - T)
    size_t updated = 0;
    const float minDepth = config_.minDepth, maxDepth = config_.maxDepth;
    const float minMeshWeight = config_.minMeshWeight, maxWeight = config_.maxWeight;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : updated)
    for (size_t i = 0; i < work.size(); ++i) {
        const uint64_t k = work[i].first;
        Leaf& leaf = *work[i].second;
        const int lx = int((k >> 42) & 0x1fffff) - kBias;
        const int ly = int((k >> 21) & 0x1fffff) - kBias;
        const int lz = int(k & 0x1fffff) - kBias;
        float ox[kLeaf], oy[kLeaf], oz[kLeaf];
        for (int v = 0; v < kLeaf; ++v) {
            ox[v] = ((lx * kLeaf + v) + 0.5f) * voxel - T[0];
            oy[v] = ((ly * kLeaf + v) + 0.5f) * voxel - T[1];
            oz[v] = ((lz * kLeaf + v) + 0.5f) * voxel - T[2];
        }
        // Projection of all 64 voxels first (vectorisable); pixel < 0 = rejected.
        // Rounding in float and range-testing before the int conversion matches
        // int(lround(.)) for every value the range test lets through.
        float qx[64], qy[64], qz[64];
        int pix[64];
        bool anyPixel = false;
#pragma omp simd
        for (int n = 0; n < 64; ++n) {
            const float wx = ox[n & 3], wy = oy[(n >> 2) & 3], wz = oz[n >> 4];
            const float px = R[0][0] * wx + R[1][0] * wy + R[2][0] * wz;
            const float py = R[0][1] * wx + R[1][1] * wy + R[2][1] * wz;
            const float pz = R[0][2] * wx + R[1][2] * wy + R[2][2] * wz;
            const float uf = std::round(focal * px / pz + cx);
            const float vf = std::round(focal * py / pz + cy);
            const bool in = !(pz < minDepth) && uf >= 0 && vf >= 0 && uf < float(width) && vf < float(height);
            qx[n] = px; qy[n] = py; qz[n] = pz;
            pix[n] = in ? int(vf) * width + int(uf) : -1;
        }
        for (int n = 0; n < 64; ++n) anyPixel |= pix[n] >= 0;
        bool any = false;
        for (int n = 0; anyPixel && n < 64; ++n) {
            if (pix[n] < 0) continue;
            const float px = qx[n], py = qy[n], pz = qz[n];
            const size_t pixel = size_t(pix[n]);
            const float d = depth[pixel];
            if (!(d >= minDepth && d <= maxDepth)) continue;
            const float sdf = d - pz;
            if (sdf < -mu) continue;  // occluded: leave unknown/behind untouched
            const float s = std::min(1.0f, sdf / mu);
            float cos2 = pz * pz / (px * px + py * py + pz * pz);
            if (options.normalIncidence && options.normal) {
                const float* nrm = options.normal + 3 * pixel;
                const float rl = std::sqrt(px * px + py * py + pz * pz);
                const float c = std::fabs(nrm[0] * px + nrm[1] * py + nrm[2] * pz) / rl;
                cos2 = c * c;
            }
            float w = std::max(2.0f * cos2 - 1.0f, 0.1f) / (d * d);
            if (confidence) w *= confidence[pixel];
            if (options.robust) {
                const float a = std::min(std::fabs(s), 1.0f);
                w *= (1.0f - a) * (1.0f - a);
            }
            if (options.frameWeightCap) w = std::min(w, options.frameWeightCapValue);
            if (!(w > 0.0f)) continue;
            const float W = float(leaf.weight[n]);
            const float stored = float(leaf.sdf[n]);
            // A full free-space sample (s > 0.5) on an established surface is a
            // contradiction. Smaller residuals, including the far side of a thin
            // sheet, update immediately.
            if (s > 0.5f && W >= minMeshWeight && stored < 0.5f) {
                const int seen = std::min(255, int(leaf.contradict[n]) + 1);
                leaf.contradict[n] = uint8_t(seen);
                if (seen < options.carveAfter) continue;
            } else {
                leaf.contradict[n] = 0;
            }
            const float total = W + w;
            leaf.sdf[n] = __fp16((stored * W + s * w) / total);
            if (luma && s < 1.0f)
                leaf.luma[n] = uint8_t(std::lround((leaf.luma[n] * W + luma[pixel] * w) / total));
            leaf.weight[n] = __fp16(std::min(total, maxWeight));
            any = true;
            ++updated;
        }
        if (any) leaf.lastFrame = frame_;
    }
    stats.updatedVoxels = updated;
    if (mesh_.valid) {
        for (const auto& item : work) {
            Leaf& leaf = *item.second;
            if (leaf.lastFrame != frame_ || leaf.mesh.changed) continue;
            leaf.mesh.changed = 1;
            mesh_.changed.push_back(item);
        }
    }
    if (leaves_.size() > config_.maxLeaves) {
        const size_t before = leaves_.size();
        evict();
        stats.evictedLeaves = before - leaves_.size();
    }
    return stats;
}

void TsdfVolume::addToList(uint64_t k, Leaf* leaf) {
    const auto axis = [](uint64_t k, int shift) { return float(int((k >> shift) & 0x1fffff) - kBias) + 0.5f; };
    leafList_.push_back({k, leaf, {axis(k, 42), axis(k, 21), axis(k, 0)}});
}

void TsdfVolume::evict() {
    evictLeaves();
    leafList_.clear();
    for (auto& entry : leaves_) addToList(entry.first, &entry.second);
}

void TsdfVolume::evictLeaves() {
    resetMesh();
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

std::vector<std::pair<float, const TsdfVolume::LeafEntry*>> TsdfVolume::meshOrder(const float* focus) const {
    std::vector<std::pair<float, const LeafEntry*>> order;
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
    return order;
}

void TsdfVolume::extractLeaf(uint64_t k, const Leaf& self, std::vector<TsdfVertex>& out) const {
    out.clear();
    const float voxel = config_.voxel, minW = config_.minMeshWeight;
    constexpr int G = 6, C = 5;
    auto g = [](int x, int y, int z) { return ((z + 1) * G + (y + 1)) * G + (x + 1); };
    auto c = [](int x, int y, int z) { return ((z + 1) * C + (y + 1)) * C + (x + 1); };
    float s[G * G * G], w[G * G * G], l[G * G * G];
    float cv[C * C * C][4];
    bool has[C * C * C];
    bool surface = false;
    for (int i = 0; i < 64 && !surface; ++i)
        surface = float(self.weight[i]) >= minW && float(self.sdf[i]) < 0.999f;
    if (!surface) return;
    const int lx = int((k >> 42) & 0x1fffff) - kBias;
    const int ly = int((k >> 21) & 0x1fffff) - kBias;
    const int lz = int(k & 0x1fffff) - kBias;
    const Leaf* neighbor[27];
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                neighbor[(dz + 1) * 9 + (dy + 1) * 3 + dx + 1] =
                    (dx | dy | dz) ? find(lx + dx, ly + dy, lz + dz) : &self;
    bool negative = false, positive = false;
    // Grid coordinate -1..4 lives in leaf offset -1/0/0/0/0/+1 at voxel 3/0/1/2/3/0.
    static const int gridLeaf[G]{-1, 0, 0, 0, 0, 1}, gridVoxel[G]{3, 0, 1, 2, 3, 0};
    for (int gz = 0; gz < G; ++gz)
        for (int gy = 0; gy < G; ++gy) {
            const Leaf* const* row = neighbor + (gridLeaf[gz] + 1) * 9 + (gridLeaf[gy] + 1) * 3 + 1;
            const int voxelRow = (gridVoxel[gz] * kLeaf + gridVoxel[gy]) * kLeaf;
            for (int gx = 0; gx < G; ++gx) {
                const Leaf* leaf = row[gridLeaf[gx]];
                const int i = (gz * G + gy) * G + gx;
                if (!leaf) { s[i] = 1; w[i] = 0; l[i] = 0; continue; }
                const int v = voxelRow + gridVoxel[gx];
                s[i] = float(leaf->sdf[v]); w[i] = float(leaf->weight[v]); l[i] = leaf->luma[v];
                if (w[i] >= minW) {
                    negative |= s[i] < 0;
                    positive |= s[i] >= 0;
                }
            }
        }
    if (!negative || !positive) return;
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
    for (int z = 0; z < kLeaf; ++z)
        for (int y = 0; y < kLeaf; ++y)
            for (int x = 0; x < kLeaf; ++x) {
                const int i0 = g(x, y, z);
                if (w[i0] < minW) continue;
                for (int axis = 0; axis < 3; ++axis) {
                    const int i1 = g(x + (axis == 0), y + (axis == 1), z + (axis == 2));
                    if (w[i1] < minW || (s[i0] < 0) == (s[i1] < 0)) continue;
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
                    for (const int q : {quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]})
                        out.push_back({cv[q][0], cv[q][1], cv[q][2], cv[q][3]});
                }
            }
}

void TsdfVolume::extractMesh(std::vector<TsdfVertex>& out, size_t maxTriangles,
                             const float* focus) const {
    out.clear();
    if (maxTriangles < 2) return;
    std::vector<TsdfVertex> leafMesh;
    for (const auto& ordered : meshOrder(focus)) {
        extractLeaf(ordered.second->first, ordered.second->second, leafMesh);
        if (!appendLeafMesh(out, leafMesh, maxTriangles)) return;
    }
}

void TsdfVolume::extractMeshIncremental(std::vector<TsdfVertex>& out, size_t maxTriangles,
                                        const float* focus) {
    const MeshPlan plan = planMesh(maxTriangles, focus, true);
    writeMesh(out, plan, [](const TsdfVertex& v) { return v; });
}

// Brings every leaf's cached triangles up to date: a leaf is rebuilt when its own or any
// neighbouring leaf's voxels changed, because the surface-nets grid reaches one voxel
// into the 26 neighbours.
void TsdfVolume::refreshMesh() {
    lastMeshRebuiltLeaves_ = 0;
    auto& dirty = mesh_.dirty;
    dirty.clear();
    if (!mesh_.valid) {
        resetMesh();
        mesh_.valid = true;
        dirty.reserve(leaves_.size());
        for (auto& entry : leaves_) {
            Leaf& leaf = entry.second;
            leaf.mesh.index = kNoMesh; leaf.mesh.changed = 0; leaf.mesh.dirty = 1;
            dirty.push_back({entry.first, &leaf});
        }
    } else {
#pragma omp parallel
        {
            std::vector<std::pair<uint64_t, Leaf*>> mine;
#pragma omp for schedule(static) nowait
            for (size_t i = 0; i < mesh_.changed.size(); ++i) {
                const uint64_t k = mesh_.changed[i].first;
                mesh_.changed[i].second->mesh.changed = 0;
                const int lx = int((k >> 42) & 0x1fffff) - kBias;
                const int ly = int((k >> 21) & 0x1fffff) - kBias;
                const int lz = int(k & 0x1fffff) - kBias;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const uint64_t nk = key(lx + dx, ly + dy, lz + dz);
                            const auto it = leaves_.find(nk);
                            if (it == leaves_.end()) continue;
                            Leaf* leaf = &it->second;
                            if (!__atomic_exchange_n(&leaf->mesh.dirty, uint8_t(1), __ATOMIC_RELAXED))
                                mine.push_back({nk, leaf});
                        }
            }
#pragma omp critical(tsdf_mesh_dirty)
            dirty.insert(dirty.end(), mine.begin(), mine.end());
        }
        mesh_.changed.clear();
    }
    lastMeshRebuiltLeaves_ = dirty.size();

#pragma omp parallel
    {
        std::vector<TsdfVertex> scratch;
#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < dirty.size(); ++i) {
            Leaf& leaf = *dirty[i].second;
            extractLeaf(dirty[i].first, leaf, scratch);
            if (scratch.size() != leaf.mesh.count) {
                leaf.mesh.vertices.reset(scratch.empty() ? nullptr : new TsdfVertex[scratch.size()]);
                leaf.mesh.count = uint32_t(scratch.size());
            }
            if (!scratch.empty()) std::copy(scratch.begin(), scratch.end(), leaf.mesh.vertices.get());
            leaf.mesh.dirty = 0;
        }
    }

    // Publish the results in key order, so the cache order does not depend on thread timing.
    std::vector<std::pair<uint64_t, Leaf*>> added, removed;
    for (const auto& item : dirty) {
        Leaf& leaf = *item.second;
        if (leaf.mesh.index == kNoMesh) {
            if (leaf.mesh.count) added.push_back(item);
        } else if (leaf.mesh.count) {
            MeshRef& ref = mesh_.list[leaf.mesh.index];
            mesh_.vertices = mesh_.vertices + leaf.mesh.count - ref.count;
            ref = {&leaf, leaf.mesh.vertices.get(), leaf.mesh.count};
        } else {
            removed.push_back(item);
        }
    }
    const auto byKey = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::sort(removed.begin(), removed.end(), byKey);
    for (const auto& item : removed) {
        Leaf& leaf = *item.second;
        mesh_.vertices -= mesh_.list[leaf.mesh.index].count;
        mesh_.list[leaf.mesh.index] = mesh_.list.back();
        mesh_.list[leaf.mesh.index].leaf->mesh.index = leaf.mesh.index;
        mesh_.list.pop_back();
        leaf.mesh.index = kNoMesh;
    }
    std::sort(added.begin(), added.end(), byKey);
    for (const auto& item : added) {
        Leaf& leaf = *item.second;
        leaf.mesh.index = uint32_t(mesh_.list.size());
        mesh_.list.push_back({&leaf, leaf.mesh.vertices.get(), leaf.mesh.count});
        mesh_.vertices += leaf.mesh.count;
    }
}

// Chooses the leaves to publish. Under the cap the order is free, so the cache list is
// used as is; otherwise (or when exactOrder) it is extractMesh's focus order and selection.
TsdfVolume::MeshPlan TsdfVolume::planMesh(size_t maxTriangles, const float* focus, bool exactOrder) {
    refreshMesh();
    const MeshRef* refs = mesh_.list.data();
    size_t count = mesh_.list.size(), vertices = mesh_.vertices;
    if (maxTriangles < 2) {
        count = vertices = 0;
    } else if (exactOrder || vertices / 3 > maxTriangles) {
        mesh_.capped.clear();
        size_t triangles = 0;
        for (const auto& ordered : meshOrder(focus)) {
            const Leaf& leaf = ordered.second->second;
            if (!leaf.mesh.count) continue;
            const size_t take = std::min<size_t>(leaf.mesh.count / 6, (maxTriangles - triangles) / 2);
            if (take) mesh_.capped.push_back({nullptr, leaf.mesh.vertices.get(), uint32_t(take * 6)});
            triangles += take * 2;
            if (take * 6 < leaf.mesh.count) break;
        }
        refs = mesh_.capped.data();
        count = mesh_.capped.size();
        vertices = triangles * 3;
    }
    mesh_.offset.resize(count);
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i) { mesh_.offset[i] = offset; offset += refs[i].count; }
    return {refs, count, vertices};
}

}  // namespace FrameSpatial
