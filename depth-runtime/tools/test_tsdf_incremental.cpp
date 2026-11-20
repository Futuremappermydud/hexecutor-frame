#include "tsdf_volume.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace FrameSpatial;

namespace {
constexpr int N = 96;
constexpr float F = 80, C = 47.5f;

TsdfPose pose(int frame) {
    TsdfPose p;
    const float yaw = 0.02f * (frame % 5 - 2);
    p.rotation[0][0] = std::cos(yaw);
    p.rotation[0][2] = std::sin(yaw);
    p.rotation[2][0] = -std::sin(yaw);
    p.rotation[2][2] = std::cos(yaw);
    p.translation[0] = 0.01f * (frame % 3);
    p.translation[1] = 0.005f * (frame % 2);
    return p;
}

void render(std::vector<float>& depth, std::vector<uint8_t>& luma, std::vector<float>& confidence,
            bool foreground) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            const size_t i = size_t(y) * N + x;
            const float wall = 1.2f + 0.001f * x - 0.001f * y;
            depth[i] = foreground && x > 30 && x < 58 && y > 25 && y < 65 ? 0.65f : wall;
            luma[i] = uint8_t((x * 3 + y * 5) & 255);
            confidence[i] = 1;
        }
    }
}

bool sameMesh(const std::vector<TsdfVertex>& a, const std::vector<TsdfVertex>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(TsdfVertex)) == 0);
}

// Triangle multiset equality: the fast path may reorder triangles when the cap is not hit.
bool sameTriangles(std::vector<TsdfVertex> a, std::vector<TsdfVertex> b) {
    if (a.size() != b.size()) return false;
    auto sorted = [](std::vector<TsdfVertex>& v) {
        std::vector<std::array<TsdfVertex, 3>> t(v.size() / 3);
        for (size_t i = 0; i < t.size(); ++i) std::copy(v.begin() + 3 * i, v.begin() + 3 * i + 3, t[i].begin());
        std::sort(t.begin(), t.end(), [](const auto& x, const auto& y) {
            return std::memcmp(x.data(), y.data(), sizeof(x)) < 0;
        });
        return t;
    };
    const auto ta = sorted(a), tb = sorted(b);
    return a.empty() || std::memcmp(ta.data(), tb.data(), ta.size() * sizeof(ta[0])) == 0;
}

size_t lastRebuilt = 0;  // leaves the incremental call of the latest compare() rebuilt

bool compare(TsdfVolume& volume, size_t cap = 100000, const float* focus = nullptr) {
    std::vector<TsdfVertex> reference, actual, fast;
    volume.extractMesh(reference, cap, focus);
    volume.extractMeshIncremental(actual, cap, focus);
    lastRebuilt = volume.lastMeshRebuiltLeaves();
    volume.extractMeshAs(fast, cap, focus, [](const TsdfVertex& v) { return v; });
    std::vector<TsdfVertex> all;
    volume.extractMesh(all, size_t(1) << 40, focus);
    const bool capped = all.size() / 3 > cap;  // the fast path then keeps extractMesh's order
    const bool fastOk = capped ? sameMesh(reference, fast) : sameTriangles(reference, fast);
    if (sameMesh(reference, actual) && fastOk) return true;
    std::fprintf(stderr, "mesh mismatch: full %zu vertices, incremental %zu, fast %zu\n", reference.size(),
                 actual.size(), fast.size());
    return false;
}
}

int main() {
    bool pass = true;
    TsdfVolume volume;
    std::vector<float> depth(N * N), confidence(N * N);
    std::vector<uint8_t> luma(N * N);
    size_t triangles = 0;
    for (int frame = 0; frame < 10; ++frame) {
        render(depth, luma, confidence, true);
        volume.integrate(depth.data(), luma.data(), N, N, F, C, C, pose(frame), confidence.data());
        pass &= compare(volume);
        triangles = std::max(triangles, [&] {
            std::vector<TsdfVertex> mesh;
            volume.extractMesh(mesh, 100000);
            return mesh.size() / 3;
        }());
    }
    const size_t rebuilt = lastRebuilt;
    std::vector<TsdfVertex> before, after;
    volume.extractMeshIncremental(before, 100000);
    pass &= rebuilt > 0 && volume.lastMeshRebuiltLeaves() == 0;
    volume.extractMeshIncremental(after, 100000);
    pass &= sameMesh(before, after) && volume.lastMeshRebuiltLeaves() == 0;
    for (int frame = 10; frame < 18; ++frame) {
        render(depth, luma, confidence, false);
        volume.integrate(depth.data(), luma.data(), N, N, F, C, C, pose(frame), confidence.data());
        pass &= compare(volume);
    }
    for (int frame = 18; frame < 24; ++frame) {  // several integrations between extractions
        render(depth, luma, confidence, frame % 2);
        volume.integrate(depth.data(), luma.data(), N, N, F, C, C, pose(frame), confidence.data());
        if (frame % 3 == 2) pass &= compare(volume);
    }
    const float focus[3]{0, 0, 0.8f};
    for (const size_t cap : {size_t(0), size_t(1), size_t(2), size_t(17), size_t(1000)})
        pass &= compare(volume, cap, focus);
    volume.clear();
    pass &= compare(volume);
    render(depth, luma, confidence, true);
    for (int frame = 0; frame < 3; ++frame) {
        volume.integrate(depth.data(), luma.data(), N, N, F, C, C, pose(frame), confidence.data());
        pass &= compare(volume);
    }

    TsdfConfig bounded;
    bounded.maxLeaves = 24;
    TsdfVolume evicting(bounded);
    size_t evicted = 0;
    for (int frame = 0; frame < 4; ++frame) {
        render(depth, luma, confidence, frame < 2);
        const auto stats = evicting.integrate(depth.data(), luma.data(), N, N, F, C, C, pose(frame), confidence.data());
        evicted += stats.evictedLeaves;
        pass &= compare(evicting);
    }
    pass &= evicted > 0 && triangles > 100;
    std::printf("incremental parity: triangles %zu, evicted %zu, %s\n", triangles, evicted, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
