// Synthetic check of the Vogon-style TSDF: a wall at z=2.5 m and a sphere,
// seen from a few poses; mesh vertices must lie on the true surface, and
// removing the sphere must carve it away.
#include "tsdf_volume.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace FrameSpatial;
constexpr int N = 616; constexpr float F = 380, CX = 308, CY = 308;
struct Scene { bool sphere; };
const float SC[3]{0.1f, 0.0f, 1.6f}; const float SR = 0.35f; const float WALL = 2.5f;

TsdfPose poseYaw(float yaw, float tx, float ty) {
    TsdfPose p; const float c = std::cos(yaw), s = std::sin(yaw);
    float r[3][3]{{c, 0, s}, {0, 1, 0}, {-s, 0, c}};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) p.rotation[i][j] = r[i][j];
    p.translation[0] = tx; p.translation[1] = ty; p.translation[2] = 0; return p;
}
std::vector<float> render(const TsdfPose& p, Scene scene) {
    std::vector<float> depth(N * N, 0);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const float rc[3]{(x - CX) / F, (y - CY) / F, 1};
        float d[3]; for (int i = 0; i < 3; ++i) d[i] = p.rotation[i][0] * rc[0] + p.rotation[i][1] * rc[1] + p.rotation[i][2] * rc[2];
        const float* o = p.translation;
        float best = 1e9f;
        if (d[2] > 1e-6f) best = (WALL - o[2]) / d[2];  // ray parameter == camera z since rc.z = 1
        if (scene.sphere) {
            float oc[3]{o[0] - SC[0], o[1] - SC[1], o[2] - SC[2]};
            const float a = d[0]*d[0] + d[1]*d[1] + d[2]*d[2], b = 2 * (oc[0]*d[0] + oc[1]*d[1] + oc[2]*d[2]);
            const float c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - SR * SR, disc = b * b - 4 * a * c;
            if (disc >= 0) { const float t = (-b - std::sqrt(disc)) / (2 * a); if (t > 0 && t < best) best = t; }
        }
        if (best < 1e8f) depth[y * N + x] = best;
    }
    return depth;
}
float surfaceDistance(const TsdfVertex& v, bool sphere) {
    float d = std::abs(v.z - WALL);
    if (sphere) d = std::min(d, std::abs(std::sqrt((v.x-SC[0])*(v.x-SC[0]) + (v.y-SC[1])*(v.y-SC[1]) + (v.z-SC[2])*(v.z-SC[2])) - SR));
    return d;
}
int main() {
    TsdfVolume volume;
    const TsdfPose poses[]{poseYaw(0, 0, 0), poseYaw(0.15f, -0.1f, 0), poseYaw(-0.15f, 0.1f, 0.05f), poseYaw(0.05f, 0, -0.05f)};
    double ms = 0; TsdfStats st;
    for (int round = 0; round < 3; ++round) for (const auto& p : poses) {
        const auto depth = render(p, {true});
        const auto t0 = std::chrono::steady_clock::now();
        st = volume.integrate(depth.data(), nullptr, N, N, F, CX, CY, p);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::vector<TsdfVertex> mesh;
    auto t0 = std::chrono::steady_clock::now();
    volume.extractMesh(mesh, 1000000);
    const double meshMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    double sum = 0, worst = 0; size_t nearSphere = 0;
    for (const auto& v : mesh) { const float d = surfaceDistance(v, true); sum += d; worst = std::max(worst, double(d)); nearSphere += std::abs(v.z - WALL) > 0.1f; }
    const double mean = mesh.empty() ? 1 : sum / mesh.size();
    std::printf("integrate %.1f ms/frame (last: touched %zu, updated %zu), leaves %zu, mesh %zu tris in %.1f ms\n",
                ms / 12, st.touchedLeaves, st.updatedVoxels, volume.leafCount(), mesh.size() / 3, meshMs);
    std::printf("vertex-to-surface mean %.4f m, max %.4f m, sphere vertices %zu\n", mean, worst, nearSphere);
    // Carving: the sphere leaves the scene.
    for (int round = 0; round < 12; ++round) for (const auto& p : poses) {
        const auto depth = render(p, {false});
        volume.integrate(depth.data(), nullptr, N, N, F, CX, CY, p);
    }
    volume.extractMesh(mesh, 1000000);
    size_t remaining = 0; double worstWall = 0;
    for (const auto& v : mesh) { if (std::abs(v.z - WALL) > 0.1f) ++remaining; worstWall = std::max(worstWall, double(surfaceDistance(v, false))); }
    std::printf("after removal: %zu tris, sphere vertices %zu, max wall error %.4f m\n", mesh.size() / 3, remaining, worstWall);
    const bool pass = !mesh.empty() && mean < 0.01 && worst < 0.03 && nearSphere > 1000 && remaining == 0;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
