// E07 synthetic components. Thresholds below are the gate, written before the run.
// The default integrate() path is the baseline. Oracle normals are analytic,
// not the E04 estimator.
#include "tsdf_volume.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
using namespace FrameSpatial;

constexpr int N = 200;
constexpr float F = 380, CX = 100, CY = 100;
constexpr float WALL = 2.5f;
constexpr float SC[3]{0.05f, 0.0f, 1.6f};
constexpr float SR = 0.35f;
constexpr float SHEET = 1.5f;
constexpr float THIN = 0.02f;

TsdfPose poseYaw(float yaw, float tx, float ty, float tz) {
    TsdfPose p;
    const float c = std::cos(yaw), s = std::sin(yaw);
    float r[3][3]{{c, 0, s}, {0, 1, 0}, {-s, 0, c}};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) p.rotation[i][j] = r[i][j];
    p.translation[0] = tx;
    p.translation[1] = ty;
    p.translation[2] = tz;
    return p;
}

struct Hit { float z = 0; float n[3]{0, 0, -1}; };

Hit rayHit(const TsdfPose& p, int x, int y, int scene) {
    const float rc[3]{(x - CX) / F, (y - CY) / F, 1};
    float d[3];
    for (int i = 0; i < 3; ++i)
        d[i] = p.rotation[i][0] * rc[0] + p.rotation[i][1] * rc[1] + p.rotation[i][2] * rc[2];
    const float* o = p.translation;
    auto consider = [](float t, float& best) { if (t > 0.05f && t < best) best = t; };
    float best = 1e9f;
    if (scene == 0 || scene == 1 || scene == 4) {
        if (d[2] > 1e-6f) consider((WALL - o[2]) / d[2], best);
    }
    if (scene == 1) {
        float oc[3]{o[0] - SC[0], o[1] - SC[1], o[2] - SC[2]};
        const float a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const float b = 2 * (oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2]);
        const float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - SR * SR;
        const float disc = b * b - 4 * a * c;
        if (disc >= 0) consider((-b - std::sqrt(disc)) / (2 * a), best);
    }
    if (scene == 2) {
        if (d[2] > 1e-6f) consider((2.0f - o[2]) / d[2], best);
        if (d[0] > 1e-6f) consider((0.6f - o[0]) / d[0], best);
    }
    if (scene == 3) {
        const float face = d[0] >= 0 ? SHEET - THIN * 0.5f : SHEET + THIN * 0.5f;
        if (std::fabs(d[0]) > 1e-6f) consider((face - o[0]) / d[0], best);
    }
    if (scene == 5) {
        // z - x = 1.8
        const float den = d[2] - d[0];
        if (std::fabs(den) > 1e-6f) consider((1.8f - (o[2] - o[0])) / den, best);
    }
    Hit hit;
    if (best > 1e8f) return hit;
    hit.z = best;
    if (scene == 5) {
        const float s = std::sqrt(0.5f);
        hit.n[0] = s;
        hit.n[1] = 0;
        hit.n[2] = -s;
    }
    return hit;
}

struct Frame {
    std::vector<float> depth;
    std::vector<float> normal;
};

Frame render(const TsdfPose& p, int scene, bool hole, float noise, uint32_t seed) {
    Frame f;
    f.depth.assign(N * N, 0);
    f.normal.assign(3 * N * N, 0);
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0, noise);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        Hit hit = rayHit(p, x, y, scene);
        if (!(hit.z > 0)) continue;
        if (hole && x >= 80 && x < 120 && y >= 80 && y < 120) continue;
        float z = hit.z;
        if (noise > 0) z = std::max(0.3f, z + gauss(rng));
        f.depth[y * N + x] = z;
        for (int k = 0; k < 3; ++k) f.normal[3 * (y * N + x) + k] = hit.n[k];
    }
    return f;
}

const TsdfPose kPoses[]{poseYaw(0, 0, 0, 0), poseYaw(0.12f, -0.08f, 0, 0),
                        poseYaw(-0.12f, 0.08f, 0, 0.04f), poseYaw(0.04f, 0, -0.04f, 0)};

void fuse(TsdfVolume& v, int scene, int rounds, const TsdfIntegrateOptions& opt, bool hole = false,
          float noise = 0) {
    for (int r = 0; r < rounds; ++r) for (const auto& p : kPoses) {
        Frame f = render(p, scene, hole, noise, 29 + r);
        TsdfIntegrateOptions o = opt;
        o.normal = opt.normalIncidence ? f.normal.data() : nullptr;
        v.integrate(f.depth.data(), nullptr, N, N, F, CX, CY, p, nullptr, o);
    }
}

struct Score {
    size_t tris = 0, sphere = 0, front = 0, sheet = 0;
    double wallMean = 0, wallMax = 0, cornerMean = 0, obliqueMean = 0;
    size_t leaves = 0;
};

Score score(const TsdfVolume& v, int scene) {
    std::vector<TsdfVertex> mesh;
    v.extractMesh(mesh, 1000000);
    Score s;
    s.tris = mesh.size() / 3;
    s.leaves = v.leafCount();
    double wallSum = 0, cornerSum = 0, obliqueSum = 0;
    size_t wallN = 0, cornerN = 0, obliqueN = 0;
    for (const auto& p : mesh) {
        const float sphereD = std::sqrt((p.x - SC[0]) * (p.x - SC[0]) + (p.y - SC[1]) * (p.y - SC[1]) +
                                         (p.z - SC[2]) * (p.z - SC[2]));
        if (std::fabs(sphereD - SR) < 0.04f) {
            ++s.sphere;
            if (p.z < SC[2]) ++s.front;
        }
        if (std::fabs(p.z - WALL) < 0.15f && std::fabs(sphereD - SR) >= 0.04f) {
            wallSum += std::fabs(p.z - WALL);
            s.wallMax = std::max(s.wallMax, double(std::fabs(p.z - WALL)));
            ++wallN;
        }
        if (scene == 2) {
            const double dist = std::min(std::fabs(p.z - 2.0), std::fabs(p.x - 0.6));
            cornerSum += dist;
            ++cornerN;
        }
        if (scene == 3 && std::fabs(p.x - SHEET) < 0.06f) ++s.sheet;
        if (scene == 5) {
            obliqueSum += std::fabs(-p.x + p.z - 1.8) / std::sqrt(2.0);
            ++obliqueN;
        }
    }
    if (wallN) s.wallMean = wallSum / wallN;
    if (cornerN) s.cornerMean = cornerSum / cornerN;
    if (obliqueN) s.obliqueMean = obliqueSum / obliqueN;
    return s;
}

int main() {
    int fails = 0;
    auto expect = [&](bool ok, const std::string& name, const std::string& detail) {
        std::printf("%s %s  %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
        if (!ok) ++fails;
    };

    TsdfIntegrateOptions base;
    TsdfIntegrateOptions robust;
    robust.robust = true;
    TsdfIntegrateOptions cap;
    cap.frameWeightCap = true;
    TsdfIntegrateOptions carve;
    carve.carveAfter = 3;
    TsdfIntegrateOptions incidence;
    incidence.normalIncidence = true;
    TsdfIntegrateOptions combo;
    combo.robust = true;
    combo.frameWeightCap = true;
    combo.carveAfter = 3;

    auto fusedVolume = [](const TsdfIntegrateOptions& opt) {
        TsdfVolume v;
        fuse(v, 1, 2, opt);
        return v;
    };
    TsdfVolume baseVol = fusedVolume(base);
    Score fused = score(baseVol, 1);
    expect(fused.front > 200 && fused.wallMax < 0.03, "baseline fuses the wall and the sphere",
           "front " + std::to_string(fused.front) + " wall max " + std::to_string(fused.wallMax) +
               " leaves " + std::to_string(fused.leaves));

    TsdfVolume one = baseVol;
    {
        Frame empty = render(kPoses[0], 0, false, 0, 1);
        one.integrate(empty.depth.data(), nullptr, N, N, F, CX, CY, kPoses[0]);
    }
    Score oneBase = score(one, 1);
    TsdfVolume carveVol = fusedVolume(carve);
    Score carveFused = score(carveVol, 1);
    {
        Frame empty = render(kPoses[0], 0, false, 0, 1);
        carveVol.integrate(empty.depth.data(), nullptr, N, N, F, CX, CY, kPoses[0], nullptr, carve);
    }
    Score oneCarve = score(carveVol, 1);
    expect(oneCarve.front * 2 >= carveFused.front, "one empty view does not remove the sphere",
           "front before " + std::to_string(carveFused.front) + " after " + std::to_string(oneCarve.front) +
               " baseline after " + std::to_string(oneBase.front));

    fuse(one, 0, 12, base);
    fuse(carveVol, 0, 12, carve);
    Score gone = score(one, 0);
    Score goneCarve = score(carveVol, 0);
    expect(gone.sphere == 0 && gone.wallMax < 0.03, "baseline removes the sphere and keeps the wall",
           "sphere " + std::to_string(gone.sphere) + " wall max " + std::to_string(gone.wallMax));
    expect(goneCarve.sphere == 0 && goneCarve.wallMax < 0.03,
           "repeated empty views still remove the sphere",
           "sphere " + std::to_string(goneCarve.sphere) + " wall max " + std::to_string(goneCarve.wallMax));

    TsdfVolume hole;
    fuse(hole, 0, 2, base);
    Score beforeHole = score(hole, 0);
    fuse(hole, 0, 2, base, true);
    Score afterHole = score(hole, 0);
    expect(beforeHole.tris > 0 && afterHole.tris * 100 >= beforeHole.tris * 98 && afterHole.wallMax < 0.03,
           "missing depth does not carve the wall",
           "tris " + std::to_string(beforeHole.tris) + " -> " + std::to_string(afterHole.tris) +
               " wall max " + std::to_string(afterHole.wallMax));

    // Same two poses for baseline and carveAfter=3. The 0.5 full-carve cut is frozen:
    // if truncation overlap removes the sheet, record it.
    const TsdfPose sheetFrontPose = poseYaw(1.5707963f, 0, 0, 1.5f);
    const TsdfPose sheetBackPose = poseYaw(-1.5707963f, 3.0f, 0, 1.5f);
    TsdfVolume sheetBaseVol, sheetCarveVol;
    for (int r = 0; r < 4; ++r) {
        Frame f = render(sheetFrontPose, 3, false, 0, 3);
        sheetBaseVol.integrate(f.depth.data(), nullptr, N, N, F, CX, CY, sheetFrontPose);
        sheetCarveVol.integrate(f.depth.data(), nullptr, N, N, F, CX, CY, sheetFrontPose, nullptr, carve);
    }
    Score sheetFront = score(sheetBaseVol, 3);
    Score sheetFrontCarve = score(sheetCarveVol, 3);
    for (int r = 0; r < 4; ++r) {
        Frame f = render(sheetBackPose, 3, false, 0, 7);
        sheetBaseVol.integrate(f.depth.data(), nullptr, N, N, F, CX, CY, sheetBackPose);
        sheetCarveVol.integrate(f.depth.data(), nullptr, N, N, F, CX, CY, sheetBackPose, nullptr, carve);
    }
    Score sheetBase = score(sheetBaseVol, 3);
    Score sheetCarve = score(sheetCarveVol, 3);
    expect(sheetFront.sheet > 50, "one view of the 2 cm sheet produces a surface",
           "baseline " + std::to_string(sheetFront.sheet) + " carve " + std::to_string(sheetFrontCarve.sheet));
    std::printf("REPORT sheet after opposite view baseline %zu carve %zu (from %zu / %zu)\n",
                sheetBase.sheet, sheetCarve.sheet, sheetFront.sheet, sheetFrontCarve.sheet);

    TsdfVolume corner;
    fuse(corner, 2, 2, base);
    {
        TsdfPose side = poseYaw(1.5707963f, -0.8f, 0, 1.0f);
        for (int r = 0; r < 4; ++r) {
            Frame f = render(side, 2, false, 0, 4);
            corner.integrate(f.depth.data(), nullptr, N, N, F, CX, CY, side);
        }
    }
    Score cornerScore = score(corner, 2);
    expect(cornerScore.tris > 100 && cornerScore.cornerMean < 0.03, "corner mesh stays within a voxel",
           "tris " + std::to_string(cornerScore.tris) + " mean " + std::to_string(cornerScore.cornerMean));

    auto noiseScore = [](const TsdfIntegrateOptions& opt) {
        TsdfVolume v;
        fuse(v, 0, 2, opt, false, 0.02f);
        return score(v, 0);
    };
    Score noiseBase = noiseScore(base);
    Score noiseRobust = noiseScore(robust);
    std::printf("REPORT robust wall mean %.4f m vs baseline %.4f m\n", noiseRobust.wallMean, noiseBase.wallMean);

    TsdfVolume robustGone;
    fuse(robustGone, 1, 2, robust);
    fuse(robustGone, 0, 12, robust);
    Score robustLeft = score(robustGone, 0);
    std::printf("REPORT robust after removal sphere %zu wall max %.4f\n", robustLeft.sphere, robustLeft.wallMax);

    TsdfVolume capGone;
    fuse(capGone, 1, 2, cap);
    fuse(capGone, 0, 12, cap);
    Score capLeft = score(capGone, 0);
    std::printf("REPORT frame cap after removal sphere %zu wall max %.4f leaves %zu vs baseline leaves %zu\n",
                capLeft.sphere, capLeft.wallMax, capLeft.leaves, gone.leaves);

    TsdfVolume obliqueB, obliqueN;
    fuse(obliqueB, 5, 2, base);
    fuse(obliqueN, 5, 2, incidence);
    Score ob = score(obliqueB, 5), on = score(obliqueN, 5);
    std::printf("REPORT oracle incidence oblique mean %.4f m vs baseline %.4f m\n", on.obliqueMean, ob.obliqueMean);

    TsdfVolume all;
    fuse(all, 1, 2, combo);
    {
        Frame empty = render(kPoses[0], 0, false, 0, 1);
        all.integrate(empty.depth.data(), nullptr, N, N, F, CX, CY, kPoses[0], nullptr, combo);
    }
    Score comboOne = score(all, 1);
    fuse(all, 0, 12, combo);
    Score comboGone = score(all, 0);
    std::printf("REPORT combination one-view front %zu then sphere %zu wall max %.4f\n", comboOne.front,
                comboGone.sphere, comboGone.wallMax);
    std::vector<Frame> cached;
    for (const auto& p : kPoses) cached.push_back(render(p, 1, false, 0, 1));
    auto percentile = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        const double i = q * (v.size() - 1);
        const size_t lo = size_t(i);
        const double f = i - lo;
        return v[lo] * (1.0 - f) + v[std::min(lo + 1, v.size() - 1)] * f;
    };
    auto timeIntegrate = [&](const TsdfIntegrateOptions& opt) {
        TsdfVolume warm;
        for (size_t i = 0; i < cached.size(); ++i)
            warm.integrate(cached[i].depth.data(), nullptr, N, N, F, CX, CY, kPoses[i], nullptr, opt);
        std::vector<double> ms;
        for (int rep = 0; rep < 20; ++rep) {
            TsdfVolume v;
            for (size_t i = 0; i < cached.size(); ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                v.integrate(cached[i].depth.data(), nullptr, N, N, F, CX, CY, kPoses[i], nullptr, opt);
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        return std::pair<double, double>{percentile(ms, 0.50), percentile(ms, 0.95)};
    };
    const auto [baseP50, baseP95] = timeIntegrate(base);
    const auto [carveP50, carveP95] = timeIntegrate(carve);
    std::printf("TIMING integrate-only steady_clock n=80 baseline p50 %.3f p95 %.3f carve p50 %.3f p95 %.3f ms\n",
                baseP50, baseP95, carveP50, carveP95);
    std::printf("MEMORY removal leaves baseline %zu carve %zu\n", gone.leaves, goneCarve.leaves);
    std::printf("%s %d\n", fails ? "E07 FAIL" : "E07 PASS", fails);
    return fails ? 1 : 0;
}
