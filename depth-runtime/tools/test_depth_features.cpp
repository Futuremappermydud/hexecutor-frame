// Synthetic gate for the E04 edge classes. Thresholds are the defaults in
// DepthFeatureConfig, declared before any real frame is scored.
#include "depth_features.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
using namespace FrameSpatial;

static int gFails = 0;

static void expect(bool ok, const std::string& name, const std::string& detail) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << "  " << detail << "\n";
    if (!ok) ++gFails;
}

static float deg(const float* a, const float* b) {
    float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    d = std::max(-1.f, std::min(1.f, d));
    return std::acos(d) * 180.f / std::acos(-1.f);
}

struct Scene {
    int w, h;
    float focal, cx, cy;
    std::vector<float> z;
    std::vector<uint8_t> luma;
    std::vector<uint8_t> prov;
    explicit Scene(int w, int h, float focal)
        : w(w), h(h), focal(focal), cx(w / 2.f), cy(h / 2.f),
          z(w * h, 1.5f), luma(w * h, 40), prov(w * h, uint8_t(DepthProvenance::Measured)) {}
    FeatureInput input() const {
        FeatureInput in;
        in.width = w; in.height = h; in.depth = z.data(); in.luma = luma.data();
        in.provenance = prov.data(); in.focal = focal; in.cx = cx; in.cy = cy;
        return in;
    }
};

static bool geometric(uint8_t c) {
    return c == uint8_t(EdgeClass::Silhouette) || c == uint8_t(EdgeClass::Crease);
}

static void writePpm(const std::string& path, const Scene& s, const std::vector<uint8_t>& edges) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << s.w << " " << s.h << "\n255\n";
    for (size_t i = 0; i < edges.size(); ++i) {
        unsigned char px[3] = {0, 0, 0};
        switch (EdgeClass(edges[i])) {
        case EdgeClass::Interior: px[0] = px[1] = px[2] = 80; break;
        case EdgeClass::Appearance: px[2] = 255; break;
        case EdgeClass::Silhouette: px[0] = 255; break;
        case EdgeClass::Crease: px[1] = 255; break;
        case EdgeClass::Uncertain: px[0] = px[1] = 255; break;
        case EdgeClass::Invalid: break;
        }
        out.write(reinterpret_cast<char*>(px), 3);
    }
}

static void appearance(const char* overlayDir) {
    Scene s(160, 96, 160.f);
    for (int y = 0; y < s.h; ++y) for (int x = 0; x < s.w; ++x) {
        uint8_t luma = 40;
        if (x >= 100 && x <= 120) luma = uint8_t(20 + (x - 100) * 4);  // 4 levels per px, under the gate
        else if (x >= 50 && x <= 51) luma = 255;                         // painted line
        else if (x >= 30) luma = 180;                                    // hard shadow
        s.luma[y * s.w + x] = luma;
    }
    auto combined = extractDepthFeatures(s.input(), EdgeMode::Combined);
    auto lumaOnly = extractDepthFeatures(s.input(), EdgeMode::LumaOnly);
    int app = 0, geo = 0, lumaSil = 0, softGeo = 0, softApp = 0, softN = 0;
    int planeBad = 0, planeN = 0;
    const float wall[3] = {0, 0, -1};
    for (int y = 20; y < 76; ++y) for (int x = 16; x < 144; ++x) {
        const int i = y * s.w + x;
        const float gx = std::fabs(float(s.luma[i + 1]) - float(s.luma[i - 1]));
        const bool lumaEdge = gx >= 24.f;
        if (lumaEdge) {
            ++app;
            geo += geometric(combined.edgeClass[i]);
            lumaSil += lumaOnly.edgeClass[i] == uint8_t(EdgeClass::Silhouette);
        }
        if (x >= 104 && x <= 116) {
            ++softN;
            softGeo += geometric(combined.edgeClass[i]);
            softApp += combined.edgeClass[i] == uint8_t(EdgeClass::Appearance);
        }
        if (x >= 16 && x <= 24 && combined.normalValid[i]) {
            ++planeN;
            planeBad += deg(&combined.normal[3 * i], wall) > 1.f;
        }
    }
    expect(app > 50 && geo == 0, "appearance geometric ridges",
           "luma-edge px " + std::to_string(app) + " geometric " + std::to_string(geo));
    expect(app > 0 && lumaSil * 10 >= app * 9, "luma-only labels appearance as silhouette",
           "silhouette " + std::to_string(lumaSil) + "/" + std::to_string(app));
    expect(softN > 0 && softGeo == 0 && softApp == 0, "soft ramp stays interior",
           "n " + std::to_string(softN) + " geometric " + std::to_string(softGeo)
           + " appearance " + std::to_string(softApp));
    expect(planeN > 100 && planeBad == 0, "flat wall normal",
           "n " + std::to_string(planeN) + " over 1 deg " + std::to_string(planeBad));
    if (overlayDir) writePpm(std::string(overlayDir) + "/appearance.ppm", s, combined.edgeClass);
}

static void silhouette(const char* overlayDir) {
    Scene s(160, 96, 160.f);
    for (int y = 0; y < s.h; ++y) for (int x = 0; x < s.w; ++x)
        s.z[y * s.w + x] = x < 70 ? 1.2f : 2.f;
    auto maps = extractDepthFeatures(s.input(), EdgeMode::Combined);
    auto depthOnly = extractDepthFeatures(s.input(), EdgeMode::DepthOnly);
    int hit = 0, expectN = 0, extra = 0, depthHit = 0;
    double sumX = 0; int nSil = 0;
    for (int y = 30; y < 66; ++y) for (int x = 10; x < 150; ++x) {
        const int i = y * s.w + x;
        const bool ref = x == 69 || x == 70;
        if (ref) {
            ++expectN;
            hit += maps.edgeClass[i] == uint8_t(EdgeClass::Silhouette);
            depthHit += depthOnly.edgeClass[i] == uint8_t(EdgeClass::Silhouette);
        } else extra += maps.edgeClass[i] == uint8_t(EdgeClass::Silhouette);
        if (maps.edgeClass[i] == uint8_t(EdgeClass::Silhouette)) { sumX += x; ++nSil; }
    }
    const double mean = nSil ? sumX / nSil : -1;
    expect(expectN > 0 && hit * 10 >= expectN * 9 && extra == 0, "silhouette recall and location",
           "hit " + std::to_string(hit) + "/" + std::to_string(expectN)
           + " extra " + std::to_string(extra) + " mean x " + std::to_string(mean));
    expect(std::fabs(mean - 69.5) <= 1.0, "silhouette location error",
           "mean x " + std::to_string(mean));
    expect(depthHit * 10 >= expectN * 9, "depth-only finds the step",
           "hit " + std::to_string(depthHit) + "/" + std::to_string(expectN));
    const float wall[3] = {0, 0, -1};
    int bad = 0, n = 0;
    for (int y = 30; y < 66; ++y) for (int x : {64, 74}) {
        const int i = y * s.w + x;
        if (!maps.normalValid[i]) continue;
        ++n; bad += deg(&maps.normal[3 * i], wall) > 5.f;
    }
    expect(n > 20 && bad == 0, "silhouette normals stay on their plane",
           "n " + std::to_string(n) + " bad " + std::to_string(bad));
    if (overlayDir) writePpm(std::string(overlayDir) + "/silhouette.ppm", s, maps.edgeClass);
}

static void crease(const char* overlayDir) {
    Scene s(160, 96, 160.f);
    const float th = 12.f * std::acos(-1.f) / 180.f;
    const float c = std::cos(th), sn = std::sin(th);
    const float leftN[3] = {-sn, 0, -c};
    const float rightN[3] = {sn, 0, -c};
    for (int y = 0; y < s.h; ++y) for (int x = 0; x < s.w; ++x) {
        const float rx = (x + 0.5f - s.cx) / s.focal;
        s.z[y * s.w + x] = x < 80 ? 1.4f * c / (c + sn * rx) : 1.4f * c / (c - sn * rx);
    }
    auto combined = extractDepthFeatures(s.input(), EdgeMode::Combined);
    auto depthOnly = extractDepthFeatures(s.input(), EdgeMode::DepthOnly);
    int creaseHit = 0, creaseN = 0, depthSil = 0;
    int badOne = 0, oneN = 0, centeredWeak = 0, centeredN = 0;
    int faceBad = 0, faceN = 0, faceNotInterior = 0;
    for (int y = 30; y < 66; ++y) {
        for (int x = 70; x <= 90; ++x) {
            const int i = y * s.w + x;
            if (x == 79 || x == 80) {
                ++creaseN;
                creaseHit += combined.edgeClass[i] == uint8_t(EdgeClass::Crease);
                depthSil += depthOnly.edgeClass[i] == uint8_t(EdgeClass::Silhouette)
                    || depthOnly.edgeClass[i] == uint8_t(EdgeClass::Crease);
                if (combined.normalValid[i]) {
                    const float* n = &combined.normal[3 * i];
                    ++oneN;
                    badOne += deg(n, leftN) > 5.f && deg(n, rightN) > 5.f;
                }
                if (combined.centeredValid[i]) {
                    const float* n = &combined.centeredNormal[3 * i];
                    ++centeredN;
                    centeredWeak += std::min(deg(n, leftN), deg(n, rightN)) <= 8.f;
                }
            }
            if (x <= 76 || x >= 83) {
                const float* expectN = x < 80 ? leftN : rightN;
                if (combined.edgeClass[i] != uint8_t(EdgeClass::Interior)) ++faceNotInterior;
                if (combined.normalValid[i]) {
                    ++faceN;
                    faceBad += deg(&combined.normal[3 * i], expectN) > 5.f;
                }
            }
        }
    }
    expect(creaseN > 0 && creaseHit * 10 >= creaseN * 9, "combined labels the roof crease",
           "hit " + std::to_string(creaseHit) + "/" + std::to_string(creaseN));
    expect(depthSil == 0, "depth-only misses the crease",
           "geometric labels " + std::to_string(depthSil));
    expect(badOne == 0, "one-sided normal is not the average",
           "emitted " + std::to_string(oneN) + " farther than 5 deg from both planes " + std::to_string(badOne));
    expect(centeredN > 20 && centeredWeak == 0, "centered normal averages across the crease",
           "n " + std::to_string(centeredN) + " within 8 deg of a plane " + std::to_string(centeredWeak));
    expect(faceN > 50 && faceBad == 0 && faceNotInterior == 0, "crease faces stay interior",
           "normals " + std::to_string(faceN) + " bad " + std::to_string(faceBad)
           + " not interior " + std::to_string(faceNotInterior));
    if (overlayDir) writePpm(std::string(overlayDir) + "/crease.ppm", s, combined.edgeClass);
}

static void openingAndProvenance(const char* overlayDir) {
    Scene hole(160, 96, 160.f);
    for (int y = 40; y < 56; ++y) for (int x = 50; x < 70; ++x) {
        hole.z[y * hole.w + x] = 0.f;
        hole.prov[y * hole.w + x] = uint8_t(DepthProvenance::Missing);
    }
    auto maps = extractDepthFeatures(hole.input(), EdgeMode::Combined);
    int holeBad = 0, holeN = 0, ring = 0, ringN = 0;
    for (int y = 40; y < 56; ++y) for (int x = 50; x < 70; ++x) {
        ++holeN;
        holeBad += maps.edgeClass[y * hole.w + x] != uint8_t(EdgeClass::Invalid);
    }
    for (int y = 42; y < 54; ++y) for (int x : {49, 70}) {
        ++ringN;
        ring += maps.edgeClass[y * hole.w + x] == uint8_t(EdgeClass::Silhouette);
    }
    expect(holeN > 0 && holeBad == 0, "opening interior stays invalid",
           "bad " + std::to_string(holeBad) + "/" + std::to_string(holeN));
    expect(ringN > 0 && ring == ringN, "opening boundary is a silhouette",
           "hit " + std::to_string(ring) + "/" + std::to_string(ringN));

    Scene pred(160, 96, 160.f);
    for (int y = 40; y < 56; ++y) for (int x = 50; x < 70; ++x) {
        pred.z[y * pred.w + x] = 1.9f;
        pred.prov[y * pred.w + x] = uint8_t(DepthProvenance::Predicted);
    }
    auto obs = observeDepth(pred.input(), EdgeMode::Combined);
    int support = 0, predN = 0;
    for (int y = 40; y < 56; ++y) for (int x = 50; x < 70; ++x) {
        ++predN;
        support += measuredSupport(obs, size_t(y * pred.w + x));
    }
    const float wall[3] = {0, 0, -1};
    int rimN = 0, rimBad = 0, rimSil = 0;
    for (int y = 42; y < 54; ++y) for (int x : {46, 73}) {
        const int i = y * pred.w + x;
        if (!obs.normalValid[i]) continue;
        ++rimN;
        rimBad += deg(&obs.normal[3 * i], wall) > 5.f;
        rimSil += obs.edgeClass[i] == uint8_t(EdgeClass::Silhouette);
    }
    expect(predN > 0 && support == 0, "predicted depth is not measured support",
           "support " + std::to_string(support) + "/" + std::to_string(predN));
    expect(rimN > 10 && rimBad == 0 && rimSil == 0, "predicted patch does not tilt the rim",
           "n " + std::to_string(rimN) + " bad " + std::to_string(rimBad)
           + " silhouette " + std::to_string(rimSil));
    expect(obs.nativeWidth == pred.w && obs.nativeHeight == pred.h, "native size is retained", "");
    if (overlayDir) writePpm(std::string(overlayDir) + "/opening.ppm", hole, maps.edgeClass);
}

static void decoy(const char* overlayDir) {
    Scene s(160, 96, 160.f);
    for (int y = 0; y < s.h; ++y) for (int x = 0; x < s.w; ++x) {
        s.z[y * s.w + x] = x < 70 ? 1.2f : 2.f;
        s.luma[y * s.w + x] = x < 75 ? 40 : 200;  // decoy 5 px to the right of the step
    }
    auto combined = extractDepthFeatures(s.input(), EdgeMode::Combined);
    auto lumaOnly = extractDepthFeatures(s.input(), EdgeMode::LumaOnly);
    double sumC = 0, sumL = 0, sumSnap = 0;
    int nC = 0, nL = 0, nSnap = 0;
    for (int y = 30; y < 66; ++y) {
        int lumaX = -1;
        for (int x = 16; x < 144; ++x) {
            const float gx = std::fabs(float(s.luma[y * s.w + x + 1]) - s.luma[y * s.w + x - 1]);
            if (gx >= 24.f) lumaX = x;
        }
        for (int x = 16; x < 144; ++x) {
            const int i = y * s.w + x;
            if (combined.edgeClass[i] == uint8_t(EdgeClass::Silhouette)) { sumC += x; ++nC; }
            if (lumaOnly.edgeClass[i] == uint8_t(EdgeClass::Silhouette)) { sumL += x; ++nL; }
        }
        // Rejected baseline: pull each depth silhouette onto the luma edge.
        for (int x = 16; x < 144; ++x) {
            const int i = y * s.w + x;
            if (combined.edgeClass[i] != uint8_t(EdgeClass::Silhouette)) continue;
            const int snapped = lumaX >= 0 && std::abs(lumaX - x) <= 6 ? lumaX : x;
            sumSnap += snapped; ++nSnap;
        }
    }
    const double meanC = nC ? sumC / nC : -1;
    const double meanL = nL ? sumL / nL : -1;
    const double meanS = nSnap ? sumSnap / nSnap : -1;
    expect(nC > 0 && std::fabs(meanC - 69.5) < 1.0, "combined silhouette stays on the depth step",
           "mean x " + std::to_string(meanC));
    expect(nL > 0 && std::fabs(meanL - 69.5) > 3.0, "luma-only moves the silhouette onto the decoy",
           "mean x " + std::to_string(meanL));
    expect(nSnap > 0 && std::fabs(meanS - 69.5) > 3.0, "luma snap shifts the boundary",
           "mean x " + std::to_string(meanS));
    if (overlayDir) writePpm(std::string(overlayDir) + "/decoy.ppm", s, combined.edgeClass);
}

static void upsample(const char* overlayDir) {
    const int W = 80, H = 48, step = 40;
    Scene native(W, H, 80.f);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x)
        native.z[y * W + x] = x < step ? 1.2f : 2.f;
    auto nativeMaps = extractDepthFeatures(native.input(), EdgeMode::Combined);
    const int yN = H / 2;
    double sum = 0; int n = 0, lo = 1 << 30, hi = -1;
    for (int x = 0; x < W; ++x) if (nativeMaps.edgeClass[yN * W + x] == uint8_t(EdgeClass::Silhouette)) {
        sum += x; ++n; lo = std::min(lo, x); hi = std::max(hi, x);
    }
    const double nativeMean = n ? sum / n : -1;
    const double nativeSpan = n ? double(hi - lo) : -1;

    auto fine = [&](bool linear) {
        Scene up(W * 2, H * 2, native.focal * 2.f);
        up.cx = native.cx * 2.f; up.cy = native.cy * 2.f;
        for (int y = 0; y < up.h; ++y) for (int x = 0; x < up.w; ++x) {
            const float src = (x + 0.5f) / 2.f - 0.5f;
            float z;
            if (!linear) {
                int ix = int(std::lround(src));
                ix = std::max(0, std::min(W - 1, ix));
                z = native.z[(y / 2) * W + ix];
            } else {
                const int i0 = std::max(0, std::min(W - 1, int(std::floor(src))));
                const int i1 = std::max(0, std::min(W - 1, i0 + 1));
                const float t = src - std::floor(src);
                const int yy = std::min(H - 1, y / 2);
                z = (1.f - t) * native.z[yy * W + i0] + t * native.z[yy * W + i1];
            }
            up.z[y * up.w + x] = z;
        }
        return up;
    };
    Scene nearest = fine(false), linear = fine(true);
    auto nearMaps = extractDepthFeatures(nearest.input(), EdgeMode::Combined);
    auto linMaps = extractDepthFeatures(linear.input(), EdgeMode::Combined);
    auto spanOf = [](const Scene& s, const FeatureMaps& m) {
        const int y = s.h / 2;
        int lo = 1 << 30, hi = -1;
        double sum = 0; int n = 0;
        for (int x = 0; x < s.w; ++x) if (m.edgeClass[y * s.w + x] == uint8_t(EdgeClass::Silhouette)) {
            lo = std::min(lo, x); hi = std::max(hi, x); sum += x; ++n;
        }
        // Fine pixel x sits at coarse coordinate (x+0.5)/2 - 0.5.
        const double a = n ? ((lo + 0.5) / 2.0 - 0.5) : -1;
        const double b = n ? ((hi + 0.5) / 2.0 - 0.5) : -1;
        const double mean = n ? (sum / n + 0.5) / 2.0 - 0.5 : -1;
        return std::tuple<double, double, int>(b - a, mean, n);
    };
    auto [nearSpan, nearMean, nearN] = spanOf(nearest, nearMaps);
    auto [linSpan, linMean, linN] = spanOf(linear, linMaps);
    expect(n >= 2 && std::fabs(nativeMean - (step - 0.5)) < 1e-6, "native step location",
           "mean " + std::to_string(nativeMean) + " span " + std::to_string(nativeSpan));
    expect(nearN >= 2 && nearSpan <= nativeSpan + 0.25 && std::fabs(nearMean - (step - 0.5)) < 0.5,
           "nearest expansion keeps the silhouette",
           "span " + std::to_string(nearSpan) + " mean " + std::to_string(nearMean));
    expect(linN >= 3 && linSpan >= nativeSpan + 0.5, "linear upsample thickens the silhouette",
           "span " + std::to_string(linSpan) + " native " + std::to_string(nativeSpan)
           + " mean " + std::to_string(linMean));
    (void)overlayDir;
}

static void multiview() {
    Scene a(160, 96, 160.f), b = a;
    for (int y = 0; y < a.h; ++y) for (int x = 0; x < a.w; ++x) {
        const float wall = x < 40 ? 1.2f : 2.f;
        a.z[y * a.w + x] = wall;
        b.z[y * b.w + x] = wall;
        if (x >= 100 && x < 120 && y >= 30 && y < 60) a.z[y * a.w + x] = 0.8f;
    }
    auto ea = extractDepthFeatures(a.input(), EdgeMode::Combined);
    auto eb = extractDepthFeatures(b.input(), EdgeMode::Combined);
    int boxBefore = 0, stepBefore = 0;
    for (int y = 0; y < a.h; ++y) for (int x = 0; x < a.w; ++x) {
        const int i = y * a.w + x;
        if (ea.edgeClass[i] != uint8_t(EdgeClass::Silhouette)) continue;
        const bool box = x >= 96 && x <= 122 && y >= 26 && y <= 62 && !(x < 40);
        const bool step = x == 39 || x == 40;
        if (box) ++boxBefore;
        if (step) ++stepBefore;
    }
    confirmSilhouettes(ea.edgeClass, eb.edgeClass, a.w, a.h);
    int boxAfter = 0, stepAfter = 0;
    for (int y = 0; y < a.h; ++y) for (int x = 0; x < a.w; ++x) {
        const int i = y * a.w + x;
        if (ea.edgeClass[i] != uint8_t(EdgeClass::Silhouette)) continue;
        const bool box = x >= 96 && x <= 122 && y >= 26 && y <= 62;
        const bool step = x == 39 || x == 40;
        if (box) ++boxAfter;
        if (step) ++stepAfter;
    }
    expect(boxBefore > 20 && boxAfter * 10 <= boxBefore, "multiview drops a one-view false surface",
           "before " + std::to_string(boxBefore) + " after " + std::to_string(boxAfter));
    expect(stepBefore > 20 && stepAfter * 10 >= stepBefore * 9, "multiview keeps the step seen twice",
           "before " + std::to_string(stepBefore) + " after " + std::to_string(stepAfter));
}

static void timing() {
    Scene s(320, 256, 160.f);
    std::vector<double> ms;
    FeatureMaps maps;
    for (int i = 0; i < 21; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        maps = extractDepthFeatures(s.input(), EdgeMode::Combined);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    ms.erase(ms.begin());  // drop the first call
    std::sort(ms.begin(), ms.end());
    const double p50 = ms[ms.size() / 2];
    const double p95 = ms[std::min(ms.size() - 1, size_t(ms.size() * 95 / 100))];
    int interior = 0;
    for (uint8_t c : maps.edgeClass) interior += c == uint8_t(EdgeClass::Interior);
    expect(interior > 70000 && p50 < 5000.0, "cpu reference timing",
           "p50 " + std::to_string(p50) + " ms p95 " + std::to_string(p95)
           + " ms over " + std::to_string(ms.size()) + " runs, steady_clock, not a gpu timestamp");
    std::cout << "TIMING_P50_MS " << p50 << "\nTIMING_P95_MS " << p95 << "\n";
}

static int classifyFile(int argc, char** argv) {
    // --classify W H focal cx cy depth.f32 luma.u8|- prov.u8|- outPrefix [xyz.f32]
    if (argc != 11 && argc != 12) {
        std::cerr << "usage: --classify W H focal cx cy depth.f32 luma.u8|- prov.u8|- outPrefix [xyz.f32]\n";
        return 2;
    }
    const int w = std::atoi(argv[2]), h = std::atoi(argv[3]);
    FeatureInput in;
    in.width = w; in.height = h;
    in.focal = std::atof(argv[4]); in.cx = std::atof(argv[5]); in.cy = std::atof(argv[6]);
    const size_t n = size_t(w) * h;
    std::vector<float> z(n);
    std::vector<uint8_t> luma(n, 0), prov(n, uint8_t(DepthProvenance::Measured));
    std::ifstream dz(argv[7], std::ios::binary);
    dz.read(reinterpret_cast<char*>(z.data()), std::streamsize(n * sizeof(float)));
    if (!dz) { std::cerr << "depth read failed\n"; return 2; }
    in.depth = z.data();
    if (std::strcmp(argv[8], "-") != 0) {
        std::ifstream f(argv[8], std::ios::binary);
        f.read(reinterpret_cast<char*>(luma.data()), std::streamsize(n));
        if (!f) { std::cerr << "luma read failed\n"; return 2; }
        in.luma = luma.data();
    }
    if (std::strcmp(argv[9], "-") != 0) {
        std::ifstream f(argv[9], std::ios::binary);
        f.read(reinterpret_cast<char*>(prov.data()), std::streamsize(n));
        if (!f) { std::cerr << "provenance read failed\n"; return 2; }
        in.provenance = prov.data();
    }
    std::vector<float> xyz;
    if (argc == 12) {
        xyz.resize(3 * n);
        std::ifstream f(argv[11], std::ios::binary);
        f.read(reinterpret_cast<char*>(xyz.data()), std::streamsize(xyz.size() * sizeof(float)));
        if (!f) { std::cerr << "xyz read failed\n"; return 2; }
        in.position = xyz.data();
    }
    const auto t0 = std::chrono::steady_clock::now();
    FeatureMaps maps = extractDepthFeatures(in, EdgeMode::Combined);
    const auto t1 = std::chrono::steady_clock::now();
    const std::string prefix = argv[10];
    std::ofstream edge(prefix + "-edge.u8", std::ios::binary);
    std::ofstream valid(prefix + "-nvalid.u8", std::ios::binary);
    std::ofstream normal(prefix + "-normal.f32", std::ios::binary);
    edge.write(reinterpret_cast<char*>(maps.edgeClass.data()), std::streamsize(n));
    valid.write(reinterpret_cast<char*>(maps.normalValid.data()), std::streamsize(n));
    normal.write(reinterpret_cast<char*>(maps.normal.data()), std::streamsize(n * 3 * sizeof(float)));
    if (!edge || !valid || !normal) { std::cerr << "write failed\n"; return 2; }
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CLASSIFY_MS " << ms << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--classify") == 0) return classifyFile(argc, argv);
    const char* overlay = nullptr;
    if (argc > 2 && std::strcmp(argv[1], "--overlays") == 0) overlay = argv[2];
    appearance(overlay);
    silhouette(overlay);
    crease(overlay);
    openingAndProvenance(overlay);
    decoy(overlay);
    upsample(overlay);
    multiview();
    timing();
    std::cout << (gFails ? "DEPTH FEATURES FAIL " : "DEPTH FEATURES PASS ") << gFails << "\n";
    return gFails ? 1 : 0;
}
