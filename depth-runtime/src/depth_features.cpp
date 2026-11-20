#include "depth_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace FrameSpatial {
namespace {

struct Fit {
    bool ok = false;
    float n[3]{};
    float rms = 0;
    int count = 0;
};

float depthAt(const FeatureInput& in, int i) {
    if (in.position) return in.position[3 * i + 2];
    return in.depth ? in.depth[i] : 0.f;
}

bool samplePoint(const FeatureInput& in, int i, float p[3]) {
    const float z = depthAt(in, i);
    if (!(z > 0.f) || !std::isfinite(z)) return false;
    if (in.provenance && in.provenance[i] != uint8_t(DepthProvenance::Measured)) return false;
    if (in.position) {
        p[0] = in.position[3 * i];
        p[1] = in.position[3 * i + 1];
        p[2] = z;
        return std::isfinite(p[0]) && std::isfinite(p[1]);
    }
    const int x = i % in.width, y = i / in.width;
    const float rx = (x + 0.5f - in.cx) / in.focal;
    const float ry = (y + 0.5f - in.cy) / in.focal;
    p[0] = rx * z;
    p[1] = ry * z;
    p[2] = z;
    return true;
}

bool depthCompatible(float z0, float z1, const DepthFeatureConfig& c) {
    return std::fabs(z0 - z1) <= std::max(c.absDepthTol, c.relDepthTol * z0);
}

void jacobiSmallest(float a[3][3], float out[3]) {
    float ev[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int iter = 0; iter < 16; ++iter) {
        int p = 0, q = 1;
        float maxv = std::fabs(a[0][1]);
        if (std::fabs(a[0][2]) > maxv) { maxv = std::fabs(a[0][2]); p = 0; q = 2; }
        if (std::fabs(a[1][2]) > maxv) { maxv = std::fabs(a[1][2]); p = 1; q = 2; }
        if (maxv < 1e-12f) break;
        const float app = a[p][p], aqq = a[q][q], apq = a[p][q];
        const float tau = (aqq - app) / (2.f * apq);
        const float t = std::copysign(1.f / (std::fabs(tau) + std::sqrt(1.f + tau * tau)), tau);
        const float c = 1.f / std::sqrt(1.f + t * t), s = t * c;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = a[q][p] = 0.f;
        for (int r = 0; r < 3; ++r) if (r != p && r != q) {
            const float arp = a[r][p], arq = a[r][q];
            a[r][p] = a[p][r] = c * arp - s * arq;
            a[r][q] = a[q][r] = s * arp + c * arq;
        }
        for (int r = 0; r < 3; ++r) {
            const float vp = ev[r][p], vq = ev[r][q];
            ev[r][p] = c * vp - s * vq;
            ev[r][q] = s * vp + c * vq;
        }
    }
    int col = 0;
    if (a[1][1] < a[col][col]) col = 1;
    if (a[2][2] < a[col][col]) col = 2;
    out[0] = ev[0][col];
    out[1] = ev[1][col];
    out[2] = ev[2][col];
}

Fit fitQuadrant(const FeatureInput& in, int x, int y, int sx, int sy,
                const DepthFeatureConfig& config) {
    Fit fit;
    const int w = in.width, h = in.height, r = config.fitRadius;
    const int c = y * w + x;
    const float zc = depthAt(in, c);
    float pts[3 * 64];
    int n = 0;
    for (int dy = 1; dy <= r; ++dy) for (int dx = 1; dx <= r; ++dx) {
        const int xx = x + sx * dx, yy = y + sy * dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const int i = yy * w + xx;
        if (!depthCompatible(zc, depthAt(in, i), config)) continue;
        float p[3];
        if (!samplePoint(in, i, p)) continue;
        if (n >= 64) break;
        pts[3 * n] = p[0];
        pts[3 * n + 1] = p[1];
        pts[3 * n + 2] = p[2];
        ++n;
    }
    fit.count = n;
    if (n < config.minQuadrantSamples) return fit;
    float mean[3]{};
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) mean[k] += pts[3 * i + k];
    for (int k = 0; k < 3; ++k) mean[k] /= float(n);
    float cov[3][3]{};
    for (int i = 0; i < n; ++i) {
        float d[3];
        for (int k = 0; k < 3; ++k) d[k] = pts[3 * i + k] - mean[k];
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) cov[a][b] += d[a] * d[b];
    }
    jacobiSmallest(cov, fit.n);
    float len = std::sqrt(fit.n[0] * fit.n[0] + fit.n[1] * fit.n[1] + fit.n[2] * fit.n[2]);
    if (!(len > 1e-8f)) return fit;
    for (int k = 0; k < 3; ++k) fit.n[k] /= len;
    float center[3];
    if (!samplePoint(in, c, center)) return fit;
    if (fit.n[0] * center[0] + fit.n[1] * center[1] + fit.n[2] * center[2] > 0.f)
        for (int k = 0; k < 3; ++k) fit.n[k] = -fit.n[k];
    double acc = 0;
    for (int i = 0; i < n; ++i) {
        float d = 0;
        for (int k = 0; k < 3; ++k) d += fit.n[k] * (pts[3 * i + k] - mean[k]);
        acc += double(d) * double(d);
    }
    fit.rms = float(std::sqrt(acc / n));
    fit.ok = fit.rms <= config.maxFitRms;
    return fit;
}

float angleDeg(const float a[3], const float b[3]) {
    float d = std::clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.f, 1.f);
    return std::acos(d) * 180.f / std::acos(-1.f);
}

float lumaMag(const FeatureInput& in, int x, int y) {
    if (!in.luma) return 0.f;
    const int w = in.width, h = in.height;
    auto at = [&](int xx, int yy) { return float(in.luma[yy * w + xx]); };
    const float gx = (x + 1 < w && x > 0) ? at(x + 1, y) - at(x - 1, y) : 0.f;
    const float gy = (y + 1 < h && y > 0) ? at(x, y + 1) - at(x, y - 1) : 0.f;
    return std::hypot(gx, gy);
}

bool jumped(const FeatureInput& in, int x, int y, const DepthFeatureConfig& c) {
    const int w = in.width, h = in.height, i = y * w + x;
    const float z = depthAt(in, i);
    if (!(z > 0.f)) return false;
    if (in.provenance && in.provenance[i] != uint8_t(DepthProvenance::Measured)) return false;
    const int step[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (auto d : step) {
        const int xx = x + d[0], yy = y + d[1];
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const int j = yy * w + xx;
        const float zn = depthAt(in, j);
        if (!(zn > 0.f)) return true;  // measured pixel against a hole or opening
        if (in.provenance && in.provenance[j] != uint8_t(DepthProvenance::Measured)) continue;
        if (!depthCompatible(z, zn, c)) return true;
    }
    return false;
}

}  // namespace

FeatureMaps extractDepthFeatures(const FeatureInput& in, EdgeMode mode,
                                 const DepthFeatureConfig& config) {
    const int w = in.width, h = in.height, n = w * h;
    FeatureMaps maps;
    maps.edgeClass.assign(n, uint8_t(EdgeClass::Invalid));
    maps.normalValid.assign(n, 0);
    maps.normal.assign(3 * n, 0.f);
    maps.centeredValid.assign(n, 0);
    maps.centeredNormal.assign(3 * n, 0.f);
    if (w < 3 || h < 3 || (!(in.focal > 0.f) && !in.position)) return maps;

    const int sign[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        const int i = y * w + x;
        float center[3];
        const bool measured = samplePoint(in, i, center);
        if (!measured) continue;
        Fit quadrants[4];
        for (int q = 0; q < 4; ++q)
            quadrants[q] = fitQuadrant(in, x, y, sign[q][0], sign[q][1], config);
        // Centered fit uses the whole window. A depth jump drops the far side;
        // a crease does not, because the per-pixel step stays inside the tolerance.
        {
            float pts[3 * 128];
            int count = 0;
            const float zc = depthAt(in, i);
            for (int dy = -config.fitRadius; dy <= config.fitRadius; ++dy)
                for (int dx = -config.fitRadius; dx <= config.fitRadius; ++dx) {
                    if (!dx && !dy) continue;
                    const int xx = x + dx, yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    const int j = yy * w + xx;
                    if (!depthCompatible(zc, depthAt(in, j), config)) continue;
                    float p[3];
                    if (!samplePoint(in, j, p) || count >= 128) continue;
                    pts[3 * count] = p[0];
                    pts[3 * count + 1] = p[1];
                    pts[3 * count + 2] = p[2];
                    ++count;
                }
            if (count >= config.minQuadrantSamples) {
                float mean[3]{};
                for (int k = 0; k < count; ++k) for (int a = 0; a < 3; ++a) mean[a] += pts[3 * k + a];
                for (int a = 0; a < 3; ++a) mean[a] /= float(count);
                float cov[3][3]{};
                for (int k = 0; k < count; ++k) {
                    float d[3];
                    for (int a = 0; a < 3; ++a) d[a] = pts[3 * k + a] - mean[a];
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) cov[a][b] += d[a] * d[b];
                }
                float nn[3];
                jacobiSmallest(cov, nn);
                float len = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                if (len > 1e-8f) {
                    for (int a = 0; a < 3; ++a) nn[a] /= len;
                    if (nn[0] * center[0] + nn[1] * center[1] + nn[2] * center[2] > 0.f)
                        for (int a = 0; a < 3; ++a) nn[a] = -nn[a];
                    maps.centeredValid[i] = 1;
                    for (int a = 0; a < 3; ++a) maps.centeredNormal[3 * i + a] = nn[a];
                }
            }
        }

        const bool jump = jumped(in, x, y, config);
        const bool lumaEdge = lumaMag(in, x, y) >= config.lumaGradient;
        bool crease = false;
        int best = -1;
        for (int q = 0; q < 4; ++q) if (quadrants[q].ok && (best < 0 || quadrants[q].rms < quadrants[best].rms))
            best = q;
        if (best >= 0) {
            // A second quadrant is another plane only when it fits as tightly as
            // the best one. A quadrant that merely reaches across a fold has a
            // larger residual and does not vote.
            for (int q = 0; q < 4; ++q) if (q != best && quadrants[q].ok &&
                                            quadrants[q].rms <= quadrants[best].rms * 2.f + 0.001f &&
                                            angleDeg(quadrants[best].n, quadrants[q].n) > config.creaseDegrees)
                crease = true;
            if (!crease) {
                maps.normalValid[i] = 1;
                for (int a = 0; a < 3; ++a) maps.normal[3 * i + a] = quadrants[best].n[a];
            }
        }

        EdgeClass cls = EdgeClass::Interior;
        if (mode == EdgeMode::LumaOnly) cls = lumaEdge ? EdgeClass::Silhouette : EdgeClass::Interior;
        else if (mode == EdgeMode::DepthOnly) cls = jump ? EdgeClass::Silhouette : EdgeClass::Interior;
        else if (jump) cls = EdgeClass::Silhouette;
        else if (crease) cls = EdgeClass::Crease;
        else if (lumaEdge) cls = EdgeClass::Appearance;
        maps.edgeClass[i] = uint8_t(cls);
    }
    return maps;
}

void confirmSilhouettes(std::vector<uint8_t>& edges, const std::vector<uint8_t>& other,
                        int width, int height) {
    std::vector<uint8_t> kept = edges;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const int i = y * width + x;
        if (edges[i] != uint8_t(EdgeClass::Silhouette)) continue;
        bool hit = false;
        for (int dy = -1; dy <= 1 && !hit; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
            if (other[yy * width + xx] == uint8_t(EdgeClass::Silhouette)) hit = true;
        }
        if (!hit) kept[i] = uint8_t(EdgeClass::Uncertain);
    }
    edges.swap(kept);
}

DepthObservation observeDepth(const FeatureInput& in, EdgeMode mode,
                              const DepthFeatureConfig& config) {
    DepthObservation o;
    o.width = in.width;
    o.height = in.height;
    o.nativeWidth = in.nativeWidth ? in.nativeWidth : in.width;
    o.nativeHeight = in.nativeHeight ? in.nativeHeight : in.height;
    o.focal = in.focal;
    o.cx = in.cx;
    o.cy = in.cy;
    const size_t n = size_t(in.width) * in.height;
    o.depth.resize(n);
    o.provenance.resize(n, uint8_t(DepthProvenance::Missing));
    o.validity.assign(n, 0);
    o.reliability.assign(n, 0.f);
    for (size_t i = 0; i < n; ++i) {
        const float z = depthAt(in, i);
        o.depth[i] = z > 0.f ? z : 0.f;
        uint8_t prov = uint8_t(DepthProvenance::Missing);
        if (in.provenance) prov = in.provenance[i];
        else if (z > 0.f) prov = uint8_t(DepthProvenance::Measured);
        o.provenance[i] = prov;
        if (prov == uint8_t(DepthProvenance::Measured) && z > 0.f) {
            o.validity[i] = 1;
            o.reliability[i] = 1.f;
        }
    }
    if (in.luma) o.luma.assign(in.luma, in.luma + n);
    if (in.position) o.position.assign(in.position, in.position + 3 * n);
    FeatureMaps maps = extractDepthFeatures(in, mode, config);
    o.edgeClass = std::move(maps.edgeClass);
    o.normal = std::move(maps.normal);
    o.normalValid = std::move(maps.normalValid);
    return o;
}

}  // namespace FrameSpatial
