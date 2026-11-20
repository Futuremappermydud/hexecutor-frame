#pragma once

// Offline replay kernel for the tag-free IR motion-stereo yaw estimator.
//
// Port of priv-experiments/temporal-selfcal-v103/spline-motion-yaw/spline_motion_yaw.py main()
// (Python reference, run with --lens spline). Geometry is double precision; the arithmetic that
// NumPy keeps in float32 (remapped images, np.gradient, the cumulative-sum box blur, the Harris-style
// corner response) is kept in float32 with the same operation order. The header does no file I/O:
// the caller supplies the problem and a loader for raw 1056x1024 uint8 IR frames.
//
// problem.txt (written by priv-experiments/temporal-selfcal-v107/cpp-motion-yaw/export_problem.py;
// tokens whitespace-separated, reals in shortest round-trip repr; a `ref` line is tab-separated):
//
//   format frame-motion-yaw-v1
//   grid W_HI H_HI K_HI HALF          512 640 <radians per pixel> 5
//   intrinsics left  fx fy cx cy k1 k2 k3 k4 spacing0 spacing1 sU[25] sV[25]   (kb_spline slam_left)
//   intrinsics right <same layout>                                             (kb_spline slam_right)
//   rotation rect0.RL r00..r22         3x3 row-major
//   rotation rect0.RR r00..r22
//   rotation rect1.RR r00..r22
//   baseline rect0.baseline
//   keyframes N
//   inverse M v0 .. v(M-1)             inverse distance hypotheses (1/m)
//   references K
//   ref<TAB>when_ns<TAB>R(9, row-major, = R @ A)<TAB>t(3, = R @ a + t)<TAB>left_path<TAB>right_path
//
// Per-reference raw images are loaded by the caller through RawLoader.

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

#include "ir_kb_spline.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace FrameSpatial {
namespace MotionYaw {

constexpr int kWidth = 512;          // W_HI
constexpr int kHeight = 640;         // H_HI
constexpr int kHalf = 5;             // HALF
constexpr int kPatch = 121;          // 11 x 11 window
constexpr int kRawWidth = 1056;
constexpr int kRawHeight = 1024;
constexpr int kMargin = kHalf + 6;   // corner mask margin
constexpr int kCell = 24;            // corner grid cell
constexpr std::size_t kCornerCount = 140;
constexpr double kCornerThreshold = 4.0;
constexpr double kPartnerWindowNs = 2.6e9;
constexpr double kCenterU = (kWidth - 1) / 2.0;   // 255.5
constexpr double kCenterV = (kHeight - 1) / 2.0;  // 319.5
constexpr std::size_t kLeftCacheLimit = 40;
constexpr double kPi = 3.141592653589793;   // np.pi

using Point2 = std::array<double, 2>;

struct Reference {
    std::int64_t when = 0;             // left_monotonic_ns - lead, ns
    std::array<double, 9> R{};         // R @ A, row-major
    std::array<double, 3> t{};         // R @ a + t
};

struct Problem {
    double kHi = 0;                    // K_HI, radians per pixel
    int keyframes = 150;
    IrKBSplineIntrinsics leftIntr{}, rightIntr{};
    std::array<double, 9> rect0RL{}, rect0RR{}, rect1RR{};
    double baseline = 0;
    std::vector<double> inverse;
    std::vector<Reference> references;
};

struct Sample {
    int i = 0;
    double tS = 0, u = 0, v = 0, distance = 0, expected = 0, d0 = 0, d1 = 0, s0 = 0, sweepScore = 0;
};

struct Summary {
    bool valid = false;                // false when there are no samples
    std::size_t samples = 0, keyframes = 0;
    double perDegree = 0, yaw = 0;
};

struct KeyframeTrace {
    int i = 0;
    std::vector<std::size_t> partners;
    std::vector<Point2> corners;
};

struct RunResult {
    std::vector<Sample> samples;
    std::vector<KeyframeTrace> trace;
    Summary summary;
};

// Returns the raw uint8 frame (kRawWidth * kRawHeight bytes) for reference `index`.
using RawLoader = std::function<std::vector<std::uint8_t>(std::size_t index, bool right)>;

namespace detail {

// Map of 512x640 grid pixels to raw-image coordinates (x: column, y: row), row-major.
struct Map {
    std::vector<double> x, y;
};

struct Remapped {
    std::vector<float> image;          // float32, as REF casts the remap result
    std::vector<std::uint8_t> ok;      // raw-bounds validity from the map
};

// Centered 11x11 template, as REF's ncc() centers both windows.
struct Template {
    std::array<double, kPatch> values{};
    std::array<bool, kPatch> ok{};
    std::array<double, kPatch> centered{};
    double ss = 0;
};

struct Partner {
    const std::vector<float>* image = nullptr;
    std::array<double, 9> R{};
    std::array<double, 3> t{};
};

inline double offsetU(int k) { return double(k % 11 - kHalf); }   // OFFSET_U after ravel, row-major
inline double offsetV(int k) { return double(k / 11 - kHalf); }   // OFFSET_V after ravel, row-major

// np.add.reduce on a contiguous row: first element, then pairwise sum of the rest.
inline double pairwiseSum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
}

inline double reduceSum(const double* a, std::size_t n) {
    if (n == 0) return 0.;
    return a[0] + pairwiseSum(a + 1, n - 1);
}

// np.nan_to_num with nan=-1.
inline double nanToNum(double x) {
    if (std::isnan(x)) return -1.0;
    if (std::isinf(x)) return x > 0 ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
    return x;
}

// Bilinear sample of a 512x640 float32 image, as REF sample(). Sets ok like REF.
inline double sampleImage(const std::vector<float>& img, double x, double y, bool& ok) {
    const long long x0 = static_cast<long long>(std::floor(x));
    const long long y0 = static_cast<long long>(std::floor(y));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    ok = x0 >= 0 && y0 >= 0 && x0 < kWidth - 1 && y0 < kHeight - 1;
    const long long cx = std::min(std::max(x0, 0LL), static_cast<long long>(kWidth - 2));
    const long long cy = std::min(std::max(y0, 0LL), static_cast<long long>(kHeight - 2));
    const std::size_t i00 = static_cast<std::size_t>(cy * kWidth + cx);
    const double a = img[i00];
    const double b = img[i00 + 1];
    const double c = img[i00 + kWidth];
    const double d = img[i00 + kWidth + 1];
    const double t1 = a * (1 - fx) * (1 - fy);
    const double t2 = b * fx * (1 - fy);
    const double t3 = c * (1 - fx) * fy;
    const double t4 = d * fx * fy;
    return t1 + t2 + t3 + t4;
}

// Lens map: for each 512x640 grid pixel, rays (tan th, sin ph, cos ph) rotated by R and projected
// through the spline lens (same guards as SplineLens.project).
inline Map buildLensMap(const IrKBSplineProjector& lens, const IrKBSplineIntrinsics& intr,
                        const std::array<double, 9>& R, double K) {
    double k1 = double(float(intr.k[0])), k2 = double(float(intr.k[1]));
    double k3 = double(float(intr.k[2])), k4 = double(float(intr.k[3]));
    Map map;
    map.x.resize(std::size_t(kWidth) * kHeight);
    map.y.resize(std::size_t(kWidth) * kHeight);
    for (int v = 0; v < kHeight; ++v) {
        for (int u = 0; u < kWidth; ++u) {
            const double theta_a = (u - kCenterU) * K;
            const double phi_a = (v - kCenterV) * K;
            const double r0 = std::tan(theta_a), r1 = std::sin(phi_a), r2 = std::cos(phi_a);
            const double p0 = R[0] * r0 + R[1] * r1 + R[2] * r2;
            const double p1 = R[3] * r0 + R[4] * r1 + R[5] * r2;
            const double p2 = R[6] * r0 + R[7] * r1 + R[8] * r2;
            const double theta = std::atan2(std::hypot(p0, p1), p2);
            const double t2 = theta * theta;
            const double derivative = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)));
            if (theta >= 1.5 || derivative <= 0)
                throw std::runtime_error("spline projection outside monotonic forward domain");
            const auto uv = lens.project(p0, p1, p2);
            const std::size_t idx = std::size_t(v) * kWidth + std::size_t(u);
            map.x[idx] = uv[0];
            map.y[idx] = uv[1];
        }
    }
    return map;
}

// REF load(): bilinear remap of a raw 1056x1024 frame, cast to float32.
inline Remapped remapRaw(const std::vector<std::uint8_t>& raw, const Map& map) {
    if (raw.size() != std::size_t(kRawWidth) * kRawHeight)
        throw std::runtime_error("raw IR frame has the wrong size");
    Remapped out;
    out.image.resize(std::size_t(kWidth) * kHeight);
    out.ok.resize(std::size_t(kWidth) * kHeight);
    for (std::size_t idx = 0; idx < out.image.size(); ++idx) {
        const double x = map.x[idx], y = map.y[idx];
        const long long x0 = static_cast<long long>(std::floor(x));
        const long long y0 = static_cast<long long>(std::floor(y));
        const double fx = x - static_cast<double>(x0);
        const double fy = y - static_cast<double>(y0);
        out.ok[idx] = (x0 >= 0 && y0 >= 0 && x0 < kRawWidth - 1 && y0 < kRawHeight - 1) ? 1 : 0;
        const long long cx = std::min(std::max(x0, 0LL), static_cast<long long>(kRawWidth - 2));
        const long long cy = std::min(std::max(y0, 0LL), static_cast<long long>(kRawHeight - 2));
        const std::size_t i00 = std::size_t(cy) * kRawWidth + std::size_t(cx);
        const double a = raw[i00];
        const double b = raw[i00 + 1];
        const double c = raw[i00 + kRawWidth];
        const double d = raw[i00 + kRawWidth + 1];
        const double t1 = a * (1 - fx) * (1 - fy);
        const double t2 = b * fx * (1 - fy);
        const double t3 = c * (1 - fx) * fy;
        const double t4 = d * fx * fy;
        out.image[idx] = static_cast<float>(t1 + t2 + t3 + t4);
    }
    return out;
}

inline void rayOf(double u, double v, double K, double out[3]) {
    const double theta = (u - kCenterU) * K;
    const double phi = (v - kCenterV) * K;
    out[0] = std::tan(theta);
    out[1] = std::sin(phi);
    out[2] = std::cos(phi);
}

inline Template makeTemplate(const std::vector<float>& reference, const Point2& p, double K) {
    (void)K;
    Template tpl;
    for (int k = 0; k < kPatch; ++k) {
        bool ok = false;
        tpl.values[k] = sampleImage(reference, p[0] + offsetU(k), p[1] + offsetV(k), ok);
        tpl.ok[k] = ok;
    }
    const double mean = reduceSum(tpl.values.data(), kPatch) / kPatch;
    std::array<double, kPatch> sq{};
    for (int k = 0; k < kPatch; ++k) tpl.centered[k] = tpl.values[k] - mean;
    for (int k = 0; k < kPatch; ++k) sq[k] = tpl.centered[k] * tpl.centered[k];
    tpl.ss = reduceSum(sq.data(), kPatch);
    return tpl;
}

// REF ncc(template, values, valid) along the patch axis.
inline double warpedNcc(const Template& tpl, const double* values, const bool* valid) {
    bool all = true;
    for (int k = 0; k < kPatch; ++k) all = all && valid[k];
    if (!all) return -1.0;
    const double mean = reduceSum(values, kPatch) / kPatch;
    std::array<double, kPatch> bc{}, prod{};
    for (int k = 0; k < kPatch; ++k) bc[k] = values[k] - mean;
    for (int k = 0; k < kPatch; ++k) prod[k] = tpl.centered[k] * bc[k];
    const double num = reduceSum(prod.data(), kPatch);
    for (int k = 0; k < kPatch; ++k) prod[k] = bc[k] * bc[k];
    const double bss = reduceSum(prod.data(), kPatch);
    const double score = num / std::sqrt(tpl.ss * bss);
    return nanToNum(score);
}

// Warped NCC of one template against one partner image at one inverse-distance hypothesis.
inline double partnerScore(const Template& tpl, const Partner& partner, const Point2& p, double distance,
                           double K) {
    std::array<double, kPatch> values{};
    std::array<bool, kPatch> valid{};
    for (int k = 0; k < kPatch; ++k) {
        double ray[3];
        rayOf(p[0] + offsetU(k), p[1] + offsetV(k), K, ray);
        const double X0 = ray[0] * distance, X1 = ray[1] * distance, X2 = ray[2] * distance;
        const auto& R = partner.R;
        const double Y0 = R[0] * X0 + R[1] * X1 + R[2] * X2 + partner.t[0];
        const double Y1 = R[3] * X0 + R[4] * X1 + R[5] * X2 + partner.t[1];
        const double Y2 = R[6] * X0 + R[7] * X1 + R[8] * X2 + partner.t[2];
        const double rho = std::hypot(Y1, Y2);
        const double theta = std::atan2(Y0, rho);
        const double phi = std::atan2(Y1, Y2);
        const double x = theta / K + kCenterU;
        const double y = phi / K + kCenterV;
        bool ok = false;
        values[k] = sampleImage(*partner.image, x, y, ok);
        valid[k] = ok && (Y2 > 0.05) && tpl.ok[k];
    }
    return warpedNcc(tpl, values.data(), valid.data());
}

// REF sweep(): mean and min over partners of warped NCC, for each point and hypothesis.
struct SweepResult {
    std::vector<double> mean, low;     // [point * hypotheses + h]
};

inline SweepResult sweep(const std::vector<float>& reference, const std::vector<Partner>& partners,
                         const std::vector<Point2>& points, const std::vector<double>& inverse, double K) {
    const std::size_t H = inverse.size();
    SweepResult out;
    out.mean.assign(points.size() * H, 0.);
    out.low.assign(points.size() * H, 0.);
    std::vector<double> distance(H);
    for (std::size_t h = 0; h < H; ++h) distance[h] = 1.0 / inverse[h];
    for (std::size_t p = 0; p < points.size(); ++p) {
        const Template tpl = makeTemplate(reference, points[p], K);
        std::vector<double> sum(H, 0.), low(H, 0.);
        for (std::size_t j = 0; j < partners.size(); ++j) {
            for (std::size_t h = 0; h < H; ++h) {
                const double score = partnerScore(tpl, partners[j], points[p], distance[h], K);
                if (j == 0) {
                    sum[h] = score;
                    low[h] = score;
                } else {
                    sum[h] += score;
                    low[h] = std::min(low[h], score);
                }
            }
        }
        for (std::size_t h = 0; h < H; ++h) {
            out.mean[p * H + h] = sum[h] / double(partners.size());
            out.low[p * H + h] = low[h];
        }
    }
    return out;
}

inline double parabola(const double* values, std::size_t n, std::size_t index) {
    if (index == 0 || index + 1 >= n) return 0.0;
    const double denominator = values[index - 1] - 2.0 * values[index] + values[index + 1];
    return std::abs(denominator) > 1e-9 ? 0.5 * (values[index - 1] - values[index + 1]) / denominator : 0.0;
}

// REF corners(): float32 gradient, cumulative-sum box blur, Harris-style response, per-cell argmax.
inline std::vector<Point2> corners(const std::vector<float>& image, const std::vector<std::uint8_t>& ok) {
    const std::size_t W = kWidth, H = kHeight;
    std::vector<float> gy(W * H), gx(W * H);
    // np.gradient, uniform spacing 1: interior (f[i+1]-f[i-1])/2, edges (f[1]-f[0])/1.
    for (std::size_t y = 0; y < H; ++y) {
        for (std::size_t x = 0; x < W; ++x) {
            const std::size_t idx = y * W + x;
            if (y == 0) gy[idx] = (image[idx + W] - image[idx]) / 1.0f;
            else if (y == H - 1) gy[idx] = (image[idx] - image[idx - W]) / 1.0f;
            else gy[idx] = (image[idx + W] - image[idx - W]) / 2.0f;
            if (x == 0) gx[idx] = (image[idx + 1] - image[idx]) / 1.0f;
            else if (x == W - 1) gx[idx] = (image[idx] - image[idx - 1]) / 1.0f;
            else gx[idx] = (image[idx + 1] - image[idx - 1]) / 2.0f;
        }
    }
    std::vector<float> gxx(W * H), gxy(W * H), gyy(W * H);
    for (std::size_t i = 0; i < W * H; ++i) {
        gxx[i] = gx[i] * gx[i];
        gxy[i] = gx[i] * gy[i];
        gyy[i] = gy[i] * gy[i];
    }
    // blur(values): pad ((4,3),(4,3)) with zeros, cumsum axis 0 then axis 1 in float32,
    // then (c[7:,7:] - c[:-7,7:] - c[7:,:-7] + c[:-7,:-7]) / 49.
    auto blur = [&](const std::vector<float>& values) {
        const std::size_t PW = W + 7, PH = H + 7;
        std::vector<float> c(PW * PH, 0.f);
        for (std::size_t y = 0; y < H; ++y)
            for (std::size_t x = 0; x < W; ++x) c[(y + 4) * PW + (x + 4)] = values[y * W + x];
        for (std::size_t y = 1; y < PH; ++y)
            for (std::size_t x = 0; x < PW; ++x) c[y * PW + x] = c[(y - 1) * PW + x] + c[y * PW + x];
        for (std::size_t y = 0; y < PH; ++y)
            for (std::size_t x = 1; x < PW; ++x) c[y * PW + x] = c[y * PW + x - 1] + c[y * PW + x];
        std::vector<float> out(W * H);
        for (std::size_t y = 0; y < H; ++y) {
            for (std::size_t x = 0; x < W; ++x) {
                const float A = c[(y + 7) * PW + (x + 7)];
                const float B = c[y * PW + (x + 7)];
                const float C = c[(y + 7) * PW + x];
                const float D = c[y * PW + x];
                const float s = ((A - B) - C) + D;
                out[y * W + x] = s / 49.0f;
            }
        }
        return out;
    };
    const std::vector<float> a = blur(gxx), b = blur(gxy), d = blur(gyy);
    std::vector<float> response(W * H);
    for (std::size_t i = 0; i < W * H; ++i) {
        const float half_sum = (a[i] + d[i]) / 2.0f;
        const float half_diff = (a[i] - d[i]) / 2.0f;
        const float sq = half_diff * half_diff;
        const float bb = b[i] * b[i];
        const float root = std::sqrt(sq + bb);
        response[i] = half_sum - root;
    }
    // Mask: rows and columns [margin, size - margin) and ok.
    for (std::size_t y = 0; y < H; ++y)
        for (std::size_t x = 0; x < W; ++x) {
            const bool inside = y >= std::size_t(kMargin) && y < H - kMargin && x >= std::size_t(kMargin) &&
                                x < W - kMargin;
            if (!(inside && ok[y * W + x])) response[y * W + x] = 0.f;
        }
    struct Found {
        float value;
        int x, y;
    };
    std::vector<Found> found;
    for (int y0 = 0; y0 < int(H); y0 += kCell) {
        for (int x0 = 0; x0 < int(W); x0 += kCell) {
            const int bh = std::min(kCell, int(H) - y0);
            const int bw = std::min(kCell, int(W) - x0);
            int best = 0;
            float bestValue = response[std::size_t(y0) * W + x0];
            for (int i = 1; i < bw * bh; ++i) {
                const float value = response[std::size_t(y0 + i / bw) * W + std::size_t(x0 + i % bw)];
                if (value > bestValue) {
                    bestValue = value;
                    best = i;
                }
            }
            if (double(bestValue) > kCornerThreshold)
                found.push_back({bestValue, x0 + best % bw, y0 + best / bw});
        }
    }
    std::sort(found.begin(), found.end(), [](const Found& l, const Found& r) {
        return std::tie(l.value, l.x, l.y) > std::tie(r.value, r.x, r.y);
    });
    std::vector<Point2> out;
    for (std::size_t k = 0; k < found.size() && k < kCornerCount; ++k)
        out.push_back({double(found[k].x), double(found[k].y)});
    return out;
}

inline double medianOf(std::vector<double> values) {
    const std::size_t n = values.size();
    const std::size_t mid = n / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    if (n % 2 == 1) return values[mid];
    const double upper = values[mid];
    const double lower = *std::max_element(values.begin(), values.begin() + mid);
    return (lower + upper) / 2.0;
}

}  // namespace detail

// Median-based summary, as in REF main(): per_degree = median((d1 - d0) / -0.3),
// yaw = median((expected - d0) / per_degree).
inline Summary summarize(const std::vector<Sample>& samples) {
    Summary summary;
    if (samples.empty()) return summary;
    std::vector<double> perDegree;
    perDegree.reserve(samples.size());
    for (const auto& s : samples) perDegree.push_back((s.d1 - s.d0) / -0.3);
    summary.perDegree = detail::medianOf(perDegree);
    std::vector<double> dataYaw;
    dataYaw.reserve(samples.size());
    std::vector<int> frames;
    for (const auto& s : samples) {
        dataYaw.push_back((s.expected - s.d0) / summary.perDegree);
        frames.push_back(s.i);
    }
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    summary.keyframes = frames.size();
    summary.samples = samples.size();
    summary.yaw = detail::medianOf(dataYaw);
    summary.valid = true;
    return summary;
}

// REF main() after the reference list is built: keyframe loop, partner choice, corners, sweep,
// fine refinement, stereo search, acceptance, and the summary.
inline RunResult runMotionYaw(const Problem& prob, const RawLoader& loadRaw) {
    using namespace detail;
    RunResult result;
    if (prob.references.empty()) return result;
    if (prob.keyframes <= 0) throw std::invalid_argument("keyframes must be positive");
    const std::size_t hypotheses = prob.inverse.size();
    if (hypotheses < 3) throw std::invalid_argument("at least three inverse hypotheses required");
    const double K = prob.kHi;
    const IrKBSplineProjector leftLens(prob.leftIntr), rightLens(prob.rightIntr);
    const Map mapLeft = buildLensMap(leftLens, prob.leftIntr, prob.rect0RL, K);
    const Map mapRight0 = buildLensMap(rightLens, prob.rightIntr, prob.rect0RR, K);
    const Map mapRight1 = buildLensMap(rightLens, prob.rightIntr, prob.rect1RR, K);

    const auto& refs = prob.references;
    const std::size_t N = refs.size();
    const std::size_t step = std::max<std::size_t>(1, N / std::size_t(prob.keyframes));
    const std::int64_t start = refs[0].when;

    std::map<std::size_t, std::shared_ptr<Remapped>> leftCache;
    std::deque<std::size_t> leftOrder;
    auto left = [&](std::size_t i) {
        auto found = leftCache.find(i);
        if (found != leftCache.end()) return found->second;
        if (leftCache.size() > kLeftCacheLimit) {
            leftCache.erase(leftOrder.front());
            leftOrder.pop_front();
        }
        auto image = std::make_shared<Remapped>(remapRaw(loadRaw(i, false), mapLeft));
        leftCache[i] = image;
        leftOrder.push_back(i);
        return image;
    };

    for (std::size_t i = 0; i < N; i += step) {
        const Reference& Ri = refs[i];
        // Partner candidates: same time window, baseline in [0.06, 0.45] m, angle < 18 deg.
        std::vector<std::pair<double, std::size_t>> candidates;
        for (std::size_t j = 0; j < N; ++j) {
            if (j == i) continue;
            const double dt = double(refs[j].when - Ri.when);
            if (std::abs(dt) > kPartnerWindowNs) continue;
            const Reference& Rj = refs[j];
            const double dx = Rj.t[0] - Ri.t[0], dy = Rj.t[1] - Ri.t[1], dz = Rj.t[2] - Ri.t[2];
            const double baseline = std::sqrt(dx * dx + dy * dy + dz * dz);
            // trace(Rj^T Ri)
            double trace = 0;
            for (int k = 0; k < 3; ++k) {
                double acc = 0;
                for (int m = 0; m < 3; ++m) acc += Rj.R[m * 3 + k] * Ri.R[m * 3 + k];
                trace += acc;
            }
            const double cosine = std::min(std::max((trace - 1) / 2, -1.0), 1.0);
            const double angle = std::acos(cosine) * (180.0 / kPi);
            if (0.06 <= baseline && baseline <= 0.45 && angle < 18) candidates.push_back({baseline, j});
        }
        std::sort(candidates.begin(), candidates.end(), std::greater<>());
        const std::size_t stride = std::max<std::size_t>(1, candidates.size() / 3);
        std::vector<std::size_t> chosen;
        for (std::size_t k = 0; k < candidates.size() && chosen.size() < 3; k += stride)
            chosen.push_back(candidates[k].second);
        if (chosen.size() < 2) continue;

        const auto imageI = left(i);
        const std::vector<Point2> points = corners(imageI->image, imageI->ok);
        if (points.empty()) continue;

        KeyframeTrace trace;
        trace.i = int(i);
        trace.partners = chosen;
        trace.corners = points;
        result.trace.push_back(trace);

        std::vector<std::shared_ptr<Remapped>> partnerImages;
        std::vector<Partner> partners;
        for (std::size_t j : chosen) {
            const Reference& Rj = refs[j];
            partnerImages.push_back(left(j));
            Partner partner;
            partner.image = &partnerImages.back()->image;
            // Rj.T @ Ri and Rj.T @ (ti - tj)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double acc = 0;
                    for (int m = 0; m < 3; ++m) acc += Rj.R[m * 3 + r] * Ri.R[m * 3 + c];
                    partner.R[r * 3 + c] = acc;
                }
            for (int r = 0; r < 3; ++r) {
                double acc = 0;
                for (int m = 0; m < 3; ++m) acc += Rj.R[m * 3 + r] * (Ri.t[m] - Rj.t[m]);
                partner.t[r] = acc;
            }
            partners.push_back(partner);
        }
        const SweepResult sw = sweep(imageI->image, partners, points, prob.inverse, K);

        // Accept per point: strong, unique peak away from the ends of the hypothesis range.
        std::vector<std::pair<std::size_t, std::size_t>> kept;
        for (std::size_t p = 0; p < points.size(); ++p) {
            const double* mean = &sw.mean[p * hypotheses];
            const double* low = &sw.low[p * hypotheses];
            std::size_t best = 0;
            for (std::size_t h = 1; h < hypotheses; ++h)
                if (mean[h] > mean[best]) best = h;
            if (mean[best] < 0.8 || low[best] < 0.65 || best == 0 || best == hypotheses - 1) continue;
            bool anyFar = false;
            double farMax = -DBL_MAX;
            for (std::size_t h = 0; h < hypotheses; ++h) {
                const std::size_t diff = h > best ? h - best : best - h;
                if (diff > 3) {
                    anyFar = true;
                    farMax = std::max(farMax, mean[h]);
                }
            }
            if (anyFar && farMax > mean[best] - 0.12) continue;
            kept.push_back({p, best});
        }
        if (kept.empty()) continue;

        const std::vector<std::uint8_t> rawRight = loadRaw(i, true);
        const Remapped right0 = remapRaw(rawRight, mapRight0);
        const Remapped right1 = remapRaw(rawRight, mapRight1);

        for (const auto& keptItem : kept) {
            const std::size_t p = keptItem.first;
            const std::size_t best = keptItem.second;
            // np.linspace(lo, hi, 21): fine[k] = k * step + lo, last point exactly hi.
            const double lo = prob.inverse[best - 1], hi = prob.inverse[best + 1];
            const double stepFine = (hi - lo) / 20.0;
            std::vector<double> fine(21);
            for (int k = 0; k < 21; ++k) fine[k] = double(k) * stepFine + lo;
            fine[20] = hi;
            const SweepResult fs = sweep(imageI->image, partners, {points[p]}, fine, K);
            const double* scores = fs.mean.data();
            std::size_t index = 0;
            for (std::size_t k = 1; k < 21; ++k)
                if (scores[k] > scores[index]) index = k;
            const double inverse = fine[index] + parabola(scores, 21, index) * (fine[1] - fine[0]);
            const double u = points[p][0], v = points[p][1];
            const double distance = 1.0 / inverse;
            double ray[3];
            rayOf(u, v, K, ray);
            const double X0 = ray[0] * distance, X1 = ray[1] * distance, X2 = ray[2] * distance;
            const double px = X0 - prob.baseline;
            const double rightX = std::atan2(px, std::hypot(X1, X2)) / K + kCenterU;
            const double expected = u - rightX;

            // Left-right search around the motion depth.
            const long long center = static_cast<long long>(std::trunc(expected));
            std::array<double, 25> search{};
            for (int s = 0; s < 25; ++s) search[s] = double(center - 12 + s);
            const Template tpl = makeTemplate(imageI->image, points[p], K);

            struct Stereo {
                bool ok = false;
                double d = 0, s = 0, side = 0;
            };
            auto stereo = [&](const std::vector<float>& rightImage) {
                Stereo out;
                std::array<double, 25> score{};
                for (int s = 0; s < 25; ++s) {
                    std::array<double, kPatch> values{};
                    std::array<bool, kPatch> valid{};
                    for (int k = 0; k < kPatch; ++k) {
                        // xs = uw - search, ys = vw (REF stereo()).
                        const double xs = (points[p][0] + offsetU(k)) - search[s];
                        const double ys = points[p][1] + offsetV(k);
                        bool ok = false;
                        values[k] = sampleImage(rightImage, xs, ys, ok);
                        valid[k] = ok;
                    }
                    score[s] = warpedNcc(tpl, values.data(), valid.data());
                }
                std::size_t m = 0;
                for (std::size_t s = 1; s < 25; ++s)
                    if (score[s] > score[m]) m = s;
                if (m == 0 || m == 24) return out;
                double side = -DBL_MAX;
                for (std::size_t s = 0; s < 25; ++s) {
                    const std::size_t diff = s > m ? s - m : m - s;
                    if (diff > 3) side = std::max(side, score[s]);
                }
                out.ok = true;
                out.d = search[m] + parabola(score.data(), 25, m);
                out.s = score[m];
                out.side = side;
                return out;
            };
            const Stereo st0 = stereo(right0.image);
            const Stereo st1 = stereo(right1.image);
            if (!st0.ok || !st1.ok || st0.s < 0.85 || st0.s - st0.side < 0.1) continue;

            Sample sample;
            sample.i = int(i);
            sample.tS = double(refs[i].when - start) / 1e9;
            sample.u = u;
            sample.v = v;
            sample.distance = distance;
            sample.expected = expected;
            sample.d0 = st0.d;
            sample.d1 = st1.d;
            sample.s0 = st0.s;
            // REF: float(scores[0].max()) over the fine hypotheses.
            sample.sweepScore = scores[0];
            for (std::size_t k = 1; k < 21; ++k) sample.sweepScore = std::max(sample.sweepScore, scores[k]);
            result.samples.push_back(sample);
        }
    }
    result.summary = summarize(result.samples);
    return result;
}

}  // namespace MotionYaw
}  // namespace FrameSpatial
