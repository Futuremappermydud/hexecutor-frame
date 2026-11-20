#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace FrameSpatial {

// Explicit 5x5 IR KB-spline projection for a recorded calibration entry.
// This is a geometry primitive only; callers must supply and verify calibration
// identity, source/time and any accepted stereo correction separately.
struct IrKBSplineIntrinsics {
    double fx, fy, cx, cy;
    std::array<double, 4> k;
    std::array<double, 2> nodeSpacing;
    std::array<double, 25> sU, sV;
};

class IrKBSplineProjector {
public:
    explicit IrKBSplineProjector(const IrKBSplineIntrinsics& source) {
        intr_ = source;
        // Recorded schema fields are float32 values even when JSON stores more digits.
        auto quantize = [](double x) { return double(float(x)); };
        for (double* value : {&intr_.fx, &intr_.fy, &intr_.cx, &intr_.cy}) *value = quantize(*value);
        for (auto& v : intr_.k) v = quantize(v);
        for (auto& v : intr_.nodeSpacing) v = quantize(v);
        for (auto& v : intr_.sU) v = quantize(v);
        for (auto& v : intr_.sV) v = quantize(v);
        if (!std::isfinite(intr_.fx) || !std::isfinite(intr_.fy) ||
            !std::isfinite(intr_.cx) || !std::isfinite(intr_.cy) ||
            intr_.fx <= 0 || intr_.fy <= 0) throw std::invalid_argument("invalid IR spline focal/center");
        for (const auto& v : intr_.k) if (!std::isfinite(v)) throw std::invalid_argument("invalid IR spline radial term");
        for (const auto& v : intr_.nodeSpacing)
            if (!std::isfinite(v) || v <= 0) throw std::invalid_argument("invalid IR spline spacing");
        for (const auto* grid : {&intr_.sU, &intr_.sV}) {
            for (const auto& v : *grid) if (!std::isfinite(v)) throw std::invalid_argument("invalid IR spline node");
            for (int i : {7, 11, 12}) if ((*grid)[i] != 0) throw std::invalid_argument("IR spline fixed node differs");
        }
        for (int yi = 0; yi <= 100; ++yi)
            for (int xi = 0; xi <= 100; ++xi) {
                const double x = 3. + (xi - 50) * (6. * .84 / 100.);
                const double y = 3. + (yi - 50) * (6. * .84 / 100.);
                lutU_[size_t(yi) * 101 + xi] = sampleGrid(intr_.sU, x, y);
                lutV_[size_t(yi) * 101 + xi] = sampleGrid(intr_.sV, x, y);
            }
    }

    std::array<double, 2> project(double x, double y, double z,
                                  double deltaK1 = 0., double focalScale = 1.) const {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
            (x == 0 && y == 0 && z == 0) || !std::isfinite(deltaK1) ||
            !std::isfinite(focalScale) || focalScale <= 0)
            throw std::invalid_argument("invalid IR spline ray or lens change");
        const double radius = std::hypot(x, y);
        const double theta = std::atan2(radius, z);
        const double t2 = theta * theta;
        const auto& k = intr_.k;
        const double radial = theta * (1 + t2 * (k[0] + deltaK1 + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
        const double scale = radius > 1e-12 ? radial / radius : 1.;
        const double u = intr_.fx * focalScale * x * scale + intr_.cx;
        const double v = intr_.fy * focalScale * y * scale + intr_.cy;
        const auto correction = bilinearCorrection(u, v);
        return {u + correction[0], v + correction[1]};
    }

private:
    static double cubic(double a, double b, double c, double d, double t) {
        return b + .5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
    }

    static double padded(const std::array<double, 25>& grid, int x, int y) {
        x = std::clamp(x, 0, 6);
        y = std::clamp(y, 0, 6);
        if (x == 0 || x == 6 || y == 0 || y == 6) return 0.;
        return grid[size_t(x - 1) * 5 + size_t(y - 1)];
    }

    static double sampleGrid(const std::array<double, 25>& grid, double x, double y) {
        const int ix = int(std::floor(x)), iy = int(std::floor(y));
        double rows[4];
        for (int dx = -1; dx <= 2; ++dx) {
            rows[dx + 1] = cubic(padded(grid, ix + dx, iy - 1), padded(grid, ix + dx, iy),
                                 padded(grid, ix + dx, iy + 1), padded(grid, ix + dx, iy + 2), y - iy);
        }
        return cubic(rows[0], rows[1], rows[2], rows[3], x - ix);
    }

    std::array<double, 2> bilinearCorrection(double u, double v) const {
        const double x = std::clamp((u - intr_.cx) * 100. / (6. * intr_.nodeSpacing[0] * .84) + 50., 0., 100.);
        const double y = std::clamp((v - intr_.cy) * 100. / (6. * intr_.nodeSpacing[1] * .84) + 50., 0., 100.);
        const int ix = std::min(int(std::floor(x)), 99), iy = std::min(int(std::floor(y)), 99);
        const double tx = x - ix, ty = y - iy;
        auto sample = [&](const std::array<double, 10201>& lut) {
            const size_t i = size_t(iy) * 101 + size_t(ix);
            return (1 - ty) * ((1 - tx) * lut[i] + tx * lut[i + 1]) +
                   ty * ((1 - tx) * lut[i + 101] + tx * lut[i + 102]);
        };
        return {sample(lutU_), sample(lutV_)};
    }

    IrKBSplineIntrinsics intr_{};
    std::array<double, 10201> lutU_{}, lutV_{};
};

}  // namespace FrameSpatial
