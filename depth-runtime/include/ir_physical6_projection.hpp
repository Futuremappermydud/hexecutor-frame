#pragma once

#include "ir_kb_spline.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace FrameSpatial {

using IrMat3 = std::array<std::array<double, 3>, 3>;
using IrVec3 = std::array<double, 3>;

struct IrPhysical6Parameters {
    IrVec3 rightToLeftRotationVectorRad{};
    double rightFocalLogScale = 0.;
    double leftDeltaK1 = 0.;
    double rightDeltaK1 = 0.;
};

// An isolated raw-pixel projection of the frozen physical6 IR geometry.
// The physical-from-virtual rotations and spline intrinsics must be supplied
// from the same explicit calibration identity as the fitted parameters.
class IrPhysical6Projection {
public:
    IrPhysical6Projection(const IrKBSplineIntrinsics& leftIntr,
                          const IrKBSplineIntrinsics& rightIntr,
                          const IrMat3& leftPhysicalFromVirtual,
                          const IrMat3& rightPhysicalFromVirtual,
                          double baselineM, const IrPhysical6Parameters& parameters)
        : left_(leftIntr), right_(rightIntr), leftR_(leftPhysicalFromVirtual),
          rightR_(rightPhysicalFromVirtual), baseline_(baselineM), parameters_(parameters) {
        if (!std::isfinite(baseline_) || baseline_ <= 0 ||
            !std::isfinite(parameters_.rightFocalLogScale) ||
            !std::isfinite(parameters_.leftDeltaK1) || !std::isfinite(parameters_.rightDeltaK1))
            throw std::invalid_argument("invalid IR physical6 scale/lens parameters");
        for (double v : parameters_.rightToLeftRotationVectorRad)
            if (!std::isfinite(v)) throw std::invalid_argument("invalid IR physical6 rotation");
        for (const auto& r : {leftR_, rightR_}) {
            for (const auto& row : r)
                for (double v : row) if (!std::isfinite(v)) throw std::invalid_argument("invalid IR camera matrix");
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double dot = 0.;
                    for (int k = 0; k < 3; ++k) dot += r[k][i] * r[k][j];
                    if (std::abs(dot - (i == j ? 1. : 0.)) > 1e-6)
                        throw std::invalid_argument("IR camera matrix not orthogonal");
                }
            if (determinant(r) < 0.) throw std::invalid_argument("IR camera matrix reflected");
        }
        relative_ = rodrigues(parameters_.rightToLeftRotationVectorRad);
        rightFocalScale_ = std::exp(parameters_.rightFocalLogScale);
        if (!std::isfinite(rightFocalScale_) || rightFocalScale_ <= 0)
            throw std::invalid_argument("invalid IR physical6 focal scale");
    }

    std::array<double, 2> leftRawRay(const IrVec3& rectifiedRay) const {
        const auto physical = multiply(leftR_, rectifiedRay);
        return left_.project(physical[0], physical[1], physical[2], parameters_.leftDeltaK1);
    }

    std::array<double, 2> rightRawRay(const IrVec3& rectifiedRay) const {
        const auto rightVirtual = multiplyTransposed(relative_, rectifiedRay);
        const auto physical = multiply(rightR_, rightVirtual);
        return right_.project(physical[0], physical[1], physical[2],
                              parameters_.rightDeltaK1, rightFocalScale_);
    }

    std::array<double, 2> leftRawPixel(const IrVec3& leftVirtualPoint) const {
        return leftRawRay(leftVirtualPoint);
    }

    std::array<double, 2> rightRawPixel(const IrVec3& leftVirtualPoint) const {
        const IrVec3 shifted{leftVirtualPoint[0] - baseline_, leftVirtualPoint[1], leftVirtualPoint[2]};
        return rightRawRay(shifted);
    }

private:
    static IrVec3 multiply(const IrMat3& m, const IrVec3& p) {
        IrVec3 q{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) q[i] += m[i][j] * p[j];
        return q;
    }

    static IrVec3 multiplyTransposed(const IrMat3& m, const IrVec3& p) {
        IrVec3 q{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) q[i] += m[j][i] * p[j];
        return q;
    }

    static double determinant(const IrMat3& m) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    static IrMat3 rodrigues(const IrVec3& v) {
        for (double component : v)
            if (std::abs(component) > .15)
                throw std::invalid_argument("IR physical6 rotation outside fitted bound");
        const double angle = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double a = angle < 1e-8 ? 1. - angle * angle / 6. : std::sin(angle) / angle;
        const double b = angle < 1e-8 ? .5 - angle * angle / 24. :
                         (1. - std::cos(angle)) / (angle * angle);
        const IrMat3 skew{{{0., -v[2], v[1]}, {v[2], 0., -v[0]}, {-v[1], v[0], 0.}}};
        IrMat3 result{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double squared = 0.;
                for (int k = 0; k < 3; ++k) squared += skew[i][k] * skew[k][j];
                result[i][j] = (i == j ? 1. : 0.) + a * skew[i][j] + b * squared;
            }
        return result;
    }

    IrKBSplineProjector left_, right_;
    IrMat3 leftR_{}, rightR_{}, relative_{};
    double baseline_ = 0., rightFocalScale_ = 1.;
    IrPhysical6Parameters parameters_{};
};

}  // namespace FrameSpatial
