#include "model_temporal.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace FrameSpatial {
namespace {
constexpr int HalfWidth = ModelTemporalState::Width / 2, HalfHeight = ModelTemporalState::Height / 2;
constexpr int HalfCells = HalfWidth * HalfHeight;
constexpr double PinholeFocal = 380.0, PinholeCenter = 308.0, RectifiedSize = 616.0;
constexpr double ScaleX = ModelTemporalState::Width / RectifiedSize;
constexpr double ScaleY = ModelTemporalState::Height / RectifiedSize;
constexpr double CenterU = (ModelTemporalState::Width - 1) / 2.0;
constexpr double CenterV = (ModelTemporalState::Height - 1) / 2.0;
constexpr double NearLimit = 0.05;
constexpr float MinimumSaved = 0.125f;  // half-resolution value of 0.25 model pixels
}

ModelTemporalState::ModelTemporalState(double sphericalStepRadians)
    : step_(sphericalStepRadians), baseline_(24.3048 / 380.0), previous_(2 * HalfCells), best_(HalfCells),
      tangent_(HalfWidth), sine_(HalfHeight), cosine_(HalfHeight) {
    for (int j = 0; j < HalfWidth; ++j) tangent_[j] = std::tan((j * 2 + 0.5 - CenterU) * step_);
    for (int i = 0; i < HalfHeight; ++i) {
        const double phi = (i * 2 + 0.5 - CenterV) * step_;
        sine_[i] = std::sin(phi);
        cosine_[i] = std::cos(phi);
    }
}

ModelTemporalState::History ModelTemporalState::admit(bool aligned, bool resetRequested, const double rotation[3][3],
                                                      int64_t gapNs, int64_t maxGapNs,
                                                      double maxRotationDegrees) const {
    if (resetRequested) return History::Reset;
    if (!hasHistory_) return History::None;
    if (!aligned) return History::Unaligned;
    if (maxGapNs > 0 && (gapNs <= 0 || gapNs > maxGapNs)) return History::TimeGap;
    if (maxRotationDegrees > 0) {
        const double cosine = (rotation[0][0] + rotation[1][1] + rotation[2][2] - 1.0) / 2.0;
        const double degrees = std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
        if (degrees > maxRotationDegrees) return History::Rotation;
    }
    return History::Usable;
}

void ModelTemporalState::save(const float* left, const float* right) {
    const float* planes[2]{left, right};
    for (int eye = 0; eye < 2; ++eye) {
        const float* source = planes[eye];
        float* target = previous_.data() + size_t(eye) * HalfCells;
        for (int i = 0; i < HalfHeight; ++i) {
            const float* top = source + size_t(i) * 2 * Width;
            const float* bottom = top + Width;
            for (int j = 0; j < HalfWidth; ++j) {
                const float sum = top[j * 2] + top[j * 2 + 1] + bottom[j * 2] + bottom[j * 2 + 1];
                target[size_t(i) * HalfWidth + j] = sum * 0.25f * 0.5f;
            }
        }
    }
    hasHistory_ = true;
}

void ModelTemporalState::warpEye(int eye, const double R[3][3], const double t[3], float* hidden) {
    std::fill(best_.begin(), best_.end(), -std::numeric_limits<double>::infinity());
    const float* low = previous_.data() + size_t(eye) * HalfCells;
    const double step = step_, b = baseline_;
    for (int i = 0; i < HalfHeight; ++i) {
        for (int j = 0; j < HalfWidth; ++j) {
            const float saved = low[size_t(i) * HalfWidth + j];
            if (!(saved > MinimumSaved)) continue;
            const double d = double(saved) * 2.0;
            double x, y, z;
            if (step > 0) {
                const double theta = (j * 2 + 0.5 - CenterU) * step;
                double tangentLeft, tangentRight;
                if (eye) {
                    tangentRight = tangent_[j];
                    tangentLeft = std::tan(theta + d * step);
                } else {
                    tangentLeft = tangent_[j];
                    tangentRight = std::tan(theta - d * step);
                }
                const double rho = b / (tangentLeft - tangentRight);
                x = rho * tangentLeft;
                y = rho * sine_[i];
                z = rho * cosine_[i];
            } else {
                const double xr = (j * 2 + 0.5 + 0.5) / ScaleX - 0.5;
                const double yr = (i * 2 + 0.5 + 0.5) / ScaleY - 0.5;
                z = PinholeFocal * b / (d / ScaleX);
                x = (xr - PinholeCenter) * z / PinholeFocal + (eye ? b : 0.0);
                y = (yr - PinholeCenter) * z / PinholeFocal;
            }
            const double px = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0];
            const double py = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1];
            const double pz = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2];
            if (!(pz > NearLimit) || !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;
            double column, row, disparity;
            if (step > 0) {
                const double rho = std::hypot(py, pz);
                const double thetaLeft = std::atan2(px, rho);
                const double thetaRight = std::atan2(px - b, rho);
                disparity = (thetaLeft - thetaRight) / step;
                column = std::floor((eye ? thetaRight : thetaLeft) / step + CenterU);
                row = std::floor(std::atan2(py, pz) / step + CenterV);
            } else {
                const double left = PinholeFocal * px / pz + PinholeCenter;
                const double right = PinholeFocal * (px - b) / pz + PinholeCenter;
                const double vertical = PinholeFocal * py / pz + PinholeCenter;
                disparity = (left - right) * ScaleX;
                column = std::floor(((eye ? right : left) + 0.5) * ScaleX - 0.5);
                row = std::floor((vertical + 0.5) * ScaleY - 0.5);
            }
            if (column < 0 || column >= Width || row < 0 || row >= Height) continue;
            double& cell = best_[size_t(int(row) >> 1) * HalfWidth + (int(column) >> 1)];
            cell = std::max(cell, disparity);
        }
    }
    int filled = 0;
    float* plane = hidden + size_t(eye) * Height * Width;
    for (int i = 0; i < HalfHeight; ++i) {
        float* top = plane + size_t(i) * 2 * Width;
        float* bottom = top + Width;
        for (int j = 0; j < HalfWidth; ++j) {
            const double value = best_[size_t(i) * HalfWidth + j];
            const float output = std::isinf(value) ? 0.f : float(value);
            filled += output > 0;
            top[j * 2] = top[j * 2 + 1] = bottom[j * 2] = bottom[j * 2 + 1] = output;
        }
    }
    splatFraction_ += double(filled) / (2.0 * HalfCells);
}

void ModelTemporalState::inputs(const double R[3][3], const double t[3], float* hidden, float* transforms) {
    splatFraction_ = 0;
    if (!hasHistory_) {
        std::memset(hidden, 0, sizeof(float) * HiddenValues);
        std::memset(transforms, 0, sizeof(float) * TransformValues);
        return;
    }
    // Right-eye translation is R * (b, 0, 0) + t - (b, 0, 0).
    const double rightEye[3]{R[0][0] * baseline_ + t[0] - baseline_, R[1][0] * baseline_ + t[1], R[2][0] * baseline_ + t[2]};
    for (int eye = 0; eye < 2; ++eye) {
        const double* eyeTranslation = eye ? rightEye : t;
        float* out = transforms + eye * 16;
        // Row-major storage of the column-major motion minus identity.
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column)
                out[column * 4 + row] = float(R[row][column] - (row == column ? 1.0 : 0.0));
            out[3 * 4 + row] = float(eyeTranslation[row]);
            out[row * 4 + 3] = 0.f;
        }
        out[15] = 0.f;
        warpEye(eye, R, t, hidden);
    }
}

}  // namespace FrameSpatial
