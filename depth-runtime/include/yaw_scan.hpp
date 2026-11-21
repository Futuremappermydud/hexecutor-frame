#pragma once

#include "hand_viewer_xr.hpp"
#include "ir_motion_stereo_yaw.hpp"

#include <filesystem>
#include <optional>

namespace FrameSpatial {

struct YawScanCalibration {
    MotionYaw::Problem problem;
    std::array<double, 9> headFromRectified{};
    std::array<double, 3> leftOriginInHead{};
};

YawScanCalibration loadYawScanCalibration(const std::string& xrservice, const std::string& device);
MotionYaw::Reference yawScanReference(const YawScanCalibration& calibration, int64_t when, const HeadPose& head);

// Both repeatability checks (time halves, and matches nearer and farther than
// kYawScanSplitDistance) must agree within this limit for a Ready fit.
constexpr double kYawScanAgreementDegrees = 0.087;
constexpr double kYawScanSplitDistance = 2.0;
// Fixed right-camera corrections built into the rectification on top of the factory calibration.
// The fitted roll and the row fit are corrections relative to these.
constexpr double kYawScanAssumedRollDegrees = 0.4;
constexpr double kYawScanAssumedPitchDegrees = 0.25;
// Motion-stereo yaw minus row yaw on 16 IR scans from 2026-11-05, -06 and -20: mean -0.156,
// sd 0.032 per scan, r = 0.89. Probably a lens-model error at the image corners, where the row
// yaw signal is; re-check it after any IR recalibration.
constexpr double kYawScanRowYawOffsetDegrees = -0.156;

struct YawScanFit {
    // NeedNearAndFar: enough matches overall, but too few on one side of the split distance.
    enum class State { MoreSupport, NeedNearAndFar, Unstable, Ready };
    struct FreeScale {
        double yawDegrees, scaleTerm;
    };
    // Fixed-scale yaw of the matches with minDistance <= distance < maxDistance (metres).
    struct Band {
        double minDistance = 0, maxDistance = 0;
        size_t samples = 0, keyframes = 0;
        std::optional<double> yawDegrees;  // absent with too few matches or keyframes
    };
    // Roll, pitch and yaw corrections from the vertical offset of the stereo matches alone,
    // without poses or motion depth. A yaw rotation shifts rows in proportion to the product of
    // the horizontal and vertical image angles, so yaw needs matches across all four quadrants.
    struct RowFit {
        size_t samples = 0, keyframes = 0;
        double rollDegrees = 0, pitchDegrees = 0, residualPx = 0;
        std::optional<double> yawDegrees;  // absent when the matches do not span the quadrants
        // yawDegrees + kYawScanRowYawOffsetDegrees, comparable with the motion-stereo yaw.
        std::optional<double> calibratedYawDegrees;
    };
    State state = State::MoreSupport;
    size_t samples = 0, keyframes = 0, farSamples = 0;
    // yawDegrees is fitted together with rollDegrees when the matches span enough image height
    // (rollFitted); the bands, halves and near/far sides hold roll at that value.
    double yawDegrees = 0, rollDegrees = 0, yawWithoutRollDegrees = 0, residualScaleDegrees = 0;
    bool rollFitted = false;
    double splitDifferenceDegrees = 0;
    bool timeSplitAgrees = false, depthSplitAgrees = false;
    Band nearSide, farSide;
    std::optional<double> depthDifferenceDegrees;  // far minus near, when both sides have support
    std::vector<Band> bands;
    std::optional<FreeScale> freeScale;  // yaw, roll and scale together
    std::optional<RowFit> rows;
};

YawScanFit fitYawScan(const std::vector<MotionYaw::Sample>& samples, double step, double midpointSeconds);

std::string yawScanImageName(size_t index, bool right);
void writeYawScanProblem(const std::filesystem::path& path, const MotionYaw::Problem& problem);
void writeYawScanSamples(const std::filesystem::path& path, const std::vector<MotionYaw::Sample>& samples);

}  // namespace FrameSpatial
