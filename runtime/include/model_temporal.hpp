#pragma once
#include <cstdint>
#include <vector>

namespace FrameSpatial {

// Model-style recurrent inputs for the stereo model: hidden is the previous left and
// right disparity output, warped to the current pose. Port of scripts/model_temporal.py,
// which stays as the reference for tools/test_model_temporal.cpp.
class ModelTemporalState {
public:
    static constexpr int Width = 256, Height = 320;
    static constexpr int HiddenValues = 2 * Height * Width, TransformValues = 2 * 4 * 4;

    enum class History { Usable, None, Reset, Unaligned, TimeGap, Rotation };

    // sphericalStepRadians 0 selects the resized rectified pinhole inputs.
    explicit ModelTemporalState(double sphericalStepRadians = 0);

    void reset() { hasHistory_ = false; }
    bool hasHistory() const { return hasHistory_; }

    // Whether saved history may be applied to this frame. Anything but Usable means the
    // caller should reset and send zeros. A limit of zero or less disables that check.
    History admit(bool aligned, bool resetRequested, const double rotation[3][3], int64_t gapNs,
                  int64_t maxGapNs, double maxRotationDegrees) const;

    // left, right: Width x Height disparity planes of the model output.
    void save(const float* left, const float* right);

    // rotation, translation map previous rectified-left camera coordinates to current
    // ones (metres, x right, y down, z forward). hidden holds HiddenValues floats
    // (plane, row, column); transforms holds TransformValues floats (eye, row, column).
    // Without history both are zeroed.
    void inputs(const double rotation[3][3], const double translation[3], float* hidden, float* transforms);

    double splatFraction() const { return splatFraction_; }

private:
    void warpEye(int eye, const double rotation[3][3], const double translation[3], float* hidden);

    double step_;
    double baseline_;
    bool hasHistory_ = false;
    double splatFraction_ = 0;
    std::vector<float> previous_;   // 2 planes at half resolution, values halved
    std::vector<double> best_;      // per-cell maximum disparity while splatting
    std::vector<double> tangent_;   // tan(theta) per half-resolution column
    std::vector<double> sine_, cosine_;  // sin and cos of phi per half-resolution row
};

}  // namespace FrameSpatial
