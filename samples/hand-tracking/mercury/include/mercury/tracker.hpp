// C++17 port of Monado's Mercury stereo hand tracker (BSL-1.0, see ../../NOTICE).
//
// The two neural networks are behind Networks so that the DSP executor can run
// them; this library contains no CPU, ONNX Runtime or Python inference path.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mercury {

constexpr int kDetectionSize = 160;
constexpr int kKeypointSize = 128;
constexpr int kHeatmapSize = 22;
constexpr int kJointCount = 26;  // OpenXR XrHandJointEXT order: palm, wrist, thumb..little.

struct Quat { float x = 0, y = 0, z = 0, w = 1; };

// Inputs and outputs use the ONNX models' float32 layouts. A backend whose
// compiled graph uses another layout (QNN returns heatmap_xy as NHWC) converts.
// Fields after the tensors describe the request for logging and tests; a
// network backend must not change its results based on them.
struct DetectionInput {
    std::array<float, kDetectionSize * kDetectionSize> image;  // [1,1,160,160], Mercury-normalized
    int view = 0;  // 0 left camera, 1 right camera
};

struct DetectionOutput {
    // Slot 0 is the left hand, slot 1 the right hand.
    std::array<float, 2> hand_exists, cx, cy, size;
};

struct KeypointInput {
    std::array<float, kKeypointSize * kKeypointSize> image;  // [1,1,128,128], Mercury-normalized
    std::array<float, 42> last_keypoints;                    // [1,42]
    float use_last_keypoints = 0.0f;                         // [1]
    int view = 0, hand = 0;                                  // hand 0 left, 1 right (crop mirrored)
    Quat look_dir;                                           // crop center direction, camera axes
    float stereographic_radius = 0;
};

struct KeypointOutput {
    std::array<float, 21 * kHeatmapSize * kHeatmapSize> heatmap_xy;  // [1,21,22,22] (NCHW)
    std::array<float, 21 * kHeatmapSize> heatmap_depth;              // [1,21,22]
    std::array<float, 8> scalar_extras;                              // [1,8]; [0] is the hand logit
    std::array<float, 10> curls;                                     // [1,10]; 5 values, 5 variances
};

// Runs grayscale_detection_160x160 and grayscale_keypoint_jan18. Each call
// receives every inference needed at that point of the frame (up to two
// detections, up to four keypoint crops) so a backend can batch or pipeline.
// Throw on failure; the tracker does not retry or substitute another path.
class Networks {
public:
    virtual ~Networks() = default;
    virtual void detect(const DetectionInput* inputs, DetectionOutput* outputs, size_t count) = 0;
    virtual void keypoints(const KeypointInput* inputs, KeypointOutput* outputs, size_t count) = 0;
};

// Pinhole intrinsics of a rectified image, in pixels of the image passed to process().
struct PinholeCamera {
    float fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
};

struct Vec3 { float x = 0, y = 0, z = 0; };
struct Pose { Quat orientation; Vec3 position; };

// Camera axes follow OpenXR/Monado: +X right, +Y up, -Z forward (the image's
// +x right, +y down, +z forward with y and z negated).
struct StereoCalibration {
    PinholeCamera left, right;
    // Transforms a point in left-camera axes into right-camera axes. For an ideal
    // rectified pair with baseline b, this is identity rotation and position {-b, 0, 0}.
    Pose left_in_right;
};

struct GrayImage {
    const uint8_t* data = nullptr;
    int width = 0, height = 0;
    size_t stride = 0;  // bytes per row
};

struct Joint {
    Pose pose;  // left-camera axes, meters
    float radius = 0;
};

struct Hand {
    bool active = false;  // Mercury's is_active, including its display warm-up
    bool tracked = false; // optimized this frame, even before the warm-up allows display
    std::array<Joint, kJointCount> joints;
    float reprojection_error = 0;
};

struct Result {
    std::array<Hand, 2> hands;  // 0 = left hand, 1 = right hand
    int64_t timestamp_ns = 0;
    int detections_run = 0;
    int keypoints_run = 0;
};

// Mercury's tuneable defaults (hg_sync.cpp, Monado 3acbda6b).
struct Options {
    float after_detection_fac = 0.65f;
    float dyn_radii_fac = 1.7f;
    float amount_to_lerp_prediction = 0.4f;
    float amt_use_depth = 0.01f;
    float mpiou_any = 0.7f;
    float mpiou_double_detection = 0.4f;
    float mpiou_single_detection = 0.2f;
    float max_reprojection_error = 15.0f;
    float opt_smooth_factor = 2.0f;
    float max_hand_dist = 1.7f;
    float min_detection_confidence = 0.3f;
    float detection_confidence_sum = 0.92f;  // hard-coded in Mercury
    float min_hand_probability = 0.97f;      // hard-coded in Mercury
    int max_num_outside_view = 6;
    int num_frames_before_display = 10;
    bool always_run_detection_model = false;
    bool optimize_hand_size = true;
    bool enable_pose_predicted_input = true;
    bool detection_model_in_both_views = false;
    bool log_debug = false;
};

class Tracker {
public:
    Tracker(const StereoCalibration& calibration, Networks& networks, const Options& options = {});
    ~Tracker();
    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;

    // Both images must match the calibration sizes. Timestamps must increase.
    Result process(const GrayImage& left, const GrayImage& right, int64_t timestamp_ns);

    // Forget hand state and hand-size estimates, as for a new stream or user.
    void reset();

    float target_hand_size() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace mercury
