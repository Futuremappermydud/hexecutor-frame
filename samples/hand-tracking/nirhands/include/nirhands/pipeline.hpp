// Native stereo NIR hand pipeline for the Steam Frame slam pair: detector -> stereo seeds ->
// keypoint network -> tracks -> One Euro smoothing. A port of samples/hand-tracking/tools/nirhands_pipeline.py
// (and the stereo_nir_hands detector/crop/decode code it imports); keep the two in step.
// Models come from export_nirhands_onnx.py. The viewer runs prepared QNN16 programs on the CDSP;
// ONNX Runtime is available for host replay and as a numerical oracle.
#pragma once
#include "nirhands/geometry.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nirhands {

constexpr int kJoints = 21;
constexpr int kRoot = 9;  // middle MCP: crop centre and track position
using Joints = std::array<Vec3, kJoints>;

// One uint8 view, row stride == width.
struct Image {
    const uint8_t* data = nullptr;
    int width = 0, height = 0;
    uint8_t at(int x, int y) const { return data[size_t(y) * width + x]; }
};

struct Detection {
    int side = 0;  // 0 left, 1 right
    double score = 0, u = 0, v = 0, radius_px = 0, distance_m = 0;
    bool is_right() const { return side == 1; }
};

struct Seed {
    Vec3 center{};
    bool is_right = false;
    double score = 0;
    std::vector<int> views;          // e.g. {0,1} or {1}
    std::vector<Detection> detections;
    double ray_gap_m = NAN;
};

// ---- detector pre/post-processing (nirhands.training.detector) ----
// cv2.resize INTER_AREA to size x size (uint8, rounded), then per-image standardization.
void detector_preprocess(const Image& view, int size, float* out);
std::vector<Detection> decode_view(const float* maps, int h, int w, int image_w, int image_h, double threshold,
                                   int top_k, double merge_fraction = 0.6);
struct SeedConfig {
    double max_ray_gap_m = 0.04, depth_lo = 0.08, depth_hi = 1.3, max_distance_ratio = 0.6;
    double side_mismatch_cost_m = 0.02, dedupe_m = 0.06, single_view_threshold = 0.3;
    int max_hands = 2;
};
std::vector<Seed> seed_hands(const std::array<std::vector<Detection>, 2>& detections, const Rig& rig,
                             const SeedConfig& cfg);

// ---- keypoint crops (nirhands.training.samples / adapter.rectified_crop) ----
struct VirtualCamera {
    Vec3 center{};
    Mat3 R;           // world -> virtual, rows x y z
    double f = 0;
    Vec2 offset{};
    int size = 0;
    double depth(const Vec3& p) const { return (R * (p - center))[2]; }
};
struct Geometry {
    VirtualCamera camera;
    double baseline_m = 0, center_depth_m = 0;
    int primary_view = 0;
    bool mirrored = false, stereo = false;
};
struct Sample {
    std::vector<float> input;  // [2,S,S] robust-normalized network input, slot 1 zero when mono
    bool second_valid = false;
    Geometry geometry;
};
Sample build_sample(const std::array<Image, 2>& images, const Rig& rig, const Vec3& center, bool is_right, int size,
                    double radius = 0.10, double fill = 0.8);
// Raw crop and valid mask of one view (diagnostics/tests): resample through a virtual camera.
void resample(const VirtualCamera& v, const Image& image, const KbCamera& camera, uint8_t* crop, uint8_t* valid);
Joints decode_prediction(const Geometry& g, const float* keypoints, const float* disparity);

// ---- ONNX models ----
class Models {
public:
    struct Buffer { const void* data; size_t bytes; };
    using Bytes = std::vector<uint8_t>;
    using Run = std::function<std::vector<Bytes>(size_t, const std::vector<Buffer>&, const Geometry*)>;
    // dir holds detector.onnx, keypoints.onnx and manifest.json; threads per keypoint session.
    Models(const std::string& dir, int threads = 2);
    // Run prepared NPU programs: program 0 detector, program 1 keypoints.
    // Inputs and outputs use compact float32 NHWC QNN order.
    Models(const std::string& dir, Run run);
    ~Models();
    int detector_size() const;
    int keypoint_size() const;
    bool tight_crop() const;
    bool is_skeleton() const;
    // input [2,1,S,S] -> maps [2,10,S/4,S/4]
    void detect(const float* input, std::vector<float>& maps);
    // One hand: input [2,K,K] -> keypoints [21,2], disparity [21], visibility logits [21].
    // slot selects an independent session so two hands can run concurrently.
    // Skeleton models also return their auxiliary heatmap keypoints/disparity when kp_hm/disp_hm are given
    // (same encoding as kp/disp; left untouched for keypoint models).
    void keypoints(int slot, const float* input, bool second_valid, const Geometry& geometry,
                   float* kp, float* disp, float* vis_logit, float* kp_hm = nullptr, float* disp_hm = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- tracking ----
struct TrackerConfig {
    int max_hands = 2, max_tentative = 2, detect_every = 3;
    double associate_m = 0.10, associate_radius = 0.8;
    int confirm_hits = 2, max_missed = 3;
    double max_unseen_s = 0.3, max_unseen_visible_s = 1.0, visible_mean = 0.6;
    double start_stereo_score = 0.35, start_mono_score = 0.5;
    double min_visibility = 0.25;
    int max_weak = 3;
    double duplicate_m = 0.06, confirm_visibility = 0.45;
    int max_tentative_frames = 8;
    double confirm_span_m = 0.045;
    int confirm_span_frames = 3;
    double keep_span_m = 0.035;
    int span_window = 10;
    double reuse_id_s = 1.0, reuse_id_m = 0.15, max_speed_mps = 3.0, max_jump_m = 0.12;
    double min_cutoff = 1.5, beta = 10.0, d_cutoff = 1.0;
    // Detector thresholds (nirhands_take.py defaults).
    double detect_threshold = 0.3, candidate_threshold = 0.15, single_view_threshold = 0.3;
    int top_k = 3;
    // Shown joints of a skeleton model. Tracking (crops, gating) always uses the skeleton output.
    //   Skeleton: the skeleton output (template hand at a predicted scale).
    //   Heatmap:  the auxiliary heatmap keypoints (no bone constraints).
    //   Combined: the wearer's bones (median over recent heatmap hands), finger angles blended
    //             (1 - heatmap_angle_weight) skeleton + heatmap_angle_weight heatmap, palm from the heatmap.
    //             The skeleton output is shown until profile_min_samples stereo hands of that side were seen.
    enum class Output { Skeleton, Heatmap, Combined };
    Output output = Output::Skeleton;
    double heatmap_angle_weight = 0.5;
    int profile_window = 300, profile_min_samples = 20;
    double profile_min_visibility = 0.6;  // mean joint visibility for a hand to update the profile
};

struct HandResult {
    int id = 0;
    bool is_right = false;
    Joints joints{};  // smoothed, world (head) frame, metres
    Joints raw{};          // unsmoothed shown joints (see TrackerConfig::output)
    bool combined = false; // raw/joints are the combined output (false: skeleton output, profile not ready yet)
    std::array<float, kJoints> visibility{};
    bool stereo = false;
    double score = 0;
    int age = 0;
};

struct FrameResult {
    std::vector<HandResult> hands;
    bool detected = false;
    std::array<std::vector<Detection>, 2> detections;  // shown detections (score >= detect_threshold)
    std::vector<Seed> seeds;
    double detect_ms = 0, keypoints_ms = 0, total_ms = 0;
};

class OneEuro {
public:
    OneEuro(double min_cutoff, double beta, double d_cutoff) : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff) {}
    Joints operator()(const Joints& x, double t);

private:
    double min_cutoff_, beta_, d_cutoff_;
    bool init_ = false;
    Joints x_{}, dx_{};
    double t_ = 0;
};

class Pipeline {
public:
    Pipeline(const Rig& rig, std::shared_ptr<Models> models, TrackerConfig cfg = {});
    ~Pipeline();
    // One synchronized pair at capture time t (seconds, increasing).
    FrameResult step(const std::array<Image, 2>& images, double t);
    void reset();
    const Rig& rig() const { return rig_; }

private:
    struct Track;
    struct Impl;
    Rig rig_;
    std::shared_ptr<Models> models_;
    TrackerConfig cfg_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nirhands
