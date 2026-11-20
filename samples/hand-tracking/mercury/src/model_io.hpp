// Mercury model pre- and post-processing without OpenCV. Ported from Monado's
// hg_model.cpp and hg_image_distorter.cpp (BSL-1.0, see ../NOTICE) for a
// rectified pinhole camera.
#pragma once

#include "mercury/tracker.hpp"

#include "mercury/kine_common.hpp"
#include "xrt/xrt_defines.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstdint>

namespace mercury::detail {

namespace hm = xrt::tracking::hand::mercury;
using hand21_2d = std::array<hm::vec2_5, 21>;
using Joints21 = Eigen::Array<float, 3, 21>;

// Row-major 2x3 affine transform: {a, b, c, d, e, f} maps (x, y) to (ax+by+c, dx+ey+f).
using Affine = std::array<float, 6>;

inline xrt_vec2 apply(const Affine& m, xrt_vec2 p) {
    return {p.x * m[0] + p.y * m[1] + m[2], p.x * m[3] + p.y * m[4] + m[5]};
}

// cv::warpAffine(INTER_LINEAR, BORDER_CONSTANT 0) of an 8-bit image with the
// forward transform `forward`, including OpenCV's 1/32-pixel fixed point.
void warp_affine_u8(const GrayImage& src, const Affine& forward, uint8_t* dst, int width, int height);

// Mercury's blackbar() for CAMERA_ORIENTATION_0: letterbox into size x size.
// Returns the transform from letterboxed pixels back to source pixels.
Affine letterbox(const GrayImage& src, int size, uint8_t* dst);

// Mercury's normalizeGrayscaleImage(): scale to std 0.25 and mean 0.5. Returns
// false for a constant image.
bool normalize_gray(const uint8_t* src, size_t count, float* dst);

// Monado t_camera_models_* for a pinhole camera, in Monado camera axes
// (+Y up, -Z forward) after the flip.
bool unproject_and_flip(const PinholeCamera& camera, float x, float y, xrt_vec3& out);
bool flip_and_project(const PinholeCamera& camera, float x, float y, float z, float& out_x, float& out_y);

struct Projection {
    Eigen::Quaternionf rot_quat = Eigen::Quaternionf::Identity();
    float stereographic_radius = 0;
    bool flip = false;
};

void make_projection_instructions(bool flip_after, float expand_val, float twist, const Joints21& joints,
                                  Projection& out, hand21_2d& out_hand);
void make_projection_instructions_angular(xrt_vec3 direction_3d, bool flip_after, float angular_radius,
                                          float expand_val, float twist, Projection& out);

// StereographicDistort: 128x128 nearest (truncating) resample of the view.
void stereographic_project_image(const PinholeCamera& camera, const Projection& projection,
                                 const GrayImage& image, uint8_t* out128);

// Decode one keypoint inference into Mercury's per-view observation.
// Returns Mercury's is_hand (hand probability at least min_hand_probability).
bool decode_keypoints(const KeypointOutput& out, int hand_idx, float min_hand_probability,
                      hm::one_frame_one_view& view);

// One detection inference in letterbox coordinates mapped to source pixels.
struct Region {
    xrt_vec2 center_px{};
    float size_px = 0;
    bool found = false;
    float confidence = 0;
};
void decode_detection(const DetectionOutput& out, const Affine& go_back, float min_confidence,
                      std::array<Region, 2>& regions);

}  // namespace mercury::detail
