// Copyright 2021-2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
//
// Ported from Monado's src/xrt/tracking/hand/mercury/hg_model.cpp and
// hg_image_distorter.cpp at 3acbda6b7dacbe014749020e8543fded54ef076b
// (authors Moshi Turner, Jakob Bornecrantz, Charlton Rodda). Changes for Frame
// Spatial Runtime: no OpenCV or ONNX Runtime, pinhole camera only, debug
// drawing removed.
#include "model_io.hpp"

#include "mercury/hg_stereographic_unprojection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace mercury::detail {

namespace {

constexpr int wsize = kKeypointSize;

template <typename T>
T map_ranges(T value, T from_low, T from_high, T to_low, T to_high) {
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
}

float vec2_len(xrt_vec2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

int argmax(const float* data, int size) {
    float max_value = data[0];
    int out_idx = 0;
    for (int i = 1; i < size; i++) {
        if (data[i] > max_value) {
            max_value = data[i];
            out_idx = i;
        }
    }
    return out_idx;
}

bool hand_depth_center_of_mass(float data[22], float* out_depth, float* out_confidence) {
    float avg = 0, sum = 0;
    for (int i = 0; i < 22; i++) {
        data[i] = std::fmin(1.0f, std::fmax(data[i], 0.0f));
        sum += data[i];
        avg += data[i] * float(i);
    }
    if (sum < 1e-5f) return false;
    avg /= sum;
    if (avg < 0 || avg > 21) return false;
    if (avg != avg) {
        *out_depth = 0;
        *out_confidence = 0;
        return false;
    }
    *out_confidence = data[int(avg)];
    float depth_value = avg + 0.5f;
    depth_value /= 22;
    depth_value -= 0.5f;
    depth_value *= 2 * 1.5f;
    *out_depth = depth_value;
    return true;
}

bool refine_center_of_distribution(const float* data, int coarse_x, int coarse_y, int w, int h, float* out_x,
                                   float* out_y) {
    float sum = 0, sum_x = 0, sum_y = 0;
    const int max_kern_width = 10;
    // Mercury's kernel bounds, including its noted edge cases, kept unchanged.
    int kern_width_x = std::max(0, std::min(coarse_x, std::min(max_kern_width, std::abs(coarse_x - w) - 1)));
    int kern_width_y = std::max(0, std::min(coarse_y, std::min(max_kern_width, std::abs(coarse_y - h) - 1)));
    for (int y = coarse_y - kern_width_y; y <= coarse_y + kern_width_y; y++) {
        for (int x = coarse_x - kern_width_x; x <= coarse_x + kern_width_x; x++) {
            float val = data[y * w + x];
            sum += val;
            sum_y += val * (float(y) + 0.5f);
            sum_x += val * (float(x) + 0.5f);
        }
    }
    if (sum == 0) {
        *out_x = float(coarse_x);
        *out_y = float(coarse_y);
        return false;
    }
    *out_x = sum_x / sum;
    *out_y = sum_y / sum;
    return true;
}

Eigen::Quaternionf direction(const Eigen::Vector3f& dir, float twist) {
    Eigen::Quaternionf one = Eigen::Quaternionf().setFromTwoVectors(-Eigen::Vector3f::UnitZ(), dir);
    Eigen::Quaternionf two;
    two = Eigen::AngleAxisf(twist, -Eigen::Vector3f::UnitZ());
    return one * two;
}

void project_21_points_unscaled(const Joints21& joints, const Eigen::Quaternionf& rot_quat, hand21_2d& out) {
    for (int i = 0; i < 21; i++) {
        Eigen::Vector3f d = joints.col(i);
        d.normalize();
        d = rot_quat.conjugate() * d;
        float denom = 1 - d.z();
        out[i].pos_2d.x = d.x() / denom;
        out[i].pos_2d.y = d.y() / denom;
    }
}

void add_rel_depth(const Joints21& joints, hand21_2d& out) {
    float hand_size = (Eigen::Vector3f(joints.col(0)) - Eigen::Vector3f(joints.col(9))).norm();
    float midpxm_depth = Eigen::Vector3f(joints.col(9)).norm();
    for (int i = 0; i < 21; i++) {
        float jd = Eigen::Vector3f(joints.col(i)).norm();
        out[i].depth_relative_to_midpxm = (jd - midpxm_depth) / hand_size;
    }
}

float palm_length(const hand21_2d& joints) {
    xrt_vec2 a = joints[0].pos_2d, b = joints[9].pos_2d, c = joints[5].pos_2d, d = joints[17].pos_2d;
    float fwd = vec2_len({a.x - b.x, a.y - b.y});
    float side = vec2_len({c.x - d.x, c.y - d.y});
    return std::fmax(fwd, side);
}

}  // namespace

void warp_affine_u8(const GrayImage& src, const Affine& forward, uint8_t* dst, int width, int height) {
    // cv::warpAffine without WARP_INVERSE_MAP inverts in double precision.
    double M[6] = {forward[0], forward[1], forward[2], forward[3], forward[4], forward[5]};
    double D = M[0] * M[4] - M[1] * M[3];
    D = D != 0 ? 1. / D : 0;
    double A11 = M[4] * D, A22 = M[0] * D;
    M[0] = A11;
    M[1] *= -D;
    M[3] *= -D;
    M[4] = A22;
    double b1 = -M[0] * M[2] - M[1] * M[5];
    double b2 = -M[3] * M[2] - M[4] * M[5];
    M[2] = b1;
    M[5] = b2;

    constexpr int INTER_BITS = 5, INTER_TAB_SIZE = 1 << INTER_BITS;
    constexpr int AB_BITS = 10, AB_SCALE = 1 << AB_BITS;
    constexpr int round_delta = AB_SCALE / INTER_TAB_SIZE / 2;
    auto pixel = [&](int x, int y) -> int {
        if (x < 0 || y < 0 || x >= src.width || y >= src.height) return 0;
        return src.data[size_t(y) * src.stride + size_t(x)];
    };
    for (int y = 0; y < height; y++) {
        int X0 = int(std::lrint((M[1] * y + M[2]) * AB_SCALE)) + round_delta;
        int Y0 = int(std::lrint((M[4] * y + M[5]) * AB_SCALE)) + round_delta;
        for (int x = 0; x < width; x++) {
            int X = (X0 + int(std::lrint(M[0] * x * AB_SCALE))) >> (AB_BITS - INTER_BITS);
            int Y = (Y0 + int(std::lrint(M[3] * x * AB_SCALE))) >> (AB_BITS - INTER_BITS);
            int xi = X >> INTER_BITS, yi = Y >> INTER_BITS;
            int fx = X & (INTER_TAB_SIZE - 1), fy = Y & (INTER_TAB_SIZE - 1);
            // Products of 1/32 steps scaled by 2^15 are exact, so OpenCV's
            // table normalization never adjusts these weights.
            int w00 = 32 * (32 - fx) * (32 - fy), w01 = 32 * fx * (32 - fy);
            int w10 = 32 * (32 - fx) * fy, w11 = 32 * fx * fy;
            int sum = w00 * pixel(xi, yi) + w01 * pixel(xi + 1, yi) + w10 * pixel(xi, yi + 1) +
                      w11 * pixel(xi + 1, yi + 1);
            dst[size_t(y) * size_t(width) + size_t(x)] = uint8_t(std::min(255, (sum + (1 << 14)) >> 15));
        }
    }
}

Affine letterbox(const GrayImage& src, int size, uint8_t* dst) {
    float in_w = float(src.width), in_h = float(src.height);
    float scale_down = std::fmin(float(size) / in_w, float(size) / in_h);
    float width_inside = in_w * scale_down, height_inside = in_h * scale_down;
    float translate_x = (float(size) - width_inside) / 2;
    float translate_y = (float(size) - height_inside) / 2;
    Affine go = {scale_down, 0.0f, translate_x, 0.0f, scale_down, translate_y};
    if (scale_down < 0.5f) {
        // The Frame rectifier provides 616px images to a 160px detector. A
        // point-sampled warp aliases the dark finger edges into the background
        // and collapses detector confidence. Use a separable, antialiased
        // triangle filter at this reduction, keeping the same affine map.
        std::fill(dst, dst + size_t(size) * size, uint8_t(0));
        std::vector<float> horizontal(size_t(src.height) * size, 0.0f);
        const float support = 1.0f / scale_down;
        auto weight = [&](float distance) { return std::max(0.0f, 1.0f - std::abs(distance) * scale_down); };
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < size; ++x) {
                if (x < translate_x || x >= translate_x + width_inside) continue;
                const float center = (float(x) + 0.5f - translate_x) / scale_down - 0.5f;
                float sum = 0, total = 0;
                for (int sx = std::max(0, int(std::floor(center - support)));
                     sx <= std::min(src.width - 1, int(std::ceil(center + support))); ++sx) {
                    const float w = weight(float(sx) - center);
                    sum += w * src.data[size_t(y) * src.stride + size_t(sx)];
                    total += w;
                }
                horizontal[size_t(y) * size + size_t(x)] = total ? sum / total : 0;
            }
        }
        for (int y = 0; y < size; ++y) {
            if (y < translate_y || y >= translate_y + height_inside) continue;
            const float center = (float(y) + 0.5f - translate_y) / scale_down - 0.5f;
            for (int x = 0; x < size; ++x) {
                if (x < translate_x || x >= translate_x + width_inside) continue;
                float sum = 0, total = 0;
                for (int sy = std::max(0, int(std::floor(center - support)));
                     sy <= std::min(src.height - 1, int(std::ceil(center + support))); ++sy) {
                    const float w = weight(float(sy) - center);
                    sum += w * horizontal[size_t(sy) * size + size_t(x)];
                    total += w;
                }
                dst[size_t(y) * size + size_t(x)] = uint8_t(std::clamp(std::lround(total ? sum / total : 0), 0l, 255l));
            }
        }
    } else {
        warp_affine_u8(src, go, dst, size, size);
    }
    // Mercury inverts the 3x3 float matrix; for this diagonal form that is exact.
    return {1.0f / scale_down, 0.0f, -translate_x / scale_down, 0.0f, 1.0f / scale_down, -translate_y / scale_down};
}

bool normalize_gray(const uint8_t* src, size_t count, float* dst) {
    // cv::meanStdDev accumulates in double and reports the population deviation.
    auto stats = [&](double& mean, double& stddev) {
        double s = 0, s2 = 0;
        for (size_t i = 0; i < count; i++) {
            s += dst[i];
            s2 += double(dst[i]) * dst[i];
        }
        mean = s / double(count);
        stddev = std::sqrt(std::max(0.0, s2 / double(count) - mean * mean));
    };
    for (size_t i = 0; i < count; i++) dst[i] = float(double(src[i]) * (1 / 255.0));
    double mean, stddev;
    stats(mean, stddev);
    if (stddev == 0) return false;
    const float scale = float(0.25 / stddev);
    for (size_t i = 0; i < count; i++) dst[i] *= scale;
    stats(mean, stddev);
    const float offset = float(0.5 - mean);
    for (size_t i = 0; i < count; i++) dst[i] += offset;
    return true;
}

bool unproject_and_flip(const PinholeCamera& c, float x, float y, xrt_vec3& out) {
    float nx = (x - c.cx) / c.fx, ny = (y - c.cy) / c.fy;
    float inv = 1.0f / std::sqrt(nx * nx + ny * ny + 1.0f);
    out = {nx * inv, -ny * inv, -inv};
    return true;
}

bool flip_and_project(const PinholeCamera& c, float x, float y, float z, float& out_x, float& out_y) {
    // Monado camera axes to image axes: (x, -y, -z), then project with +Z forward.
    float iy = -y, iz = -z;
    if (std::fabs(iz) < 1e-9f) {
        out_x = c.cx;
        out_y = c.cy;
        return false;
    }
    out_x = c.fx * x / iz + c.cx;
    out_y = c.fy * iy / iz + c.cy;
    return iz > 0;
}

void make_projection_instructions(bool flip_after, float expand_val, float twist, const Joints21& joints,
                                  Projection& out, hand21_2d& out_hand) {
    out.flip = flip_after;
    Eigen::Vector3f dir = Eigen::Vector3f(joints.col(9)).normalized();
    out.rot_quat = direction(dir, twist);
    Eigen::Vector3f old_direction = dir;
    for (int iter = 0; iter < 8; iter++) {
        project_21_points_unscaled(joints, out.rot_quat, out_hand);
        xrt_vec2 mn = out_hand[0].pos_2d, mx = out_hand[0].pos_2d;
        for (int i = 0; i < 21; i++) {
            xrt_vec2 pt = out_hand[i].pos_2d;
            mn.x = std::fmin(pt.x, mn.x);
            mn.y = std::fmin(pt.y, mn.y);
            mx.x = std::fmax(pt.x, mx.x);
            mx.y = std::fmax(pt.y, mx.y);
        }
        xrt_vec2 center = {(mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f};
        out.stereographic_radius = std::fmax(center.x - mn.x, center.y - mn.y);
        Eigen::Vector3f new_direction = stereographic_unprojection(center.x, center.y);
        new_direction = out.rot_quat * new_direction;
        out.rot_quat = direction(new_direction, twist);
        if ((old_direction - new_direction).norm() < 0.0001f) break;
        old_direction = new_direction;
    }
    project_21_points_unscaled(joints, out.rot_quat, out_hand);
    float palm_l = palm_length(out_hand);
    float radius_around_palm = palm_l * 0.5f * (2.2f / 1.65f) * expand_val;
    out.stereographic_radius *= expand_val;
    out.stereographic_radius = std::fmax(out.stereographic_radius, radius_around_palm);
    for (int i = 0; i < 21; i++) {
        hm::vec2_5& v = out_hand[i];
        v.pos_2d.x = map_ranges<float>(v.pos_2d.x, -out.stereographic_radius, out.stereographic_radius, -1, 1);
        if (flip_after) v.pos_2d.x *= -1;
        v.pos_2d.y = map_ranges<float>(v.pos_2d.y, out.stereographic_radius, -out.stereographic_radius, -1, 1);
    }
    add_rel_depth(joints, out_hand);
}

void make_projection_instructions_angular(xrt_vec3 direction_3d, bool flip_after, float angular_radius,
                                          float expand_val, float twist, Projection& out) {
    out.flip = flip_after;
    xrt_vec3 imaginary = {0, std::sin(angular_radius), -std::cos(angular_radius)};
    out.stereographic_radius = imaginary.y / (1 - imaginary.z);
    Eigen::Vector3f dir(direction_3d.x, direction_3d.y, direction_3d.z);
    dir.normalize();
    out.rot_quat = direction(dir, twist);
    out.stereographic_radius *= expand_val;
}

void stereographic_project_image(const PinholeCamera& camera, const Projection& p, const GrayImage& image,
                                 uint8_t* out) {
    const Eigen::Quaternionf& q = p.rot_quat;
    const float r = p.stereographic_radius;
    for (int y = 0; y < wsize; y++) {
        const float sg_y = map_ranges<float>(float(y), 0.0f, float(wsize), r, -r);
        for (int x = 0; x < wsize; x++) {
            const float sg_x = p.flip ? map_ranges<float>(float(x), 0.0f, float(wsize), r, -r)
                                      : map_ranges<float>(float(x), 0.0f, float(wsize), -r, r);
            // Unnormalized stereographic direction, rotated by q (Mercury's expanded form).
            Eigen::Vector3f dir(sg_x + sg_x, sg_y + sg_y, sg_x * sg_x + sg_y * sg_y - 1);
            Eigen::Vector3f rot = q * dir;
            uint8_t value = 0;
            // Image axes: +y down, +z forward.
            const float ix = rot.x(), iy = -rot.y(), iz = -rot.z();
            if (iz > 0) {
                const float u = camera.fx * ix / iz + camera.cx;
                const float v = camera.fy * iy / iz + camera.cy;
                if (std::fabs(u) < 32767.0f && std::fabs(v) < 32767.0f) {
                    // Eigen's cast<int16_t> truncates toward zero.
                    const int xi = int(int16_t(u)), yi = int(int16_t(v));
                    if (xi >= 0 && yi >= 0 && xi < image.width && yi < image.height)
                        value = image.data[size_t(yi) * image.stride + size_t(xi)];
                }
            }
            out[y * wsize + x] = value;
        }
    }
}

bool decode_keypoints(const KeypointOutput& out, int hand_idx, float min_hand_probability,
                      hm::one_frame_one_view& view) {
    constexpr int plane = kHeatmapSize * kHeatmapSize;
    hm::MLOutput2D& px = view.keypoints_in_scaled_stereographic;
    for (int i = 0; i < 21; i++) {
        const float* data = &out.heatmap_xy[size_t(i) * plane];
        int idx = argmax(data, plane);
        int row = idx / kHeatmapSize, col = idx % kHeatmapSize;
        xrt_vec2 loc{};
        refine_center_of_distribution(data, col, row, kHeatmapSize, kHeatmapSize, &loc.x, &loc.y);
        px[i].pos_2d.x = hand_idx == 0 ? map_ranges<float>(loc.x, 0, 22, -1, 1) : map_ranges<float>(loc.x, 0, 22, 1, -1);
        px[i].pos_2d.y = map_ranges<float>(loc.y, 0, 22, 1, -1);
        px[i].confidence_xy = data[idx];
    }
    for (int j = 0; j < 21; j++) {
        float depth_row[22];
        std::copy_n(&out.heatmap_depth[size_t(j) * 22], 22, depth_row);
        float depth = 0, confidence = 0;
        if (hand_depth_center_of_mass(depth_row, &depth, &confidence)) {
            px[j].depth_relative_to_midpxm = depth;
            px[j].confidence_depth = confidence;
        } else {
            px[j].depth_relative_to_midpxm = 0;
            px[j].confidence_depth = 0;
        }
    }
    const float is_hand_explicit = 1.0f / (1.0f + std::exp(-out.scalar_extras[0]));
    for (int i = 0; i < 5; i++) {
        view.curls[i].value = out.curls[size_t(i)];
        view.curls[i].variance = std::fabs(out.curls[size_t(5 + i)]) + 0.01f;
    }
    return is_hand_explicit >= min_hand_probability;
}

void decode_detection(const DetectionOutput& out, const Affine& go_back, float min_confidence,
                      std::array<Region, 2>& regions) {
    for (int h = 0; h < 2; h++) {
        Region& r = regions[size_t(h)];
        r = {};
        r.found = out.hand_exists[size_t(h)] > min_confidence;
        if (!r.found) continue;
        r.confidence = out.hand_exists[size_t(h)];
        xrt_vec2 pt = {map_ranges<float>(out.cx[size_t(h)], -1, 1, 0, kDetectionSize),
                       map_ranges<float>(out.cy[size_t(h)], -1, 1, 0, kDetectionSize)};
        float size = out.size[size_t(h)] * kDetectionSize * 2.0f;
        size *= vec2_len({go_back[0], go_back[1]});
        r.center_px = apply(go_back, pt);
        r.size_px = size;
    }
}

}  // namespace mercury::detail
