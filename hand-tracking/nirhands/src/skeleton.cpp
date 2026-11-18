#include "nirhands/skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nirhands {
namespace {

constexpr int kPalmJoints[] = {0, 1, 5, 9, 13, 17};
constexpr int kChains[5][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}};

Vec3 safe_unit(const Vec3& v) {
    const double n = norm(v);
    if (!(n > 1e-12) || !std::isfinite(n)) throw std::runtime_error("degenerate skeleton frame");
    return (1.0 / n) * v;
}

Mat3 local_rotation(double flex, double side) {
    const double cf = std::cos(flex), sf = std::sin(flex);
    const double cs = std::cos(side), ss = std::sin(side);
    const Mat3 ra{{cf, 0, sf, 0, 1, 0, -sf, 0, cf}};
    const Mat3 rc{{cs, -ss, 0, ss, cs, 0, 0, 0, 1}};
    return ra * rc;
}

Joints forward_kinematics(const SkeletonProfile& mean_profile, const float* angles, float scale) {
    std::array<Vec3, 6> palm;
    for (int p = 0; p < 6; ++p)
        for (int k = 0; k < 3; ++k) palm[p][k] = double(mean_profile[3 * p + k]) * scale;
    Joints out{};
    for (int p = 0; p < 6; ++p) out[kPalmJoints[p]] = palm[p];
    for (int digit = 0; digit < 5; ++digit) {
        const Vec3 d = safe_unit(palm[digit + 1]);
        const Vec3 a = safe_unit(cross(Vec3{0, 0, 1}, d));
        const Vec3 c = cross(d, a);
        Mat3 frame{{d[0], a[0], c[0], d[1], a[1], c[1], d[2], a[2], c[2]}};
        Vec3 pos = out[kChains[digit][0]];
        for (int i = 0; i < 3; ++i) {
            const int bone = 3 * digit + i;
            frame = frame * local_rotation(angles[2 * bone], angles[2 * bone + 1]);
            const double length = double(mean_profile[18 + bone]) * scale;
            pos = pos + length * Vec3{frame.m[0], frame.m[3], frame.m[6]};
            out[kChains[digit][i + 1]] = pos;
        }
    }
    return out;
}

Mat3 quaternion_rotation(const std::array<std::array<double, 4>, 4>& input) {
    auto a = input;
    std::array<std::array<double, 4>, 4> vectors{};
    for (int i = 0; i < 4; ++i) vectors[i][i] = 1;
    for (int iteration = 0; iteration < 64; ++iteration) {
        int p = 0, q = 1;
        double largest = std::fabs(a[p][q]);
        for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) {
            if (std::fabs(a[i][j]) > largest) { largest = std::fabs(a[i][j]); p = i; q = j; }
        }
        if (largest < 1e-18) break;
        const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const double t = (theta < 0 ? -1.0 : 1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 4; ++k) if (k != p && k != q) {
            const double akp = a[k][p], akq = a[k][q];
            a[k][p] = a[p][k] = c * akp - s * akq;
            a[k][q] = a[q][k] = s * akp + c * akq;
        }
        const double app = a[p][p], aqq = a[q][q], apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = a[q][p] = 0;
        for (int k = 0; k < 4; ++k) {
            const double vkp = vectors[k][p], vkq = vectors[k][q];
            vectors[k][p] = c * vkp - s * vkq;
            vectors[k][q] = s * vkp + c * vkq;
        }
    }
    int best = 0;
    for (int i = 1; i < 4; ++i) if (a[i][i] > a[best][best]) best = i;
    std::array<double, 4> q{};
    double qnorm = 0;
    for (int i = 0; i < 4; ++i) { q[i] = vectors[i][best]; qnorm += q[i] * q[i]; }
    qnorm = std::sqrt(qnorm);
    if (!(qnorm > 0) || !std::isfinite(qnorm)) throw std::runtime_error("invalid skeleton Procrustes rotation");
    for (double& v : q) v /= qnorm;
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    return {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
             2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
}

Mat3 fit_palm(const Joints& local, const std::array<Vec3, 6>& target, Vec3& translation) {
    Vec3 sc{}, tc{};
    for (int p = 0; p < 6; ++p) {
        sc = sc + local[kPalmJoints[p]];
        tc = tc + target[p];
    }
    sc = (1.0 / 6) * sc;
    tc = (1.0 / 6) * tc;
    std::array<std::array<double, 3>, 3> h{};
    double energy = 0;
    for (int p = 0; p < 6; ++p) {
        const Vec3 s = local[kPalmJoints[p]] - sc, t = target[p] - tc;
        energy += dot(s, s) + dot(t, t);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) h[i][j] += s[i] * t[j];
    }
    if (!(energy > 1e-18) || !std::isfinite(energy)) throw std::runtime_error("degenerate skeleton palm");
    const double xx = h[0][0], xy = h[0][1], xz = h[0][2];
    const double yx = h[1][0], yy = h[1][1], yz = h[1][2];
    const double zx = h[2][0], zy = h[2][1], zz = h[2][2];
    const std::array<std::array<double, 4>, 4> n{{
        {{xx + yy + zz, yz - zy, zx - xz, xy - yx}},
        {{yz - zy, xx - yy - zz, xy + yx, zx + xz}},
        {{zx - xz, xy + yx, -xx + yy - zz, yz + zy}},
        {{xy - yx, zx + xz, yz + zy, -xx - yy + zz}}
    }};
    const Mat3 rotation = quaternion_rotation(n);
    translation = tc - rotation * sc;
    return rotation;
}

Vec3 decode_root(const Geometry& g, const float* keypoints, const float* disparity) {
    const int size = g.camera.size;
    double u = keypoints[2 * kRoot] * (size / 2.0) + size / 2.0;
    const double v = keypoints[2 * kRoot + 1] * (size / 2.0) + size / 2.0;
    if (g.mirrored) u = (size - 1) - u;
    const double dc = disparity[kRoot] * (size / 8.0);
    const double z = 1.0 / (dc / (g.camera.f * g.baseline_m) + 1.0 / g.center_depth_m);
    double x = (u + g.camera.offset[0]) * z / g.camera.f;
    const double y = (v + g.camera.offset[1]) * z / g.camera.f;
    if (g.mirrored) x = -x;
    return {x, y, z};
}

}  // namespace

void decode_skeleton(const Geometry& g, const SkeletonProfile& mean_profile,
                     const float* keypoints_hm, const float* disparity_hm,
                     const float* palm_points, const float* angles, float scale,
                     float* keypoints, float* disparity) {
    const Joints local = forward_kinematics(mean_profile, angles, scale);
    std::array<Vec3, 6> palm;
    const Vec3 root{palm_points[9], palm_points[10], palm_points[11]};
    for (int p = 0; p < 6; ++p) {
        const Vec3 point{palm_points[3 * p], palm_points[3 * p + 1], palm_points[3 * p + 2]};
        palm[p] = point - root;
    }
    Vec3 translation;
    const Mat3 rotation = fit_palm(local, palm, translation);
    Joints joints;
    for (int j = 0; j < kJoints; ++j) joints[j] = rotation * local[j] + translation;
    const Vec3 fitted_root = joints[kRoot];
    const Vec3 crop_root = decode_root(g, keypoints_hm, disparity_hm);
    for (int j = 0; j < kJoints; ++j) {
        const Vec3 point = joints[j] - fitted_root + crop_root;
        const double x = g.mirrored ? -point[0] : point[0];
        const double z = std::max(0.02, point[2]);
        double u = g.camera.f * x / z - g.camera.offset[0];
        const double v = g.camera.f * point[1] / z - g.camera.offset[1];
        if (g.mirrored) u = (g.camera.size - 1) - u;
        keypoints[2 * j] = float((u - g.camera.size / 2.0) / (g.camera.size / 2.0));
        keypoints[2 * j + 1] = float((v - g.camera.size / 2.0) / (g.camera.size / 2.0));
        disparity[j] = float(g.camera.f * g.baseline_m * (1.0 / z - 1.0 / g.center_depth_m) /
                             (g.camera.size / 8.0));
    }
}

// ------------------------------------------------------------------------------------------- combined output

namespace {

// _unit in skeleton.py: clamp instead of throwing (the combined output must never take the pipeline down).
Vec3 unit_clamped(const Vec3& v) { return (1.0 / std::max(norm(v), 1e-9)) * v; }

// palm_frame: rows x (wrist -> middle MCP), y, z (palm normal).
Mat3 palm_axes(const Joints& j) {
    const Vec3 w = j[0];
    const Vec3 x = unit_clamped(j[9] - w);
    const Vec3 n = cross(j[5] - w, j[17] - w);
    const Vec3 z = unit_clamped(n - dot(n, x) * x);
    const Vec3 y = cross(z, x);
    return Mat3::rows(x, y, z);
}

// _rest_frames: rest frame of a digit's first bone, columns (d, a, c), in palm coordinates.
Mat3 rest_frame(const SkeletonProfile& profile, int digit) {
    const Vec3 base{profile[3 * (digit + 1)], profile[3 * (digit + 1) + 1], profile[3 * (digit + 1) + 2]};
    const Vec3 d = unit_clamped(base);
    const Vec3 a = unit_clamped(cross(Vec3{0, 0, 1}, d));
    const Vec3 c = cross(d, a);
    return {{d[0], a[0], c[0], d[1], a[1], c[1], d[2], a[2], c[2]}};
}

Joints mirrored(const Joints& j, bool mirror) {
    Joints out = j;
    if (mirror) for (auto& p : out) p[0] = -p[0];
    return out;
}

bool finite(const Joints& j) {
    for (const auto& p : j) for (double v : p) if (!std::isfinite(v)) return false;
    return true;
}

}  // namespace

SkeletonProfile measure_profile(const Joints& joints) {
    const Mat3 axes = palm_axes(joints);
    SkeletonProfile out{};
    for (int p = 0; p < 6; ++p) {
        const Vec3 local = axes * (joints[kPalmJoints[p]] - joints[0]);
        for (int k = 0; k < 3; ++k) out[3 * p + k] = float(local[k]);
    }
    for (int digit = 0; digit < 5; ++digit)
        for (int i = 0; i < 3; ++i)
            out[18 + 3 * digit + i] = float(norm(joints[kChains[digit][i + 1]] - joints[kChains[digit][i]]));
    return out;
}

SkeletonAngles inverse_kinematics(const Joints& joints, const SkeletonProfile& profile) {
    const Mat3 axes = palm_axes(joints);
    Joints local;
    for (int j = 0; j < kJoints; ++j) local[j] = axes * (joints[j] - joints[0]);
    SkeletonAngles angles{};
    for (int digit = 0; digit < 5; ++digit) {
        Mat3 frame = rest_frame(profile, digit);
        for (int i = 0; i < 3; ++i) {
            const Vec3 v = unit_clamped(local[kChains[digit][i + 1]] - local[kChains[digit][i]]);
            const Vec3 vl = frame.tmul(v);  // v in (d, a, c) coordinates
            const double side = std::asin(std::clamp(vl[1], -1.0, 1.0));
            const double flex = std::atan2(-vl[2], vl[0]);
            angles[2 * (3 * digit + i)] = flex;
            angles[2 * (3 * digit + i) + 1] = side;
            frame = frame * local_rotation(flex, side);
        }
    }
    return angles;
}

Joints forward_kinematics_profile(const SkeletonProfile& profile, const SkeletonAngles& angles) {
    Joints out{};
    for (int p = 0; p < 6; ++p) out[kPalmJoints[p]] = {profile[3 * p], profile[3 * p + 1], profile[3 * p + 2]};
    for (int digit = 0; digit < 5; ++digit) {
        Mat3 frame = rest_frame(profile, digit);
        Vec3 pos = out[kChains[digit][0]];
        for (int i = 0; i < 3; ++i) {
            const int bone = 3 * digit + i;
            frame = frame * local_rotation(angles[2 * bone], angles[2 * bone + 1]);
            pos = pos + double(profile[18 + bone]) * Vec3{frame.m[0], frame.m[3], frame.m[6]};
            out[kChains[digit][i + 1]] = pos;
        }
    }
    return out;
}

bool combine_skeleton(const Joints& skeleton, const Joints& heatmap, bool is_right, const SkeletonProfile& profile,
                      double heatmap_weight, Joints& out) {
    if (!finite(skeleton) || !finite(heatmap)) return false;
    const Joints sk = mirrored(skeleton, is_right), hm = mirrored(heatmap, is_right);
    const SkeletonAngles a_sk = inverse_kinematics(sk, profile), a_hm = inverse_kinematics(hm, profile);
    SkeletonAngles angles;
    for (int k = 0; k < kSkeletonAngles; ++k) angles[k] = (1 - heatmap_weight) * a_sk[k] + heatmap_weight * a_hm[k];
    const Joints local = forward_kinematics_profile(profile, angles);
    std::array<Vec3, 6> palm;
    for (int p = 0; p < 6; ++p) palm[p] = hm[kPalmJoints[p]];
    Joints posed;
    try {
        Vec3 translation;
        const Mat3 rotation = fit_palm(local, palm, translation);
        for (int j = 0; j < kJoints; ++j) posed[j] = rotation * local[j] + translation;
    } catch (const std::runtime_error&) {
        return false;
    }
    posed = mirrored(posed, is_right);
    if (!finite(posed)) return false;
    out = posed;
    return true;
}

void ProfileEstimator::add(const Joints& heatmap_joints, bool is_right) {
    if (!finite(heatmap_joints)) return;
    const SkeletonProfile p = measure_profile(mirrored(heatmap_joints, is_right));
    for (float v : p) if (!std::isfinite(v)) return;
    auto& h = history_[is_right];
    h.push_back(p);
    while (int(h.size()) > window_) h.pop_front();
    dirty_[is_right] = true;
}

bool ProfileEstimator::profile(bool is_right, SkeletonProfile& out) {
    const auto& h = history_[is_right];
    if (int(h.size()) < min_samples_) return false;
    if (dirty_[is_right]) {
        std::vector<float> column(h.size());
        for (int k = 0; k < kSkeletonProfileValues; ++k) {
            for (size_t i = 0; i < h.size(); ++i) column[i] = h[i][k];
            const size_t n = column.size();
            std::nth_element(column.begin(), column.begin() + n / 2, column.end());
            float med = column[n / 2];
            if (n % 2 == 0) {
                const float lower = *std::max_element(column.begin(), column.begin() + n / 2);
                med = 0.5f * (med + lower);
            }
            cached_[is_right][k] = med;
        }
        dirty_[is_right] = false;
    }
    out = cached_[is_right];
    return true;
}

void ProfileEstimator::reset() {
    for (auto& h : history_) h.clear();
    dirty_ = {true, true};
}

}  // namespace nirhands
