// Geometry for the native stereo NIR hand pipeline: small vector types, the Kannala-Brandt
// slam camera, and the factory rig. Mirrors stereo_nir_hands (camera/calibration.py,
// frametop/session.py) so native and Python results agree to floating-point precision.
#pragma once
#include <array>
#include <cmath>
#include <string>

namespace nirhands {

using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 operator*(double s, const Vec3& a) { return {s * a[0], s * a[1], s * a[2]}; }
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 unit(const Vec3& a) { return (1.0 / norm(a)) * a; }

// Row-major 3x3.
struct Mat3 {
    std::array<double, 9> m{1, 0, 0, 0, 1, 0, 0, 0, 1};
    static Mat3 rows(const Vec3& a, const Vec3& b, const Vec3& c) {
        return {{a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]}};
    }
    Vec3 row(int r) const { return {m[3 * r], m[3 * r + 1], m[3 * r + 2]}; }
    Vec3 operator*(const Vec3& v) const { return {dot(row(0), v), dot(row(1), v), dot(row(2), v)}; }
    // Transpose times v (v @ M in numpy).
    Vec3 tmul(const Vec3& v) const {
        return {m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
                m[2] * v[0] + m[5] * v[1] + m[8] * v[2]};
    }
    Mat3 transposed() const { return {{m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]}}; }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[3 * i + j] = m[3 * i] * o.m[j] + m[3 * i + 1] * o.m[3 + j] + m[3 * i + 2] * o.m[6 + j];
        return r;
    }
};

// Rigid transform: p_parent = R * p_child + t.
struct Pose {
    Mat3 R;
    Vec3 t{0, 0, 0};
    Vec3 operator*(const Vec3& p) const { return R * p + t; }
    Pose operator*(const Pose& o) const { return {R * o.R, R * o.t + t}; }
    Pose inverse() const { const Mat3 Rt = R.transposed(); return {Rt, -1.0 * (Rt * t)}; }
};

// Kannala-Brandt fisheye camera; world -> camera is R_cw * p + t_cw.
struct KbCamera {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::array<double, 4> k{};
    Mat3 R_cw;
    Vec3 t_cw{0, 0, 0};
    int width = 1056, height = 1024;

    Vec3 to_camera(const Vec3& world) const { return R_cw * world + t_cw; }
    Vec3 center() const { return -1.0 * R_cw.tmul(t_cw); }
    double distort(double theta) const {
        const double t2 = theta * theta;
        return theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
    }
    Vec2 project_camera(const Vec3& p) const {
        const double r = std::hypot(p[0], p[1]);
        const double theta = std::atan2(r, p[2]);
        const double scale = r > 1e-12 ? distort(theta) / r : 0.0;
        return {fx * p[0] * scale + cx, fy * p[1] * scale + cy};
    }
    Vec2 project(const Vec3& world) const { return project_camera(to_camera(world)); }
    // Unit bearing in camera coordinates (8 Newton steps, as unproject_kb).
    Vec3 unproject(const Vec2& uv) const;
};

struct Rig {
    KbCamera left, right;  // slam_left (video13), slam_right (video9); world = OpenVR head frame
};

// Rig from the headset's factory files (/persist/xrservice.json, /persist/device_config.json)
// or a Frametop session's calibration.json/device.json (same schema). The world frame is the
// OpenVR head frame (+x right, +y up, -z forward), so results go straight to the viewer.
Rig load_rig(const std::string& xrservice_json_path, const std::string& device_config_json_path);

}  // namespace nirhands
