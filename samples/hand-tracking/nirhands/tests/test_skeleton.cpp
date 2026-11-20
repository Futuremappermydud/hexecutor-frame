// Independent checks for nirhands::decode_skeleton (src/skeleton.cpp).
// Expected values are computed here from the written spec with separate helpers: own Rodrigues
// rotation, zero-angle forward kinematics, root encode (inverse of the root decode), re-encode
// (crop projection) and un-projection. Nothing from skeleton.cpp is reused.
#include "nirhands/skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace {

using nirhands::Geometry;
using nirhands::SkeletonProfile;

using Pt = std::array<double, 3>;
using M3 = std::array<double, 9>;  // row-major

constexpr double kPi = 3.14159265358979323846;
constexpr double kTolCrop = 2e-5;  // keypoints and disparity (float32 outputs)
constexpr double kTol3d = 1e-4;    // metres, un-projected checks
constexpr int kRootJoint = 9;
constexpr int kPalmJoint[6] = {0, 1, 5, 9, 13, 17};
constexpr int kChain[5][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}};

// Synthetic profile: 6 rest palm points for joints {0,1,5,9,13,17}, then 15 bone lengths.
constexpr float kRest[6][3] = {
    {0.00f, 0.00f, 0.000f},
    {0.02f, -0.02f, -0.015f},
    {0.08f, -0.02f, -0.005f},
    {0.09f, 0.00f, 0.000f},
    {0.08f, 0.02f, 0.002f},
    {0.07f, 0.04f, -0.004f},
};
constexpr float kLen[15] = {0.0240f, 0.0212f, 0.0298f, 0.0255f, 0.0221f, 0.0333f, 0.0268f, 0.0307f,
                            0.0229f, 0.0276f, 0.0352f, 0.0215f, 0.0319f, 0.0263f, 0.0242f};

int g_failures = 0;

bool check_true(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
    return ok;
}

bool check_near(const char* tag, const char* field, int idx, double observed, double expected, double tol) {
    const bool ok = std::isfinite(observed) && std::fabs(observed - expected) <= tol;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL [%s] %s[%d]: observed %.9g expected %.9g |diff| %.3g > tol %.1g\n", tag, field, idx,
                    observed, expected, std::fabs(observed - expected), tol);
    }
    return ok;
}

// ---- small independent vector / matrix helpers ----
Pt add(const Pt& a, const Pt& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Pt sub(const Pt& a, const Pt& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Pt scl(double s, const Pt& a) { return {s * a[0], s * a[1], s * a[2]}; }
double len3(const Pt& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
double dist(const Pt& a, const Pt& b) { return len3(sub(a, b)); }
Pt unit3(const Pt& a) { return scl(1.0 / len3(a), a); }
Pt mul(const M3& m, const Pt& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}
M3 identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }

// Rodrigues: R = cos(a) I + (1 - cos(a)) n n^T + sin(a) [n]_x, n = axis normalised.
M3 rodrigues(Pt axis, double angle) {
    const double n = len3(axis);
    const double x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
    const double c = std::cos(angle), s = std::sin(angle), C = 1 - c;
    return {c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C};
}

SkeletonProfile make_profile() {
    SkeletonProfile p{};
    for (int k = 0; k < 6; ++k)
        for (int c = 0; c < 3; ++c) p[3 * k + c] = kRest[k][c];
    for (int b = 0; b < 15; ++b) p[18 + b] = kLen[b];
    return p;
}
Pt rest_point(const SkeletonProfile& p, int k) { return {double(p[3 * k]), double(p[3 * k + 1]), double(p[3 * k + 2])}; }
double bone_len(const SkeletonProfile& p, int b) { return double(p[18 + b]); }

Geometry make_geometry(bool mirrored) {
    Geometry g;
    g.camera.center = {0.0, 0.0, 0.0};
    g.camera.R = nirhands::Mat3{};  // identity
    g.camera.f = 310.0;
    g.camera.offset = {-70.5, -58.25};
    g.camera.size = 128;
    g.baseline_m = 0.0642;
    g.center_depth_m = 0.37;
    g.primary_view = 0;
    g.mirrored = mirrored;
    g.stereo = true;
    return g;
}

// Spec item 5: model-frame point -> crop keypoint (kx, ky) and disparity d, with z clamp.
void project(const Geometry& g, const Pt& P, double& kx, double& ky, double& d) {
    const double S = g.camera.size, f = g.camera.f, ox = g.camera.offset[0], oy = g.camera.offset[1];
    const double B = g.baseline_m, Zc = g.center_depth_m;
    const double xc = g.mirrored ? -P[0] : P[0];
    const double z = std::max(0.02, P[2]);
    double u = f * xc / z - ox;
    const double v = f * P[1] / z - oy;
    if (g.mirrored) u = (S - 1) - u;
    kx = (u - S / 2) / (S / 2);
    ky = (v - S / 2) / (S / 2);
    d = f * B * (1.0 / z - 1.0 / Zc) / (S / 8);
}

// Inverse of spec item 4 (root decode), no clamp: root X -> kx, ky, d.
void encode_root(const Geometry& g, const Pt& X, double& kx, double& ky, double& d) {
    const double S = g.camera.size, f = g.camera.f, ox = g.camera.offset[0], oy = g.camera.offset[1];
    const double B = g.baseline_m, Zc = g.center_depth_m;
    const double xc = g.mirrored ? -X[0] : X[0];  // decode: x = -(...) when mirrored
    double u = f * xc / X[2] - ox;                // u after the decode's mirror step
    if (g.mirrored) u = (S - 1) - u;              // undo: pre-mirror value
    const double v = f * X[1] / X[2] - oy;
    kx = (u - S / 2) / (S / 2);
    ky = (v - S / 2) / (S / 2);
    d = f * B * (1.0 / X[2] - 1.0 / Zc) / (S / 8);
}

// Inverse of spec item 5: crop keypoint and disparity -> 3D point (no clamp).
Pt unproject(const Geometry& g, double kx, double ky, double d) {
    const double S = g.camera.size, f = g.camera.f, ox = g.camera.offset[0], oy = g.camera.offset[1];
    const double B = g.baseline_m, Zc = g.center_depth_m;
    double u = kx * (S / 2) + S / 2;
    const double v = ky * (S / 2) + S / 2;
    if (g.mirrored) u = (S - 1) - u;
    const double z = 1.0 / (d * (S / 8) / (f * B) + 1.0 / Zc);
    const double xc = (u + ox) * z / f;
    const double y = (v + oy) * z / f;
    return {g.mirrored ? -xc : xc, y, z};
}

// Forward kinematics at zero angles, written from spec item 2.
void fk_zero(const SkeletonProfile& p, double scale, Pt* L) {
    for (int k = 0; k < 6; ++k) L[kPalmJoint[k]] = scl(scale, rest_point(p, k));
    for (int digit = 0; digit < 5; ++digit) {
        const Pt base = L[kChain[digit][0]];
        const Pt d = unit3(rest_point(p, digit + 1));
        double acc = 0;
        for (int i = 0; i < 3; ++i) {
            acc += scale * bone_len(p, 3 * digit + i);
            L[kChain[digit][i + 1]] = add(base, scl(acc, d));
        }
    }
}

struct CaseSpec {
    const char* name;
    M3 R;           // palm rotation used to build palm_points
    Pt t;           // translation added to palm_points
    float scale;
    bool mirrored;
    Pt X;           // root joint position in model frame
    bool expect_clamp;
};

struct Run {
    float kp[42];
    float disp[21];
};

Run decode_case(const CaseSpec& c, const float* angles) {
    const SkeletonProfile prof = make_profile();
    const Geometry g = make_geometry(c.mirrored);
    float palm[18];
    for (int p = 0; p < 6; ++p) {
        const Pt q = add(mul(c.R, scl(c.scale, rest_point(prof, p))), c.t);
        for (int k = 0; k < 3; ++k) palm[3 * p + k] = float(q[k]);
    }
    // Aux crop outputs: garbage everywhere except the encoded root at joint 9.
    float kh[42], dh[21];
    for (int j = 0; j < 21; ++j) {
        kh[2 * j] = kh[2 * j + 1] = float(0.37 * j);
        dh[j] = float(0.37 * j);
    }
    double kx, ky, d;
    encode_root(g, c.X, kx, ky, d);
    kh[2 * kRootJoint] = float(kx);
    kh[2 * kRootJoint + 1] = float(ky);
    dh[kRootJoint] = float(d);

    Run r{};
    nirhands::decode_skeleton(g, prof, kh, dh, palm, angles, c.scale, r.kp, r.disp);
    return r;
}

void check_finite(const char* tag, const Run& r) {
    for (int i = 0; i < 42; ++i) check_true(std::isfinite(r.kp[i]), "keypoint not finite");
    for (int j = 0; j < 21; ++j) check_true(std::isfinite(r.disp[j]), "disparity not finite");
    (void)tag;
}

// Expected outputs from spec items 2, 4, 5 for the given palm rotation and root.
void check_expected(const CaseSpec& c, const Run& r) {
    const SkeletonProfile prof = make_profile();
    const Geometry g = make_geometry(c.mirrored);
    const double s = c.scale;
    Pt L[21];
    fk_zero(prof, s, L);
    int clamped = 0;
    for (int j = 0; j < 21; ++j) {
        const Pt P = add(mul(c.R, sub(L[j], L[kRootJoint])), c.X);
        if (P[2] < 0.02) ++clamped;
        double ekx, eky, ed;
        project(g, P, ekx, eky, ed);
        check_near(c.name, "keypoints", 2 * j, r.kp[2 * j], ekx, kTolCrop);
        check_near(c.name, "keypoints", 2 * j + 1, r.kp[2 * j + 1], eky, kTolCrop);
        check_near(c.name, "disparity", j, r.disp[j], ed, kTolCrop);
    }
    if (c.expect_clamp) check_true(clamped > 0, "clamp case: no expected joint has pz < 0.02");
    std::printf("case %s: %d of 21 expected joints have pz < 0.02 (clamped)\n", c.name, clamped);
}

// Case b, zero angles: middle finger length, measured in 3D from joints 9 and 12.
void check_zero_angle_3d(const CaseSpec& c, const Run& r) {
    const Geometry g = make_geometry(c.mirrored);
    const SkeletonProfile prof = make_profile();
    const Pt P9 = unproject(g, r.kp[2 * 9], r.kp[2 * 9 + 1], r.disp[9]);
    const Pt P12 = unproject(g, r.kp[2 * 12], r.kp[2 * 12 + 1], r.disp[12]);
    const double expect = double(c.scale) * (bone_len(prof, 6) + bone_len(prof, 7) + bone_len(prof, 8));
    check_near(c.name, "3D middle finger length", 0, dist(P9, P12), expect, kTol3d);
    for (int k = 0; k < 3; ++k) check_near(c.name, "3D root (joint 9)", k, P9[k], c.X[k], kTol3d);
}

// Case b, non-zero angles: bone lengths and palm distances after un-projection.
void check_nonzero_angles(const CaseSpec& c, const Run& r) {
    const Geometry g = make_geometry(c.mirrored);
    const SkeletonProfile prof = make_profile();
    const double s = c.scale;
    Pt P[21];
    for (int j = 0; j < 21; ++j) {
        P[j] = unproject(g, r.kp[2 * j], r.kp[2 * j + 1], r.disp[j]);
        check_true(P[j][2] > 0.02, "unprojected z unexpectedly clamped");
    }
    for (int digit = 0; digit < 5; ++digit) {
        for (int i = 0; i < 3; ++i) {
            const int bone = 3 * digit + i;
            check_near("nonzero-angles", "bone length", bone, dist(P[kChain[digit][i + 1]], P[kChain[digit][i]]),
                       s * bone_len(prof, bone), kTol3d);
        }
    }
    for (int p = 0; p < 6; ++p)
        for (int q = p + 1; q < 6; ++q)
            check_near("nonzero-angles", "palm pair distance", 6 * p + q,
                       dist(P[kPalmJoint[p]], P[kPalmJoint[q]]),
                       s * dist(rest_point(prof, p), rest_point(prof, q)), kTol3d);
}

}  // namespace

int main() {
    const float zero[30] = {};
    const CaseSpec cases[] = {
        {"a", identity(), {0.0, 0.0, 0.0}, 1.0f, false, {0.01, -0.02, 0.40}, false},
        {"b", rodrigues({1, 2, 3}, 0.9), {0.3, -0.2, 0.1}, 1.15f, false, {-0.03, 0.02, 0.35}, false},
        {"c", rodrigues({-2, 1, 0.5}, 2.4), {0.3, -0.2, 0.1}, 0.85f, true, {0.04, 0.01, 0.30}, false},
        {"d", rodrigues({0.3, -1, 0.2}, 3.1), {-0.1, 0.05, 0.2}, 1.0f, false, {0.02, 0.01, 0.36}, false},
        // 90 degrees about y maps the +x finger direction to -z (toward the camera).
        {"e", rodrigues({0, 1, 0}, kPi / 2), {0.0, 0.0, 0.0}, 1.0f, false, {0.005, -0.01, 0.03}, true},
    };

    for (const CaseSpec& c : cases) {
        const int before = g_failures;
        const Run r = decode_case(c, zero);
        check_finite(c.name, r);
        check_expected(c, r);
        if (g_failures == before) std::printf("case %s: all 63 values within tolerance\n", c.name);
    }

    const CaseSpec& caseB = cases[1];
    check_zero_angle_3d(caseB, decode_case(caseB, zero));

    float angles[30];
    for (int b = 0; b < 15; ++b) {
        angles[2 * b] = 0.3f;
        angles[2 * b + 1] = 0.1f;
    }
    const Run rn = decode_case(caseB, angles);
    check_finite("nonzero-angles", rn);
    check_nonzero_angles(caseB, rn);

    if (g_failures != 0) {
        std::printf("test_skeleton: %d failing check(s)\n", g_failures);
        return 1;
    }
    std::printf("test_skeleton: ok\n");
    return 0;
}
