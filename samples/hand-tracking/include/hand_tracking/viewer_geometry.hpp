#pragma once
#include "hand_tracking/tracker.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace hand_tracking::viewer {
// MediaPipe hand topology: wrist, thumb, index, middle, ring, pinky, palm.
inline constexpr int bones[21][2] = {{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},
    {11,12},{9,13},{13,14},{14,15},{15,16},{13,17},{0,17},{17,18},{18,19},{19,20}};

// OpenXR / Mercury 26-joint skeletal topology: wrist (1) connects to palm (0)
// and each finger's metacarpal (2, 6, 11, 16, 21), followed by distal segments to tips.
inline constexpr int mercury_bones[25][2] = {
    {1, 0},
    {1, 2}, {2, 3}, {3, 4}, {4, 5},
    {1, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10},
    {1, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 15},
    {1, 16}, {16, 17}, {17, 18}, {18, 19}, {19, 20},
    {1, 21}, {21, 22}, {22, 23}, {23, 24}, {24, 25}
};

// Rotate vector v by quaternion q = (x, y, z, w).
inline std::array<float, 3> rotate(const std::array<float, 4>& q, const std::array<float, 3>& v) {
    const std::array<float, 3> t{
        2.0f * (q[1] * v[2] - q[2] * v[1]),
        2.0f * (q[2] * v[0] - q[0] * v[2]),
        2.0f * (q[0] * v[1] - q[1] * v[0])
    };
    return {
        v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1],
        v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
        v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0]
    };
}

// Joints in rectified pinhole camera coordinates (meters; x right, y down,
// z forward). The model-relative metric hand shape is translated so that it
// projects onto the image landmarks in the least-squares sense. This is a
// monocular estimate: depth scales with the model's assumed hand size.
// Returns false when the fit is degenerate or nonfinite.
inline bool camera_joints(const Hand& hand, float width, float height, float focal, float cx, float cy,
                          std::array<Point,21>& out, float min_z = .12f, float max_z = 1.5f) {
    // For each joint i with ray (u, v): tx - u*tz = u*Wz - Wx, ty - v*tz = v*Wz - Wy.
    // Eliminate tx/ty through their means, solve tz, then back-substitute.
    double su=0,sv=0,sa=0,sb=0,suu=0,sua=0;
    std::array<double,21> u{}, v{};
    for (size_t i=0;i<21;++i) {
        const auto p=hand.model_world_landmarks[i];
        u[i]=(hand.landmarks[i].x*width-cx)/focal;
        v[i]=(hand.landmarks[i].y*height-cy)/focal;
        const double a=p.x-u[i]*p.z, b=p.y-v[i]*p.z;
        if (!std::isfinite(u[i]+v[i]+a+b)) return false;
        su+=u[i]; sv+=v[i]; sa+=a; sb+=b; suu+=u[i]*u[i]+v[i]*v[i]; sua+=u[i]*a+v[i]*b;
    }
    const double denominator=suu-(su*su+sv*sv)/21;
    if (!(denominator>1e-8)) return false;
    const double tz=std::clamp((sua-(su*sa+sv*sb)/21)/denominator,double(min_z),double(max_z));
    const double tx=(su*tz-sa)/21, ty=(sv*tz-sb)/21;
    if (!std::isfinite(tx+ty+tz)) return false;
    for (size_t i=0;i<21;++i) {
        const auto p=hand.model_world_landmarks[i];
        out[i]={float(p.x+tx),float(p.y+ty),float(p.z+tz)};
    }
    return true;
}
} // namespace hand_tracking::viewer
