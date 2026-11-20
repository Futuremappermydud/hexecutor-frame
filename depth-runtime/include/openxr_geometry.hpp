#pragma once
#include <openxr/openxr.h>
#include <cmath>
namespace FrameSpatialCloud {
struct Mat4 {
    float m[16]{};
};

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                result.m[column * 4 + row] +=
                    a.m[k * 4 + row] * b.m[column * 4 + k];
            }
        }
    }
    return result;
}

inline Mat4 viewFromPose(const XrPosef& pose) {
    const auto& q = pose.orientation;
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float xw = q.x * q.w;
    const float yw = q.y * q.w;
    const float zw = q.z * q.w;
    const float r[3][3] = {
        {1.0f - 2.0f * (yy + zz), 2.0f * (xy - zw),
         2.0f * (xz + yw)},
        {2.0f * (xy + zw), 1.0f - 2.0f * (xx + zz),
         2.0f * (yz - xw)},
        {2.0f * (xz - yw), 2.0f * (yz + xw),
         1.0f - 2.0f * (xx + yy)},
    };
    Mat4 view{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            view.m[column * 4 + row] = r[column][row];
        }
    }
    const float p[3]{pose.position.x, pose.position.y, pose.position.z};
    for (int row = 0; row < 3; ++row) {
        view.m[12 + row] = -(r[0][row] * p[0] + r[1][row] * p[1] +
                             r[2][row] * p[2]);
    }
    view.m[15] = 1.0f;
    return view;
}

inline Mat4 projectionFromFov(const XrFovf& fov, float nearZ, float farZ) {
    const float left = std::tan(fov.angleLeft);
    const float right = std::tan(fov.angleRight);
    const float down = std::tan(fov.angleDown);
    const float up = std::tan(fov.angleUp);
    const float width = right - left;
    const float height = up - down;
    Mat4 projection{};
    projection.m[0] = 2.0f / width;
    // Vulkan framebuffer coordinates point down; flip clip-space Y.
    projection.m[5] = -2.0f / height;
    projection.m[8] = (right + left) / width;
    projection.m[9] = -(up + down) / height;
    projection.m[10] = -farZ / (farZ - nearZ);
    projection.m[11] = -1.0f;
    projection.m[14] = -(farZ * nearZ) / (farZ - nearZ);
    return projection;
}

}
