#pragma once

#include "spatial_types.hpp"
#include <cmath>

namespace FrameSpatial {

// Hardware Camera Dimensions
constexpr int RAW_CAMERA_WIDTH  = 2464;
constexpr int RAW_CAMERA_HEIGHT = 2464;

// Rectified Virtual Camera Dimensions
constexpr int RECT_WIDTH  = 616;
constexpr int RECT_HEIGHT = 616;
constexpr int MAX_DISPARITY = 64;

// Virtual Rectified Pin-hole Intrinsics
constexpr float RECT_FOCAL = 380.0f;
constexpr float RECT_CX    = 308.0f;
constexpr float RECT_CY    = 308.0f;

// Physical Stereo Baseline (Valve Frame CAD: 63.96 mm)
constexpr float STEREO_BASELINE_M = 0.0639016f;
constexpr float FOCAL_BASELINE    = RECT_FOCAL * STEREO_BASELINE_M; // 24.3048

// Left Camera Factory Calibration (Kannala-Brandt Fisheye)
// Current device EEPROM values reported by XRService (passthroughCam1).
constexpr float FX_L = 1477.96f;
constexpr float FY_L = 1480.84f;
constexpr float CX_L = 1086.97f;
constexpr float CY_L = 1237.34f;
constexpr float K1_L = -0.030051f;
constexpr float K2_L = -0.161829f;
constexpr float K3_L =  0.091782f;
constexpr float K4_L = -0.012317f;

// Right Camera Factory Calibration (Kannala-Brandt Fisheye)
// Note: Right sensor is mounted with 180-degree physical rotation
// Current device EEPROM values reported by XRService (passthroughCam0).
// Its raw principal point is near x=1388; the previous x=1079 value was an
// already-flipped coordinate and was incorrect for sampling the DMA image.
constexpr float FX_R = 1479.22f;
constexpr float FY_R = 1481.90f;
constexpr float CX_R = 1387.79f;
constexpr float CY_R = 1220.44f;
constexpr float K1_R = -0.032597f;
constexpr float K2_R = -0.152976f;
constexpr float K3_R =  0.082236f;
constexpr float K4_R = -0.008963f;

// Camera-to-HMD Rig Extrinsics (relative to HMD center of mass / eye midpoint)
constexpr float CAM_TO_HMD_X = -0.0528f; // ~5.3 cm left of center
constexpr float CAM_TO_HMD_Y = -0.0186f; // ~1.9 cm below eye plane
constexpr float CAM_TO_HMD_Z = -0.0368f; // ~3.7 cm forward from eye plane

// Kannala-Brandt Equidistant Projection Model
inline Point2D projectKannalaBrandt(float x, float y, float z,
                                   float fx, float fy, float cx, float cy,
                                   float k1, float k2, float k3, float k4) {
    float r_xy = std::sqrt(x * x + y * y);
    if (r_xy < 1e-6f) return { cx, cy };

    float theta = std::atan2(r_xy, z);
    float t2 = theta * theta;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;

    float r_dist = theta * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
    float scale = r_dist / r_xy;

    return { cx + fx * x * scale, cy + fy * y * scale };
}

} // namespace FrameSpatial
