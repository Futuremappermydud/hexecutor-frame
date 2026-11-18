#include "depth_pipeline.hpp"
#include "calibration.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <stdexcept>
#include <omp.h>

namespace FrameSpatial {

namespace {

using Vec3 = std::array<float, 3>;
using Mat3 = std::array<Vec3, 3>;

float dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 normalized(Vec3 value) {
    const float length = std::sqrt(dot(value, value));
    if (length <= 1e-8f) return {0.0f, 0.0f, 0.0f};
    for (float& component : value) component /= length;
    return value;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

Vec3 multiply(const Mat3& matrix, const Vec3& value) {
    return {dot(matrix[0], value), dot(matrix[1], value), dot(matrix[2], value)};
}

Mat3 quaternionMatrix(float x, float y, float z, float w) {
    const float scale = 1.0f / std::sqrt(x * x + y * y + z * z + w * w);
    x *= scale; y *= scale; z *= scale; w *= scale;
    return {{{1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w),
              2.0f * (x * z + y * w)},
             {2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z),
              2.0f * (y * z - x * w)},
             {2.0f * (x * z - y * w), 2.0f * (y * z + x * w),
              1.0f - 2.0f * (x * x + y * y)}}};
}

// XRService continuously refines this seven-value transform:
// [translation xyz, quaternion xyzw], passthroughCam0 (right) from
// passthroughCam1 (left). Reading its latest entry keeps epipolar alignment
// tied to the headset rather than to a calibration copied from another unit.
bool loadStereoTransform(const std::string& name, Vec3& translation,
                         std::array<float, 4>& quaternion) {
    const char* overridePath = std::getenv("FRAME_STEREO_CALIBRATION");
    const std::string path = overridePath && *overridePath
        ? overridePath
        : "/home/steamos/.config/openvr/config/cv/xrservice/"
          "colorPassthroughCalibrationHistory.json";
    std::ifstream input(path);
    if (!input) return false;
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    const std::string key = "\"" + name + "\"";
    size_t position = 0;
    size_t latest = std::string::npos;
    while ((position = json.find(key, position)) != std::string::npos) {
        latest = position++;
    }
    if (latest == std::string::npos) return false;
    const size_t bracket = json.find('[', latest + key.size());
    if (bracket == std::string::npos) return false;
    const char* cursor = json.c_str() + bracket + 1;
    float values[7]{};
    for (float& value : values) {
        while (*cursor && !std::isdigit(static_cast<unsigned char>(*cursor)) &&
               *cursor != '-' && *cursor != '+' && *cursor != '.') ++cursor;
        if (!*cursor) return false;
        char* end = nullptr;
        value = std::strtof(cursor, &end);
        if (end == cursor) return false;
        cursor = end;
    }
    translation = {values[0], values[1], values[2]};
    quaternion = {values[3], values[4], values[5], values[6]};
    return true;
}

} // namespace

static inline float sampleBilinear(const uint8_t *img, float px, float py) {
    if (px < 0.0f || px >= RAW_CAMERA_WIDTH - 1 || py < 0.0f || py >= RAW_CAMERA_HEIGHT - 1) return 0.0f;
    int x0 = (int)px;
    int y0 = (int)py;
    float wx = px - (float)x0;
    float wy = py - (float)y0;

    float p00 = img[y0 * RAW_CAMERA_WIDTH + x0];
    float p10 = img[y0 * RAW_CAMERA_WIDTH + x0 + 1];
    float p01 = img[(y0 + 1) * RAW_CAMERA_WIDTH + x0];
    float p11 = img[(y0 + 1) * RAW_CAMERA_WIDTH + x0 + 1];

    return (p00 * (1.0f - wx) + p10 * wx) * (1.0f - wy) + (p01 * (1.0f - wx) + p11 * wx) * wy;
}

DepthPipeline::DepthPipeline()
    : m_initialized(false) {}

DepthPipeline::~DepthPipeline() = default;

void DepthPipeline::initialize() {
    m_lutL.resize(RECT_WIDTH * RECT_HEIGHT);
    m_lutR.resize(RECT_WIDTH * RECT_HEIGHT);
    m_rectL.resize(RECT_WIDTH * RECT_HEIGHT);
    m_rectR.resize(RECT_WIDTH * RECT_HEIGHT);
    m_censusL.resize(RECT_WIDTH * RECT_HEIGHT);
    m_censusR.resize(RECT_WIDTH * RECT_HEIGHT);
    m_depthMap.resize(RECT_WIDTH * RECT_HEIGHT, 0.0f);

    // Factory fallback from this headset's aligned stereo history. The values
    // are replaced by the newest XRService record whenever it is available.
    Vec3 rightFromLeftTranslation{0.00228127f, 0.06266509f, -0.01230039f};
    std::array<float, 4> rightFromLeftQuaternion{
        -0.00618799f, 0.18649586f, 0.98183560f, 0.03434955f};
    const bool loadedAlignment = loadStereoTransform(
        "passthroughCam0FromPassthroughCam1Extrinsics",
        rightFromLeftTranslation, rightFromLeftQuaternion);
    const Mat3 rightFromLeft = quaternionMatrix(
        rightFromLeftQuaternion[0], rightFromLeftQuaternion[1],
        rightFromLeftQuaternion[2], rightFromLeftQuaternion[3]);

    // Build a common rectified basis in left-camera coordinates. Its +X axis
    // follows the physical baseline, +Z bisects the two optical axes, and +Y
    // completes a right-handed frame. This makes corresponding rows epipolar.
    Vec3 baselineLeft{};
    for (int row = 0; row < 3; ++row) {
        baselineLeft[row] = -(rightFromLeft[0][row] * rightFromLeftTranslation[0] +
                              rightFromLeft[1][row] * rightFromLeftTranslation[1] +
                              rightFromLeft[2][row] * rightFromLeftTranslation[2]);
    }
    const Vec3 rectX = normalized(baselineLeft);
    m_rectifiedBaseline = std::sqrt(dot(baselineLeft, baselineLeft));
    const Vec3 rightOpticalInLeft{
        rightFromLeft[2][0], rightFromLeft[2][1], rightFromLeft[2][2]};
    Vec3 rectZ = normalized(Vec3{rightOpticalInLeft[0], rightOpticalInLeft[1],
                                 rightOpticalInLeft[2] + 1.0f});
    const float opticalAlongBaseline = dot(rectZ, rectX);
    rectZ = normalized(Vec3{rectZ[0] - opticalAlongBaseline * rectX[0],
                            rectZ[1] - opticalAlongBaseline * rectX[1],
                            rectZ[2] - opticalAlongBaseline * rectX[2]});
    const Vec3 rectY = normalized(cross(rectZ, rectX));

    // SteamVR device_config cv.cad_from_cal maps frontCam0 calibration axes
    // into CAD; device_config head maps CAD into the tracked head axes.
    // This unit's factory values are retained as a fallback.
    const Vec3 cadX{0.648667109f, -0.760370276f, 0.032680637f};
    const Vec3 cadZ{-0.456154753f, -0.354052366f, 0.816437238f};
    const Vec3 cadY = cross(cadZ, cadX);
    const Mat3 headFromFront{{{-cadX[0], -cadY[0], -cadZ[0]},
                              {cadX[1], cadY[1], cadZ[1]},
                              {-cadX[2], -cadY[2], -cadZ[2]}}};
    const Vec3 frontOriginInHead{0.052832839f, -0.018683300f, -0.036808733f};
    Vec3 frontFromRightTranslation{0.003687401f, 0.032272208f, 0.001879808f};
    std::array<float, 4> frontFromRightQuaternion{
        -0.163944200f, -0.019652419f, -0.357536495f, 0.919186473f};
    const bool loadedHeadAlignment = loadStereoTransform(
        "frontCam0FromPassthroughCam0Extrinsics",
        frontFromRightTranslation, frontFromRightQuaternion);
    const Mat3 frontFromRight = quaternionMatrix(
        frontFromRightQuaternion[0], frontFromRightQuaternion[1],
        frontFromRightQuaternion[2], frontFromRightQuaternion[3]);
    const auto leftToHead = [&](const Vec3& v) {
        return multiply(headFromFront, multiply(frontFromRight, multiply(rightFromLeft, v)));
    };
    const Vec3 headX = leftToHead(rectX);
    const Vec3 headY = leftToHead(rectY);
    const Vec3 headZ = leftToHead(rectZ);
    for (int row = 0; row < 3; ++row) {
        m_headFromRectified[row] = {headX[row], headY[row], headZ[row]};
    }
    Vec3 leftOriginInFront = multiply(frontFromRight, rightFromLeftTranslation);
    for (int i = 0; i < 3; ++i) leftOriginInFront[i] += frontFromRightTranslation[i];
    m_leftOriginInHead = multiply(headFromFront, leftOriginInFront);
    for (int i = 0; i < 3; ++i) m_leftOriginInHead[i] += frontOriginInHead[i];
    std::cout << "[DepthPipeline] Rectified forward in head: "
              << headZ[0] << ", " << headZ[1] << ", " << headZ[2]
              << "; head alignment="
              << (loadedHeadAlignment ? "XRService latest history" : "built-in fallback")
              << std::endl;

    // Precompute Kannala-Brandt rectification LUTs using the full relative
    // rotation. A simple (-x,-y,z) flip is not sufficient on this headset.
    for (int y = 0; y < RECT_HEIGHT; y++) {
        for (int x = 0; x < RECT_WIDTH; x++) {
            float rx = ((float)x - RECT_CX) / RECT_FOCAL;
            float ry = ((float)y - RECT_CY) / RECT_FOCAL;

            const Vec3 leftRay{
                rectX[0] * rx + rectY[0] * ry + rectZ[0],
                rectX[1] * rx + rectY[1] * ry + rectZ[1],
                rectX[2] * rx + rectY[2] * ry + rectZ[2]};
            const Vec3 rightRay = multiply(rightFromLeft, leftRay);

            m_lutL[y * RECT_WIDTH + x] = projectKannalaBrandt(
                leftRay[0], leftRay[1], leftRay[2],
                FX_L, FY_L, CX_L, CY_L, K1_L, K2_L, K3_L, K4_L);

            m_lutR[y * RECT_WIDTH + x] = projectKannalaBrandt(
                rightRay[0], rightRay[1], rightRay[2],
                FX_R, FY_R, CX_R, CY_R, K1_R, K2_R, K3_R, K4_R);
        }
    }

    // Model-style spherical model inputs, 3x3 supersampled against aliasing
    // (the raw fisheye is ~7.5x finer than 0.29 deg/px).
    if (const char* enabled = std::getenv("FRAME_DEPTH_SPHERICAL"); enabled && std::string(enabled) == "1") {
        const char* degrees = std::getenv("FRAME_DEPTH_SPHERICAL_DEG");
        m_sphK = float((degrees ? std::atof(degrees) : 0.29) * M_PI / 180.0);
        const int w = SPHERICAL_WIDTH, h = SPHERICAL_HEIGHT;
        m_sphLutL.resize(size_t(w) * h * 9);
        m_sphLutR.resize(size_t(w) * h * 9);
        m_sphL.assign(size_t(w) * h, 0);
        m_sphR.assign(size_t(w) * h, 0);
        for (int v = 0; v < h; ++v)
            for (int u = 0; u < w; ++u)
                for (int s = 0; s < 9; ++s) {
                    const float az = (u + (s % 3 - 1) / 3.0f - (w - 1) * 0.5f) * m_sphK;
                    const float el = (v + (s / 3 - 1) / 3.0f - (h - 1) * 0.5f) * m_sphK;
                    const float a = std::tan(az), b = std::sin(el), c = std::cos(el);
                    const Vec3 leftRay{rectX[0] * a + rectY[0] * b + rectZ[0] * c,
                                       rectX[1] * a + rectY[1] * b + rectZ[1] * c,
                                       rectX[2] * a + rectY[2] * b + rectZ[2] * c};
                    const Vec3 rightRay = multiply(rightFromLeft, leftRay);
                    const size_t i = (size_t(v) * w + u) * 9 + s;
                    m_sphLutL[i] = projectKannalaBrandt(leftRay[0], leftRay[1], leftRay[2],
                        FX_L, FY_L, CX_L, CY_L, K1_L, K2_L, K3_L, K4_L);
                    m_sphLutR[i] = projectKannalaBrandt(rightRay[0], rightRay[1], rightRay[2],
                        FX_R, FY_R, CX_R, CY_R, K1_R, K2_R, K3_R, K4_R);
                }
        std::cout << "[DepthPipeline] Model spherical inputs " << w << "x" << h << " at "
                  << m_sphK * 180.0 / M_PI << " deg/px (" << w * m_sphK * 180.0 / M_PI << " x "
                  << h * m_sphK * 180.0 / M_PI << " deg)" << std::endl;
    }

    m_initialized = true;
    std::cout << "[DepthPipeline] Initialized full-extrinsic rectification ("
              << RECT_WIDTH << "x" << RECT_HEIGHT << ", 64 disparities)" << std::endl;
    std::cout << "[DepthPipeline] Stereo alignment source: "
              << (loadedAlignment ? "XRService latest history" : "built-in fallback")
              << ", baseline=" << m_rectifiedBaseline
              << " m" << std::endl;
}

std::array<float, 3> DepthPipeline::rectifiedPointInHead(int x, int y, float z) const {
    const Vec3 point{(x - RECT_CX) * z / RECT_FOCAL,
                     (y - RECT_CY) * z / RECT_FOCAL, z};
    Vec3 head = multiply(m_headFromRectified, point);
    for (int i = 0; i < 3; ++i) head[i] += m_leftOriginInHead[i];
    return head;
}

std::array<float, 3> DepthPipeline::rectifiedCameraPointInHead(const Vec3& point) const {
    Vec3 head = multiply(m_headFromRectified, point);
    for (int i = 0; i < 3; ++i) head[i] += m_leftOriginInHead[i];
    return head;
}

std::array<float, 3> DepthPipeline::headPointInRectified(const Vec3& head) const {
    Vec3 rect{};
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            rect[col] += m_headFromRectified[row][col] *
                         (head[row] - m_leftOriginInHead[row]);
    return rect;
}

std::vector<Point3D> DepthPipeline::process(const uint8_t* leftRaw,
                                            const uint8_t* rightRaw,
                                            const HmdPose& hmdPose) {
    rectify(leftRaw, rightRaw);

    // 2. Compute one illumination-invariant 7x7 Census descriptor per pixel.
    auto census = [](const std::vector<uint8_t>& image, std::vector<uint64_t>& out) {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < RECT_HEIGHT; ++y) {
            for (int x = 0; x < RECT_WIDTH; ++x) {
                const int idx = y * RECT_WIDTH + x;
                if (x < 3 || x >= RECT_WIDTH - 3 || y < 3 || y >= RECT_HEIGHT - 3) {
                    out[idx] = 0;
                    continue;
                }
                const uint8_t center = image[idx];
                uint64_t code = 0;
                int bit = 0;
                for (int dy = -3; dy <= 3; ++dy) {
                    for (int dx = -3; dx <= 3; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (image[(y + dy) * RECT_WIDTH + x + dx] > center) {
                            code |= uint64_t{1} << bit;
                        }
                        ++bit;
                    }
                }
                out[idx] = code;
            }
        }
    };
    census(m_rectL, m_censusL);
    census(m_rectR, m_censusR);

    // 3. Winner-take-all horizontal disparity search on the rectified pair.
    // Keep the HUD dense; a stricter uniqueness/speckle pass belongs in the mesh path.
    constexpr int kMaxDisp = 64;
    constexpr int kCostThreshold = 28;
    constexpr float kMinDepth = 0.20f;
    constexpr float kMaxDepth = 12.0f;
    std::fill(m_depthMap.begin(), m_depthMap.end(), 0.0f);

    #pragma omp parallel for collapse(2)
    for (int y = 3; y < RECT_HEIGHT - 3; ++y) {
        for (int x = 4; x < RECT_WIDTH - 3; ++x) {
            const int idx = y * RECT_WIDTH + x;
            const uint64_t left = m_censusL[idx];
            int bestDisp = 0;
            int bestCost = std::numeric_limits<int>::max();
            const int maxDisp = std::min(kMaxDisp - 1, x - 3);

            for (int d = 1; d <= maxDisp; ++d) {
                const int cost = __builtin_popcountll(
                    left ^ m_censusR[y * RECT_WIDTH + x - d]);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestDisp = d;
                }
            }

            if (bestDisp <= 1 || bestCost >= kCostThreshold) continue;

            float subpixelDisp = static_cast<float>(bestDisp);
            if (bestDisp > 1 && bestDisp < maxDisp) {
                auto costAt = [&](int d) {
                    return static_cast<float>(__builtin_popcountll(
                        left ^ m_censusR[y * RECT_WIDTH + x - d]));
                };
                const float c0 = costAt(bestDisp - 1);
                const float c1 = static_cast<float>(bestCost);
                const float c2 = costAt(bestDisp + 1);
                const float denom = 2.0f * (c0 - 2.0f * c1 + c2);
                if (std::fabs(denom) > 1e-4f) {
                    subpixelDisp += std::clamp((c0 - c2) / denom, -1.0f, 1.0f);
                }
            }

            const float depth = FOCAL_BASELINE / subpixelDisp;
            if (depth >= kMinDepth && depth <= kMaxDepth) m_depthMap[idx] = depth;
        }
    }

    ++m_frameCount;
    if (m_frameCount == 1 || m_frameCount % 60 == 0) {
        size_t valid = 0;
        float minDepth = kMaxDepth;
        float maxDepth = kMinDepth;
        for (float z : m_depthMap) {
            if (z <= 0.0f) continue;
            ++valid;
            minDepth = std::min(minDepth, z);
            maxDepth = std::max(maxDepth, z);
        }
        const float coverage = 100.0f * static_cast<float>(valid) /
                               static_cast<float>(m_depthMap.size());
        std::cout << "[DepthPipeline] metric depth coverage=" << coverage << "%";
        if (valid) std::cout << " range=" << minDepth << ".." << maxDepth << "m";
        std::cout << std::endl;
    }

    (void)hmdPose;
    return {};
}

std::vector<uint8_t> DepthPipeline::rectifyLeftRgb(const uint8_t* y, const uint8_t* uv) {
    if (!y || !uv) throw std::invalid_argument("left RGB rectification requires Y and UV");
    if (!m_initialized) initialize();
    std::vector<uint8_t> rgb(size_t(RECT_WIDTH) * RECT_HEIGHT * 3);
    #pragma omp parallel for
    for (int i = 0; i < RECT_WIDTH * RECT_HEIGHT; ++i) {
        const auto p = m_lutL[i];
        if (p.x < 0 || p.y < 0 || p.x >= RAW_CAMERA_WIDTH - 1 || p.y >= RAW_CAMERA_HEIGHT - 1)
            continue;
        const float c = sampleBilinear(y, p.x, p.y) - 16;
        // NV12 chroma is sampled at its native half resolution. Match the
        // production passthrough's BT.601 limited-range conversion.
        const size_t index = (size_t(int(p.y) / 2) * (RAW_CAMERA_WIDTH / 2) + int(p.x) / 2) * 2;
        const float u = uv[index] - 128.f, v = uv[index + 1] - 128.f;
        auto byte = [](float value) { return uint8_t(std::clamp(std::lround(value), 0l, 255l)); };
        rgb[size_t(i) * 3] = byte(1.164f * c + 1.596f * v);
        rgb[size_t(i) * 3 + 1] = byte(1.164f * c - .392f * u - .813f * v);
        rgb[size_t(i) * 3 + 2] = byte(1.164f * c + 2.017f * u);
    }
    return rgb;
}

void DepthPipeline::rectify(const uint8_t* leftRaw, const uint8_t* rightRaw) {
    if (!m_initialized) initialize();

    // 1. Bilinear Rectification
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < RECT_HEIGHT; y++) {
        for (int x = 0; x < RECT_WIDTH; x++) {
            int idx = y * RECT_WIDTH + x;
            m_rectL[idx] = (uint8_t)sampleBilinear(leftRaw, m_lutL[idx].x, m_lutL[idx].y);
            m_rectR[idx] = (uint8_t)sampleBilinear(rightRaw, m_lutR[idx].x, m_lutR[idx].y);
        }
    }
    if (!m_sphLutL.empty()) {
        const int pixels = SPHERICAL_WIDTH * SPHERICAL_HEIGHT;
        #pragma omp parallel for
        for (int i = 0; i < pixels; i++) {
            float l = 0, r = 0;
            for (int s = 0; s < 9; ++s) {
                const auto& pl = m_sphLutL[size_t(i) * 9 + s];
                const auto& pr = m_sphLutR[size_t(i) * 9 + s];
                l += sampleBilinear(leftRaw, pl.x, pl.y);
                r += sampleBilinear(rightRaw, pr.x, pr.y);
            }
            m_sphL[i] = (uint8_t)std::lround(l / 9);
            m_sphR[i] = (uint8_t)std::lround(r / 9);
        }
    }
}

} // namespace FrameSpatial
