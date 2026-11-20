#pragma once
#include <string>

#include "spatial_types.hpp"
#include <vector>
#include <array>
#include <cstdint>

namespace FrameSpatial {

class DepthPipeline {
public:
    DepthPipeline();
    ~DepthPipeline();

    void initialize();

    // Calibrated luma rectification shared by both the CPU matcher and the
    // asynchronous learned Vulkan backend.
    void rectify(const uint8_t* leftRaw, const uint8_t* rightRaw);
    // Calibration provenance (valid after the first rectify/initialize): the
    // XRService history text read at initialization, empty for the built-in fallback.
    const std::string& calibrationSource() const { return m_calibrationSource; }
    const std::array<float, 4>& stereoQuaternion() const { return m_stereoQuaternion; }
    const std::array<float, 3>& stereoTranslation() const { return m_stereoTranslation; }
    // FRAME_STEREO_CALIBRATION_POLICY=medoid (default) | latest; chosen history index.
    const std::string& calibrationPolicy() const { return m_calibrationPolicy; }
    int calibrationEntry() const { return m_calibrationEntry; }
    int calibrationEntries() const { return m_calibrationEntries; }

    // RGB view of the left NV12 camera, using the same calibrated LUT as luma.
    // Chroma is interleaved UV at half resolution. No raw DMA storage is retained.
    std::vector<uint8_t> rectifyLeftRgb(const uint8_t* y, const uint8_t* uv);

    // Process stereo pair into 3D metric point cloud in world coordinates
    std::vector<Point3D> process(const uint8_t* leftRaw,
                                const uint8_t* rightRaw,
                                const HmdPose& hmdPose);

    // Apply the same rectified basis used by the LUT, then calibrated rig extrinsics.
    std::array<float, 3> rectifiedPointInHead(int x, int y, float z) const;
    // Metric point in the rectified left camera (x right, y down, z forward) to head.
    std::array<float, 3> rectifiedCameraPointInHead(const std::array<float, 3>& point) const;

    // Inverse rig transform for visibility checks in the captured depth image.
    std::array<float, 3> headPointInRectified(const std::array<float, 3>& head) const;

    // Accessors for debug / visualization
    const std::vector<uint8_t>& getRectifiedLeft() const { return m_rectL; }
    const std::vector<uint8_t>& getRectifiedRight() const { return m_rectR; }
    float rectifiedBaseline() const { return m_rectifiedBaseline; }
    // Model-style rectified spherical images (256x320, same rectified basis):
    // column -> azimuth (u - 127.5) * k, row -> elevation (v - 159.5) * k,
    // ray ~ (tan az, sin el, cos el). Empty unless FRAME_DEPTH_SPHERICAL=1.
    static constexpr int SPHERICAL_WIDTH = 256, SPHERICAL_HEIGHT = 320;
    const std::vector<uint8_t>& getSphericalLeft() const { return m_sphL; }
    const std::vector<uint8_t>& getSphericalRight() const { return m_sphR; }
    float sphericalStepRadians() const { return m_sphK; }
    const std::vector<float>& getDepthMap() const { return m_depthMap; }

private:
    std::vector<Point2D> m_lutL;
    std::vector<Point2D> m_lutR;
    std::vector<uint8_t> m_rectL;
    std::vector<uint8_t> m_rectR;
    std::vector<Point2D> m_sphLutL, m_sphLutR;  // 9 supersamples per pixel
    std::vector<uint8_t> m_sphL, m_sphR;
    float m_sphK = 0;
    float m_rectifiedBaseline = 0;
    std::vector<uint64_t> m_censusL;
    std::vector<uint64_t> m_censusR;
    std::vector<float> m_depthMap;
    std::array<std::array<float, 3>, 3> m_headFromRectified{};
    std::array<float, 3> m_leftOriginInHead{};
    bool m_initialized;
    std::string m_calibrationSource;
    std::string m_calibrationPolicy = "medoid";
    int m_calibrationEntry = -1, m_calibrationEntries = 0;
    std::array<float, 4> m_stereoQuaternion{};
    std::array<float, 3> m_stereoTranslation{};
    uint64_t m_frameCount = 0;
};

} // namespace FrameSpatial
