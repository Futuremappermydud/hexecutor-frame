#pragma once

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
    
    // Process stereo pair into 3D metric point cloud in world coordinates
    std::vector<Point3D> process(const uint8_t* leftRaw,
                                const uint8_t* rightRaw,
                                const HmdPose& hmdPose);

    // Apply the same rectified basis used by the LUT, then calibrated rig extrinsics.
    std::array<float, 3> rectifiedPointInHead(int x, int y, float z) const;

    // Inverse rig transform for visibility checks in the captured depth image.
    std::array<float, 3> headPointInRectified(const std::array<float, 3>& head) const;

    // Accessors for debug / visualization
    const std::vector<uint8_t>& getRectifiedLeft() const { return m_rectL; }
    const std::vector<uint8_t>& getRectifiedRight() const { return m_rectR; }
    const std::vector<float>& getDepthMap() const { return m_depthMap; }

private:
    std::vector<Point2D> m_lutL;
    std::vector<Point2D> m_lutR;
    std::vector<uint8_t> m_rectL;
    std::vector<uint8_t> m_rectR;
    std::vector<uint64_t> m_censusL;
    std::vector<uint64_t> m_censusR;
    std::vector<float> m_depthMap;
    std::array<std::array<float, 3>, 3> m_headFromRectified{};
    std::array<float, 3> m_leftOriginInHead{};
    bool m_initialized;
    uint64_t m_frameCount = 0;
};

} // namespace FrameSpatial
