#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace FrameSpatial {

// Native Vulkan stereo matcher. Rectified 616x616 luma is uploaded once, then
// census, bidirectional disparity, confidence rejection, and metric conversion
// execute in one command buffer. Intermediate readback is explicitly opt-in.
// Rejection bits are stage-local: a pixel rejected upstream is not counted again
// as a filter or support failure. Match flags may contain multiple reasons.
enum StereoMatchRejection : uint32_t {
    StereoMatchBorder = 1, StereoMatchSearch = 2, StereoMatchLowDisparity = 4,
    StereoMatchCost = 8, StereoMatchAmbiguous = 16
};
enum StereoFilterRejection : uint32_t { StereoFilterLR = 1, StereoFilterRange = 2, StereoFilterLowDisparity = 4 };
struct StereoDiagnostics {
    std::vector<float> leftDisparity, rightDisparity, consistencyDepth;
    // Interleaved RGBA: integer best cost, competitor-minus-best cost,
    // integer winning disparity, match rejection bits. Border/search use -1
    // for unevaluated cost/margin; margin -1 also means no competitor exists.
    std::vector<float> leftMatch, rightMatch;
    std::vector<uint32_t> filterRejection, supportRejection;
};
class VulkanStereoPipeline {
public:
    VulkanStereoPipeline();
    ~VulkanStereoPipeline();

    VulkanStereoPipeline(const VulkanStereoPipeline&) = delete;
    VulkanStereoPipeline& operator=(const VulkanStereoPipeline&) = delete;

    bool initialize(const std::string& shaderDirectory = "shaders",
                    bool enableDiagnostics = false);
    bool process(const std::vector<uint8_t>& rectifiedLeft,
                 const std::vector<uint8_t>& rectifiedRight,
                 std::vector<float>& depthMeters,
                 StereoDiagnostics* diagnostics = nullptr);

    bool isReady() const;
    float lastComputeMs() const;
    const std::string& lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace FrameSpatial
