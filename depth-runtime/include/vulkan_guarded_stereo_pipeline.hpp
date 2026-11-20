#pragma once

#include "vulkan_stereo_pipeline.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace FrameSpatial {

// Opt-in guarded research matcher, fixed g3/e1/r8/t32/c120/m3.
// Exposure percentiles and final speckle cleanup remain host-side.
// Native Vulkan stereo matcher. Rectified 616x616 luma is uploaded once, then
// census, bidirectional disparity, confidence rejection, and metric conversion
// execute on the GPU. Optional row submissions yield to the scene renderer.
// Intermediate readback is explicitly opt-in.
// Rejection bits are stage-local: a pixel rejected upstream is not counted again
// as a filter or support failure. Match flags may contain multiple reasons.
struct GuardedStereoDiagnostics {
    std::vector<float> leftDisparity, rightDisparity, consistencyDepth;
    // Interleaved RGBA: integer best cost, competitor-minus-best cost,
    // integer winning disparity, match rejection bits. Border/search use -1
    // for unevaluated cost/margin; margin -1 also means no competitor exists.
    std::vector<float> leftMatch, rightMatch;
    std::vector<uint32_t> filterRejection, supportRejection;
    // Evidence RGBA: selected samples, joint census bits, stopped arms, reserved.
    std::vector<float> leftAnchors, rightAnchors, leftEvidence, rightEvidence;
    // Two uint32 words per 48-bit descriptor boundary mask.
    std::vector<uint32_t> leftMasks, rightMasks;
};
struct GuardedStereoOptions {
    bool profile = false;
    bool precomputeArms = false;
    bool cacheNeighbors = false;
    uint32_t batchRows = 8, yieldMicroseconds = 3000;
    uint32_t localSizeX = 8, localSizeY = 8;
    static GuardedStereoOptions fromEnvironment();
};
struct GuardedStereoProfile {
    bool complete = false, timestampsAvailable = false;
    uint32_t timestampValidBits = 0;
    float timestampPeriodNs = 0;
    double pairMs = 0, uploadHostMs = 0, histogramMs = 0, recordSubmitMs = 0;
    double fenceWaitMs = 0, yieldMs = 0, readbackMs = 0;
    uint64_t allocatedBytes = 0;
    std::vector<std::string> stages;
    std::vector<double> gpuMs;
};
class VulkanGuardedStereoPipeline {
public:
    VulkanGuardedStereoPipeline();
    ~VulkanGuardedStereoPipeline();

    VulkanGuardedStereoPipeline(const VulkanGuardedStereoPipeline&) = delete;
    VulkanGuardedStereoPipeline& operator=(const VulkanGuardedStereoPipeline&) = delete;

    bool initialize(const std::string& shaderDirectory = "shaders",
                    bool enableDiagnostics = false, bool integerRefinement = false,
                    bool interleaveRendering = false,
                    const GuardedStereoOptions& options = GuardedStereoOptions::fromEnvironment());
    bool process(const std::vector<uint8_t>& rectifiedLeft,
                 const std::vector<uint8_t>& rectifiedRight,
                 std::vector<float>& depthMeters,
                 GuardedStereoDiagnostics* diagnostics = nullptr);

    bool isReady() const;
    float lastComputeMs() const;
    const GuardedStereoProfile& lastProfile() const;
    const std::string& lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace FrameSpatial
