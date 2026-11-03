#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace FrameSpatial {

// Learned stereo disparity on Frame's native Mesa/Turnip Vulkan driver.
// MNN is kept behind a pImpl so the main headers remain buildable before the
// optional Vulkan-AI dependency has been installed.
class GpuDepthPipeline {
public:
    GpuDepthPipeline();
    ~GpuDepthPipeline();

    GpuDepthPipeline(const GpuDepthPipeline&) = delete;
    GpuDepthPipeline& operator=(const GpuDepthPipeline&) = delete;

    bool initialize(const std::string& modelPath);
    bool process(const std::vector<uint8_t>& rectifiedLeft,
                 const std::vector<uint8_t>& rectifiedRight,
                 std::vector<float>& depthMeters);

    bool isReady() const;
    float lastInferenceMs() const;
    const std::string& lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace FrameSpatial
