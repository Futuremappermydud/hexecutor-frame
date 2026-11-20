#pragma once
#include <vulkan/vulkan.h>
#include "cloud_worker.hpp"
#include "openxr_geometry.hpp"
#include <string>
#include <vector>

namespace FrameSpatialCloud {
struct OcclusionMaskTarget {
    VkImage color, depth;
    VkFormat colorFormat, depthFormat;
    uint32_t colorLayer, depthLayer;
    XrRect2Di colorRect, depthRect;
    Mat4 viewProjection;
    XrCompositionLayerDepthInfoKHR range;
    float displayAge=-1; // Negative only for untimed standalone diagnostic draws.
};
struct OcclusionRenderStats {
    uint64_t submissions=0, fenceWaits=0, cacheHits=0, cacheMisses=0;
    double prepareMs=0, recordMs=0, submitMs=0, waitMs=0;
};
class OcclusionRenderer {
public:
    ~OcclusionRenderer();
    bool initialize(VkPhysicalDevice physical, VkDevice device, uint32_t family,
                    uint32_t queueIndex, const std::string& shaderDir);
    // World-space triangle list, with three vertices per triangle.
    bool upload(const std::vector<PointVertex>& points);
    bool mask(VkImage color, VkFormat colorFormat, VkImage depth, VkFormat depthFormat,
              uint32_t colorLayer, uint32_t depthLayer, const XrRect2Di& colorRect,
              const XrRect2Di& depthRect, const Mat4& viewProjection,
              const XrCompositionLayerDepthInfoKHR& range);
    // Compatible targets share one synchronous submission and fence wait.
    // All views/descriptors are prepared before recording either eye.
    bool maskBatch(const std::vector<OcclusionMaskTarget>& targets);
    void releaseImage(VkImage image);
    size_t cachedTargetCount() const { return targets_.size(); }
    const OcclusionRenderStats& stats() const { return stats_; }
private:
    struct CachedTarget {
        OcclusionMaskTarget key;
        VkRenderPass renderPass;
        VkImageView colorView=VK_NULL_HANDLE, depthView=VK_NULL_HANDLE;
        VkFramebuffer framebuffer=VK_NULL_HANDLE;
        VkDescriptorSet descriptor=VK_NULL_HANDLE;
    };
    CachedTarget* cachedTarget(const OcclusionMaskTarget& target);
    void destroyTarget(CachedTarget& target);
    void clearTargets();
    std::vector<CachedTarget> targets_;
    OcclusionRenderStats stats_;
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags);
    bool createPipeline(VkFormat format);
    VkDevice device_=VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_{};
    VkQueue queue_=VK_NULL_HANDLE;
    VkCommandPool pool_=VK_NULL_HANDLE;
    VkCommandBuffer command_=VK_NULL_HANDLE;
    VkFence fence_=VK_NULL_HANDLE;
    VkBuffer buffer_=VK_NULL_HANDLE;
    VkDeviceMemory allocation_=VK_NULL_HANDLE;
    void* mapped_=nullptr;
    VkDescriptorSetLayout descriptorLayout_=VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_=VK_NULL_HANDLE;
    VkSampler sampler_=VK_NULL_HANDLE;
    VkPipelineLayout layout_=VK_NULL_HANDLE;
    VkPipeline pipeline_=VK_NULL_HANDLE;
    VkRenderPass renderPass_=VK_NULL_HANDLE;
    VkFormat format_=VK_FORMAT_UNDEFINED;
    std::string shaderDir_;
    uint32_t count_=0;
};
}
