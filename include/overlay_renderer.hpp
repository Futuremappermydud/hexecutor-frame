#pragma once

#include "spatial_types.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <openvr.h>
#include <vulkan/vulkan.h>

namespace FrameSpatial {

class OverlayRenderer {
public:
    OverlayRenderer();
    ~OverlayRenderer();

    bool initialize(vr::IVRSystem* vrSys, int eyeWidth = 256, int eyeHeight = 96,
                    const char* overlayKey = "frame.spatial.imx_monitor.vulkan");
    void shutdown();

    HmdPose getHmdPose();
    void render(const RenderSnapshot& snap, float liveFps);
    bool renderComparison(const std::vector<uint8_t>& image,
                          const std::vector<float>& baseline, const std::vector<float>& experiment,
                          int width, int height, const std::string& title, const std::string& status);

private:
    void initFont();
    void fillRect(int x, int y, int w, int h, uint32_t color);
    void drawRect(int x, int y, int w, int h, uint32_t color);
    void drawLine(int x0, int y0, int x1, int y1, uint32_t color);
    void drawText(int startX, int startY, const std::string& str, uint32_t color, int scale = 1, int clipX0 = 0, int clipX1 = -1);
    void splat(int x, int y, uint32_t color, int radius);
    void blitGray(const uint8_t* src, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH);
    void blitRgba(const uint32_t* src, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH);
    void blitDepth(const float* src, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH);
    bool projectWorld(float wx, float wy, float wz,
                      const float R[3][3], const float hmdPos[3],
                      bool leftEye, int& sx, int& sy);
    void drawWorldLine(float ax, float ay, float az,
                       float bx, float by, float bz,
                       const float R[3][3], const float hmdPos[3],
                       uint32_t color);
    static uint32_t depthColor(float zMeters);
    bool initializeVulkanTexture();
    bool uploadVulkanTexture();
    void releaseVulkanTexture();
    uint32_t findVulkanMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const;

    vr::IVRSystem* m_vrSys;
    vr::IVROverlay* m_overlay;
    vr::VROverlayHandle_t m_ovHandle;

    int m_eyeWidth;
    int m_eyeHeight;
    int m_totalWidth;
    int m_totalHeight;

    std::vector<uint32_t> m_pixels;
    struct OverlayTexture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory imageMemory = VK_NULL_HANDLE;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        bool initialized = false;
    };
    static constexpr int kTextureCount = 3;
    OverlayTexture m_textures[kTextureCount];
    VkInstance m_vkInstance;
    VkPhysicalDevice m_vkPhysicalDevice;
    VkDevice m_vkDevice;
    VkQueue m_vkQueue;
    uint32_t m_vkQueueFamily;
    VkCommandPool m_vkCommandPool;
    VkCommandBuffer m_vkCommandBuffer;
    vr::VRVulkanTextureData_t m_vulkanData;
    int m_textureIdx;
    uint32_t m_uploadSeq;
    uint32_t m_textureFails;
    std::chrono::steady_clock::time_point m_lastUpload;
    vr::HmdMatrix34_t m_hmdTransform;

    vr::HmdMatrix44_t m_projL;
    vr::HmdMatrix44_t m_projR;
    float m_projScaleXL, m_projScaleYL, m_projCenterXL, m_projCenterYL;
    float m_projScaleXR, m_projScaleYR, m_projCenterXR, m_projCenterYR;

    vr::HmdMatrix34_t m_eyeToHeadL;
    vr::HmdMatrix34_t m_eyeToHeadR;

    uint8_t m_font5x7[128][5];
    bool m_fontInitialized;
    bool m_isInitialized;
};

} // namespace FrameSpatial
