#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <vulkan/vulkan.h>

namespace FrameSpatial {

// Import compositor NV12 DMA-BUFs (QCOM UBWC) and copy the Y plane to linear CPU memory.
class VkYDetile {
public:
    static constexpr int kMaxSlots = 4;

    VkYDetile(int width = 2464, int height = 2464, VkDeviceSize dmaSize = 9564160);
    ~VkYDetile();

    bool initialize();
    bool importStereoGroups(const std::vector<int>& leftFds, const std::vector<int>& rightFds);
    // Luma-only reads preserve Y slot selection; UV getters return null until refreshed.
    // Left-only consumers avoid copying/detiling the unused right camera.
    // Right getters are unavailable until a subsequent full stereo capture.
    bool copyYPlanes(bool includeUV = true, bool leftOnly = false);
    void release();

    const uint8_t* leftY() const { return m_leftY.empty() ? nullptr : m_leftY.data(); }
    const uint8_t* rightY() const { return !m_rightCurrent || m_rightY.empty() ? nullptr : m_rightY.data(); }
    const uint8_t* leftUV() const { return !m_uvCurrent || m_leftUV.empty() ? nullptr : m_leftUV.data(); }
    const uint8_t* rightUV() const { return !m_rightCurrent || !m_uvCurrent || m_rightUV.empty() ? nullptr : m_rightUV.data(); }
    int uvWidth() const { return m_uvW; }
    int uvHeight() const { return m_uvH; }
    uint64_t leftHash() const { return m_leftHash; }
    uint64_t rightHash() const { return m_rightHash; }
    int leftSlot() const { return m_curLeft; }
    int rightSlot() const { return m_curRight; }
    // Sampled Y hash of every imported slot from the latest copy (diagnostics).
    int leftSlotCount() const { return m_nLeft; }
    int rightSlotCount() const { return m_nRight; }
    uint64_t slotHash(bool left, int i) const { return left ? m_prevLeftHash[i] : m_prevRightHash[i]; }
    int width() const { return m_width; }
    int height() const { return m_height; }
    bool isReady() const { return m_imported; }

private:
    struct ImportedImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    bool createImportedImage(int dmaFd, ImportedImage& out);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const;
    void destroyImage(ImportedImage& img);
    static uint64_t hashPlane(const uint8_t* p, size_t n);

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_phys = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VkCommandPool m_cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkBuffer m_hostBuf = VK_NULL_HANDLE;
    VkDeviceMemory m_hostMem = VK_NULL_HANDLE;
    uint8_t* m_hostMapped = nullptr;
    VkDeviceSize m_ySize = 0;
    VkBuffer m_uvBuf = VK_NULL_HANDLE;
    VkDeviceMemory m_uvMem = VK_NULL_HANDLE;
    uint8_t* m_uvMapped = nullptr;
    VkDeviceSize m_uvSize = 0;
    int m_uvW = 0;
    int m_uvH = 0;

    ImportedImage m_leftSlots[kMaxSlots];
    ImportedImage m_rightSlots[kMaxSlots];
    int m_nLeft = 0;
    int m_nRight = 0;
    uint64_t m_prevLeftHash[kMaxSlots]{};
    uint64_t m_prevRightHash[kMaxSlots]{};
    int m_curLeft = 0;
    int m_curRight = 0;
    uint64_t m_leftHash = 0;
    uint64_t m_rightHash = 0;

    std::vector<uint8_t> m_leftY;
    std::vector<uint8_t> m_rightY;
    std::vector<uint8_t> m_leftUV;
    std::vector<uint8_t> m_rightUV;

    PFN_vkGetMemoryFdPropertiesKHR m_vkGetMemoryFdPropertiesKHR = nullptr;
    VkPhysicalDeviceMemoryProperties m_memProps{};
    bool m_uvCurrent = false;
    bool m_rightCurrent = false;
    bool m_initialized = false;
    bool m_imported = false;
    int m_width;
    int m_height;
    VkDeviceSize m_dmaSize;
};

} // namespace FrameSpatial
