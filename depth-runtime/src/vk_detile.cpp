#define _GNU_SOURCE
#include "vk_detile.hpp"
#include "calibration.hpp"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace FrameSpatial {

namespace {
constexpr uint64_t QCOM_COMPRESSED = 0x500000000000001ULL;
} // namespace

VkYDetile::VkYDetile(int width, int height, VkDeviceSize dmaSize)
    : m_width(width), m_height(height), m_dmaSize(dmaSize) {}

VkYDetile::~VkYDetile() {
    release();
}

uint32_t VkYDetile::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const {
    for (uint32_t i = 0; i < m_memProps.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) &&
            (m_memProps.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    for (uint32_t i = 0; i < m_memProps.memoryTypeCount; i++) {
        if (typeBits & (1u << i)) return i;
    }
    return 0;
}

uint64_t VkYDetile::hashPlane(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i += 64) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void VkYDetile::destroyImage(ImportedImage& img) {
    if (!m_device) { img = {}; return; }
    if (img.image) vkDestroyImage(m_device, img.image, nullptr);
    if (img.memory) vkFreeMemory(m_device, img.memory, nullptr);
    img = {};
}

bool VkYDetile::initialize() {
    release();

    const char *instExts[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME
    };
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "frame-spatial-runtime";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instExts;
    if (vkCreateInstance(&ici, nullptr, &m_instance) != VK_SUCCESS) {
        std::cerr << "[VkYDetile] vkCreateInstance failed" << std::endl;
        return false;
    }

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(m_instance, &gpuCount, gpus.data());
    for (auto g : gpus) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(g, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            m_phys = g;
            std::cout << "[VkYDetile] Using GPU: " << props.deviceName << std::endl;
            break;
        }
    }
    if (m_phys == VK_NULL_HANDLE && !gpus.empty()) m_phys = gpus[0];
    if (m_phys == VK_NULL_HANDLE) {
        std::cerr << "[VkYDetile] No Vulkan GPU" << std::endl;
        return false;
    }

    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_phys, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_phys, &qCount, qps.data());
    m_queueFamily = 0;
    for (uint32_t i = 0; i < qCount; i++) {
        if (qps[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            m_queueFamily = i;
            break;
        }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char *devExts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME
    };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 6;
    dci.ppEnabledExtensionNames = devExts;
    if (vkCreateDevice(m_phys, &dci, nullptr, &m_device) != VK_SUCCESS) {
        std::cerr << "[VkYDetile] vkCreateDevice failed" << std::endl;
        return false;
    }
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);
    vkGetPhysicalDeviceMemoryProperties(m_phys, &m_memProps);
    m_vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(m_device, "vkGetMemoryFdPropertiesKHR");
    if (!m_vkGetMemoryFdPropertiesKHR) {
        std::cerr << "[VkYDetile] vkGetMemoryFdPropertiesKHR missing" << std::endl;
        return false;
    }

    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = m_queueFamily;
    if (vkCreateCommandPool(m_device, &cpci, nullptr, &m_cmdPool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = m_cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(m_device, &cbai, &m_cmd) != VK_SUCCESS) return false;

    m_ySize = (VkDeviceSize)m_width * m_height;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = m_ySize * (kMaxSlots * 2);
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(m_device, &bci, nullptr, &m_hostBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements bufReq{};
    vkGetBufferMemoryRequirements(m_device, m_hostBuf, &bufReq);
    VkMemoryAllocateInfo hostAlloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    hostAlloc.allocationSize = bufReq.size;
    hostAlloc.memoryTypeIndex = findMemoryType(bufReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (vkAllocateMemory(m_device, &hostAlloc, nullptr, &m_hostMem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(m_device, m_hostBuf, m_hostMem, 0) != VK_SUCCESS) return false;
    if (vkMapMemory(m_device, m_hostMem, 0, bci.size, 0, (void**)&m_hostMapped) != VK_SUCCESS) return false;

    m_leftY.resize((size_t)m_ySize);
    m_rightY.resize((size_t)m_ySize);

    m_uvW = m_width / 2;
    m_uvH = m_height / 2;
    m_uvSize = (VkDeviceSize)m_uvW * m_uvH * 2; // R8G8
    VkBufferCreateInfo uvBci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    uvBci.size = m_uvSize * (kMaxSlots * 2);
    uvBci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(m_device, &uvBci, nullptr, &m_uvBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements uvReq{};
    vkGetBufferMemoryRequirements(m_device, m_uvBuf, &uvReq);
    VkMemoryAllocateInfo uvAlloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    uvAlloc.allocationSize = uvReq.size;
    uvAlloc.memoryTypeIndex = findMemoryType(uvReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (vkAllocateMemory(m_device, &uvAlloc, nullptr, &m_uvMem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(m_device, m_uvBuf, m_uvMem, 0) != VK_SUCCESS) return false;
    if (vkMapMemory(m_device, m_uvMem, 0, uvBci.size, 0, (void**)&m_uvMapped) != VK_SUCCESS) return false;
    m_leftUV.resize((size_t)m_uvSize);
    m_rightUV.resize((size_t)m_uvSize);
    m_initialized = true;
    return true;
}

bool VkYDetile::createImportedImage(int dmaFd, ImportedImage& out) {
    int importFd = dup(dmaFd);
    if (importFd < 0) {
        perror("[VkYDetile] dup");
        return false;
    }

    VkMemoryFdPropertiesKHR fdProps = { VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    if (m_vkGetMemoryFdPropertiesKHR(m_device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                     importFd, &fdProps) != VK_SUCCESS) {
        std::cerr << "[VkYDetile] vkGetMemoryFdPropertiesKHR failed" << std::endl;
        close(importFd);
        return false;
    }

    uint64_t modifier = QCOM_COMPRESSED;
    VkImageDrmFormatModifierListCreateInfoEXT modInfo = {
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT
    };
    modInfo.drmFormatModifierCount = 1;
    modInfo.pDrmFormatModifiers = &modifier;

    VkExternalMemoryImageCreateInfo extImg = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    extImg.pNext = &modInfo;
    extImg.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkImageCreateInfo imgInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imgInfo.pNext = &extImg;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    imgInfo.extent = { (uint32_t)m_width, (uint32_t)m_height, 1 };
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_device, &imgInfo, nullptr, &out.image) != VK_SUCCESS) {
        std::cerr << "[VkYDetile] vkCreateImage failed" << std::endl;
        close(importFd);
        return false;
    }

    VkImageMemoryRequirementsInfo2 reqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
    reqInfo.image = out.image;
    VkMemoryDedicatedRequirements dedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
    VkMemoryRequirements2 req2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedReq };
    vkGetImageMemoryRequirements2(m_device, &reqInfo, &req2);

    uint32_t bits = req2.memoryRequirements.memoryTypeBits & fdProps.memoryTypeBits;
    if (!bits) bits = fdProps.memoryTypeBits;
    uint32_t memType = findMemoryType(bits, 0);

    VkMemoryDedicatedAllocateInfo dedAlloc = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    dedAlloc.image = out.image;
    VkImportMemoryFdInfoKHR import = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
    import.pNext = &dedAlloc;
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import.fd = importFd;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.pNext = &import;
    mai.allocationSize = req2.memoryRequirements.size ? req2.memoryRequirements.size : m_dmaSize;
    mai.memoryTypeIndex = memType;
    if (vkAllocateMemory(m_device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
        std::cerr << "[VkYDetile] vkAllocateMemory import failed" << std::endl;
        vkDestroyImage(m_device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        close(importFd);
        return false;
    }
    if (vkBindImageMemory(m_device, out.image, out.memory, 0) != VK_SUCCESS) {
        std::cerr << "[VkYDetile] vkBindImageMemory failed" << std::endl;
        return false;
    }
    return true;
}

bool VkYDetile::importStereoGroups(const std::vector<int>& leftFds, const std::vector<int>& rightFds) {
    if (!m_initialized && !initialize()) return false;
    for (int i = 0; i < kMaxSlots; i++) {
        destroyImage(m_leftSlots[i]);
        destroyImage(m_rightSlots[i]);
        m_prevLeftHash[i] = 0;
        m_prevRightHash[i] = 0;
    }
    m_nLeft = 0;
    m_nRight = 0;
    m_imported = false;

    for (int fd : leftFds) {
        if (m_nLeft >= kMaxSlots) break;
        if (!createImportedImage(fd, m_leftSlots[m_nLeft])) return false;
        m_nLeft++;
    }
    for (int fd : rightFds) {
        if (m_nRight >= kMaxSlots) break;
        if (!createImportedImage(fd, m_rightSlots[m_nRight])) return false;
        m_nRight++;
    }
    if (m_nLeft < 1 || m_nRight < 1) return false;
    m_imported = true;
    std::cout << "[VkYDetile] Imported " << m_nLeft << " left + " << m_nRight
              << " right QCOM UBWC NV12 slots" << std::endl;
    return true;
}

bool VkYDetile::copyYPlanes(bool includeUV, bool leftOnly) {
    m_uvCurrent = false;
    m_rightCurrent = false;
    if (!m_imported || !m_hostMapped) return false;
    const int total = m_nLeft + m_nRight;
    if (total <= 0) return false;

    vkResetCommandBuffer(m_cmd, 0);
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(m_cmd, &begin) != VK_SUCCESS) return false;

    auto transition = [&](VkImage img, VkImageAspectFlags aspect,
                          VkImageLayout oldLayout, VkImageLayout newLayout,
                          uint32_t srcFamily, uint32_t dstFamily,
                          VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                          VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        bar.srcAccessMask = srcAccess;
        bar.dstAccessMask = dstAccess;
        bar.oldLayout = oldLayout;
        bar.newLayout = newLayout;
        bar.srcQueueFamilyIndex = srcFamily;
        bar.dstQueueFamilyIndex = dstFamily;
        bar.image = img;
        bar.subresourceRange.aspectMask = aspect;
        bar.subresourceRange.levelCount = 1;
        bar.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(m_cmd, srcStage, dstStage, 0,
                             0, nullptr, 0, nullptr, 1, &bar);
    };
    auto copyY = [&](VkImage img, VkDeviceSize offset) {
        VkBufferImageCopy region{};
        region.bufferOffset = offset;
        region.bufferRowLength = m_width;
        region.bufferImageHeight = m_height;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { (uint32_t)m_width, (uint32_t)m_height, 1 };
        vkCmdCopyImageToBuffer(m_cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_hostBuf, 1, &region);
    };
    auto copyUV = [&](VkImage img, VkDeviceSize offset) {
        VkBufferImageCopy region{};
        region.bufferOffset = offset;
        region.bufferRowLength = (uint32_t)m_uvW;
        region.bufferImageHeight = (uint32_t)m_uvH;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { (uint32_t)m_uvW, (uint32_t)m_uvH, 1 };
        vkCmdCopyImageToBuffer(m_cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_uvBuf, 1, &region);
    };
    auto copyImage = [&](VkImage img, int idx) {
        transition(img, VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_QUEUE_FAMILY_EXTERNAL, m_queueFamily,
                   0, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        copyY(img, m_ySize * idx);
        if (includeUV) copyUV(img, m_uvSize * idx);
        transition(img, VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                   m_queueFamily, VK_QUEUE_FAMILY_EXTERNAL,
                   VK_ACCESS_TRANSFER_READ_BIT, 0,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    };

    int idx = 0;
    for (int i = 0; i < m_nLeft; i++, idx++) {
        copyImage(m_leftSlots[i].image, idx);
    }
    for (int i = 0; !leftOnly && i < m_nRight; i++, idx++) {
        copyImage(m_rightSlots[i].image, idx);
    }

    VkBufferMemoryBarrier bb = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bb.buffer = m_hostBuf;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &bb, 0, nullptr);
    VkBufferMemoryBarrier ub = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    ub.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ub.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    ub.buffer = m_uvBuf;
    ub.size = VK_WHOLE_SIZE;
    if (includeUV)
        vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &ub, 0, nullptr);

    if (vkEndCommandBuffer(m_cmd) != VK_SUCCESS) return false;
    VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &m_cmd;
    if (vkQueueSubmit(m_queue, 1, &sub, VK_NULL_HANDLE) != VK_SUCCESS) return false;
    if (vkQueueWaitIdle(m_queue) != VK_SUCCESS) return false;

    VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    range.memory = m_hostMem;
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(m_device, 1, &range);
    VkMappedMemoryRange uvRange = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    uvRange.memory = m_uvMem;
    uvRange.size = VK_WHOLE_SIZE;
    if (includeUV) vkInvalidateMappedMemoryRanges(m_device, 1, &uvRange);

    auto pick = [&](int n, int baseIdx, uint64_t* prev, int last, int* changedOut) {
        int chosen = last;
        int changed = -1;
        for (int i = 0; i < n; i++) {
            const uint8_t* plane = m_hostMapped + m_ySize * (baseIdx + i);
            uint64_t h = hashPlane(plane, (size_t)m_ySize);
            int center = plane[(m_height / 2) * m_width + m_width / 2];
            bool empty = (center == 0 && plane[0] == 0 && plane[(size_t)m_ySize / 2] == 0);
            if (!empty && h != prev[i]) changed = i;
            prev[i] = h;
        }
        // All planes from every slot were captured by one queue submission, so
        // the changed slot is already a coherent snapshot and needs no guessed
        // predecessor or extra frame of delay.
        if (changed >= 0) {
            chosen = changed;
        }
        if (chosen < 0 || chosen >= n) chosen = 0;
        if (changedOut) *changedOut = changed;
        return chosen;
    };

    int leftChanged = -1, rightChanged = -1;
    m_curLeft = pick(m_nLeft, 0, m_prevLeftHash, m_curLeft, &leftChanged);
    if (!leftOnly) m_curRight = pick(m_nRight, m_nLeft, m_prevRightHash, m_curRight, &rightChanged);
    m_leftHash = m_prevLeftHash[m_curLeft];
    m_rightHash = m_prevRightHash[m_curRight];
    std::memcpy(m_leftY.data(), m_hostMapped + m_ySize * m_curLeft, (size_t)m_ySize);
    if (!leftOnly) std::memcpy(m_rightY.data(), m_hostMapped + m_ySize * (m_nLeft + m_curRight), (size_t)m_ySize);
    if (includeUV) {
        std::memcpy(m_leftUV.data(), m_uvMapped + m_uvSize * m_curLeft, (size_t)m_uvSize);
        if (!leftOnly) std::memcpy(m_rightUV.data(),
                    m_uvMapped + m_uvSize * (m_nLeft + m_curRight), (size_t)m_uvSize);
        m_uvCurrent = true;
    }
    m_rightCurrent = !leftOnly;
    return true;
}

void VkYDetile::release() {
    m_uvCurrent = false;
    m_imported = false;
    m_initialized = false;
    m_nLeft = 0;
    m_nRight = 0;
    if (m_device) {
        vkDeviceWaitIdle(m_device);
        for (int i = 0; i < kMaxSlots; i++) {
            destroyImage(m_leftSlots[i]);
            destroyImage(m_rightSlots[i]);
        }
        if (m_hostMapped) vkUnmapMemory(m_device, m_hostMem);
        if (m_hostBuf) vkDestroyBuffer(m_device, m_hostBuf, nullptr);
        if (m_hostMem) vkFreeMemory(m_device, m_hostMem, nullptr);
        if (m_uvMapped) vkUnmapMemory(m_device, m_uvMem);
        if (m_uvBuf) vkDestroyBuffer(m_device, m_uvBuf, nullptr);
        if (m_uvMem) vkFreeMemory(m_device, m_uvMem, nullptr);
        if (m_cmdPool) vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        vkDestroyDevice(m_device, nullptr);
    }
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_phys = VK_NULL_HANDLE;
    m_cmdPool = VK_NULL_HANDLE;
    m_cmd = VK_NULL_HANDLE;
    m_hostBuf = VK_NULL_HANDLE;
    m_hostMem = VK_NULL_HANDLE;
    m_hostMapped = nullptr;
    m_uvBuf = VK_NULL_HANDLE;
    m_uvMem = VK_NULL_HANDLE;
    m_uvMapped = nullptr;
}

} // namespace FrameSpatial
