#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <vulkan/vulkan.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

static const uint64_t QCOM_COMPRESSED = 0x500000000000001ULL;
static const uint32_t CAM_W = 2464;
static const uint32_t CAM_H = 2464;
static const VkDeviceSize DMA_SIZE = 9564160;

#define CHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "FAIL %s:%d %s -> %d\n", __FILE__, __LINE__, #expr, (int)_r); return 1; } } while (0)

static pid_t findPid() {
    FILE *p = popen("pgrep -n vrcompositor", "r");
    if (!p) return 0;
    pid_t pid = 0;
    if (fscanf(p, "%d", &pid) != 1) pid = 0;
    pclose(p);
    return pid;
}

static std::vector<int> findCamFds(pid_t pid) {
    std::vector<int> fds;
    char dirPath[256];
    snprintf(dirPath, sizeof(dirPath), "/proc/%d/fd", pid);
    DIR *d = opendir(dirPath);
    if (!d) return fds;
    while (dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        char linkPath[512], target[512] = {0};
        snprintf(linkPath, sizeof(linkPath), "/proc/%d/fd/%s", pid, e->d_name);
        ssize_t n = readlink(linkPath, target, sizeof(target) - 1);
        if (n <= 0) continue;
        target[n] = 0;
        struct stat st;
        if (strstr(target, "dmabuf") && stat(linkPath, &st) == 0 && st.st_size == (off_t)DMA_SIZE) {
            fds.push_back(atoi(e->d_name));
        }
    }
    closedir(d);
    std::sort(fds.begin(), fds.end());
    return fds;
}

static void stats(const char *label, const uint8_t *p, size_t n) {
    double sum = 0, sum2 = 0;
    int mn = 255, mx = 0;
    uint32_t hist[256] = {0};
    size_t step = n / 200000 + 1;
    size_t count = 0;
    double corr = 0;
    size_t corrN = 0;
    for (size_t i = 0; i < n; i += step) {
        int v = p[i];
        hist[v]++;
        sum += v; sum2 += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        count++;
        if (i + 1 < n) { corr += std::abs((int)p[i] - (int)p[i + 1]); corrN++; }
    }
    double mean = sum / count;
    double var = sum2 / count - mean * mean;
    double ent = 0;
    for (int i = 0; i < 256; i++) {
        if (!hist[i]) continue;
        double pr = (double)hist[i] / count;
        ent -= pr * std::log2(pr);
    }
    printf("[%s] mean=%.1f std=%.1f min=%d max=%d entropy=%.3f neighAbsDiff=%.2f n=%zu\n",
           label, mean, std::sqrt(std::max(0.0, var)), mn, mx, ent,
           corrN ? corr / corrN : 0, count);
}

int main() {
    pid_t pid = findPid();
    if (!pid) { fprintf(stderr, "no vrcompositor\n"); return 1; }
    auto camFds = findCamFds(pid);
    printf("compositor pid=%d camFds=%zu:", pid, camFds.size());
    for (int fd : camFds) printf(" %d", fd);
    printf("\n");
    if (camFds.empty()) return 1;

    int pfd = syscall(__NR_pidfd_open, pid, 0);
    if (pfd < 0) { perror("pidfd_open"); return 1; }

    const char *instExts[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME
    };
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instExts;
    VkInstance instance;
    CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    for (auto g : gpus) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(g, &props);
        printf("GPU: %s type=%d\n", props.deviceName, props.deviceType);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && phys == VK_NULL_HANDLE)
            phys = g;
    }
    if (phys == VK_NULL_HANDLE) phys = gpus[0];

    auto vkGetPhysFmt2 = (PFN_vkGetPhysicalDeviceFormatProperties2)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFormatProperties2");
    VkDrmFormatModifierPropertiesListEXT modList = { VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT };
    VkFormatProperties2 fp2 = { VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &modList };
    vkGetPhysFmt2(phys, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fp2);
    std::vector<VkDrmFormatModifierPropertiesEXT> mods(modList.drmFormatModifierCount);
    modList.pDrmFormatModifierProperties = mods.data();
    vkGetPhysFmt2(phys, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fp2);
    printf("NV12 modifiers (%u):\n", modList.drmFormatModifierCount);
    uint32_t qcomPlanes = 2;
    for (auto &m : mods) {
        printf("  mod=0x%llx planes=%u features=0x%x\n",
               (unsigned long long)m.drmFormatModifier, m.drmFormatModifierPlaneCount,
               m.drmFormatModifierTilingFeatures);
        if (m.drmFormatModifier == QCOM_COMPRESSED) qcomPlanes = m.drmFormatModifierPlaneCount;
    }

    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, qps.data());
    uint32_t qfam = 0;
    for (uint32_t i = 0; i < qCount; i++) {
        if (qps[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) {
            qfam = i; break;
        }
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qfam;
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
    VkDevice device;
    CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, qfam, 0, &queue);

    auto vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR");

    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qfam;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(device, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

    const VkDeviceSize ySize = (VkDeviceSize)CAM_W * CAM_H;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = ySize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer dstBuf;
    CHECK(vkCreateBuffer(device, &bci, nullptr, &dstBuf));
    VkMemoryRequirements bufReq{};
    vkGetBufferMemoryRequirements(device, dstBuf, &bufReq);
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    uint32_t hostType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((bufReq.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            hostType = i; break;
        }
    }
    printf("host mem type=%u flags=0x%x bufSize=%llu\n", hostType,
           memProps.memoryTypes[hostType].propertyFlags, (unsigned long long)bufReq.size);
    VkMemoryAllocateInfo hostAlloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    hostAlloc.allocationSize = bufReq.size;
    hostAlloc.memoryTypeIndex = hostType;
    VkDeviceMemory hostMem;
    CHECK(vkAllocateMemory(device, &hostAlloc, nullptr, &hostMem));
    CHECK(vkBindBufferMemory(device, dstBuf, hostMem, 0));
    void *mapped = nullptr;
    CHECK(vkMapMemory(device, hostMem, 0, ySize, 0, &mapped));

    int slots = std::min((int)camFds.size(), 8);
    for (int i = 0; i < slots; i++) {
        int dma = syscall(__NR_pidfd_getfd, pfd, camFds[i], 0);
        if (dma < 0) { perror("pidfd_getfd"); continue; }

        VkMemoryFdPropertiesKHR fdProps = { VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
        VkResult fdRes = vkGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dma, &fdProps);
        printf("\nslot %d compositor_fd=%d local_fd=%d getFdProps=%d memTypes=0x%x qcomPlanes=%u\n",
               i, camFds[i], dma, (int)fdRes, fdProps.memoryTypeBits, qcomPlanes);

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
        imgInfo.extent = { CAM_W, CAM_H, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage img;
        VkResult cr = vkCreateImage(device, &imgInfo, nullptr, &img);
        printf("  vkCreateImage(LIST)=%d\n", (int)cr);
        if (cr != VK_SUCCESS) { close(dma); continue; }

        VkImageMemoryRequirementsInfo2 reqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
        reqInfo.image = img;
        VkMemoryDedicatedRequirements dedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
        VkMemoryRequirements2 req2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedReq };
        vkGetImageMemoryRequirements2(device, &reqInfo, &req2);
        printf("  imgReq size=%llu align=%llu bits=0x%x dedicated req=%d pref=%d\n",
               (unsigned long long)req2.memoryRequirements.size,
               (unsigned long long)req2.memoryRequirements.alignment,
               req2.memoryRequirements.memoryTypeBits,
               (int)dedReq.requiresDedicatedAllocation, (int)dedReq.prefersDedicatedAllocation);

        uint32_t memType = UINT32_MAX;
        uint32_t bits = req2.memoryRequirements.memoryTypeBits & fdProps.memoryTypeBits;
        if (!bits) bits = fdProps.memoryTypeBits ? fdProps.memoryTypeBits : req2.memoryRequirements.memoryTypeBits;
        for (uint32_t t = 0; t < memProps.memoryTypeCount; t++) {
            if (bits & (1u << t)) { memType = t; break; }
        }
        printf("  import memType=%u\n", memType);

        VkMemoryDedicatedAllocateInfo dedAlloc = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        dedAlloc.image = img;
        VkImportMemoryFdInfoKHR import = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
        import.pNext = &dedAlloc;
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        import.fd = dma;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.pNext = &import;
        mai.allocationSize = req2.memoryRequirements.size ? req2.memoryRequirements.size : DMA_SIZE;
        mai.memoryTypeIndex = memType;
        VkDeviceMemory mem;
        VkResult ar = vkAllocateMemory(device, &mai, nullptr, &mem);
        printf("  vkAllocateMemory(import)=%d\n", (int)ar);
        if (ar != VK_SUCCESS) {
            vkDestroyImage(device, img, nullptr);
            close(dma);
            continue;
        }
        CHECK(vkBindImageMemory(device, img, mem, 0));

        VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        VkImageMemoryBarrier bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        bar.srcAccessMask = 0;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        bar.dstQueueFamilyIndex = qfam;
        bar.image = img;
        bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        bar.subresourceRange.levelCount = 1;
        bar.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &bar);
        VkBufferImageCopy region{};
        region.bufferRowLength = CAM_W;
        region.bufferImageHeight = CAM_H;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { CAM_W, CAM_H, 1 };
        vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuf, 1, &region);
        VkBufferMemoryBarrier bb = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        bb.buffer = dstBuf;
        bb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &bb, 0, nullptr);
        CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cmd;
        CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(queue));
        VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
        range.memory = hostMem;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device, 1, &range);

        auto *y = (const uint8_t*)mapped;
        char label[64];
        snprintf(label, sizeof(label), "slot%d", i);
        stats(label, y, (size_t)ySize);

        if (i == 0 || i == 4) {
            char path[64];
            snprintf(path, sizeof(path), "/tmp/detile_%s.pgm", i == 0 ? "right" : "left");
            FILE *f = fopen(path, "wb");
            if (f) {
                fprintf(f, "P5\n%u %u\n255\n", CAM_W, CAM_H);
                fwrite(y, 1, (size_t)ySize, f);
                fclose(f);
                printf("  wrote %s\n", path);
            }
        }

        vkDestroyImage(device, img, nullptr);
        vkFreeMemory(device, mem, nullptr);
    }

    close(pfd);
    printf("done\n");
    return 0;
}
