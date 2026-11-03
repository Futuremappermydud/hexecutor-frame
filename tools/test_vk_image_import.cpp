#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>
#include <vulkan/vulkan.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

int main() {
    const char *instExts[] = {
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_get_physical_device_properties2"
    };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instExts;
    VkInstance instance;
    vkCreateInstance(&ici, NULL, &instance);

    uint32_t gpuCount = 1;
    VkPhysicalDevice phys;
    vkEnumeratePhysicalDevices(instance, &gpuCount, &phys);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char *devExts[] = {
        "VK_KHR_external_memory",
        "VK_KHR_external_memory_fd",
        "VK_EXT_image_drm_format_modifier",
        "VK_KHR_dedicated_allocation",
        "VK_KHR_get_memory_requirements2"
    };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 5;
    dci.ppEnabledExtensionNames = devExts;
    VkDevice device;
    vkCreateDevice(phys, &dci, NULL, &device);

    auto vkCreateImage = (PFN_vkCreateImage)vkGetDeviceProcAddr(device, "vkCreateImage");
    auto vkAllocateMemory = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device, "vkAllocateMemory");
    auto vkBindImageMemory = (PFN_vkBindImageMemory)vkGetDeviceProcAddr(device, "vkBindImageMemory");

    // Grab FD 189
    int pfd = syscall(__NR_pidfd_open, 59153, 0);
    int dma_fd = syscall(__NR_pidfd_getfd, pfd, 189, 0);
    close(pfd);

    // Create Image with DRM format modifier
    uint64_t modifier = 0x500000000000001ULL;
    VkImageDrmFormatModifierListCreateInfoEXT modCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT };
    modCreateInfo.drmFormatModifierCount = 1;
    modCreateInfo.pDrmFormatModifiers = &modifier;

    VkExternalMemoryImageCreateInfo extImageInfo = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    extImageInfo.pNext = &modCreateInfo;
    extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.pNext = &extImageInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    imageInfo.extent.width = 2464;
    imageInfo.extent.height = 2464;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img;
    VkResult res = vkCreateImage(device, &imageInfo, NULL, &img);
    printf("vkCreateImage with QCOM_COMPRESSED modifier: %d\n", res);

    if (res == VK_SUCCESS) {
        // Import memory
        VkImportMemoryFdInfoKHR importInfo = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
        importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importInfo.fd = dma_fd;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = &importInfo;
        allocInfo.allocationSize = 9564160;
        allocInfo.memoryTypeIndex = 0; // memoryTypeBits was 0x7, index 0 is valid

        VkDeviceMemory mem;
        res = vkAllocateMemory(device, &allocInfo, NULL, &mem);
        printf("vkAllocateMemory (import dma_buf): %d\n", res);

        if (res == VK_SUCCESS) {
            res = vkBindImageMemory(device, img, mem, 0);
            printf("vkBindImageMemory: %d\n", res);
            vkFreeMemory(device, mem, NULL);
        }
        vkDestroyImage(device, img, NULL);
    } else {
        close(dma_fd);
    }

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
