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
    // 1. Vulkan instance
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
        "VK_EXT_image_drm_format_modifier"
    };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = devExts;
    VkDevice device;
    VkResult res = vkCreateDevice(phys, &dci, NULL, &device);
    printf("vkCreateDevice (with drm modifier ext): %d\n", res);
    if (res != VK_SUCCESS) return 1;

    // Get function pointers
    auto vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR");
    if (!vkGetMemoryFdPropertiesKHR) {
        printf("vkGetMemoryFdPropertiesKHR not found\n");
        return 1;
    }

    // Grab FD 189 from compositor
    int pfd = syscall(__NR_pidfd_open, 59153, 0);
    int dma_fd = syscall(__NR_pidfd_getfd, pfd, 189, 0);
    close(pfd);
    printf("Acquired DMA-BUF fd=%d\n", dma_fd);

    VkMemoryFdPropertiesKHR fdProps = { VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    res = vkGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dma_fd, &fdProps);
    printf("vkGetMemoryFdPropertiesKHR: %d, memoryTypeBits = 0x%x\n", res, fdProps.memoryTypeBits);

    // Query supported modifiers for G8_B8R8_2PLANE_420_UNORM
    auto vkGetPhysicalDeviceFormatProperties2 = (PFN_vkGetPhysicalDeviceFormatProperties2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFormatProperties2KHR");
    if (!vkGetPhysicalDeviceFormatProperties2) vkGetPhysicalDeviceFormatProperties2 = vkGetPhysicalDeviceFormatProperties2;

    VkDrmFormatModifierPropertiesListEXT modPropsList = { VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT };
    VkFormatProperties2 fmtProps2 = { VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2 };
    fmtProps2.pNext = &modPropsList;

    vkGetPhysicalDeviceFormatProperties2(phys, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fmtProps2);
    std::vector<VkDrmFormatModifierPropertiesEXT> modProps(modPropsList.drmFormatModifierCount);
    modPropsList.pDrmFormatModifierProperties = modProps.data();
    vkGetPhysicalDeviceFormatProperties2(phys, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fmtProps2);

    printf("Supported DRM Format Modifiers for G8_B8R8_2PLANE_420_UNORM (%u):\n", modPropsList.drmFormatModifierCount);
    for (uint32_t i = 0; i < modPropsList.drmFormatModifierCount; i++) {
        printf("  [%u] modifier = 0x%llx, planes = %u, features = 0x%x\n",
               i, (unsigned long long)modProps[i].drmFormatModifier,
               modProps[i].drmFormatModifierPlaneCount,
               modProps[i].drmFormatModifierTilingFeatures);
    }

    close(dma_fd);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
