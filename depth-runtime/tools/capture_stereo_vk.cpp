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

// Shader to sample the Y plane from 2464x2464 and write into a linear buffer
// Or we can just sample G8_B8R8_2PLANE_420_UNORM plane 0 (Y plane) into an R8 buffer!
// Let's create an RGBA output image or write a compute shader.
// Actually, let's look at VkImageSubresource:
// plane 0 is the Y plane (2464x2464, R8_UNORM equivalent).

int main() {
    printf("[StereoCapture] Initializing Vulkan...\n");
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

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    // Command pool & buffer
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = 0;
    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &cpci, NULL, &cmdPool);

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbai, &cmd);

    // Grab compositor pidfd
    int pfd = syscall(__NR_pidfd_open, 59153, 0);

    int camera_fds[] = { 189, 197 }; // Right slot 0, Left slot 0
    const char *out_names[] = { "/home/steamos/test_openvr/right_view.ppm", "/home/steamos/test_openvr/left_view.ppm" };

    // Linear destination buffer (2464 * 2464 bytes for Y plane)
    size_t y_size = 2464 * 2464;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = y_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer dstBuf;
    vkCreateBuffer(device, &bci, NULL, &dstBuf);

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, dstBuf, &reqs);

    // Find host visible memory
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    uint32_t hostMemType = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((reqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            hostMemType = i;
            break;
        }
    }

    VkMemoryAllocateInfo dstAlloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    dstAlloc.allocationSize = reqs.size;
    dstAlloc.memoryTypeIndex = hostMemType;
    VkDeviceMemory dstMem;
    vkAllocateMemory(device, &dstAlloc, NULL, &dstMem);
    vkBindBufferMemory(device, dstBuf, dstMem, 0);

    for (int cam = 0; cam < 2; cam++) {
        int dma_fd = syscall(__NR_pidfd_getfd, pfd, camera_fds[cam], 0);
        printf("[StereoCapture] Importing camera %d (fd=%d)...\n", cam, dma_fd);

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
        imageInfo.extent = { 2464, 2464, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage srcImg;
        vkCreateImage(device, &imageInfo, NULL, &srcImg);

        VkImportMemoryFdInfoKHR importInfo = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
        importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importInfo.fd = dma_fd;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = &importInfo;
        allocInfo.allocationSize = 9564160;
        allocInfo.memoryTypeIndex = 0;

        VkDeviceMemory srcMem;
        vkAllocateMemory(device, &allocInfo, NULL, &srcMem);
        vkBindImageMemory(device, srcImg, srcMem, 0);

        // Copy Y plane (plane 0) to buffer
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &cbbi);

        // Transition image layout to TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.image = srcImg;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, NULL, 0, NULL, 1, &barrier);

        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 2464;
        region.bufferImageHeight = 2464;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { 2464, 2464, 1 };

        vkCmdCopyImageToBuffer(cmd, srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuf, 1, &region);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        // Map and save PPM
        void *data;
        vkMapMemory(device, dstMem, 0, y_size, 0, &data);

        FILE *fp = fopen(out_names[cam], "wb");
        if (fp) {
            fprintf(fp, "P5\n2464 2464\n255\n");
            fwrite(data, 1, y_size, fp);
            fclose(fp);
            printf("[StereoCapture] Saved %s!\n", out_names[cam]);
        }
        vkUnmapMemory(device, dstMem);

        vkFreeMemory(device, srcMem, NULL);
        vkDestroyImage(device, srcImg, NULL);
    }

    close(pfd);
    vkFreeMemory(device, dstMem, NULL);
    vkDestroyBuffer(device, dstBuf, NULL);
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
