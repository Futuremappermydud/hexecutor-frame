#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <sys/syscall.h>
#include <vulkan/vulkan.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

static std::vector<char> readFile(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return {};
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(size);
    fread(buf.data(), 1, size, fp);
    fclose(fp);
    return buf;
}

int main() {
    printf("[RectifySSBO] Starting...\n");
    const char *instExts[] = { "VK_KHR_external_memory_capabilities", "VK_KHR_get_physical_device_properties2" };
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
        "VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_EXT_image_drm_format_modifier",
        "VK_KHR_dedicated_allocation", "VK_KHR_get_memory_requirements2"
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

    // Import DMA-BUFs (Left = 197, Right = 189)
    int pfd = syscall(__NR_pidfd_open, 59153, 0);
    int raw_fds[] = { 197, 189 };

    VkImage rawImgs[2];
    VkDeviceMemory rawMems[2];

    uint64_t modifier = 0x500000000000001ULL;
    VkImageDrmFormatModifierListCreateInfoEXT modCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT };
    modCreateInfo.drmFormatModifierCount = 1;
    modCreateInfo.pDrmFormatModifiers = &modifier;

    VkExternalMemoryImageCreateInfo extImageInfo = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    extImageInfo.pNext = &modCreateInfo;
    extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    for (int i = 0; i < 2; i++) {
        int dma_fd = syscall(__NR_pidfd_getfd, pfd, raw_fds[i], 0);

        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.pNext = &extImageInfo;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        imageInfo.extent = { 2464, 2464, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(device, &imageInfo, NULL, &rawImgs[i]);

        VkImportMemoryFdInfoKHR importInfo = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
        importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importInfo.fd = dma_fd;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = &importInfo;
        allocInfo.allocationSize = 9564160;
        allocInfo.memoryTypeIndex = 0;
        vkAllocateMemory(device, &allocInfo, NULL, &rawMems[i]);
        vkBindImageMemory(device, rawImgs[i], rawMems[i], 0);
    }
    close(pfd);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    auto getMemType = [&](uint32_t typeBits, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        return 0u;
    };

    // Raw SSBO Buffers for Left & Right (2464 * 2464 bytes each)
    size_t rawSize = 2464 * 2464;
    VkBuffer rawBufs[2];
    VkDeviceMemory rawBufMems[2];

    for (int i = 0; i < 2; i++) {
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = rawSize;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(device, &bci, NULL, &rawBufs[i]);

        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device, rawBufs[i], &reqs);
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = reqs.size;
        mai.memoryTypeIndex = getMemType(reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &mai, NULL, &rawBufMems[i]);
        vkBindBufferMemory(device, rawBufs[i], rawBufMems[i], 0);
    }

    // Output Rectified Images (616x616 R8_UNORM)
    const uint32_t W = 616, H = 616;
    VkImage rectImgs[2];
    VkDeviceMemory rectMems[2];
    VkImageView rectViews[2];

    for (int i = 0; i < 2; i++) {
        VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8_UNORM;
        ici.extent = { W, H, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(device, &ici, NULL, &rectImgs[i]);

        VkMemoryRequirements reqs;
        vkGetImageMemoryRequirements(device, rectImgs[i], &reqs);
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = reqs.size;
        mai.memoryTypeIndex = getMemType(reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &mai, NULL, &rectMems[i]);
        vkBindImageMemory(device, rectImgs[i], rectMems[i], 0);

        VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = rectImgs[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8_UNORM;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &vci, NULL, &rectViews[i]);
    }

    // Compute Pipeline
    auto code = readFile("/home/steamos/test_openvr/rectify_ssbo.spv");
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = code.size();
    smci.pCode = (const uint32_t*)code.data();
    VkShaderModule mod;
    vkCreateShaderModule(device, &smci, NULL, &mod);

    VkDescriptorSetLayoutBinding bindings[4] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }
    };
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 4;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout dsLayout;
    vkCreateDescriptorSetLayout(device, &dslci, NULL, &dsLayout);

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsLayout;
    VkPipelineLayout pLayout;
    vkCreatePipelineLayout(device, &plci, NULL, &pLayout);

    VkComputePipelineCreateInfo cpci_p = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci_p.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci_p.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci_p.stage.module = mod;
    cpci_p.stage.pName = "main";
    cpci_p.layout = pLayout;
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci_p, NULL, &pipeline);

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 }
    };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &dpci, NULL, &descPool);

    VkDescriptorSet descSet;
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsLayout;
    vkAllocateDescriptorSets(device, &dsai, &descSet);

    VkDescriptorBufferInfo bufInfoL = { rawBufs[0], 0, rawSize };
    VkDescriptorBufferInfo bufInfoR = { rawBufs[1], 0, rawSize };
    VkDescriptorImageInfo rectLInfo = { VK_NULL_HANDLE, rectViews[0], VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo rectRInfo = { VK_NULL_HANDLE, rectViews[1], VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bufInfoL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bufInfoR },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rectLInfo },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rectRInfo }
    };
    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    // Readback buffer (host visible)
    size_t outSize = W * H;
    VkBufferCreateInfo rbci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    rbci.size = outSize;
    rbci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readbackBuf;
    vkCreateBuffer(device, &rbci, NULL, &readbackBuf);

    VkMemoryRequirements rbReqs;
    vkGetBufferMemoryRequirements(device, readbackBuf, &rbReqs);
    VkMemoryAllocateInfo rbMai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    rbMai.allocationSize = rbReqs.size;
    rbMai.memoryTypeIndex = getMemType(rbReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory readbackMem;
    vkAllocateMemory(device, &rbMai, NULL, &readbackMem);
    vkBindBufferMemory(device, readbackBuf, readbackMem, 0);

    // Record Commands
    VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &cbbi);

    auto barrier = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL, VkAccessFlags srcA, VkAccessFlags dstA, VkImageAspectFlags aspect) {
        VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = srcA;
        b.dstAccessMask = dstA;
        b.oldLayout = oldL;
        b.newLayout = newL;
        b.image = img;
        b.subresourceRange.aspectMask = aspect;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    };

    // Copy raw DMA-BUFs plane 0 directly to SSBO buffers
    for (int i = 0; i < 2; i++) {
        barrier(rawImgs[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_PLANE_0_BIT);

        VkBufferImageCopy region = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { 2464, 2464, 1 };
        vkCmdCopyImageToBuffer(cmd, rawImgs[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rawBufs[i], 1, &region);
    }

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    barrier(rectImgs[0], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    barrier(rectImgs[1], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // Dispatch Rectify
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pLayout, 0, 1, &descSet, 0, NULL);
    vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);

    // Copy rectified Left to readback
    barrier(rectImgs[0], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    VkBufferImageCopy bic = {};
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent = { W, H, 1 };
    vkCmdCopyImageToBuffer(cmd, rectImgs[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuf, 1, &bic);

    vkEndCommandBuffer(cmd);

    // Submit
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    // Readback & verify
    uint8_t *pData;
    vkMapMemory(device, readbackMem, 0, outSize, 0, (void**)&pData);
    FILE *fp = fopen("/home/steamos/test_openvr/rect_left_real.ppm", "wb");
    if (fp) {
        fprintf(fp, "P5\n%u %u\n255\n", W, H);
        fwrite(pData, 1, outSize, fp);
        fclose(fp);
        printf("Saved /home/steamos/test_openvr/rect_left_real.ppm!\n");
    }

    int nonZero = 0;
    for (size_t i = 0; i < outSize; i++) if (pData[i] > 0) nonZero++;
    printf("Rectified Left non-zero pixels: %d / %zu (%.1f%%)\n", nonZero, outSize, nonZero * 100.0f / outSize);
    vkUnmapMemory(device, readbackMem);

    return 0;
}
