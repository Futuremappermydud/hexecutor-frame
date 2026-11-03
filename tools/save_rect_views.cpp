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

    int pfd = syscall(__NR_pidfd_open, 59153, 0);
    int raw_fds[] = { 197, 189 }; // Left (197), Right (189)

    VkImage rawImgs[2];
    VkDeviceMemory rawMems[2];
    VkImageView rawViews[2];

    uint64_t modifier = 0x500000000000001ULL;
    VkImageDrmFormatModifierListCreateInfoEXT modCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT };
    modCreateInfo.drmFormatModifierCount = 1;
    modCreateInfo.pDrmFormatModifiers = &modifier;

    VkExternalMemoryImageCreateInfo extImageInfo = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    extImageInfo.pNext = &modCreateInfo;
    extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler;
    vkCreateSampler(device, &sci, NULL, &sampler);

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
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
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

        VkImageViewCreateInfo ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        ivci.image = rawImgs[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        ivci.subresourceRange.baseMipLevel = 0;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.baseArrayLayer = 0;
        ivci.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &ivci, NULL, &rawViews[i]);
    }
    close(pfd);

    const uint32_t W = 616, H = 616;
    auto createStorageImage = [&](VkFormat fmt, VkImage &img, VkDeviceMemory &mem, VkImageView &view) {
        VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = { W, H, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(device, &ici, NULL, &img);

        VkMemoryRequirements reqs;
        vkGetImageMemoryRequirements(device, img, &reqs);

        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        uint32_t memType = 0;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((reqs.memoryTypeBits & (1 << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memType = i; break;
            }
        }
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = reqs.size;
        mai.memoryTypeIndex = memType;
        vkAllocateMemory(device, &mai, NULL, &mem);
        vkBindImageMemory(device, img, mem, 0);

        VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &vci, NULL, &view);
    };

    VkImage rectL_img, rectR_img;
    VkDeviceMemory rectL_mem, rectR_mem;
    VkImageView rectL_view, rectR_view;
    createStorageImage(VK_FORMAT_R8_UNORM, rectL_img, rectL_mem, rectL_view);
    createStorageImage(VK_FORMAT_R8_UNORM, rectR_img, rectR_mem, rectR_view);

    // Build Rectify Pipeline
    auto code = readFile("/home/steamos/test_openvr/rectify.spv");
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = code.size();
    smci.pCode = (const uint32_t*)code.data();
    VkShaderModule mod;
    vkCreateShaderModule(device, &smci, NULL, &mod);

    VkDescriptorSetLayoutBinding bindings[4] = {};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };

    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 4;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout rectDSLayout;
    vkCreateDescriptorSetLayout(device, &dslci, NULL, &rectDSLayout);

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &rectDSLayout;
    VkPipelineLayout rectPLayout;
    vkCreatePipelineLayout(device, &plci, NULL, &rectPLayout);

    VkComputePipelineCreateInfo cpci_p = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci_p.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci_p.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci_p.stage.module = mod;
    cpci_p.stage.pName = "main";
    cpci_p.layout = rectPLayout;
    VkPipeline rectPipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci_p, NULL, &rectPipeline);

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 }
    };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &dpci, NULL, &descPool);

    VkDescriptorSet rectDS;
    VkDescriptorSetAllocateInfo dsai1 = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai1.descriptorPool = descPool;
    dsai1.descriptorSetCount = 1;
    dsai1.pSetLayouts = &rectDSLayout;
    vkAllocateDescriptorSets(device, &dsai1, &rectDS);

    VkDescriptorImageInfo rawLInfo = { sampler, rawViews[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo rawRInfo = { sampler, rawViews[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo rectLInfo = { VK_NULL_HANDLE, rectL_view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo rectRInfo = { VK_NULL_HANDLE, rectR_view, VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet rectWrites[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, rectDS, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rawLInfo },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, rectDS, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rawRInfo },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, rectDS, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rectLInfo },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, rectDS, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rectRInfo }
    };
    vkUpdateDescriptorSets(device, 4, rectWrites, 0, NULL);

    // Readback buffer
    size_t rectSize = W * H;
    VkBufferCreateInfo rbBci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    rbBci.size = rectSize;
    rbBci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readbackBuf;
    vkCreateBuffer(device, &rbBci, NULL, &readbackBuf);

    VkMemoryRequirements rbReqs;
    vkGetBufferMemoryRequirements(device, readbackBuf, &rbReqs);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t hostMemType = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((rbReqs.memoryTypeBits & (1 << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            hostMemType = i; break;
        }
    }
    VkMemoryAllocateInfo rbMai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    rbMai.allocationSize = rbReqs.size;
    rbMai.memoryTypeIndex = hostMemType;
    VkDeviceMemory readbackMem;
    vkAllocateMemory(device, &rbMai, NULL, &readbackMem);
    vkBindBufferMemory(device, readbackBuf, readbackMem, 0);

    // Record
    VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &cbbi);

    auto transition = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL, VkAccessFlags srcA, VkAccessFlags dstA, VkImageAspectFlags aspect) {
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

    transition(rawImgs[0], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_PLANE_0_BIT);
    transition(rawImgs[1], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_PLANE_0_BIT);
    transition(rectL_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    transition(rectR_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rectPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rectPLayout, 0, 1, &rectDS, 0, NULL);
    vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);

    transition(rectL_img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    VkBufferImageCopy bic = {};
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent = { W, H, 1 };
    vkCmdCopyImageToBuffer(cmd, rectL_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuf, 1, &bic);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    uint8_t *pData;
    vkMapMemory(device, readbackMem, 0, rectSize, 0, (void**)&pData);
    FILE *fp = fopen("/home/steamos/test_openvr/rect_left.ppm", "wb");
    if (fp) {
        fprintf(fp, "P5\n%u %u\n255\n", W, H);
        fwrite(pData, 1, rectSize, fp);
        fclose(fp);
        printf("Saved /home/steamos/test_openvr/rect_left.ppm\n");
    }

    int nonZero = 0;
    for (size_t i = 0; i < rectSize; i++) if (pData[i] > 0) nonZero++;
    printf("Rectified Left non-zero pixels: %d / %zu (%.1f%%)\n", nonZero, rectSize, nonZero * 100.0f / rectSize);

    vkUnmapMemory(device, readbackMem);
    return 0;
}
