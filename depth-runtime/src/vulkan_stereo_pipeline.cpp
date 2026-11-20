#include "vulkan_stereo_pipeline.hpp"
#include "calibration.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace FrameSpatial {

namespace {

struct ImageResource {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct ComputePass {
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

std::vector<uint32_t> readSpirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return {};
    file.seekg(0);
    std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
    file.read(reinterpret_cast<char*>(code.data()), bytes);
    return file.good() ? code : std::vector<uint32_t>{};
}

} // namespace

struct VulkanStereoPipeline::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    uint8_t* stagingMapped = nullptr;
    bool stagingCoherent = false;
    VkDeviceSize planeBytes = static_cast<VkDeviceSize>(RECT_WIDTH) * RECT_HEIGHT;
    VkDeviceSize outputOffset = planeBytes * 2;
    VkDeviceSize stagingBytes = outputOffset + planeBytes * sizeof(float);

    ImageResource left;
    ImageResource right;
    ImageResource leftCensus;
    ImageResource rightCensus;
    ImageResource leftDisparity;
    ImageResource rightDisparity;
    ImageResource depth;
    ImageResource filteredDepth;
    ImageResource leftMatch, rightMatch, filterRejection, supportRejection;
    bool diagnosticsEnabled = false;
    std::vector<ImageResource*> diagnosticImages;
    std::vector<ImageResource*> allImages;

    ComputePass censusPass;
    ComputePass matchPass;
    ComputePass filterPass;
    ComputePass supportPass;

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    bool ready = false;
    float computeMs = 0.0f;
    uint64_t frames = 0;
    std::string error;

    ~Impl() { release(); }

    void setError(const std::string& message) {
        error = message;
        std::cerr << "[VulkanStereo] " << message << std::endl;
    }

    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred = 0) const {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) && (flags & required) == required &&
                (flags & preferred) == preferred) return i;
        }
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) && (flags & required) == required) return i;
        }
        return std::numeric_limits<uint32_t>::max();
    }

    bool checkStorageFormat(VkFormat format, const char* name) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
            setError(std::string(name) + " storage images are unsupported");
            return false;
        }
        return true;
    }

    bool createImage(ImageResource& resource, VkFormat format,
                     VkImageUsageFlags extraUsage) {
        resource.format = format;
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = {static_cast<uint32_t>(RECT_WIDTH),
                            static_cast<uint32_t>(RECT_HEIGHT), 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | extraUsage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &imageInfo, nullptr, &resource.image) != VK_SUCCESS) {
            setError("vkCreateImage failed");
            return false;
        }

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, resource.image, &requirements);
        const uint32_t memoryType = findMemoryType(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryType == std::numeric_limits<uint32_t>::max()) {
            setError("no device-local image memory type");
            return false;
        }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType;
        if (vkAllocateMemory(device, &allocation, nullptr, &resource.memory) != VK_SUCCESS ||
            vkBindImageMemory(device, resource.image, resource.memory, 0) != VK_SUCCESS) {
            setError("image memory allocation failed");
            return false;
        }

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = resource.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &viewInfo, nullptr, &resource.view) != VK_SUCCESS) {
            setError("vkCreateImageView failed");
            return false;
        }
        allImages.push_back(&resource);
        return true;
    }

    bool createStagingBuffer() {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = stagingBytes;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
            setError("staging buffer creation failed");
            return false;
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, stagingBuffer, &requirements);
        const uint32_t memoryType = findMemoryType(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memoryType == std::numeric_limits<uint32_t>::max()) {
            setError("no host-visible staging memory type");
            return false;
        }
        stagingCoherent = (memoryProperties.memoryTypes[memoryType].propertyFlags &
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType;
        if (vkAllocateMemory(device, &allocation, nullptr, &stagingMemory) != VK_SUCCESS ||
            vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS ||
            vkMapMemory(device, stagingMemory, 0, stagingBytes, 0,
                        reinterpret_cast<void**>(&stagingMapped)) != VK_SUCCESS) {
            setError("staging memory setup failed");
            return false;
        }
        return true;
    }

    bool createPass(ComputePass& pass, uint32_t bindingCount,
                    const std::string& shaderPath) {
        std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
        for (uint32_t i = 0; i < bindingCount; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo descriptorInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptorInfo.bindingCount = bindingCount;
        descriptorInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &descriptorInfo, nullptr,
                                        &pass.descriptorLayout) != VK_SUCCESS) {
            setError("descriptor layout creation failed");
            return false;
        }
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &pass.descriptorLayout;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr,
                                   &pass.pipelineLayout) != VK_SUCCESS) {
            setError("pipeline layout creation failed");
            return false;
        }

        const auto code = readSpirv(shaderPath);
        if (code.empty()) {
            setError("could not read shader " + shaderPath);
            return false;
        }
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = code.size() * sizeof(uint32_t);
        moduleInfo.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
            setError("shader module creation failed for " + shaderPath);
            return false;
        }
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pass.pipelineLayout;
        const VkResult result = vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pass.pipeline);
        vkDestroyShaderModule(device, module, nullptr);
        if (result != VK_SUCCESS) {
            setError("compute pipeline creation failed for " + shaderPath);
            return false;
        }

        VkDescriptorSetAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pass.descriptorLayout;
        if (vkAllocateDescriptorSets(device, &allocateInfo, &pass.descriptorSet) != VK_SUCCESS) {
            setError("descriptor set allocation failed");
            return false;
        }
        return true;
    }

    void updatePass(const ComputePass& pass,
                    const std::vector<ImageResource*>& resources) {
        std::vector<VkDescriptorImageInfo> imageInfos(resources.size());
        std::vector<VkWriteDescriptorSet> writes(resources.size());
        for (size_t i = 0; i < resources.size(); ++i) {
            imageInfos[i].imageView = resources[i]->view;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = pass.descriptorSet;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &imageInfos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    static VkImageMemoryBarrier imageBarrier(ImageResource& resource,
                                              VkImageLayout oldLayout,
                                              VkImageLayout newLayout,
                                              VkAccessFlags sourceAccess,
                                              VkAccessFlags destinationAccess) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = resource.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstAccessMask = destinationAccess;
        return barrier;
    }

    bool submitAndWait() {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) return false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer;
        if (vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS) return false;
        return vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ULL) == VK_SUCCESS;
    }

    bool initializeLayouts() {
        vkResetFences(device, 1, &fence);
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) return false;
        std::vector<VkImageMemoryBarrier> barriers;
        barriers.reserve(allImages.size());
        for (ImageResource* image : allImages) {
            barriers.push_back(imageBarrier(*image, VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_GENERAL, 0,
                                            VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT));
        }
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, static_cast<uint32_t>(barriers.size()),
                             barriers.data());
        return submitAndWait();
    }

    bool initialize(const std::string& shaderDirectory, bool enableDiagnostics) {
        diagnosticsEnabled = enableDiagnostics;
        if (diagnosticsEnabled) stagingBytes += planeBytes * 13 * sizeof(float);
        VkApplicationInfo applicationInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        applicationInfo.pApplicationName = "frame-vulkan-stereo";
        applicationInfo.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &applicationInfo;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            setError("vkCreateInstance failed");
            return false;
        }

        uint32_t physicalCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
        std::vector<VkPhysicalDevice> devices(physicalCount);
        vkEnumeratePhysicalDevices(instance, &physicalCount, devices.data());
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
                physicalDevice = candidate;
                std::cout << "[VulkanStereo] Using GPU: " << properties.deviceName
                          << std::endl;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE && !devices.empty()) physicalDevice = devices[0];
        if (physicalDevice == VK_NULL_HANDLE) {
            setError("no Vulkan physical device");
            return false;
        }
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        if (!checkStorageFormat(VK_FORMAT_R8_UINT, "R8_UINT") ||
            !checkStorageFormat(VK_FORMAT_R32G32_UINT, "R32G32_UINT") ||
            !checkStorageFormat(VK_FORMAT_R32_SFLOAT, "R32_SFLOAT")) return false;

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());
        bool foundFamily = false;
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamily = i;
                foundFamily = true;
                break;
            }
        }
        if (!foundFamily) {
            setError("no Vulkan compute queue");
            return false;
        }
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            setError("vkCreateDevice failed");
            return false;
        }
        vkGetDeviceQueue(device, queueFamily, 0, &queue);

        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamily;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            setError("command pool creation failed");
            return false;
        }
        VkCommandBufferAllocateInfo commandInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commandInfo.commandPool = commandPool;
        commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &commandInfo, &commandBuffer) != VK_SUCCESS) {
            setError("command buffer allocation failed");
            return false;
        }
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            setError("fence creation failed");
            return false;
        }
        if (!createStagingBuffer()) return false;
        if (!createImage(left, VK_FORMAT_R8_UINT, VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
            !createImage(right, VK_FORMAT_R8_UINT, VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
            !createImage(leftCensus, VK_FORMAT_R32G32_UINT, 0) ||
            !createImage(rightCensus, VK_FORMAT_R32G32_UINT, 0) ||
            !createImage(leftDisparity, VK_FORMAT_R32_SFLOAT, diagnosticsEnabled ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0) ||
            !createImage(rightDisparity, VK_FORMAT_R32_SFLOAT, diagnosticsEnabled ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0) ||
            !createImage(depth, VK_FORMAT_R32_SFLOAT, diagnosticsEnabled ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0) ||
            !createImage(filteredDepth, VK_FORMAT_R32_SFLOAT,
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            return false;
        }

        if (diagnosticsEnabled) {
            if (!checkStorageFormat(VK_FORMAT_R32G32B32A32_SFLOAT, "RGBA32_SFLOAT") ||
                !checkStorageFormat(VK_FORMAT_R32_UINT, "R32_UINT") ||
                !createImage(leftMatch, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
                !createImage(rightMatch, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
                !createImage(filterRejection, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
                !createImage(supportRejection, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return false;
            diagnosticImages = {&leftDisparity, &rightDisparity, &depth, &leftMatch,
                                &rightMatch, &filterRejection, &supportRejection};
        }
        const std::string suffix = diagnosticsEnabled ? "_diagnostic.spv" : ".spv";
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize.descriptorCount = diagnosticsEnabled ? 17 : 13;
        VkDescriptorPoolCreateInfo descriptorPoolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        descriptorPoolInfo.maxSets = 4;
        descriptorPoolInfo.poolSizeCount = 1;
        descriptorPoolInfo.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr,
                                   &descriptorPool) != VK_SUCCESS) {
            setError("descriptor pool creation failed");
            return false;
        }
        if (!createPass(censusPass, 4, shaderDirectory + "/stereo_census.spv") ||
            !createPass(matchPass, diagnosticsEnabled ? 6 : 4, shaderDirectory + "/stereo_match" + suffix) ||
            !createPass(filterPass, diagnosticsEnabled ? 4 : 3, shaderDirectory + "/stereo_filter" + suffix) ||
            !createPass(supportPass, diagnosticsEnabled ? 3 : 2, shaderDirectory + "/stereo_support" + suffix)) {
            return false;
        }
        updatePass(censusPass, {&left, &right, &leftCensus, &rightCensus});
        updatePass(matchPass, {&leftCensus, &rightCensus,
                               &leftDisparity, &rightDisparity});
        updatePass(filterPass, {&leftDisparity, &rightDisparity, &depth});
        updatePass(supportPass, {&depth, &filteredDepth});
        if (diagnosticsEnabled) {
            // Each update starts at binding zero; include all bindings.
            updatePass(matchPass, {&leftCensus, &rightCensus, &leftDisparity,
                                  &rightDisparity, &leftMatch, &rightMatch});
            updatePass(filterPass, {&leftDisparity, &rightDisparity, &depth, &filterRejection});
            updatePass(supportPass, {&depth, &filteredDepth, &supportRejection});
        }
        if (!initializeLayouts()) {
            setError("initial image layout transition failed");
            return false;
        }
        ready = true;
        std::cout << "[VulkanStereo] Census + bidirectional confidence pipeline ready ("
                  << RECT_WIDTH << "x" << RECT_HEIGHT << ", 96 disparities)"
                  << std::endl;
        return true;
    }

    bool process(const std::vector<uint8_t>& rectifiedLeft,
                 const std::vector<uint8_t>& rectifiedRight,
                 std::vector<float>& depthMeters, StereoDiagnostics* diagnostics) {
        if (diagnostics && !diagnosticsEnabled) {
            setError("diagnostics requested without enableDiagnostics initialization");
            return false;
        }
        if (!ready || rectifiedLeft.size() < static_cast<size_t>(planeBytes) ||
            rectifiedRight.size() < static_cast<size_t>(planeBytes)) return false;

        std::memcpy(stagingMapped, rectifiedLeft.data(), static_cast<size_t>(planeBytes));
        std::memcpy(stagingMapped + planeBytes, rectifiedRight.data(),
                    static_cast<size_t>(planeBytes));
        if (!stagingCoherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = stagingMemory;
            range.size = VK_WHOLE_SIZE;
            vkFlushMappedMemoryRanges(device, 1, &range);
        }

        const auto start = std::chrono::steady_clock::now();
        vkResetFences(device, 1, &fence);
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            setError("vkBeginCommandBuffer failed");
            return false;
        }

        VkImageMemoryBarrier uploadBarriers[2] = {
            imageBarrier(left, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT),
            imageBarrier(right, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT)};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             2, uploadBarriers);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {static_cast<uint32_t>(RECT_WIDTH),
                            static_cast<uint32_t>(RECT_HEIGHT), 1};
        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, left.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        copy.bufferOffset = planeBytes;
        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, right.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        VkImageMemoryBarrier computeInputs[2] = {
            imageBarrier(left, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            imageBarrier(right, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             2, computeInputs);

        const uint32_t groupsX = (RECT_WIDTH + 15) / 16;
        const uint32_t groupsY = (RECT_HEIGHT + 15) / 16;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, censusPass.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                censusPass.pipelineLayout, 0, 1,
                                &censusPass.descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        VkImageMemoryBarrier censusBarriers[2] = {
            imageBarrier(leftCensus, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            imageBarrier(rightCensus, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             2, censusBarriers);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, matchPass.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                matchPass.pipelineLayout, 0, 1,
                                &matchPass.descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 2);

        VkImageMemoryBarrier disparityBarriers[2] = {
            imageBarrier(leftDisparity, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            imageBarrier(rightDisparity, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             2, disparityBarriers);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, filterPass.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                filterPass.pipelineLayout, 0, 1,
                                &filterPass.descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        VkImageMemoryBarrier supportInput = imageBarrier(
            depth, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &supportInput);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          supportPass.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                supportPass.pipelineLayout, 0, 1,
                                &supportPass.descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        VkImageMemoryBarrier downloadBarrier = imageBarrier(
            filteredDepth, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &downloadBarrier);
        copy.bufferOffset = outputOffset;
        vkCmdCopyImageToBuffer(commandBuffer, filteredDepth.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               stagingBuffer, 1, &copy);
        VkImageMemoryBarrier restoreDepth = imageBarrier(
            filteredDepth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        if (diagnostics) {
            VkDeviceSize diagnosticOffset = outputOffset + planeBytes * sizeof(float);
            for (ImageResource* image : diagnosticImages) {
                auto barrier = imageBarrier(*image, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                copy.bufferOffset = diagnosticOffset;
                vkCmdCopyImageToBuffer(commandBuffer, image->image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &copy);
                barrier = imageBarrier(*image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                diagnosticOffset += planeBytes * (image->format == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : 4);
            }
        }
        VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.buffer = stagingBuffer;
        hostBarrier.offset = outputOffset;
        hostBarrier.size = diagnostics ? stagingBytes - outputOffset : planeBytes * sizeof(float);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             1, &hostBarrier, 1, &restoreDepth);

        if (!submitAndWait()) {
            setError("Vulkan stereo submission timed out or failed");
            ready = false;
            return false;
        }
        computeMs = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (!stagingCoherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = stagingMemory;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(device, 1, &range);
        }
        depthMeters.resize(static_cast<size_t>(planeBytes));
        std::memcpy(depthMeters.data(), stagingMapped + outputOffset,
                    static_cast<size_t>(planeBytes) * sizeof(float));

        if (diagnostics) {
            VkDeviceSize offset = outputOffset + planeBytes * sizeof(float);
            const auto extract = [&](auto& out, size_t channels) {
                out.resize(static_cast<size_t>(planeBytes) * channels);
                std::memcpy(out.data(), stagingMapped + offset, out.size() * sizeof(out[0]));
                offset += out.size() * sizeof(out[0]);
            };
            extract(diagnostics->leftDisparity, 1); extract(diagnostics->rightDisparity, 1);
            extract(diagnostics->consistencyDepth, 1); extract(diagnostics->leftMatch, 4);
            extract(diagnostics->rightMatch, 4); extract(diagnostics->filterRejection, 1);
            extract(diagnostics->supportRejection, 1);
        }
        ++frames;
        if (frames == 1 || frames % 30 == 0) {
            size_t valid = 0;
            float minimum = 12.0f;
            float maximum = 0.0f;
            for (float value : depthMeters) {
                if (!std::isfinite(value) || value <= 0.0f) continue;
                ++valid;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            std::cout << "[VulkanStereo] " << computeMs << " ms, coverage="
                      << 100.0f * static_cast<float>(valid) / depthMeters.size()
                      << "%, range=" << (valid ? minimum : 0.0f) << ".."
                      << maximum << " m" << std::endl;
        }
        return true;
    }

    void destroyPass(ComputePass& pass) {
        if (!device) return;
        if (pass.pipeline) vkDestroyPipeline(device, pass.pipeline, nullptr);
        if (pass.pipelineLayout) vkDestroyPipelineLayout(device, pass.pipelineLayout, nullptr);
        if (pass.descriptorLayout) {
            vkDestroyDescriptorSetLayout(device, pass.descriptorLayout, nullptr);
        }
        pass = {};
    }

    void destroyImage(ImageResource& resource) {
        if (!device) return;
        if (resource.view) vkDestroyImageView(device, resource.view, nullptr);
        if (resource.image) vkDestroyImage(device, resource.image, nullptr);
        if (resource.memory) vkFreeMemory(device, resource.memory, nullptr);
        resource = {};
    }

    void release() {
        if (device) vkDeviceWaitIdle(device);
        destroyPass(supportPass);
        destroyPass(filterPass);
        destroyPass(matchPass);
        destroyPass(censusPass);
        if (device && descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        destroyImage(supportRejection); destroyImage(filterRejection);
        destroyImage(rightMatch); destroyImage(leftMatch);
        destroyImage(filteredDepth);
        destroyImage(depth);
        destroyImage(rightDisparity);
        destroyImage(leftDisparity);
        destroyImage(rightCensus);
        destroyImage(leftCensus);
        destroyImage(right);
        destroyImage(left);
        if (device && stagingMapped) vkUnmapMemory(device, stagingMemory);
        if (device && stagingBuffer) vkDestroyBuffer(device, stagingBuffer, nullptr);
        if (device && stagingMemory) vkFreeMemory(device, stagingMemory, nullptr);
        if (device && fence) vkDestroyFence(device, fence, nullptr);
        if (device && commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        ready = false;
    }
};

VulkanStereoPipeline::VulkanStereoPipeline() : m_impl(std::make_unique<Impl>()) {}
VulkanStereoPipeline::~VulkanStereoPipeline() = default;

bool VulkanStereoPipeline::initialize(const std::string& shaderDirectory, bool enableDiagnostics) {
    return m_impl->initialize(shaderDirectory, enableDiagnostics);
}

bool VulkanStereoPipeline::process(const std::vector<uint8_t>& rectifiedLeft,
                                   const std::vector<uint8_t>& rectifiedRight,
                                   std::vector<float>& depthMeters, StereoDiagnostics* diagnostics) {
    return m_impl->process(rectifiedLeft, rectifiedRight, depthMeters, diagnostics);
}

bool VulkanStereoPipeline::isReady() const { return m_impl->ready; }
float VulkanStereoPipeline::lastComputeMs() const { return m_impl->computeMs; }
const std::string& VulkanStereoPipeline::lastError() const { return m_impl->error; }

} // namespace FrameSpatial
