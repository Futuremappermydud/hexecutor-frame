#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h>
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "calibration.hpp"
#include "depth_pipeline.hpp"
#include "depth_speckle_filter.hpp"
#include "cloud_visibility.hpp"
#include "cloud_worker.hpp"
#include "application_render_profile.hpp"
#include "openxr_geometry.hpp"
#include "dma_capture.hpp"
#include "vulkan_stereo_pipeline.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace {

using namespace FrameSpatialCloud;

void signalHandler(int) {
    gRunning.store(false);
}

const char* xrResultName(XrResult result) {
    switch (result) {
        case XR_SUCCESS: return "XR_SUCCESS";
        case XR_TIMEOUT_EXPIRED: return "XR_TIMEOUT_EXPIRED";
        case XR_SESSION_LOSS_PENDING: return "XR_SESSION_LOSS_PENDING";
        case XR_ERROR_RUNTIME_FAILURE: return "XR_ERROR_RUNTIME_FAILURE";
        case XR_ERROR_INSTANCE_LOST: return "XR_ERROR_INSTANCE_LOST";
        case XR_ERROR_SESSION_LOST: return "XR_ERROR_SESSION_LOST";
        case XR_ERROR_INITIALIZATION_FAILED: return "XR_ERROR_INITIALIZATION_FAILED";
        case XR_ERROR_API_VERSION_UNSUPPORTED: return "XR_ERROR_API_VERSION_UNSUPPORTED";
        case XR_ERROR_GRAPHICS_DEVICE_INVALID: return "XR_ERROR_GRAPHICS_DEVICE_INVALID";
        case XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING:
            return "XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING";
        default: return "XR_RESULT";
    }
}

bool xrCheck(XrResult result, const char* operation) {
    if (XR_SUCCEEDED(result)) return true;
    std::cerr << "[OpenXR] " << operation << " failed: " << xrResultName(result)
              << " (" << result << ")" << std::endl;
    return false;
}

std::vector<uint32_t> readSpirv(const std::string& path) {
    auto tryRead = [](const std::string& p) -> std::vector<uint32_t> {
        std::ifstream input(p, std::ios::binary | std::ios::ate);
        if (!input) return {};
        const auto bytes = input.tellg();
        if (bytes <= 0 || (bytes % 4) != 0) return {};
        input.seekg(0);
        std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
        input.read(reinterpret_cast<char*>(code.data()), bytes);
        return input.good() ? code : std::vector<uint32_t>{};
    };

    auto code = tryRead(path);
    if (!code.empty()) return code;

    if (const char* envDir = std::getenv("FRAME_SHADER_DIR")) {
        auto pos = path.find_last_of("/\\");
        std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
        code = tryRead(std::string(envDir) + "/" + filename);
        if (!code.empty()) return code;
    }

    char exePath[1024];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len > 0) {
        exePath[len] = '\0';
        std::string exeStr(exePath);
        auto pos = exeStr.find_last_of("/\\");
        if (pos != std::string::npos) {
            std::string binDir = exeStr.substr(0, pos);
            if (binDir.size() >= 4 && binDir.substr(binDir.size() - 4) == "/bin") {
                std::string root = binDir.substr(0, binDir.size() - 4);
                code = tryRead(root + "/" + path);
                if (!code.empty()) return code;
            }
            code = tryRead(binDir + "/" + path);
            if (!code.empty()) return code;
        }
    }

    return tryRead("/home/steamos/frame-spatial-runtime/" + path);
}

class OpenXrPointRenderer {
public:
    explicit OpenXrPointRenderer(bool demo = false, bool mesh = false, bool wireframe = false)
        : mDemo(demo), mMesh(mesh), mWireframe(wireframe) {
        if (mMesh) {
            // Room for up to 250,000 triangles (750,000 vertices)
            mPointCapacity = 1500000;
        }
    }
    ~OpenXrPointRenderer() { release(); }

    bool initialize();
    int run(FrameSpatial::DMACapture& capture);

private:
    struct DepthTarget {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    struct EyeSwapchain {
        XrSwapchain handle = XR_NULL_HANDLE;
        XrSwapchain depthHandle = XR_NULL_HANDLE;
        std::vector<XrSwapchainImageVulkan2KHR> depthImages;
        std::vector<VkImageView> depthViews;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<XrSwapchainImageVulkan2KHR> images;
        std::vector<VkImageView> views;
        std::vector<DepthTarget> depths;
        std::vector<VkFramebuffer> framebuffers;
    };

    bool createXrInstance();
    bool createVulkan();
    bool createSession();
    bool createRenderResources();
    bool createSwapchains();
    bool createDepthTarget(uint32_t width, uint32_t height, DepthTarget& target);
    bool pollEvents();
    bool renderFrame();
    bool drawEye(uint32_t eye, uint32_t imageIndex, const Mat4& viewProjection,
                 uint32_t vertexCount, uint32_t depthIndex);
    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;
    void release();

    bool mDemo = false;
    bool mMesh = false;
    bool mWireframe = false;
    std::vector<PointVertex> mDemoVertices;
    XrInstance mInstance = XR_NULL_HANDLE;
    XrSystemId mSystemId = XR_NULL_SYSTEM_ID;
    XrSession mSession = XR_NULL_HANDLE;
    XrSpace mStageSpace = XR_NULL_HANDLE;
    XrSpace mViewSpace = XR_NULL_HANDLE;
    XrSessionState mSessionState = XR_SESSION_STATE_UNKNOWN;
    bool mSessionRunning = false;
    XrEnvironmentBlendMode mBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

    PFN_xrGetVulkanGraphicsRequirements2KHR mGetRequirements = nullptr;
    PFN_xrCreateVulkanInstanceKHR mCreateVulkanInstance = nullptr;
    PFN_xrGetVulkanGraphicsDevice2KHR mGetVulkanDevice = nullptr;
    PFN_xrCreateVulkanDeviceKHR mCreateVulkanDevice = nullptr;

    VkInstance mVkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    VkQueue mQueue = VK_NULL_HANDLE;
    uint32_t mQueueFamily = 0;
    VkPhysicalDeviceMemoryProperties mMemoryProperties{};
    VkFormat mColorFormat = VK_FORMAT_UNDEFINED;
    VkFormat mDepthFormat = VK_FORMAT_D32_SFLOAT;
    VkRenderPass mRenderPass = VK_NULL_HANDLE;
    VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;
    VkPipeline mPipeline = VK_NULL_HANDLE;
    VkCommandPool mCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer mCommandBuffer = VK_NULL_HANDLE;
    VkFence mFence = VK_NULL_HANDLE;
    VkBuffer mPointBuffer = VK_NULL_HANDLE;
    VkDeviceMemory mPointMemory = VK_NULL_HANDLE;
    void* mPointMapped = nullptr;
    bool mPointCoherent = false;
    size_t mPointCapacity = 260000;
    std::vector<XrViewConfigurationView> mViewConfig;
    std::vector<XrView> mViews;
    std::vector<EyeSwapchain> mEyes;
    uint64_t mRenderedFrames = 0;
    std::chrono::steady_clock::time_point mLastFrameReport = std::chrono::steady_clock::now();
};

bool OpenXrPointRenderer::createXrInstance() {
    const char* extensions[]{XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
                             XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME,
                             XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME};
    XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strncpy(createInfo.applicationInfo.applicationName,
                 "Frame Spatial Point Cloud", XR_MAX_APPLICATION_NAME_SIZE - 1);
    createInfo.applicationInfo.applicationVersion = 1;
    std::strncpy(createInfo.applicationInfo.engineName, "FrameSpatial",
                 XR_MAX_ENGINE_NAME_SIZE - 1);
    createInfo.applicationInfo.engineVersion = 1;
    // SteamVR's Frame runtime currently advertises OpenXR 1.0 even though the
    // distro headers are newer. Request only the core version used here.
    createInfo.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    createInfo.enabledExtensionCount = mDemo ? 3 : 2;
    createInfo.enabledExtensionNames = extensions;
    if (!xrCheck(xrCreateInstance(&createInfo, &mInstance), "xrCreateInstance")) {
        return false;
    }

    XrInstanceProperties properties{XR_TYPE_INSTANCE_PROPERTIES};
    xrGetInstanceProperties(mInstance, &properties);
    std::cout << "[OpenXR] Runtime: " << properties.runtimeName << " "
              << XR_VERSION_MAJOR(properties.runtimeVersion) << "."
              << XR_VERSION_MINOR(properties.runtimeVersion) << "."
              << XR_VERSION_PATCH(properties.runtimeVersion) << std::endl;

    XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
    systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (!xrCheck(xrGetSystem(mInstance, &systemInfo, &mSystemId), "xrGetSystem")) {
        return false;
    }

    auto load = [&](const char* name, PFN_xrVoidFunction* function) {
        return xrCheck(xrGetInstanceProcAddr(mInstance, name, function), name);
    };
    const bool loaded = load("xrGetVulkanGraphicsRequirements2KHR",
                reinterpret_cast<PFN_xrVoidFunction*>(&mGetRequirements)) &&
           load("xrCreateVulkanInstanceKHR",
                reinterpret_cast<PFN_xrVoidFunction*>(&mCreateVulkanInstance)) &&
           load("xrGetVulkanGraphicsDevice2KHR",
                reinterpret_cast<PFN_xrVoidFunction*>(&mGetVulkanDevice)) &&
           load("xrCreateVulkanDeviceKHR",
                reinterpret_cast<PFN_xrVoidFunction*>(&mCreateVulkanDevice));
    if (!loaded || !load("xrConvertTimespecTimeToTimeKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&gConvertTimespecTime))) {
        return false;
    }
    gLocateSpace = xrLocateSpace;
    gXrInstance = mInstance;
    return true;
}

bool OpenXrPointRenderer::createVulkan() {
    XrGraphicsRequirementsVulkan2KHR requirements{
        XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    if (!xrCheck(mGetRequirements(mInstance, mSystemId, &requirements),
                 "xrGetVulkanGraphicsRequirements2KHR")) return false;

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Frame Spatial Point Cloud";
    application.applicationVersion = 1;
    application.pEngineName = "FrameSpatial";
    application.engineVersion = 1;
    auto xrVersionToVk = [](XrVersion version) {
        return VK_MAKE_API_VERSION(0, XR_VERSION_MAJOR(version),
                                   XR_VERSION_MINOR(version),
                                   XR_VERSION_PATCH(version));
    };
    const uint32_t minimumVulkan = xrVersionToVk(requirements.minApiVersionSupported);
    const uint32_t maximumVulkan = xrVersionToVk(requirements.maxApiVersionSupported);
    application.apiVersion = std::min(VK_API_VERSION_1_1, maximumVulkan);
    if (application.apiVersion < minimumVulkan) {
        application.apiVersion = minimumVulkan;
    }
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &application;
    XrVulkanInstanceCreateInfoKHR xrInstanceInfo{
        XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    xrInstanceInfo.systemId = mSystemId;
    xrInstanceInfo.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xrInstanceInfo.vulkanCreateInfo = &instanceInfo;
    VkResult vkResult = VK_SUCCESS;
    if (!xrCheck(mCreateVulkanInstance(mInstance, &xrInstanceInfo, &mVkInstance,
                                       &vkResult), "xrCreateVulkanInstanceKHR") ||
        vkResult != VK_SUCCESS) {
        std::cerr << "[Vulkan] Instance creation result " << vkResult << std::endl;
        return false;
    }

    XrVulkanGraphicsDeviceGetInfoKHR deviceInfo{
        XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    deviceInfo.systemId = mSystemId;
    deviceInfo.vulkanInstance = mVkInstance;
    if (!xrCheck(mGetVulkanDevice(mInstance, &deviceInfo, &mPhysicalDevice),
                 "xrGetVulkanGraphicsDevice2KHR")) return false;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &familyCount,
                                             families.data());
    bool foundQueue = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            mQueueFamily = i;
            foundQueue = true;
            break;
        }
    }
    if (!foundQueue) return false;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = mQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures available{};
    vkGetPhysicalDeviceFeatures(mPhysicalDevice, &available);
    VkPhysicalDeviceFeatures enabled{};
    enabled.largePoints = available.largePoints;
    VkDeviceCreateInfo vkDeviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    vkDeviceInfo.queueCreateInfoCount = 1;
    vkDeviceInfo.pQueueCreateInfos = &queueInfo;
    vkDeviceInfo.pEnabledFeatures = &enabled;
    XrVulkanDeviceCreateInfoKHR xrDeviceInfo{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    xrDeviceInfo.systemId = mSystemId;
    xrDeviceInfo.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xrDeviceInfo.vulkanPhysicalDevice = mPhysicalDevice;
    xrDeviceInfo.vulkanCreateInfo = &vkDeviceInfo;
    if (!xrCheck(mCreateVulkanDevice(mInstance, &xrDeviceInfo, &mDevice, &vkResult),
                 "xrCreateVulkanDeviceKHR") || vkResult != VK_SUCCESS) {
        std::cerr << "[Vulkan] Device creation result " << vkResult << std::endl;
        return false;
    }
    vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &mMemoryProperties);
    std::cout << "[OpenXR] Vulkan API " << VK_VERSION_MAJOR(application.apiVersion)
              << "." << VK_VERSION_MINOR(application.apiVersion)
              << ", graphics queue " << mQueueFamily << std::endl;
    return true;
}

bool OpenXrPointRenderer::createSession() {
    XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    binding.instance = mVkInstance;
    binding.physicalDevice = mPhysicalDevice;
    binding.device = mDevice;
    binding.queueFamilyIndex = mQueueFamily;
    binding.queueIndex = 0;
    XrSessionCreateInfo sessionInfo{XR_TYPE_SESSION_CREATE_INFO};
    sessionInfo.next = &binding;
    sessionInfo.systemId = mSystemId;
    if (!xrCheck(xrCreateSession(mInstance, &sessionInfo, &mSession),
                 "xrCreateSession")) return false;

    uint32_t spaceCount = 0;
    xrEnumerateReferenceSpaces(mSession, 0, &spaceCount, nullptr);
    std::vector<XrReferenceSpaceType> spaces(spaceCount);
    xrEnumerateReferenceSpaces(mSession, spaceCount, &spaceCount, spaces.data());
    XrReferenceSpaceType stageType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    if (std::find(spaces.begin(), spaces.end(), XR_REFERENCE_SPACE_TYPE_STAGE) !=
        spaces.end()) {
        stageType = XR_REFERENCE_SPACE_TYPE_STAGE;
    }
#ifdef XR_EXT_local_floor
    else if (std::find(spaces.begin(), spaces.end(),
                       XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR_EXT) != spaces.end()) {
        stageType = XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR_EXT;
    }
#endif
    const XrPosef identity{{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
    XrReferenceSpaceCreateInfo spaceInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    spaceInfo.referenceSpaceType = stageType;
    spaceInfo.poseInReferenceSpace = identity;
    if (!xrCheck(xrCreateReferenceSpace(mSession, &spaceInfo, &mStageSpace),
                 "xrCreateReferenceSpace(stage)")) return false;
    spaceInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    if (!xrCheck(xrCreateReferenceSpace(mSession, &spaceInfo, &mViewSpace),
                 "xrCreateReferenceSpace(view)")) return false;
    gStageSpace = mStageSpace;
    gViewSpace = mViewSpace;

    uint32_t blendCount = 0;
    xrEnumerateEnvironmentBlendModes(mInstance, mSystemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &blendCount, nullptr);
    std::vector<XrEnvironmentBlendMode> modes(blendCount);
    xrEnumerateEnvironmentBlendModes(mInstance, mSystemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, blendCount, &blendCount,
        modes.data());
    if (std::find(modes.begin(), modes.end(), XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) !=
        modes.end()) {
        mBlendMode = XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
    } else if (!modes.empty()) {
        mBlendMode = modes.front();
    }
    std::cout << "[OpenXR] Reference space=" << stageType
              << ", blend mode=" << mBlendMode << std::endl;
    return true;
}

uint32_t OpenXrPointRenderer::findMemoryType(
    uint32_t bits, VkMemoryPropertyFlags flags) const {
    for (uint32_t i = 0; i < mMemoryProperties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (mMemoryProperties.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

bool OpenXrPointRenderer::createRenderResources() {
    VkAttachmentDescription attachments[2]{};
    attachments[0].format = mColorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1].format = mDepthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = mDemo ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthReference{1,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;
    subpass.pDepthStencilAttachment = &depthReference;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = dependency.srcStageMask;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    if (vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderPass) !=
        VK_SUCCESS) return false;

    const auto vertexCode = readSpirv(mMesh ? "shaders/mesh.vert.spv" : "shaders/point_cloud.vert.spv");
    const auto fragmentCode = readSpirv(mDemo ? "shaders/occlusion_demo.frag.spv" : (mMesh ? "shaders/mesh.frag.spv" : "shaders/point_cloud.frag.spv"));
    if (vertexCode.empty() || fragmentCode.empty()) {
        std::cerr << "[Vulkan] " << (mMesh ? "Mesh" : "Point-cloud") << " shaders are missing" << std::endl;
        return false;
    }
    auto createShader = [&](const std::vector<uint32_t>& code, VkShaderModule& module) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = code.size() * sizeof(uint32_t);
        info.pCode = code.data();
        return vkCreateShaderModule(mDevice, &info, nullptr, &module) == VK_SUCCESS;
    };
    VkShaderModule vertexShader = VK_NULL_HANDLE;
    VkShaderModule fragmentShader = VK_NULL_HANDLE;
    if (!createShader(vertexCode, vertexShader) ||
        !createShader(fragmentCode, fragmentShader)) return false;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertexShader;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragmentShader;
    stages[1].pName = "main";
    VkVertexInputBindingDescription binding{0, sizeof(PointVertex),
                                             VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[2]{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                     static_cast<uint32_t>(offsetof(PointVertex, x))};
    attributes[1] = {1, 0, VK_FORMAT_R8G8B8A8_UNORM,
                     static_cast<uint32_t>(offsetof(PointVertex, rgba))};
    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = (mDemo || mMesh) ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = mWireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                     VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT |
                                     VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;
    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT,
                                   VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset = 0;
    push.size = sizeof(Mat4);
    VkPipelineLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(mDevice, &layoutInfo, nullptr, &mPipelineLayout) !=
        VK_SUCCESS) return false;
    VkGraphicsPipelineCreateInfo pipelineInfo{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewport;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = mPipelineLayout;
    pipelineInfo.renderPass = mRenderPass;
    if (vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &mPipeline) != VK_SUCCESS) return false;
    vkDestroyShaderModule(mDevice, fragmentShader, nullptr);
    vkDestroyShaderModule(mDevice, vertexShader, nullptr);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = mQueueFamily;
    if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool) != VK_SUCCESS)
        return false;
    VkCommandBufferAllocateInfo commandInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandInfo.commandPool = mCommandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(mDevice, &commandInfo, &mCommandBuffer) != VK_SUCCESS)
        return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(mDevice, &fenceInfo, nullptr, &mFence) != VK_SUCCESS) return false;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = mPointCapacity * sizeof(PointVertex);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(mDevice, &bufferInfo, nullptr, &mPointBuffer) != VK_SUCCESS)
        return false;
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(mDevice, mPointBuffer, &memoryRequirements);
    uint32_t memoryType = findMemoryType(memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memoryType == std::numeric_limits<uint32_t>::max()) {
        memoryType = findMemoryType(memoryRequirements.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    } else {
        mPointCoherent = true;
    }
    if (memoryType == std::numeric_limits<uint32_t>::max()) return false;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = memoryRequirements.size;
    allocation.memoryTypeIndex = memoryType;
    if (vkAllocateMemory(mDevice, &allocation, nullptr, &mPointMemory) != VK_SUCCESS ||
        vkBindBufferMemory(mDevice, mPointBuffer, mPointMemory, 0) != VK_SUCCESS ||
        vkMapMemory(mDevice, mPointMemory, 0, VK_WHOLE_SIZE, 0, &mPointMapped) !=
            VK_SUCCESS) return false;
    return true;
}

bool OpenXrPointRenderer::createDepthTarget(uint32_t width, uint32_t height,
                                            DepthTarget& target) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = mDepthFormat;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (vkCreateImage(mDevice, &imageInfo, nullptr, &target.image) != VK_SUCCESS)
        return false;
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(mDevice, target.image, &requirements);
    const uint32_t memoryType = findMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryType == std::numeric_limits<uint32_t>::max()) return false;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    if (vkAllocateMemory(mDevice, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
        vkBindImageMemory(mDevice, target.image, target.memory, 0) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = target.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = mDepthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    return vkCreateImageView(mDevice, &viewInfo, nullptr, &target.view) == VK_SUCCESS;
}

bool OpenXrPointRenderer::createSwapchains() {
    uint32_t viewCount = 0;
    if (!xrCheck(xrEnumerateViewConfigurationViews(
            mInstance, mSystemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
            0, &viewCount, nullptr), "xrEnumerateViewConfigurationViews")) return false;
    mViewConfig.resize(viewCount);
    for (auto& view : mViewConfig) view.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
    if (!xrCheck(xrEnumerateViewConfigurationViews(
            mInstance, mSystemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
            viewCount, &viewCount, mViewConfig.data()),
            "xrEnumerateViewConfigurationViews")) return false;
    if (viewCount != 2) {
        std::cerr << "[OpenXR] Expected stereo views, got " << viewCount << std::endl;
        return false;
    }
    mViews.resize(viewCount);
    for (auto& view : mViews) view.type = XR_TYPE_VIEW;

    uint32_t formatCount = 0;
    xrEnumerateSwapchainFormats(mSession, 0, &formatCount, nullptr);
    std::vector<int64_t> formats(formatCount);
    xrEnumerateSwapchainFormats(mSession, formatCount, &formatCount, formats.data());
    const VkFormat preferred[]{VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    for (VkFormat candidate : preferred) {
        if (std::find(formats.begin(), formats.end(), static_cast<int64_t>(candidate)) !=
            formats.end()) {
            mColorFormat = candidate;
            break;
        }
    }
    if (mColorFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[OpenXR] No supported RGBA swapchain format" << std::endl;
        return false;
    }
    if (mDemo && std::find(formats.begin(), formats.end(), int64_t(mDepthFormat)) == formats.end()) {
        std::cerr << "[Demo] Runtime does not support D32 depth swapchains" << std::endl;
        return false;
    }
    if (!createRenderResources()) return false;

    mEyes.resize(viewCount);
    for (uint32_t eye = 0; eye < viewCount; ++eye) {
        EyeSwapchain& output = mEyes[eye];
        output.width = mViewConfig[eye].recommendedImageRectWidth;
        output.height = mViewConfig[eye].recommendedImageRectHeight;
        XrSwapchainCreateInfo swapchainInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        swapchainInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        swapchainInfo.format = mColorFormat;
        swapchainInfo.sampleCount = 1;
        swapchainInfo.width = output.width;
        swapchainInfo.height = output.height;
        swapchainInfo.faceCount = 1;
        swapchainInfo.arraySize = 1;
        swapchainInfo.mipCount = 1;
        if (!xrCheck(xrCreateSwapchain(mSession, &swapchainInfo, &output.handle),
                     "xrCreateSwapchain")) return false;
        uint32_t imageCount = 0;
        xrEnumerateSwapchainImages(output.handle, 0, &imageCount, nullptr);
        output.images.resize(imageCount);
        for (auto& image : output.images) image.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
        if (!xrCheck(xrEnumerateSwapchainImages(
                output.handle, imageCount, &imageCount,
                reinterpret_cast<XrSwapchainImageBaseHeader*>(output.images.data())),
                "xrEnumerateSwapchainImages")) return false;
        output.views.resize(imageCount);
        output.depths.resize(imageCount);
        uint32_t depthCount = 1;
        if (mDemo) {
            auto depthInfo = swapchainInfo;
            depthInfo.format = mDepthFormat;
            depthInfo.usageFlags = XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            if (!xrCheck(xrCreateSwapchain(mSession, &depthInfo, &output.depthHandle), "xrCreateSwapchain(depth)")) return false;
            if (!xrCheck(xrEnumerateSwapchainImages(output.depthHandle, 0, &depthCount, nullptr), "xrEnumerateSwapchainImages(depth)")) return false;
            output.depthImages.resize(depthCount);
            output.depthViews.resize(depthCount);
            for (auto& image : output.depthImages) image.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
            if (!xrCheck(xrEnumerateSwapchainImages(output.depthHandle, depthCount, &depthCount,
                reinterpret_cast<XrSwapchainImageBaseHeader*>(output.depthImages.data())), "xrEnumerateSwapchainImages(depth)")) return false;
            for (uint32_t d = 0; d < depthCount; ++d) {
                VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                vi.image = output.depthImages[d].image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = mDepthFormat;
                vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                if (vkCreateImageView(mDevice, &vi, nullptr, &output.depthViews[d]) != VK_SUCCESS) return false;
            }
        }
        output.framebuffers.resize(imageCount * depthCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = output.images[i].image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = mColorFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(mDevice, &viewInfo, nullptr, &output.views[i]) !=
                VK_SUCCESS) return false;
            if (!mDemo && !createDepthTarget(output.width, output.height, output.depths[i])) return false;
            for (uint32_t d = 0; d < depthCount; ++d) {
            VkImageView attachments[]{output.views[i], mDemo ? output.depthViews[d] : output.depths[i].view};
            VkFramebufferCreateInfo framebufferInfo{
                VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            framebufferInfo.renderPass = mRenderPass;
            framebufferInfo.attachmentCount = 2;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = output.width;
            framebufferInfo.height = output.height;
            framebufferInfo.layers = 1;
            if (vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr,
                                    &output.framebuffers[i * depthCount + d]) != VK_SUCCESS) return false;
            }
        }
        std::cout << "[OpenXR] Eye " << eye << ": " << output.width << "x"
                  << output.height << ", " << imageCount << " images" << std::endl;
    }
    return true;
}

bool OpenXrPointRenderer::initialize() {
    return createXrInstance() && createVulkan() && createSession() && createSwapchains();
}

bool OpenXrPointRenderer::pollEvents() {
    XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(mInstance, &event) == XR_SUCCESS) {
        if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const auto* changed =
                reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
            mSessionState = changed->state;
            std::cout << "[OpenXR] Session state " << mSessionState << std::endl;
            if (mSessionState == XR_SESSION_STATE_READY && !mSessionRunning) {
                XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
                begin.primaryViewConfigurationType =
                    XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                if (!xrCheck(xrBeginSession(mSession, &begin), "xrBeginSession"))
                    return false;
                mSessionRunning = true;
            } else if (mSessionState == XR_SESSION_STATE_STOPPING && mSessionRunning) {
                xrEndSession(mSession);
                mSessionRunning = false;
            } else if (mSessionState == XR_SESSION_STATE_EXITING ||
                       mSessionState == XR_SESSION_STATE_LOSS_PENDING) {
                gRunning.store(false);
            }
        } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
            gRunning.store(false);
        }
        event = {XR_TYPE_EVENT_DATA_BUFFER};
    }
    return true;
}

bool OpenXrPointRenderer::drawEye(uint32_t eye, uint32_t imageIndex,
                                  const Mat4& viewProjection,
                                  uint32_t vertexCount, uint32_t depthIndex) {
    if (vkWaitForFences(mDevice, 1, &mFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
        vkResetFences(mDevice, 1, &mFence) != VK_SUCCESS ||
        vkResetCommandBuffer(mCommandBuffer, 0) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(mCommandBuffer, &begin) != VK_SUCCESS) return false;
    VkClearValue clears[2]{};
    clears[0].color = {{0.004f, 0.006f, 0.010f,
                        mBlendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND ? 0.0f : 1.0f}};
    clears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render.renderPass = mRenderPass;
    render.framebuffer = mEyes[eye].framebuffers[imageIndex * (mDemo ? mEyes[eye].depthViews.size() : 1) + depthIndex];
    render.renderArea.extent = {mEyes[eye].width, mEyes[eye].height};
    render.clearValueCount = 2;
    render.pClearValues = clears;
    vkCmdBeginRenderPass(mCommandBuffer, &render, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(mEyes[eye].width),
                        static_cast<float>(mEyes[eye].height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {mEyes[eye].width, mEyes[eye].height}};
    vkCmdSetViewport(mCommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(mCommandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(mCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(mCommandBuffer, 0, 1, &mPointBuffer, &offset);
    vkCmdPushConstants(mCommandBuffer, mPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(Mat4), &viewProjection);
    if (vertexCount > 0) vkCmdDraw(mCommandBuffer, vertexCount, 1, 0, 0);
    vkCmdEndRenderPass(mCommandBuffer);
    if (vkEndCommandBuffer(mCommandBuffer) != VK_SUCCESS) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &mCommandBuffer;
    return vkQueueSubmit(mQueue, 1, &submit, mFence) == VK_SUCCESS;
}

bool OpenXrPointRenderer::renderFrame() {
    static ApplicationRenderProfile profile;
    ApplicationRenderProfile::Sample timing{};
    const auto frameStart=profile.now();
    auto stageStart=frameStart;
    bool rendered=false;
    XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState state{XR_TYPE_FRAME_STATE};
    if (!xrCheck(xrWaitFrame(mSession, &waitInfo, &state), "xrWaitFrame")) return false;
    profile.mark(timing,ApplicationRenderProfile::WaitFrame,stageStart);
    XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    if (!xrCheck(xrBeginFrame(mSession, &beginInfo), "xrBeginFrame")) return false;

    stageStart=profile.now();
    XrViewState viewState{XR_TYPE_VIEW_STATE};
    XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
    locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    locateInfo.displayTime = state.predictedDisplayTime;
    locateInfo.space = mStageSpace;
    uint32_t viewCount = 0;
    const XrResult locateResult = xrLocateViews(mSession, &locateInfo, &viewState,
                                                mViews.size(), &viewCount,
                                                mViews.data());

    XrSpaceLocation headLocation{XR_TYPE_SPACE_LOCATION};
    xrLocateSpace(mViewSpace, mStageSpace, state.predictedDisplayTime, &headLocation);
    const XrSpaceLocationFlags required = XR_SPACE_LOCATION_POSITION_VALID_BIT |
                                          XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    if ((headLocation.locationFlags & required) == required) {
        std::lock_guard<std::mutex> lock(gPoseMutex);
        gLatestHeadPose = {headLocation.pose, state.predictedDisplayTime, true};
    }

    std::array<XrCompositionLayerProjectionView, 2> projectionViews{};
    std::array<XrCompositionLayerDepthInfoKHR, 2> depthInfos{};
    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    std::vector<const XrCompositionLayerBaseHeader*> layers;
    const bool viewsValid = XR_SUCCEEDED(locateResult) && viewCount == 2 &&
        (viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) &&
        (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT);
    profile.mark(timing,ApplicationRenderProfile::Locate,stageStart);
    if (state.shouldRender && viewsValid) {
        stageStart=profile.now();
        std::vector<PointVertex> points;
        if (!mDemo) {
            std::lock_guard<std::mutex> lock(gCloudMutex);
            points = gCloud;
        }
        if (mDemo) {
            if (mDemoVertices.empty()) {
                // Three stage-locked panels at different metric distances.
                const float distances[]{1.0f, 2.0f, 3.0f};
                const uint32_t colors[]{0xff4040ffu, 0xff40ff40u, 0xffff8040u};
                const int indices[]{0,1,2,0,2,3};
                for (int panel = 0; panel < 3; ++panel) {
                    const float cx = (panel - 1) * 0.65f;
                    const float corners[4][2]{{cx-0.24f,-0.4f},{cx+0.24f,-0.4f},{cx+0.24f,0.4f},{cx-0.24f,0.4f}};
                    // Inverse current left-eye view maps the demo into stage.
                    const Mat4 toEye = viewFromPose(mViews[0].pose);
                    for (int index : indices) {
                        const float v[3]{corners[index][0], corners[index][1], -distances[panel]};
                        const auto& pose=mViews[0].pose;
                        float world[3]{pose.position.x,pose.position.y,pose.position.z};
                        for (int row=0;row<3;++row) for (int col=0;col<3;++col) world[row]+=toEye.m[row*4+col]*v[col];
                        mDemoVertices.push_back({world[0],world[1],world[2],colors[panel]});
                    }
                }
                std::cout << "[Demo] Stage-locked panels at 1, 2, and 3 metres; submitting application depth" << std::endl;
            }
            points=mDemoVertices;
        }
        const uint32_t pointCount = static_cast<uint32_t>(
            std::min(points.size(), mPointCapacity));
        if (pointCount > 0) {
            const VkDeviceSize bytes = pointCount * sizeof(PointVertex);
            std::memcpy(mPointMapped, points.data(), bytes);
            if (!mPointCoherent) {
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = mPointMemory;
                range.offset = 0;
                range.size = VK_WHOLE_SIZE;
                vkFlushMappedMemoryRanges(mDevice, 1, &range);
            }
        }

        profile.mark(timing,ApplicationRenderProfile::Upload,stageStart);
        stageStart=profile.now();
        std::array<uint32_t, 2> indices{}, depthIndices{};
        bool acquired = true;
        for (uint32_t eye = 0; eye < 2; ++eye) {
            XrSwapchainImageAcquireInfo acquireInfo{
                XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            acquired &= xrCheck(xrAcquireSwapchainImage(
                mEyes[eye].handle, &acquireInfo, &indices[eye]),
                "xrAcquireSwapchainImage");
            XrSwapchainImageWaitInfo imageWait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            imageWait.timeout = XR_INFINITE_DURATION;
            acquired &= xrCheck(xrWaitSwapchainImage(mEyes[eye].handle, &imageWait),
                                "xrWaitSwapchainImage");
            if (mDemo) {
                acquired &= xrCheck(xrAcquireSwapchainImage(mEyes[eye].depthHandle, &acquireInfo, &depthIndices[eye]), "xrAcquireSwapchainImage(depth)");
                acquired &= xrCheck(xrWaitSwapchainImage(mEyes[eye].depthHandle, &imageWait), "xrWaitSwapchainImage(depth)");
            }
        }
        profile.mark(timing,ApplicationRenderProfile::Acquire,stageStart);
        if (acquired) {
            for (uint32_t eye = 0; eye < 2; ++eye) {
                stageStart=profile.now();
                const Mat4 view = viewFromPose(mViews[eye].pose);
                const Mat4 projection = projectionFromFov(mViews[eye].fov, 0.08f, 12.0f);
                if (!drawEye(eye, indices[eye], multiply(projection, view), pointCount, depthIndices[eye]))
                    return false;
                if (vkWaitForFences(mDevice, 1, &mFence, VK_TRUE, UINT64_MAX) !=
                    VK_SUCCESS) return false;
                profile.mark(timing,eye?ApplicationRenderProfile::Eye1:ApplicationRenderProfile::Eye0,stageStart);
                XrSwapchainImageReleaseInfo releaseInfo{
                    XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                if (!xrCheck(xrReleaseSwapchainImage(mEyes[eye].handle, &releaseInfo),
                             "xrReleaseSwapchainImage")) return false;
                projectionViews[eye].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
                projectionViews[eye].pose = mViews[eye].pose;
                projectionViews[eye].fov = mViews[eye].fov;
                projectionViews[eye].subImage.swapchain = mEyes[eye].handle;
                projectionViews[eye].subImage.imageRect.offset = {0, 0};
                projectionViews[eye].subImage.imageRect.extent = {
                    static_cast<int32_t>(mEyes[eye].width),
                    static_cast<int32_t>(mEyes[eye].height)};
                projectionViews[eye].subImage.imageArrayIndex = 0;
                if (mDemo) {
                    if (!xrCheck(xrReleaseSwapchainImage(mEyes[eye].depthHandle, &releaseInfo), "xrReleaseSwapchainImage(depth)")) return false;
                    auto& d=depthInfos[eye]; d.type=XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR;
                    d.subImage=projectionViews[eye].subImage; d.subImage.swapchain=mEyes[eye].depthHandle;
                    d.minDepth=0; d.maxDepth=1; d.nearZ=0.08f; d.farZ=12;
                    projectionViews[eye].next=&d;
                }
            }
            layer.space = mStageSpace;
            layer.viewCount = 2;
            layer.views = projectionViews.data();
            if (mBlendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) {
                layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            }
            layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));
            rendered=true;
            ++mRenderedFrames;
            if ((mRenderedFrames % 300u) == 0u) {
                const auto now = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(now - mLastFrameReport).count();
                std::cout << "[OpenXR] rendered=" << mRenderedFrames
                          << ", FPS=" << 300.0 / seconds
                          << (mMesh ? ", triangles=" : ", points=") << (mMesh ? pointCount / 3 : pointCount)
                          << ", depth frames=" << gDepthFrames.load() << std::endl;
                mLastFrameReport = now;
            }
        }
    }

    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime = state.predictedDisplayTime;
    endInfo.environmentBlendMode = mBlendMode;
    endInfo.layerCount = static_cast<uint32_t>(layers.size());
    endInfo.layers = layers.empty() ? nullptr : layers.data();
    stageStart=profile.now();
    const bool success=xrCheck(xrEndFrame(mSession, &endInfo), "xrEndFrame");
    profile.mark(timing,ApplicationRenderProfile::EndFrame,stageStart);
    if(success) profile.finish(timing,frameStart,rendered);
    return success;
}

int OpenXrPointRenderer::run(FrameSpatial::DMACapture& capture) {
    std::thread worker;
    if (!mDemo) worker = std::thread(depthWorker, &capture);
    while (gRunning.load()) {
        if (!pollEvents()) {
            gRunning.store(false);
            break;
        }
        if (mSessionRunning) {
            if (!renderFrame()) {
                gRunning.store(false);
                break;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (worker.joinable()) worker.join();
    return 0;
}

void OpenXrPointRenderer::release() {
    gViewSpace = XR_NULL_HANDLE;
    gStageSpace = XR_NULL_HANDLE;
    gXrInstance = XR_NULL_HANDLE;
    if (mDevice != VK_NULL_HANDLE) vkDeviceWaitIdle(mDevice);
    for (auto& eye : mEyes) {
        for (auto framebuffer : eye.framebuffers) {
            if (framebuffer) vkDestroyFramebuffer(mDevice, framebuffer, nullptr);
        }
        for (auto& depth : eye.depths) {
            if (depth.view) vkDestroyImageView(mDevice, depth.view, nullptr);
            if (depth.image) vkDestroyImage(mDevice, depth.image, nullptr);
            if (depth.memory) vkFreeMemory(mDevice, depth.memory, nullptr);
        }
        for (auto view : eye.views) {
            if (view) vkDestroyImageView(mDevice, view, nullptr);
        }
        for (auto view : eye.depthViews) if (view) vkDestroyImageView(mDevice, view, nullptr);
        if (eye.depthHandle != XR_NULL_HANDLE) xrDestroySwapchain(eye.depthHandle);
        if (eye.handle != XR_NULL_HANDLE) xrDestroySwapchain(eye.handle);
    }
    if (mPointMapped) vkUnmapMemory(mDevice, mPointMemory);
    if (mPointBuffer) vkDestroyBuffer(mDevice, mPointBuffer, nullptr);
    if (mPointMemory) vkFreeMemory(mDevice, mPointMemory, nullptr);
    if (mFence) vkDestroyFence(mDevice, mFence, nullptr);
    if (mCommandPool) vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
    if (mPipeline) vkDestroyPipeline(mDevice, mPipeline, nullptr);
    if (mPipelineLayout) vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
    if (mRenderPass) vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
    if (mViewSpace != XR_NULL_HANDLE) xrDestroySpace(mViewSpace);
    if (mStageSpace != XR_NULL_HANDLE) xrDestroySpace(mStageSpace);
    if (mSession != XR_NULL_HANDLE) xrDestroySession(mSession);
    if (mDevice) vkDestroyDevice(mDevice, nullptr);
    if (mVkInstance) vkDestroyInstance(mVkInstance, nullptr);
    if (mInstance != XR_NULL_HANDLE) xrDestroyInstance(mInstance);
}

} // namespace

int main(int argc, char** argv) {
    bool demo = false;
    bool modelPrepared = false;
    bool modelQnn = false;
    bool model = false;
    bool mesh = false;
    bool wireframe = false;
    bool tsdfMesh = false;

    // Check executable name: if invoked as frame_openxr_mesh_viewer or similar, default to mesh mode
    if (argv[0] && std::strstr(argv[0], "mesh") != nullptr) {
        mesh = true;
    }

    // Check environment variable
    if (const char* envMode = std::getenv("FRAME_VIEWER_MODE")) {
        if (std::strcmp(envMode, "mesh") == 0) {
            mesh = true;
        } else if (std::strcmp(envMode, "wireframe") == 0) {
            mesh = true;
            wireframe = true;
        } else if (std::strcmp(envMode, "points") == 0 || std::strcmp(envMode, "point") == 0) {
            mesh = false;
            wireframe = false;
        }
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--occlusion-demo") == 0) {
            demo = true;
        } else if (std::strcmp(argv[i], "--tsdf") == 0) {
            mesh = true;
            tsdfMesh = true;
        } else if (std::strcmp(argv[i], "--model-temporal") == 0) {
            setenv("FRAME_DEPTH_TEMPORAL", "1", 1);
        } else if (std::strcmp(argv[i], "--depth-model-qnn") == 0) {
            modelPrepared = true;
            modelQnn = true;
            model = true;
        } else if (std::strcmp(argv[i], "--depth-model-prepared") == 0) {
            modelPrepared = true;
            model = true;
        } else if (std::strcmp(argv[i], "--depth-model") == 0) {
            model = true;
        } else if (std::strcmp(argv[i], "--mesh") == 0 || std::strcmp(argv[i], "-m") == 0) {
            mesh = true;
        } else if (std::strcmp(argv[i], "--wireframe") == 0 || std::strcmp(argv[i], "-w") == 0) {
            mesh = true;
            wireframe = true;
        } else if (std::strcmp(argv[i], "--points") == 0 || std::strcmp(argv[i], "-p") == 0) {
            mesh = false;
            wireframe = false;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --mesh, -m               Render as 3D continuous surface mesh (Power-mean filter + tear gating)\n"
                      << "  --wireframe, -w          Render mesh in wireframe mode\n"
                      << "  --points, -p             Render as point cloud\n"
                      << "  --depth-model            Use Model depth neural model\n"
                      << "  --depth-model-prepared   Use prepared Model depth executor\n"
                      << "  --depth-model-qnn        Use QNN run-list Model depth executor (~23 ms)\n"
                      << "  --tsdf                   Fused room mesh from a Vogon-style sparse TSDF\n"
                      << "  --model-temporal         Feed Model-style warped history (hidden/t) to the model\n"
                      << "  --occlusion-demo         Render test occlusion panels\n"
                      << "  --help, -h               Show this help\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n"
                      << "Run with --help for options\n";
            return 2;
        }
    }

    if (model) setenv("FRAME_CLOUD_STEREO", "model", 1);
    if (modelPrepared) setenv("FRAME_DEPTH_BACKEND", modelQnn ? "prepared-qnn" : "prepared", 1);

    // Set cloud worker mode: 0 = PointCloud, 1 = LiveMesh, 3 = Wireframe
    FrameSpatialCloud::gMeshViewerMode.store(tsdfMesh ? 2 : mesh ? (wireframe ? 3 : 1) : 0);

    // If working directory doesn't have shaders, fallback to install directory
    if (access("shaders/stereo_census.spv", R_OK) != 0 &&
        access("/home/steamos/frame-spatial-runtime/shaders/stereo_census.spv", R_OK) == 0) {
        FrameSpatialCloud::gShaderDirectory = "/home/steamos/frame-spatial-runtime/shaders";
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "========================================================\n";
    if (mesh) {
        std::cout << "          FRAME OPENXR 3D MESH VIEWER                   \n"
                  << "========================================================\n"
                  << " Pipeline: Meta-inspired power-mean blur (p=4) +        \n"
                  << "           discontinuity-gated triangulation            \n"
                  << " Display:  " << (wireframe ? "Wireframe" : "Flat-shaded lit surface") << "\n";
    } else {
        std::cout << "       FRAME OPENXR WORLD POINT CLOUD                  \n"
                  << "========================================================\n"
                  << " Display: Point cloud voxels\n";
    }
    std::cout << " Stereo:   " << (model ? (modelQnn ? "Model (QNN run lists)" : modelPrepared ? "Model (Prepared)" : "Model") : "GPU Stereo") << "\n"
              << "========================================================\n";

    FrameSpatial::DMACapture capture;
    const pid_t compositor = FrameSpatial::DMACapture::findCompositorPid();
    if (!demo && (compositor == 0 || !capture.initialize(compositor, -1, -1))) {
        std::cerr << "[Camera] Could not attach to vrcompositor camera DMA-BUFs. "
                  << "Check kernel.yama.ptrace_scope=0." << std::endl;
        return 1;
    }

    OpenXrPointRenderer renderer(demo, mesh, wireframe);
    if (!renderer.initialize()) return 2;
    return renderer.run(capture);
}
