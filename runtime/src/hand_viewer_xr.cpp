#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h>
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include "hand_viewer_xr.hpp"
#include "openxr_geometry.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using FrameSpatialCloud::Mat4;
void check(XrResult result, const char* operation) {
    if (XR_FAILED(result)) throw std::runtime_error(std::string(operation) + ": OpenXR " + std::to_string(result));
}
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": Vulkan " + std::to_string(result));
}
template<class T> T proc(XrInstance instance, const char* name) {
    PFN_xrVoidFunction function = nullptr;
    check(xrGetInstanceProcAddr(instance,name,&function),name);
    return reinterpret_cast<T>(function);
}
std::vector<uint32_t> spirv(const std::string& name) {
    for (const std::string& path : {"shaders/"+name, "../shaders/"+name, "runtime/shaders/"+name}) {
        std::ifstream file(path, std::ios::binary|std::ios::ate);
        if (!file) continue;
        const auto size = size_t(file.tellg());
        if (!size || size%4) break;
        std::vector<uint32_t> code(size/4);
        file.seekg(0); file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));
        if (file) return code;
    }
    throw std::runtime_error("missing shader "+name+" (run make hand-viewer from runtime/)");
}

struct Vertex { float position[3]; float normal[3]; uint32_t rgba; };
struct Push { float view_projection[16]; float eye[4]; };
using Vec3 = std::array<float,3>;
Vec3 operator+(Vec3 a, Vec3 b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Vec3 operator*(Vec3 a, float s) { return {a[0]*s,a[1]*s,a[2]*s}; }
Vec3 cross(Vec3 a, Vec3 b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
float length(Vec3 a) { return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
Vec3 rotate(const XrQuaternionf& q, Vec3 v) {
    const Vec3 u{q.x,q.y,q.z}; const Vec3 t=cross(u,v)*2.f;
    return v+t*q.w+cross(u,t);
}
uint32_t rgba(int r, int g, int b) { return uint32_t(r)|(uint32_t(g)<<8)|(uint32_t(b)<<16)|0xff000000u; }

constexpr float kPi=3.14159265358979323846f;
constexpr int kSphereStacks=6, kSphereSlices=10, kCylinderSides=8, kMaxHands=4;
constexpr int kBones[21][2]={{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},
    {11,12},{9,13},{13,14},{14,15},{15,16},{13,17},{0,17},{17,18},{18,19},{19,20}};
constexpr int kMercuryBones[25][2]={{1,0},{1,2},{2,3},{3,4},{4,5},
    {1,6},{6,7},{7,8},{8,9},{9,10},{1,11},{11,12},{12,13},{13,14},{14,15},
    {1,16},{16,17},{17,18},{18,19},{19,20},{1,21},{21,22},{22,23},{23,24},{24,25}};
// Unit sphere as a non-indexed triangle list (position == normal).
std::vector<Vec3> unitSphere() {
    std::vector<Vec3> out;
    auto point=[](int stack,int slice) {
        const float theta=kPi*stack/kSphereStacks, phi=2*kPi*slice/kSphereSlices;
        return Vec3{std::sin(theta)*std::cos(phi),std::cos(theta),std::sin(theta)*std::sin(phi)};
    };
    for (int stack=0;stack<kSphereStacks;++stack)
        for (int slice=0;slice<kSphereSlices;++slice) {
            const Vec3 a=point(stack,slice),b=point(stack+1,slice),c=point(stack+1,slice+1),d=point(stack,slice+1);
            if (stack>0) { out.push_back(a); out.push_back(b); out.push_back(d); }
            if (stack<kSphereStacks-1) { out.push_back(d); out.push_back(b); out.push_back(c); }
        }
    return out;
}
constexpr size_t kSphereVertices=size_t(kSphereSlices)*(kSphereStacks-1)*2*3;
constexpr size_t kCylinderVertices=size_t(kCylinderSides)*6;
constexpr size_t kVerticesPerHand=26*kSphereVertices+25*kCylinderVertices;
}

struct HandViewerXr::Impl {
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace local = XR_NULL_HANDLE, view = XR_NULL_HANDLE;
    XrSessionState state = XR_SESSION_STATE_UNKNOWN;
    XrEnvironmentBlendMode blend = XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
    PFN_xrConvertTimespecTimeToTimeKHR convertTime = nullptr;
    std::atomic<bool> running{false};  // also read by headPoseAt on the worker
    VkInstance vk_instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkFormat format = VK_FORMAT_UNDEFINED, depth_format = VK_FORMAT_D16_UNORM;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    struct Eye {
        XrSwapchain swapchain = XR_NULL_HANDLE;
        uint32_t width = 0, height = 0;
        std::vector<XrSwapchainImageVulkan2KHR> images;
        std::vector<VkImageView> views;
        std::vector<VkFramebuffer> framebuffers;
        VkImage depth = VK_NULL_HANDLE;
        VkDeviceMemory depth_memory = VK_NULL_HANDLE;
        VkImageView depth_view = VK_NULL_HANDLE;
    };
    std::array<Eye,2> eyes;
    // Two frames in flight: the CPU never waits for the frame it just submitted.
    struct InFlight {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkBuffer vertices = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        Vertex* mapped = nullptr;
    };
    std::array<InFlight,2> in_flight;
    std::vector<Vec3> sphere = unitSphere();
    uint64_t frames = 0;

    ~Impl() {
        if (device) vkDeviceWaitIdle(device);
        for (auto& f : in_flight) {
            if (f.mapped) vkUnmapMemory(device,f.memory);
            if (f.vertices) vkDestroyBuffer(device,f.vertices,nullptr);
            if (f.memory) vkFreeMemory(device,f.memory,nullptr);
            if (f.fence) vkDestroyFence(device,f.fence,nullptr);
        }
        if (pool) vkDestroyCommandPool(device,pool,nullptr);
        for (auto& eye : eyes) {
            for (auto framebuffer : eye.framebuffers) vkDestroyFramebuffer(device,framebuffer,nullptr);
            for (auto image_view : eye.views) vkDestroyImageView(device,image_view,nullptr);
            if (eye.depth_view) vkDestroyImageView(device,eye.depth_view,nullptr);
            if (eye.depth) vkDestroyImage(device,eye.depth,nullptr);
            if (eye.depth_memory) vkFreeMemory(device,eye.depth_memory,nullptr);
            if (eye.swapchain) xrDestroySwapchain(eye.swapchain);
        }
        if (pipeline) vkDestroyPipeline(device,pipeline,nullptr);
        if (layout) vkDestroyPipelineLayout(device,layout,nullptr);
        if (render_pass) vkDestroyRenderPass(device,render_pass,nullptr);
        if (view) xrDestroySpace(view);
        if (local) xrDestroySpace(local);
        if (session) xrDestroySession(session);
        if (device) vkDestroyDevice(device,nullptr);
        if (vk_instance) vkDestroyInstance(vk_instance,nullptr);
        if (instance) xrDestroyInstance(instance);
    }
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
        VkPhysicalDeviceMemoryProperties properties{}; vkGetPhysicalDeviceMemoryProperties(physical,&properties);
        for (uint32_t i=0;i<properties.memoryTypeCount;++i)
            if ((bits&(1u<<i)) && (properties.memoryTypes[i].propertyFlags&flags)==flags) return i;
        throw std::runtime_error("no suitable Vulkan memory type");
    }
    void createInstance() {
        uint32_t count=0;
        check(xrEnumerateInstanceExtensionProperties(nullptr,0,&count,nullptr),"xrEnumerateInstanceExtensionProperties");
        std::vector<XrExtensionProperties> available(count,{XR_TYPE_EXTENSION_PROPERTIES});
        check(xrEnumerateInstanceExtensionProperties(nullptr,count,&count,available.data()),"xrEnumerateInstanceExtensionProperties");
        auto has=[&](const char* name) {
            return std::any_of(available.begin(),available.end(),[&](const auto& e) { return !std::strcmp(e.extensionName,name); });
        };
        std::vector<const char*> extensions{XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
        const bool timespec=has(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
        if (timespec) extensions.push_back(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
        XrInstanceCreateInfo info{XR_TYPE_INSTANCE_CREATE_INFO};
        std::strcpy(info.applicationInfo.applicationName,"Frame Hand Joints");
        info.applicationInfo.apiVersion = XR_MAKE_VERSION(1,0,0);
        info.enabledExtensionCount = uint32_t(extensions.size()); info.enabledExtensionNames = extensions.data();
        check(xrCreateInstance(&info,&instance),"xrCreateInstance");
        if (timespec) convertTime=proc<PFN_xrConvertTimespecTimeToTimeKHR>(instance,"xrConvertTimespecTimeToTimeKHR");
        else std::cerr << "[Hand OpenXR] XR_KHR_convert_timespec_time unavailable; joints follow the head\n";
        XrSystemGetInfo get{XR_TYPE_SYSTEM_GET_INFO}; get.formFactor=XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        check(xrGetSystem(instance,&get,&system),"xrGetSystem");
    }
    void createDevice() {
        XrGraphicsRequirementsVulkan2KHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
        check(proc<PFN_xrGetVulkanGraphicsRequirements2KHR>(instance,"xrGetVulkanGraphicsRequirements2KHR")(
            instance,system,&requirements),"xrGetVulkanGraphicsRequirements2KHR");
        const auto vkVersion=[](XrVersion v) { return VK_MAKE_API_VERSION(0,XR_VERSION_MAJOR(v),XR_VERSION_MINOR(v),XR_VERSION_PATCH(v)); };
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName="Frame Hand Joints";
        app.apiVersion=std::max(vkVersion(requirements.minApiVersionSupported),std::min(VK_API_VERSION_1_1,vkVersion(requirements.maxApiVersionSupported)));
        VkInstanceCreateInfo vi{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; vi.pApplicationInfo=&app;
        XrVulkanInstanceCreateInfoKHR xvi{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
        xvi.systemId=system; xvi.pfnGetInstanceProcAddr=vkGetInstanceProcAddr; xvi.vulkanCreateInfo=&vi;
        VkResult result=VK_SUCCESS;
        check(proc<PFN_xrCreateVulkanInstanceKHR>(instance,"xrCreateVulkanInstanceKHR")(instance,&xvi,&vk_instance,&result),"xrCreateVulkanInstanceKHR");
        check(result,"vkCreateInstance");
        XrVulkanGraphicsDeviceGetInfoKHR gpu{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        gpu.systemId=system; gpu.vulkanInstance=vk_instance;
        check(proc<PFN_xrGetVulkanGraphicsDevice2KHR>(instance,"xrGetVulkanGraphicsDevice2KHR")(instance,&gpu,&physical),"xrGetVulkanGraphicsDevice2KHR");
        uint32_t count=0; vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
        std::vector<VkQueueFamilyProperties> families(count); vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
        auto found=std::find_if(families.begin(),families.end(),[](const auto& p) { return p.queueFlags&VK_QUEUE_GRAPHICS_BIT; });
        if (found==families.end()) throw std::runtime_error("no Vulkan graphics queue");
        family=uint32_t(found-families.begin());
        const float priority=1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi;
        XrVulkanDeviceCreateInfoKHR xd{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
        xd.systemId=system; xd.pfnGetInstanceProcAddr=vkGetInstanceProcAddr; xd.vulkanPhysicalDevice=physical; xd.vulkanCreateInfo=&di;
        check(proc<PFN_xrCreateVulkanDeviceKHR>(instance,"xrCreateVulkanDeviceKHR")(instance,&xd,&device,&result),"xrCreateVulkanDeviceKHR");
        check(result,"vkCreateDevice"); vkGetDeviceQueue(device,family,0,&queue);
    }
    void createSession() {
        XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
        binding.instance=vk_instance; binding.physicalDevice=physical; binding.device=device; binding.queueFamilyIndex=family;
        XrSessionCreateInfo si{XR_TYPE_SESSION_CREATE_INFO}; si.next=&binding; si.systemId=system;
        check(xrCreateSession(instance,&si,&session),"xrCreateSession");
        XrReferenceSpaceCreateInfo sp{XR_TYPE_REFERENCE_SPACE_CREATE_INFO}; sp.poseInReferenceSpace.orientation.w=1;
        sp.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_LOCAL;
        check(xrCreateReferenceSpace(session,&sp,&local),"xrCreateReferenceSpace(LOCAL)");
        sp.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_VIEW;
        check(xrCreateReferenceSpace(session,&sp,&view),"xrCreateReferenceSpace(VIEW)");
        uint32_t count=0;
        check(xrEnumerateEnvironmentBlendModes(instance,system,XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,0,&count,nullptr),"xrEnumerateEnvironmentBlendModes");
        std::vector<XrEnvironmentBlendMode> blends(count);
        check(xrEnumerateEnvironmentBlendModes(instance,system,XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,count,&count,blends.data()),"xrEnumerateEnvironmentBlendModes");
        // Frame keeps its color camera producer active for alpha-blend MR.
        if (std::find(blends.begin(),blends.end(),XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND)==blends.end())
            throw std::runtime_error("OpenXR alpha-blended passthrough is required");
        check(xrEnumerateSwapchainFormats(session,0,&count,nullptr),"xrEnumerateSwapchainFormats");
        std::vector<int64_t> formats(count); check(xrEnumerateSwapchainFormats(session,count,&count,formats.data()),"xrEnumerateSwapchainFormats");
        for (const auto candidate : {VK_FORMAT_R8G8B8A8_SRGB,VK_FORMAT_B8G8R8A8_SRGB,VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_B8G8R8A8_UNORM})
            if (std::find(formats.begin(),formats.end(),candidate)!=formats.end()) { format=candidate; break; }
        if (format==VK_FORMAT_UNDEFINED) throw std::runtime_error("no RGBA8 OpenXR swapchain format");
    }
    void createRenderPass() {
        VkAttachmentDescription attachments[2]{};
        attachments[0].format=format; attachments[0].samples=VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; attachments[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachments[0].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; attachments[0].finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[1]=attachments[0]; attachments[1].format=depth_format; attachments[1].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference color{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=1; subpass.pColorAttachments=&color; subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        info.attachmentCount=2; info.pAttachments=attachments; info.subpassCount=1; info.pSubpasses=&subpass;
        info.dependencyCount=1; info.pDependencies=&dependency;
        check(vkCreateRenderPass(device,&info,nullptr,&render_pass),"vkCreateRenderPass");
    }
    void createPipeline() {
        auto module=[&](const std::vector<uint32_t>& code) {
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize=code.size()*4; info.pCode=code.data();
            VkShaderModule result=VK_NULL_HANDLE; check(vkCreateShaderModule(device,&info,nullptr,&result),"vkCreateShaderModule");
            return result;
        };
        const VkShaderModule vertex=module(spirv("hand_joints.vert.spv")), fragment=module(spirv("hand_joints.frag.spv"));
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vertex; stages[0].pName="main";
        stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=fragment; stages[1].pName="main";
        VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attributes[3]{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT,uint32_t(offsetof(Vertex,position))},
            {1,0,VK_FORMAT_R32G32B32_SFLOAT,uint32_t(offsetof(Vertex,normal))},
            {2,0,VK_FORMAT_R8G8B8A8_UNORM,uint32_t(offsetof(Vertex,rgba))}};
        VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        input.vertexBindingDescriptionCount=1; input.pVertexBindingDescriptions=&binding;
        input.vertexAttributeDescriptionCount=3; input.pVertexAttributeDescriptions=attributes;
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount=1; viewport.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode=VK_POLYGON_MODE_FILL; raster.cullMode=VK_CULL_MODE_NONE; raster.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable=VK_TRUE; depth.depthWriteEnable=VK_TRUE; depth.depthCompareOp=VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount=1; blending.pAttachments=&attachment;
        const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states;
        VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(Push)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount=1; layout_info.pPushConstantRanges=&push;
        check(vkCreatePipelineLayout(device,&layout_info,nullptr,&layout),"vkCreatePipelineLayout");
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount=2; info.pStages=stages; info.pVertexInputState=&input; info.pInputAssemblyState=&assembly;
        info.pViewportState=&viewport; info.pRasterizationState=&raster; info.pMultisampleState=&multisample;
        info.pDepthStencilState=&depth; info.pColorBlendState=&blending; info.pDynamicState=&dynamic;
        info.layout=layout; info.renderPass=render_pass;
        const VkResult result=vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&info,nullptr,&pipeline);
        vkDestroyShaderModule(device,vertex,nullptr); vkDestroyShaderModule(device,fragment,nullptr);
        check(result,"vkCreateGraphicsPipelines");
    }
    void createEyes() {
        uint32_t count=0;
        check(xrEnumerateViewConfigurationViews(instance,system,XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,0,&count,nullptr),"xrEnumerateViewConfigurationViews");
        std::vector<XrViewConfigurationView> configs(count,{XR_TYPE_VIEW_CONFIGURATION_VIEW});
        check(xrEnumerateViewConfigurationViews(instance,system,XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,count,&count,configs.data()),"xrEnumerateViewConfigurationViews");
        if (count!=2) throw std::runtime_error("stereo views required");
        for (int i=0;i<2;++i) {
            auto& eye=eyes[i];
            eye.width=configs[i].recommendedImageRectWidth; eye.height=configs[i].recommendedImageRectHeight;
            XrSwapchainCreateInfo sc{XR_TYPE_SWAPCHAIN_CREATE_INFO};
            sc.usageFlags=XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT; sc.format=format; sc.sampleCount=1;
            sc.width=eye.width; sc.height=eye.height; sc.faceCount=1; sc.arraySize=1; sc.mipCount=1;
            check(xrCreateSwapchain(session,&sc,&eye.swapchain),"xrCreateSwapchain");
            check(xrEnumerateSwapchainImages(eye.swapchain,0,&count,nullptr),"xrEnumerateSwapchainImages");
            eye.images.resize(count,{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
            check(xrEnumerateSwapchainImages(eye.swapchain,count,&count,reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())),"xrEnumerateSwapchainImages");
            VkImageCreateInfo di{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            di.imageType=VK_IMAGE_TYPE_2D; di.format=depth_format; di.extent={eye.width,eye.height,1};
            di.mipLevels=1; di.arrayLayers=1; di.samples=VK_SAMPLE_COUNT_1_BIT; di.tiling=VK_IMAGE_TILING_OPTIMAL;
            di.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            check(vkCreateImage(device,&di,nullptr,&eye.depth),"vkCreateImage(depth)");
            VkMemoryRequirements needs{}; vkGetImageMemoryRequirements(device,eye.depth,&needs);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=needs.size;
            ai.memoryTypeIndex=memoryType(needs.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device,&ai,nullptr,&eye.depth_memory),"vkAllocateMemory(depth)");
            check(vkBindImageMemory(device,eye.depth,eye.depth_memory,0),"vkBindImageMemory(depth)");
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.image=eye.depth; vi.format=depth_format;
            vi.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
            check(vkCreateImageView(device,&vi,nullptr,&eye.depth_view),"vkCreateImageView(depth)");
            for (const auto& image : eye.images) {
                vi.image=image.image; vi.format=format; vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                VkImageView color=VK_NULL_HANDLE; check(vkCreateImageView(device,&vi,nullptr,&color),"vkCreateImageView");
                eye.views.push_back(color);
                const VkImageView attachments[]{color,eye.depth_view};
                VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
                fi.renderPass=render_pass; fi.attachmentCount=2; fi.pAttachments=attachments;
                fi.width=eye.width; fi.height=eye.height; fi.layers=1;
                VkFramebuffer framebuffer=VK_NULL_HANDLE; check(vkCreateFramebuffer(device,&fi,nullptr,&framebuffer),"vkCreateFramebuffer");
                eye.framebuffers.push_back(framebuffer);
            }
        }
    }
    void createFrames() {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex=family; pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device,&pi,nullptr,&pool),"vkCreateCommandPool");
        for (auto& f : in_flight) {
            VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ca.commandPool=pool; ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount=1;
            check(vkAllocateCommandBuffers(device,&ca,&f.command),"vkAllocateCommandBuffers");
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
            check(vkCreateFence(device,&fi,nullptr,&f.fence),"vkCreateFence");
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size=kMaxHands*kVerticesPerHand*sizeof(Vertex); bi.usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            check(vkCreateBuffer(device,&bi,nullptr,&f.vertices),"vkCreateBuffer");
            VkMemoryRequirements needs{}; vkGetBufferMemoryRequirements(device,f.vertices,&needs);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=needs.size;
            ai.memoryTypeIndex=memoryType(needs.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check(vkAllocateMemory(device,&ai,nullptr,&f.memory),"vkAllocateMemory");
            check(vkBindBufferMemory(device,f.vertices,f.memory,0),"vkBindBufferMemory");
            void* mapped=nullptr; check(vkMapMemory(device,f.memory,0,VK_WHOLE_SIZE,0,&mapped),"vkMapMemory");
            f.mapped=static_cast<Vertex*>(mapped);
        }
    }
    void initialize() {
        if (sphere.size()!=kSphereVertices) throw std::runtime_error("hand sphere template size");
        createInstance(); createDevice(); createSession(); createRenderPass(); createPipeline(); createEyes(); createFrames();
        std::cout << "[Hand OpenXR] Session created, eyes " << eyes[0].width << 'x' << eyes[0].height
                  << (convertTime ? ", world-locked at capture time\n" : ", head-locked\n");
    }
    bool poll() {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        XrResult result;
        while ((result=xrPollEvent(instance,&event))==XR_SUCCESS) {
            if (event.type==XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) return false;
            if (event.type==XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                state=reinterpret_cast<XrEventDataSessionStateChanged*>(&event)->state;
                std::cout << "[Hand OpenXR] state=" << state << '\n';
                if (state==XR_SESSION_STATE_READY && !running) {
                    XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO}; begin.primaryViewConfigurationType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    check(xrBeginSession(session,&begin),"xrBeginSession"); running=true;
                } else if (state==XR_SESSION_STATE_STOPPING && running) {
                    check(xrEndSession(session),"xrEndSession"); running=false;
                } else if (state==XR_SESSION_STATE_EXITING || state==XR_SESSION_STATE_LOSS_PENDING) return false;
            }
            event={XR_TYPE_EVENT_DATA_BUFFER};
        }
        if (result!=XR_EVENT_UNAVAILABLE) check(result,"xrPollEvent");
        return true;
    }
    bool locate(XrTime time, XrPosef& pose) const {
        XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
        if (XR_FAILED(xrLocateSpace(view,local,time,&location))) return false;
        const XrSpaceLocationFlags required=XR_SPACE_LOCATION_POSITION_VALID_BIT|XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
        if ((location.locationFlags&required)!=required) return false;
        pose=location.pose; return true;
    }
    bool headPoseAt(uint64_t ns, HeadPose& out) const {
        if (!convertTime || !running) return false;
        timespec when{}; when.tv_sec=time_t(ns/1000000000ull); when.tv_nsec=long(ns%1000000000ull);
        XrTime time=0; XrPosef pose{};
        if (XR_FAILED(convertTime(instance,&when,&time)) || !locate(time,pose)) return false;
        out.orientation={pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w};
        out.position={pose.position.x,pose.position.y,pose.position.z};
        return true;
    }
    // Lit spheres at joints and cylinders along bones, in LOCAL space.
    size_t build(const HandScene& scene, XrTime display, Vertex* out) const {
        XrPosef head{}; head.orientation.w=1;
        if (!scene.local_space && !locate(display,head)) return 0;
        size_t n=0;
        auto emit=[&](Vec3 p, Vec3 normal, uint32_t color) {
            auto& v=out[n++];
            std::memcpy(v.position,p.data(),12); std::memcpy(v.normal,normal.data(),12); v.rgba=color;
        };
        const size_t count=std::min(scene.hands.size(),size_t(kMaxHands));
        for (size_t h=0;h<count;++h) {
            const auto& hand=scene.hands[h];
            std::array<Vec3,26> joint;
            bool finite=true;
            if (hand.joint_count!=21 && hand.joint_count!=26) continue;
            for (int i=0;i<hand.joint_count;++i) {
                joint[i]=hand.joints[i];
                if (!scene.local_space) joint[i]=rotate(head.orientation,joint[i])+Vec3{head.position.x,head.position.y,head.position.z};
                finite=finite && std::isfinite(joint[i][0]+joint[i][1]+joint[i][2]);
            }
            if (!finite) continue;
            const uint32_t joint_color=hand.right?rgba(110,215,255):rgba(255,180,90);
            const uint32_t tip_color=hand.right?rgba(200,245,255):rgba(255,230,170);
            const uint32_t bone_color=hand.right?rgba(40,140,215):rgba(215,115,40);
            for (int i=0;i<hand.joint_count;++i) {
                const bool tip=hand.joint_count==26 ? (i==5||i==10||i==15||i==20||i==25)
                                                    : (i==4||i==8||i==12||i==16||i==20);
                const float radius=hand.radii[i]>0 ? hand.radii[i] : (i==0?.0095f:tip?.0065f:.0075f);
                for (const auto& unit : sphere) emit(joint[i]+unit*radius,unit,tip?tip_color:joint_color);
            }
            auto draw_bone=[&](const int (&bone)[2]) {
                const Vec3 a=joint[bone[0]], b=joint[bone[1]], axis=b-a;
                const float len=length(axis);
                if (len<1e-5f) return;
                const Vec3 d=axis*(1/len);
                const Vec3 helper=std::abs(d[0])<.9f?Vec3{1,0,0}:Vec3{0,1,0};
                Vec3 x=cross(d,helper); x=x*(1/length(x)); const Vec3 y=cross(d,x);
                constexpr float r=.0035f;
                for (int s=0;s<kCylinderSides;++s) {
                    const float t0=2*kPi*s/kCylinderSides, t1=2*kPi*(s+1)/kCylinderSides;
                    const Vec3 n0=x*std::cos(t0)+y*std::sin(t0), n1=x*std::cos(t1)+y*std::sin(t1);
                    emit(a+n0*r,n0,bone_color); emit(b+n0*r,n0,bone_color); emit(b+n1*r,n1,bone_color);
                    emit(a+n0*r,n0,bone_color); emit(b+n1*r,n1,bone_color); emit(a+n1*r,n1,bone_color);
                }
            };
            if (hand.joint_count==26) {
                for (const auto& bone : kMercuryBones) draw_bone(bone);
            } else {
                for (const auto& bone : kBones) draw_bone(bone);
            }
        }
        return n;
    }
    void render(const HandScene& scene) {
        if (!running) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); return; }
        XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO}; XrFrameState frame{XR_TYPE_FRAME_STATE};
        check(xrWaitFrame(session,&wait,&frame),"xrWaitFrame");
        XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO}; check(xrBeginFrame(session,&begin),"xrBeginFrame");
        std::array<XrView,2> views{{{XR_TYPE_VIEW},{XR_TYPE_VIEW}}};
        std::array<XrCompositionLayerProjectionView,2> projection_views{{{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}}};
        XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        bool submitted=false;
        if (frame.shouldRender) {
            XrViewLocateInfo locate_info{XR_TYPE_VIEW_LOCATE_INFO}; locate_info.viewConfigurationType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            locate_info.displayTime=frame.predictedDisplayTime; locate_info.space=local;
            XrViewState located{XR_TYPE_VIEW_STATE}; uint32_t count=0;
            check(xrLocateViews(session,&locate_info,&located,2,&count,views.data()),"xrLocateViews");
            const XrViewStateFlags valid=XR_VIEW_STATE_POSITION_VALID_BIT|XR_VIEW_STATE_ORIENTATION_VALID_BIT;
            if (count==2 && (located.viewStateFlags&valid)==valid) {
                auto& f=in_flight[frames%2];
                check(vkWaitForFences(device,1,&f.fence,VK_TRUE,2000000000ull),"vkWaitForFences");
                const size_t vertices=build(scene,frame.predictedDisplayTime,f.mapped);
                std::array<uint32_t,2> index{};
                for (int i=0;i<2;++i) {
                    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
                    check(xrAcquireSwapchainImage(eyes[i].swapchain,&acquire,&index[i]),"xrAcquireSwapchainImage");
                    XrSwapchainImageWaitInfo ready{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO}; ready.timeout=XR_INFINITE_DURATION;
                    check(xrWaitSwapchainImage(eyes[i].swapchain,&ready),"xrWaitSwapchainImage");
                }
                check(vkResetFences(device,1,&f.fence),"vkResetFences");
                check(vkResetCommandBuffer(f.command,0),"vkResetCommandBuffer");
                VkCommandBufferBeginInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                check(vkBeginCommandBuffer(f.command,&cb),"vkBeginCommandBuffer");
                for (int i=0;i<2;++i) {
                    const auto& eye=eyes[i];
                    VkClearValue clear[2]{}; clear[0].color={{0,0,0,0}}; clear[1].depthStencil={1,0};
                    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                    rp.renderPass=render_pass; rp.framebuffer=eye.framebuffers.at(index[i]);
                    rp.renderArea.extent={eye.width,eye.height}; rp.clearValueCount=2; rp.pClearValues=clear;
                    vkCmdBeginRenderPass(f.command,&rp,VK_SUBPASS_CONTENTS_INLINE);
                    if (vertices) {
                        const VkViewport viewport{0,0,float(eye.width),float(eye.height),0,1};
                        const VkRect2D scissor{{0,0},{eye.width,eye.height}};
                        vkCmdSetViewport(f.command,0,1,&viewport); vkCmdSetScissor(f.command,0,1,&scissor);
                        vkCmdBindPipeline(f.command,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
                        const VkDeviceSize offset=0; vkCmdBindVertexBuffers(f.command,0,1,&f.vertices,&offset);
                        Push push{};
                        const Mat4 matrix=FrameSpatialCloud::multiply(FrameSpatialCloud::projectionFromFov(views[i].fov,.03f,20.f),
                                                                      FrameSpatialCloud::viewFromPose(views[i].pose));
                        std::memcpy(push.view_projection,matrix.m,sizeof(matrix.m));
                        push.eye[0]=views[i].pose.position.x; push.eye[1]=views[i].pose.position.y; push.eye[2]=views[i].pose.position.z;
                        vkCmdPushConstants(f.command,layout,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(push),&push);
                        vkCmdDraw(f.command,uint32_t(vertices),1,0,0);
                    }
                    vkCmdEndRenderPass(f.command);
                }
                check(vkEndCommandBuffer(f.command),"vkEndCommandBuffer");
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&f.command;
                check(vkQueueSubmit(queue,1,&submit,f.fence),"vkQueueSubmit");
                for (int i=0;i<2;++i) {
                    XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                    check(xrReleaseSwapchainImage(eyes[i].swapchain,&release),"xrReleaseSwapchainImage");
                    projection_views[i].pose=views[i].pose; projection_views[i].fov=views[i].fov;
                    projection_views[i].subImage.swapchain=eyes[i].swapchain;
                    projection_views[i].subImage.imageRect.extent={int32_t(eyes[i].width),int32_t(eyes[i].height)};
                }
                // SteamVR's Frame passthrough is revealed through a transparent projection layer.
                projection.layerFlags=XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
                projection.space=local; projection.viewCount=2; projection.views=projection_views.data();
                submitted=true;
            }
        }
        const XrCompositionLayerBaseHeader* layer=reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection);
        XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO}; end.displayTime=frame.predictedDisplayTime; end.environmentBlendMode=blend;
        end.layerCount=submitted?1:0; end.layers=submitted?&layer:nullptr;
        check(xrEndFrame(session,&end),"xrEndFrame");
        if (++frames%600==0) std::cout << "[Hand OpenXR] submitted=" << frames << " visible=" << frame.shouldRender << '\n';
    }
};
HandViewerXr::HandViewerXr() : impl_(std::make_unique<Impl>()) {}
HandViewerXr::~HandViewerXr()=default;
void HandViewerXr::initialize() { impl_->initialize(); }
bool HandViewerXr::pollEvents() { return impl_->poll(); }
bool HandViewerXr::active() const { return impl_->running && (impl_->state==XR_SESSION_STATE_VISIBLE || impl_->state==XR_SESSION_STATE_FOCUSED); }
bool HandViewerXr::headPoseAt(uint64_t ns, HeadPose& pose) const { return impl_->headPoseAt(ns,pose); }
void HandViewerXr::render(const HandScene& scene) { impl_->render(scene); }
