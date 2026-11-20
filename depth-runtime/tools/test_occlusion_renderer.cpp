#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h>
#include "occlusion_renderer.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <limits>
using namespace FrameSpatialCloud;
void check(VkResult r) { if (r!=VK_SUCCESS) throw std::runtime_error("Vulkan failure "+std::to_string(r)); }
struct TestGpu {
    VkInstance instance{}; VkPhysicalDevice physical{}; VkDevice device{}; VkQueue queue{};
    VkCommandPool pool{}; VkCommandBuffer cmd{}; VkImage color{},depth{},colorSecond{},depthSecond{},colorAlternate{};
    VkDeviceMemory colorMemory{},depthMemory{},colorSecondMemory{},depthSecondMemory{},colorAlternateMemory{},readMemory{}; VkBuffer read{};
    uint32_t family=0; static constexpr uint32_t size=65;
    uint32_t memoryType(uint32_t bits,VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties p{}; vkGetPhysicalDeviceMemoryProperties(physical,&p);
        for (uint32_t i=0;i<p.memoryTypeCount;++i) if ((bits&(1u<<i)) && (p.memoryTypes[i].propertyFlags&flags)==flags) return i;
        throw std::runtime_error("No memory type");
    }
    void image(VkFormat format,VkImageUsageFlags usage,VkImage& output,VkDeviceMemory& memory, uint32_t layers=1) {
        VkImageCreateInfo i{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; i.imageType=VK_IMAGE_TYPE_2D;
        i.format=format; i.extent={size,size,1}; i.mipLevels=1; i.arrayLayers=layers; i.samples=VK_SAMPLE_COUNT_1_BIT;
        i.tiling=VK_IMAGE_TILING_OPTIMAL; i.usage=usage; check(vkCreateImage(device,&i,nullptr,&output));
        VkMemoryRequirements r{}; vkGetImageMemoryRequirements(device,output,&r);
        VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; a.allocationSize=r.size;
        a.memoryTypeIndex=memoryType(r.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device,&a,nullptr,&memory)); check(vkBindImageMemory(device,output,memory,0));
    }
    TestGpu() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo i{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; i.pApplicationInfo=&app;
        check(vkCreateInstance(&i,nullptr,&instance)); uint32_t n=1; check(vkEnumeratePhysicalDevices(instance,&n,&physical));
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&n,nullptr); std::vector<VkQueueFamilyProperties> families(n);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&n,families.data());
        while (family<n && !(families[family].queueFlags&VK_QUEUE_GRAPHICS_BIT)) ++family;
        float priority=1; VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex=family; q.queueCount=1; q.pQueuePriorities=&priority;
        VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; d.queueCreateInfoCount=1; d.pQueueCreateInfos=&q;
        check(vkCreateDevice(physical,&d,nullptr,&device)); vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; p.queueFamilyIndex=family; p.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device,&p,nullptr,&pool));
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ca.commandPool=pool; ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&ca,&cmd));
        image(VK_FORMAT_R8G8B8A8_UNORM,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,color,colorMemory);
        image(VK_FORMAT_D32_SFLOAT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,depth,depthMemory);
        image(VK_FORMAT_R8G8B8A8_UNORM,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,colorSecond,colorSecondMemory,2);
        image(VK_FORMAT_D32_SFLOAT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,depthSecond,depthSecondMemory,2);
        image(VK_FORMAT_B8G8R8A8_UNORM,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,colorAlternate,colorAlternateMemory);
        VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; b.size=size*size*4; b.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device,&b,nullptr,&read)); VkMemoryRequirements r{}; vkGetBufferMemoryRequirements(device,read,&r);
        VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; a.allocationSize=r.size;
        a.memoryTypeIndex=memoryType(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device,&a,nullptr,&readMemory)); check(vkBindBufferMemory(device,read,readMemory,0));
    }
    void begin() { check(vkResetCommandBuffer(cmd,0)); VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(cmd,&b)); }
    void end() { check(vkEndCommandBuffer(cmd)); VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO}; s.commandBufferCount=1; s.pCommandBuffers=&cmd; check(vkQueueSubmit(queue,1,&s,VK_NULL_HANDLE)); check(vkQueueWaitIdle(queue)); }
    void barrier(VkImage image,VkImageAspectFlags aspect,VkImageLayout from,VkImageLayout to, uint32_t layer=0) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout=from; b.newLayout=to;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=image; b.subresourceRange={aspect,0,1,layer,1};
        b.srcAccessMask=from==VK_IMAGE_LAYOUT_UNDEFINED?0:VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
    }
    void clear(float virtualDepth, bool second=false, uint32_t layer=0) {
        const VkImage color=second?colorSecond:this->color, depth=second?depthSecond:this->depth;
        begin(); barrier(color,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,layer);
        barrier(depth,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,layer);
        VkClearColorValue c{{1,0.5f,0.25f,1}}; VkImageSubresourceRange cr{VK_IMAGE_ASPECT_COLOR_BIT,0,1,layer,1};
        VkImageSubresourceRange dr{VK_IMAGE_ASPECT_DEPTH_BIT,0,1,layer,1}; VkClearDepthStencilValue d{virtualDepth,0};
        vkCmdClearColorImage(cmd,color,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&c,1,&cr);
        vkCmdClearDepthStencilImage(cmd,depth,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&d,1,&dr);
        barrier(color,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,layer);
        barrier(depth,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,layer); end();
    }
    void verify(bool masked, uint32_t x=32, uint32_t y=32, bool second=false, uint32_t layer=0) {
        const VkImage color=second?colorSecond:this->color;
        begin(); barrier(color,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,layer);
        VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,layer,1}; copy.imageExtent={size,size,1};
        vkCmdCopyImageToBuffer(cmd,color,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,read,1,&copy); end();
        void* data; check(vkMapMemory(device,readMemory,0,VK_WHOLE_SIZE,0,&data));
        auto pixels=static_cast<uint8_t*>(data); size_t center=(y*size+x)*4;
        bool result=pixels[center+3]==(masked?0:255) && pixels[3]==255;
        if (masked) result=result && pixels[center]==0 && pixels[center+1]==0 && pixels[center+2]==0;
        vkUnmapMemory(device,readMemory);
        if (!result) throw std::runtime_error("Unexpected occlusion/foreground or outside-surface pixel");
    }
    ~TestGpu() {
        vkDeviceWaitIdle(device); vkDestroyBuffer(device,read,nullptr); vkFreeMemory(device,readMemory,nullptr);
        vkDestroyImage(device,color,nullptr); vkDestroyImage(device,depth,nullptr);
        vkFreeMemory(device,colorMemory,nullptr); vkFreeMemory(device,depthMemory,nullptr);
        vkDestroyImage(device,colorSecond,nullptr); vkDestroyImage(device,depthSecond,nullptr);
        vkFreeMemory(device,colorSecondMemory,nullptr); vkFreeMemory(device,depthSecondMemory,nullptr);
        vkDestroyImage(device,colorAlternate,nullptr); vkFreeMemory(device,colorAlternateMemory,nullptr);
        vkDestroyCommandPool(device,pool,nullptr); vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    }
};
int main() {
    try {
        TestGpu gpu; OcclusionRenderer renderer;
        if (!renderer.initialize(gpu.physical,gpu.device,gpu.family,0,"shaders")) throw std::runtime_error("Renderer initialization failed");
        auto plane=[&](float right) {
            return std::vector<PointVertex>{{-0.25f,-0.25f,-1,0xffffffff}, {right,-0.25f,-1,0xffffffff},
                {right,0.25f,-1,0xffffffff}, {-0.25f,-0.25f,-1,0xffffffff},
                {right,0.25f,-1,0xffffffff}, {-0.25f,0.25f,-1,0xffffffff}};
        };
        if (!renderer.upload(plane(0.25f))) throw std::runtime_error("Surface upload failed");
        XrFovf fov{-0.785398f,0.785398f,0.785398f,-0.785398f};
        auto matrix=projectionFromFov(fov,0.08f,12);
        XrRect2Di rect{{0,0},{65,65}};
        XrCompositionLayerDepthInfoKHR range{XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR}; range.minDepth=0; range.maxDepth=1;
        auto run=[&](float virtualZ,float nearZ,float farZ,bool masked) {
            range.nearZ=nearZ; range.farZ=farZ;
            float normalized=std::isinf(farZ)?1-nearZ/virtualZ:std::isinf(nearZ)?farZ/virtualZ:(farZ-nearZ*farZ/virtualZ)/(farZ-nearZ);
            gpu.clear(normalized);
            if (!renderer.mask(gpu.color,VK_FORMAT_R8G8B8A8_UNORM,gpu.depth,VK_FORMAT_D32_SFLOAT,0,0,rect,rect,matrix,range)) throw std::runtime_error("Mask failed");
            gpu.verify(masked);
        };
        run(2,0.1f,10,true); run(0.5f,0.1f,10,false); run(1.02f,0.1f,10,false);
        run(2,10,0.1f,true); run(0.5f,10,0.1f,false);
        run(2,0.1f,INFINITY,true); run(0.5f,0.1f,INFINITY,false);
        run(2,INFINITY,0.1f,true); run(0.5f,INFINITY,0.1f,false);
        renderer.upload(plane(1.0f/65));
        run(2,0.1f,10,true);
        gpu.verify(true,31); gpu.verify(false,33);
        // Perspective depth must vary across a slanted measured surface.
        renderer.upload({{-0.5f,-0.5f,-1,0xffffffff}, {1.5f,-1.5f,-3,0xffffffff},
            {1.5f,1.5f,-3,0xffffffff}, {-0.5f,-0.5f,-1,0xffffffff},
            {1.5f,1.5f,-3,0xffffffff}, {-0.5f,0.5f,-1,0xffffffff}});
        run(2,0.1f,10,true);
        gpu.verify(true,24); gpu.verify(false,40);
        // A stereo batch must retain two distinct immutable descriptors. The
        // left virtual surface is behind the real plane; right is in front.
        renderer.upload(plane(0.25f));
        range.nearZ=0.1f; range.farZ=10;
        const auto normalized=[](float z) { return (10.f-1.f/z)/9.9f; };
        gpu.clear(normalized(2)); gpu.clear(normalized(0.5f),true);
        std::vector<OcclusionMaskTarget> targets{
            {gpu.color,gpu.depth,VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_D32_SFLOAT,0,0,rect,rect,matrix,range},
            {gpu.colorSecond,gpu.depthSecond,VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_D32_SFLOAT,0,0,rect,rect,matrix,range}};
        const auto individual=renderer.stats();
        for (const auto& target:targets)
            if (!renderer.maskBatch({target})) throw std::runtime_error("Per-eye reference failed");
        if (renderer.stats().submissions!=individual.submissions+2 ||
            renderer.stats().fenceWaits!=individual.fenceWaits+2)
            throw std::runtime_error("Per-eye reference count changed");
        gpu.verify(true); gpu.verify(false,32,32,true);
        gpu.clear(normalized(2)); gpu.clear(normalized(0.5f),true);
        const auto previous=renderer.stats();
        if (!renderer.maskBatch(targets)) throw std::runtime_error("Stereo batch failed");
        if (renderer.stats().submissions!=previous.submissions+1 ||
            renderer.stats().fenceWaits!=previous.fenceWaits+1)
            throw std::runtime_error("Stereo batch submitted/waited more than once");
        gpu.verify(true); gpu.verify(false,32,32,true);
        const auto cached=renderer.cachedTargetCount(), hits=renderer.stats().cacheHits;
        gpu.clear(normalized(2)); gpu.clear(normalized(0.5f),true);
        if (!renderer.maskBatch(targets) || renderer.cachedTargetCount()!=cached ||
            renderer.stats().cacheHits!=hits+2) throw std::runtime_error("Swapchain cache was not reused");
        gpu.verify(true); gpu.verify(false,32,32,true);
        // Stereo array layers require distinct views/descriptors despite sharing
        // the same image handles. The unmasked eye must not sample the other eye.
        targets[0].color=gpu.colorSecond; targets[0].depth=gpu.depthSecond;
        targets[1].colorLayer=targets[1].depthLayer=1;
        gpu.clear(normalized(2),true,0); gpu.clear(normalized(0.5f),true,1);
        if (!renderer.maskBatch(targets)) throw std::runtime_error("Array-layer stereo batch failed");
        gpu.verify(true,32,32,true,0); gpu.verify(false,32,32,true,1);
        renderer.releaseImage(gpu.colorSecond);
        if (renderer.cachedTargetCount()!=1) throw std::runtime_error("Color swapchain cache did not release both layers");
        renderer.releaseImage(gpu.depth);
        if (renderer.cachedTargetCount()!=0) throw std::runtime_error("Depth swapchain cache did not release paired color views");
        // Recreation after lifecycle invalidation and a changed framebuffer
        // extent creates fresh entries. Existing images remain runtime-owned.
        gpu.clear(normalized(2));
        XrRect2Di smaller{{0,0},{64,64}};
        if (!renderer.mask(gpu.color,VK_FORMAT_R8G8B8A8_UNORM,gpu.depth,VK_FORMAT_D32_SFLOAT,0,0,
                           smaller,smaller,matrix,range) || renderer.cachedTargetCount()!=1)
            throw std::runtime_error("Cache recreation failed");
        gpu.verify(true);
        gpu.clear(normalized(2));
        if (!renderer.mask(gpu.color,VK_FORMAT_R8G8B8A8_UNORM,gpu.depth,VK_FORMAT_D32_SFLOAT,0,0,
                           rect,rect,matrix,range) || renderer.cachedTargetCount()!=2)
            throw std::runtime_error("Framebuffer extent missing from cache key");
        gpu.verify(true);
        // Changing the color render-pass format invalidates every cached
        // framebuffer before destroying the old render pass.
        std::swap(gpu.color,gpu.colorAlternate);
        gpu.clear(normalized(2));
        if (!renderer.mask(gpu.color,VK_FORMAT_B8G8R8A8_UNORM,gpu.depth,VK_FORMAT_D32_SFLOAT,0,0,
                           rect,rect,matrix,range) || renderer.cachedTargetCount()!=1)
            throw std::runtime_error("Pipeline format failed to invalidate cache");
        gpu.verify(true);
        std::swap(gpu.color,gpu.colorAlternate);
        run(2,0.1f,10,true);
        if (renderer.cachedTargetCount()!=1) throw std::runtime_error("Pipeline format cache restore failed");
        // Mixed-age triangles expire individually at predicted display age.
        // Updating age alone never uploads or republishes the surface.
        std::vector<PointVertex> aged{
            {-.45f,-.3f,-1,0xffffffff,.1f}, {-.1f,-.3f,-1,0xffffffff,.1f},
            {-.1f,.3f,-1,0xffffffff,.1f}, {-.45f,-.3f,-1,0xffffffff,.1f},
            {-.1f,.3f,-1,0xffffffff,.1f}, {-.45f,.3f,-1,0xffffffff,.1f},
            {.1f,-.3f,-1,0xffffffff,.3f}, {.45f,-.3f,-1,0xffffffff,.3f},
            {.45f,.3f,-1,0xffffffff,.3f}, {.1f,-.3f,-1,0xffffffff,.3f},
            {.45f,.3f,-1,0xffffffff,.3f}, {.1f,.3f,-1,0xffffffff,.3f}};
        if (!renderer.upload(aged)) throw std::runtime_error("Timed surface upload failed");
        targets[0]={gpu.color,gpu.depth,VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_D32_SFLOAT,
                    0,0,rect,rect,matrix,range};
        targets[1]={gpu.colorSecond,gpu.depthSecond,VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_D32_SFLOAT,
                    0,0,rect,rect,matrix,range};
        const auto timed=[&](float age, bool oldVisible, bool youngVisible) {
            gpu.clear(normalized(2)); gpu.clear(normalized(2),true);
            for (auto& target:targets) target.displayAge=age;
            const auto before=renderer.stats();
            if (!renderer.maskBatch(targets) || renderer.stats().submissions!=before.submissions+1 ||
                renderer.stats().fenceWaits!=before.fenceWaits+1)
                throw std::runtime_error("Timed batch failed or added submissions");
            gpu.verify(oldVisible,24); gpu.verify(youngVisible,40);
            gpu.verify(oldVisible,24,32,true); gpu.verify(youngVisible,40,32,true);
        };
        timed(.05f,true,true);
        timed(.1f,false,true); // Inclusive deadline: equality is already expired.
        timed(.2f,false,true);
        timed(.3f,false,false);
        // Invalid lifetime must never mask under the runtime's timed path.
        for(float invalid : {0.f,-1.f,std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
            for(auto& vertex:aged) vertex.supportLifetime=invalid;
            if (!renderer.upload(aged)) throw std::runtime_error("Invalid lifetime mesh upload failed");
            timed(0,false,false);
        }
        if (renderer.upload({{0,0,-1,0xffffffff}})) throw std::runtime_error("Incomplete triangle accepted");
        run(2,0.1f,10,false);
        renderer.upload({}); run(2,0.1f,10,false);
        std::cout<<"Stereo operation counts: per-eye reference submits/waits=2/2; batch=1/1; "
                 <<"warm batch cache hits/misses=2/0 (zero new views/framebuffers)\n";
        std::cout<<"Occlusion GPU tests passed: behind/in front, bias, reverse Z, infinite planes, empty surface, sharp silhouette, slanted surface, malformed mesh, batch eye/layer descriptors, one submit/wait, cache reuse/release/extent/format, mixed triangle display expiry and invalid lifetimes\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
