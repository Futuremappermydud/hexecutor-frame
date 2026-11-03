#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h>
#include "occlusion_renderer.hpp"
#include "calibration.hpp"
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <chrono>
#include <fstream>
#include <limits>

namespace FrameSpatialCloud {
namespace {
constexpr size_t kMaxSurfaceVertices = size_t((FrameSpatial::RECT_WIDTH - 1) / 2) *
    ((FrameSpatial::RECT_HEIGHT - 1) / 2) * 24;
struct Params { Mat4 matrix; float range[4]; float viewport[4]; float tuning[4]; };
std::vector<uint32_t> spirv(const std::string& path) {
    std::ifstream in(path, std::ios::binary|std::ios::ate);
    if (!in) return {};
    auto size=in.tellg();
    if (size<=0 || size%4) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size)/4);
    in.seekg(0); in.read(reinterpret_cast<char*>(code.data()),size);
    return in ? code : std::vector<uint32_t>{};
}
}
uint32_t OcclusionRenderer::memoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i=0;i<memory_.memoryTypeCount;++i)
        if ((bits&(1u<<i)) && (memory_.memoryTypes[i].propertyFlags&flags)==flags) return i;
    return UINT32_MAX;
}
bool OcclusionRenderer::initialize(VkPhysicalDevice physical, VkDevice device,
                                  uint32_t family, uint32_t index, const std::string& dir) {
    device_=device; shaderDir_=dir;
    if (spirv(dir+"/occlusion.vert.spv").empty() || spirv(dir+"/occlusion.frag.spv").empty()) return false;
    vkGetPhysicalDeviceMemoryProperties(physical,&memory_);
    vkGetDeviceQueue(device,family,index,&queue_);
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex=family; pool.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device,&pool,nullptr,&pool_)!=VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd.commandPool=pool_; cmd.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmd.commandBufferCount=1;
    if (vkAllocateCommandBuffers(device,&cmd,&command_)!=VK_SUCCESS) return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device,&fence,nullptr,&fence_)!=VK_SUCCESS) return false;
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size=kMaxSurfaceVertices*sizeof(PointVertex); buffer.usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (vkCreateBuffer(device,&buffer,nullptr,&buffer_)!=VK_SUCCESS) return false;
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device,buffer_,&req);
    uint32_t type=memoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type==UINT32_MAX) return false;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize=req.size; alloc.memoryTypeIndex=type;
    if (vkAllocateMemory(device,&alloc,nullptr,&allocation_)!=VK_SUCCESS ||
        vkBindBufferMemory(device,buffer_,allocation_,0)!=VK_SUCCESS ||
        vkMapMemory(device,allocation_,0,VK_WHOLE_SIZE,0,&mapped_)!=VK_SUCCESS) return false;
    VkDescriptorSetLayoutBinding binding{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo ds{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ds.bindingCount=1; ds.pBindings=&binding;
    if (vkCreateDescriptorSetLayout(device,&ds,nullptr,&descriptorLayout_)!=VK_SUCCESS) return false;
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,256};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp.maxSets=256; dp.poolSizeCount=1; dp.pPoolSizes=&size;
    if (vkCreateDescriptorPool(device,&dp,nullptr,&descriptorPool_)!=VK_SUCCESS) return false;
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter=sampler.minFilter=VK_FILTER_NEAREST;
    sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device,&sampler,nullptr,&sampler_)!=VK_SUCCESS) return false;
    VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Params)};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount=1; pl.pSetLayouts=&descriptorLayout_; pl.pushConstantRangeCount=1; pl.pPushConstantRanges=&push;
    return vkCreatePipelineLayout(device,&pl,nullptr,&layout_)==VK_SUCCESS;
}
bool OcclusionRenderer::createPipeline(VkFormat format) {
    if (format==format_ && pipeline_) return true;
    // All masks finish synchronously. Framebuffers must die before their pass.
    clearTargets();
    if (pipeline_) vkDestroyPipeline(device_,pipeline_,nullptr);
    if (renderPass_) vkDestroyRenderPass(device_,renderPass_,nullptr);
    pipeline_=VK_NULL_HANDLE; renderPass_=VK_NULL_HANDLE; format_=VK_FORMAT_UNDEFINED;
    VkAttachmentDescription attachment{};
    attachment.format=format; attachment.samples=VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout=attachment.finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ref{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount=1; sub.pColorAttachments=&ref;
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount=1; rp.pAttachments=&attachment; rp.subpassCount=1; rp.pSubpasses=&sub;
    if (vkCreateRenderPass(device_,&rp,nullptr,&renderPass_)!=VK_SUCCESS) return false;
    VkShaderModule modules[2]{};
    const char* names[]{"occlusion.vert.spv","occlusion.frag.spv"};
    bool ok=true;
    for (int i=0;i<2;++i) {
        auto code=spirv(shaderDir_+"/"+names[i]);
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize=code.size()*4; sm.pCode=code.data();
        if (code.empty() || vkCreateShaderModule(device_,&sm,nullptr,&modules[i])!=VK_SUCCESS) { ok=false; break; }
    }
    if (ok) {
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (int i=0;i<2;++i) { stages[i].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[i].stage=i==0?VK_SHADER_STAGE_VERTEX_BIT:VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[i].module=modules[i]; stages[i].pName="main"; }
        VkVertexInputBindingDescription binding{0,sizeof(PointVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription attrs[]{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT,0},
            {1,0,VK_FORMAT_R32_SFLOAT,offsetof(PointVertex,supportLifetime)}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&binding;
        vi.vertexAttributeDescriptionCount=2; vi.pVertexAttributeDescriptions=attrs;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vs.viewportCount=vs.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask=15;
        VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        bs.attachmentCount=1; bs.pAttachments=&ba;
        VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount=2; dyn.pDynamicStates=states;
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount=2; gp.pStages=stages; gp.pVertexInputState=&vi; gp.pInputAssemblyState=&ia;
        gp.pViewportState=&vs; gp.pRasterizationState=&rs; gp.pMultisampleState=&ms;
        gp.pColorBlendState=&bs; gp.pDynamicState=&dyn; gp.layout=layout_; gp.renderPass=renderPass_;
        ok=vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gp,nullptr,&pipeline_)==VK_SUCCESS;
    }
    for (auto module:modules) if (module) vkDestroyShaderModule(device_,module,nullptr);
    if (ok) format_=format;
    return ok;
}
bool OcclusionRenderer::upload(const std::vector<PointVertex>& points) {
    if (points.size() > kMaxSurfaceVertices || points.size() % 3 != 0) { count_=0; return false; }
    count_=static_cast<uint32_t>(points.size());
    if (count_) std::memcpy(mapped_,points.data(),count_*sizeof(PointVertex));
    return true;
}
void OcclusionRenderer::destroyTarget(CachedTarget& target) {
    if (target.framebuffer) vkDestroyFramebuffer(device_,target.framebuffer,nullptr);
    if (target.descriptor) vkFreeDescriptorSets(device_,descriptorPool_,1,&target.descriptor);
    if (target.colorView) vkDestroyImageView(device_,target.colorView,nullptr);
    if (target.depthView) vkDestroyImageView(device_,target.depthView,nullptr);
}
void OcclusionRenderer::clearTargets() {
    for (auto& target:targets_) destroyTarget(target);
    targets_.clear();
}
void OcclusionRenderer::releaseImage(VkImage image) {
    for (auto it=targets_.begin();it!=targets_.end();) {
        if (it->key.color==image || it->key.depth==image) {
            destroyTarget(*it); it=targets_.erase(it);
        } else ++it;
    }
}
OcclusionRenderer::CachedTarget* OcclusionRenderer::cachedTarget(const OcclusionMaskTarget& target) {
    const auto extent=[](const XrRect2Di& rect) {
        return std::pair<uint32_t,uint32_t>{uint32_t(rect.offset.x)+uint32_t(rect.extent.width),
                                            uint32_t(rect.offset.y)+uint32_t(rect.extent.height)};
    };
    for (auto& cached:targets_) {
        const auto& key=cached.key;
        if (key.color==target.color && key.depth==target.depth &&
            key.colorFormat==target.colorFormat && key.depthFormat==target.depthFormat &&
            key.colorLayer==target.colorLayer && key.depthLayer==target.depthLayer &&
            extent(key.colorRect)==extent(target.colorRect) && cached.renderPass==renderPass_) {
            ++stats_.cacheHits; return &cached;
        }
    }
    ++stats_.cacheMisses;
    CachedTarget cached{target,renderPass_};
    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.viewType=VK_IMAGE_VIEW_TYPE_2D;
    iv.image=target.color; iv.format=target.colorFormat;
    iv.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,target.colorLayer,1};
    if (vkCreateImageView(device_,&iv,nullptr,&cached.colorView)!=VK_SUCCESS) return nullptr;
    iv.image=target.depth; iv.format=target.depthFormat;
    iv.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,target.depthLayer,1};
    if (vkCreateImageView(device_,&iv,nullptr,&cached.depthView)!=VK_SUCCESS) {
        destroyTarget(cached); return nullptr;
    }
    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass=renderPass_; fb.attachmentCount=1; fb.pAttachments=&cached.colorView;
    const auto size=extent(target.colorRect); fb.width=size.first; fb.height=size.second; fb.layers=1;
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool=descriptorPool_; da.descriptorSetCount=1; da.pSetLayouts=&descriptorLayout_;
    if (vkCreateFramebuffer(device_,&fb,nullptr,&cached.framebuffer)!=VK_SUCCESS ||
        vkAllocateDescriptorSets(device_,&da,&cached.descriptor)!=VK_SUCCESS) {
        destroyTarget(cached); return nullptr;
    }
    VkDescriptorImageInfo ii{sampler_,cached.depthView,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet=cached.descriptor; write.descriptorCount=1;
    write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&ii;
    vkUpdateDescriptorSets(device_,1,&write,0,nullptr);
    targets_.push_back(cached);
    return &targets_.back();
}
bool OcclusionRenderer::mask(VkImage color,VkFormat format,VkImage depth,VkFormat depthFormat,
    uint32_t colorLayer,uint32_t depthLayer,const XrRect2Di& cr,const XrRect2Di& dr,
    const Mat4& matrix,const XrCompositionLayerDepthInfoKHR& range) {
    return maskBatch({{color,depth,format,depthFormat,colorLayer,depthLayer,cr,dr,matrix,range}});
}
bool OcclusionRenderer::maskBatch(const std::vector<OcclusionMaskTarget>& batch) {
    using Clock=std::chrono::steady_clock;
    const auto ms=[](Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    };
    stats_.prepareMs=stats_.recordMs=stats_.submitMs=stats_.waitMs=0;
    if (!count_ || batch.empty()) return true;
    for (const auto& target:batch) {
        if (target.colorFormat!=batch[0].colorFormat || !target.color || !target.depth ||
            target.colorRect.offset.x<0 || target.colorRect.offset.y<0 ||
            target.colorRect.extent.width<=0 || target.colorRect.extent.height<=0) return false;
    }
    auto start=Clock::now();
    if (!createPipeline(batch[0].colorFormat)) return false;
    // Reserve before retaining pointers: insertion cannot invalidate earlier eyes.
    targets_.reserve(targets_.size()+batch.size());
    std::vector<CachedTarget*> prepared;
    prepared.reserve(batch.size());
    for (const auto& target:batch) {
        auto cached=cachedTarget(target);
        if (!cached) return false;
        prepared.push_back(cached);
    }
    stats_.prepareMs=ms(start); start=Clock::now();
    if (vkResetCommandBuffer(command_,0)!=VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_,&begin)!=VK_SUCCESS) return false;
    for (size_t eye=0;eye<batch.size();++eye) {
        const auto& target=batch[eye]; const auto& cached=*prepared[eye];
        const auto& cr=target.colorRect; const auto& dr=target.depthRect;
        const auto& range=target.range;
        const VkImage images[]{target.color,target.depth};
        const uint32_t layers[]{target.colorLayer,target.depthLayer};
        VkImageMemoryBarrier barriers[2]{};
        for (int i=0;i<2;++i) {
            auto& b=barriers[i]; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask=i==0?VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT:VK_ACCESS_SHADER_READ_BIT;
            b.oldLayout=i==0?VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            b.newLayout=i==0?VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.image=images[i]; b.subresourceRange={i==0?VK_IMAGE_ASPECT_COLOR_BIT:VK_IMAGE_ASPECT_DEPTH_BIT,0,1,layers[i],1};
            if (i==1 && (target.depthFormat==VK_FORMAT_D24_UNORM_S8_UINT || target.depthFormat==VK_FORMAT_D32_SFLOAT_S8_UINT)) b.subresourceRange.aspectMask|=VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        vkCmdPipelineBarrier(command_,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,2,barriers);
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass=renderPass_; rp.framebuffer=cached.framebuffer; rp.renderArea={{cr.offset.x,cr.offset.y},{uint32_t(cr.extent.width),uint32_t(cr.extent.height)}};
        vkCmdBeginRenderPass(command_,&rp,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport{float(cr.offset.x),float(cr.offset.y),float(cr.extent.width),float(cr.extent.height),0,1};
        vkCmdSetViewport(command_,0,1,&viewport); vkCmdSetScissor(command_,0,1,&rp.renderArea);
        vkCmdBindPipeline(command_,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_);
        vkCmdBindDescriptorSets(command_,VK_PIPELINE_BIND_POINT_GRAPHICS,layout_,0,1,&cached.descriptor,0,nullptr);
        VkDeviceSize offset=0; vkCmdBindVertexBuffers(command_,0,1,&buffer_,&offset);
        Params params{target.viewProjection,{range.minDepth,range.maxDepth,range.nearZ,range.farZ},
            {float(cr.extent.width),float(cr.extent.height),float(dr.offset.x-cr.offset.x),float(dr.offset.y-cr.offset.y)},
            {target.displayAge,0.04f,0,0}};
        vkCmdPushConstants(command_,layout_,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(params),&params);
        vkCmdDraw(command_,count_,1,0,0); vkCmdEndRenderPass(command_);
        auto& b=barriers[1]; b.srcAccessMask=VK_ACCESS_SHADER_READ_BIT; b.dstAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::swap(b.oldLayout,b.newLayout);
        vkCmdPipelineBarrier(command_,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,0,0,nullptr,0,nullptr,1,&b);
    }
    if (vkEndCommandBuffer(command_)!=VK_SUCCESS || vkResetFences(device_,1,&fence_)!=VK_SUCCESS) return false;
    stats_.recordMs=ms(start); start=Clock::now();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&command_;
    if (vkQueueSubmit(queue_,1,&submit,fence_)!=VK_SUCCESS) return false;
    ++stats_.submissions; stats_.submitMs=ms(start); start=Clock::now();
    ++stats_.fenceWaits;
    const bool ok=vkWaitForFences(device_,1,&fence_,VK_TRUE,UINT64_MAX)==VK_SUCCESS;
    stats_.waitMs=ms(start);
    return ok;
}
OcclusionRenderer::~OcclusionRenderer() {
    if (!device_) return;
    clearTargets();
    if (mapped_) vkUnmapMemory(device_,allocation_);
    if (buffer_) vkDestroyBuffer(device_,buffer_,nullptr);
    if (allocation_) vkFreeMemory(device_,allocation_,nullptr);
    if (pipeline_) vkDestroyPipeline(device_,pipeline_,nullptr);
    if (renderPass_) vkDestroyRenderPass(device_,renderPass_,nullptr);
    if (layout_) vkDestroyPipelineLayout(device_,layout_,nullptr);
    if (sampler_) vkDestroySampler(device_,sampler_,nullptr);
    if (descriptorPool_) vkDestroyDescriptorPool(device_,descriptorPool_,nullptr);
    if (descriptorLayout_) vkDestroyDescriptorSetLayout(device_,descriptorLayout_,nullptr);
    if (fence_) vkDestroyFence(device_,fence_,nullptr);
    if (pool_) vkDestroyCommandPool(device_,pool_,nullptr);
}
}
