#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h>
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_loader_negotiation.h>
#include "cloud_worker.hpp"
#include "occlusion_capture_gate.hpp"
#include "occlusion_renderer.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace {
using namespace FrameSpatialCloud;
void profileMask(const OcclusionRenderer& renderer) {
    static const bool enabled=[] {
        const auto setting=std::getenv("FRAME_OCCLUSION_RENDER_PROFILE");
        return setting && std::string(setting)=="1";
    }();
    if (!enabled) return;
    static std::vector<std::array<double,4>> samples;
    const auto& stats=renderer.stats();
    samples.push_back({stats.prepareMs,stats.recordMs,stats.submitMs,stats.waitMs});
    if (samples.size()<300) return;
    const char* names[]{"prepare","record","submit","fence_wait"};
    std::cout<<"[OcclusionRenderProfile] "<<samples.size()<<" successful mask batches";
    for (size_t stage=0;stage<4;++stage) {
        std::vector<double> values; values.reserve(samples.size());
        for (const auto& sample:samples) values.push_back(sample[stage]);
        std::sort(values.begin(),values.end());
        std::cout<<", "<<names[stage]<<"_p50/p95_ms="
                 <<values[values.size()/2]<<'/'<<values[values.size()*95/100];
    }
    std::cout<<", total_submits="<<stats.submissions<<", total_waits="<<stats.fenceWaits
             <<", cache_hits/misses="<<stats.cacheHits<<'/'<<stats.cacheMisses<<'\n';
    samples.clear();
}
constexpr const char* kName="XR_APILAYER_FRAME_spatial_occlusion";
std::mutex mutex;
PFN_xrGetInstanceProcAddr nextGipa=nullptr;
XrInstance instance=XR_NULL_HANDLE;
XrSession activeSession=XR_NULL_HANDLE;
bool appDepth=false, timespecEnabled=false, alphaSupported=false, warned=false;
std::unique_ptr<OcclusionRenderer> renderer;
double uploadedCloudTime=0;
FrameSpatial::OcclusionCaptureGate captureGate;
bool uploadedSurface=false;
std::thread worker;
struct Swapchain {
    XrSwapchainCreateInfo info{};
    std::vector<XrSwapchainImageVulkanKHR> images;
    uint32_t acquired=0, released=0;
    bool waiting=false, waited=false, pending=false, eligible=false;
};
std::unordered_map<XrSwapchain,Swapchain> chains;
#define XR_DISPATCH(F) PFN_##F next##F=nullptr
XR_DISPATCH(xrDestroyInstance); XR_DISPATCH(xrCreateSession); XR_DISPATCH(xrDestroySession);
XR_DISPATCH(xrBeginSession); XR_DISPATCH(xrEndSession); XR_DISPATCH(xrCreateReferenceSpace);
XR_DISPATCH(xrDestroySpace); XR_DISPATCH(xrLocateSpace); XR_DISPATCH(xrCreateSwapchain);
XR_DISPATCH(xrDestroySwapchain); XR_DISPATCH(xrEnumerateSwapchainImages);
XR_DISPATCH(xrAcquireSwapchainImage); XR_DISPATCH(xrWaitSwapchainImage);
XR_DISPATCH(xrReleaseSwapchainImage); XR_DISPATCH(xrEndFrame);
XR_DISPATCH(xrEnumerateEnvironmentBlendModes);
XR_DISPATCH(xrConvertTimeToTimespecTimeKHR);
#undef XR_DISPATCH
bool isDepth(int64_t f) {
    return f==VK_FORMAT_D16_UNORM || f==VK_FORMAT_D32_SFLOAT || f==VK_FORMAT_D24_UNORM_S8_UINT || f==VK_FORMAT_D32_SFLOAT_S8_UINT;
}
bool isColor(int64_t f) {
    return f==VK_FORMAT_R8G8B8A8_UNORM || f==VK_FORMAT_R8G8B8A8_SRGB || f==VK_FORMAT_B8G8R8A8_UNORM || f==VK_FORMAT_B8G8R8A8_SRGB;
}
XrResult flush(XrSwapchain handle, Swapchain& chain) {
    if (!chain.pending) return XR_SUCCESS;
    XrSwapchainImageReleaseInfo info{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    XrResult r=nextxrReleaseSwapchainImage(handle,&info);
    if (XR_SUCCEEDED(r)) chain.pending=false;
    return r;
}
void stopWorker() {
    gRunning.store(false);
    if (worker.joinable()) worker.join();
    clearPublishedCloud();
}
void cleanupSession() {
    stopWorker(); renderer.reset();
    uploadedCloudTime=0; uploadedSurface=false; captureGate.reset();
    if (gViewSpace) nextxrDestroySpace(gViewSpace);
    if (gStageSpace) nextxrDestroySpace(gStageSpace);
    gViewSpace=gStageSpace=XR_NULL_HANDLE;
    gLatestHeadPose={}; activeSession=XR_NULL_HANDLE; alphaSupported=false;
}
XrResult XRAPI_CALL destroyInstance(XrInstance handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (activeSession) cleanupSession();
    chains.clear();
    auto r=nextxrDestroyInstance(handle);
    instance=gXrInstance=XR_NULL_HANDLE; return r;
}
XrResult XRAPI_CALL createSession(XrInstance handle,const XrSessionCreateInfo* info,XrSession* out) {
    std::lock_guard<std::mutex> lock(mutex);
    auto result=nextxrCreateSession(handle,info,out);

    if (XR_FAILED(result) || activeSession || !appDepth || !timespecEnabled) return result;
    const XrGraphicsBindingVulkanKHR* binding=nullptr;
    for (auto b=reinterpret_cast<const XrBaseInStructure*>(info->next);b;b=b->next)
        if (b->type==XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR) binding=reinterpret_cast<const XrGraphicsBindingVulkanKHR*>(b);
    if (!binding) { std::cerr<<"[Occlusion] Only Vulkan sessions are supported; passing through\n"; return result; }
    uint32_t count=0;
    if (XR_FAILED(nextxrEnumerateEnvironmentBlendModes(handle,info->systemId,XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,0,&count,nullptr))) return result;
    std::vector<XrEnvironmentBlendMode> modes(count);
    if (XR_FAILED(nextxrEnumerateEnvironmentBlendModes(handle,info->systemId,XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,count,&count,modes.data()))) return result;
    alphaSupported=std::find(modes.begin(),modes.end(),XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND)!=modes.end();
    if (!alphaSupported) { std::cerr<<"[Occlusion] Runtime has no alpha passthrough; passing through\n"; return result; }
    activeSession=*out;
    XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_STAGE; space.poseInReferenceSpace.orientation.w=1;
    if (XR_FAILED(nextxrCreateReferenceSpace(*out,&space,&gStageSpace))) {
        space.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_LOCAL;
        if (XR_FAILED(nextxrCreateReferenceSpace(*out,&space,&gStageSpace))) { cleanupSession(); return result; }
    }
    space.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_VIEW;
    if (XR_FAILED(nextxrCreateReferenceSpace(*out,&space,&gViewSpace))) { cleanupSession(); return result; }
    renderer=std::make_unique<OcclusionRenderer>();
    const char* dir=std::getenv("FRAME_OCCLUSION_SHADER_DIR");
    gShaderDirectory=dir?dir:"shaders";
    if (!renderer->initialize(binding->physicalDevice,binding->device,binding->queueFamilyIndex,binding->queueIndex,gShaderDirectory)) {
        std::cerr<<"[Occlusion] Vulkan setup failed; passing through\n"; cleanupSession();
    } else std::cout<<"[Occlusion] Vulkan depth masking ready\n";
    return result;
}
void startWorker() {
    if (worker.joinable()) return;
    gOcclusionOnly=true;
    gRunning.store(true);
    worker=std::thread([] {
        try {
        FrameSpatial::DMACapture capture;
        auto pid=FrameSpatial::DMACapture::findCompositorPid();
        if (!pid || !capture.initialize(pid,-1,-1)) {
            std::cerr<<"[Occlusion] Camera attach failed; check ptrace_scope=0\n";
            gRunning.store(false); clearPublishedCloud(); return;
        }
        depthWorker(&capture);
        } catch(const std::exception& e) {
            std::cerr<<"[Occlusion] Worker failed: "<<e.what()<<'\n';
        } catch(...) {
            std::cerr<<"[Occlusion] Worker failed\n";
        }
        gRunning.store(false); clearPublishedCloud();
    });
}
XrResult XRAPI_CALL beginSession(XrSession session,const XrSessionBeginInfo* info) {
    std::lock_guard<std::mutex> lock(mutex);
    if (session==activeSession) { captureGate.reset(); setOcclusionCaptureEnabled(false); }
    return nextxrBeginSession(session,info);
}
XrResult XRAPI_CALL endSession(XrSession session) {
    std::lock_guard<std::mutex> lock(mutex);
    if (session==activeSession) {
        stopWorker();
        for (auto& entry:chains) flush(entry.first,entry.second);
    }
    return nextxrEndSession(session);
}
XrResult XRAPI_CALL destroySession(XrSession session) {
    std::lock_guard<std::mutex> lock(mutex);
    if (session==activeSession) {
        for (auto& entry:chains) flush(entry.first,entry.second);
        cleanupSession(); chains.clear();
    }
    return nextxrDestroySession(session);
}
XrResult XRAPI_CALL createSwapchain(XrSession session,const XrSwapchainCreateInfo* info,XrSwapchain* out) {
    std::lock_guard<std::mutex> lock(mutex);
    auto modified=*info;
    bool eligible=session==activeSession && renderer && info->sampleCount==1 && info->faceCount==1 &&
        !(info->createFlags&(XR_SWAPCHAIN_CREATE_PROTECTED_CONTENT_BIT|XR_SWAPCHAIN_CREATE_STATIC_IMAGE_BIT)) &&
        (isDepth(info->format) || isColor(info->format));
    if (eligible) modified.usageFlags|=isDepth(info->format)?XR_SWAPCHAIN_USAGE_SAMPLED_BIT:XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    auto r=nextxrCreateSwapchain(session,&modified,out);
    if (XR_FAILED(r) && eligible) { eligible=false; r=nextxrCreateSwapchain(session,info,out); }
    if (XR_FAILED(r)) return r;
    if (session!=activeSession) return r;
    Swapchain chain; chain.info=modified; chain.eligible=eligible;
    if (eligible) {
        uint32_t count=0;
        auto e=nextxrEnumerateSwapchainImages(*out,0,&count,nullptr);
        chain.images.resize(count);
        for (auto& image:chain.images) image.type=XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR;
        if (XR_SUCCEEDED(e)) e=nextxrEnumerateSwapchainImages(*out,count,&count,reinterpret_cast<XrSwapchainImageBaseHeader*>(chain.images.data()));
        if (XR_FAILED(e)) chain.eligible=false;
    }
    chains.emplace(*out,std::move(chain)); return r;
}
XrResult XRAPI_CALL destroySwapchain(XrSwapchain handle) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it=chains.find(handle);
    if (it!=chains.end()) {
        auto r=flush(handle,it->second); if (XR_FAILED(r)) return r;
        // Masks finish before deferred release. Drop views/framebuffers before
        // the runtime destroys images or later reuses their Vulkan handles.
        if (renderer) for (const auto& image:it->second.images) renderer->releaseImage(image.image);
        chains.erase(it);
    }
    return nextxrDestroySwapchain(handle);
}
XrResult XRAPI_CALL acquireSwapchain(XrSwapchain handle,const XrSwapchainImageAcquireInfo* info,uint32_t* index) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it=chains.find(handle);
    if (it!=chains.end()) {
        auto r=flush(handle,it->second); if (XR_FAILED(r)) return r;
        // Multiple outstanding images are legal but outside this layer's path.
        if (it->second.waiting) it->second.eligible=false;
    }
    auto r=nextxrAcquireSwapchainImage(handle,info,index);
    if (XR_SUCCEEDED(r) && it!=chains.end()) { it->second.acquired=*index; it->second.waiting=true; it->second.waited=false; }
    return r;
}
XrResult XRAPI_CALL waitSwapchain(XrSwapchain handle,const XrSwapchainImageWaitInfo* info) {
    std::lock_guard<std::mutex> lock(mutex);
    auto r=nextxrWaitSwapchainImage(handle,info);
    auto it=chains.find(handle);
    if (r==XR_SUCCESS && it!=chains.end()) it->second.waited=true;
    return r;
}
XrResult XRAPI_CALL releaseSwapchain(XrSwapchain handle,const XrSwapchainImageReleaseInfo* info) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it=chains.find(handle);
    if (it==chains.end() || !it->second.eligible || !renderer || !info || info->next || !it->second.waited) {
        auto r=nextxrReleaseSwapchainImage(handle,info);
        if (XR_SUCCEEDED(r) && it!=chains.end()) it->second.waiting=false;
        return r;
    }
    auto& chain=it->second;
    if (!chain.waiting || chain.pending) return XR_ERROR_CALL_ORDER_INVALID;
    // Images stay runtime-acquired until xrEndFrame gives us pose/FOV/depth metadata.
    chain.released=chain.acquired; chain.pending=true; chain.waiting=false;
    return XR_SUCCESS;
}
const XrCompositionLayerDepthInfoKHR* depthInfo(const XrCompositionLayerProjectionView& view) {
    for (auto p=reinterpret_cast<const XrBaseInStructure*>(view.next);p;p=p->next)
        if (p->type==XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR) return reinterpret_cast<const XrCompositionLayerDepthInfoKHR*>(p);
    return nullptr;
}
bool validSubImage(const XrSwapchainSubImage& image,const Swapchain& chain) {
    auto& r=image.imageRect;
    return chain.eligible && chain.pending && chain.released<chain.images.size() &&
        image.imageArrayIndex<chain.info.arraySize && r.offset.x>=0 && r.offset.y>=0 && r.extent.width>0 && r.extent.height>0 &&
        uint64_t(r.offset.x)+r.extent.width<=chain.info.width && uint64_t(r.offset.y)+r.extent.height<=chain.info.height;
}
XrResult XRAPI_CALL endFrame(XrSession session,const XrFrameEndInfo* info) {
    std::lock_guard<std::mutex> lock(mutex);
    if (session!=activeSession || !renderer || !info || (info->layerCount && !info->layers)) return nextxrEndFrame(session,info);
    bool captureEnabled=false;
    if (info->environmentBlendMode==XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) {
        for (uint32_t i=0;i<info->layerCount;++i) {
            auto base=info->layers[i];
            if (!base || base->type!=XR_TYPE_COMPOSITION_LAYER_PROJECTION) continue;
            auto layer=reinterpret_cast<const XrCompositionLayerProjection*>(base);
            if (layer->viewCount==2 && layer->views && depthInfo(layer->views[0]) && depthInfo(layer->views[1])) captureEnabled=true;
        }
    }
    timespec now{}; clock_gettime(CLOCK_MONOTONIC,&now);
    const double nowSeconds=now.tv_sec+now.tv_nsec*1e-9;
    const bool keepCapturing = captureGate.update(captureEnabled,
        info->layerCount == 0 && info->environmentBlendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND,
        nowSeconds);
    setOcclusionCaptureEnabled(keepCapturing);
    if (keepCapturing) startWorker();

    timespec display{};
    const bool displayKnown=nextxrConvertTimeToTimespecTimeKHR &&
        XR_SUCCEEDED(nextxrConvertTimeToTimespecTimeKHR(instance,info->displayTime,&display));
    const double displaySeconds=displayKnown?std::max(nowSeconds,display.tv_sec+display.tv_nsec*1e-9):0;
    bool fresh=false;
    bool haveSurface=false;
    double maskCaptureTime=0;
    // Keep the last mesh on the GPU between camera frames. Copying the full
    // triangle list at headset refresh rate adds work without new coverage.
    {
        std::lock_guard<std::mutex> cloudLock(gCloudMutex);
        const double cloudTime=gCloudTime.load();
        fresh=captureEnabled && gRunning.load() && displayKnown &&
            occlusionCloudFresh(cloudTime,gCloudExpiry.load(),nowSeconds,displaySeconds);
        if (!fresh) {
            if (uploadedSurface && displayKnown && cloudTime>0 &&
                nowSeconds>=cloudTime &&
                !occlusionCloudFresh(cloudTime,gCloudExpiry.load(),nowSeconds,displaySeconds)) {
                static uint64_t meshExpiries=0;
                ++meshExpiries;
                const char* profile=std::getenv("FRAME_OCCLUSION_PROFILE");
                if(profile && std::string(profile)=="1")
                    std::cout<<"[OcclusionExpiry] meshes="<<meshExpiries<<" age_ms="
                             <<(displaySeconds-cloudTime)*1000<<std::endl;
            }
            renderer->upload({});
            uploadedCloudTime=0; uploadedSurface=false;
        } else if (uploadedCloudTime!=cloudTime) {
            uploadedSurface=renderer->upload(gCloud) && !gCloud.empty();
            uploadedCloudTime=cloudTime;
        }
        haveSurface=uploadedSurface;
        maskCaptureTime=uploadedCloudTime;
    }
    std::vector<XrCompositionLayerProjection> projections;
    std::vector<const XrCompositionLayerBaseHeader*> layers;
    projections.reserve(info->layerCount); layers.reserve(info->layerCount);
    bool masked=false, renderFailed=false;
    for (uint32_t i=0;i<info->layerCount;++i) {
        auto base=info->layers[i];
        layers.push_back(base);
        if (!base) continue;
        if (!fresh || !haveSurface || base->type!=XR_TYPE_COMPOSITION_LAYER_PROJECTION) continue;
        auto projection=reinterpret_cast<const XrCompositionLayerProjection*>(base);
        if (projection->viewCount!=2 || !projection->views) continue;
        XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
        auto r=nextxrLocateSpace(gStageSpace,projection->space,info->displayTime,&location);
        auto required=XR_SPACE_LOCATION_POSITION_VALID_BIT|XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
        if (XR_FAILED(r) || (location.locationFlags&required)!=required) {
            captureGate.reset(); setOcclusionCaptureEnabled(false); renderer->upload({});
            uploadedCloudTime=0; uploadedSurface=false; fresh=false;
            continue;
        }
        // viewFromPose maps projection-space points into cloud space; inverse
        // gives cloud-space points in the application's projection space.
        XrPosef inverse{};
        inverse.orientation={-location.pose.orientation.x,-location.pose.orientation.y,-location.pose.orientation.z,location.pose.orientation.w};
        const auto& q=inverse.orientation;
        float x=-location.pose.position.x,y=-location.pose.position.y,z=-location.pose.position.z;
        float tx=2*(q.y*z-q.z*y),ty=2*(q.z*x-q.x*z),tz=2*(q.x*y-q.y*x);
        inverse.position={x+q.w*tx+q.y*tz-q.z*ty,y+q.w*ty+q.z*tx-q.x*tz,z+q.w*tz+q.x*ty-q.y*tx};
        Mat4 cloudToLayer=viewFromPose(inverse);
        bool layerMasked=false;
        // Validate both eyes before changing either color image.
        bool valid=true;
        for (uint32_t eye=0;eye<2;++eye) {
            const auto& v=projection->views[eye]; auto d=depthInfo(v);
            if (!d) { valid=false; break; }
            auto c=chains.find(v.subImage.swapchain), dc=chains.find(d->subImage.swapchain);
            if (c==chains.end() || dc==chains.end() || !isColor(c->second.info.format) || !isDepth(dc->second.info.format) ||
                !validSubImage(v.subImage,c->second) || !validSubImage(d->subImage,dc->second) ||
                v.subImage.imageRect.extent.width!=d->subImage.imageRect.extent.width ||
                v.subImage.imageRect.extent.height!=d->subImage.imageRect.extent.height ||
                !std::isfinite(d->minDepth) || !std::isfinite(d->maxDepth) || d->maxDepth<=d->minDepth ||
                !(d->nearZ>0) || !(d->farZ>0) || d->nearZ==d->farZ) { valid=false; break; }
        }
        if (!valid) continue;
        std::vector<OcclusionMaskTarget> targets;
        targets.reserve(2);
        for (uint32_t eye=0;eye<2;++eye) {
            const auto& v=projection->views[eye]; auto d=depthInfo(v);
            auto& c=chains.at(v.subImage.swapchain); auto& dc=chains.at(d->subImage.swapchain);
            Mat4 matrix=multiply(projectionFromFov(v.fov,0.08f,12.0f),multiply(viewFromPose(v.pose),cloudToLayer));
            targets.push_back({c.images[c.released].image,dc.images[dc.released].image,
                VkFormat(c.info.format),VkFormat(dc.info.format),
                v.subImage.imageArrayIndex,d->subImage.imageArrayIndex,
                v.subImage.imageRect,d->subImage.imageRect,matrix,*d,
                float(displaySeconds-maskCaptureTime)});
        }
        // The usual stereo pair shares a render-pass format: one queue submit,
        // one wait, with separate immutable depth descriptors for both eyes.
        // Preserve support for unusual heterogeneous-format stereo swapchains.
        if (targets[0].colorFormat==targets[1].colorFormat) {
            layerMasked=renderer->maskBatch(targets);
            if (layerMasked) profileMask(*renderer);
        } else {
            layerMasked=true;
            for (const auto& target:targets) {
                if (!renderer->maskBatch({target})) { layerMasked=false; break; }
                profileMask(*renderer);
            }
        }
        if (!layerMasked) {
            std::cerr<<"[Occlusion] Mask rendering failed\n";
            fresh=false; renderFailed=true;
        }
        if (layerMasked) {
            projections.push_back(*projection);
            projections.back().layerFlags|=XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            layers.back()=reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projections.back());
            masked=true;
        }
    }
    for (auto& entry:chains) { auto r=flush(entry.first,entry.second); if (XR_FAILED(r)) return r; }
    if (renderFailed) { clearPublishedCloud(); renderer.reset(); }
    auto modified=*info;
    modified.layers=layers.empty()?nullptr:layers.data();
    // Preserve the application's chosen environment blend mode.
    if (masked && !warned) { warned=true; std::cout<<"[Occlusion] Masked stereo projection using the current depth surface\n"; }
    return nextxrEndFrame(session,&modified);
}
XrResult XRAPI_CALL getProc(XrInstance handle,const char* name,PFN_xrVoidFunction* out) {
    if (!name || !out) return XR_ERROR_VALIDATION_FAILURE;
#define HOOK(NAME,FUNC) if (!std::strcmp(name,#NAME)) { *out=reinterpret_cast<PFN_xrVoidFunction>(FUNC); return XR_SUCCESS; }
    HOOK(xrGetInstanceProcAddr,getProc)
    HOOK(xrDestroyInstance,destroyInstance) HOOK(xrCreateSession,createSession)
    HOOK(xrDestroySession,destroySession) HOOK(xrBeginSession,beginSession) HOOK(xrEndSession,endSession)
    HOOK(xrCreateSwapchain,createSwapchain) HOOK(xrDestroySwapchain,destroySwapchain)
    HOOK(xrAcquireSwapchainImage,acquireSwapchain) HOOK(xrWaitSwapchainImage,waitSwapchain) HOOK(xrReleaseSwapchainImage,releaseSwapchain)
    HOOK(xrEndFrame,endFrame)
#undef HOOK
    return nextGipa?nextGipa(handle,name,out):XR_ERROR_FUNCTION_UNSUPPORTED;
}
XrResult XRAPI_CALL createLayer(const XrInstanceCreateInfo* info,const XrApiLayerCreateInfo* layer,XrInstance* out) {
    if (!layer || !layer->nextInfo) return XR_ERROR_INITIALIZATION_FAILED;
    nextGipa=layer->nextInfo->nextGetInstanceProcAddr;
    std::vector<const char*> extensions;
    for (uint32_t i=0;i<info->enabledExtensionCount;++i) extensions.push_back(info->enabledExtensionNames[i]);
    appDepth=std::any_of(extensions.begin(),extensions.end(),[](const char* s){return !std::strcmp(s,XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);});
    PFN_xrEnumerateInstanceExtensionProperties enumerate=nullptr;
    PFN_xrVoidFunction function=nullptr;
    if (XR_SUCCEEDED(nextGipa(XR_NULL_HANDLE,"xrEnumerateInstanceExtensionProperties",&function))) enumerate=reinterpret_cast<PFN_xrEnumerateInstanceExtensionProperties>(function);
    uint32_t count=0; timespecEnabled=false;
    if (enumerate && XR_SUCCEEDED(enumerate(nullptr,0,&count,nullptr))) {
        std::vector<XrExtensionProperties> available(count,{XR_TYPE_EXTENSION_PROPERTIES});
        if (XR_SUCCEEDED(enumerate(nullptr,count,&count,available.data())))
            for (auto& e:available) if (!std::strcmp(e.extensionName,XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME)) timespecEnabled=true;
    }
    if (timespecEnabled && std::none_of(extensions.begin(),extensions.end(),[](const char* s){return !std::strcmp(s,XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);}))
        extensions.push_back(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    auto modified=*info; modified.enabledExtensionCount=extensions.size(); modified.enabledExtensionNames=extensions.data();
    auto next=*layer; next.nextInfo=layer->nextInfo->next;
    auto r=layer->nextInfo->nextCreateApiLayerInstance(&modified,&next,out);
    if (XR_FAILED(r)) return r;
    instance=gXrInstance=*out;
#define LOAD(F) nextGipa(*out,#F,reinterpret_cast<PFN_xrVoidFunction*>(&next##F))
    LOAD(xrDestroyInstance); LOAD(xrCreateSession); LOAD(xrDestroySession); LOAD(xrBeginSession); LOAD(xrEndSession);
    LOAD(xrCreateReferenceSpace); LOAD(xrDestroySpace); LOAD(xrLocateSpace); LOAD(xrCreateSwapchain); LOAD(xrDestroySwapchain);
    LOAD(xrEnumerateSwapchainImages); LOAD(xrAcquireSwapchainImage); LOAD(xrWaitSwapchainImage); LOAD(xrReleaseSwapchainImage);
    LOAD(xrEndFrame); LOAD(xrEnumerateEnvironmentBlendModes);
#undef LOAD
    gLocateSpace=nextxrLocateSpace;
    if (timespecEnabled) {
        nextGipa(*out,"xrConvertTimespecTimeToTimeKHR",reinterpret_cast<PFN_xrVoidFunction*>(&gConvertTimespecTime));
        nextGipa(*out,"xrConvertTimeToTimespecTimeKHR",reinterpret_cast<PFN_xrVoidFunction*>(&nextxrConvertTimeToTimespecTimeKHR));
    }
    std::cout<<"[Occlusion] API layer loaded; application depth="<<(appDepth?"enabled":"absent (passing through)")<<'\n';
    return r;
}
} // namespace
extern "C" __attribute__((visibility("default"))) XrResult XRAPI_CALL
xrNegotiateLoaderApiLayerInterface(const XrNegotiateLoaderInfo* loader,const char* name,XrNegotiateApiLayerRequest* request) {
    if (!loader || !request || !name || std::strcmp(name,kName) ||
        loader->structType!=XR_LOADER_INTERFACE_STRUCT_LOADER_INFO || loader->structVersion!=XR_LOADER_INFO_STRUCT_VERSION ||
        loader->structSize!=sizeof(XrNegotiateLoaderInfo) || request->structType!=XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST ||
        request->structVersion!=XR_API_LAYER_INFO_STRUCT_VERSION || request->structSize!=sizeof(XrNegotiateApiLayerRequest) ||
        loader->minInterfaceVersion>1 || loader->maxInterfaceVersion<1 || loader->maxApiVersion<XR_MAKE_VERSION(1,0,0)) return XR_ERROR_INITIALIZATION_FAILED;
    request->layerInterfaceVersion=1; request->layerApiVersion=XR_MAKE_VERSION(1,0,0);
    request->getInstanceProcAddr=getProc; request->createApiLayerInstance=createLayer;
    return XR_SUCCESS;
}
