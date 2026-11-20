// Reuse the GPU fixture and readback assertions without running its main.
#define main rendererSmokeMain
#include "test_occlusion_renderer.cpp"
#undef main
#include "../src/openxr_occlusion_layer.cpp"
#include <cassert>
namespace {
double predictedDisplay;
bool expectedMasked;
bool trackingValid=true;
int submitted=0;
XrResult XRAPI_CALL convertDisplay(XrInstance,XrTime,timespec* value) {
    value->tv_sec=static_cast<time_t>(predictedDisplay);
    value->tv_nsec=static_cast<long>((predictedDisplay-value->tv_sec)*1e9);
    return XR_SUCCESS;
}
XrResult XRAPI_CALL locateIdentity(XrSpace,XrSpace,XrTime,XrSpaceLocation* location) {
    location->locationFlags=trackingValid?(XR_SPACE_LOCATION_POSITION_VALID_BIT|XR_SPACE_LOCATION_ORIENTATION_VALID_BIT):0;
    location->pose={{0,0,0,1},{0,0,0}};
    return XR_SUCCESS;
}
XrResult XRAPI_CALL releaseImage(XrSwapchain,const XrSwapchainImageReleaseInfo*) { return XR_SUCCESS; }
XrResult XRAPI_CALL submitFrame(XrSession,const XrFrameEndInfo* frame) {
    const bool flagged=(frame->layers[0]->layerFlags&XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT)!=0;
    assert(flagged==expectedMasked); ++submitted; return XR_SUCCESS;
}
}
int main() {
    TestGpu gpu;
    activeSession=reinterpret_cast<XrSession>(uintptr_t(1));
    gStageSpace=reinterpret_cast<XrSpace>(uintptr_t(2));
    // A joinable idle thread models a worker stalled after its last publication.
    worker=std::thread([]{}); gRunning.store(true);
    renderer=std::make_unique<OcclusionRenderer>();
    assert(renderer->initialize(gpu.physical,gpu.device,gpu.family,0,"shaders"));
    nextxrConvertTimeToTimespecTimeKHR=convertDisplay;
    nextxrLocateSpace=locateIdentity; nextxrReleaseSwapchainImage=releaseImage; nextxrEndFrame=submitFrame;
    auto color=reinterpret_cast<XrSwapchain>(uintptr_t(3));
    auto depth=reinterpret_cast<XrSwapchain>(uintptr_t(4));
    Swapchain c,d;
    c.eligible=d.eligible=true;
    c.info.format=VK_FORMAT_R8G8B8A8_UNORM; d.info.format=VK_FORMAT_D32_SFLOAT;
    c.info.width=d.info.width=c.info.height=d.info.height=TestGpu::size;
    c.info.arraySize=d.info.arraySize=1;
    c.images.resize(1); d.images.resize(1); c.images[0].image=gpu.color; d.images[0].image=gpu.depth;
    chains[color]=c; chains[depth]=d;
    XrCompositionLayerDepthInfoKHR range{XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR};
    range.subImage={depth,{{0,0},{65,65}},0}; range.minDepth=0; range.maxDepth=1; range.nearZ=.1f; range.farZ=10;
    XrCompositionLayerProjectionView views[2]{};
    for(auto& view:views) {
        view.type=XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW; view.next=&range;
        view.pose.orientation.w=1; view.fov={-.785398f,.785398f,.785398f,-.785398f};
        view.subImage={color,{{0,0},{65,65}},0};
    }
    XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    projection.viewCount=2; projection.views=views;
    const auto* base=reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection);
    XrFrameEndInfo frame{XR_TYPE_FRAME_END_INFO}; frame.layerCount=1; frame.layers=&base;
    frame.environmentBlendMode=XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
    timespec now{}; clock_gettime(CLOCK_MONOTONIC,&now);
    const double current=now.tv_sec+now.tv_nsec*1e-9;
    const double capture=current-.1, expiry=current+.2;
    {
        std::lock_guard<std::mutex> lock(gCloudMutex);
        gCloud={{-.25f,-.25f,-1,0xffffffff},{.25f,-.25f,-1,0xffffffff},{.25f,.25f,-1,0xffffffff},
                {-.25f,-.25f,-1,0xffffffff},{.25f,.25f,-1,0xffffffff},{-.25f,.25f,-1,0xffffffff}};
        for(auto& vertex:gCloud) vertex.supportLifetime=float(expiry-capture);
        gCloudTime.store(capture); gCloudExpiry.store(expiry);
    }
    auto run=[&](double display,bool masked) {
        predictedDisplay=display; expectedMasked=masked;
        gpu.clear((10-.1f*10/2)/(10-.1f));
        chains[color].pending=chains[depth].pending=true;
        assert(endFrame(activeSession,&frame)==XR_SUCCESS);
        gpu.verify(masked);
    };
    run(current+.01,true);
    run(expiry+.001,false); // Predicted display expires while CPU wall time is younger.
    run(expiry+.02,false);  // Repeated rendering cannot revive the GPU mesh.
    assert(gCloudTime.load()==capture && gCloudExpiry.load()==expiry);
    // Missing display conversion fails closed without inventing a clock mapping.
    nextxrConvertTimeToTimespecTimeKHR=nullptr;
    run(current+.025,false);
    nextxrConvertTimeToTimespecTimeKHR=convertDisplay;
    // Restart with a new publication. Capture disable clears it immediately.
    for(auto& vertex:gCloud) vertex.supportLifetime=.4f;
    gCloudTime.store(current); gCloudExpiry.store(current+.4);
    run(current+.03,true);
    // Two independently supported cells must expire separately; publication age
    // exceeding the older deadline must retain the freshly supported region.
    timespec mixedNow{}; clock_gettime(CLOCK_MONOTONIC,&mixedNow);
    const double mixedCapture=mixedNow.tv_sec+mixedNow.tv_nsec*1e-9;
    {
        std::lock_guard<std::mutex> lock(gCloudMutex);
        gCloud.clear();
        const auto cell=[&](float left,float right,float lifetime) {
            gCloud.insert(gCloud.end(),{{left,-.25f,-1,0xffffffff,lifetime},{right,-.25f,-1,0xffffffff,lifetime},
                {right,.25f,-1,0xffffffff,lifetime},{left,-.25f,-1,0xffffffff,lifetime},
                {right,.25f,-1,0xffffffff,lifetime},{left,.25f,-1,0xffffffff,lifetime}});
        };
        cell(-.5f,0,.1f); cell(0,.5f,.4f);
        gCloudTime.store(mixedCapture); gCloudExpiry.store(mixedCapture+.4);
    }
    run(mixedCapture+.01,true); gpu.verify(true,24); gpu.verify(true,40);
    run(mixedCapture+.12,true); gpu.verify(false,24); gpu.verify(true,40);
    run(mixedCapture+.2,true); gpu.verify(false,24); gpu.verify(true,40);
    // Invalid display tracking clears both GPU state and publication.
    trackingValid=false; run(mixedCapture+.25,false);
    assert(gCloud.empty() && gCloudExpiry.load()==0);
    trackingValid=true;
    {
        std::lock_guard<std::mutex> lock(gCloudMutex);
        gCloud={{-.25f,-.25f,-1,0xffffffff},{.25f,-.25f,-1,0xffffffff},{.25f,.25f,-1,0xffffffff}};
        for(auto& vertex:gCloud) vertex.supportLifetime=.4f;
        gCloudTime.store(current); gCloudExpiry.store(current+.4);
    }
    frame.environmentBlendMode=XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    run(current+.04,false);
    assert(gCloud.empty() && gCloudTime.load()==0 && gCloudExpiry.load()==0);
    assert(submitted==10);
    worker.join(); renderer.reset(); chains.clear(); activeSession=XR_NULL_HANDLE;
    std::cout<<"Expiry GPU tests passed: stalled worker readback, per-cell mixed-age display deadlines, repeated rendering, restart, missing time conversion, invalid tracking, disable clearing\n";
}
