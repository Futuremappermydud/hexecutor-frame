// Exercise the frame contract with mocked downstream OpenXR calls. No cameras
// or headset session are opened by this test.
#include "../src/openxr_occlusion_layer.cpp"
#include "calibration.hpp"
#include "depth_pipeline.hpp"
#include <cassert>
#include <stdexcept>
namespace {
XrEnvironmentBlendMode expectedBlend;
const XrCompositionLayerBaseHeader* expectedLayer;
int releases=0, frames=0;
XrResult XRAPI_CALL recordRelease(XrSwapchain,const XrSwapchainImageReleaseInfo*) { ++releases; return XR_SUCCESS; }
XrResult XRAPI_CALL recordEnd(XrSession,const XrFrameEndInfo* frame) {
    if (frame->environmentBlendMode!=expectedBlend || frame->layerCount!=1 || frame->layers[0]!=expectedLayer)
        throw std::runtime_error("Layer altered an ineligible frame");
    ++frames; return XR_SUCCESS;
}
void testFrame(XrEnvironmentBlendMode mode,double cloudTime) {
    gCloudTime.store(cloudTime);
    XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    // Intentionally no views: opaque or stale frames must never inspect them.
    projection.viewCount=2;
    expectedLayer=reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection); expectedBlend=mode;
    XrFrameEndInfo frame{XR_TYPE_FRAME_END_INFO}; frame.environmentBlendMode=mode;
    frame.layerCount=1; frame.layers=&expectedLayer;
    if (endFrame(activeSession,&frame)!=XR_SUCCESS) throw std::runtime_error("Frame forwarding failed");
}
}
int main() {
    try {
        assert(occlusionCloudFresh(10,10.2,10.1,10.199));
        assert(!occlusionCloudFresh(10,10.2,10.1,10.2));
        assert(!occlusionCloudFresh(10,10.2,10.21,10.21));
        assert(!occlusionCloudFresh(10,11,10.4,10.5));
        assert(!occlusionCloudFresh(10,10.2,9.9,10.1));
        assert(!occlusionCloudFresh(10,0,10.1,10.1));
        assert(!occlusionCloudFresh(10,10.2,10.1,std::numeric_limits<double>::quiet_NaN()));
        // The temporal transform must match calibrated point backprojection
        // through a translated/rotated head pose, including the optical offset.
        FrameSpatial::DepthPipeline rectifier;
        rectifier.initialize();
        XrPosef previous{{0,0,0,1},{0,0,0}};
        XrPosef currentPose{{0,std::sin(0.1f),0,std::cos(0.1f)},{0.08f,-0.03f,0.04f}};
        const auto alignment=occlusionReprojection(rectifier,previous,currentPose);
        const auto inHead=rectifier.rectifiedPointInHead(360,280,1.5f);
        const float x=inHead[0]-currentPose.position.x;
        const float y=inHead[1]-currentPose.position.y;
        const float z=inHead[2]-currentPose.position.z;
        const auto expected=rectifier.headPointInRectified({
            std::cos(0.2f)*x-std::sin(0.2f)*z,y,
            std::sin(0.2f)*x+std::cos(0.2f)*z});
        const std::array<float,3> point{
            (360-FrameSpatial::RECT_CX)*1.5f/FrameSpatial::RECT_FOCAL,
            (280-FrameSpatial::RECT_CY)*1.5f/FrameSpatial::RECT_FOCAL,1.5f};
        for(int row=0;row<3;++row) {
            float result=alignment.translation[row];
            for(int col=0;col<3;++col) result+=alignment.rotation[row][col]*point[col];
            assert(std::abs(result-expected[row])<1e-5f);
        }
        XrNegotiateLoaderInfo loader{};
        loader.structType=XR_LOADER_INTERFACE_STRUCT_LOADER_INFO; loader.structVersion=XR_LOADER_INFO_STRUCT_VERSION;
        loader.structSize=sizeof(loader); loader.minInterfaceVersion=loader.maxInterfaceVersion=1;
        loader.minApiVersion=loader.maxApiVersion=XR_MAKE_VERSION(1,0,0);
        XrNegotiateApiLayerRequest request{};
        request.structType=XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST; request.structVersion=XR_API_LAYER_INFO_STRUCT_VERSION; request.structSize=sizeof(request);
        if (xrNegotiateLoaderApiLayerInterface(&loader,kName,&request)!=XR_SUCCESS || !request.getInstanceProcAddr || !request.createApiLayerInstance)
            throw std::runtime_error("Loader negotiation failed");
        loader.maxInterfaceVersion=0;
        if (xrNegotiateLoaderApiLayerInterface(&loader,kName,&request)!=XR_ERROR_INITIALIZATION_FAILED)
            throw std::runtime_error("Invalid loader accepted");
        activeSession=reinterpret_cast<XrSession>(uintptr_t(1));
        renderer=std::make_unique<OcclusionRenderer>();
        nextxrEndFrame=recordEnd; nextxrReleaseSwapchainImage=recordRelease;
        timespec now{}; clock_gettime(CLOCK_MONOTONIC,&now); double current=now.tv_sec+now.tv_nsec*1e-9;
        { std::lock_guard<std::mutex> lock(gCloudMutex); gCloud={{0,0,-1,0xffffffff}}; }
        Swapchain chain; chain.pending=true;
        chains.emplace(reinterpret_cast<XrSwapchain>(uintptr_t(2)),chain);
        testFrame(XR_ENVIRONMENT_BLEND_MODE_OPAQUE,current);
        testFrame(XR_ENVIRONMENT_BLEND_MODE_ADDITIVE,current);
        testFrame(XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND,current-1);
        testFrame(XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND,0);
        if (frames!=4 || releases!=1) throw std::runtime_error("Deferred release forwarding failed");
        renderer.reset(); activeSession=XR_NULL_HANDLE;
        std::cout<<"Layer contract tests passed: calibrated capture-pose reprojection, loader ABI, opaque/additive passthrough gating, stale/absent cloud, predicted display expiry, deferred releases\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
