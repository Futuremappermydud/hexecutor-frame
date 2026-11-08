// CPU-only synthetic downstream benchmark. Never opens a camera or NPU session.
// Compile with FRAME_PROCESSING_BASELINE against a saved pre-change include/src
// directory to compare the same workload and checksums with the old implementation.
#include "meshing_pipeline.hpp"
#include "model_depth_geometry.hpp"
#include "tsdf_volume.hpp"
#include "depth_speckle_filter.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace FrameSpatial;
using Clock = std::chrono::steady_clock;
namespace { size_t projections = 0; }
// Test fixture: calibrated pinhole with a camera offset, no hardware initialization.
namespace FrameSpatial {
DepthPipeline::DepthPipeline() : m_initialized(false) {}
DepthPipeline::~DepthPipeline() = default;
std::array<float,3> DepthPipeline::rectifiedPointInHead(int x, int y, float z) const {
    ++projections;
    return {(x - 308.f) * z / 380.f + .05f, -(y - 308.f) * z / 380.f, -z};
}
}
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
static void report(const char* name, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::printf("%s p50_ms=%.4f p95_ms=%.4f\n", name, values[values.size()/2], values[(values.size()-1)*95/100]);
}
static void hashFloat(uint64_t& hash, float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ull;
}
int main(int argc, char** argv) {
    const int count = argc > 1 ? std::max(3, std::atoi(argv[1])) : 30;
    constexpr int n = 616;
    const float step = .29f * .01745329252f, baseline = .075f;
    auto map = modelSphericalMap(step, baseline, n, n, 380, 308, 308);
    std::vector<float> disparity(MODEL_WIDTH * MODEL_HEIGHT);
    std::vector<uint8_t> uncertainty(disparity.size(), 30), luma(n*n, 180);
    for (int y=0;y<MODEL_HEIGHT;++y) for(int x=0;x<MODEL_WIDTH;++x) {
        const float a = (x - (MODEL_WIDTH-1)*.5f)*step;
        const float rho = 2.5f/std::cos((y-(MODEL_HEIGHT-1)*.5f)*step);
        disparity[size_t(y)*MODEL_WIDTH+x] = (a-std::atan(std::tan(a)-baseline/rho))/step;
    }
    // Depth steps and invalid patches exercise validity and meshing gates.
    for(int y=60;y<120;++y) for(int x=80;x<150;++x)
        disparity[size_t(y)*MODEL_WIDTH+x] *= 2.f;
    for(int y=220;y<240;++y) for(int x=120;x<140;++x)
        uncertainty[size_t(y)*MODEL_WIDTH+x] = 100;
    std::vector<double> postTimes, speckleTimes, filterTimes, liveTimes, integrateTimes, meshTimes;
    std::vector<float> depth, filtered;
    const std::vector<float>* lastFiltered = nullptr;
    std::vector<FrameSpatialCloud::PointVertex> live;
    std::vector<TsdfVertex> mesh;
#ifndef FRAME_PROCESSING_BASELINE
    DepthFilterWorkspace workspace;
    DepthSpeckleWorkspace speckleWorkspace;
#endif
    DepthPipeline rectifier;
    const XrPosef head{{0,.2588190451f,0,.9659258263f},{.2f,.1f,-.3f}};
    TsdfVolume volume;
    for(int frame=-3; frame<count; ++frame) {
        auto begin = Clock::now();
        depth = modelSphericalDepth(disparity, uncertainty, map);
        const auto postMs = elapsed(begin);
        begin = Clock::now();
#ifdef FRAME_PROCESSING_BASELINE
        removeDepthSpeckles(depth,n,n);
#else
        removeDepthSpeckles(depth,n,n,speckleWorkspace);
#endif
        const auto speckleMs = elapsed(begin);
        begin = Clock::now();
#ifdef FRAME_PROCESSING_BASELINE
        filtered = powerMeanFilterDepth(depth,n,n);
        const auto& smoothed = filtered;
#else
        const auto& smoothed = powerMeanFilterDepth(depth,n,n,workspace);
#endif
        const auto filterMs = elapsed(begin);
        lastFiltered = &smoothed;
        begin = Clock::now();
        projections = 0;
#ifdef FRAME_PROCESSING_BASELINE
        live = triangulateDepthMap(smoothed,luma,rectifier,head,n,n);
#else
        triangulateDepthMap(smoothed,luma,rectifier,head,n,n,live);
#endif
        const auto liveMs = elapsed(begin);
        begin = Clock::now();
        volume.integrate(depth.data(),luma.data(),n,n,380,308,308,TsdfPose{});
        const auto integrateMs = elapsed(begin);
        begin = Clock::now();
        const float focus[3]{0,0,0};
        volume.extractMesh(mesh,480000,focus);
        const auto meshMs = elapsed(begin);
        if(frame>=0) {
            postTimes.push_back(postMs); speckleTimes.push_back(speckleMs);
            filterTimes.push_back(filterMs); liveTimes.push_back(liveMs);
            integrateTimes.push_back(integrateMs); meshTimes.push_back(meshMs);
        }
    }
    uint64_t depthHash=1469598103934665603ull, filterHash=depthHash, liveHash=depthHash, meshHash=depthHash;
    for(float z:depth) hashFloat(depthHash,z);
    for(float z:*lastFiltered) hashFloat(filterHash,z);
    for(const auto& v:live) {
        hashFloat(liveHash,v.x);hashFloat(liveHash,v.y);hashFloat(liveHash,v.z);
        liveHash=(liveHash^v.rgba)*1099511628211ull;
    }
    for(const auto& v:mesh) {
        hashFloat(meshHash,v.x);hashFloat(meshHash,v.y);hashFloat(meshHash,v.z);hashFloat(meshHash,v.luma);
    }
    report("postprocess",postTimes);report("speckles",speckleTimes);
    report("filter",filterTimes);report("live_mesh",liveTimes);
    report("tsdf_integrate",integrateTimes);report("tsdf_mesh",meshTimes);
    std::printf("depth_hash=%llx filter_hash=%llx live_hash=%llx tsdf_hash=%llx\n",(unsigned long long)depthHash,(unsigned long long)filterHash,(unsigned long long)liveHash,(unsigned long long)meshHash);
    std::printf("live_triangles=%zu tsdf_triangles=%zu leaves=%zu projections=%zu\n",live.size()/3,mesh.size()/3,volume.leafCount(),projections);
}
