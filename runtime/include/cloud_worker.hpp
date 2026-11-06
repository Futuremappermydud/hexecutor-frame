#pragma once
#ifndef XR_USE_TIMESPEC
#define XR_USE_TIMESPEC
#endif
#include <time.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include "dma_capture.hpp"
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "occlusion_surface.hpp"

namespace FrameSpatial { class DepthPipeline; }

namespace FrameSpatialCloud {
struct PointVertex {
    float x;
    float y;
    float z;
    uint32_t rgba;
    // Seconds after this mesh's capture-end timestamp; constant across each cell.
    float supportLifetime = 0;
};

struct TimedHeadPose {
    XrPosef pose{{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
    XrTime time = 0;
    bool valid = false;
};

// Optical-frame history alignment includes the calibrated camera offset.
FrameSpatial::OcclusionReprojection occlusionReprojection(
    const FrameSpatial::DepthPipeline& rectifier, const XrPosef& previous,
    const XrPosef& current);


extern std::atomic<bool> gRunning, gCaptureEnabled;
// 0 = PointCloud, 1 = LiveMesh, 2 = PersistentMesh, 3 = WireframeMesh
extern std::atomic<int> gMeshViewerMode;
// Set before launching the worker. The API layer publishes capture-aligned
// confirmed surface triangles; the viewer retains its world voxel fusion.
extern bool gOcclusionOnly;
extern std::mutex gPoseMutex;
extern TimedHeadPose gLatestHeadPose;
extern XrInstance gXrInstance;
extern XrSpace gStageSpace, gViewSpace;
extern PFN_xrConvertTimespecTimeToTimeKHR gConvertTimespecTime;
extern PFN_xrLocateSpace gLocateSpace;
extern std::mutex gCloudMutex;
extern std::vector<PointVertex> gCloud;
extern std::atomic<uint32_t> gDepthFrames, gValidDepth, gStablePoints;
extern std::atomic<float> gStereoMs;
extern std::atomic<double> gCloudTime;
// Latest cell CLOCK_MONOTONIC deadline, published under gCloudMutex with the mesh.
extern std::atomic<double> gCloudExpiry;
void clearPublishedCloud();
extern std::atomic<uint64_t> gCaptureGeneration;
void setOcclusionCaptureEnabled(bool enabled);
bool resetOcclusionHistoryOnCaptureChange(FrameSpatial::OcclusionSurfaceHistory& history,
                                          uint64_t& generation);
// Consumes surface on success; rejects computations begun before capture disable.
bool publishOcclusionCloud(std::vector<PointVertex>& surface, double captureTime,
                           double expiry, uint64_t generation, bool captureAligned = true);
// Evaluate at predicted display time, retaining capture freshness as a guard.
bool occlusionCloudFresh(double captureTime, double expiry, double now, double display);
extern std::string gShaderDirectory;
void depthWorker(FrameSpatial::DMACapture* capture);
} // namespace FrameSpatialCloud
