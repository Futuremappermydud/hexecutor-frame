#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h>
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "calibration.hpp"
#include "depth_pipeline.hpp"
#include "persistent_depth.hpp"
#include "tsdf_volume.hpp"
#include "depth_speckle_filter.hpp"
#include "occlusion_surface.hpp"
#include "occlusion_completion.hpp"
#include "occlusion_benchmark.hpp"
#include <memory>
#include "cloud_visibility.hpp"
#include "dma_capture.hpp"
#include "vulkan_stereo_pipeline.hpp"
#include "vulkan_guarded_stereo_pipeline.hpp"
#include "model_depth_pipeline.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cloud_worker.hpp"
#include "meshing_pipeline.hpp"

namespace FrameSpatialCloud {
std::atomic<bool> gRunning{true};
std::atomic<bool> gCaptureEnabled{true};
std::atomic<int> gMeshViewerMode{0};
bool gOcclusionOnly = false;
PFN_xrLocateSpace gLocateSpace = nullptr;
std::atomic<double> gCloudTime{0};
std::atomic<double> gCloudExpiry{0};
std::atomic<double> gOcclusionMaxAge{0.5};
std::atomic<uint64_t> gCaptureGeneration{0};
std::string gShaderDirectory = "shaders";
std::mutex gPoseMutex;
TimedHeadPose gLatestHeadPose;
XrInstance gXrInstance = XR_NULL_HANDLE;
XrSpace gStageSpace = XR_NULL_HANDLE;
XrSpace gViewSpace = XR_NULL_HANDLE;
PFN_xrConvertTimespecTimeToTimeKHR gConvertTimespecTime = nullptr;

std::mutex gCloudMutex;
std::vector<PointVertex> gCloud;
std::atomic<uint32_t> gDepthFrames{0};
std::atomic<uint32_t> gValidDepth{0};
std::atomic<uint32_t> gStablePoints{0};
std::atomic<float> gStereoMs{0.0f};

void clearPublishedCloud() {
    std::lock_guard<std::mutex> lock(gCloudMutex);
    gCloud.clear(); gCloudTime.store(0); gCloudExpiry.store(0);
}

void setOcclusionCaptureEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(gCloudMutex);
    const bool previous=gCaptureEnabled.exchange(enabled);
    if(!enabled) {
        if(previous) gCaptureGeneration.fetch_add(1);
        gCloud.clear(); gCloudTime.store(0); gCloudExpiry.store(0);
    }
}

bool resetOcclusionHistoryOnCaptureChange(FrameSpatial::OcclusionSurfaceHistory& history,
                                          uint64_t& generation) {
    const auto current=gCaptureGeneration.load();
    if(current==generation) return false;
    history.reset(); generation=current; return true;
}

bool publishOcclusionCloud(std::vector<PointVertex>& surface,double captureTime,
                           double expiry,uint64_t generation,bool captureAligned) {
    std::lock_guard<std::mutex> lock(gCloudMutex);
    if(!captureAligned || !gCaptureEnabled.load() || !gRunning.load() || generation!=gCaptureGeneration.load()) {
        gCloud.clear(); gCloudTime.store(0); gCloudExpiry.store(0); return false;
    }
    gCloud.swap(surface); gCloudTime.store(captureTime);
    gCloudExpiry.store(gCloud.empty()?0:expiry);
    return true;
}

bool occlusionCloudFresh(double captureTime, double expiry, double now, double display) {
    return std::isfinite(captureTime) && std::isfinite(expiry) &&
        std::isfinite(now) && std::isfinite(display) && captureTime>0 &&
        expiry>captureTime && now>=captureTime && display>=now &&
        display-captureTime<gOcclusionMaxAge.load() && display<expiry;
}

struct VoxelKey {
    int x;
    int y;
    int z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    size_t operator()(const VoxelKey& key) const {
        uint64_t value = static_cast<uint32_t>(key.x) * 0x9e3779b1u;
        value ^= static_cast<uint32_t>(key.y) * 0x85ebca77u +
                 (value << 6) + (value >> 2);
        value ^= static_cast<uint32_t>(key.z) * 0xc2b2ae3du +
                 (value << 6) + (value >> 2);
        return static_cast<size_t>(value);
    }
};

struct Voxel {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float luminance = 0.0f;
    uint32_t observations = 0;
    uint32_t lastFrame = 0;
    float anchorX = 0.0f;
    float anchorY = 0.0f;
    float anchorZ = 0.0f;
    float anchorQx = 0.0f;
    float anchorQy = 0.0f;
    float anchorQz = 0.0f;
    float anchorQw = 1.0f;
    FrameSpatial::CloudAging aging;
};

std::array<float, 3> rotateVector(const XrQuaternionf& q,
                                  const std::array<float, 3>& v) {
    const float tx = 2.0f * (q.y * v[2] - q.z * v[1]);
    const float ty = 2.0f * (q.z * v[0] - q.x * v[2]);
    const float tz = 2.0f * (q.x * v[1] - q.y * v[0]);
    return {
        v[0] + q.w * tx + (q.y * tz - q.z * ty),
        v[1] + q.w * ty + (q.z * tx - q.x * tz),
        v[2] + q.w * tz + (q.x * ty - q.y * tx),
    };
}

FrameSpatial::OcclusionReprojection occlusionReprojection(
    const FrameSpatial::DepthPipeline& rectifier, const XrPosef& previous,
    const XrPosef& current) {
    const XrQuaternionf inverse{-current.orientation.x, -current.orientation.y,
                               -current.orientation.z, current.orientation.w};
    const auto map = [&](const std::array<float, 3>& inHead) {
        auto world = rotateVector(previous.orientation, inHead);
        world[0] += previous.position.x - current.position.x;
        world[1] += previous.position.y - current.position.y;
        world[2] += previous.position.z - current.position.z;
        return rectifier.headPointInRectified(rotateVector(inverse, world));
    };
    const int cx = int(FrameSpatial::RECT_CX), cy = int(FrameSpatial::RECT_CY);
    const int focal = int(FrameSpatial::RECT_FOCAL);
    FrameSpatial::OcclusionReprojection transform;
    transform.translation = map(rectifier.rectifiedPointInHead(cx, cy, 0));
    const auto forward = map(rectifier.rectifiedPointInHead(cx, cy, 1));
    const auto right = map(rectifier.rectifiedPointInHead(cx + focal, cy, 1));
    const auto down = map(rectifier.rectifiedPointInHead(cx, cy + focal, 1));
    for (int row = 0; row < 3; ++row) {
        transform.rotation[row][0] = right[row] - forward[row];
        transform.rotation[row][1] = down[row] - forward[row];
        transform.rotation[row][2] = forward[row] - transform.translation[row];
    }
    return transform;
}

uint32_t packGray(float luminance, uint32_t observations) {
    const float confidence = std::min(1.0f, observations / 8.0f);
    const int gray = std::clamp(static_cast<int>(luminance), 35, 255);
    const int red = std::clamp(static_cast<int>(gray * (0.55f + 0.30f * confidence)), 0, 255);
    const int green = std::clamp(static_cast<int>(gray * (0.80f + 0.20f * confidence)), 0, 255);
    const int blue = std::clamp(static_cast<int>(gray * (1.00f + 0.15f * confidence)), 0, 255);
    return static_cast<uint32_t>(red) |
           (static_cast<uint32_t>(green) << 8) |
           (static_cast<uint32_t>(blue) << 16) | 0xff000000u;
}

// Opt-in bounded timing samples. This stores scalar timings only, and runs on
// the depth thread; ordinary runs retain the existing logging behavior.
class OcclusionTimingProfile {
public:
    explicit OcclusionTimingProfile(bool enabled) : enabled_(enabled) {
        if (enabled_) samples_.reserve(kWindow);
    }
    enum Event { CaptureAttempt, CaptureFailure, DuplicatePair, MatchFailure, CancelledPublication,
                 ResetGeneration, ResetCaptureDisabled, ResetNoPose, ResetPoseFallback, Published,
                 CaptureUnavailable, ResetAlignment, EventCount };
    ~OcclusionTimingProfile() { reportEvents(true); }
    void event(Event index) { if (enabled_) ++events_[index]; }
    void reportEvents(bool final=false, uint64_t generation=0) {
        if (!enabled_) return;
        if (!final) lastGeneration_=generation;
        const auto now=std::chrono::steady_clock::now();
        if (!final && std::chrono::duration<double>(now-lastReport_).count()<5) return;
        lastReport_=now;
        static const char* names[]{"capture_attempts","capture_failures","duplicate_pairs","match_failures",
            "cancelled_publications","reset_generation","reset_capture_disabled","reset_no_pose","reset_pose_fallback","published","capture_unavailable","reset_alignment"};
        std::cout<<"[OcclusionEvents] elapsed_s="<<std::chrono::duration<double>(now-started_).count()
                 <<" capture_generation="<<lastGeneration_;
        for(size_t i=0;i<events_.size();++i)std::cout<<' '<<names[i]<<'='<<events_[i];
        std::ifstream status("/proc/self/status");std::string line;
        while(std::getline(status,line)) if(line.rfind("VmHWM:",0)==0)std::cout<<" rss_high_water="<<line.substr(6);
        std::cout<<std::endl;
    }
    void add(const FrameSpatial::OcclusionBenchmark::Metadata& metadata,
             bool recording, bool capturePose) {
        if (!enabled_) return;
        if (samples_.empty()) beginNs_ = metadata.captureBeginNs;
        samples_.push_back(metadata.timings);
        recorded_ += recording;
        fallback_ += !capturePose;
        if (samples_.size() != kWindow) return;
        const double seconds = (metadata.publishNs - beginNs_) * 1e-9;
        std::cout << "[OcclusionProfile] frames=" << samples_.size()
                  << ", hz=" << samples_.size() / seconds
                  << ", recording_frames=" << recorded_
                  << ", pose_fallbacks=" << fallback_;
        std::array<float, kWindow> values{};
        for (int stage = 0; stage < FrameSpatial::OcclusionBenchmark::TimingCount; ++stage) {
            for (size_t i = 0; i < kWindow; ++i) values[i] = samples_[i][stage];
            std::sort(values.begin(), values.end());
            // Floor-index percentiles, matching the benchmark analyzer.
            std::cout << ", " << FrameSpatial::OcclusionBenchmark::timingName(stage)
                      << "_p50=" << values[size_t(.5f * (kWindow - 1))]
                      << ", " << FrameSpatial::OcclusionBenchmark::timingName(stage)
                      << "_p95=" << values[size_t(.95f * (kWindow - 1))]
                      << ", " << FrameSpatial::OcclusionBenchmark::timingName(stage)
                      << "_p99=" << values[size_t(.99f * (kWindow - 1))];
        }
        std::cout << std::endl;
        samples_.clear();
        recorded_ = fallback_ = 0;
    }
private:
    static constexpr size_t kWindow = 120;
    bool enabled_;
    std::array<uint64_t,EventCount> events_{};
    uint64_t lastGeneration_=0;
    std::chrono::steady_clock::time_point started_=std::chrono::steady_clock::now(),lastReport_=started_;
    uint64_t beginNs_ = 0;
    size_t recorded_ = 0, fallback_ = 0;
    std::vector<std::array<float, FrameSpatial::OcclusionBenchmark::TimingCount>> samples_;
};

FrameSpatial::OcclusionConfidenceConfig confidenceConfigFromEnvironment() {
    FrameSpatial::OcclusionConfidenceConfig config;
    const char* value = std::getenv("FRAME_OCCLUSION_CONFIDENCE");
    const char* temporal = std::getenv("FRAME_OCCLUSION_TEMPORAL");
    if (!value || !temporal || std::string(temporal) != "confidence") return config;
    std::istringstream input(value);
    std::array<char, 5> separators{};
    bool parsed = bool(input >> config.strongObservations >> separators[0]
        >> config.measuredObservations >> separators[1]
        >> config.inferredObservations >> separators[2]
        >> config.inferredMaxAge >> separators[3]
        >> config.measuredMaxAge >> separators[4] >> config.pendingMaxAge);
    input >> std::ws;
    parsed = parsed && input.eof() &&
        std::all_of(separators.begin(), separators.end(), [](char c) { return c == ','; });
    config.inferredMaxAge *= .001f;
    config.measuredMaxAge *= .001f;
    config.pendingMaxAge *= .001f;
    try {
        if (!parsed) throw std::invalid_argument("expected counts,counts,counts,ms,ms,ms");
        FrameSpatial::OcclusionSurfaceHistory validate(config);
        std::cout << "[Occlusion] confidence_config=" << value << std::endl;
        return config;
    } catch (const std::exception& e) {
        std::cerr << "[Occlusion] Invalid FRAME_OCCLUSION_CONFIDENCE: " << e.what()
                  << "; using baseline confidence settings" << std::endl;
        return {};
    }
}

void depthWorker(FrameSpatial::DMACapture* capture) {
    // On normal exit and exceptions no published occluder survives the worker.
    struct PublicationGuard {
        ~PublicationGuard() { if(gOcclusionOnly) clearPublishedCloud(); }
    } publicationGuard;
    FrameSpatial::DepthPipeline rectifier;
    const char* matcherSetting = std::getenv(gOcclusionOnly ? "FRAME_OCCLUSION_STEREO" : "FRAME_CLOUD_STEREO");
    const bool guarded = matcherSetting &&
                         std::string(matcherSetting) == "guarded";
    const bool model = !gOcclusionOnly && matcherSetting && std::string(matcherSetting) == "model";
    const bool sgbm = matcherSetting && std::string(matcherSetting) == "sgbm";
    FrameSpatial::VulkanStereoPipeline baselineStereo;
    FrameSpatial::VulkanGuardedStereoPipeline guardedStereo;
    FrameSpatial::ModelDepthPipeline modelStereo;
    // Capture-time pose of the previous Model pair, for the temporal warp.
    TimedHeadPose previousModelPose;
    bool previousModelAtCapture = false;
    double sgbmMilliseconds = 0;
    const auto stereoError = [&]() -> const std::string& {
        if (model) return modelStereo.lastError();
        return guarded ? guardedStereo.lastError() : baselineStereo.lastError();
    };
    const auto stereoMs = [&]() {
        if (model) return modelStereo.lastComputeMs();
        if (sgbm) return static_cast<float>(sgbmMilliseconds);
        return guarded ? guardedStereo.lastComputeMs() : baselineStereo.lastComputeMs();
    };
    if ((model && !modelStereo.initialize()) ||
        (!model && !sgbm && !(guarded ? guardedStereo.initialize(gShaderDirectory, false, false, true)
                                    : baselineStereo.initialize(gShaderDirectory)))) {
        std::cerr << "[Cloud] Stereo initialization failed: "
                  << stereoError() << std::endl;
        gRunning.store(false);
        return;
    }
    std::cout << "[Cloud] Stereo matcher: "
              << (model ? "Model learned depth (experimental, census-validated disparity)"
                       : sgbm ? "sgbm, persistent confirmed voxels"
                       : guarded ? "guarded GPU g3/e1 (experimental)" : "baseline GPU")
              << std::endl;

    constexpr float kVoxelSize = 0.020f;
    constexpr float kMinDepth = 0.25f;
    constexpr float kMaxDepth = 5.0f;
    constexpr uint32_t kKeepFrames = 300;
    constexpr uint32_t kUnconfirmedKeepFrames = 30;
    constexpr uint32_t kMinObservations = 3;
    constexpr size_t kMaxVoxels = 350000;
    constexpr size_t kMaxRenderPoints = 260000;
    std::unordered_map<VoxelKey, Voxel, VoxelHash> voxels;
    voxels.reserve(kMaxVoxels);
    // Mesh mode 2: Vogon-style sparse TSDF, meshed about once per second.
    FrameSpatial::TsdfConfig tsdfConfig;
    if (const char* v = std::getenv("FRAME_TSDF_VOXEL")) tsdfConfig.voxel = std::max(0.01f, float(std::atof(v)));
    tsdfConfig.truncation = 4.0f * tsdfConfig.voxel;
    if (const char* v = std::getenv("FRAME_TSDF_TRUNCATION")) tsdfConfig.truncation = std::max(tsdfConfig.voxel, float(std::atof(v)));
    FrameSpatial::TsdfVolume tsdf(tsdfConfig);
    constexpr uint32_t kTsdfMeshEvery = 5;
    constexpr size_t kTsdfMaxTriangles = 480000;  // renderer holds 1.5M mesh vertices
    double tsdfIntegrateMs = 0, tsdfMeshMs = 0;
    size_t tsdfTriangles = 0;

    uint32_t frame = 0;
    double lastVisibilityCheck = 0.0;
    uint64_t totalPurged = 0;
    uint64_t lastLeftHash = 0, lastRightHash = 0;
    bool havePair = false;
    // Guarded preview updates are slower than the baseline's 200 ms budget.
    // Retain capture-anchored legacy surfaces for a bounded 2.5 seconds.
    const float historyMaxAge = guarded ? 2.5f : 0.2f;
    gOcclusionMaxAge.store(guarded ? 2.5 : 0.5);
    FrameSpatial::OcclusionSurfaceHistory occlusionHistory(confidenceConfigFromEnvironment(), historyMaxAge);
    // Both completion stages only use measured anchors from the current capture.
    const std::vector<float> currentAnchorAges(gOcclusionOnly?
        size_t(FrameSpatial::RECT_WIDTH)*FrameSpatial::RECT_HEIGHT:0,0);
    TimedHeadPose previousOcclusionPose;
    TimedHeadPose previousFusionPose;
    double previousOcclusionTime = 0.0;
    const char* completionSetting = std::getenv("FRAME_OCCLUSION_COMPLETION");
    const bool completePlanes = !completionSetting || std::string(completionSetting) != "0";
    if (gOcclusionOnly)
        std::cout << "[Occlusion] Image-guided plane completion "
                  << (completePlanes ? "enabled" : "disabled") << std::endl;
    const char* temporalSetting = std::getenv("FRAME_OCCLUSION_TEMPORAL");
    const bool confidenceTemporal = !guarded && temporalSetting && std::string(temporalSetting)=="confidence";
    std::unique_ptr<FrameSpatial::OcclusionBenchmark::Recorder> benchmark;
    if (gOcclusionOnly) if (const char* path=std::getenv("FRAME_OCCLUSION_BENCHMARK")) {
        FrameSpatial::OcclusionBenchmark::Header h;
        h.focal=FrameSpatial::RECT_FOCAL; h.cx=FrameSpatial::RECT_CX; h.cy=FrameSpatial::RECT_CY;
        const auto origin=rectifier.rectifiedPointInHead(int(h.cx),int(h.cy),0);
        const auto forward=rectifier.rectifiedPointInHead(int(h.cx),int(h.cy),1);
        const auto right=rectifier.rectifiedPointInHead(int(h.cx+h.focal),int(h.cy),1);
        const auto down=rectifier.rectifiedPointInHead(int(h.cx),int(h.cy+h.focal),1);
        for(int r=0;r<3;++r) {
            h.headFromOptical[r*4]=right[r]-forward[r];
            h.headFromOptical[r*4+1]=down[r]-forward[r];
            h.headFromOptical[r*4+2]=forward[r]-origin[r];
            h.headFromOptical[r*4+3]=origin[r];
        }
        if (const char* smoke=std::getenv("FRAME_OCCLUSION_BENCHMARK_SMOKE"))
            if(std::string(smoke)=="1") { h.warmup=0; h.segment=1; h.recovery=0; }
        try {
            uint32_t budget=FrameSpatial::OcclusionBenchmark::FrameLimit;
            if(const char* value=std::getenv("FRAME_OCCLUSION_FRAME_BUDGET")) {
                size_t used=0; const auto parsed=std::stoul(value,&used);
                if(used!=std::string(value).size() || parsed==0 ||
                   parsed>FrameSpatial::OcclusionBenchmark::MaxFrameLimit)
                    throw std::invalid_argument("Invalid FRAME_OCCLUSION_FRAME_BUDGET");
                budget=uint32_t(parsed);
            }
            benchmark=std::make_unique<FrameSpatial::OcclusionBenchmark::Recorder>(path,h,budget);
        }
        catch(const std::exception& e) { std::cerr<<"[Benchmark] "<<e.what()<<std::endl; }
    }
    if(gOcclusionOnly) std::cout<<"[Occlusion] temporal="<<(confidenceTemporal?"confidence":"legacy")<<std::endl;
    // This worker consumes luma only. Retain a switch for matched capture A/B runs.
    const char* uvSetting = std::getenv("FRAME_CAPTURE_UV");
    const bool captureUV = uvSetting && std::string(uvSetting) == "1";
    std::cout << "[Cloud] capture=" << (captureUV ? "Y+UV" : "Y only") << std::endl;
    const char* profileSetting = std::getenv("FRAME_OCCLUSION_PROFILE");
    OcclusionTimingProfile timingProfile(gOcclusionOnly && profileSetting &&
                                        std::string(profileSetting) == "1");
    using Clock=std::chrono::steady_clock;
    const auto milliseconds=[](Clock::time_point begin) {
        return float(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
    };
    const auto nanoseconds=[](const timespec& t) {
        return uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec);
    };
    // Optional bounded sequence recording for coverage-flicker regression tests.
    // Only processed metric depth is recorded, never DMA camera buffers.
    std::ofstream occlusionRecording;
    unsigned recordedFrames = 0;
    if (gOcclusionOnly) if (const char* path = std::getenv("FRAME_OCCLUSION_RECORD")) {
        occlusionRecording.open(path, std::ios::binary | std::ios::trunc);
        const uint32_t dimensions[2]{FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT};
        occlusionRecording.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    }
    uint64_t historyGeneration=gCaptureGeneration.load();
    while (gRunning.load()) {
        const uint64_t captureGeneration=gCaptureGeneration.load();
        timingProfile.reportEvents(false,captureGeneration);
        if(gOcclusionOnly && resetOcclusionHistoryOnCaptureChange(occlusionHistory,historyGeneration)) {
            timingProfile.event(OcclusionTimingProfile::ResetGeneration); previousOcclusionPose.valid=false;
        }
        if (!gCaptureEnabled.load()) {
            if(previousOcclusionPose.valid) timingProfile.event(OcclusionTimingProfile::ResetCaptureDisabled);
            if(gOcclusionOnly) clearPublishedCloud();
            occlusionHistory.reset();
            previousOcclusionPose.valid = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (!capture->isReady()) {
            timingProfile.event(OcclusionTimingProfile::CaptureUnavailable);
            // Published cells keep their original bounded expiry on brief gaps.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        timespec captureStart{};
        timespec captureEnd{};
        clock_gettime(CLOCK_MONOTONIC, &captureStart);
        timingProfile.event(OcclusionTimingProfile::CaptureAttempt);
        if (!capture->captureFrame(captureUV)) {
            timingProfile.event(OcclusionTimingProfile::CaptureFailure);
            // A failed read is not evidence that every room surface disappeared.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        clock_gettime(CLOCK_MONOTONIC, &captureEnd);
        const uint64_t leftHash = capture->leftHash();
        const uint64_t rightHash = capture->rightHash();
        if (havePair && (leftHash == lastLeftHash || rightHash == lastRightHash)) {
            timingProfile.event(OcclusionTimingProfile::DuplicatePair);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        lastLeftHash = leftHash;
        lastRightHash = rightHash;
        havePair = true;


        TimedHeadPose head;
        bool locatedAtCapture = false;
        if (gLocateSpace && gConvertTimespecTime && gXrInstance != XR_NULL_HANDLE &&
            gViewSpace != XR_NULL_HANDLE && gStageSpace != XR_NULL_HANDLE) {
            const int64_t startNs = static_cast<int64_t>(captureStart.tv_sec) *
                                    1000000000ll + captureStart.tv_nsec;
            const int64_t endNs = static_cast<int64_t>(captureEnd.tv_sec) *
                                  1000000000ll + captureEnd.tv_nsec;
            const int64_t middleNs = startNs + (endNs - startNs) / 2;
            timespec midpoint{};
            midpoint.tv_sec = middleNs / 1000000000ll;
            midpoint.tv_nsec = middleNs % 1000000000ll;
            XrTime captureTime = 0;
            if (XR_SUCCEEDED(gConvertTimespecTime(gXrInstance, &midpoint,
                                                  &captureTime))) {
                XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
                if (XR_SUCCEEDED(gLocateSpace(gViewSpace, gStageSpace,
                                               captureTime, &location))) {
                    const XrSpaceLocationFlags required =
                        XR_SPACE_LOCATION_POSITION_VALID_BIT |
                        XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
                    if ((location.locationFlags & required) == required) {
                        head = {location.pose, captureTime, true};
                        locatedAtCapture = true;
                    }
                }
            }
        }
        if (!locatedAtCapture) {
            std::lock_guard<std::mutex> lock(gPoseMutex);
            head = gLatestHeadPose;
        }
        if (!head.valid) {
            timingProfile.event(OcclusionTimingProfile::ResetNoPose);
            if(gOcclusionOnly) clearPublishedCloud();
            occlusionHistory.reset();
            previousOcclusionPose.valid = false;
            continue;
        }

        const bool recording=benchmark && benchmark->active();
        FrameSpatial::OcclusionBenchmark::Metadata metadata;
        metadata.captureBeginNs=nanoseconds(captureStart);
        metadata.captureEndNs=nanoseconds(captureEnd);
        metadata.timings[FrameSpatial::OcclusionBenchmark::Capture]=float((metadata.captureEndNs-metadata.captureBeginNs)*1e-6);
        auto stageStart=Clock::now();
        rectifier.rectify(capture->getLeftBuffer(), capture->getRightBuffer());
        metadata.timings[FrameSpatial::OcclusionBenchmark::Rectify]=milliseconds(stageStart);
        stageStart=Clock::now();
        std::vector<float> depth;
        bool matched = false;
        if (model) {
            FrameSpatial::ModelMotion modelMotion;
            if (locatedAtCapture && previousModelAtCapture) {  // always sent; used only with --temporal
                const auto motion = occlusionReprojection(rectifier, previousModelPose.pose, head.pose);
                modelMotion.aligned = true;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) modelMotion.rotation[i][j] = motion.rotation[i][j];
                    modelMotion.translation[i] = motion.translation[i];
                }
            }
            matched = modelStereo.process(rectifier.getRectifiedLeft(), rectifier.getRectifiedRight(),
                                          depth, modelMotion, &rectifier.getSphericalLeft(),
                                          &rectifier.getSphericalRight(), rectifier.sphericalStepRadians());
            previousModelPose = head;
            previousModelAtCapture = locatedAtCapture;
        } else if (sgbm) {
            depth.assign(FrameSpatial::RECT_WIDTH * FrameSpatial::RECT_HEIGHT, 0.0f);
            const auto matchStart = Clock::now();
            FrameSpatial::matchSgbm(rectifier.getRectifiedLeft().data(),
                                     rectifier.getRectifiedRight().data(),
                                     FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT,
                                     FrameSpatial::FOCAL_BASELINE, depth.data());
            sgbmMilliseconds = std::chrono::duration<double, std::milli>(
                                   Clock::now() - matchStart)
                                   .count();
            matched = true;
        } else {
            matched = guarded
                ? guardedStereo.process(rectifier.getRectifiedLeft(), rectifier.getRectifiedRight(), depth)
                : baselineStereo.process(rectifier.getRectifiedLeft(), rectifier.getRectifiedRight(), depth);
        }
        if (!matched) {
            timingProfile.event(OcclusionTimingProfile::MatchFailure);
            std::cerr << "[Cloud] Stereo failed: " << stereoError()
                      << std::endl;
            gRunning.store(false);
            break;
        }

        metadata.timings[FrameSpatial::OcclusionBenchmark::Stereo]=milliseconds(stageStart);
        stageStart=Clock::now();
        FrameSpatial::removeDepthSpeckles(depth, FrameSpatial::RECT_WIDTH,
                                           FrameSpatial::RECT_HEIGHT);
        metadata.timings[FrameSpatial::OcclusionBenchmark::Speckles]=milliseconds(stageStart);
        ++frame;
        if (gOcclusionOnly) {
            const auto surfaceStart = std::chrono::steady_clock::now();
            const double captureSeconds = captureEnd.tv_sec + captureEnd.tv_nsec * 1e-9;
            stageStart=Clock::now();
            const auto smoothed = FrameSpatial::smoothOcclusionDepth(
                depth, FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT);
            metadata.timings[FrameSpatial::OcclusionBenchmark::Smooth]=milliseconds(stageStart);
            stageStart=Clock::now();
            FrameSpatial::OcclusionCompletionStats completion;
            FrameSpatial::OcclusionCompletionContract completionContract;
            const auto completed = completePlanes ? FrameSpatial::completeOcclusionPlanes(
                smoothed, rectifier.getRectifiedLeft(), FrameSpatial::RECT_WIDTH,
                FrameSpatial::RECT_HEIGHT, &completion, &completionContract) : smoothed;
            if (!completePlanes)
                completion.measured = std::count_if(smoothed.begin(), smoothed.end(),
                                                     FrameSpatial::surfaceDepth);
            metadata.timings[FrameSpatial::OcclusionBenchmark::Completion]=milliseconds(stageStart);
            stageStart=Clock::now();
            // Even with plane completion disabled, retain image boundaries in
            // the measured-bracket hole pass.
            if (!completePlanes) {
                completionContract.veto.assign(smoothed.size(),0);
                completionContract.provenance.resize(smoothed.size());
                for (size_t i=0; i<smoothed.size(); ++i)
                    completionContract.provenance[i]=FrameSpatial::surfaceDepth(smoothed[i]) ?
                        FrameSpatial::CompletionMeasured : FrameSpatial::CompletionUnknown;
            }
            auto surfaceDepth = FrameSpatial::fillOcclusionHoles(
                completed, smoothed, FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT,
                completionContract, rectifier.getRectifiedLeft());
            metadata.timings[FrameSpatial::OcclusionBenchmark::Holes]=milliseconds(stageStart);
            stageStart=Clock::now();
            const auto evidence=FrameSpatial::occlusionEvidence(
                surfaceDepth,depth,FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT);
            metadata.timings[FrameSpatial::OcclusionBenchmark::Evidence]=milliseconds(stageStart);
            const auto benchmarkInput=recording?surfaceDepth:std::vector<float>{};
            const float elapsed = float(captureSeconds - previousOcclusionTime);
            const bool aligned = locatedAtCapture && previousOcclusionPose.valid;
            const auto transform = aligned ?
                occlusionReprojection(rectifier, previousOcclusionPose.pose, head.pose) :
                FrameSpatial::OcclusionReprojection{};
            if (occlusionRecording.is_open()) {
                const uint8_t alignment = aligned;
                occlusionRecording.write(reinterpret_cast<const char*>(&elapsed), sizeof(elapsed));
                occlusionRecording.write(reinterpret_cast<const char*>(&alignment), sizeof(alignment));
                for (const auto& row : transform.rotation)
                    occlusionRecording.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * 3);
                occlusionRecording.write(reinterpret_cast<const char*>(transform.translation.data()), sizeof(float) * 3);
                occlusionRecording.write(reinterpret_cast<const char*>(surfaceDepth.data()), surfaceDepth.size() * sizeof(float));
                if (++recordedFrames == 48) occlusionRecording.close();
            }
            // A fallback render pose cannot anchor an occluder to camera capture.
            // Preserve pre-history input for recording, suppress final geometry,
            // and restart confirmation once capture-time poses return.
            stageStart=Clock::now();
            // Both completion passes use this frame's measured anchors exclusively.
            // Inferred provenance renews anchor support, never direct-measurement age.
            if (locatedAtCapture) {
                if(!aligned)timingProfile.event(OcclusionTimingProfile::ResetAlignment);
                surfaceDepth = occlusionHistory.update(surfaceDepth,
                    FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT,
                    FrameSpatial::RECT_FOCAL, FrameSpatial::RECT_CX, FrameSpatial::RECT_CY,
                    aligned ? &transform : nullptr, elapsed, confidenceTemporal?&evidence:nullptr,
                    &evidence, &currentAnchorAges);
            } else {
                timingProfile.event(OcclusionTimingProfile::ResetPoseFallback);
                occlusionHistory.reset();
                std::fill(surfaceDepth.begin(),surfaceDepth.end(),0);
            }
            metadata.timings[FrameSpatial::OcclusionBenchmark::History]=milliseconds(stageStart);
            stageStart=Clock::now();
            const auto indices = FrameSpatial::occlusionSurfaceIndices(
                surfaceDepth, FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT);
            const auto cellLifetimes=locatedAtCapture?FrameSpatial::occlusionSurfaceCellLifetimes(
                surfaceDepth,occlusionHistory.supportLifetimes(),FrameSpatial::RECT_WIDTH,
                FrameSpatial::RECT_HEIGHT,indices):std::vector<float>{};
            float maxLifetime=0;
            std::vector<PointVertex> surface;
            surface.reserve(indices.size());
            for(size_t cell=0;cell<cellLifetimes.size();++cell) {
                const float lifetime=cellLifetimes[cell];
                if(!std::isfinite(lifetime) || lifetime<=0) continue;
                maxLifetime=std::max(maxLifetime,lifetime);
                // Every cell's vertices share the earliest deadline of ALL covered
                // samples, including coarse-cell interiors and shared edges.
                for(size_t vertex=0;vertex<6;++vertex) {
                    const size_t index=indices[cell*6+vertex];
                    const auto inHead = rectifier.rectifiedPointInHead(
                        index % FrameSpatial::RECT_WIDTH, index / FrameSpatial::RECT_WIDTH,
                        surfaceDepth[index]);
                    const auto rotated = rotateVector(head.pose.orientation, inHead);
                    surface.push_back({head.pose.position.x + rotated[0],
                                       head.pose.position.y + rotated[1],
                                       head.pose.position.z + rotated[2], 0xffffffffu, lifetime});
                }
            }
            const auto count = static_cast<uint32_t>(surface.size());
            previousOcclusionPose = head;
            previousOcclusionPose.valid = locatedAtCapture;
            previousOcclusionTime = captureSeconds;
            const auto surfaceMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - surfaceStart).count();
            const bool meshPublished=publishOcclusionCloud(surface,captureSeconds,
                captureSeconds+maxLifetime,captureGeneration,locatedAtCapture);
            if(locatedAtCapture && !meshPublished) {
                timingProfile.event(OcclusionTimingProfile::CancelledPublication);
                occlusionHistory.reset(); previousOcclusionPose.valid=false;
                // A cancelled computation is not a published/recorded depth frame.
                --frame;
                continue;
            }
            metadata.timings[FrameSpatial::OcclusionBenchmark::Mesh]=milliseconds(stageStart);
            timespec published{}; clock_gettime(CLOCK_MONOTONIC,&published);
            metadata.publishNs=nanoseconds(published);
            metadata.timings[FrameSpatial::OcclusionBenchmark::PublishAge]=float((metadata.publishNs-
                (metadata.captureBeginNs+(metadata.captureEndNs-metadata.captureBeginNs)/2))*1e-6);
            if(meshPublished)timingProfile.event(OcclusionTimingProfile::Published);
            timingProfile.add(metadata, recording, locatedAtCapture);
            if(recording) {
                metadata.frame=frame;
                metadata.poseTime=head.time; metadata.elapsed=elapsed; metadata.transform=transform;
                metadata.flags=(aligned?uint32_t(FrameSpatial::OcclusionBenchmark::Aligned):0u)|
                    (locatedAtCapture?uint32_t(FrameSpatial::OcclusionBenchmark::CapturePose):0u)|
                    (completePlanes?uint32_t(FrameSpatial::OcclusionBenchmark::Completed):0u)|
                    (confidenceTemporal?uint32_t(FrameSpatial::OcclusionBenchmark::Confidence):0u)|
                    (guarded?uint32_t(FrameSpatial::OcclusionBenchmark::Guarded):0u);
                metadata.headPose={head.pose.position.x,head.pose.position.y,head.pose.position.z,
                    head.pose.orientation.x,head.pose.orientation.y,head.pose.orientation.z,head.pose.orientation.w};
                benchmark->append(metadata,rectifier.getRectifiedLeft(),rectifier.getRectifiedRight(),
                    depth,benchmarkInput,evidence);
            }
            gDepthFrames.store(frame);
            gStablePoints.store(count / 3);
            gStereoMs.store(stereoMs());
            if (frame == 1 || frame % ((guarded || model) ? 5u : 30u) == 0u)
                std::cout << "[Occlusion] frame=" << frame << " stereo="
                          << stereoMs() << " ms, triangles=" << count / 3
                          << ", surface=" << surfaceMs << " ms"
                          << ", measured=" << completion.measured
                          << ", completed=" << completion.filled
                          << ", planes=" << completion.planes
                          << ", conflicts=" << completion.conflicts
                          << ", confirmed=" << occlusionHistory.confirmed
                          << ", pending=" << occlusionHistory.pending
                          << ", held=" << occlusionHistory.held
                          << ", pose=" << (locatedAtCapture ? "capture-time" : "latest")
                          << std::endl;
            continue;
        }

        // Reimplemented Meshing Pipeline: Power-mean filtering + Discontinuity-gated Triangulation
        const int activeMeshMode = gMeshViewerMode.load();
        if (activeMeshMode == 2) {
            FrameSpatial::TsdfStats tsdfStats;
            // Integrate only with a capture-time pose: a fallback pose would smear the volume.
            if (locatedAtCapture) {
                const int cx = int(FrameSpatial::RECT_CX), cy = int(FrameSpatial::RECT_CY);
                const int f = int(FrameSpatial::RECT_FOCAL);
                const auto origin = rectifier.rectifiedPointInHead(cx, cy, 0);
                const auto forward = rectifier.rectifiedPointInHead(cx, cy, 1);
                const auto rightAxis = rectifier.rectifiedPointInHead(cx + f, cy, 1);
                const auto downAxis = rectifier.rectifiedPointInHead(cx, cy + f, 1);
                FrameSpatial::TsdfPose cameraToWorld;
                std::array<float, 3> columns[3];
                for (int i = 0; i < 3; ++i) {
                    columns[0][i] = rightAxis[i] - forward[i];
                    columns[1][i] = downAxis[i] - forward[i];
                    columns[2][i] = forward[i] - origin[i];
                }
                for (int col = 0; col < 3; ++col) {
                    const auto world = rotateVector(head.pose.orientation, columns[col]);
                    for (int row = 0; row < 3; ++row) cameraToWorld.rotation[row][col] = world[row];
                }
                const auto originWorld = rotateVector(head.pose.orientation, origin);
                cameraToWorld.translation[0] = head.pose.position.x + originWorld[0];
                cameraToWorld.translation[1] = head.pose.position.y + originWorld[1];
                cameraToWorld.translation[2] = head.pose.position.z + originWorld[2];
                const auto integrateStart = Clock::now();
                tsdfStats = tsdf.integrate(depth.data(), rectifier.getRectifiedLeft().data(),
                    FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT, FrameSpatial::RECT_FOCAL,
                    FrameSpatial::RECT_CX, FrameSpatial::RECT_CY, cameraToWorld,
                    model && modelStereo.lastConfidence().size() == depth.size() ?
                        modelStereo.lastConfidence().data() : nullptr);
                tsdfIntegrateMs = milliseconds(integrateStart);
            }
            if (frame == 1 || frame % kTsdfMeshEvery == 0) {
                const auto meshStart = Clock::now();
                std::vector<FrameSpatial::TsdfVertex> triangles;
                const float focus[3]{head.pose.position.x, head.pose.position.y, head.pose.position.z};
                tsdf.extractMesh(triangles, kTsdfMaxTriangles, focus);
                std::vector<PointVertex> meshVertices;
                meshVertices.reserve(triangles.size());
                for (const auto& v : triangles)
                    meshVertices.push_back({v.x, v.y, v.z, packGray(v.luma, 8)});
                tsdfTriangles = triangles.size() / 3;
                {
                    std::lock_guard<std::mutex> lock(gCloudMutex);
                    gCloud.swap(meshVertices);
                }
                gCloudTime.store(captureEnd.tv_sec + captureEnd.tv_nsec * 1e-9);
                tsdfMeshMs = milliseconds(meshStart);
            }
            gDepthFrames.store(frame);
            gStablePoints.store(uint32_t(tsdfTriangles));
            gStereoMs.store(stereoMs());
            if (frame == 1 || frame % ((guarded || model) ? 5u : 30u) == 0u)
                std::cout << "[TSDF] frame=" << frame << " stereo=" << stereoMs()
                          << " ms, integrate=" << tsdfIntegrateMs << " ms (touched "
                          << tsdfStats.touchedLeaves << ", new " << tsdfStats.allocatedLeaves
                          << ", evicted " << tsdfStats.evictedLeaves << "), leaves=" << tsdf.leafCount()
                          << ", mesh=" << tsdfMeshMs << " ms, triangles=" << tsdfTriangles
                          << ", pose=" << (locatedAtCapture ? "capture-time" : "latest") << std::endl;
            continue;
        }
        if (activeMeshMode == 1 || activeMeshMode == 3) {
            // 1. Meta-style power-mean filter (p=4) on inverse depth to prevent edge bleeding
            const auto smoothedDepth = FrameSpatial::powerMeanFilterDepth(
                depth, FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT, 4, 2, kMinDepth, kMaxDepth);

            // 2. Continuous 3D surface mesh generation with depth tear rejection
            const auto& luma = rectifier.getRectifiedLeft();
            auto meshSurface = FrameSpatial::triangulateDepthMap(
                smoothedDepth, luma, rectifier, head.pose,
                FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT,
                2, 0.045f, 0.05f);

            const uint32_t triangleCount = static_cast<uint32_t>(meshSurface.size() / 3);
            {
                std::lock_guard<std::mutex> lock(gCloudMutex);
                gCloud.swap(meshSurface);
            }
            gCloudTime.store(captureEnd.tv_sec + captureEnd.tv_nsec * 1e-9);
            gDepthFrames.store(frame);
            gValidDepth.store(triangleCount * 3);
            gStablePoints.store(triangleCount);
            gStereoMs.store(stereoMs());

            if (frame == 1 || (frame % ((guarded || model) ? 5u : 30u)) == 0u) {
                std::cout << "[Mesh] frame=" << frame << " stereo=" << stereoMs()
                          << " ms, triangles=" << triangleCount
                          << ", vertices=" << triangleCount * 3
                          << ", pose=" << (locatedAtCapture ? "capture-time" : "latest")
                          << std::endl;
            }
            continue;
        }

        uint32_t valid = 0;
        uint32_t purged = 0;
        bool poseBlurred = false;
        if (sgbm && previousFusionPose.valid) {
            const auto& prior = previousFusionPose.pose;
            const auto& current = head.pose;
            const float qdot = std::abs(prior.orientation.x * current.orientation.x +
                                        prior.orientation.y * current.orientation.y +
                                        prior.orientation.z * current.orientation.z +
                                        prior.orientation.w * current.orientation.w);
            const float turned = 2.0f * std::acos(std::clamp(qdot, 0.0f, 1.0f));
            const float dx = prior.position.x - current.position.x;
            const float dy = prior.position.y - current.position.y;
            const float dz = prior.position.z - current.position.z;
            // Skip a frame whose head moved enough that the buffer-midpoint
            // pose no longer belongs to the exposure.
            poseBlurred = turned > 0.22f || dx * dx + dy * dy + dz * dz > 0.08f * 0.08f;
        }
        if (head.valid) previousFusionPose = head;
        // A straight-on wall beyond 1 m is where a pixel of disparity error
        // becomes several centimeters. Close range and a real slant lock on.
        auto reliableSurface = [&](int x, int y, float z) {
            if (!(z >= kMinDepth && z <= kMaxDepth)) return false;
            if (z <= 1.0f) return true;
            auto disparityAt = [&](int sx, int sy) {
                if (sx < 2 || sy < 2 || sx >= FrameSpatial::RECT_WIDTH - 2 ||
                    sy >= FrameSpatial::RECT_HEIGHT - 2) return 0.0f;
                const float sample = depth[size_t(sy) * FrameSpatial::RECT_WIDTH + sx];
                return sample > 0.0f ? FrameSpatial::FOCAL_BASELINE / sample : 0.0f;
            };
            const float left = disparityAt(x - 24, y);
            const float right = disparityAt(x + 24, y);
            const float up = disparityAt(x, y - 24);
            const float down = disparityAt(x, y + 24);
            return (left > 0.0f && right > 0.0f && std::abs(left - right) > 0.45f) ||
                   (up > 0.0f && down > 0.0f && std::abs(up - down) > 0.45f);
        };
        // Check only against a pose located at capture time. A fallback render
        // pose can shift the depth image during head motion and erase good points.
        if (locatedAtCapture && (frame % 3u) == 0u) {
            const double now = captureEnd.tv_sec + captureEnd.tv_nsec * 1e-9;
            const float elapsed = lastVisibilityCheck > 0.0 ?
                static_cast<float>(now - lastVisibilityCheck) : 0.0f;
            lastVisibilityCheck = now;
            const auto& q = head.pose.orientation;
            const XrQuaternionf inverse{-q.x, -q.y, -q.z, q.w};
            std::vector<std::pair<VoxelKey, std::array<float, 3>>> snaps;
            for (auto it = voxels.begin(); it != voxels.end();) {
                Voxel& voxel = it->second;
                const auto inHead = rotateVector(inverse, {
                    voxel.x - head.pose.position.x,
                    voxel.y - head.pose.position.y,
                    voxel.z - head.pose.position.z});
                const auto inRect = rectifier.headPointInRectified(inHead);
                if (inRect[2] >= kMinDepth && inRect[2] <= kMaxDepth) {
                    const float px = FrameSpatial::RECT_CX +
                        FrameSpatial::RECT_FOCAL * inRect[0] / inRect[2];
                    const float py = FrameSpatial::RECT_CY +
                        FrameSpatial::RECT_FOCAL * inRect[1] / inRect[2];
                    if (px >= 2 && px < FrameSpatial::RECT_WIDTH - 2 &&
                        py >= 2 && py < FrameSpatial::RECT_HEIGHT - 2) {
                        const int ix = static_cast<int>(std::round(px));
                        const int iy = static_cast<int>(std::round(py));
                        const auto evidence = FrameSpatial::classifyCloudVisibility(
                            depth, FrameSpatial::RECT_WIDTH, FrameSpatial::RECT_HEIGHT,
                            ix, iy, inRect[2], FrameSpatial::FOCAL_BASELINE);
                        voxel.aging.observe(evidence, elapsed);
                        const float measured = depth[size_t(iy) * FrameSpatial::RECT_WIDTH + ix];
                        const float noise = std::max(
                            0.04f, 0.5f * inRect[2] * inRect[2] / FrameSpatial::FOCAL_BASELINE);
                        const float limit = std::max(0.18f, 0.15f * inRect[2]);
                        const float gap = std::abs(measured - inRect[2]);
                        if (sgbm && !poseBlurred && !voxel.aging.expired() &&
                            reliableSurface(ix, iy, measured) && gap > noise && gap < limit) {
                            const auto snapped = rectifier.rectifiedPointInHead(ix, iy, measured);
                            const auto world = rotateVector(head.pose.orientation, snapped);
                            snaps.push_back({it->first, {
                                head.pose.position.x + world[0],
                                head.pose.position.y + world[1],
                                head.pose.position.z + world[2]}});
                        }
                    }
                }
                if (voxel.aging.expired()) {
                    it = voxels.erase(it);
                    ++purged;
                } else ++it;
            }
            totalPurged += purged;
            for (const auto& snap : snaps) {
                const auto found = voxels.find(snap.first);
                if (found == voxels.end()) continue;
                Voxel moved = found->second;
                voxels.erase(found);
                moved.x = snap.second[0];
                moved.y = snap.second[1];
                moved.z = snap.second[2];
                moved.lastFrame = frame;
                const VoxelKey key{
                    static_cast<int>(std::floor(moved.x / kVoxelSize)),
                    static_cast<int>(std::floor(moved.y / kVoxelSize)),
                    static_cast<int>(std::floor(moved.z / kVoxelSize)),
                };
                auto [placed, inserted] = voxels.try_emplace(key, moved);
                if (!inserted && placed->second.lastFrame != frame) {
                    placed->second.observations = std::max(placed->second.observations, moved.observations);
                    placed->second.lastFrame = frame;
                    placed->second.aging.reset();
                }
            }
        }

        const auto& luma = rectifier.getRectifiedLeft();
        for (int y = 2; y < FrameSpatial::RECT_HEIGHT - 2; y += 2) {
            for (int x = 2; x < FrameSpatial::RECT_WIDTH - 2; x += 2) {
                const size_t index = static_cast<size_t>(y) * FrameSpatial::RECT_WIDTH + x;
                const float z = depth[index];
                if (!(z >= kMinDepth && z <= kMaxDepth)) continue;
                if (sgbm && poseBlurred) continue;
                ++valid;

                // Depth is referenced to the left optical center. Undo the
                // rectified basis and apply the measured camera-to-head rig.
                const auto inHead = rectifier.rectifiedPointInHead(x, y, z);
                const auto rotated = rotateVector(head.pose.orientation, inHead);
                const float wx = head.pose.position.x + rotated[0];
                const float wy = head.pose.position.y + rotated[1];
                const float wz = head.pose.position.z + rotated[2];
                const VoxelKey key{
                    static_cast<int>(std::floor(wx / kVoxelSize)),
                    static_cast<int>(std::floor(wy / kVoxelSize)),
                    static_cast<int>(std::floor(wz / kVoxelSize)),
                };

                auto [it, inserted] = voxels.try_emplace(key);
                Voxel& voxel = it->second;
                if (inserted) {
                    voxel.x = wx;
                    voxel.y = wy;
                    voxel.z = wz;
                    voxel.luminance = static_cast<float>(luma[index]);
                    voxel.observations = 1;
                    voxel.lastFrame = frame;
                    voxel.anchorX = head.pose.position.x;
                    voxel.anchorY = head.pose.position.y;
                    voxel.anchorZ = head.pose.position.z;
                    voxel.anchorQx = head.pose.orientation.x;
                    voxel.anchorQy = head.pose.orientation.y;
                    voxel.anchorQz = head.pose.orientation.z;
                    voxel.anchorQw = head.pose.orientation.w;
                } else if (voxel.lastFrame != frame) {
                    if (sgbm) {
                        const float band = std::max(
                            0.06f, z * z * 0.75f / FrameSpatial::FOCAL_BASELINE);
                        const float dx = wx - voxel.x;
                        const float dy = wy - voxel.y;
                        const float dz = wz - voxel.z;
                        if (dx * dx + dy * dy + dz * dz > band * band) continue;
                        // The same viewpoint repeats the same disparity bias.
                        // Count another observation only after the head has moved,
                        // so a straight-on stare cannot confirm a floating sheet.
                        const float viewDot = std::abs(
                            voxel.anchorQx * head.pose.orientation.x +
                            voxel.anchorQy * head.pose.orientation.y +
                            voxel.anchorQz * head.pose.orientation.z +
                            voxel.anchorQw * head.pose.orientation.w);
                        const float viewTurn = 2.0f * std::acos(std::clamp(viewDot, 0.0f, 1.0f));
                        const float vx = head.pose.position.x - voxel.anchorX;
                        const float vy = head.pose.position.y - voxel.anchorY;
                        const float vz = head.pose.position.z - voxel.anchorZ;
                        if (viewTurn < 0.26f && vx * vx + vy * vy + vz * vz < 0.20f * 0.20f)
                            continue;
                        voxel.anchorX = head.pose.position.x;
                        voxel.anchorY = head.pose.position.y;
                        voxel.anchorZ = head.pose.position.z;
                        voxel.anchorQx = head.pose.orientation.x;
                        voxel.anchorQy = head.pose.orientation.y;
                        voxel.anchorQz = head.pose.orientation.z;
                        voxel.anchorQw = head.pose.orientation.w;
                    }
                    const float weight = voxel.observations < 8 ?
                        1.0f / static_cast<float>(voxel.observations + 1) : 0.125f;
                    voxel.x += (wx - voxel.x) * weight;
                    voxel.y += (wy - voxel.y) * weight;
                    voxel.z += (wz - voxel.z) * weight;
                    voxel.luminance +=
                        (static_cast<float>(luma[index]) - voxel.luminance) * weight;
                    voxel.observations = std::min(voxel.observations + 1, 255u);
                    voxel.lastFrame = frame;
                    voxel.aging.reset();
                }
            }
        }

        if ((frame % ((guarded || model) ? 5u : 30u)) == 0u || voxels.size() > kMaxVoxels) {
            const uint32_t oldest = frame > kKeepFrames ? frame - kKeepFrames : 0;
            for (auto it = voxels.begin(); it != voxels.end();) {
                const bool kept = sgbm && it->second.observations >= kMinObservations;
                if ((!kept && it->second.lastFrame < oldest) ||
                    (it->second.observations < kMinObservations &&
                     frame - it->second.lastFrame > kUnconfirmedKeepFrames) ||
                    (voxels.size() > kMaxVoxels &&
                     it->second.observations < kMinObservations)) {
                    it = voxels.erase(it);
                } else {
                    ++it;
                }
            }
        }

        std::vector<PointVertex> stable;
        stable.reserve(std::min(voxels.size(), kMaxRenderPoints));
        for (const auto& entry : voxels) {
            const Voxel& voxel = entry.second;
            if (voxel.observations < kMinObservations) continue;
            // Repeated mismatches can occupy the same voxel. Require a local
            // surface of stable voxels within 4 cm, not just temporal repeats.
            int neighbors = 0;
            const VoxelKey& key = entry.first;
            for (int dz = -2; dz <= 2 && neighbors < 4; ++dz) {
                for (int dy = -2; dy <= 2 && neighbors < 4; ++dy) {
                    for (int dx = -2; dx <= 2 && neighbors < 4; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const auto found = voxels.find({key.x + dx, key.y + dy, key.z + dz});
                        if (found == voxels.end() ||
                            found->second.observations < kMinObservations) continue;
                        const Voxel& other = found->second;
                        const float vx = other.x - voxel.x;
                        const float vy = other.y - voxel.y;
                        const float vz = other.z - voxel.z;
                        if (vx * vx + vy * vy + vz * vz <= 0.04f * 0.04f) ++neighbors;
                    }
                }
            }
            if (neighbors < 4) continue;
            stable.push_back({voxel.x, voxel.y, voxel.z,
                              packGray(voxel.luminance, voxel.observations)});
            if (stable.size() == kMaxRenderPoints) break;
        }
        const uint32_t stableCount = static_cast<uint32_t>(stable.size());
        {
            std::lock_guard<std::mutex> lock(gCloudMutex);
            gCloud.swap(stable);
        }
        gCloudTime.store(captureEnd.tv_sec + captureEnd.tv_nsec * 1e-9);
        gDepthFrames.store(frame);
        gValidDepth.store(valid);
        gStablePoints.store(stableCount);
        gStereoMs.store(stereoMs());

        if (frame == 1 || (frame % ((guarded || model) ? 5u : 30u)) == 0u) {
            std::cout << "[Cloud] frame=" << frame << " stereo="
                      << stereoMs() << " ms, accepted=" << valid
                      << ", stable=" << gStablePoints.load()
                      << ", voxels=" << voxels.size()
                      << ", purged=" << purged << " (total " << totalPurged << ")"
                      << ", pose=" << (locatedAtCapture ? "capture-time" : "latest")
                      << (sgbm ? (poseBlurred ? ", blurred" : ", steady") : "")
                      << std::endl;
        }
    }
}


} // namespace FrameSpatialCloud
