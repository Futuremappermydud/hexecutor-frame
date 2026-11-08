#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "ir_dot_bridge.hpp"
#include "ir_pair_capture.hpp"
#include "tsdf_volume.hpp"

namespace FrameSpatial {

// CPU IR+dot refine beside the RGB Model TSDF. No second DSP session. Opt-in: FRAME_IR_DOT=1.
//
// RGB disparity is the search prior for a ZNCC match on the IR pair. The result
// reaches the map two ways, and neither adds support the IR data does not have:
//  - takeObservation() hands each IR depth map to the TSDF exactly once, at the
//    pose of the IR exposure. It is the only way IR adds evidence.
//  - apply() corrects RGB depth on the fitted world plane for a short time. It
//    keeps each pixel's RGB confidence and never fills RGB holes, so a corrected
//    pixel counts as the one RGB sample it already was.
class IrDotRefine {
public:
    struct WorldPlane {
        float n[3]{0.f, 0.f, 1.f};
        float c = 0.f;
        uint32_t inliers = 0;
        float centroid[3]{0.f, 0.f, 0.f};
        float axisU[3]{1.f, 0.f, 0.f}, axisV[3]{0.f, 1.f, 0.f};
        float radius = 0.f;
        std::vector<uint8_t> footprint;  // kIrFootprintSize^2 IrFootprintState cells
        bool valid = false;
        std::chrono::steady_clock::time_point timestamp{};  // IR capture
        // Marks every cell whose centre is within `radius` as supported.
        void fillDiskFootprint();
    };
    struct Observation {
        std::shared_ptr<const std::vector<float>> depth;  // RECT_WIDTH * RECT_HEIGHT, 0 = none
        TsdfPose pose;
        uint32_t kept = 0;
    };
    // Rectified-camera-to-world pose at a CLOCK_MONOTONIC time.
    using PoseLocator = std::function<bool(int64_t monotonicNs, TsdfPose& pose)>;

    IrDotRefine();
    ~IrDotRefine();
    void setPoseLocator(PoseLocator locator) { locator_ = std::move(locator); }
    bool initialize();
    bool active() const { return active_; }
    // rgbPose must be the capture-time pose of this disparity.
    void submit(const std::vector<float>& rgbDisparity, const std::vector<uint8_t>& rgbUncertainty,
                const TsdfPose& rgbPose);
    void apply(std::vector<float>& rgbDepth, const std::vector<uint8_t>& rgbLuma, const TsdfPose* currentPose);
    bool takeObservation(Observation& observation);
    const std::string& lastError() const { return error_; }
    WorldPlane worldPlane() const;
    void setTestWorldPlane(const WorldPlane& wp) {
        std::lock_guard<std::mutex> lock(mutex_);
        worldPlane_ = wp;
        active_ = true;
    }

private:
    void worker();
    bool collectDotPair(IrPairCapture::Pair& pair);
    void waitUntil(std::chrono::steady_clock::time_point deadline) const;
    void turnEmitterOff();
    void publish(const IrBridgeReply& reply, const TsdfPose& irPose,
                 std::chrono::steady_clock::time_point captured);
    void stop();

    IrPairCapture capture_;
    std::atomic<bool> running_{false};
    IrBridgeClient bridge_;
    IrPulseProcess pulse_;
    PoseLocator locator_;
    bool active_ = false;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<float> pendingDisp_;
    std::vector<uint8_t> pendingUnc_;
    TsdfPose pendingPose_{};
    bool havePending_ = false, haveObservation_ = false;
    Observation observation_;
    WorldPlane worldPlane_{};
    int periodMs_ = 1500;
    int exposureOffsetNs_ = 8000000;
    float pulseSeconds_ = 0.12f, minDotRatio_ = 1.04f, maxPriorDegrees_ = 6.f;
    std::string error_, pulseCmd_, pulseOffCmd_;
};

}  // namespace FrameSpatial
