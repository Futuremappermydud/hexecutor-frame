#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "model_depth_geometry.hpp"
#include "model_temporal.hpp"
#include <memory>

namespace FrameSpatial {
// Rigid motion of the rectified left camera since the previous pair:
// P_now = rotation * P_prev + translation (metres, x right, y down, z forward).
// Only used with FRAME_DEPTH_TEMPORAL=1. aligned=false or reset=true drops the history,
// as does a gap or rotation beyond FRAME_DEPTH_TEMPORAL_MAX_GAP_MS / _MAX_DEG.
struct ModelMotion {
    bool aligned=false;
    bool reset=false;
    float rotation[3][3]{{1,0,0},{0,1,0},{0,0,1}};
    float translation[3]{};
};
struct ModelCaptureMetadata {
    int64_t captureBeginNs=0, captureEndNs=0, poseTimeNs=0;
    bool captureTimePose=false, poseValid=false;
    float position[3]{};
    float orientation[4]{0,0,0,1};
    float cameraToWorldRotation[9]{1,0,0,0,1,0,0,0,1};
    float cameraToWorldTranslation[3]{};
    // E02 timing diagnostics (v2 recording only): head poses located at
    // poseSampleOffsetMs relative to the buffer-read midpoint, and the DMA slots used.
    static constexpr int POSE_SAMPLES=13;
    static constexpr int poseSampleOffsetMs[POSE_SAMPLES]{-120,-100,-80,-66,-50,-40,-33,-25,-16,-8,0,8,16};
    bool poseSampleValid[POSE_SAMPLES]{};
    float poseSamplePosition[POSE_SAMPLES][3]{};
    float poseSampleOrientation[POSE_SAMPLES][4]{};
    int leftSlot=-1, rightSlot=-1;
    // E02b V4L2-paired capture (zero in the legacy hash-selected mode).
    uint32_t cameraSequence=0;
    int64_t leftEndOfReadoutNs=0, rightEndOfReadoutNs=0, leftWriteStartNs=0, rightWriteStartNs=0;
    uint64_t skippedSequences=0;
};
// One persistent native C++/DSP worker, used only by the background depth thread.
// The bridge exchanges resized rectified luma and float disparity, never raw DMA.
// FRAME_DEPTH_BACKEND selects prepared-qnn (default) or the exact prepared executor.
class ModelDepthPipeline {
public:
    ModelDepthPipeline()=default;
    ~ModelDepthPipeline();
    ModelDepthPipeline(const ModelDepthPipeline&)=delete;
    ModelDepthPipeline& operator=(const ModelDepthPipeline&)=delete;
    bool initialize(const std::string& executable="bin/frame_depth_worker");
    // left/right: 616x616 rectified pinhole luma. With FRAME_DEPTH_SPHERICAL=1 the
    // model instead receives sphericalLeft/Right (256x320, step sphericalStep rad/px,
    // from DepthPipeline) and its disparity is triangulated in that model.
    bool process(const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,
                 std::vector<float>& depthMeters,const ModelMotion& motion=ModelMotion{},
                 const std::vector<uint8_t>* sphericalLeft=nullptr,
                 const std::vector<uint8_t>* sphericalRight=nullptr,float sphericalStep=0,
                 const ModelCaptureMetadata* captureMetadata=nullptr);
    bool spherical() const {return spherical_;}
    bool temporal() const {return temporal_;}
    // Per rectified pixel weight in [0.05,1] from the model uncertainty of the
    // last frame (nearest expansion), for confidence-weighted fusion.
    const std::vector<float>& lastConfidence() const {return confidence_;}
    // Model-resolution plane 0 (256x320) from the last reply, for IR-dot refine.
    const std::vector<float>& lastDisparity() const {return disparity_;}
    const std::vector<uint8_t>& lastUncertainty() const {return uncertainty_;}
    float lastComputeMs() const {return milliseconds_;}
    // CPU disparity validation, metric reprojection and confidence expansion.
    float lastPostprocessMs() const {return postprocessMs_;}
    const std::string& lastError() const {return error_;}
private:
    bool transfer(void* data,size_t bytes,bool sending);
    void stop();
    int fd_=-1,pid_=-1;
    int timeoutSeconds_=30;
    bool temporal_=false;
    int recordFd_=-1;
    bool censusGate_=false;
    bool spherical_=false;
    int recordSphericalFd_=-1;
    std::string recordV2Directory_, backend_, model_, package_;
    bool secondDisparity_=false;
    std::unique_ptr<ModelTemporalState> temporalState_;
    std::vector<float> hidden_,transforms_;
    int64_t temporalMaxGapNs_=0,lastPoseNs_=0;
    double temporalMaxDegrees_=0;
    uint64_t temporalApplied_=0,temporalDropped_[6]{};
    float temporalWarpMs_=0;
    uint32_t recordedV2_=0;
    std::string sphericalDegrees_;
    int maxUncertainty_=89;
    std::vector<float> confidence_;
    std::vector<float> disparity_;
    std::vector<uint8_t> uncertainty_;
    ModelSphericalMap sphericalMap_;
    uint32_t recordLimit_=0,recorded_=0;
    uint32_t sequence_=0;
    float milliseconds_=0;
    float postprocessMs_=0;
    std::string error_;
};
}
