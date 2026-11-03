#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace FrameSpatial {
// Rigid motion of the rectified left camera since the previous pair:
// P_now = rotation * P_prev + translation (metres, x right, y down, z forward).
// Only used with FRAME_DEPTH_TEMPORAL=1; aligned=false drops the history.
struct ModelMotion {
    bool aligned=false;
    bool reset=false;
    float rotation[3][3]{{1,0,0},{0,1,0},{0,0,1}};
    float translation[3]{};
};
// One persistent Python/DSP worker, used only by the background depth thread.
// The bridge exchanges resized rectified luma and float disparity, never raw DMA.
// FRAME_DEPTH_BACKEND selects the v5 hybrid (default) or the exact prepared executor.
class ModelDepthPipeline {
public:
    ModelDepthPipeline()=default;
    ~ModelDepthPipeline();
    ModelDepthPipeline(const ModelDepthPipeline&)=delete;
    ModelDepthPipeline& operator=(const ModelDepthPipeline&)=delete;
    bool initialize(const std::string& script="scripts/model_depth_bridge.py");
    bool process(const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,
                 std::vector<float>& depthMeters,const ModelMotion& motion=ModelMotion{});
    bool temporal() const {return temporal_;}
    // Per rectified pixel weight in [0.05,1] from the model uncertainty of the
    // last frame (nearest expansion), for confidence-weighted fusion.
    const std::vector<float>& lastConfidence() const {return confidence_;}
    float lastComputeMs() const {return milliseconds_;}
    const std::string& lastError() const {return error_;}
private:
    bool transfer(void* data,size_t bytes,bool sending);
    void stop();
    int fd_=-1,pid_=-1;
    int timeoutSeconds_=30;
    bool temporal_=false;
    int recordFd_=-1;
    bool censusGate_=false;
    int maxUncertainty_=89;
    std::vector<float> confidence_;
    uint32_t recordLimit_=0,recorded_=0;
    uint32_t sequence_=0;
    float milliseconds_=0;
    std::string error_;
};
}
