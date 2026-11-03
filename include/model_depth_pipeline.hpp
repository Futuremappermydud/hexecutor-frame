#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "model_depth_geometry.hpp"

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
    // left/right: 616x616 rectified pinhole luma. With FRAME_DEPTH_SPHERICAL=1 the
    // model instead receives sphericalLeft/Right (256x320, step sphericalStep rad/px,
    // from DepthPipeline) and its disparity is triangulated in that model.
    bool process(const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,
                 std::vector<float>& depthMeters,const ModelMotion& motion=ModelMotion{},
                 const std::vector<uint8_t>* sphericalLeft=nullptr,
                 const std::vector<uint8_t>* sphericalRight=nullptr,float sphericalStep=0);
    bool spherical() const {return spherical_;}
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
    bool spherical_=false;
    int recordSphericalFd_=-1;
    std::string sphericalDegrees_;
    int maxUncertainty_=89;
    std::vector<float> confidence_;
    ModelSphericalMap sphericalMap_;
    uint32_t recordLimit_=0,recorded_=0;
    uint32_t sequence_=0;
    float milliseconds_=0;
    std::string error_;
};
}
