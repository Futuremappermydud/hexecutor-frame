#pragma once
// CPU reference for edge classes and one-sided normals. Classify at native
// resolution; a linearly upsampled depth map thickens and shifts silhouettes.
#include "depth_observation.hpp"

namespace FrameSpatial {

// Frozen before any real-image measurement. The jump tolerance matches the
// existing spherical support gate. The crease angle is three times the 5°
// normal tolerance, so planar noise is not a crease. Quadrant fits may be
// noisy up to the stationary depth scatter; a fold is a disagreement between
// two fits that each pass that scatter, not a tighter RMS cut.
struct DepthFeatureConfig {
    float absDepthTol = 0.04f;
    float relDepthTol = 0.04f;
    float creaseDegrees = 15.f;
    float lumaGradient = 24.f;
    int fitRadius = 5;
    int minQuadrantSamples = 8;
    float maxFitRms = 0.02f;
};

enum class EdgeMode : uint8_t {
    DepthOnly = 0,
    LumaOnly = 1,
    Combined = 2,
};

struct FeatureInput {
    int width = 0, height = 0;
    const float* depth = nullptr;          // camera z; 0 = none
    const float* position = nullptr;       // optional xyz; position z <= 0 = none
    const uint8_t* luma = nullptr;
    const uint8_t* provenance = nullptr;   // null: depth > 0 is Measured
    float focal = 0, cx = 0, cy = 0;
    int nativeWidth = 0, nativeHeight = 0;
};

struct FeatureMaps {
    std::vector<uint8_t> edgeClass;
    std::vector<uint8_t> normalValid;
    std::vector<float> normal;
    std::vector<uint8_t> centeredValid;
    std::vector<float> centeredNormal;
};

FeatureMaps extractDepthFeatures(const FeatureInput& in, EdgeMode mode,
                                 const DepthFeatureConfig& config = {});

// Drop silhouettes the other view does not also mark within a 3x3. Other
// classes are unchanged. This is not part of the single-view default.
void confirmSilhouettes(std::vector<uint8_t>& edges, const std::vector<uint8_t>& other,
                        int width, int height);

DepthObservation observeDepth(const FeatureInput& in, EdgeMode mode,
                              const DepthFeatureConfig& config = {});

}  // namespace FrameSpatial
