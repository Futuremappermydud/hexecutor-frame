#pragma once
// Shared per-pixel observation for room reconstruction.
// Predicted and missing samples are never measured support. Reliability is a
// weight in [0,1], not a variance (E03).
#include <cstdint>
#include <vector>

namespace FrameSpatial {

enum class DepthProvenance : uint8_t {
    Missing = 0,
    Measured = 1,
    Filtered = 2,
    Predicted = 3,
};

enum class EdgeClass : uint8_t {
    Invalid = 0,
    Interior = 1,
    Appearance = 2,   // luma or shadow; geometry is continuous
    Silhouette = 3,   // depth jump or opening outline
    Crease = 4,       // both sides measured, normals disagree
    Uncertain = 5,
};

struct DepthObservation {
    int width = 0, height = 0;
    int nativeWidth = 0, nativeHeight = 0;  // feature grid; 0 means same as width
    float focal = 0, cx = 0, cy = 0;
    int generation = 0;
    std::vector<float> disparity;
    std::vector<uint8_t> uncertainty;
    std::vector<float> depth;          // camera z, metres; 0 = none
    std::vector<float> position;       // xyz, 3 per pixel; empty if unused
    std::vector<uint8_t> luma;
    std::vector<uint8_t> validity;     // 1 only for measured usable depth
    std::vector<float> reliability;
    std::vector<float> normal;         // xyz, toward the camera; 0 if invalid
    std::vector<uint8_t> normalValid;
    std::vector<uint8_t> edgeClass;
    std::vector<uint8_t> provenance;
};

inline bool measuredSupport(const DepthObservation& o, size_t i) {
    return i < o.provenance.size() && i < o.depth.size()
        && o.provenance[i] == uint8_t(DepthProvenance::Measured)
        && o.depth[i] > 0.f
        && (o.validity.empty() || o.validity[i]);
}

}  // namespace FrameSpatial
