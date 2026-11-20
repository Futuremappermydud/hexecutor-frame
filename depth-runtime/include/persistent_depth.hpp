#pragma once

#include <cstdint>
#include <unordered_map>

namespace FrameSpatial {

// Four-path semi-global matching on a rectified luma pair.
// Invalid pixels are 0. Metric Z is focalBaseline / disparity.
// Disparity search is 1..63, so the near limit at focal*baseline 24.3 is
// about 0.38 m. The path is meant for surfaces past that, where census
// winner-take-all stopped being metric.
void matchSgbm(const uint8_t* left, const uint8_t* right, int width, int height,
               float focalBaseline, float* depth);

// World-space points that stay after the camera looks away.
// A confirmed cell is kept when a later frame does not see it.
// A new measurement replaces a cell only inside the distance-dependent band
// mu = max(minBandM, z^2 * disparitySigmaPx / focalBaseline).
struct PersistentCell {
    float x = 0;
    float y = 0;
    float z = 0;
    int observations = 0;
};

struct PersistentKey {
    int x = 0;
    int y = 0;
    int z = 0;
    bool operator==(const PersistentKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct PersistentKeyHash {
    size_t operator()(const PersistentKey& key) const {
        size_t h = static_cast<size_t>(key.x) * 73856093u;
        h ^= static_cast<size_t>(key.y) * 19349663u;
        h ^= static_cast<size_t>(key.z) * 83492791u;
        return h;
    }
};

class PersistentMap {
public:
    void integrate(const float* depth, int width, int height, float focalPx,
                   float principalX, float principalY, float focalBaseline,
                   float cameraX, float cameraY, float cameraZ, float yawRad);
    const std::unordered_map<PersistentKey, PersistentCell, PersistentKeyHash>&
    cells() const { return cells_; }
    int confirmed(int minimumObservations) const;

private:
    std::unordered_map<PersistentKey, PersistentCell, PersistentKeyHash> cells_;
};

}  // namespace FrameSpatial
