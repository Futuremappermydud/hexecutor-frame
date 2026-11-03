#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace FrameSpatial {

enum class CloudEvidence { Unknown, Confirmed, Occluded, FreeSpace };

// Stereo uncertainty grows quadratically with distance. A conservative disparity
// error allowance avoids treating normal depth jitter as a disappearing object.
inline CloudEvidence classifyCloudVisibility(const std::vector<float>& depth,
                                             int width, int height,
                                             int x, int y, float z,
                                             float focalBaseline) {
    if (width < 5 || height < 5 || depth.size() != static_cast<size_t>(width) * height ||
        x < 2 || x >= width - 2 || y < 2 || y >= height - 2 ||
        !std::isfinite(z) || z <= 0.0f || focalBaseline <= 0.0f)
        return CloudEvidence::Unknown;
    const float tolerance = std::max(0.06f,
        std::max(z * 0.03f, 0.75f * z * z / focalBaseline));
    int agreeing = 0, nearer = 0, farther = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const float measured = depth[(y + dy) * width + x + dx];
            if (!std::isfinite(measured) || measured <= 0.0f) continue;
            if (std::abs(measured - z) <= tolerance) ++agreeing;
            else if (measured < z - tolerance) ++nearer;
            else ++farther;
        }
    }
    if (agreeing > 0) return CloudEvidence::Confirmed;
    if (nearer > 0) return CloudEvidence::Occluded;
    // Require a measured patch entirely behind the old point. Sparse/no depth
    // (including highlights) is not evidence of empty space.
    return farther >= 6 ? CloudEvidence::FreeSpace : CloudEvidence::Unknown;
}

struct CloudAging {
    float freeSpaceSeconds = 0.0f;
    uint32_t freeSpaceChecks = 0;

    void observe(CloudEvidence evidence, float elapsedSeconds) {
        if (evidence == CloudEvidence::Confirmed || evidence == CloudEvidence::Occluded) {
            reset();
        } else if (evidence == CloudEvidence::FreeSpace) {
            // A long capture/tracking gap cannot count as continuous evidence.
            freeSpaceSeconds += std::clamp(elapsedSeconds, 0.0f, 0.2f);
            ++freeSpaceChecks;
        }
    }
    void reset() { freeSpaceSeconds = 0.0f; freeSpaceChecks = 0; }
    bool expired() const { return freeSpaceSeconds >= 1.0f && freeSpaceChecks >= 5; }
};

} // namespace FrameSpatial
