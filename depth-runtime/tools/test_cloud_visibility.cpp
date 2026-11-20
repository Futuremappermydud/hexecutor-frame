#include "cloud_visibility.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main() {
    using namespace FrameSpatial;
    constexpr int width = 12, height = 12;
    constexpr float fb = 24.282608f;
    std::vector<float> depth(width * height, 0.0f);
    auto evidence = [&] (float z = 1.0f) {
        return classifyCloudVisibility(depth, width, height, 6, 6, z, fb);
    };
    CloudAging aging;
    // A highlight/no stereo depth must not clear the table.
    assert(evidence() == CloudEvidence::Unknown);
    for (int i = 0; i < 30; ++i) aging.observe(evidence(), 0.1f);
    assert(!aging.expired());
    depth[6 * width + 6] = std::numeric_limits<float>::quiet_NaN();
    assert(evidence() == CloudEvidence::Unknown);
    // One far mismatch cannot remove an object.
    depth[6 * width + 6] = 2.0f;
    assert(evidence() == CloudEvidence::Unknown);
    // When the object moves away, supported background clears its old points.
    for (int y = 4; y <= 8; ++y)
        for (int x = 4; x <= 8; ++x) depth[y * width + x] = 2.0f;
    assert(evidence() == CloudEvidence::FreeSpace);
    aging.observe(evidence(), 10.0f); // A paused capture is not ten seconds of evidence.
    assert(!aging.expired());
    for (int i = 0; i < 4; ++i) aging.observe(evidence(), 0.1f);
    assert(!aging.expired());
    for (int i = 0; i < 5; ++i) aging.observe(evidence(), 0.1f);
    assert(aging.expired());
    // Seeing the object again cancels removal, even beside distant background.
    depth[6 * width + 6] = 1.03f;
    assert(evidence() == CloudEvidence::Confirmed);
    aging.observe(evidence(), 0.1f);
    assert(!aging.expired() && aging.freeSpaceChecks == 0);
    // A foreground occluder protects the old surface behind it.
    depth[6 * width + 6] = 0.5f;
    assert(evidence() == CloudEvidence::Occluded);
    aging.observe(CloudEvidence::FreeSpace, 0.2f);
    aging.observe(evidence(), 0.1f);
    assert(aging.freeSpaceSeconds == 0.0f);
    // Out of view/border points and distant stereo noise are protected.
    assert(classifyCloudVisibility(depth, width, height, -1, 6, 1, fb) == CloudEvidence::Unknown);
    std::fill(depth.begin(), depth.end(), 4.4f);
    assert(evidence(4.0f) == CloudEvidence::Confirmed);
    std::cout << "Cloud visibility and moving-object aging checks passed\n";
}
