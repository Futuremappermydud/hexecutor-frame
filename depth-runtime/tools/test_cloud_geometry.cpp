#include "depth_pipeline.hpp"
#include "depth_speckle_filter.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    // Keep a sizeable plane, discard isolated/adjacent mismatches and a small
    // foreground patch touching the plane across a depth discontinuity.
    std::vector<float> depth(20 * 20, 0.0f);
    for (int y = 2; y < 10; ++y)
        for (int x = 2; x < 10; ++x) depth[y * 20 + x] = 2.0f;
    depth[2 * 20 + 10] = 0.5f;
    depth[15 * 20 + 15] = depth[15 * 20 + 16] = 1.0f;
    depth[0] = std::numeric_limits<float>::quiet_NaN();
    FrameSpatial::removeDepthSpeckles(depth, 20, 20);
    assert(depth[5 * 20 + 5] == 2.0f);
    assert(depth[2 * 20 + 10] == 0.0f);
    assert(depth[15 * 20 + 15] == 0.0f && depth[15 * 20 + 16] == 0.0f);

    FrameSpatial::DepthPipeline pipeline;
    pipeline.initialize();
    auto origin = pipeline.rectifiedPointInHead(308, 308, 0.0f);
    auto forward = pipeline.rectifiedPointInHead(308, 308, 1.0f);
    auto right = pipeline.rectifiedPointInHead(688, 308, 1.0f);
    auto down = pipeline.rectifiedPointInHead(308, 688, 1.0f);
    // Visibility purging uses this inverse; check off-axis metric backprojection.
    const auto inverse = pipeline.headPointInRectified(right);
    assert(std::abs(inverse[0] - 1.0f) < 1e-4f);
    assert(std::abs(inverse[1]) < 1e-4f);
    assert(std::abs(inverse[2] - 1.0f) < 1e-4f);
    // This rig's cameras look slightly down, and the left optical center is
    // left of the head origin. These checks catch the old axis-flip/midpoint map.
    assert(origin[0] < -0.02f && origin[0] > -0.05f);
    assert(origin[2] < -0.04f && origin[2] > -0.08f);
    assert(forward[1] - origin[1] < -0.05f);
    assert(forward[2] - origin[2] < -0.9f);
    assert(right[0] - forward[0] > 0.95f);
    assert(down[1] - forward[1] < -0.9f);
    float lengthSquared = 0;
    for (int i = 0; i < 3; ++i) lengthSquared += std::pow(forward[i] - origin[i], 2);
    assert(std::abs(lengthSquared - 1.0f) < 1e-4f);
    std::cout << "Cloud geometry and speckle regression checks passed\n";
}
