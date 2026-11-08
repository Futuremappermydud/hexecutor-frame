#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace FrameSpatial {

// Scratch belongs to one depth worker and retains capacity between frames.
// Reject disconnected stereo patches before they can accumulate in world space.
// Use eight-connectivity to preserve diagonal/slanted surfaces. Never fill holes.
struct DepthSpeckleWorkspace {
    std::vector<unsigned char> visited;
    std::vector<size_t> component;
};
inline void removeDepthSpeckles(std::vector<float>& depth, int width, int height,
                               DepthSpeckleWorkspace& workspace, size_t minimumPixels = 32) {
    if (width <= 0 || height <= 0 ||
        depth.size() != static_cast<size_t>(width) * height) return;
    if (minimumPixels <= 1) return;
    auto& visited = workspace.visited;
    auto& component = workspace.component;
    visited.assign(depth.size(), 0);
    component.reserve(depth.size());
    for (size_t seed = 0; seed < depth.size(); ++seed) {
        if (visited[seed] || !std::isfinite(depth[seed]) || depth[seed] <= 0.0f) continue;
        component.clear();
        component.push_back(seed);
        visited[seed] = 1;
        for (size_t next = 0; next < component.size(); ++next) {
            const size_t index = component[next];
            const int x = index % width;
            const int y = index / width;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    const size_t neighbor = static_cast<size_t>(ny) * width + nx;
                    if (visited[neighbor] || !std::isfinite(depth[neighbor]) ||
                        depth[neighbor] <= 0.0f) continue;
                    const float tolerance = std::max(0.04f,
                        std::min(depth[index], depth[neighbor]) * 0.04f);
                    if (std::abs(depth[index] - depth[neighbor]) > tolerance) continue;
                    visited[neighbor] = 1;
                    component.push_back(neighbor);
                }
            }
        }
        if (component.size() < minimumPixels) {
            for (size_t index : component) depth[index] = 0.0f;
        }
    }
}

inline void removeDepthSpeckles(std::vector<float>& depth, int width, int height,
                               size_t minimumPixels = 32) {
    DepthSpeckleWorkspace workspace;
    removeDepthSpeckles(depth, width, height, workspace, minimumPixels);
}

} // namespace FrameSpatial
