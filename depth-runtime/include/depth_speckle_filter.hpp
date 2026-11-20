#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include "frame_omp.hpp"

namespace FrameSpatial {

// Scratch belongs to one depth worker and retains capacity between frames.
// Reject disconnected stereo patches before they can accumulate in world space.
// Use eight-connectivity to preserve diagonal/slanted surfaces. Never fill holes.
struct DepthSpeckleWorkspace {
    std::vector<uint32_t> label;   // per pixel, 0 = invalid; labels are 1-based, unique per band
    std::vector<uint32_t> parent;  // union-find forest; a root is its smallest member label
    std::vector<uint32_t> size;    // pixels per root
    std::vector<unsigned char> small;
};
inline void removeDepthSpeckles(std::vector<float>& depth, int width, int height,
                               DepthSpeckleWorkspace& workspace, size_t minimumPixels = 32) {
    if (width <= 0 || height <= 0 ||
        depth.size() != static_cast<size_t>(width) * height) return;
    if (minimumPixels <= 1) return;
    const size_t pixels = depth.size();
    if (pixels >= UINT32_MAX) return;
    auto& labels = workspace.label;
    auto& parent = workspace.parent;
    auto& size = workspace.size;
    auto& small = workspace.small;
    if (labels.size() < pixels) labels.resize(pixels);
    if (parent.size() < pixels + 1) { parent.resize(pixels + 1); size.resize(pixels + 1); small.resize(pixels + 1); }
    uint32_t* const lab = labels.data();
    uint32_t* const par = parent.data();
    uint32_t* const cnt = size.data();
    const float* const z = depth.data();
    const float infinity = std::numeric_limits<float>::infinity();
    auto find = [par](uint32_t a) {
        while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; }
        return a;
    };
    // Joins two labels' components; returns the surviving root.
    auto unite = [&](uint32_t a, uint32_t b) {
        const uint32_t ra = find(a), rb = find(b);
        if (ra == rb) return ra;
        if (ra < rb) { par[rb] = ra; cnt[ra] += cnt[rb]; return ra; }
        par[ra] = rb; cnt[rb] += cnt[ra]; return rb;
    };
    // Adjacent pixels are linked when both are valid and within the tolerance;
    // the expression is symmetric in its operands, so the order is free.
    auto linked = [z](size_t index, size_t neighbor) {
        const float a = z[index], b = z[neighbor];
        return std::abs(a - b) <= std::max(0.04f, std::min(a, b) * 0.04f);
    };
    // Rows are split into bands labelled independently (label ids of band b lie in
    // (first pixel of band, last pixel of band], so they ascend with the band);
    // bands are then joined along their boundary rows.
    const int bands = pixels >= 65536 && height >= 64 ? 4 : 1;
    std::array<int, 5> start{};
    std::array<uint32_t, 4> used{};
    for (int b = 0; b <= bands; ++b) start[b] = static_cast<int>(static_cast<long long>(height) * b / bands);
    FRAME_OMP_FOR
    for (int b = 0; b < bands; ++b) {
        uint32_t last = static_cast<uint32_t>(start[b]) * width;
        // Raster order; each pixel meets its earlier 8-neighbours (W, NW, N, NE) once.
        for (int y = start[b]; y < start[b + 1]; ++y) {
            const size_t row = static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                const size_t index = row + x;
                const float a = z[index];
                if (!(a > 0.0f && a < infinity)) { lab[index] = 0; continue; }
                uint32_t current = 0;
                auto link = [&](size_t neighbor) {
                    const uint32_t other = lab[neighbor];
                    if (!other || other == current || !linked(index, neighbor)) return;
                    current = current ? unite(current, other) : other;
                };
                if (x > 0) link(index - 1);
                if (y > start[b]) {
                    const size_t up = index - width;
                    if (x > 0) link(up - 1);
                    link(up);
                    if (x + 1 < width) link(up + 1);
                }
                if (!current) {
                    current = ++last;
                    par[current] = current;
                    cnt[current] = 1;
                } else {
                    ++cnt[find(current)];
                }
                lab[index] = current;
            }
        }
        used[b] = last;
    }
    for (int b = 1; b < bands; ++b) {
        const size_t row = static_cast<size_t>(start[b]) * width;
        for (int x = 0; x < width; ++x) {
            const size_t index = row + x;
            const uint32_t own = lab[index];
            if (!own) continue;
            const size_t up = index - width;
            for (int dx = std::max(-1, -x); dx <= (x + 1 < width ? 1 : 0); ++dx)
                if (lab[up + dx] && linked(index, up + dx)) unite(own, lab[up + dx]);
        }
    }
    // Parents always have smaller labels, so one ascending pass resolves every root.
    bool any = false;
    small[0] = 0;
    for (int b = 0; b < bands; ++b)
        for (uint32_t l = static_cast<uint32_t>(start[b]) * width + 1; l <= used[b]; ++l) {
            par[l] = par[par[l]];
            small[l] = static_cast<size_t>(cnt[par[l]]) < minimumPixels;
            any |= small[l] != 0;
        }
    if (!any) return;
    float* const out = depth.data();
    FRAME_OMP_FOR
    for (int b = 0; b < bands; ++b)
        for (size_t index = static_cast<size_t>(start[b]) * width, e = static_cast<size_t>(start[b + 1]) * width;
             index < e; ++index)
            if (small[lab[index]]) out[index] = 0.0f;
}

inline void removeDepthSpeckles(std::vector<float>& depth, int width, int height,
                               size_t minimumPixels = 32) {
    DepthSpeckleWorkspace workspace;
    removeDepthSpeckles(depth, width, height, workspace, minimumPixels);
}

} // namespace FrameSpatial
