#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <stdexcept>
#include <utility>
#include "depth_pipeline.hpp"
#include "cloud_worker.hpp"
#include "openxr_geometry.hpp"

namespace FrameSpatial {

enum class ViewerMode {
    PointCloud,      // Classic fused stage-space voxels (points)
    LiveMesh,        // Real-time camera surface triangulation (solid shaded triangles)
    PersistentMesh,  // Fused multi-frame stage-space surface mesh
    WireframeMesh    // Wireframe surface mesh
};

inline const char* viewerModeName(ViewerMode mode) {
    switch (mode) {
        case ViewerMode::PointCloud: return "Point Cloud (fused stage-space voxels)";
        case ViewerMode::LiveMesh: return "Live Surface Mesh (power-mean filtered, tear-gated)";
        case ViewerMode::PersistentMesh: return "Persistent Room Mesh (fused multi-frame)";
        case ViewerMode::WireframeMesh: return "Wireframe Mesh";
        default: return "Unknown";
    }
}

// Power-mean (L^p, p=4) separable depth filter inspired by Meta's Zelda depth post-processing.
// Evaluates power-mean accumulation on inverse depth (1/Z) to strongly bias
// smoothed edges toward the foreground surface, preventing background depth bleed.
struct DepthFilterWorkspace {
    std::vector<float> invP, validMask, tempVal, tempWeight, filtered;
};

// Workspace is owned by the depth worker, never shared with the render thread.
// The returned reference remains valid until the next call using this workspace.
inline const std::vector<float>& powerMeanFilterDepth(
    const std::vector<float>& depth, int width, int height, DepthFilterWorkspace& workspace,
    uint32_t p = 4, int radius = 2, float minDepth = 0.25f, float maxDepth = 5.0f) {

    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height) return depth;

    if (radius < 0 || radius > 2) throw std::invalid_argument("depth filter radius must be in [0,2]");

    // Precomputed 1D Gaussian kernel weights (radius 2 -> 5 taps, sigma ~ 1.0)
    constexpr float kKernel[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};

    // Convert valid depth to inverse depth and encode into power-mean space (x^p)
    auto& invP = workspace.invP;
    auto& validMask = workspace.validMask;
    invP.resize(depth.size()); validMask.resize(depth.size());

    #pragma omp parallel for
    for (size_t i = 0; i < depth.size(); ++i) {
        invP[i] = 0.0f;
        validMask[i] = 0.0f;
        float z = depth[i];
        if (std::isfinite(z) && z >= minDepth && z <= maxDepth) {
            float inv = 1.0f / z;
            validMask[i] = 1.0f;
            if (p == 1) invP[i] = inv;
            else if (p == 2) invP[i] = inv * inv;
            else /* p == 4 */ { float sq = inv * inv; invP[i] = sq * sq; }
        }
    }

    // Pass 1: Horizontal Gaussian in power-mean space
    auto& tempVal = workspace.tempVal;
    auto& tempWeight = workspace.tempWeight;
    tempVal.resize(depth.size()); tempWeight.resize(depth.size());

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float vSum = 0.0f;
            float wSum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int nx = x + k;
                if (nx >= 0 && nx < width) {
                    size_t idx = size_t(y) * width + nx;
                    // Invalid samples carry zero value and zero weight.
                    float w = kKernel[k + radius];
                    vSum += invP[idx] * w;
                    wSum += validMask[idx] * w;
                }
            }
            size_t outIdx = size_t(y) * width + x;
            tempVal[outIdx] = vSum;
            tempWeight[outIdx] = wSum;
        }
    }

    // Pass 2: Vertical Gaussian + decode p-th root
    auto& filtered = workspace.filtered;
    filtered.resize(depth.size());

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float vSum = 0.0f;
            float wSum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int ny = y + k;
                if (ny >= 0 && ny < height) {
                    size_t idx = size_t(ny) * width + x;
                    float w = kKernel[k + radius];
                    vSum += tempVal[idx] * w;
                    wSum += tempWeight[idx] * w;
                }
            }
            size_t outIdx = size_t(y) * width + x;
            filtered[outIdx] = 0.0f;
            // Require minimum weight support to produce a valid depth pixel (and inpaint small holes)
            if (wSum >= 0.12f) {
                float avgP = vSum / wSum;
                if (avgP > 1e-12f) {
                    float decodedInv = 0.0f;
                    if (p == 1) decodedInv = avgP;
                    else if (p == 2) decodedInv = std::sqrt(avgP);
                    else /* p == 4 */ decodedInv = std::sqrt(std::sqrt(avgP));

                    if (decodedInv > 1e-4f) {
                        float z = 1.0f / decodedInv;
                        if (z >= minDepth && z <= maxDepth) {
                            filtered[outIdx] = z;
                        }
                    }
                }
            }
        }
    }

    return filtered;
}

inline std::vector<float> powerMeanFilterDepth(
    const std::vector<float>& depth, int width, int height,
    uint32_t p = 4, int radius = 2, float minDepth = 0.25f, float maxDepth = 5.0f) {
    DepthFilterWorkspace workspace;
    powerMeanFilterDepth(depth, width, height, workspace, p, radius, minDepth, maxDepth);
    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height) return depth;
    return std::move(workspace.filtered);
}

// Generate continuous 3D surface mesh triangles with depth-discontinuity tear gating.
// Prevents "rubber sheeting" across foreground/background silhouettes.
inline void triangulateDepthMap(
    const std::vector<float>& depth,
    const std::vector<uint8_t>& luma,
    const DepthPipeline& rectifier,
    const XrPosef& headPose,
    int width, int height,
    std::vector<FrameSpatialCloud::PointVertex>& vertices,
    int step = 2,
    float maxRelDepthJump = 0.045f,
    float maxAbsDepthJump = 0.05f) {

    vertices.clear();
    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height) return;
    if (step <= 0) throw std::invalid_argument("mesh step must be positive");
    if (!luma.empty() && luma.size() != depth.size())
        throw std::invalid_argument("mesh luma dimensions do not match depth");

    vertices.reserve(size_t(width / step) * (height / step) * 6);

    const auto rotateVector = [](const XrQuaternionf& q, const std::array<float, 3>& v) -> std::array<float, 3> {
        const float tx = 2.0f * (q.y * v[2] - q.z * v[1]);
        const float ty = 2.0f * (q.z * v[0] - q.x * v[2]);
        const float tz = 2.0f * (q.x * v[1] - q.y * v[0]);
        return {
            v[0] + q.w * tx + (q.y * tz - q.z * ty),
            v[1] + q.w * ty + (q.z * tx - q.x * tz),
            v[2] + q.w * tz + (q.x * ty - q.y * tx),
        };
    };

    const auto unprojectToStage = [&](int x, int y, float z) -> std::array<float, 3> {
        const auto inHead = rectifier.rectifiedPointInHead(x, y, z);
        const auto rotated = rotateVector(headPose.orientation, inHead);
        return {
            headPose.position.x + rotated[0],
            headPose.position.y + rotated[1],
            headPose.position.z + rotated[2]
        };
    };

    // Each grid point is shared by up to six triangles. Keep two lazily
    // projected rows, so tear-rejected triangles still avoid projection work.
    const size_t columns = (size_t(width) - 1) / step + 1;
    std::vector<std::array<float, 3>> projected[2];
    std::vector<uint8_t> projectedValid[2];
    for (int row = 0; row < 2; ++row) {
        projected[row].resize(columns);
        projectedValid[row].assign(columns, 0);
    }
    const auto projectedPoint = [&](int x, int y, float z) -> const std::array<float, 3>& {
        const size_t row = size_t(y / step) & 1, col = size_t(x / step);
        if (!projectedValid[row][col]) {
            projected[row][col] = unprojectToStage(x, y, z);
            projectedValid[row][col] = 1;
        }
        return projected[row][col];
    };

    const auto makeColor = [](uint8_t lumaVal, const std::array<float, 3>& norm) -> uint32_t {
        // Key light from upper-front
        constexpr float lx = 0.2f, ly = 0.85f, lz = 0.45f;
        float diff = std::max(0.0f, norm[0]*lx + norm[1]*ly + norm[2]*lz);
        float ambient = 0.35f;
        float light = std::clamp(ambient + 0.65f * diff, 0.25f, 1.25f);

        // Normal-based tint: floors/ceilings (high |Ny|) get subtle cyan tint, walls get warm slate tint
        float floorness = std::abs(norm[1]);
        float wallness = 1.0f - floorness;

        int r = std::clamp(int(lumaVal * (0.80f + 0.20f * wallness) * light), 15, 255);
        int g = std::clamp(int(lumaVal * (0.85f + 0.15f * floorness) * light), 15, 255);
        int b = std::clamp(int(lumaVal * (0.95f + 0.25f * floorness) * light), 15, 255);
        return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | 0xff000000u;
    };

    auto addTriangle = [&](int x0, int y0, float z0, uint8_t l0,
                           int x1, int y1, float z1, uint8_t l1,
                           int x2, int y2, float z2, uint8_t l2) {
        float zMin = std::min({z0, z1, z2});
        float zMax = std::max({z0, z1, z2});
        float tearThreshold = std::max(maxAbsDepthJump, zMin * maxRelDepthJump);
        if (zMax - zMin > tearThreshold) return;

        const auto& v0 = projectedPoint(x0, y0, z0);
        const auto& v1 = projectedPoint(x1, y1, z1);
        const auto& v2 = projectedPoint(x2, y2, z2);

        // Triangle face normal
        std::array<float, 3> d1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        std::array<float, 3> d2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        std::array<float, 3> norm = {
            d1[1] * d2[2] - d1[2] * d2[1],
            d1[2] * d2[0] - d1[0] * d2[2],
            d1[0] * d2[1] - d1[1] * d2[0]
        };
        float normLen = std::sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
        if (normLen > 1e-6f) {
            norm[0] /= normLen; norm[1] /= normLen; norm[2] /= normLen;
        } else {
            norm = {0.0f, 1.0f, 0.0f};
        }

        uint32_t c0 = makeColor(l0, norm);
        uint32_t c1 = makeColor(l1, norm);
        uint32_t c2 = makeColor(l2, norm);

        vertices.push_back({v0[0], v0[1], v0[2], c0, 1.0f});
        vertices.push_back({v1[0], v1[1], v1[2], c1, 1.0f});
        vertices.push_back({v2[0], v2[1], v2[2], c2, 1.0f});
    };

    for (int y = 0; y + step < height; y += step) {
        // The top row was the preceding strip's bottom row; only the new
        // bottom row needs invalidating as its ring-buffer slot is reused.
        auto& valid = projectedValid[size_t(y / step + 1) & 1];
        std::fill(valid.begin(), valid.end(), uint8_t(0));
        for (int x = 0; x + step < width; x += step) {
            float z00 = depth[size_t(y) * width + x];
            float z10 = depth[size_t(y) * width + (x + step)];
            float z01 = depth[size_t(y + step) * width + x];
            float z11 = depth[size_t(y + step) * width + (x + step)];

            bool v00Valid = (z00 > 0.0f);
            bool v10Valid = (z10 > 0.0f);
            bool v01Valid = (z01 > 0.0f);
            bool v11Valid = (z11 > 0.0f);

            int validCount = int(v00Valid) + int(v10Valid) + int(v01Valid) + int(v11Valid);
            if (validCount < 3) continue;

            uint8_t l00 = (luma.empty() || !v00Valid) ? 180 : luma[size_t(y) * width + x];
            uint8_t l10 = (luma.empty() || !v10Valid) ? 180 : luma[size_t(y) * width + (x + step)];
            uint8_t l01 = (luma.empty() || !v01Valid) ? 180 : luma[size_t(y + step) * width + x];
            uint8_t l11 = (luma.empty() || !v11Valid) ? 180 : luma[size_t(y + step) * width + (x + step)];

            if (v00Valid && v10Valid && v11Valid) {
                addTriangle(x, y, z00, l00,  x + step, y, z10, l10,  x + step, y + step, z11, l11);
            }
            if (v00Valid && v11Valid && v01Valid) {
                addTriangle(x, y, z00, l00,  x + step, y + step, z11, l11,  x, y + step, z01, l01);
            } else if (!v00Valid && v10Valid && v11Valid && v01Valid) {
                // Diagonal alternate if top-left is missing
                addTriangle(x + step, y, z10, l10,  x + step, y + step, z11, l11,  x, y + step, z01, l01);
            } else if (!v11Valid && v00Valid && v10Valid && v01Valid) {
                // Diagonal alternate if bottom-right is missing
                addTriangle(x, y, z00, l00,  x + step, y, z10, l10,  x, y + step, z01, l01);
            }
        }
    }
}

inline std::vector<FrameSpatialCloud::PointVertex> triangulateDepthMap(
    const std::vector<float>& depth, const std::vector<uint8_t>& luma,
    const DepthPipeline& rectifier, const XrPosef& headPose, int width, int height,
    int step = 2, float maxRelDepthJump = 0.045f, float maxAbsDepthJump = 0.05f) {
    std::vector<FrameSpatialCloud::PointVertex> vertices;
    triangulateDepthMap(depth, luma, rectifier, headPose, width, height, vertices,
                        step, maxRelDepthJump, maxAbsDepthJump);
    return vertices;
}

} // namespace FrameSpatial
