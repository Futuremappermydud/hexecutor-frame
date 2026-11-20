// Independent scalar spherical oracle: preserve gates, support and projection.
#include "model_depth_geometry.hpp"
#include "depth_speckle_filter.hpp"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
using namespace FrameSpatial;

static std::vector<float> reference(const std::vector<float>& disparity,
    const std::vector<uint8_t>& uncertainty, const ModelSphericalMap& map) {
    std::vector<float> rho(disparity.size()), supported(disparity.size());
    for (int y = 2; y < MODEL_HEIGHT - 2; ++y) for (int x = 2; x < MODEL_WIDTH - 2; ++x) {
        const size_t i = size_t(y) * MODEL_WIDTH + x;
        const float d = disparity[i];
        if (!std::isfinite(d) || d <= 0 || uncertainty[i] >= 89 || x - d < 2) continue;
        const float a = (x - (MODEL_WIDTH - 1) * .5f) * map.step;
        const float b = a - d * map.step;
        if (a > 1.5f || b < -1.5f) continue;
        const float gap = std::tan(a) - std::tan(b);
        if (!(gap > 0)) continue;
        const float r = map.baseline / gap;
        const float z = r * std::cos((y - (MODEL_HEIGHT - 1) * .5f) * map.step);
        if (z >= .25f && z <= 5.f) rho[i] = r;
    }
    for (int y = 2; y < MODEL_HEIGHT - 2; ++y) for (int x = 2; x < MODEL_WIDTH - 2; ++x) {
        const size_t i = size_t(y) * MODEL_WIDTH + x;
        const float r = rho[i];
        if (!(r > 0)) continue;
        int support = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const float n = rho[size_t(y + dy) * MODEL_WIDTH + x + dx];
            support += n > 0 && std::abs(n - r) <= std::max(.04f, .04f * r);
        }
        if (support >= 3) supported[i] = r;
    }
    std::vector<float> result(size_t(map.width) * map.height);
    for (size_t i = 0; i < result.size(); ++i) {
        const int m = map.modelIndex[i];
        if (m >= 0 && supported[m] > 0) result[i] = supported[m] * map.cosElevation[i];
    }
    return result;
}

static void referenceSpeckles(std::vector<float>& depth, int width, int height,
                               size_t minimumPixels = 32) {
    if (width <= 0 || height <= 0 ||
        depth.size() != static_cast<size_t>(width) * height) return;
    std::vector<unsigned char> visited(depth.size(), 0);
    std::vector<size_t> component;
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

static void checkSpeckles() {
    std::mt19937 rng(817);
    DepthSpeckleWorkspace workspace;
    for (int width : {1,2,13,64}) for (int height : {1,7,39})
        for (size_t minimum : {size_t(0),size_t(1),size_t(2),size_t(8),size_t(32),size_t(128)}) {
            std::vector<float> depth(size_t(width)*height);
            for (float& z : depth) {
                z = 1.f + float(rng()%4)*.039f;
                switch (rng()%11) {
                    case 0: z=0; break;
                    case 1: z=-1; break;
                    case 2: z=std::numeric_limits<float>::quiet_NaN(); break;
                    case 3: z=std::numeric_limits<float>::infinity(); break;
                    default: break;
                }
            }
            auto expected=depth;
            referenceSpeckles(expected,width,height,minimum);
            removeDepthSpeckles(depth,width,height,workspace,minimum);
            assert(std::memcmp(depth.data(),expected.data(),depth.size()*sizeof(float))==0);
        }
    std::vector<float> diagonal(36,0);
    for(int i=0;i<6;++i) diagonal[i*6+i]=2.f;
    auto expected=diagonal;
    removeDepthSpeckles(diagonal,6,6,workspace,6);
    assert(diagonal==expected);
    removeDepthSpeckles(diagonal,6,6,workspace,7);
    for(float z:diagonal) assert(z==0);
}

int main() {
    checkSpeckles();
    std::mt19937 rng(1147);
    for (const float degrees : {.29f, .42f}) {
        const auto map = modelSphericalMap(degrees * .01745329252f, .075f, 616, 616, 380, 308, 308);
        std::vector<float> disparity(MODEL_WIDTH * MODEL_HEIGHT);
        std::vector<uint8_t> uncertainty(disparity.size(), 30);
        for (size_t i = 0; i < disparity.size(); ++i) {
            // Mostly coherent geometry, plus isolated outliers and invalid heads.
            disparity[i] = 7.f + float(rng() % 100) * .001f;
            switch (rng() % 37) {
                case 0: disparity[i] = std::numeric_limits<float>::quiet_NaN(); break;
                case 1: disparity[i] = std::numeric_limits<float>::infinity(); break;
                case 2: disparity[i] = 0; break;
                case 3: disparity[i] = -2; break;
                case 4: disparity[i] = 100; break;
                case 5: disparity[i] = .00001f; break;
                case 6: uncertainty[i] = 89; break;
                case 7: uncertainty[i] = 255; break;
                default: break;
            }
        }
        const auto expected = reference(disparity, uncertainty, map);
        const auto actual = modelSphericalDepth(disparity, uncertainty, map);
        assert(actual.size() == expected.size());
        size_t valid = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            assert(std::isfinite(actual[i]));
            assert(actual[i] == expected[i]);
            valid += actual[i] > 0;
        }
        assert(valid > 10000 && valid < actual.size());
        std::fill(uncertainty.begin(), uncertainty.end(), 89);
        for (float z : modelSphericalDepth(disparity, uncertainty, map)) assert(z == 0);
    }
    std::cout << "Depth postprocess PASS: scalar spherical and speckle identity, range, nonfinite, uncertainty, support and projection\n";
}
