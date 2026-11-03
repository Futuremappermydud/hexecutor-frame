#include "persistent_depth.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace FrameSpatial {
namespace {

constexpr int kDisparities = 64;
constexpr int kBlock = 5;
constexpr int kP1 = 8 * kBlock * kBlock;
constexpr int kP2 = 32 * kBlock * kBlock;
constexpr int kPrefilterCap = 63;

void accumulateHorizontal(const uint16_t* cost, uint32_t* summed, int width, int height,
                          bool leftToRight) {
    const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
    const int rows = std::max(1, height / int(workers));
    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < workers; ++worker) {
        const int y0 = int(worker) * rows;
        const int y1 = worker + 1 == workers ? height : std::min(height, y0 + rows);
        if (y0 >= y1) continue;
        threads.emplace_back([=] {
            std::vector<uint32_t> previous(kDisparities, 0);
            std::vector<uint32_t> current(kDisparities, 0);
            for (int y = y0; y < y1; ++y) {
                const int xBegin = leftToRight ? 0 : width - 1;
                const int xEnd = leftToRight ? width : -1;
                const int step = leftToRight ? 1 : -1;
                bool first = true;
                for (int x = xBegin; x != xEnd; x += step) {
                    const size_t base = (size_t(y) * width + x) * kDisparities;
                    if (first) {
                        for (int d = 0; d < kDisparities; ++d) {
                            previous[d] = cost[base + d];
                            summed[base + d] += previous[d];
                        }
                        first = false;
                        continue;
                    }
                    uint32_t minimum = previous[0];
                    for (int d = 1; d < kDisparities; ++d)
                        minimum = std::min(minimum, previous[d]);
                    for (int d = 0; d < kDisparities; ++d) {
                        uint32_t best = previous[d];
                        if (d > 0) best = std::min(best, previous[d - 1] + kP1);
                        if (d + 1 < kDisparities) best = std::min(best, previous[d + 1] + kP1);
                        best = std::min(best, minimum + uint32_t(kP2));
                        current[d] = cost[base + d] + best - minimum;
                        summed[base + d] += current[d];
                    }
                    previous.swap(current);
                }
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
}

void accumulateVertical(const uint16_t* cost, uint32_t* summed, int width, int height,
                        bool topToBottom) {
    std::vector<uint32_t> previous(size_t(width) * kDisparities, 0);
    std::vector<uint32_t> current(size_t(width) * kDisparities, 0);
    const int yBegin = topToBottom ? 0 : height - 1;
    const int yEnd = topToBottom ? height : -1;
    const int step = topToBottom ? 1 : -1;
    bool first = true;
    for (int y = yBegin; y != yEnd; y += step) {
        for (int x = 0; x < width; ++x) {
            const size_t base = (size_t(y) * width + x) * kDisparities;
            const size_t prior = size_t(x) * kDisparities;
            if (first) {
                for (int d = 0; d < kDisparities; ++d) {
                    current[prior + d] = cost[base + d];
                    summed[base + d] += current[prior + d];
                }
                continue;
            }
            uint32_t minimum = previous[prior];
            for (int d = 1; d < kDisparities; ++d)
                minimum = std::min(minimum, previous[prior + d]);
            for (int d = 0; d < kDisparities; ++d) {
                uint32_t best = previous[prior + d];
                if (d > 0) best = std::min(best, previous[prior + d - 1] + kP1);
                if (d + 1 < kDisparities) best = std::min(best, previous[prior + d + 1] + kP1);
                best = std::min(best, minimum + uint32_t(kP2));
                current[prior + d] = cost[base + d] + best - minimum;
                summed[base + d] += current[prior + d];
            }
        }
        previous.swap(current);
        first = false;
    }
}

void accumulateDiagonal(const uint16_t* cost, uint32_t* summed, int width, int height,
                       int xStep, int yStep) {
    std::vector<uint32_t> previous(size_t(width) * kDisparities, 0);
    std::vector<uint32_t> current(size_t(width) * kDisparities, 0);
    const int yBegin = yStep > 0 ? 0 : height - 1;
    const int yEnd = yStep > 0 ? height : -1;
    bool first = true;
    for (int y = yBegin; y != yEnd; y += yStep) {
        for (int x = 0; x < width; ++x) {
            const size_t base = (size_t(y) * width + x) * kDisparities;
            const int priorX = x - xStep;
            if (first || priorX < 0 || priorX >= width) {
                for (int d = 0; d < kDisparities; ++d) {
                    current[size_t(x) * kDisparities + d] = cost[base + d];
                    summed[base + d] += cost[base + d];
                }
                continue;
            }
            const size_t prior = size_t(priorX) * kDisparities;
            uint32_t minimum = previous[prior];
            for (int d = 1; d < kDisparities; ++d)
                minimum = std::min(minimum, previous[prior + d]);
            for (int d = 0; d < kDisparities; ++d) {
                uint32_t best = previous[prior + d];
                if (d > 0) best = std::min(best, previous[prior + d - 1] + kP1);
                if (d + 1 < kDisparities) best = std::min(best, previous[prior + d + 1] + kP1);
                best = std::min(best, minimum + uint32_t(kP2));
                current[size_t(x) * kDisparities + d] = cost[base + d] + best - minimum;
                summed[base + d] += current[size_t(x) * kDisparities + d];
            }
        }
        previous.swap(current);
        first = false;
    }
}

void removeSpeckles(float* depth, int width, int height, float focalBaseline) {
    const size_t count = size_t(width) * height;
    std::vector<int> labels(count, -1);
    std::vector<int> stack;
    int component = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int start = y * width + x;
            if (labels[start] >= 0 || !(depth[start] > 0)) continue;
            stack.clear();
            stack.push_back(start);
            labels[start] = component;
            std::vector<int> members;
            while (!stack.empty()) {
                const int current = stack.back();
                stack.pop_back();
                members.push_back(current);
                const int cx = current % width;
                const int cy = current / width;
                const float currentDisparity = focalBaseline / depth[current];
                const int neighbors[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (const auto& offset : neighbors) {
                    const int nx = cx + offset[0];
                    const int ny = cy + offset[1];
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                    const int next = ny * width + nx;
                    if (labels[next] >= 0 || !(depth[next] > 0)) continue;
                    if (std::fabs(focalBaseline / depth[next] - currentDisparity) > 2.0f) continue;
                    labels[next] = component;
                    stack.push_back(next);
                }
            }
            if (members.size() < 100) {
                for (int index : members) depth[index] = 0;
            }
            ++component;
        }
    }
}

}  // namespace

void matchSgbm(const uint8_t* left, const uint8_t* right, int width, int height,
               float focalBaseline, float* depth) {
    const size_t pixels = size_t(width) * height;
    std::fill(depth, depth + pixels, 0.0f);
    if (width < 16 || height < 16 || !(focalBaseline > 0)) return;

    std::vector<int16_t> leftFiltered(pixels, 0), rightFiltered(pixels, 0);
    auto prefilter = [](const uint8_t* source, int16_t* dest, int width, int height) {
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int gradient = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const uint8_t* row = source + size_t(y + dy) * width;
                    gradient += int(row[x + 1]) - int(row[x - 1]);
                }
                dest[size_t(y) * width + x] =
                    static_cast<int16_t>(std::clamp(gradient, -kPrefilterCap, kPrefilterCap));
            }
        }
    };
    prefilter(left, leftFiltered.data(), width, height);
    prefilter(right, rightFiltered.data(), width, height);

    std::vector<uint16_t> cost(pixels * kDisparities, 4000);
    {
        const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
        const int span = std::max(1, (kDisparities - 1) / int(workers));
        std::vector<std::thread> threads;
        for (unsigned worker = 0; worker < workers; ++worker) {
            const int d0 = 1 + int(worker) * span;
            const int d1 = worker + 1 == workers ? kDisparities : std::min(kDisparities, d0 + span);
            if (d0 >= d1) continue;
            threads.emplace_back([=, &cost, &leftFiltered, &rightFiltered] {
                const int stride = width + 1;
                std::vector<uint32_t> integral(size_t(stride) * (height + 1), 0);
                std::vector<uint16_t> difference(pixels, 0);
                for (int d = d0; d < d1; ++d) {
                    std::fill(difference.begin(), difference.end(), 0);
                    for (int y = 1; y < height - 1; ++y) {
                        const int16_t* lrow = leftFiltered.data() + size_t(y) * width;
                        const int16_t* rrow = rightFiltered.data() + size_t(y) * width;
                        uint16_t* out = difference.data() + size_t(y) * width;
                        for (int x = d + 2; x < width - 2; ++x) {
                            const int u = lrow[x];
                            const int um = lrow[x - 1];
                            const int up = lrow[x + 1];
                            const int v = rrow[x - d];
                            const int vm = rrow[x - d - 1];
                            const int vp = rrow[x - d + 1];
                            const int halfRightPlus = (v + vp) / 2;
                            const int halfRightMinus = (v + vm) / 2;
                            const int rightMin = std::min(v, std::min(halfRightPlus, halfRightMinus));
                            const int rightMax = std::max(v, std::max(halfRightPlus, halfRightMinus));
                            const int leftToRight = std::max(0, std::max(u - rightMax, rightMin - u));
                            const int halfLeftPlus = (u + up) / 2;
                            const int halfLeftMinus = (u + um) / 2;
                            const int leftMin = std::min(u, std::min(halfLeftPlus, halfLeftMinus));
                            const int leftMax = std::max(u, std::max(halfLeftPlus, halfLeftMinus));
                            const int rightToLeft = std::max(0, std::max(v - leftMax, leftMin - v));
                            out[x] = static_cast<uint16_t>(std::min(leftToRight, rightToLeft));
                        }
                    }
                    for (int y = 0; y < height; ++y) {
                        uint32_t row = 0;
                        for (int x = 0; x < width; ++x) {
                            row += difference[size_t(y) * width + x];
                            integral[size_t(y + 1) * stride + (x + 1)] =
                                integral[size_t(y) * stride + (x + 1)] + row;
                        }
                    }
                    for (int y = 2; y < height - 2; ++y) {
                        for (int x = std::max(d + 4, 4); x < width - 3; ++x) {
                            const int x0 = x - 2;
                            const int x1 = x + 3;
                            const int y0 = y - 2;
                            const int y1 = y + 3;
                            const uint32_t sum = integral[size_t(y1) * stride + x1] -
                                                 integral[size_t(y0) * stride + x1] -
                                                 integral[size_t(y1) * stride + x0] +
                                                 integral[size_t(y0) * stride + x0];
                            cost[(size_t(y) * width + x) * kDisparities + d] =
                                static_cast<uint16_t>(std::min<uint32_t>(sum, 4000));
                        }
                    }
                }
            });
        }
        for (std::thread& thread : threads) thread.join();
    }

    std::vector<uint32_t> summed(pixels * kDisparities, 0);
    accumulateHorizontal(cost.data(), summed.data(), width, height, true);
    accumulateHorizontal(cost.data(), summed.data(), width, height, false);
    accumulateVertical(cost.data(), summed.data(), width, height, true);
    accumulateVertical(cost.data(), summed.data(), width, height, false);
    accumulateDiagonal(cost.data(), summed.data(), width, height, 1, 1);
    accumulateDiagonal(cost.data(), summed.data(), width, height, -1, 1);
    accumulateDiagonal(cost.data(), summed.data(), width, height, 1, -1);
    accumulateDiagonal(cost.data(), summed.data(), width, height, -1, -1);

    std::vector<int> best(pixels, 0);
    std::vector<uint32_t> bestCost(pixels, std::numeric_limits<uint32_t>::max());
    for (int y = 2; y < height - 2; ++y) {
        for (int x = kDisparities + 2; x < width - 2; ++x) {
            const size_t pixel = size_t(y) * width + x;
            const size_t base = pixel * kDisparities;
            for (int d = 1; d < kDisparities; ++d) {
                const uint32_t value = summed[base + d];
                if (value < bestCost[pixel]) {
                    bestCost[pixel] = value;
                    best[pixel] = d;
                }
            }
            // A correct valley is a few disparities wide, so the uniqueness
            // rival starts two steps away from the winner.
            uint32_t rival = std::numeric_limits<uint32_t>::max();
            for (int d = 1; d < kDisparities; ++d) {
                if (std::abs(d - best[pixel]) <= 2) continue;
                rival = std::min(rival, summed[base + d]);
            }
            if (rival < bestCost[pixel] + bestCost[pixel] / 10) best[pixel] = 0;
        }
    }

    std::vector<int> rightBest(pixels, 0);
    std::vector<uint32_t> rightCost(pixels, std::numeric_limits<uint32_t>::max());
    for (int y = 2; y < height - 2; ++y) {
        for (int x = kDisparities + 2; x < width - 2; ++x) {
            const size_t base = (size_t(y) * width + x) * kDisparities;
            for (int d = 1; d < kDisparities; ++d) {
                const size_t right = size_t(y) * width + (x - d);
                const uint32_t value = cost[base + d];
                if (value < rightCost[right]) {
                    rightCost[right] = value;
                    rightBest[right] = d;
                }
            }
        }
    }

    for (int y = 2; y < height - 2; ++y) {
        for (int x = kDisparities + 2; x < width - 2; ++x) {
            const size_t pixel = size_t(y) * width + x;
            const int d = best[pixel];
            if (d <= 1 || d >= kDisparities - 1) continue;
            if (std::abs(rightBest[pixel - d] - d) > 1) continue;
            const size_t base = pixel * kDisparities;
            float disparity = float(d);
            const float c0 = float(summed[base + d - 1]);
            const float c1 = float(summed[base + d]);
            const float c2 = float(summed[base + d + 1]);
            const float denom = 2.0f * (c0 - 2.0f * c1 + c2);
            if (std::fabs(denom) > 1.0f)
                disparity += std::clamp((c0 - c2) / denom, -1.0f, 1.0f);
            if (disparity > 1.0f) depth[pixel] = focalBaseline / disparity;
        }
    }
    removeSpeckles(depth, width, height, focalBaseline);
    // Foreground fattening leaves a sheet in front of the real surface.
    // Drop pixels that are nearer than the local disparity median.
    std::vector<float> disparity(pixels, 0.0f);
    for (size_t i = 0; i < pixels; ++i)
        if (depth[i] > 0.0f) disparity[i] = focalBaseline / depth[i];
    for (int y = 6; y < height - 6; ++y) {
        for (int x = 6; x < width - 6; ++x) {
            const size_t pixel = size_t(y) * width + x;
            if (!(disparity[pixel] > 0.0f)) continue;
            std::vector<float> window;
            window.reserve(49);
            for (int dy = -6; dy <= 6; dy += 2) {
                for (int dx = -6; dx <= 6; dx += 2) {
                    const float sample = disparity[size_t(y + dy) * width + (x + dx)];
                    if (sample > 0.0f) window.push_back(sample);
                }
            }
            if (window.size() < 8) continue;
            const size_t middle = window.size() / 2;
            std::nth_element(window.begin(), window.begin() + middle, window.end());
            if (disparity[pixel] > window[middle] + 1.25f) depth[pixel] = 0.0f;
        }
    }
}

void PersistentMap::integrate(const float* depth, int width, int height, float focalPx,
                              float principalX, float principalY, float focalBaseline,
                              float cameraX, float cameraY, float cameraZ, float yawRad) {
    const float cosine = std::cos(yawRad);
    const float sine = std::sin(yawRad);
    constexpr float kVoxel = 0.02f;
    constexpr float kMinimumBand = 0.06f;
    constexpr float kDisparitySigma = 0.75f;
    for (int y = 2; y < height - 2; y += 2) {
        for (int x = 2; x < width - 2; x += 2) {
            const float z = depth[size_t(y) * width + x];
            if (!(z >= 0.25f && z <= 5.0f)) continue;
            const float rx = (float(x) - principalX) * z / focalPx;
            const float ry = (float(y) - principalY) * z / focalPx;
            const float wx = cameraX + cosine * rx + sine * z;
            const float wy = cameraY + ry;
            const float wz = cameraZ - sine * rx + cosine * z;
            const PersistentKey key{
                int(std::floor(wx / kVoxel)),
                int(std::floor(wy / kVoxel)),
                int(std::floor(wz / kVoxel)),
            };
            const float band = std::max(kMinimumBand, z * z * kDisparitySigma / focalBaseline);
            auto [it, inserted] = cells_.try_emplace(key);
            PersistentCell& cell = it->second;
            if (inserted) {
                cell.x = wx;
                cell.y = wy;
                cell.z = wz;
                cell.observations = 1;
                continue;
            }
            const float dx = wx - cell.x;
            const float dy = wy - cell.y;
            const float dz = wz - cell.z;
            if (dx * dx + dy * dy + dz * dz > band * band) continue;
            const float weight = 1.0f / float(cell.observations + 1);
            cell.x += dx * weight;
            cell.y += dy * weight;
            cell.z += dz * weight;
            cell.observations = std::min(cell.observations + 1, 255);
        }
    }
}

int PersistentMap::confirmed(int minimumObservations) const {
    int count = 0;
    for (const auto& entry : cells_)
        if (entry.second.observations >= minimumObservations) ++count;
    return count;
}

}  // namespace FrameSpatial
