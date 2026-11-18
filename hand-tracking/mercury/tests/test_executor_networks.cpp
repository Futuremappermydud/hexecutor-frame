#include "mercury/executor_networks.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace {
template <size_t N>
hexecutor::Bytes bytes(const std::array<float, N>& values) {
    const auto* p = reinterpret_cast<const uint8_t*>(values.data());
    return {p, p + sizeof(values)};
}
}  // namespace

int main() {
    int calls[2]{};
    mercury::ExecutorNetworks networks([&](size_t program, const std::vector<hexecutor::Buffer>& in) {
        assert(program < 2);
        ++calls[program];
        if (program == 0) {
            assert(in.size() == 1 && in[0].bytes == 160 * 160 * sizeof(float));
            return std::vector<hexecutor::Bytes>{bytes(std::array<float, 2>{1, 2}),
                bytes(std::array<float, 2>{3, 4}), bytes(std::array<float, 2>{5, 6}),
                bytes(std::array<float, 2>{7, 8})};
        }
        assert(in.size() == 3 && in[0].bytes == 128 * 128 * sizeof(float) &&
               in[1].bytes == 42 * sizeof(float) && in[2].bytes == sizeof(float));
        std::array<float, 21 * 22 * 22> nhwc{};
        for (size_t p = 0; p < 22 * 22; ++p)
            for (size_t j = 0; j < 21; ++j) nhwc[p * 21 + j] = float(j * 1000 + p);
        std::array<float, 21 * 22> depth{}; depth[0] = 42;
        std::array<float, 8> extras{}; extras[0] = 13;
        return std::vector<hexecutor::Bytes>{bytes(depth), bytes(extras), bytes(nhwc),
            bytes(std::array<float, 10>{})};
    });
    std::array<mercury::DetectionInput, 2> di{};
    std::array<mercury::DetectionOutput, 2> d{};
    networks.detect(di.data(), d.data(), d.size());
    assert(calls[0] == 2 && d[1].hand_exists[1] == 2 && d[0].size[0] == 7);
    std::array<mercury::KeypointInput, 2> ki{};
    std::array<mercury::KeypointOutput, 2> k{};
    networks.keypoints(ki.data(), k.data(), k.size());
    assert(calls[1] == 2);
    assert(k[0].heatmap_depth[0] == 42 && k[0].scalar_extras[0] == 13);
    for (size_t j : {size_t(0), size_t(7), size_t(20)})
        for (size_t p : {size_t(0), size_t(121), size_t(483)})
            assert(k[1].heatmap_xy[j * 22 * 22 + p] == float(j * 1000 + p));
    bool threw = false;
    try { networks.detect(nullptr, d.data(), 1); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    return 0;
}
