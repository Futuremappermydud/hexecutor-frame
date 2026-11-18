#include "hexecutor/xrservice_featurenn.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace hexecutor;
using namespace hexecutor::xrservice;

static void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
template <class T> static void put(std::vector<uint8_t> &b, size_t at, T value) {
    std::memcpy(b.data() + at, &value, sizeof value);
}
template <class T> static T get(const void *base, size_t at) {
    T value;
    std::memcpy(&value, static_cast<const uint8_t *>(base) + at, sizeof value);
    return value;
}

int main() {
    try {
        std::vector<uint8_t> shared(featurenn_output_offset + 4 * featurenn_output_record_bytes);
        put<uint32_t>(shared, 0x80, 2);
        put<uint32_t>(shared, 0x84, 2);
        put<uint32_t>(shared, 0x88, 128);
        put<uint32_t>(shared, 0x8c, 128);
        auto request = read_featurenn_request(shared.data(), shared.size());
        require(request.eyes == 2 && request.controllers == 1 && request.input_bytes() == 32768,
                "stock request layout");
        for (size_t i = 0; i < 16384; i++) {
            shared[featurenn_input_offset + i] = uint8_t(i);
            shared[featurenn_input_offset + 16384 + i] = uint8_t(255 - i);
        }
        auto input = make_featurenn_input(request, shared.data(), shared.size(), 0);
        require(input.size() == 196608, "float NHWC input size");
        auto *f = reinterpret_cast<const float *>(input.data());
        auto normalized = [](uint8_t x) { return (float(x) / 255.0f - 0.449f) / 0.226f; };
        require(f[0] == normalized(0) && f[1] == normalized(255) && f[2] == 0,
                "first normalized pixel");
        size_t last = (128 * 128 - 1) * 3;
        require(f[last] == normalized(255) && f[last + 1] == normalized(0) && f[last + 2] == 0,
                "last normalized pixel");

        Bytes gemm(552 * sizeof(float));
        auto *g = reinterpret_cast<float *>(gemm.data());
        for (size_t i = 0; i < 552; i++)
            g[i] = float(i) * 0.25f - 40.0f;
        std::vector<Bytes> parts;
        parts.emplace_back(gemm.begin(), gemm.begin() + 352 * sizeof(float));
        parts.emplace_back(gemm.begin() + 352 * sizeof(float), gemm.end());
        size_t calls = 0;
        run_featurenn_request(request, shared.data(), shared.size(), [&](const Bytes &input) {
            require(input.size() == 196608, "full request input");
            calls++;
            return parts;
        });
        require(calls == request.controllers, "one broker run per controller");
        for (size_t eye = 0; eye < 2; eye++)
            for (size_t head = 0; head < 4; head++)
                for (size_t i = 0; i < 69; i++) {
                    size_t source = (eye * 92 + head * 23) * 3 + i;
                    float want = g[source];
                    if (head == 1)
                        want = 2.0f * std::min(want, 10.0f);
                    float got = get<float>(shared.data(), featurenn_output_offset +
                                                       eye * featurenn_output_record_bytes +
                                                       head * 69 * sizeof(float) + i * sizeof(float));
                    require(got == want, "raw output record layout");
                }

        g[100] = std::numeric_limits<float>::infinity();
        bool threw = false;
        try {
            write_featurenn_output(request, {gemm}, 0, shared.data(), shared.size());
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw, "non-finite output rejected");
        put<uint32_t>(shared, 0x88, 64);
        threw = false;
        try {
            (void)read_featurenn_request(shared.data(), shared.size());
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw, "unsupported crop rejected");

        for (size_t eye = 0; eye < 2; eye++)
            for (size_t pixel = 0; pixel < 128 * 128; pixel++)
                put<float>(shared, gazenet_input_offset + (eye * 128 * 128 + pixel) * sizeof(float),
                           float(eye * 128 * 128 + pixel) * 0.5f - 100.0f);
        auto gaze_input = make_gazenet_input(shared.data(), shared.size());
        require(gaze_input.size() == 128 * 128 * 3 * sizeof(float), "Gazenet NHWC input size");
        auto *gi = reinterpret_cast<const float *>(gaze_input.data());
        require(gi[0] == -100.0f && gi[1] == 8092.0f && gi[2] == 0,
                "Gazenet first pixel");
        size_t gaze_last = (128 * 128 - 1) * 3;
        require(gi[gaze_last] == get<float>(shared.data(), gazenet_input_offset +
                                                        (128 * 128 - 1) * sizeof(float)) &&
                    gi[gaze_last + 1] == get<float>(shared.data(), gazenet_input_offset +
                                                               (2 * 128 * 128 - 1) * sizeof(float)) &&
                    gi[gaze_last + 2] == 0,
                "Gazenet last pixel");
        Bytes gaze_gemm(64 * sizeof(float));
        auto *gg = reinterpret_cast<float *>(gaze_gemm.data());
        for (size_t i = 0; i < 64; i++)
            gg[i] = float(i) - 20.0f;
        std::vector<Bytes> gaze_parts;
        gaze_parts.emplace_back(gaze_gemm.begin(), gaze_gemm.begin() + 16 * sizeof(float));
        gaze_parts.emplace_back(gaze_gemm.begin() + 16 * sizeof(float), gaze_gemm.end());
        run_gazenet_request(shared.data(), shared.size(), [&](const Bytes &input) {
            require(input == gaze_input, "Gazenet full input");
            return gaze_parts;
        });
        for (size_t i = 0; i < 64; i++)
            require(get<float>(shared.data(), gazenet_output_offset + i * sizeof(float)) == gg[i],
                    "Gazenet image output layout");

        for (size_t i = 0; i < gazenet_vector_floats; i++)
            put<float>(shared, gazenet_vector_offset + i * sizeof(float), float(i) + 0.25f);
        auto vector_input = make_gazenet_vector_input(shared.data(), shared.size(),
                                                      gazenet_vector_floats);
        require(vector_input.size() == gazenet_vector_floats * sizeof(float),
                "Gazenet vector input size");
        std::vector<Bytes> vector_outputs(2);
        vector_outputs[0].resize(2 * sizeof(float));
        vector_outputs[1].resize(3 * sizeof(float));
        for (size_t i = 0; i < 5; i++)
            put<float>(vector_outputs[i < 2 ? 0 : 1], (i < 2 ? i : i - 2) * sizeof(float),
                       float(i) - 3.0f);
        require(write_gazenet_vector_output(vector_outputs, shared.data(), shared.size()) == 5,
                "Gazenet vector output count");
        for (size_t i = 0; i < 5; i++)
            require(get<float>(shared.data(), gazenet_vector_offset + i * sizeof(float)) == float(i) - 3.0f,
                    "Gazenet vector output layout");
        threw = false;
        try {
            (void)make_gazenet_vector_input(shared.data(), shared.size(),
                                            gazenet_vector_floats + 1);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw, "oversize Gazenet vector rejected");
    } catch (const std::exception &e) {
        std::cerr << "test_xrservice_featurenn: " << e.what() << '\n';
        return 1;
    }
    std::cout << "test_xrservice_featurenn: ok\n";
    return 0;
}
