// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/xrservice_featurenn.hpp"
#include "hexecutor/broker.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

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

struct FakeAdapterBackend : broker::Backend {
    std::vector<broker::GraphInfo> graphs() const override {
        return {
            {"featurenn", {128 * 128 * 3 * sizeof(float)}, {352 * sizeof(float), 200 * sizeof(float)}},
            {"gazenet", {128 * 128 * 3 * sizeof(float)}, {64 * sizeof(float)}},
            {"gazenet_vector", {gazenet_vector_floats * sizeof(float)}, {5 * sizeof(float)}}
        };
    }
    std::vector<Bytes> run(size_t graph, const std::vector<Buffer> &inputs, double &dsp_ms) override {
        dsp_ms = 0.5;
        if (graph == 0) {
            require(inputs.size() == 1 && inputs[0].bytes == 128 * 128 * 3 * sizeof(float),
                    "featurenn input shape");
            std::vector<Bytes> out(2);
            out[0].resize(352 * sizeof(float));
            out[1].resize(200 * sizeof(float));
            auto *p0 = reinterpret_cast<float *>(out[0].data());
            for (size_t i = 0; i < 352; i++)
                p0[i] = float(i) * 0.1f;
            auto *p1 = reinterpret_cast<float *>(out[1].data());
            for (size_t i = 0; i < 200; i++)
                p1[i] = float(352 + i) * 0.1f;
            return out;
        } else if (graph == 1) {
            require(inputs.size() == 1 && inputs[0].bytes == 128 * 128 * 3 * sizeof(float),
                    "gazenet input shape");
            std::vector<Bytes> out(1);
            out[0].resize(64 * sizeof(float));
            auto *p = reinterpret_cast<float *>(out[0].data());
            for (size_t i = 0; i < 64; i++)
                p[i] = float(i) * 0.5f;
            return out;
        } else if (graph == 2) {
            require(inputs.size() == 1 && inputs[0].bytes == gazenet_vector_floats * sizeof(float),
                    "gazenet_vector input shape");
            std::vector<Bytes> out(1);
            out[0].resize(5 * sizeof(float));
            auto *p = reinterpret_cast<float *>(out[0].data());
            for (size_t i = 0; i < 5; i++)
                p[i] = float(i) + 42.0f;
            return out;
        }
        throw std::runtime_error("unknown graph in FakeAdapterBackend");
    }
};

static void test_adapter_bridge() {
    std::string socket_path = "/tmp/test_xrservice_bridge_" + std::to_string(::getpid()) + ".sock";
    ::unlink(socket_path.c_str());

    struct Cleaner {
        std::string path;
        ~Cleaner() { ::unlink(path.c_str()); }
    } cleaner{socket_path};

    broker::ServerOptions server_options;
    server_options.socket = socket_path;
    broker::Server server(std::make_unique<FakeAdapterBackend>(), server_options);
    server.start();

    std::vector<uint8_t> dma(dma_bytes, 0);
    put<uint64_t>(dma, 0xa90000, 0x1614df651d03f542ULL);
    put<uint64_t>(dma, 0xa90008, 9ULL);

    std::atomic<int> yields{0};
    AdapterOptions opts;
    opts.set_stereo_featurenn_graph("featurenn");
    require(opts.stereo_featurenn_graph() == "featurenn", "stereo_featurenn alias works");
    opts.gazenet_graph = "gazenet";
    opts.gazenet_vector_graph = "gazenet_vector";

    Adapter adapter(socket_path, opts, [&] { yields++; }, dma.data(), dma.size());
    adapter.start();

    // 1. StereoFeatureNN Job (job 8)
    auto *fnn = dma.data() + featurenn_control_offset;
    put<uint32_t>(dma, featurenn_control_offset + 0x80, 2);   // eyes = 2
    put<uint32_t>(dma, featurenn_control_offset + 0x84, 2);   // views = 2
    put<uint32_t>(dma, featurenn_control_offset + 0x88, 128); // width = 128
    put<uint32_t>(dma, featurenn_control_offset + 0x8c, 128); // height = 128
    for (size_t i = 0; i < 2 * 128 * 128; i++)
        dma[featurenn_control_offset + featurenn_input_offset + i] = uint8_t(i & 0xff);

    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(fnn + 8), featurenn_job, __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(fnn + 0), 2, __ATOMIC_SEQ_CST); // state = 2
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(fnn + 4), 1, __ATOMIC_SEQ_CST); // active = 1

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (__atomic_load_n(reinterpret_cast<volatile uint32_t *>(fnn + 0), __ATOMIC_SEQ_CST) != 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(fnn + 0), __ATOMIC_SEQ_CST) == 3,
            "StereoFeatureNN job finished with state 3");

    // Output check
    float first_out = get<float>(dma.data(), featurenn_control_offset + featurenn_output_offset);
    require(first_out == 0.0f, "StereoFeatureNN first output float");

    // XRService consumes output: active = 0
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(fnn + 4), 0, __ATOMIC_SEQ_CST);
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (__atomic_load_n(reinterpret_cast<volatile uint32_t *>(fnn + 8), __ATOMIC_SEQ_CST) != 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(fnn + 8), __ATOMIC_SEQ_CST) == 0,
            "StereoFeatureNN taken reset to 0");
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(fnn + 0), __ATOMIC_SEQ_CST) == 1,
            "StereoFeatureNN state reset to 1");

    // 2. Gazenet Image Job: first job 4 yields to stock buffer acquisition
    auto *gaze = dma.data() + gazenet_control_offset;
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 8), gazenet_job, __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), 2, __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 4), 1, __ATOMIC_SEQ_CST);

    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (yields.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(yields.load() > 0, "Adapter yields on stock buffer acquisition");

    // Simulate stock buffer acquisition completing:
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), 3, __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 4), 0, __ATOMIC_SEQ_CST);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Now post evaluation job 4:
    for (size_t i = 0; i < 2 * 128 * 128; i++)
        put<float>(dma, gazenet_control_offset + gazenet_input_offset + i * sizeof(float), float(i));
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 8), gazenet_job, __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), 2, __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 4), 1, __ATOMIC_SEQ_CST);

    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), __ATOMIC_SEQ_CST) != 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), __ATOMIC_SEQ_CST) == 3,
            "Gazenet image evaluation completed with state 3");
    require(get<float>(dma.data(), gazenet_control_offset + gazenet_output_offset + sizeof(float)) == 0.5f,
            "Gazenet output float layout");

    // XRService consumes output: active = 0
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 4), 0, __ATOMIC_SEQ_CST);
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), __ATOMIC_SEQ_CST) != 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 0), __ATOMIC_SEQ_CST) == 1,
            "Gazenet state reset to 1");

    // 3. Gazenet Vector Job
    for (size_t i = 0; i < gazenet_vector_floats; i++)
        put<float>(dma, gazenet_control_offset + gazenet_vector_offset + i * sizeof(float), float(i));
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 8), uint32_t(gazenet_vector_floats), __ATOMIC_SEQ_CST);
    __atomic_store_n(reinterpret_cast<volatile uint32_t *>(gaze + 0xc), 1, __ATOMIC_SEQ_CST); // ready = 1

    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 0xc), __ATOMIC_SEQ_CST) != 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 0xc), __ATOMIC_SEQ_CST) == 0,
            "Gazenet vector ready cleared");
    require(__atomic_load_n(reinterpret_cast<volatile uint32_t *>(gaze + 8), __ATOMIC_SEQ_CST) == 5,
            "Gazenet vector returned 5 floats");
    require(get<float>(dma.data(), gazenet_control_offset + gazenet_vector_offset) == 42.0f,
            "Gazenet vector output value");

    adapter.stop();
    server.stop();
}

static void test_featurenn_only_ignores_gazenet_block() {
    std::string socket_path = "/tmp/test_featurenn_only_" + std::to_string(::getpid()) + ".sock";
    ::unlink(socket_path.c_str());
    broker::ServerOptions server_options;
    server_options.socket = socket_path;
    broker::Server server(std::make_unique<FakeAdapterBackend>(), server_options);
    server.start();
    std::vector<uint8_t> dma(dma_bytes, 0);
    put<uint64_t>(dma, 0xa90000, 0x1614df651d03f542ULL);
    put<uint64_t>(dma, 0xa90008, 9ULL);
    put<uint32_t>(dma, gazenet_control_offset + 4, 0xbeef1234);
    put<uint32_t>(dma, gazenet_control_offset + 0xc, 0xbeef5678);
    AdapterOptions opts;
    opts.featurenn_graph = "featurenn";
    Adapter adapter(socket_path, opts, [] {}, dma.data(), dma.size());
    adapter.start();
    adapter.stop();
    server.stop();
    ::unlink(socket_path.c_str());
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

        test_adapter_bridge();
        test_featurenn_only_ignores_gazenet_block();
    } catch (const std::exception &e) {
        std::cerr << "test_xrservice_featurenn: " << e.what() << '\n';
        return 1;
    }
    std::cout << "test_xrservice_featurenn: ok\n";
    return 0;
}
