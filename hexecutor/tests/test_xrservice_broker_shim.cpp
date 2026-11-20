// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/xrservice_broker_shim.h"
#include "hexecutor/broker.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace hexecutor;

static void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}

struct FakeBackend : broker::Backend {
    std::vector<broker::GraphInfo> graphs() const override {
        return {
            {"featurenn", {128 * 128 * 3 * sizeof(float)}, {352 * sizeof(float), 200 * sizeof(float)}},
            {"gazenet", {128 * 128 * 3 * sizeof(float)}, {64 * sizeof(float)}},
            {"gazenet_vector", {4800 * sizeof(float)}, {10 * sizeof(float)}}
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
                p0[i] = float(i) * 0.25f - 10.0f;
            auto *p1 = reinterpret_cast<float *>(out[1].data());
            for (size_t i = 0; i < 200; i++)
                p1[i] = float(352 + i) * 0.25f - 10.0f;
            return out;
        } else if (graph == 1) {
            require(inputs.size() == 1 && inputs[0].bytes == 128 * 128 * 3 * sizeof(float),
                    "gazenet input shape");
            std::vector<Bytes> out(1);
            out[0].resize(64 * sizeof(float));
            auto *p = reinterpret_cast<float *>(out[0].data());
            for (size_t i = 0; i < 64; i++)
                p[i] = float(i) * 1.5f;
            return out;
        } else if (graph == 2) {
            require(inputs.size() == 1 && inputs[0].bytes == 4800 * sizeof(float),
                    "gazenet_vector input shape");
            std::vector<Bytes> out(1);
            out[0].resize(10 * sizeof(float));
            auto *p = reinterpret_cast<float *>(out[0].data());
            for (size_t i = 0; i < 10; i++)
                p[i] = float(i) + 100.0f;
            return out;
        }
        throw std::runtime_error("unknown graph in FakeBackend");
    }
};

int main() {
    try {
        std::string socket_path = "/tmp/test_xrservice_shim_" + std::to_string(::getpid()) + ".sock";
        ::unlink(socket_path.c_str());

        struct Cleaner {
            std::string path;
            ~Cleaner() { ::unlink(path.c_str()); }
        } cleaner{socket_path};

        broker::ServerOptions server_options;
        server_options.socket = socket_path;
        broker::Server server(std::make_unique<FakeBackend>(), server_options);
        server.start();

        broker_shim_set_socket(socket_path.c_str());
        require(broker_shim_is_connected() == 1, "broker shim connects to server");

        // 1. Test FeatureNN dispatch (1 controller = 2 eye views)
        std::vector<uint8_t> u8_input(2 * 128 * 128, 128);
        std::vector<uint8_t> output_records(2 * 0x450, 0);

        int rc = broker_shim_dispatch_stereofeaturenn(u8_input.data(), output_records.data(), 2, 128, 128);
        require(rc == 0, "broker_shim_dispatch_stereofeaturenn returns 0");

        auto *out_floats = reinterpret_cast<const float *>(output_records.data());
        require(out_floats[0] == -10.0f, "first head float matches");

        // 2. Test LD_PRELOAD symbol alias dispatchStereoFeatureNNOnCDSP (2 controllers = 4 eye views)
        std::vector<uint8_t> u8_input4(4 * 128 * 128, 64);
        std::vector<uint8_t> output_records4(4 * 0x450, 0);
        rc = dispatchStereoFeatureNNOnCDSP(u8_input4.data(), output_records4.data(), 4, 128, 128);
        require(rc == 0, "dispatchStereoFeatureNNOnCDSP returns 0");

        // 3. Test Gazenet image dispatch
        std::vector<float> gaze_in(2 * 128 * 128, 0.5f);
        std::vector<float> gaze_out(64, 0.0f);
        rc = broker_shim_dispatch_gazenet(gaze_in.data(), gaze_out.data());
        require(rc == 0, "broker_shim_dispatch_gazenet returns 0");
        require(gaze_out[0] == 0.0f && gaze_out[1] == 1.5f, "gazenet output floats match");

        // 4. Test Gazenet vector dispatch
        std::vector<float> vec_in(4800, 1.0f);
        std::vector<float> vec_out(64, 0.0f);
        uint32_t out_count = 0;
        rc = broker_shim_dispatch_gazenet_vector(vec_in.data(), 4800, vec_out.data(), 64, &out_count);
        require(rc == 0, "broker_shim_dispatch_gazenet_vector returns 0");
        require(out_count == 10, "gazenet vector output count is 10");
        require(vec_out[0] == 100.0f && vec_out[9] == 109.0f, "gazenet vector output values match");

        server.stop();
    } catch (const std::exception &e) {
        std::cerr << "test_xrservice_broker_shim: " << e.what() << '\n';
        return 1;
    }
    std::cout << "test_xrservice_broker_shim: ok\n";
    return 0;
}
