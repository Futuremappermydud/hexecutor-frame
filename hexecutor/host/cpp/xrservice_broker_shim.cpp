#include "hexecutor/xrservice_broker_shim.h"
#include "hexecutor/broker.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
constexpr uint32_t side = 128;
constexpr uint32_t channels = 2;
constexpr size_t head_floats = 69;
constexpr size_t head_bytes = head_floats * sizeof(float);
constexpr size_t output_record_bytes = 0x450; // 4 * 69 * sizeof(float)
constexpr size_t gemm_floats = 552;
constexpr size_t gazenet_output_floats = 64;
constexpr size_t gazenet_vector_floats = 4800;
constexpr float input_mean = 0.449f;
constexpr float input_std = 0.226f;

std::string get_default_socket() {
    const char *env = std::getenv("HEXECUTORD_SOCKET");
    if (env && *env)
        return env;
    const char *xdg = std::getenv("XDG_RUNTIME_DIR");
    if (xdg && *xdg)
        return std::string(xdg) + "/hexecutord.sock";
    return "/tmp/hexecutord.sock";
}

struct ShimState {
    std::mutex mutex;
    std::string socket_path;
    std::unique_ptr<hexecutor::broker::Client> client;
    hexecutor::broker::Client::Stream featurenn_stream;
    hexecutor::broker::Client::Stream gazenet_stream;
    hexecutor::broker::Client::Stream vector_stream;

    ShimState() : socket_path(get_default_socket()) {}

    bool ensure_connected_locked() {
        if (client)
            return true;
        try {
            client = std::make_unique<hexecutor::broker::Client>(socket_path);

            // Open stereoFeatureNN stream if graph exists
            hexecutor::broker::StreamSpec fn_spec;
            fn_spec.graph = "featurenn";
            fn_spec.cls = hexecutor::broker::Class::tracking;
            fn_spec.period_ms = 0;
            fn_spec.deadline_ms = 50;
            fn_spec.latest_only = false;
            try {
                featurenn_stream = client->open(fn_spec);
            } catch (const std::exception &e) {
                // FeatureNN graph might not be registered
            }

            // Open Gazenet image stream if graph exists
            hexecutor::broker::StreamSpec gaze_spec;
            gaze_spec.graph = "gazenet";
            gaze_spec.cls = hexecutor::broker::Class::tracking;
            gaze_spec.period_ms = 0;
            gaze_spec.deadline_ms = 50;
            gaze_spec.latest_only = false;
            try {
                gazenet_stream = client->open(gaze_spec);
            } catch (const std::exception &e) {
                // Gazenet graph might not be registered
            }

            // Open Gazenet vector stream if graph exists
            hexecutor::broker::StreamSpec vec_spec;
            vec_spec.graph = "gazenet_vector";
            vec_spec.cls = hexecutor::broker::Class::tracking;
            vec_spec.period_ms = 0;
            vec_spec.deadline_ms = 50;
            vec_spec.latest_only = false;
            try {
                vector_stream = client->open(vec_spec);
            } catch (const std::exception &e) {
                // Vector graph might not be registered
            }

            return true;
        } catch (const std::exception &e) {
            client.reset();
            return false;
        }
    }
};

ShimState &state() {
    static ShimState s;
    return s;
}

hexecutor::Bytes join_floats(const std::vector<hexecutor::Bytes> &parts, size_t min_f, size_t max_f) {
    hexecutor::Bytes joined;
    joined.reserve(max_f * sizeof(float));
    for (auto &p : parts) {
        if (p.size() % sizeof(float) || joined.size() + p.size() > max_f * sizeof(float))
            throw std::runtime_error("broker graph returned invalid float output");
        joined.insert(joined.end(), p.begin(), p.end());
    }
    size_t floats = joined.size() / sizeof(float);
    if (floats < min_f || floats > max_f)
        throw std::runtime_error("broker graph returned incorrect float count");
    auto *vals = reinterpret_cast<const float *>(joined.data());
    for (size_t i = 0; i < floats; i++) {
        if (!std::isfinite(vals[i]))
            throw std::runtime_error("broker graph returned non-finite float");
    }
    return joined;
}

} // namespace

extern "C" {

void broker_shim_set_socket(const char *socket_path) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (socket_path && *socket_path) {
        s.socket_path = socket_path;
        s.client.reset();
    }
}

int broker_shim_is_connected(void) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.ensure_connected_locked() ? 1 : 0;
}

int broker_shim_dispatch_stereofeaturenn(const void *input_u8, void *output_records,
                                        uint32_t views, uint32_t width, uint32_t height) {
    if (!input_u8 || !output_records || width != side || height != side || views == 0 || (views % 2 != 0))
        return -1;
    uint32_t controllers = views / 2;
    if (controllers > 2)
        return -1;

    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ensure_connected_locked() || !s.featurenn_stream.id)
        return -2;

    try {
        const auto *raw = static_cast<const uint8_t *>(input_u8);
        auto *out_base = static_cast<uint8_t *>(output_records);

        for (uint32_t c = 0; c < controllers; c++) {
            // Normalize stereo u8 planes to 3-channel NHWC
            hexecutor::Bytes in_buf(side * side * 3 * sizeof(float));
            auto *dst = reinterpret_cast<float *>(in_buf.data());
            const uint8_t *ctrl_raw = raw + size_t(c) * channels * side * side;

            for (uint32_t eye = 0; eye < channels; eye++) {
                for (size_t pixel = 0; pixel < side * side; pixel++) {
                    float x = float(ctrl_raw[size_t(eye) * side * side + pixel]) / 255.0f;
                    dst[pixel * 3 + eye] = (x - input_mean) / input_std;
                }
            }
            for (size_t pixel = 0; pixel < side * side; pixel++)
                dst[pixel * 3 + 2] = 0.0f;

            auto outputs = s.client->run(s.featurenn_stream, {{in_buf.data(), in_buf.size()}});
            hexecutor::Bytes joined = join_floats(outputs, gemm_floats, gemm_floats);
            const auto *gemm = reinterpret_cast<const float *>(joined.data());

            // Format heads into output record
            uint8_t *ctrl_out = out_base + size_t(c) * channels * output_record_bytes;
            for (uint32_t eye = 0; eye < channels; eye++) {
                uint8_t *record = ctrl_out + size_t(eye) * output_record_bytes;
                for (size_t head = 0; head < 4; head++) {
                    size_t source = (size_t(eye) * 92 + head * 23) * 3;
                    if (head != 1) {
                        std::memcpy(record + head * head_bytes, gemm + source, head_bytes);
                        continue;
                    }
                    float uncertainty[head_floats];
                    for (size_t i = 0; i < head_floats; i++)
                        uncertainty[i] = 2.0f * std::min(gemm[source + i], 10.0f);
                    std::memcpy(record + head_bytes, uncertainty, head_bytes);
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[xrservice_broker_shim] FeatureNN dispatch failed: " << e.what() << '\n';
        return -3;
    }
}

int broker_shim_dispatch_gazenet(const float *input_planar_f32, float *output_64_f32) {
    if (!input_planar_f32 || !output_64_f32)
        return -1;

    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ensure_connected_locked() || !s.gazenet_stream.id)
        return -2;

    try {
        // Convert 2-channel planar float input to compact 3-channel NHWC with zero 3rd channel
        hexecutor::Bytes in_buf(side * side * 3 * sizeof(float));
        auto *dst = reinterpret_cast<float *>(in_buf.data());
        for (size_t pixel = 0; pixel < side * side; pixel++) {
            for (size_t eye = 0; eye < channels; eye++) {
                dst[pixel * 3 + eye] = input_planar_f32[eye * side * side + pixel];
            }
            dst[pixel * 3 + 2] = 0.0f;
        }

        auto outputs = s.client->run(s.gazenet_stream, {{in_buf.data(), in_buf.size()}});
        hexecutor::Bytes joined = join_floats(outputs, gazenet_output_floats, gazenet_output_floats);
        std::memcpy(output_64_f32, joined.data(), gazenet_output_floats * sizeof(float));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[xrservice_broker_shim] Gazenet dispatch failed: " << e.what() << '\n';
        return -3;
    }
}

int broker_shim_dispatch_gazenet_vector(const float *input_f32, uint32_t in_floats,
                                       float *output_f32, uint32_t max_out_floats,
                                       uint32_t *out_floats) {
    if (!input_f32 || !output_f32 || in_floats == 0 || in_floats > gazenet_vector_floats || !out_floats)
        return -1;

    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ensure_connected_locked() || !s.vector_stream.id)
        return -2;

    try {
        auto outputs = s.client->run(s.vector_stream,
                                    {{input_f32, size_t(in_floats) * sizeof(float)}});
        hexecutor::Bytes joined = join_floats(outputs, 1, gazenet_vector_floats);
        size_t count = joined.size() / sizeof(float);
        if (count > max_out_floats)
            return -4;
        std::memcpy(output_f32, joined.data(), count * sizeof(float));
        *out_floats = uint32_t(count);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[xrservice_broker_shim] Gazenet vector dispatch failed: " << e.what() << '\n';
        return -3;
    }
}

// LD_PRELOAD interceptor for XRService's dispatchStereoFeatureNNOnCDSP
int dispatchStereoFeatureNNOnCDSP(const void *input_u8, void *output_records,
                                  uint32_t views, uint32_t width, uint32_t height) {
    return broker_shim_dispatch_stereofeaturenn(input_u8, output_records, views, width, height);
}

} // extern "C"
