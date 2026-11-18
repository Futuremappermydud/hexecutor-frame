#pragma once
#include "hexecutor/executor.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hexecutor::xrservice {
constexpr size_t featurenn_control_offset = 0x1522000;
constexpr size_t featurenn_input_offset = 0x100;
constexpr size_t featurenn_output_offset = 0x40080;
constexpr size_t featurenn_output_record_bytes = 0x450;
constexpr uint32_t featurenn_job = 8;
constexpr size_t gazenet_control_offset = 0x445000;
constexpr size_t gazenet_input_offset = 0x100;
constexpr size_t gazenet_output_offset = 0x20100;
constexpr size_t gazenet_vector_offset = 0x440;
constexpr size_t gazenet_vector_floats = 4800;
constexpr uint32_t gazenet_job = 4;

struct FeatureNNRequest {
    uint32_t eyes = 0, controllers = 0, height = 0, width = 0;
    size_t input_bytes() const;
    size_t output_bytes() const;
};

// `shared` points at the stock cached FeatureNN control/data block.
FeatureNNRequest read_featurenn_request(const void *shared, size_t bytes);
Bytes make_featurenn_input(const FeatureNNRequest &, const void *shared, size_t bytes,
                           uint32_t controller);
void write_featurenn_output(const FeatureNNRequest &, const std::vector<Bytes> &graph_outputs,
                            uint32_t controller, void *shared, size_t bytes);
void run_featurenn_request(
    const FeatureNNRequest &, void *shared, size_t bytes,
    const std::function<std::vector<Bytes>(const Bytes &input)> &run);

// Stock Gazenet image I/O is two planar 128x128 float images and 64 floats.
// The broker graph sees compact NHWC with a zero third channel.
Bytes make_gazenet_input(const void *shared, size_t bytes);
void write_gazenet_output(const std::vector<Bytes> &graph_outputs, void *shared, size_t bytes);
void run_gazenet_request(
    void *shared, size_t bytes,
    const std::function<std::vector<Bytes>(const Bytes &input)> &run);

// The separate stock vector request is one padded float vector, in place.
Bytes make_gazenet_vector_input(const void *shared, size_t bytes, uint32_t floats);
size_t write_gazenet_vector_output(const std::vector<Bytes> &graph_outputs, void *shared,
                                   size_t bytes);

struct AdapterOptions {
    std::string featurenn_graph, gazenet_graph, gazenet_vector_graph;
};

// Adapts the unmodified stock shared-memory ABIs to ordinary broker streams.
// Other stock CDSP jobs are left to the firmware; `yield_device` releases ours.
class Adapter {
  public:
    Adapter(std::filesystem::path broker_socket, AdapterOptions,
            std::function<void()> yield_device);
    ~Adapter();
    Adapter(const Adapter &) = delete;
    Adapter &operator=(const Adapter &) = delete;
    void start();
    void stop();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace hexecutor::xrservice
