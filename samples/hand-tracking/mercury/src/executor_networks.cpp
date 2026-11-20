#include "mercury/executor_networks.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mercury {
namespace {

template <size_t N>
void copy_float_output(std::array<float, N>& dst, const hexecutor::Bytes& src) {
    if (src.size() != sizeof(dst))
        throw std::runtime_error("Mercury QNN16 output size mismatch");
    std::memcpy(dst.data(), src.data(), sizeof(dst));
}

void check_request(const void* in, const void* out, size_t count) {
    if (count && (!in || !out))
        throw std::invalid_argument("null Mercury network request");
}

}  // namespace

ExecutorNetworks::ExecutorNetworks(Run run) : run_(std::move(run)) {
    if (!run_) throw std::invalid_argument("Mercury network runner is empty");
}

ExecutorNetworks::ExecutorNetworks(hexecutor::Session& session)
    : ExecutorNetworks([&session](size_t program, const std::vector<hexecutor::Buffer>& inputs) {
          return session.run(inputs, program);
      }) {}

void ExecutorNetworks::detect(const DetectionInput* inputs, DetectionOutput* outputs, size_t count) {
    check_request(inputs, outputs, count);
    for (size_t i = 0; i < count; ++i) {
        const std::vector<hexecutor::Buffer> request{{inputs[i].image.data(), sizeof(inputs[i].image)}};
        const auto result = run_(0, request);
        if (result.size() != 4) throw std::runtime_error("Mercury detection requires four outputs");
        copy_float_output(outputs[i].hand_exists, result[0]);
        copy_float_output(outputs[i].cx, result[1]);
        copy_float_output(outputs[i].cy, result[2]);
        copy_float_output(outputs[i].size, result[3]);
    }
}

void ExecutorNetworks::keypoints(const KeypointInput* inputs, KeypointOutput* outputs, size_t count) {
    check_request(inputs, outputs, count);
    for (size_t i = 0; i < count; ++i) {
        const std::vector<hexecutor::Buffer> request{
            {inputs[i].image.data(), sizeof(inputs[i].image)},
            {inputs[i].last_keypoints.data(), sizeof(inputs[i].last_keypoints)},
            {&inputs[i].use_last_keypoints, sizeof(float)}};
        const auto result = run_(1, request);
        if (result.size() != 4) throw std::runtime_error("Mercury keypoint requires four outputs");
        constexpr size_t plane = kHeatmapSize * kHeatmapSize;
        if (result[2].size() != sizeof(outputs[i].heatmap_xy))
            throw std::runtime_error("Mercury heatmap size mismatch");
        // QNN 2.50 returns [1,22,22,21]; Mercury decodes [1,21,22,22].
        const auto* xy = result[2].data();
        for (size_t pixel = 0; pixel < plane; ++pixel)
            for (size_t joint = 0; joint < 21; ++joint)
                std::memcpy(&outputs[i].heatmap_xy[joint * plane + pixel],
                            xy + (pixel * 21 + joint) * sizeof(float), sizeof(float));
        copy_float_output(outputs[i].heatmap_depth, result[0]);
        copy_float_output(outputs[i].scalar_extras, result[1]);
        copy_float_output(outputs[i].curls, result[3]);
    }
}

}  // namespace mercury
