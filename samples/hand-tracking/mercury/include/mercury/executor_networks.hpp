// Mercury's ONNX-order float32 interface over the two native QNN16 programs.
// The runner is supplied by the Hexecutor session; no model is evaluated here.
#pragma once

#include "mercury/tracker.hpp"
#include "hexecutor/executor.hpp"

#include <functional>
#include <vector>

namespace mercury {

class ExecutorNetworks final : public Networks {
public:
    using Run = std::function<std::vector<hexecutor::Bytes>(
        size_t program, const std::vector<hexecutor::Buffer>& inputs)>;

    explicit ExecutorNetworks(Run run);
    explicit ExecutorNetworks(hexecutor::Session& session);
    void detect(const DetectionInput* inputs, DetectionOutput* outputs, size_t count) override;
    void keypoints(const KeypointInput* inputs, KeypointOutput* outputs, size_t count) override;

private:
    Run run_;
};

}  // namespace mercury
