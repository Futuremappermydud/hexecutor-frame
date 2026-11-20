#pragma once
#include "hexecutor/executor.hpp"
#include <atomic>
#include <functional>
#include <memory>

namespace FrameSpatial {
struct WorkerOptions {
    int fd = -1;
    bool hostTemporal = false, secondDisparity = false, closeAfterReply = false;
    double idleCloseSeconds = 15;
};
class DepthInference {
  public:
    virtual ~DepthInference() = default;
    virtual std::vector<hexecutor::Bytes> run(const std::vector<hexecutor::Buffer> &) = 0;
    virtual void close() = 0;
};
using DepthFactory = std::function<std::unique_ptr<DepthInference>()>;
void serve_depth(const WorkerOptions &, const DepthFactory &, const std::atomic<bool> &stop);
} // namespace FrameSpatial
