#include "gpu_depth_pipeline.hpp"
#include "calibration.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#ifdef FRAME_WITH_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#endif

namespace FrameSpatial {

struct GpuDepthPipeline::Impl {
    bool ready = false;
    float inferenceMs = 0.0f;
    std::string error;
    uint64_t frames = 0;
    bool diagnosticDumped = false;

#ifdef FRAME_WITH_MNN
    MNN::Interpreter* network = nullptr;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* output = nullptr;

    ~Impl() {
        if (network && session) network->releaseSession(session);
        if (network) MNN::Interpreter::destroy(network);
    }
#endif
};

GpuDepthPipeline::GpuDepthPipeline() : m_impl(std::make_unique<Impl>()) {}
GpuDepthPipeline::~GpuDepthPipeline() = default;

bool GpuDepthPipeline::initialize(const std::string& modelPath) {
#ifndef FRAME_WITH_MNN
    (void)modelPath;
    m_impl->error = "runtime was built without MNN Vulkan support";
    return false;
#else
    m_impl->network = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (!m_impl->network) {
        m_impl->error = "could not load " + modelPath;
        return false;
    }

    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_Low;
    backend.power = MNN::BackendConfig::Power_High;
    backend.memory = MNN::BackendConfig::Memory_Normal;

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_VULKAN;
    config.backupType = MNN_FORWARD_VULKAN; // Never silently fall back to CPU.
    config.numThread = 4;
    config.backendConfig = &backend;

    auto start = std::chrono::steady_clock::now();
    m_impl->session = m_impl->network->createSession(config);
    if (!m_impl->session) {
        m_impl->error = "MNN could not create a Vulkan session";
        return false;
    }

    int backends[2] = {-1, -1};
    m_impl->network->getSessionInfo(
        m_impl->session, MNN::Interpreter::BACKENDS, backends);
    if (backends[0] != MNN_FORWARD_VULKAN) {
        m_impl->error = "MNN session did not select Vulkan";
        return false;
    }

    m_impl->input = m_impl->network->getSessionInput(m_impl->session, "image");
    m_impl->output = m_impl->network->getSessionOutput(m_impl->session, "disparity");
    if (!m_impl->input || !m_impl->output ||
        m_impl->input->dimensions() != 4 ||
        m_impl->input->length(0) != 1 || m_impl->input->length(1) != 2) {
        m_impl->error = "unexpected StereoNet input/output tensors";
        return false;
    }

    const float setupMs = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    m_impl->network->releaseModel();
    m_impl->ready = true;
    std::cout << "[GpuDepthPipeline] MNN Vulkan backend ready in " << setupMs
              << " ms (input " << m_impl->input->length(3) << "x"
              << m_impl->input->length(2) << ", backendType=" << backends[0]
              << ")" << std::endl;
    return true;
#endif
}

bool GpuDepthPipeline::process(const std::vector<uint8_t>& left,
                               const std::vector<uint8_t>& right,
                               std::vector<float>& depthMeters) {
#ifndef FRAME_WITH_MNN
    (void)left;
    (void)right;
    (void)depthMeters;
    return false;
#else
    if (!m_impl->ready ||
        left.size() < static_cast<size_t>(RECT_WIDTH * RECT_HEIGHT) ||
        right.size() < static_cast<size_t>(RECT_WIDTH * RECT_HEIGHT)) {
        return false;
    }

    const int inputH = m_impl->input->length(2);
    const int inputW = m_impl->input->length(3);
    MNN::Tensor hostInput(m_impl->input, MNN::Tensor::CAFFE);
    float* inputData = hostInput.host<float>();
    const size_t plane = static_cast<size_t>(inputH) * inputW;

    auto resizePlane = [&](const std::vector<uint8_t>& source, float* destination) {
        for (int y = 0; y < inputH; ++y) {
            const float sourceY = (static_cast<float>(y) + 0.5f) *
                                  static_cast<float>(RECT_HEIGHT) / inputH - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(sourceY)), 0, RECT_HEIGHT - 1);
            const int y1 = std::min(y0 + 1, RECT_HEIGHT - 1);
            const float fy = std::clamp(sourceY - y0, 0.0f, 1.0f);
            for (int x = 0; x < inputW; ++x) {
                const float sourceX = (static_cast<float>(x) + 0.5f) *
                                      static_cast<float>(RECT_WIDTH) / inputW - 0.5f;
                const int x0 = std::clamp(static_cast<int>(std::floor(sourceX)), 0, RECT_WIDTH - 1);
                const int x1 = std::min(x0 + 1, RECT_WIDTH - 1);
                const float fx = std::clamp(sourceX - x0, 0.0f, 1.0f);
                const float top = source[static_cast<size_t>(y0) * RECT_WIDTH + x0] * (1.0f - fx) +
                                  source[static_cast<size_t>(y0) * RECT_WIDTH + x1] * fx;
                const float bottom = source[static_cast<size_t>(y1) * RECT_WIDTH + x0] * (1.0f - fx) +
                                     source[static_cast<size_t>(y1) * RECT_WIDTH + x1] * fx;
                destination[static_cast<size_t>(y) * inputW + x] =
                    (top * (1.0f - fy) + bottom * fy) / 255.0f;
            }
        }
    };
    resizePlane(left, inputData);
    resizePlane(right, inputData + plane);

    if (!m_impl->input->copyFromHostTensor(&hostInput)) {
        m_impl->error = "failed to upload StereoNet input";
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto status = m_impl->network->runSession(m_impl->session);
    if (status != MNN::NO_ERROR) {
        m_impl->error = "MNN Vulkan inference failed with status " +
                        std::to_string(static_cast<int>(status));
        return false;
    }
    MNN::Tensor hostOutput(m_impl->output, MNN::Tensor::CAFFE);
    if (!m_impl->output->copyToHostTensor(&hostOutput)) {
        m_impl->error = "failed to download StereoNet disparity";
        return false;
    }
    m_impl->inferenceMs = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    const int outputH = hostOutput.length(2);
    const int outputW = hostOutput.length(3);
    const float* disparity = hostOutput.host<float>();
    const float inputScaleX = static_cast<float>(inputW) / RECT_WIDTH;
    const float scaledFocalBaseline = FOCAL_BASELINE * inputScaleX;
    depthMeters.assign(static_cast<size_t>(RECT_WIDTH) * RECT_HEIGHT, 0.0f);

    for (int y = 0; y < RECT_HEIGHT; ++y) {
        const float sourceY = (static_cast<float>(y) + 0.5f) * outputH /
                              RECT_HEIGHT - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sourceY)), 0, outputH - 1);
        const int y1 = std::min(y0 + 1, outputH - 1);
        const float fy = std::clamp(sourceY - y0, 0.0f, 1.0f);
        for (int x = 0; x < RECT_WIDTH; ++x) {
            const float sourceX = (static_cast<float>(x) + 0.5f) * outputW /
                                  RECT_WIDTH - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sourceX)), 0, outputW - 1);
            const int x1 = std::min(x0 + 1, outputW - 1);
            const float fx = std::clamp(sourceX - x0, 0.0f, 1.0f);
            const float top = disparity[static_cast<size_t>(y0) * outputW + x0] * (1.0f - fx) +
                              disparity[static_cast<size_t>(y0) * outputW + x1] * fx;
            const float bottom = disparity[static_cast<size_t>(y1) * outputW + x0] * (1.0f - fx) +
                                 disparity[static_cast<size_t>(y1) * outputW + x1] * fx;
            const float d = top * (1.0f - fy) + bottom * fy;
            if (!std::isfinite(d) || d <= 0.25f) continue;
            const float depth = scaledFocalBaseline / d;
            if (depth >= 0.20f && depth <= 12.0f) {
                depthMeters[static_cast<size_t>(y) * RECT_WIDTH + x] = depth;
            }
        }
    }

    if (!m_impl->diagnosticDumped) {
        const char* dumpPath = std::getenv("FRAME_DUMP_AI_DEPTH");
        if (dumpPath && *dumpPath) {
            std::ofstream outputFile(dumpPath, std::ios::binary);
            outputFile.write(reinterpret_cast<const char*>(depthMeters.data()),
                             static_cast<std::streamsize>(depthMeters.size() * sizeof(float)));
            if (outputFile.good()) {
                m_impl->diagnosticDumped = true;
                std::cout << "[GpuDepthPipeline] Saved metric depth to " << dumpPath
                          << std::endl;
            } else {
                std::cerr << "[GpuDepthPipeline] Failed to save metric depth to "
                          << dumpPath << std::endl;
            }
        }
    }

    ++m_impl->frames;
    if (m_impl->frames == 1 || m_impl->frames % 10 == 0) {
        size_t valid = 0;
        double sum = 0.0;
        double sumSquares = 0.0;
        float minimum = 12.0f;
        float maximum = 0.0f;
        for (float z : depthMeters) {
            if (z <= 0.0f) continue;
            ++valid;
            sum += z;
            sumSquares += static_cast<double>(z) * z;
            minimum = std::min(minimum, z);
            maximum = std::max(maximum, z);
        }
        const double mean = valid ? sum / valid : 0.0;
        const double variance = valid
            ? std::max(0.0, sumSquares / valid - mean * mean) : 0.0;
        std::cout << "[GpuDepthPipeline] Vulkan AI " << m_impl->inferenceMs
                  << " ms, coverage="
                  << 100.0f * static_cast<float>(valid) / depthMeters.size()
                  << "%, range=" << (valid ? minimum : 0.0f) << ".." << maximum
                  << " m, mean=" << mean << " m, sigma=" << std::sqrt(variance)
                  << " m" << std::endl;
    }
    return true;
#endif
}

bool GpuDepthPipeline::isReady() const { return m_impl->ready; }
float GpuDepthPipeline::lastInferenceMs() const { return m_impl->inferenceMs; }
const std::string& GpuDepthPipeline::lastError() const { return m_impl->error; }

} // namespace FrameSpatial
