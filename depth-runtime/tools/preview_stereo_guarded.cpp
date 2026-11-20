// Opt-in live comparison card. Production matcher and 3D/occlusion paths remain independent.
#include "dma_capture.hpp"
#include "depth_pipeline.hpp"
#include "depth_speckle_filter.hpp"
#include "overlay_renderer.hpp"
#include "stereo_matcher_experiment.hpp"
#include "vulkan_stereo_pipeline.hpp"
#include "vulkan_guarded_stereo_pipeline.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <omp.h>
namespace SE = Spatial::StereoExperiment;
using Clock = std::chrono::steady_clock;
std::atomic<bool> running{true};
void stop(int) { running.store(false); }
struct Comparison {
    std::vector<uint8_t> image;
    std::vector<float> baseline, experiment;
    Clock::time_point captured;
    unsigned generation = 0;
    double computeMs = 0;
};
int main(int argc, char** argv) {
    int seconds = 900; bool useCpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--cpu") useCpu = true;
        else if (flag == "--seconds" && i + 1 < argc) seconds = std::stoi(argv[++i]);
        else { std::cerr << "Usage: preview_stereo_guarded [--cpu] [--seconds 900]\n"; return 2; }
    }
    if (seconds <= 0 || seconds > 7200) return 2;
    std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
    vr::EVRInitError error = vr::VRInitError_None;
    auto* system = vr::VR_Init(&error, vr::VRApplication_Overlay);
    if (error != vr::VRInitError_None) {
        std::cerr << vr::VR_GetVRInitErrorAsEnglishDescription(error) << '\n'; return 1;
    }
    FrameSpatial::OverlayRenderer renderer;
    constexpr const char* key = "frame.spatial.guarded_preview.vulkan";
    if (!renderer.initialize(system, 1024, 444, key)) { vr::VR_Shutdown(); return 1; }
    vr::VROverlayHandle_t handle = vr::k_ulOverlayHandleInvalid;
    if (vr::VROverlay()->FindOverlay(key, &handle) == vr::VROverlayError_None)
        vr::VROverlay()->SetOverlayWidthInMeters(handle, 1.1f);
    std::mutex mutex;
    std::shared_ptr<const Comparison> latest;
    std::string workerStatus = "WAITING FOR CAMERAS";
    std::atomic<bool> gpuBusy{!useCpu};
    std::thread worker([&] {
        try {
            // Leave CPU capacity for compositor and render/pose polling.
            omp_set_num_threads(useCpu ? 6 : 2);
            FrameSpatial::VulkanStereoPipeline baselineGpu;
            FrameSpatial::VulkanGuardedStereoPipeline guardedGpu;
            if (!useCpu) {
                if (!baselineGpu.initialize("shaders") || !guardedGpu.initialize("shaders"))
                    throw std::runtime_error("GPU initialization: " + baselineGpu.lastError() + guardedGpu.lastError());
                gpuBusy.store(false);
            }
            FrameSpatial::DMACapture capture;
            FrameSpatial::DepthPipeline rectifier;
            SE::Config c; c.supportRadius = 8; c.edgeThreshold = 32;
            c.supportGuard = 3; c.supportExposure = true;
            unsigned generation = 0;
            std::vector<uint8_t> previousLeft, previousRight;
            auto retry = Clock::now() - std::chrono::seconds(3);
            while (running.load()) {
                if (!capture.isReady()) {
                    if (Clock::now() - retry >= std::chrono::seconds(2)) {
                        retry = Clock::now();
                        const auto pid = FrameSpatial::DMACapture::findCompositorPid();
                        if (pid > 0) capture.initialize(pid);
                    }
                    if (!capture.isReady()) { std::this_thread::sleep_for(std::chrono::milliseconds(25)); continue; }
                }
                if (!capture.captureFrame(false)) {
                    capture.release();
                    { std::lock_guard<std::mutex> lock(mutex); workerStatus = "WAITING FOR CAMERAS / LAST PAIR RETAINED"; }
                    continue;
                }
                auto frame = std::make_shared<Comparison>();
                frame->captured = Clock::now();
                const auto start = frame->captured;
                rectifier.rectify(capture.getLeftBuffer(), capture.getRightBuffer());
                const auto& left = rectifier.getRectifiedLeft();
                const auto& right = rectifier.getRectifiedRight();
                if (left == previousLeft && right == previousRight) {
                    { std::lock_guard<std::mutex> lock(mutex); workerStatus = "CAMERAS UNCHANGED / LAST PAIR"; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(125));
                    continue;
                }
                previousLeft = left; previousRight = right;
                frame->image = left;
                if (useCpu) {
                    const auto baseline = SE::match(left, right, c, SE::Method::Baseline);
                    if (!running.load()) break;
                    const auto guarded = SE::match(left, right, c, SE::Method::GuardedAdaptive);
                    frame->baseline = baseline.depth; frame->experiment = guarded.depth;
                } else {
                    gpuBusy.store(true);
                    if (!baselineGpu.process(left, right, frame->baseline) ||
                        !guardedGpu.process(left, right, frame->experiment))
                        throw std::runtime_error("GPU matching: " + baselineGpu.lastError() + guardedGpu.lastError());
                    gpuBusy.store(false);
                }
                FrameSpatial::removeDepthSpeckles(frame->baseline, c.width, c.height);
                FrameSpatial::removeDepthSpeckles(frame->experiment, c.width, c.height);
                frame->generation = ++generation;
                frame->computeMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                { std::lock_guard<std::mutex> lock(mutex); latest = frame; workerStatus = "LIVE"; }
                std::cout << "[Preview] pair=" << generation << " compute_ms=" << frame->computeMs
                          << " slots L" << capture.leftSlot() << "/R" << capture.rightSlot()
                          << " hashes L" << capture.leftHash() << "/R" << capture.rightHash()
                          << " backend=" << (useCpu ? "CPU" : "VULKAN GPU") << std::endl;
                // Give the overlay queue a publication window between GPU jobs.
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (const std::exception& e) {
            gpuBusy.store(false);
            std::lock_guard<std::mutex> lock(mutex);
            workerStatus = "WORKER ERROR / CHECK PREVIEW LOG";
            std::cerr << "[Preview] worker error: " << e.what() << std::endl;
        }
    });
    const auto deadline = Clock::now() + std::chrono::seconds(seconds);
    auto fpsTime = Clock::now(); unsigned loops = 0, submissions = 0; double fps = 0;
    std::vector<uint8_t> emptyImage(616*616, 0);
    std::vector<float> emptyDepth(616*616, 0);
    while (running.load() && Clock::now() < deadline) {
        vr::VREvent_t event;
        while (system->PollNextEvent(&event, sizeof(event))) if (event.eventType == vr::VREvent_Quit) running.store(false);
        renderer.getHmdPose();
        std::shared_ptr<const Comparison> frame; std::string status;
        { std::lock_guard<std::mutex> lock(mutex); frame = latest; status = workerStatus; }
        char line[160];
        if (!gpuBusy.load() && frame) {
            const double age = std::chrono::duration<double>(Clock::now() - frame->captured).count();
            std::snprintf(line, sizeof(line), "%s PAIR %u / AGE %.1FS / COMPUTE %.0FMS / LOOP %.0FFPS", status.c_str(), frame->generation, age, frame->computeMs, fps);
            if (renderer.renderComparison(frame->image, frame->baseline, frame->experiment, 616, 616,
                                          useCpu ? "LIVE CPU COMPARISON / AFTER CLEANUP" : "LIVE GPU COMPARISON / AFTER CLEANUP", line)) ++submissions;
        } else if (!gpuBusy.load() && renderer.renderComparison(emptyImage, emptyDepth, emptyDepth, 616, 616,
                                              "LIVE STEREO COMPARISON", status)) ++submissions;
        ++loops;
        const double elapsed = std::chrono::duration<double>(Clock::now() - fpsTime).count();
        if (elapsed >= 2) {
            fps = loops/elapsed;
            std::cout << "[Preview] pose_loop_fps=" << fps << " texture_submissions=" << submissions << " status=" << status << std::endl;
            loops = 0; fpsTime = Clock::now();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    running.store(false); worker.join(); renderer.shutdown(); vr::VR_Shutdown();
    return 0;
}
