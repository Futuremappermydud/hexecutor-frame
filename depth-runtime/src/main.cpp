#include "dma_capture.hpp"
#include "depth_pipeline.hpp"
#include "gpu_depth_pipeline.hpp"
#include "vulkan_stereo_pipeline.hpp"
#include "overlay_renderer.hpp"
#include "calibration.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <openvr.h>

static volatile bool g_running = true;
void sigHandler(int) { g_running = false; }

static FrameSpatial::RenderSnapshot g_sharedSnap;
static std::mutex g_snapMutex;
static std::atomic<bool> g_hasNewSnap(false);
static std::atomic<bool> g_gpuInferenceBusy(false);

static void publishStatus(const std::string& status, bool camerasReady) {
    std::lock_guard<std::mutex> lock(g_snapMutex);
    g_sharedSnap.status = status;
    g_sharedSnap.camerasReady = camerasReady;
    g_hasNewSnap.store(true);
}

void cameraWorker(FrameSpatial::DMACapture* dmaCapture) {
    std::cout << "[Worker] Camera ingestion thread started." << std::endl;
    FrameSpatial::DepthPipeline depthPipeline;
    FrameSpatial::GpuDepthPipeline gpuDepth;
    FrameSpatial::VulkanStereoPipeline vulkanStereo;

    struct AiExchange {
        std::mutex mutex;
        std::condition_variable wake;
        std::vector<uint8_t> pendingLeft;
        std::vector<uint8_t> pendingRight;
        std::vector<float> latestDepth;
        bool pending = false;
        bool stop = false;
        bool failed = false;
        bool busy = false;
        uint64_t outputGeneration = 0;
        float inferenceMs = 0.0f;
    } ai;

    const char* backendEnv = std::getenv("FRAME_DEPTH_BACKEND");
    const std::string backend = backendEnv && *backendEnv
        ? backendEnv : "vulkan-stereo";
    const bool requestCpu = backend == "cpu";
    const bool requestLearned = backend == "vulkan-ai";
    const bool requestGpu = !requestCpu;
    const char* modelEnv = std::getenv("FRAME_AI_MODEL");
    const std::string modelPath = modelEnv && *modelEnv
        ? modelEnv : "bin/models/stereonet-vulkan-256.mnn";
    const char* intervalEnv = std::getenv("FRAME_AI_COOLDOWN_MS");
    const int aiCooldownMs = intervalEnv
        ? std::clamp(std::atoi(intervalEnv), 0, 5000)
        : (requestLearned ? 350 : 0);
    const bool gpuReady = requestGpu && (requestLearned
        ? gpuDepth.initialize(modelPath)
        : vulkanStereo.initialize("shaders"));
    const std::string gpuLabel = requestLearned ? "VULKAN AI" : "VULKAN STEREO";
    if (requestGpu && !gpuReady) {
        const std::string& reason = requestLearned
            ? gpuDepth.lastError() : vulkanStereo.lastError();
        std::cerr << "[Worker] " << gpuLabel << " unavailable (" << reason
                  << "); using calibrated Census depth" << std::endl;
    }

    std::thread aiWorker;
    bool gpuDepthDumped = false;
    if (gpuReady) {
        aiWorker = std::thread([&] {
            while (true) {
                std::vector<uint8_t> left;
                std::vector<uint8_t> right;
                {
                    std::unique_lock<std::mutex> lock(ai.mutex);
                    ai.wake.wait(lock, [&] { return ai.stop || ai.pending; });
                    if (ai.stop) break;
                    left = std::move(ai.pendingLeft);
                    right = std::move(ai.pendingRight);
                    ai.pending = false;
                    ai.busy = true;
                }

                std::vector<float> result;
                g_gpuInferenceBusy.store(true);
                const bool succeeded = requestLearned
                    ? gpuDepth.process(left, right, result)
                    : vulkanStereo.process(left, right, result);
                if (!succeeded) {
                    g_gpuInferenceBusy.store(false);
                    std::lock_guard<std::mutex> lock(ai.mutex);
                    ai.failed = true;
                    ai.busy = false;
                    std::cerr << "[Worker] " << gpuLabel << " stopped: "
                              << (requestLearned ? gpuDepth.lastError()
                                                 : vulkanStereo.lastError())
                              << std::endl;
                    break;
                }
                g_gpuInferenceBusy.store(false);
                if (!gpuDepthDumped) {
                    const char* dumpPath = std::getenv("FRAME_DUMP_GPU_DEPTH");
                    if (dumpPath && *dumpPath) {
                        std::ofstream output(dumpPath, std::ios::binary);
                        output.write(reinterpret_cast<const char*>(result.data()),
                                     static_cast<std::streamsize>(
                                         result.size() * sizeof(float)));
                        gpuDepthDumped = output.good();
                        if (gpuDepthDumped) {
                            std::cout << "[Worker] Saved GPU depth to " << dumpPath
                                      << std::endl;
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(ai.mutex);
                    ai.latestDepth = std::move(result);
                    ai.inferenceMs = requestLearned
                        ? gpuDepth.lastInferenceMs() : vulkanStereo.lastComputeMs();
                    ++ai.outputGeneration;
                    ai.busy = false;
                }
                if (aiCooldownMs > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(aiCooldownMs));
                }
            }
        });
    }

    uint64_t consumedAiGeneration = 0;
    std::vector<float> latestAiDepth;
    float latestAiMs = 0.0f;
    auto lastRetry = std::chrono::steady_clock::now() - std::chrono::seconds(3);

    while (g_running) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!dmaCapture->isReady()) {
            auto now = std::chrono::steady_clock::now();
            if (now - lastRetry >= std::chrono::seconds(2)) {
                lastRetry = now;
                pid_t pid = FrameSpatial::DMACapture::findCompositorPid();
                if (pid != 0) {
                    std::cout << "[Worker] Retrying DMA capture from vrcompositor pid " << pid << std::endl;
                    dmaCapture->initialize(pid, -1, -1);
                } else {
                    std::cout << "[Worker] vrcompositor not found; waiting..." << std::endl;
                }
            }
            publishStatus("WAITING FOR CAMERAS", false);
            usleep(100000);
            continue;
        }

        if (!dmaCapture->captureFrame()) {
            publishStatus("DETILE FAILED", false);
            usleep(50000);
            continue;
        }

        static uint32_t frameIndex = 0;
        frameIndex++;

        auto yuvToRgba = [](int Y, int U, int V) -> uint32_t {
            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;
            int r = (298 * C + 409 * E + 128) >> 8;
            int g = (298 * C - 100 * D - 208 * E + 128) >> 8;
            int b = (298 * C + 516 * D + 128) >> 8;
            r = std::clamp(r, 0, 255);
            g = std::clamp(g, 0, 255);
            b = std::clamp(b, 0, 255);
            return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | 0xFF000000u;
        };
        auto downscaleRgb = [&](const uint8_t* y, const uint8_t* uv, int sw, int sh,
                                int dw, int dh, std::vector<uint32_t>& dst) {
            dst.resize((size_t)dw * dh);
            if (!y) {
                std::fill(dst.begin(), dst.end(), 0xFF101018u);
                return;
            }
            int uvW = sw / 2;
            for (int yy = 0; yy < dh; yy++) {
                int sy = yy * sh / dh;
                for (int xx = 0; xx < dw; xx++) {
                    int sx = xx * sw / dw;
                    int Yv = y[sy * sw + sx];
                    if (uv) {
                        const uint8_t* p = uv + ((sy / 2) * uvW + (sx / 2)) * 2;
                        dst[yy * dw + xx] = yuvToRgba(Yv, p[0], p[1]);
                    } else {
                        dst[yy * dw + xx] = (uint32_t)Yv | ((uint32_t)Yv << 8) |
                                            ((uint32_t)Yv << 16) | 0xFF000000u;
                    }
                }
            }
        };

        std::vector<uint32_t> nextL, nextR;
        downscaleRgb(dmaCapture->getLeftBuffer(), dmaCapture->getLeftUV(),
                     dmaCapture->width(), dmaCapture->height(),
                     616, 616, nextL);
        downscaleRgb(dmaCapture->getRightBuffer(), dmaCapture->getRightUV(),
                     dmaCapture->width(), dmaCapture->height(),
                     616, 616, nextR);

        auto rotate90 = [](std::vector<uint32_t>& image, bool clockwise) {
            std::vector<uint32_t> rotated(616u * 616u);
            for (int y = 0; y < 616; ++y) {
                for (int x = 0; x < 616; ++x) {
                    int sx = clockwise ? y : 615 - y;
                    int sy = clockwise ? 615 - x : x;
                    rotated[(size_t)y * 616 + x] = image[(size_t)sy * 616 + sx];
                }
            }
            image.swap(rotated);
        };
        rotate90(nextL, false); // Left sensor image is clockwise; correct counter-clockwise.
        rotate90(nextR, true);  // Right sensor image is counter-clockwise; correct clockwise.

        FrameSpatial::HmdPose depthPose{};
        bool useGpuThisFrame = gpuReady;
        bool gpuBusyThisFrame = false;
        if (gpuReady) {
            bool aiFailed = false;
            {
                std::lock_guard<std::mutex> lock(ai.mutex);
                aiFailed = ai.failed;
                gpuBusyThisFrame = ai.busy;
            }
            useGpuThisFrame = !aiFailed;
        }

        if (useGpuThisFrame) {
            depthPipeline.rectify(dmaCapture->getLeftBuffer(), dmaCapture->getRightBuffer());
            {
                std::lock_guard<std::mutex> lock(ai.mutex);
                // Keep exactly one queued pair and overwrite it with the newest
                // camera frame while inference is running. This bounds latency
                // and memory without ever blocking camera ingestion.
                ai.pendingLeft = depthPipeline.getRectifiedLeft();
                ai.pendingRight = depthPipeline.getRectifiedRight();
                ai.pending = true;
                if (ai.outputGeneration != consumedAiGeneration) {
                    latestAiDepth = ai.latestDepth;
                    latestAiMs = ai.inferenceMs;
                    consumedAiGeneration = ai.outputGeneration;
                }
            }
            ai.wake.notify_one();
        } else {
            depthPipeline.process(dmaCapture->getLeftBuffer(), dmaCapture->getRightBuffer(), depthPose);
        }

        // Optional one-shot diagnostic capture of the calibrated, rectified
        // luma pair. Keeping this behind an environment variable avoids disk
        // I/O in the normal real-time path.
        static bool rectifiedPairDumped = false;
        if (!rectifiedPairDumped) {
            const char* dumpPrefix = std::getenv("FRAME_DUMP_RECTIFIED_PREFIX");
            if (dumpPrefix && *dumpPrefix) {
                const auto& rectL = depthPipeline.getRectifiedLeft();
                const auto& rectR = depthPipeline.getRectifiedRight();
                const std::string leftPath = std::string(dumpPrefix) + "_left.raw";
                const std::string rightPath = std::string(dumpPrefix) + "_right.raw";
                std::ofstream leftOut(leftPath, std::ios::binary);
                std::ofstream rightOut(rightPath, std::ios::binary);
                leftOut.write(reinterpret_cast<const char*>(rectL.data()), rectL.size());
                rightOut.write(reinterpret_cast<const char*>(rectR.data()), rectR.size());
                rectifiedPairDumped = leftOut.good() && rightOut.good();
                if (rectifiedPairDumped) {
                    std::cout << "[Worker] Saved rectified stereo pair to "
                              << leftPath << " and " << rightPath << std::endl;
                } else {
                    std::cerr << "[Worker] Failed to save rectified stereo pair with prefix "
                              << dumpPrefix << std::endl;
                }
            }
        }
        const auto& depthMap = useGpuThisFrame ? latestAiDepth : depthPipeline.getDepthMap();
        const size_t validDepth = static_cast<size_t>(std::count_if(
            depthMap.begin(), depthMap.end(), [](float z) { return z > 0.0f; }));

        auto t1 = std::chrono::high_resolution_clock::now();
        float captureHz = 1.0f / std::max(0.001, std::chrono::duration<double>(t1 - t0).count());
        {
            std::lock_guard<std::mutex> lock(g_snapMutex);
            g_sharedSnap.previewL = std::move(nextL);
            g_sharedSnap.previewR = std::move(nextR);
            g_sharedSnap.depth = depthMap;
            g_sharedSnap.rectW = FrameSpatial::RECT_WIDTH;
            g_sharedSnap.rectH = FrameSpatial::RECT_HEIGHT;
            g_sharedSnap.frameIndex = frameIndex;
            g_sharedSnap.previewW = 616;
            g_sharedSnap.previewH = 616;
            g_sharedSnap.sourceW = dmaCapture->width();
            g_sharedSnap.sourceH = dmaCapture->height();
            g_sharedSnap.leftSlot = dmaCapture->leftSlot();
            g_sharedSnap.rightSlot = dmaCapture->rightSlot();
            g_sharedSnap.captureHz = captureHz;
            if (useGpuThisFrame) {
                if (validDepth > depthMap.size() / 100) {
                    g_sharedSnap.status = gpuLabel + " " +
                        std::to_string(static_cast<int>(latestAiMs)) + "MS";
                } else {
                    g_sharedSnap.status = gpuLabel + " WARMING";
                }
            } else {
                g_sharedSnap.status = validDepth > depthMap.size() / 100
                    ? "LIVE METRIC DEPTH" : "DEPTH CALIBRATING";
            }
            g_sharedSnap.camerasReady = true;
            g_sharedSnap.gpuBusy = useGpuThisFrame && gpuBusyThisFrame;
            g_hasNewSnap.store(true);
        }
    }

    if (aiWorker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(ai.mutex);
            ai.stop = true;
        }
        ai.wake.notify_one();
        aiWorker.join();
    }

    std::cout << "[Worker] Camera ingestion thread stopped." << std::endl;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    std::cout << "========================================================\n"
              << "       STEAM FRAME REAL-TIME SPATIAL ROOM MESH         \n"
              << "========================================================\n";

    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *vrSys = vr::VR_Init(&err, vr::VRApplication_Overlay);
    if (err != vr::VRInitError_None) {
        std::cerr << "VR_Init failed: " << vr::VR_GetVRInitErrorAsEnglishDescription(err) << std::endl;
        return 1;
    }

    // Frame's compositor defaults this policy to 1 and changes passthrough to
    // a half-resolution path when another overlay/app has visual focus. Keep
    // the IMX616 producer at its native resolution so an external DMA consumer
    // never has to reinterpret a live allocation after an eye-gaze transition.
    vr::EVRSettingsError settingsErr = vr::VRSettingsError_None;
    vr::VRSettings()->SetInt32("camera", "autoLowerPassthroughResolution", 0, &settingsErr);
    if (settingsErr == vr::VRSettingsError_None) {
        std::cout << "[Main] Disabled compositor automatic passthrough downscaling" << std::endl;
    } else {
        std::cerr << "[Main] Could not disable automatic passthrough downscaling: "
                  << vr::VRSettings()->GetSettingsErrorNameFromEnum(settingsErr) << std::endl;
    }

    FrameSpatial::OverlayRenderer renderer;
    if (!renderer.initialize(vrSys, 768, 444)) {
        std::cerr << "Failed to initialize overlay renderer!" << std::endl;
        vr::VR_Shutdown();
        return 1;
    }

    FrameSpatial::DMACapture dmaCapture;
    int rightFd = -1;
    int leftFd = -1;
    if (argc >= 3) {
        rightFd = std::atoi(argv[1]);
        leftFd = std::atoi(argv[2]);
    }

    pid_t compPid = FrameSpatial::DMACapture::findCompositorPid();
    if (compPid == 0) {
        std::cerr << "[Main] vrcompositor not running yet; overlay is live, worker will retry." << std::endl;
        publishStatus("WAITING FOR COMPOSITOR", false);
    } else if (!dmaCapture.initialize(compPid, rightFd, leftFd)) {
        std::cerr << "[Main] Initial DMA-BUF capture failed; overlay is live, worker will retry." << std::endl;
        publishStatus("WAITING FOR CAMERAS", false);
    } else {
        publishStatus("LIVE", true);
    }

    std::thread worker(cameraWorker, &dmaCapture);

    FrameSpatial::RenderSnapshot localSnap;
    localSnap.status = "BOOT";
    auto lastFpsTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float liveFps = 0.0f;

    std::cout << "[Main] Stereoscopic overlay active in headset." << std::endl;

    while (g_running) {
        vr::VREvent_t vrEvent;
        while (vrSys->PollNextEvent(&vrEvent, sizeof(vrEvent))) {
            if (vrEvent.eventType == vr::VREvent_Quit) {
                g_running = false;
                break;
            }
        }

        if (g_hasNewSnap.load()) {
            std::lock_guard<std::mutex> lock(g_snapMutex);
            localSnap = g_sharedSnap;
            g_hasNewSnap.store(false);
        }

        if (!g_gpuInferenceBusy.load()) {
            renderer.render(localSnap, liveFps);
        }

        frameCount++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastFpsTime).count();
        if (elapsed >= 0.5) {
            liveFps = (float)(frameCount / elapsed);
            frameCount = 0;
            lastFpsTime = now;
            std::cout << "[Render] FPS: " << liveFps
                      << " | CaptureHz: " << localSnap.captureHz
                      << " | Frame: " << localSnap.frameIndex
                      << " | slots L" << localSnap.leftSlot << "/R" << localSnap.rightSlot
                      << " | " << localSnap.status << std::endl;
        }

        usleep(11000);
    }

    std::cout << "[Main] Shutting down..." << std::endl;
    worker.join();
    dmaCapture.release();
    renderer.shutdown();
    vr::VR_Shutdown();

    std::cout << "[Main] Clean exit." << std::endl;
    return 0;
}
