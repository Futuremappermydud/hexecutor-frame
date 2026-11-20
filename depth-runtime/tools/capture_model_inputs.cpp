// E02 camera-only capture (no DSP/OpenXR): N fresh stereo pairs through the
// production DMA/UBWC/rectification path. Per frame it saves the 616x616
// rectified pair, the Model spherical model inputs and the resized pinhole
// model inputs of the SAME exposure, so projection modes can be compared
// offline on identical pixels. Set FRAME_DEPTH_SPHERICAL=1 (and optionally
// FRAME_DEPTH_SPHERICAL_DEG) before running.
#include "dma_capture.hpp"
#include "depth_pipeline.hpp"
#include "model_depth_geometry.hpp"
#include "calibration.hpp"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
using namespace FrameSpatial;
static int64_t monoNs() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return int64_t(t.tv_sec) * 1000000000ll + t.tv_nsec; }
template <class V> static bool put(const std::filesystem::path& p, const V& v) {
    std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char*>(v.data()), v.size()); return bool(f);
}
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " output-dir frames [spacing-ms]\n"; return 2; }
    const std::filesystem::path out = argv[1];
    const int frames = std::atoi(argv[2]);
    const int spacing = argc > 3 ? std::atoi(argv[3]) : 300;
    const bool saveColor = std::getenv("FRAME_SAVE_COLOR") && std::string(std::getenv("FRAME_SAVE_COLOR")) == "1";
    std::filesystem::create_directories(out);
    DMACapture capture;
    const auto pid = DMACapture::findCompositorPid();
    if (pid <= 0 || !capture.initialize(pid)) return 1;
    DepthPipeline rectifier;
    uint64_t lastL = 0, lastR = 0;
    int saved = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10 + frames * (spacing + 200) / 1000);
    while (saved < frames && std::chrono::steady_clock::now() < deadline) {
        const int64_t b = monoNs();
        if (!capture.captureFrame(saveColor)) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
        const int64_t e = monoNs();
        if (capture.leftHash() == lastL || capture.rightHash() == lastR) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
        lastL = capture.leftHash(); lastR = capture.rightHash();
        rectifier.rectify(capture.getLeftBuffer(), capture.getRightBuffer());
        const std::string pre = "frame-" + std::to_string(++saved);
        // FRAME_SAVE_RAW=1: also keep the detiled 2464x2464 luma for offline re-rectification.
        if (const char* raw = std::getenv("FRAME_SAVE_RAW"); raw && std::string(raw) == "1") {
            const size_t n = size_t(capture.width()) * capture.height();
            std::ofstream(out / (pre + "-raw-left.u8"), std::ios::binary).write(reinterpret_cast<const char*>(capture.getLeftBuffer()), n);
            std::ofstream(out / (pre + "-raw-right.u8"), std::ios::binary).write(reinterpret_cast<const char*>(capture.getRightBuffer()), n);
        }
        if (saveColor) {
            const size_t ySize = size_t(capture.width()) * capture.height();
            const size_t uvSize = ySize / 2;
            const auto saveNv12 = [&](const std::filesystem::path& path, const uint8_t* y, const uint8_t* uv) {
                if (!y || !uv) return false;
                std::ofstream file(path, std::ios::binary);
                file.write(reinterpret_cast<const char*>(y), ySize);
                file.write(reinterpret_cast<const char*>(uv), uvSize);
                return bool(file);
            };
            if (!saveNv12(out / (pre + "-raw-left.nv12"), capture.getLeftBuffer(), capture.getLeftUV()) ||
                !saveNv12(out / (pre + "-raw-right.nv12"), capture.getRightBuffer(), capture.getRightUV())) return 1;
        }
        const auto& rl = rectifier.getRectifiedLeft(); const auto& rr = rectifier.getRectifiedRight();
        bool ok = put(out / (pre + "-rect-left.u8"), rl) && put(out / (pre + "-rect-right.u8"), rr) &&
                  put(out / (pre + "-pinhole-left.u8"), modelResize(rl, RECT_WIDTH, RECT_HEIGHT)) &&
                  put(out / (pre + "-pinhole-right.u8"), modelResize(rr, RECT_WIDTH, RECT_HEIGHT));
        if (!rectifier.getSphericalLeft().empty())
            ok = ok && put(out / (pre + "-spherical-left.u8"), rectifier.getSphericalLeft()) &&
                 put(out / (pre + "-spherical-right.u8"), rectifier.getSphericalRight());
        if (saved == 1) {
            std::ofstream(out / "calibration-history-at-init.json") << rectifier.calibrationSource();
            std::ofstream(out / "calibration-used.json") << "{\"stereo_policy\":\"" << rectifier.calibrationPolicy()
                << "\",\"stereo_entry\":" << rectifier.calibrationEntry() << ",\"stereo_entries\":"
                << rectifier.calibrationEntries() << "}\n";
            const int cx = int(RECT_CX), cy = int(RECT_CY), f = int(RECT_FOCAL);
            const auto origin = rectifier.rectifiedPointInHead(cx, cy, 0);
            const auto forward = rectifier.rectifiedPointInHead(cx, cy, 1);
            const auto right = rectifier.rectifiedPointInHead(cx + f, cy, 1);
            const auto down = rectifier.rectifiedPointInHead(cx, cy + f, 1);
            std::ofstream extrinsic(out / "rectified-camera-in-head.json");
            extrinsic << std::setprecision(9) << "{\"rotation_row_major\":[";
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) {
                    const auto& axis = col == 0 ? right : col == 1 ? down : forward;
                    const float value = axis[row] - (col == 2 ? origin[row] : forward[row]);
                    extrinsic << (row || col ? "," : "") << value;
                }
            extrinsic << "],\"translation_m\":[" << origin[0] << ',' << origin[1] << ',' << origin[2] << "]}\n";
            if (!extrinsic) return 1;
        }
        const auto* timed = capture.timedPair();
        std::ofstream j(out / (pre + ".json"));
        j << "{\"sequence\":" << saved << ",\"capture_begin_ns\":" << b << ",\"capture_end_ns\":" << e
          << ",\"left_slot\":" << capture.leftSlot() << ",\"right_slot\":" << capture.rightSlot()
          << ",\"spherical_step_rad\":" << rectifier.sphericalStepRadians() << ",\"timed\":" << (timed ? "true" : "false");
        if (timed)
            j << ",\"camera_sequence\":" << timed->sequence << ",\"left_end_of_readout_ns\":" << timed->leftEndOfReadoutNs
              << ",\"right_end_of_readout_ns\":" << timed->rightEndOfReadoutNs << ",\"left_write_start_ns\":"
              << timed->leftWriteStartNs << ",\"right_write_start_ns\":" << timed->rightWriteStartNs
              << ",\"copy_begin_ns\":" << timed->copyBeginNs << ",\"copy_end_ns\":" << timed->copyEndNs
              << ",\"skipped_sequences\":" << timed->skippedSequences;
        j << ",\"raw_color\":" << (saveColor ? "true" : "false")
          << ",\"raw_color_format\":\"NV12\",\"raw_color_shape\":[" << capture.height()
          << ',' << capture.width() << "]}\n";
        if (!ok || !j) return 1;
        std::cout << pre << " slots " << capture.leftSlot() << "/" << capture.rightSlot() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(spacing));
    }
    return saved == frames ? 0 : 1;
}
