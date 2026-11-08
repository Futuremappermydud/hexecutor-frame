// Convert an offline Model batch into the existing v2 depth-array contract.
#include "calibration.hpp"
#include "model_depth_geometry.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace FrameSpatial;
namespace fs = std::filesystem;

template<class T> static std::vector<T> read(const fs::path& path, size_t n) {
    std::vector<T> result(n);
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(result.data()), n * sizeof(T)) || in.peek() != EOF)
        throw std::runtime_error("bad input size: " + path.string());
    return result;
}
template<class T> static void write(const fs::path& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!out) throw std::runtime_error("write failed: " + path.string());
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: model_batch_to_v2 CAPTURE_DIR BATCH_DIR V2_DIR FRAME_COUNT\n";
        return 2;
    }
    try {
        const fs::path capture = argv[1], batch = argv[2], out = argv[3];
        const int count = std::stoi(argv[4]);
        if (count <= 0) return 2;
        fs::create_directories(out);
        const float step = 0.29f * 3.14159265358979323846f / 180.f;
        const auto map = modelSphericalMap(step, FOCAL_BASELINE / RECT_FOCAL,
                                            RECT_WIDTH, RECT_HEIGHT, RECT_FOCAL, RECT_CX, RECT_CY);
        for (int k = 1; k <= count; ++k) {
            const std::string name = "frame-" + std::to_string(k);
            const std::string model = name + "-spherical";
            const auto disparity = read<float>(batch / (model + "-disparity.f32"), MODEL_WIDTH * MODEL_HEIGHT);
            const auto right = read<float>(batch / (model + "-right-disparity.f32"), MODEL_WIDTH * MODEL_HEIGHT);
            const auto uncertainty = read<uint8_t>(batch / (model + "-uncertainty.u8"), MODEL_WIDTH * MODEL_HEIGHT);
            const auto rightUnc = read<uint8_t>(batch / (model + "-right-uncertainty.u8"), MODEL_WIDTH * MODEL_HEIGHT);
            const auto leftInput = read<uint8_t>(capture / (name + "-spherical-left.u8"), MODEL_WIDTH * MODEL_HEIGHT);
            const auto rightInput = read<uint8_t>(capture / (name + "-spherical-right.u8"), MODEL_WIDTH * MODEL_HEIGHT);
            const auto rectLeft = read<uint8_t>(capture / (name + "-rect-left.u8"), RECT_WIDTH * RECT_HEIGHT);
            const auto rectRight = read<uint8_t>(capture / (name + "-rect-right.u8"), RECT_WIDTH * RECT_HEIGHT);
            const auto depth = modelSphericalDepth(disparity, uncertainty, map, 89);
            std::vector<float> confidence(depth.size(), .05f);
            std::vector<uint8_t> valid(depth.size());
            for (size_t i = 0; i < depth.size(); ++i) {
                valid[i] = depth[i] > 0;
                const int j = map.modelIndex[i];
                if (j >= 0) confidence[i] = std::clamp((89.f - uncertainty[size_t(j)]) / 39.f, .05f, 1.f);
            }
            write(out / (name + "-left.u8"), leftInput);
            write(out / (name + "-right.u8"), rightInput);
            write(out / (name + "-rect-left.u8"), rectLeft);
            write(out / (name + "-rect-right.u8"), rectRight);
            write(out / (name + "-disparity.f32"), disparity);
            write(out / (name + "-right-disparity.f32"), right);
            write(out / (name + "-uncertainty.u8"), uncertainty);
            write(out / (name + "-right-uncertainty.u8"), rightUnc);
            write(out / (name + "-depth.f32"), depth);
            write(out / (name + "-confidence.f32"), confidence);
            write(out / (name + "-validity.u8"), valid);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
