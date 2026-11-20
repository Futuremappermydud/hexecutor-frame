// Standalone check for the persistent-depth path.
//
//   c++ -O3 -std=c++17 -Iinclude src/persistent_depth.cpp \
//       tools/test_persistent_depth.cpp -o /tmp/test_persistent_depth
//   /tmp/test_persistent_depth
//   /tmp/test_persistent_depth left616.raw right616.raw
//
// The self-test builds a textured wall at 1.50 m, matches it from two
// viewpoints, then turns 90 degrees onto a different wall. The first wall
// has to stay, within a few centimetres, after it leaves the cameras.
// Raw inputs are 616x616 rectified luma, one byte per pixel.

#include "persistent_depth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 616;
constexpr int kHeight = 616;
constexpr float kFocal = 380.0f;
constexpr float kPrincipal = 308.0f;
constexpr float kBaseline = 0.0639016f;
constexpr float kFocalBaseline = kFocal * kBaseline;

uint8_t shade(float x, float y) {
    const int ix = static_cast<int>(std::floor(x * 250.0f));
    const int iy = static_cast<int>(std::floor(y * 250.0f));
    const int block = ((ix / 5) ^ (iy / 5)) & 1 ? 180 : 50;
    const int grain = (ix * 13 + iy * 7) & 31;
    return static_cast<uint8_t>(std::clamp(block + grain - 10, 0, 255));
}

void render(std::vector<uint8_t>& image, float cameraX, float cameraY, float cameraZ,
            float yaw, float baseline, int axis, float plane) {
    image.assign(size_t(kWidth) * kHeight, 0);
    const float cosine = std::cos(yaw);
    const float sine = std::sin(yaw);
    const float originX = cameraX + baseline * cosine;
    const float originY = cameraY;
    const float originZ = cameraZ - baseline * sine;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const float rx = (float(x) - kPrincipal) / kFocal;
            const float ry = (float(y) - kPrincipal) / kFocal;
            const float dirX = cosine * rx + sine;
            const float dirY = ry;
            const float dirZ = -sine * rx + cosine;
            const float along = axis == 0 ? dirX : dirZ;
            const float origin = axis == 0 ? originX : originZ;
            if (std::fabs(along) < 1.0e-4f) continue;
            const float distance = (plane - origin) / along;
            if (!(distance >= 0.25f && distance <= 5.0f)) continue;
            image[size_t(y) * kWidth + x] =
                shade(originX + dirX * distance, originY + dirY * distance);
        }
    }
}

float medianOf(std::vector<float> values) {
    if (values.empty()) return 0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

int selfTest() {
    std::vector<uint8_t> left, right;
    std::vector<float> depth(size_t(kWidth) * kHeight);
    FrameSpatial::PersistentMap map;

    auto capture = [&](float x, float y, float z, float yaw, int axis, float plane) {
        render(left, x, y, z, yaw, 0.0f, axis, plane);
        render(right, x, y, z, yaw, kBaseline, axis, plane);
        FrameSpatial::matchSgbm(left.data(), right.data(), kWidth, kHeight,
                                 kFocalBaseline, depth.data());
        map.integrate(depth.data(), kWidth, kHeight, kFocal, kPrincipal, kPrincipal,
                      kFocalBaseline, x, y, z, yaw);
    };

    capture(0.0f, 0.0f, 0.0f, 0.0f, 2, 1.50f);
    std::vector<float> center;
    int valid = 0;
    for (int y = 200; y < 420; ++y) {
        for (int x = 200; x < 420; ++x) {
            const float z = depth[size_t(y) * kWidth + x];
            if (z > 0) {
                center.push_back(z);
                ++valid;
            }
        }
    }
    const float matched = medianOf(center);
    std::cout << "view 1 center median Z " << matched << " m, coverage "
              << (100.0f * valid / (220 * 220)) << "%" << std::endl;

    capture(0.15f, 0.0f, 0.0f, 0.0f, 2, 1.50f);
    std::vector<FrameSpatial::PersistentKey> wall;
    for (const auto& entry : map.cells()) {
        if (entry.second.observations >= 2 && std::fabs(entry.second.z - 1.50f) < 0.08f)
            wall.push_back(entry.first);
    }
    std::cout << "confirmed wall cells before looking away: " << wall.size() << std::endl;

    capture(0.15f, 0.0f, 0.0f, 1.5707963f, 0, 1.65f);
    int survived = 0;
    double error = 0;
    for (const FrameSpatial::PersistentKey& key : wall) {
        const auto found = map.cells().find(key);
        if (found == map.cells().end()) continue;
        ++survived;
        error += std::fabs(found->second.z - 1.50f);
    }
    const double meanError = survived ? error / survived : 0;
    std::cout << "wall cells after looking away: " << survived
              << ", mean |Z error| " << meanError << " m" << std::endl;

    const bool pass = std::fabs(matched - 1.50f) < 0.04f && valid > 220 * 220 / 2 &&
                      wall.size() > 500 && survived == int(wall.size()) && meanError < 0.04;
    std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 1;
}

int replay(const std::string& leftPath, const std::string& rightPath) {
    auto load = [](const std::string& path, std::vector<uint8_t>& image) {
        std::ifstream input(path, std::ios::binary);
        image.resize(size_t(kWidth) * kHeight);
        input.read(reinterpret_cast<char*>(image.data()), image.size());
        return bool(input) && input.gcount() == std::streamsize(image.size());
    };
    std::vector<uint8_t> left, right;
    if (!load(leftPath, left) || !load(rightPath, right)) {
        std::cerr << "expected two 616x616 raw luma files\n";
        return 1;
    }
    std::vector<float> depth(left.size());
    FrameSpatial::matchSgbm(left.data(), right.data(), kWidth, kHeight, kFocalBaseline,
                             depth.data());
    std::vector<float> values;
    for (float z : depth)
        if (z > 0) values.push_back(z);
    std::cout << "valid " << values.size() << " / " << depth.size()
              << ", median Z " << medianOf(values) << " m" << std::endl;
    std::ofstream preview("persistent-depth-preview.pgm", std::ios::binary);
    preview << "P5\n" << kWidth << " " << kHeight << "\n255\n";
    for (float z : depth) {
        const int gray = z > 0 ? std::clamp(int(z / 5.0f * 255.0f), 0, 255) : 0;
        preview.put(static_cast<char>(gray));
    }
    std::cout << "wrote persistent-depth-preview.pgm (brighter is farther, 5 m is white)\n";
    return values.empty() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--self-test")) return selfTest();
    if (argc == 3) return replay(argv[1], argv[2]);
    std::cerr << "usage: test_persistent_depth [--self-test]\n"
                 "       test_persistent_depth left616.raw right616.raw\n";
    return 2;
}
