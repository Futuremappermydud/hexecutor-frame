#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>

struct ImageGrey {
    int w, h;
    std::vector<uint8_t> data;
    uint8_t at(int x, int y) const { return data[y * w + x]; }
};

ImageGrey loadPPMAsGrey(const std::string& path, int targetW = 480, int targetH = 540) {
    std::ifstream f(path, std::ios::binary);
    std::string header;
    int srcW, srcH, maxVal;
    f >> header >> srcW >> srcH >> maxVal;
    f.get();

    std::vector<uint8_t> raw(srcW * srcH * 3);
    f.read(reinterpret_cast<char*>(raw.data()), raw.size());

    ImageGrey img;
    img.w = targetW;
    img.h = targetH;
    img.data.resize(targetW * targetH);

    float sx = static_cast<float>(srcW) / targetW;
    float sy = static_cast<float>(srcH) / targetH;

    for (int y = 0; y < targetH; y++) {
        int sy_idx = std::min(static_cast<int>(y * sy), srcH - 1);
        for (int x = 0; x < targetW; x++) {
            int sx_idx = std::min(static_cast<int>(x * sx), srcW - 1);
            int idx = (sy_idx * srcW + sx_idx) * 3;
            uint8_t grey = static_cast<uint8_t>(0.299f * raw[idx] + 0.587f * raw[idx+1] + 0.114f * raw[idx+2]);
            img.data[y * targetW + x] = grey;
        }
    }
    return img;
}

// Compute 5x5 Census Transform (64-bit bitstring per pixel)
std::vector<uint64_t> computeCensus5x5(const ImageGrey& img) {
    std::vector<uint64_t> census(img.w * img.h, 0);
    for (int y = 2; y < img.h - 2; y++) {
        for (int x = 2; x < img.w - 2; x++) {
            uint8_t center = img.at(x, y);
            uint64_t code = 0;
            int bit = 0;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (img.at(x + dx, y + dy) >= center) {
                        code |= (1ULL << bit);
                    }
                    bit++;
                }
            }
            census[y * img.w + x] = code;
        }
    }
    return census;
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    int W = 480, H = 540;
    ImageGrey left = loadPPMAsGrey("/home/steamos/test_openvr/left.ppm", W, H);
    ImageGrey right = loadPPMAsGrey("/home/steamos/test_openvr/right.ppm", W, H);

    std::cout << "Computing Census Transform..." << std::endl;
    std::vector<uint64_t> censusL = computeCensus5x5(left);
    std::vector<uint64_t> censusR = computeCensus5x5(right);

    const int maxDisparity = 64;
    std::vector<float> disparity(W * H, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int y = 3; y < H - 3; y++) {
        for (int x = 3 + maxDisparity; x < W - 3; x++) {
            int bestCost = 1000;
            int bestD = 0;
            int secondCost = 1000;

            for (int d = 0; d < maxDisparity; d++) {
                // Hamming distance of 3x1 block census
                uint64_t cL0 = censusL[y * W + x];
                uint64_t cR0 = censusR[y * W + (x - d)];
                uint64_t cL1 = censusL[(y - 1) * W + x];
                uint64_t cR1 = censusR[(y - 1) * W + (x - d)];
                uint64_t cL2 = censusL[(y + 1) * W + x];
                uint64_t cR2 = censusR[(y + 1) * W + (x - d)];

                int cost = __builtin_popcountll(cL0 ^ cR0) + 
                           __builtin_popcountll(cL1 ^ cR1) + 
                           __builtin_popcountll(cL2 ^ cR2);

                if (cost < bestCost) {
                    secondCost = bestCost;
                    bestCost = cost;
                    bestD = d;
                } else if (cost < secondCost) {
                    secondCost = cost;
                }
            }

            // Uniqueness ratio filter: discard ambiguous matches in flat drywall
            if (bestCost > 25 || (secondCost > 0 && (float)bestCost / secondCost > 0.92f)) {
                disparity[y * W + x] = 0.0f;
            } else {
                disparity[y * W + x] = static_cast<float>(bestD);
            }
        }
    }

    // 3x3 Median filter to remove outliers
    std::vector<float> filtered = disparity;
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            float vals[9];
            int n = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float v = disparity[(y + dy) * W + (x + dx)];
                    if (v > 0.0f) vals[n++] = v;
                }
            }
            if (n >= 5) {
                std::sort(vals, vals + n);
                filtered[y * W + x] = vals[n / 2];
            } else {
                filtered[y * W + x] = 0.0f;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Census Stereo Depth computed in " << ms << " ms (" << (1000.0 / ms) << " FPS)!" << std::endl;

    std::ofstream outPPM("/home/steamos/test_openvr/dense_depth.ppm", std::ios::binary);
    outPPM << "P6\n" << W << " " << H << "\n255\n";

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float d = filtered[y * W + x];
            if (d < 1.0f) {
                outPPM.put(15); outPPM.put(15); outPPM.put(25); // Dark background for unknown
            } else {
                float norm = std::clamp(d / static_cast<float>(maxDisparity), 0.0f, 1.0f);
                uint8_t r = static_cast<uint8_t>(255 * std::clamp(1.5f - std::abs(norm * 4.0f - 3.0f), 0.0f, 1.0f));
                uint8_t g = static_cast<uint8_t>(255 * std::clamp(1.5f - std::abs(norm * 4.0f - 2.0f), 0.0f, 1.0f));
                uint8_t b = static_cast<uint8_t>(255 * std::clamp(1.5f - std::abs(norm * 4.0f - 1.0f), 0.0f, 1.0f));
                outPPM.put(r); outPPM.put(g); outPPM.put(b);
            }
        }
    }
    outPPM.close();
    std::cout << "Saved /home/steamos/test_openvr/dense_depth.ppm!" << std::endl;
    return 0;
}
