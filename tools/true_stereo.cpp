#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <map>
#include <tuple>

struct ImageRGB {
    int w, h;
    std::vector<uint8_t> r, g, b;
    std::vector<uint8_t> grey;
};

ImageRGB loadStereoView(const std::string& path, int tW, int tH, int dyShift = 0) {
    std::ifstream f(path, std::ios::binary);
    std::string header;
    int srcW, srcH, maxVal;
    f >> header >> srcW >> srcH >> maxVal;
    f.get();
    std::vector<uint8_t> raw(srcW * srcH * 3);
    f.read(reinterpret_cast<char*>(raw.data()), raw.size());

    ImageRGB img;
    img.w = tW;
    img.h = tH;
    img.r.resize(tW * tH, 0);
    img.g.resize(tW * tH, 0);
    img.b.resize(tW * tH, 0);
    img.grey.resize(tW * tH, 0);

    float sx = static_cast<float>(srcW) / tW;
    float sy = static_cast<float>(srcH) / tH;

    for (int y = 0; y < tH; y++) {
        int srcY = y - dyShift;
        if (srcY < 0 || srcY >= tH) continue;
        int sy_idx = std::min(static_cast<int>(srcY * sy), srcH - 1);
        for (int x = 0; x < tW; x++) {
            int sx_idx = std::min(static_cast<int>(x * sx), srcW - 1);
            int idx = (sy_idx * srcW + sx_idx) * 3;
            uint8_t cr = raw[idx];
            uint8_t cg = raw[idx+1];
            uint8_t cb = raw[idx+2];
            int outIdx = y * tW + x;
            img.r[outIdx] = cr;
            img.g[outIdx] = cg;
            img.b[outIdx] = cb;
            img.grey[outIdx] = static_cast<uint8_t>(0.299f * cr + 0.587f * cg + 0.114f * cb);
        }
    }
    return img;
}

struct Point3D {
    float x, y, z;
    uint8_t r, g, b;
};

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    int W = 320, H = 360;
    // SteamVR side-by-side: right.ppm is true Left Eye, left.ppm is true Right Eye
    ImageRGB left = loadStereoView("/home/steamos/test_openvr/right.ppm", W, H, 0);
    ImageRGB right = loadStereoView("/home/steamos/test_openvr/left.ppm", W, H, 6);

    const int minD = 10;
    const int maxD = 200;
    const int winR = 2; // 5x5 window
    std::vector<float> disparity(W * H, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int y = winR; y < H - winR; y++) {
        for (int x = maxD + winR; x < W - winR; x++) {
            int bestCost = 1e9;
            int bestD = minD;
            int secondCost = 1e9;

            for (int d = minD; d < maxD; d++) {
                int cost = 0;
                for (int dy = -winR; dy <= winR; dy++) {
                    for (int dx = -winR; dx <= winR; dx++) {
                        int idxL = (y + dy) * W + (x + dx);
                        int idxR = (y + dy) * W + (x + dx - d);
                        cost += std::abs(left.grey[idxL] - right.grey[idxR]);
                    }
                }
                if (cost < bestCost) {
                    secondCost = bestCost;
                    bestCost = cost;
                    bestD = d;
                } else if (cost < secondCost) {
                    secondCost = cost;
                }
            }

            if (secondCost > 0 && (float)bestCost / secondCost < 0.94f) {
                disparity[y * W + x] = static_cast<float>(bestD);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Stereo Disparity computed in " << ms << " ms (" << (1000.0 / ms) << " FPS)!" << std::endl;

    // Back-project into 3D metric point cloud (Camera Space in meters)
    const float baseline = 0.063f; // 63mm IPD
    const float focal = 280.0f;    // Estimated focal length for 320px
    const float cx = W / 2.0f;
    const float cy = H / 2.0f;

    std::vector<Point3D> densePoints;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float d = disparity[y * W + x];
            if (d >= (float)minD) {
                float z = (focal * baseline) / d; // metric depth in meters
                if (z > 0.2f && z < 6.0f) {
                    float px = (x - cx) * z / focal;
                    float py = (y - cy) * z / focal;
                    int idx = y * W + x;
                    densePoints.push_back({px, -py, z, left.r[idx], left.g[idx], left.b[idx]});
                }
            }
        }
    }
    std::cout << "Generated " << densePoints.size() << " dense 3D surface points!" << std::endl;

    // Voxel downsampling (2.5cm voxels) to smooth drywall and eliminate noise
    const float voxelSize = 0.025f;
    std::map<std::tuple<int,int,int>, Point3D> voxelGrid;
    for (const auto& p : densePoints) {
        auto key = std::make_tuple(
            static_cast<int>(std::floor(p.x / voxelSize)),
            static_cast<int>(std::floor(p.y / voxelSize)),
            static_cast<int>(std::floor(p.z / voxelSize))
        );
        voxelGrid[key] = p;
    }
    std::cout << "Filtered into " << voxelGrid.size() << " clean voxel points!" << std::endl;

    // Save as colorized PLY point cloud
    std::ofstream ply("/home/steamos/test_openvr/dense_room.ply");
    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << voxelGrid.size() << "\n";
    ply << "property float x\nproperty float y\nproperty float z\n";
    ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ply << "end_header\n";
    for (const auto& kv : voxelGrid) {
        const auto& p = kv.second;
        ply << p.x << " " << p.y << " " << p.z << " " 
            << (int)p.r << " " << (int)p.g << " " << (int)p.b << "\n";
    }
    ply.close();
    std::cout << "Saved /home/steamos/test_openvr/dense_room.ply!" << std::endl;

    // Save depth map PNG
    std::ofstream outPPM("/home/steamos/test_openvr/true_depth.ppm", std::ios::binary);
    outPPM << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float d = disparity[y * W + x];
            if (d < (float)minD) {
                outPPM.put(10); outPPM.put(15); outPPM.put(25);
            } else {
                float norm = std::clamp((d - minD) / static_cast<float>(maxD - minD), 0.0f, 1.0f);
                uint8_t r = static_cast<uint8_t>(255 * std::clamp(1.5f - std::abs(norm * 4.0f - 3.0f), 0.0f, 1.0f));
                uint8_t g = static_cast<uint8_t>(255 * std::clamp(1.5f - std::abs(norm * 4.0f - 2.0f), 0.0f, 1.0f));
                uint8_t b = static_cast<uint8_t>(255 * std::clamp(1.5f - std::abs(norm * 4.0f - 1.0f), 0.0f, 1.0f));
                outPPM.put(r); outPPM.put(g); outPPM.put(b);
            }
        }
    }
    outPPM.close();
    return 0;
}
