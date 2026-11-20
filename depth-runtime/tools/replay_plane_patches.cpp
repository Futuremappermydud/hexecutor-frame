// E09 replay: stream recorded model outputs and poses through PlanePatchMap.
// stdin: u32 count, then per frame: u32 valid, 12 f32 camera-to-world (row-major R, t),
//        256x320 f32 left disparity, 256x320 u8 left uncertainty.
// OUT/labels.u16: per frame, the map patch id of every model pixel (0 none).
// OUT/frames.jsonl: per-frame statistics. OUT/patches.json: the final map.
#include "plane_patches.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace FrameSpatial;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: replay_plane_patches OUT_DIR < stream\n";
        return 2;
    }
    const std::string out = argv[1];
    uint32_t count = 0;
    if (!std::cin.read(reinterpret_cast<char*>(&count), 4) || count == 0 || count > 20000) return 2;
    PlanePatchMap map;
    const auto& cfg = map.config();
    const size_t pixels = size_t(cfg.width) * cfg.height;
    std::vector<float> disparity(pixels);
    std::vector<uint8_t> uncertainty(pixels);
    std::vector<uint16_t> labels(pixels);
    std::ofstream labelFile(out + "/labels.u16", std::ios::binary), frames(out + "/frames.jsonl");
    std::vector<double> ms;
    for (uint32_t f = 0; f < count; ++f) {
        uint32_t valid = 0;
        std::array<float, 12> m{};
        if (!std::cin.read(reinterpret_cast<char*>(&valid), 4) || !std::cin.read(reinterpret_cast<char*>(m.data()), 48) ||
            !std::cin.read(reinterpret_cast<char*>(disparity.data()), pixels * 4) ||
            !std::cin.read(reinterpret_cast<char*>(uncertainty.data()), pixels))
            return 2;
        PlaneFrameStats stats;
        if (valid) {
            PlanePose pose;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) pose.rotation[r][c] = m[r * 3 + c];
            for (int r = 0; r < 3; ++r) pose.translation[r] = m[9 + r];
            const auto began = std::chrono::steady_clock::now();
            stats = map.integrate(disparity.data(), uncertainty.data(), pose, labels.data());
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
        } else {
            std::fill(labels.begin(), labels.end(), uint16_t(0));
        }
        labelFile.write(reinterpret_cast<const char*>(labels.data()), pixels * 2);
        frames << "{\"frame\":" << f + 1 << ",\"valid\":" << valid << ",\"planar_cells\":" << stats.planarCells
               << ",\"frame_patches\":" << stats.framePatches << ",\"associated\":" << stats.associated
               << ",\"created\":" << stats.created << ",\"merged\":" << stats.merged << ",\"contests\":" << stats.contests
               << ",\"confirmed\":" << stats.confirmed << ",\"tentative\":" << stats.tentative
               << ",\"contested\":" << stats.contested << ",\"retired\":" << stats.retired
               << ",\"regions_small\":" << stats.regionsSmall << ",\"regions_curved\":" << stats.regionsCurved
               << ",\"regions_noisy\":" << stats.regionsNoisy << ",\"refused_fused\":" << stats.refusedFused
               << ",\"ms\":" << (valid ? ms.back() : 0) << "}\n";
    }
    std::ofstream patches(out + "/patches.json");
    patches << "{\"patches\":[\n";
    bool first = true;
    for (const MapPatch& p : map.patches()) {
        patches << (first ? "" : ",\n") << "{\"id\":" << p.id << ",\"state\":" << int(p.state) << ",\"merged_into\":" << p.mergedInto
                << ",\"normal\":[" << p.normal[0] << "," << p.normal[1] << "," << p.normal[2] << "],\"offset\":" << p.offset
                << ",\"centroid\":[" << p.centroid[0] << "," << p.centroid[1] << "," << p.centroid[2] << "],\"rms\":" << p.rms
                << ",\"offset_sigma\":" << p.offsetSigma << ",\"views\":" << p.views.size() << ",\"support_frames\":"
                << p.supportFrames << ",\"first_frame\":" << p.firstFrame << ",\"last_support\":" << p.lastSupport
                << ",\"footprint\":[";
        bool firstKey = true;
        for (int64_t key : p.footprint) {
            patches << (firstKey ? "" : ",") << key;
            firstKey = false;
        }
        patches << "]}";
        first = false;
    }
    patches << "\n]}\n";
    std::sort(ms.begin(), ms.end());
    std::cout << "frames=" << count << " patches=" << map.patches().size()
              << " ms_p50=" << (ms.empty() ? 0 : ms[ms.size() / 2]) << " ms_p95=" << (ms.empty() ? 0 : ms[size_t(ms.size() * 0.95)])
              << '\n';
    return labelFile && frames && patches ? 0 : 1;
}
