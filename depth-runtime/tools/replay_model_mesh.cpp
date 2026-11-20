// Offline replay of the CPU path from model output to the published TSDF mesh:
// spherical metric conversion + confidence, speckle rejection, TSDF integration,
// periodic surface extraction and vertex publication. It mirrors the mesh-mode-2
// branch of src/cloud_worker.cpp and opens no camera, NPU session or renderer.
// Input stream (little endian), written by scripts/model_mesh_replay.py:
//   u32 magic 'MMR1', u32 frames, f32 spherical step (rad), u32 max uncertainty
//   per frame: f32[12] camera-to-world (R row-major, t), f32[3] head position,
//              f32[320*256] disparity, u8[320*256] uncertainty, u8[616*616] rectified left luma
// Hashes cover every frame's depth/confidence and every published vertex buffer,
// so two builds with equal hashes produced the same outputs throughout the replay.
// mesh_set_hash ignores triangle order within a published mesh: equal mesh_set_hash
// with a different mesh_hash means the same triangles in a different order.
#include "calibration.hpp"
#include "depth_speckle_filter.hpp"
#include "model_depth_geometry.hpp"
#include "tsdf_volume.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace FrameSpatial;
using Clock = std::chrono::steady_clock;

namespace {
// Same layout and colour packing as FrameSpatialCloud::PointVertex / packGray.
struct PublishedVertex { float x, y, z; uint32_t rgba; float supportLifetime = 0; };
uint32_t packGray(float luminance, uint32_t observations) {
    const float confidence = std::min(1.0f, observations / 8.0f);
    const int gray = std::clamp(static_cast<int>(luminance), 35, 255);
    const int red = std::clamp(static_cast<int>(gray * (0.55f + 0.30f * confidence)), 0, 255);
    const int green = std::clamp(static_cast<int>(gray * (0.80f + 0.20f * confidence)), 0, 255);
    const int blue = std::clamp(static_cast<int>(gray * (1.00f + 0.15f * confidence)), 0, 255);
    return uint32_t(red) | (uint32_t(green) << 8) | (uint32_t(blue) << 16) | 0xff000000u;
}
double since(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
void hashBytes(uint64_t& hash, const void* data, size_t bytes) {
    // FNV-1a over 8-byte words (tail bytewise): order-sensitive and fast enough for every frame.
    const auto* p = static_cast<const unsigned char*>(data);
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) { uint64_t w; std::memcpy(&w, p + i, 8); hash = (hash ^ w) * 1099511628211ull; }
    for (; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
}
struct Stage {
    const char* name; std::vector<double> ms;
    void report() {
        if (ms.empty()) return;
        double sum = 0; for (double v : ms) sum += v;
        std::sort(ms.begin(), ms.end());
        std::printf("%-12s n=%zu mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f max_ms=%.3f total_s=%.3f\n", name, ms.size(),
                    sum / ms.size(), ms[ms.size() / 2], ms[(ms.size() - 1) * 95 / 100], ms.back(), sum / 1000);
    }
};
int usage() {
    std::cerr << "usage: replay_model_mesh [--mesh-every N] [--full | --incremental] [--max-frames N] [--ply mesh.ply] stream|-\n";
    return 2;
}
}  // namespace

int main(int argc, char** argv) {
    int meshEvery = 5, maxFrames = 0;
    bool incremental = false, full = false;
    std::string plyPath, streamPath;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--mesh-every" && i + 1 < argc) meshEvery = std::atoi(argv[++i]);
        else if (arg == "--max-frames" && i + 1 < argc) maxFrames = std::atoi(argv[++i]);
        else if (arg == "--incremental") incremental = true;
        else if (arg == "--full") full = true;
        else if (arg == "--ply" && i + 1 < argc) plyPath = argv[++i];
        else if (arg.rfind("--", 0) == 0) return usage();
        else streamPath = arg;
    }
    if (streamPath.empty() || meshEvery < 1 || (incremental && full)) return usage();
    std::ifstream file;
    std::istream* input = &std::cin;
    if (streamPath != "-") { file.open(streamPath, std::ios::binary); input = &file; }
    uint32_t header[2]{}, maxUncertainty = 0;
    float step = 0;
    if (!input->read(reinterpret_cast<char*>(header), sizeof(header)) || header[0] != 0x31524d4d ||
        !input->read(reinterpret_cast<char*>(&step), sizeof(step)) ||
        !input->read(reinterpret_cast<char*>(&maxUncertainty), sizeof(maxUncertainty)) ||
        header[1] == 0 || !(step > 0) || maxUncertainty < 51 || maxUncertainty > 256) {
        std::cerr << "invalid replay stream header\n";
        return 2;
    }
    const uint32_t count = maxFrames > 0 ? std::min<uint32_t>(header[1], maxFrames) : header[1];
    const size_t modelPixels = size_t(MODEL_WIDTH) * MODEL_HEIGHT, rectPixels = size_t(RECT_WIDTH) * RECT_HEIGHT;
    constexpr size_t kTsdfMaxTriangles = 480000;

    std::vector<float> disparity(modelPixels), depth, confidence;
    std::vector<uint8_t> uncertainty(modelPixels), luma(rectPixels);
    ModelSphericalMap sphericalMap;
    ModelSphericalWorkspace sphericalWorkspace;
    DepthSpeckleWorkspace speckleWorkspace;
    TsdfVolume tsdf;
    std::vector<TsdfVertex> triangles;
    std::vector<PublishedVertex> meshVertices, published;
    Stage post{"postprocess", {}}, speckle{"speckles", {}}, integrate{"integrate", {}}, extract{"extract", {}},
        publish{"publish", {}}, frameTotal{"frame_total", {}}, meshFrame{"mesh_frame", {}};
    // packGray depends on the luminance only through its integer part.
    std::array<uint32_t, 256> grayLut;
    for (size_t i = 0; i < grayLut.size(); ++i) grayLut[i] = packGray(float(i), 8);
    uint64_t depthHash = 1469598103934665603ull, meshHash = depthHash, meshSetHash = 0;
    size_t validDepth = 0, peakTriangles = 0, meshes = 0;

    for (uint32_t frame = 1; frame <= count; frame++) {
        std::array<float, 12> transform{};
        float focus[3]{};
        if (!input->read(reinterpret_cast<char*>(transform.data()), sizeof(transform)) ||
            !input->read(reinterpret_cast<char*>(focus), sizeof(focus)) ||
            !input->read(reinterpret_cast<char*>(disparity.data()), modelPixels * sizeof(float)) ||
            !input->read(reinterpret_cast<char*>(uncertainty.data()), modelPixels) ||
            !input->read(reinterpret_cast<char*>(luma.data()), rectPixels)) {
            std::cerr << "truncated replay stream at frame " << frame << "\n";
            return 2;
        }
        TsdfPose pose;
        for (int row = 0; row < 3; row++) for (int col = 0; col < 3; col++) pose.rotation[row][col] = transform[row * 3 + col];
        for (int row = 0; row < 3; row++) pose.translation[row] = transform[9 + row];

        // --- postprocess: ModelDepthPipeline::compute after the worker reply (spherical branch).
        auto begin = Clock::now();
        const auto frameBegin = begin;
        if (sphericalMap.step != step)
            sphericalMap = modelSphericalMap(step, FOCAL_BASELINE / RECT_FOCAL, RECT_WIDTH, RECT_HEIGHT,
                                             RECT_FOCAL, RECT_CX, RECT_CY);
        modelSphericalPostprocess(disparity, uncertainty, sphericalMap, int(maxUncertainty), sphericalWorkspace, depth,
                                  confidence);
        if (frame > 1) post.ms.push_back(since(begin));  // frame 1 builds the spherical map

        // --- speckles
        begin = Clock::now();
        removeDepthSpeckles(depth, RECT_WIDTH, RECT_HEIGHT, speckleWorkspace);
        speckle.ms.push_back(since(begin));
        hashBytes(depthHash, depth.data(), depth.size() * sizeof(float));
        hashBytes(depthHash, confidence.data(), confidence.size() * sizeof(float));
        for (float z : depth) validDepth += z > 0;

        // --- TSDF integration
        begin = Clock::now();
        tsdf.integrate(depth.data(), luma.data(), RECT_WIDTH, RECT_HEIGHT, RECT_FOCAL, RECT_CX, RECT_CY, pose,
                       confidence.data());
        integrate.ms.push_back(since(begin));

        // --- extraction and publication, on the live schedule
        const bool meshNow = frame == 1 || frame % meshEvery == 0;
        if (meshNow) {
            begin = Clock::now();
            if (full || incremental) {
                if (incremental) tsdf.extractMeshIncremental(triangles, kTsdfMaxTriangles, focus);
                else tsdf.extractMesh(triangles, kTsdfMaxTriangles, focus);
                extract.ms.push_back(since(begin));
                begin = Clock::now();
                meshVertices.clear();
                meshVertices.reserve(triangles.size());
                for (const auto& v : triangles) meshVertices.push_back({v.x, v.y, v.z, packGray(v.luma, 8)});
            } else {
                // Live path: extraction writes the published vertices directly (the vertex
                // count only grows, so the buffer is not re-zeroed), so its time is all "extract".
                tsdf.extractMeshAs(meshVertices, kTsdfMaxTriangles, focus, [&](const TsdfVertex& v) {
                    return PublishedVertex{v.x, v.y, v.z, grayLut[std::min(int(v.luma), 255)]};
                });
                extract.ms.push_back(since(begin));
                begin = Clock::now();
            }
            published.swap(meshVertices);
            publish.ms.push_back(since(begin));
            hashBytes(meshHash, published.data(), published.size() * sizeof(PublishedVertex));
            uint64_t triangleSum = 0;
            for (size_t t = 0; t + 3 <= published.size(); t += 3) {
                uint64_t one = 1469598103934665603ull;
                hashBytes(one, &published[t], 3 * sizeof(PublishedVertex));
                triangleSum += one * 0x9e3779b97f4a7c15ull + (one >> 29);
            }
            hashBytes(meshSetHash, &triangleSum, sizeof(triangleSum));
            peakTriangles = std::max(peakTriangles, published.size() / 3);
            ++meshes;
        }
        const double total = since(frameBegin);
        if (frame > 1) (meshNow ? meshFrame : frameTotal).ms.push_back(total);
    }
    post.report(); speckle.report(); integrate.report(); extract.report(); publish.report();
    frameTotal.name = "frame_nomesh"; frameTotal.report(); meshFrame.report();
    std::printf("frames=%u meshes=%zu leaves=%zu final_triangles=%zu peak_triangles=%zu valid_depth_mean=%.1f "
                "mesh_every=%d mode=%s\n", count, meshes, tsdf.leafCount(), published.size() / 3, peakTriangles,
                double(validDepth) / count, meshEvery, full ? "full" : incremental ? "incremental" : "live");
    std::printf("depth_hash=%016llx mesh_hash=%016llx mesh_set_hash=%016llx\n", (unsigned long long)depthHash,
                (unsigned long long)meshHash, (unsigned long long)meshSetHash);
    if (!plyPath.empty()) {
        std::ofstream out(plyPath, std::ios::binary);
        out << "ply\nformat binary_little_endian 1.0\nelement vertex " << published.size()
            << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
               "property uchar blue\nelement face " << published.size() / 3
            << "\nproperty list uchar int vertex_indices\nend_header\n";
        for (const auto& v : published) {
            out.write(reinterpret_cast<const char*>(&v.x), 12);
            out.write(reinterpret_cast<const char*>(&v.rgba), 3);
        }
        for (uint32_t i = 0; i < published.size(); i += 3) {
            const unsigned char n = 3; const int32_t idx[3]{int32_t(i), int32_t(i + 1), int32_t(i + 2)};
            out.write(reinterpret_cast<const char*>(&n), 1);
            out.write(reinterpret_cast<const char*>(idx), 12);
        }
        if (!out) return 1;
    }
    return 0;
}
