#undef NDEBUG
#include "nirhands/pipeline.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Buffer = nirhands::Models::Buffer;
using Bytes = nirhands::Models::Bytes;

// First 18 values of the skeleton manifest mean profile: six palm points, x y z each.
constexpr float kPalmProfile[18] = {
    0.0f, 0.0f, 0.0f,
    0.023721564561128616f, -0.02210545726120472f, -0.018676256760954857f,
    0.08587229996919632f, -0.022199932485818863f, -0.00661707716062665f,
    0.09277191013097763f, 0.0f, 0.0f,
    0.08069457858800888f, 0.022870788350701332f, -0.0002967443724628538f,
    0.06928838044404984f, 0.0398334264755249f, -0.00533724669367075f,
};

Bytes to_bytes(const std::vector<float>& values) {
    Bytes out(values.size() * sizeof(float));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

nirhands::Geometry test_geometry(int size = 128) {
    nirhands::Geometry g;
    g.camera.size = size;
    g.camera.f = 300.0;
    g.camera.offset = {-double(size) / 2.0, -double(size) / 2.0};
    g.camera.center = {0, 0, 0};
    g.baseline_m = 0.064;
    g.center_depth_m = 0.4;
    g.mirrored = false;
    g.stereo = true;
    g.primary_view = 0;
    return g;
}

// Six skeleton outputs: kp, disp, vis, then scale (1), palm (18) and angles (30) in whatever order
// QNN compiled them; the host picks those three by size. order 0: scale, palm, angles; 1: scale,
// angles, palm; 2: palm, angles, scale; 3: invalid (two scales, no angles).
std::vector<Bytes> skeleton_outputs(int order) {
    std::vector<float> kp(42, 0.0f), disp(21, 0.0f), vis(21), scale{1.0f}, angles(30, 0.0f);
    std::vector<float> palm(kPalmProfile, kPalmProfile + 18);
    for (int j = 0; j < 21; ++j) vis[j] = float(j);
    if (order == 0)
        return {to_bytes(kp), to_bytes(disp), to_bytes(vis), to_bytes(scale), to_bytes(palm), to_bytes(angles)};
    if (order == 1)
        return {to_bytes(kp), to_bytes(disp), to_bytes(vis), to_bytes(scale), to_bytes(angles), to_bytes(palm)};
    if (order == 2)
        return {to_bytes(kp), to_bytes(disp), to_bytes(vis), to_bytes(palm), to_bytes(angles), to_bytes(scale)};
    return {to_bytes(kp), to_bytes(disp), to_bytes(vis), to_bytes(scale), to_bytes(palm), to_bytes(scale)};
}

void test_skeleton(const std::string& dir) {
    int kp_size = 128;
    {
        nirhands::Models probe(dir, [](size_t, const std::vector<Buffer>&, const nirhands::Geometry*) {
            return skeleton_outputs(0);
        });
        kp_size = probe.keypoint_size();
    }
    const nirhands::Geometry geometry = test_geometry(kp_size);
    const size_t K = size_t(kp_size);
    const std::vector<float> crops(2u * K * K, 0.0f);
    float kp[42]{}, disp[21]{}, vis[21]{};

    for (int order = 0; order < 3; ++order) {
        std::fill(kp, kp + 42, 0.0f);
        nirhands::Models models(dir, [order, K](size_t program, const std::vector<Buffer>& inputs, const nirhands::Geometry* geo) {
            assert(program == 1);
            assert(geo != nullptr);
            assert(inputs.size() == 2);
            assert(inputs[0].bytes == 2u * K * K * sizeof(float));
            assert(inputs[1].bytes == sizeof(float));
            assert(*static_cast<const float*>(inputs[1].data) == 1.0f);
            return skeleton_outputs(order);
        });
        models.keypoints(0, crops.data(), true, geometry, kp, disp, vis);
        for (int j = 0; j < 21; ++j) assert(vis[j] == float(j));
        for (float v : kp) assert(std::isfinite(v));
        for (float v : disp) assert(std::isfinite(v));
        // Aux keypoint (0,0) and zero disparity put the root on the optical axis at center_depth_m.
        assert(std::fabs(kp[18]) < 1e-4f && std::fabs(kp[19]) < 1e-4f && std::fabs(disp[9]) < 1e-4f);
        // The wrist must not sit at the root: proves palm/scale were decoded, not swapped.
        assert(std::fabs(kp[0]) > 0.05f);
    }

    {
        nirhands::Models models(dir, [](size_t program, const std::vector<Buffer>&, const nirhands::Geometry* geo) {
            assert(program == 1);
            assert(geo != nullptr);
            return skeleton_outputs(3);
        });
        bool threw = false;
        try {
            models.keypoints(0, crops.data(), true, geometry, kp, disp, vis);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 2 || argc == 3);
    const nirhands::Geometry geometry = test_geometry();

    nirhands::Models models(argv[1], [](size_t program, const std::vector<Buffer>& inputs, const nirhands::Geometry* geo) {
        if (program == 0) {
            assert(geo == nullptr);
            assert(inputs.size() == 1);
            assert(inputs[0].bytes == 2u * 224 * 224 * sizeof(float));
            std::vector<float> nhwc(2u * 56 * 56 * 10);
            for (int b = 0; b < 2; ++b) for (int y = 0; y < 56; ++y) for (int x = 0; x < 56; ++x) for (int c = 0; c < 10; ++c)
                nhwc[((b * 56 + y) * 56 + x) * 10 + c] = float(((b * 10 + c) * 56 + y) * 56 + x);
            return std::vector<Bytes>{to_bytes(nhwc)};
        }
        assert(program == 1 && geo != nullptr);
        assert(inputs.size() == 2 && inputs[0].bytes == 2u * 128 * 128 * sizeof(float));
        const auto* crop = static_cast<const float*>(inputs[0].data);
        assert(crop[0] == 0 && crop[1] == 100 && crop[2] == 1 && crop[3] == 101);
        assert(*static_cast<const float*>(inputs[1].data) == 1.0f);
        std::vector<float> kp(42), disp(21), vis(21);
        for (int j = 0; j < 21; ++j) {
            kp[2 * j] = float(j);
            disp[j] = float(j);
            vis[j] = float(j);
        }
        return std::vector<Bytes>{to_bytes(kp), to_bytes(disp), to_bytes(vis)};
    });
    std::vector<float> detector(2u * 224 * 224), maps;
    models.detect(detector.data(), maps);
    assert(maps.size() == 2u * 10 * 56 * 56);
    for (size_t i = 0; i < maps.size(); ++i) assert(maps[i] == float(i));

    std::vector<float> crops(2u * 128 * 128);
    for (int i = 0; i < 128 * 128; ++i) {
        crops[i] = float(i);
        crops[128 * 128 + i] = float(100 + i);
    }
    float kp[42]{}, disp[21]{}, vis[21]{};
    models.keypoints(0, crops.data(), true, geometry, kp, disp, vis);
    for (int j = 0; j < 21; ++j) assert(kp[2 * j] == j && disp[j] == j && vis[j] == j);

    if (argc == 3) test_skeleton(argv[2]);
    return 0;
}
