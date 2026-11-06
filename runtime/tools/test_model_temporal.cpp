// Checks src/model_temporal.cpp. With a directory from tools/model_temporal_golden.py it
// also compares against the Python reference on real model output and reports warp time.
#include "model_temporal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using FrameSpatial::ModelTemporalState;
using History = ModelTemporalState::History;

namespace {
int failures = 0;
void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}
template <class Value>
bool load(const std::string& path, std::vector<Value>& out) {
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size() * sizeof(Value)));
    return bool(file);
}
void policy() {
    const double identity[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const double yaw40[3][3]{{0.76604444, 0, 0.64278761}, {0, 1, 0}, {-0.64278761, 0, 0.76604444}};
    const int64_t limit = 250000000;
    ModelTemporalState state(0.00506145);
    std::vector<float> plane(ModelTemporalState::Width * ModelTemporalState::Height, 10.f);
    check(state.admit(true, false, identity, 30000000, limit, 30) == History::None, "no history before save");
    state.save(plane.data(), plane.data());
    check(state.admit(true, false, identity, 30000000, limit, 30) == History::Usable, "normal step is usable");
    check(state.admit(true, true, identity, 30000000, limit, 30) == History::Reset, "reset wins");
    check(state.admit(false, false, identity, 30000000, limit, 30) == History::Unaligned, "lost pose drops history");
    check(state.admit(true, false, identity, 300000000, limit, 30) == History::TimeGap, "long gap drops history");
    check(state.admit(true, false, identity, 0, limit, 30) == History::TimeGap, "missing time drops history");
    check(state.admit(true, false, yaw40, 30000000, limit, 30) == History::Rotation, "fast rotation drops history");
    check(state.admit(true, false, yaw40, 300000000, 0, 0) == History::Usable, "disabled limits");
    std::vector<float> hidden(ModelTemporalState::HiddenValues, 1.f), t(ModelTemporalState::TransformValues, 1.f);
    const double none[3]{};
    state.reset();
    state.inputs(identity, none, hidden.data(), t.data());
    check(std::all_of(hidden.begin(), hidden.end(), [](float v) { return v == 0; }), "no history sends zero hidden");
    check(std::all_of(t.begin(), t.end(), [](float v) { return v == 0; }), "no history sends zero t");
    // A still camera must return the same disparity through both eyes' warps.
    state.save(plane.data(), plane.data());
    state.inputs(identity, none, hidden.data(), t.data());
    check(std::fabs(hidden[size_t(160) * 256 + 128] - 10.f) < 0.01f, "still camera keeps left disparity");
    check(std::fabs(hidden[size_t(320 + 160) * 256 + 128] - 10.f) < 0.01f, "still camera keeps right disparity");
    check(t[0] == 0 && t[3 * 4 + 0] == 0, "still camera has zero t");
}
}  // namespace

int main(int argc, char** argv) {
    policy();
    if (argc > 1) {
        const std::string root = argv[1];
        int cases = 0;
        std::ifstream(root + "/cases.txt") >> cases;
        check(cases > 0, "golden cases found");
        std::vector<float> previous(ModelTemporalState::HiddenValues), expected(ModelTemporalState::HiddenValues),
            hidden(ModelTemporalState::HiddenValues), expectedT(ModelTemporalState::TransformValues),
            t(ModelTemporalState::TransformValues);
        std::vector<double> motion(12);
        for (const char* name : {"spherical", "pinhole"}) {
            const bool spherical = std::string(name) == "spherical";
            ModelTemporalState state(spherical ? 0.29 * 3.14159265358979323846 / 180.0 : 0.0);
            double worst = 0, worstT = 0, total = 0;
            size_t over = 0, compared = 0;
            std::vector<double> times;
            for (int c = 0; c < cases; ++c) {
                const std::string stem = root + "/case-" + std::to_string(c) + "-" + name;
                if (!load(stem + "-previous.f32", previous) || !load(stem + "-motion.f64", motion) ||
                    !load(stem + "-hidden.f32", expected) || !load(stem + "-t.f32", expectedT)) {
                    check(false, "golden case readable");
                    continue;
                }
                const double R[3][3]{{motion[0], motion[1], motion[2]}, {motion[3], motion[4], motion[5]}, {motion[6], motion[7], motion[8]}};
                const double translation[3]{motion[9], motion[10], motion[11]};
                const auto begin = std::chrono::steady_clock::now();
                state.save(previous.data(), previous.data() + ModelTemporalState::Width * ModelTemporalState::Height);
                state.inputs(R, translation, hidden.data(), t.data());
                times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
                for (size_t i = 0; i < hidden.size(); ++i) {
                    const double difference = std::fabs(double(hidden[i]) - expected[i]);
                    worst = std::max(worst, difference);
                    total += difference;
                    over += difference > 1e-3;
                }
                compared += hidden.size();
                for (size_t i = 0; i < t.size(); ++i)
                    worstT = std::max(worstT, std::fabs(double(t[i]) - expectedT[i]));
            }
            std::sort(times.begin(), times.end());
            std::printf("%s: %d cases, hidden max diff %.6f, mean %.8f, over 0.001: %zu of %zu, t max diff %.8f; "
                        "save+warp ms p50 %.2f p95 %.2f max %.2f\n",
                        name, cases, worst, total / std::max<size_t>(compared, 1), over, compared, worstT,
                        times[times.size() / 2], times[times.size() * 95 / 100], times.back());
            check(worst < 0.02, "hidden matches the Python reference");
            check(double(over) / std::max<size_t>(compared, 1) < 1e-4, "hidden differs on almost no pixels");
            check(worstT < 1e-5, "t matches the Python reference");
        }
    }
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
