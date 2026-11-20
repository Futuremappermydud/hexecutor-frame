// fhm_nn_check: load Mercury's .pte models the way the N-view tracker does (one detection and two keypoint
// models per view, plus the two-hand keypoint model when it was exported, all in one process) and compare every copy's outputs with the ONNX Runtime outputs that
// models/export_executorch.py wrote to check/.
//
//   fhm_nn_check MODELS_DIR [--views N] [--iters N] [--tolerance X]
//
// The backend follows the HEXECUTOR_* environment (the Frame DSP), or HEXECUTOR_ET_REFERENCE=1 (portable CPU
// reference). Exit status 0 only when every output of every copy is within the tolerance.
#include "fhm_nn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string name;
    std::vector<std::vector<float>> inputs, expected;
};

bool read_f32(const std::string &path, std::vector<float> *out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return false;
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || bytes % 4)
        return false;
    out->resize(size_t(bytes) / 4);
    f.seekg(0);
    return bool(f.read(reinterpret_cast<char *>(out->data()), bytes));
}

bool load_case(const std::string &dir, const std::string &name, Case *c) {
    c->name = name;
    for (int i = 0;; i++) {
        std::vector<float> v;
        if (!read_f32(dir + "/check/" + name + ".in" + std::to_string(i) + ".f32", &v))
            break;
        c->inputs.push_back(std::move(v));
    }
    for (int j = 0;; j++) {
        std::vector<float> v;
        if (!read_f32(dir + "/check/" + name + ".out" + std::to_string(j) + ".f32", &v))
            break;
        c->expected.push_back(std::move(v));
    }
    return !c->inputs.empty() && !c->expected.empty();
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fhm_nn_check MODELS_DIR [--views N] [--iters N] [--tolerance X]\n");
        return 2;
    }
    const std::string dir = argv[1];
    int views = 4, iters = 20;
    double tolerance = 0.05;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--views") && i + 1 < argc)
            views = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
            iters = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--tolerance") && i + 1 < argc)
            tolerance = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "fhm_nn_check: unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    // The tracker's order: per view one detection model, two keypoint models, then the optional pair.
    const char *names[] = {"grayscale_detection_160x160", "grayscale_keypoint_jan18",
                           "grayscale_keypoint_jan18_x2"};
    Case cases[3];
    for (int k = 0; k < 2; k++)
        if (!load_case(dir, names[k], &cases[k])) {
            std::fprintf(stderr, "fhm_nn_check: missing %s/check/%s.* (models/export_executorch.py)\n",
                         dir.c_str(), names[k]);
            return 2;
        }
    const int kinds = load_case(dir, names[2], &cases[2]) ? 3 : 2;
    std::vector<int> order = {0, 1, 1};
    if (kinds == 3)
        order.push_back(2);
    struct Copy {
        int which;
        fhm_nn::Model *model;
    };
    std::vector<Copy> copies;
    const auto load_start = std::chrono::steady_clock::now();
    for (int v = 0; v < views; v++)
        for (int which : order) {
            std::string error;
            fhm_nn::Model *m = fhm_nn::load(dir + "/" + names[which] + ".pte", &error);
            if (!m) {
                std::fprintf(stderr, "fhm_nn_check: %s\n", error.c_str());
                return 1;
            }
            copies.push_back({which, m});
        }
    std::printf("loaded %zu models in %.1f ms\n", copies.size(), ms_since(load_start));

    bool ok = true;
    for (int k = 0; k < kinds; k++) {
        std::vector<double> warm;
        double worst = 0;
        for (const Copy &c : copies) {
            if (c.which != k)
                continue;
            const Case &cs = cases[k];
            if (fhm_nn::num_inputs(c.model) != cs.inputs.size() ||
                fhm_nn::num_outputs(c.model) != cs.expected.size()) {
                std::fprintf(stderr, "fhm_nn_check: %s I/O count differs from check/\n", names[k]);
                return 1;
            }
            std::vector<const float *> in;
            for (size_t i = 0; i < cs.inputs.size(); i++) {
                if (fhm_nn::input_elements(c.model, i) != cs.inputs[i].size()) {
                    std::fprintf(stderr, "fhm_nn_check: %s input %zu size differs\n", names[k], i);
                    return 1;
                }
                in.push_back(cs.inputs[i].data());
            }
            std::vector<const float *> out(cs.expected.size());
            for (int it = 0; it < iters; it++) {
                const auto t = std::chrono::steady_clock::now();
                std::string error;
                if (!fhm_nn::run(c.model, in.data(), out.data(), &error)) {
                    std::fprintf(stderr, "fhm_nn_check: %s: %s\n", names[k], error.c_str());
                    return 1;
                }
                if (it)
                    warm.push_back(ms_since(t));
            }
            for (size_t j = 0; j < cs.expected.size(); j++) {
                if (fhm_nn::output_elements(c.model, j) != cs.expected[j].size()) {
                    std::fprintf(stderr, "fhm_nn_check: %s output %zu size differs\n", names[k], j);
                    return 1;
                }
                for (size_t e = 0; e < cs.expected[j].size(); e++) {
                    const double d = std::fabs(double(out[j][e]) - double(cs.expected[j][e]));
                    worst = std::isnan(d) ? INFINITY : std::max(worst, d);
                }
            }
        }
        std::sort(warm.begin(), warm.end());
        const bool pass = worst <= tolerance;
        ok = ok && pass;
        std::printf("%s: max abs diff vs ONNX Runtime %.3g (tolerance %.3g) %s", names[k], worst, tolerance,
                    pass ? "PASS" : "FAIL");
        if (!warm.empty())
            std::printf("; warm run median %.3f ms, p90 %.3f ms (n=%zu)", warm[warm.size() / 2],
                        warm[std::min(warm.size() - 1, warm.size() * 9 / 10)], warm.size());
        std::printf("\n");
    }
    for (const Copy &c : copies)
        fhm_nn::destroy(c.model);
    return ok ? 0 : 1;
}
