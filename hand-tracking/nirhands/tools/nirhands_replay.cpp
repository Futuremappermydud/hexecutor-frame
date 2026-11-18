// Run the native NIR hand pipeline over a recorded raw pair stream (FRNIR001, as stream_nir_pairs --raw
// writes it) and print one JSON object per pair. Used to compare against nirhands_pipeline.py and to
// time the pipeline on the Frame.
//
//   nirhands_replay --models DIR --xrservice /persist/xrservice.json --device-config /persist/device_config.json
//                   --input pairs.raw [--max-pairs N] [--threads 2] [--quiet]
//                   [--qnn16-dir DIR --loader ELF --engine ELF --bootstrap ELF --chunk-probe ELF [--check-cpu]]
// --qnn16-dir runs both networks on the CDSP through Hexecutor (built with NIRHANDS_QNN16). --check-cpu also
// runs ONNX Runtime on every NPU input and reports per-output maximum differences on stderr.
#include "nirhands/pipeline.hpp"
#ifdef NIRHANDS_QNN16
#include "hexecutor/executor.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Header {
    char magic[8];
    int64_t left_ns, right_ns;
    uint32_t left_seq, right_seq;
};
static_assert(sizeof(Header) == 32, "FRNIR header is 32 bytes");

void print_vec(std::ostream& o, const nirhands::Vec3& v) {
    o << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string models_dir, xrservice = "/persist/xrservice.json", device = "/persist/device_config.json", input;
        int max_pairs = 0, threads = 2;
        bool quiet = false, check_cpu = false;
        std::string qnn16_dir;
#ifdef NIRHANDS_QNN16
        hexecutor::Artifacts artifacts;
#endif
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
            if (a == "--models") models_dir = next();
            else if (a == "--xrservice") xrservice = next();
            else if (a == "--device-config") device = next();
            else if (a == "--input") input = next();
            else if (a == "--max-pairs") max_pairs = std::stoi(next());
            else if (a == "--threads") threads = std::stoi(next());
            else if (a == "--quiet") quiet = true;
            else if (a == "--qnn16-dir") qnn16_dir = next();
            else if (a == "--check-cpu") check_cpu = true;
#ifdef NIRHANDS_QNN16
            else if (a == "--loader") artifacts.loader = next();
            else if (a == "--engine") artifacts.engine = next();
            else if (a == "--bootstrap") artifacts.bootstrap = next();
            else if (a == "--chunk-probe") artifacts.chunk_probe = next();
#endif
            else throw std::runtime_error("unknown argument " + a);
        }
        if (models_dir.empty() || input.empty()) throw std::runtime_error("--models and --input are required");
        const nirhands::Rig rig = nirhands::load_rig(xrservice, device);
        std::shared_ptr<nirhands::Models> models;
        // Per-output worst |npu - cpu|: detector maps, keypoints, disparity, visibility logits.
        std::array<double, 4> worst{};
        std::array<double, 4> sum{};
        std::array<size_t, 4> calls{};
#ifdef NIRHANDS_QNN16
        std::unique_ptr<hexecutor::Session> session;
        std::shared_ptr<nirhands::Models> cpu, decoder;
        const std::vector<nirhands::Models::Bytes>* pending = nullptr;
        std::mutex session_mutex;
        // Function scope: the runner lambda below outlives the block that creates it.
        auto note = [&](int k, const float* a, const float* b, size_t n) {
            double m = 0;
            for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
            worst[k] = std::max(worst[k], m); sum[k] += m; ++calls[k];
        };
#endif
        if (qnn16_dir.empty()) {
            if (check_cpu) throw std::runtime_error("--check-cpu needs --qnn16-dir");
            models = std::make_shared<nirhands::Models>(models_dir, threads);
        } else {
#ifdef NIRHANDS_QNN16
            if (artifacts.loader.empty() || artifacts.engine.empty() || artifacts.bootstrap.empty() || artifacts.chunk_probe.empty())
                throw std::runtime_error("--qnn16-dir needs --loader, --engine, --bootstrap and --chunk-probe");
            std::vector<hexecutor::Program> programs;
            for (const char* name : {"nirhands_detector", "nirhands_keypoints"}) {
                const auto prefix = (std::filesystem::path(qnn16_dir) / name).string();
                programs.push_back(hexecutor::load_qnn16_program(prefix + ".q16p", prefix + ".q16c"));
            }
            session = std::make_unique<hexecutor::Session>(std::move(programs), artifacts);
            if (check_cpu) {
                cpu = std::make_shared<nirhands::Models>(models_dir, threads);
                decoder = std::make_shared<nirhands::Models>(
                    models_dir, [&](size_t, const std::vector<nirhands::Models::Buffer>&, const nirhands::Geometry*) {
                        if (!pending) throw std::runtime_error("no NPU outputs to decode");
                        return *pending;
                    });
            }
            models = std::make_shared<nirhands::Models>(models_dir, [&](size_t index, const std::vector<nirhands::Models::Buffer>& inputs,
                                                                       const nirhands::Geometry* geometry) {
                // The pipeline runs two hands concurrently; one DSP session, one ORT slot and the
                // difference counters are shared, so serialize the whole call.
                std::lock_guard<std::mutex> lock(session_mutex);
                std::vector<hexecutor::Buffer> request;
                for (const auto& input : inputs) request.push_back({input.data, input.bytes});
                auto out = session->run(request, index);
                if (!cpu) return out;
                if (index == 0) {
                    // NHWC [2,M,M,10] from the NPU vs NCHW [2,10,M,M] from ORT; the input is [2,1,S,S] either way.
                    std::vector<float> maps;
                    cpu->detect(static_cast<const float*>(inputs[0].data), maps);
                    const int C = 10, M = int(std::lround(std::sqrt(double(maps.size() / (2 * C)))));
                    std::vector<float> nhwc(maps.size());
                    for (int b = 0; b < 2; ++b) for (int y = 0; y < M; ++y) for (int x = 0; x < M; ++x) for (int c = 0; c < C; ++c)
                        nhwc[((b * M + y) * M + x) * C + c] = maps[((b * C + c) * M + y) * M + x];
                    if (out.size() == 1 && out[0].size() == nhwc.size() * sizeof(float))
                        note(0, reinterpret_cast<const float*>(out[0].data()), nhwc.data(), nhwc.size());
                } else {
                    const int K = cpu->keypoint_size();
                    const auto* nhwc = static_cast<const float*>(inputs[0].data);
                    std::vector<float> nchw(size_t(2) * K * K);
                    for (int y = 0; y < K; ++y) for (int x = 0; x < K; ++x) for (int c = 0; c < 2; ++c)
                        nchw[(c * K + y) * K + x] = nhwc[(y * K + x) * 2 + c];
                    float kp[2 * nirhands::kJoints], disp[nirhands::kJoints], vis[nirhands::kJoints];
                    if (!geometry) throw std::runtime_error("keypoint callback has no crop geometry");
                    cpu->keypoints(0, nchw.data(), *static_cast<const float*>(inputs[1].data) != 0, *geometry, kp, disp, vis);
                    // Decode the NPU outputs exactly as the pipeline will (a skeleton model's six raw
                    // outputs go through the host FK), then compare final values with the CPU's.
                    float nkp[2 * nirhands::kJoints], ndisp[nirhands::kJoints], nvis[nirhands::kJoints];
                    pending = &out;
                    decoder->keypoints(0, nchw.data(), *static_cast<const float*>(inputs[1].data) != 0, *geometry, nkp, ndisp, nvis);
                    pending = nullptr;
                    note(1, nkp, kp, 2 * nirhands::kJoints);
                    note(2, ndisp, disp, nirhands::kJoints);
                    note(3, nvis, vis, nirhands::kJoints);
                }
                return out;
            });
#else
            throw std::runtime_error("built without NIRHANDS_QNN16");
#endif
        }
        nirhands::Pipeline pipe(rig, models);
        std::ifstream in(input, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read " + input);
        const int W = 1056, H = 1024;
        std::vector<uint8_t> left(size_t(W) * H), right(size_t(W) * H);
        std::cout.precision(7);
        double total = 0;
        int n = 0;
        Header h;
        while (in.read(reinterpret_cast<char*>(&h), sizeof h)) {
            if (std::memcmp(h.magic, "FRNIR001", 8) != 0) throw std::runtime_error("expected raw FRNIR001 pairs");
            if (!in.read(reinterpret_cast<char*>(left.data()), left.size()) ||
                !in.read(reinterpret_cast<char*>(right.data()), right.size()))
                break;  // a recording cut mid-pair ends cleanly
            const auto r = pipe.step({nirhands::Image{left.data(), W, H}, nirhands::Image{right.data(), W, H}},
                                     double(h.left_ns) * 1e-9);
            total += r.total_ms;
            ++n;
            if (!quiet) {
                std::cout << "{\"pair\":" << n - 1 << ",\"seq\":" << h.left_seq << ",\"detected\":" << r.detected
                          << ",\"detect_ms\":" << r.detect_ms << ",\"keypoints_ms\":" << r.keypoints_ms
                          << ",\"total_ms\":" << r.total_ms << ",\"seeds\":[";
                for (size_t k = 0; k < r.seeds.size(); ++k) {
                    const auto& s = r.seeds[k];
                    std::cout << (k ? "," : "") << "{\"center\":";
                    print_vec(std::cout, s.center);
                    std::cout << ",\"right\":" << s.is_right << ",\"score\":" << s.score << ",\"views\":" << s.views.size() << '}';
                }
                std::cout << "],\"hands\":[";
                for (size_t k = 0; k < r.hands.size(); ++k) {
                    const auto& hand = r.hands[k];
                    std::cout << (k ? "," : "") << "{\"id\":" << hand.id << ",\"right\":" << hand.is_right
                              << ",\"stereo\":" << hand.stereo << ",\"raw\":[";
                    for (int j = 0; j < nirhands::kJoints; ++j) { if (j) std::cout << ','; print_vec(std::cout, hand.raw[j]); }
                    std::cout << "],\"joints\":[";
                    for (int j = 0; j < nirhands::kJoints; ++j) { if (j) std::cout << ','; print_vec(std::cout, hand.joints[j]); }
                    std::cout << "]}";
                }
                std::cout << "]}\n";
            }
            if (max_pairs && n >= max_pairs) break;
        }
        std::cerr << "pairs " << n << " mean total_ms " << (n ? total / n : 0.0) << '\n';
        const char* names[] = {"detector maps", "keypoints", "disparity", "visibility_logit"};
        for (int k = 0; k < 4; ++k)
            if (calls[k])
                std::cerr << "npu vs cpu " << names[k] << ": calls " << calls[k] << " max " << worst[k]
                          << " mean-of-max " << sum[k] / calls[k] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "nirhands_replay: " << e.what() << '\n';
        return 1;
    }
}
