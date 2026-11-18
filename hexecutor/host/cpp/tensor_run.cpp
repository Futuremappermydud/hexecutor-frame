#include "hexecutor/executor.hpp"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int signal) { interrupted = signal; }
} // namespace
// Portable CPU reference or native DSP run. Arguments stay explicit, so a
// benchmark never silently picks a depth package or a stale engine.
int main(int argc, char **argv) {
    try {
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        if (argc != 5 && argc != 9) {
            std::cerr << "usage: hexecutor_tensor_run IMAGE CONSTANTS INPUT_F32 OUTPUT_PREFIX "
                         "[LOADER ENGINE BOOTSTRAP CHUNK_PROBE]\n";
            return 2;
        }
        auto p = hexecutor::load_tensor_program(argv[1], argv[2]);
        auto data = hexecutor::read_file(argv[3]);
        if (p.inputs.size() != 1 || data.size() != p.inputs[0].bytes())
            throw std::runtime_error("expected one input of the declared size");
        std::vector<hexecutor::Bytes> out;
        if (argc == 9) {
            hexecutor::Artifacts a;
            a.loader = argv[5];
            a.engine = argv[6];
            a.bootstrap = argv[7];
            a.chunk_probe = argv[8];
            a.tensor_hmx = std::getenv("HEXECUTOR_TENSOR_PRECISE") == nullptr;
            a.tensor_fp16 = std::getenv("HEXECUTOR_TENSOR_F32") == nullptr;
            // Wake-cost measurement: a tiny wake limit re-wakes before every run.
            if (const char *v = std::getenv("HEXECUTOR_MAX_WAKE_SECONDS"))
                a.max_wake_seconds = std::stod(v);
            int runs = 4;
            if (const char *v = std::getenv("HEXECUTOR_RUNS"))
                runs = std::stoi(v);
            if (runs < 1 || runs > 100000)
                throw std::runtime_error("HEXECUTOR_RUNS must be 1..100000");
            hexecutor::Session session(p, a);
            for (int i = 0; i < runs && !interrupted; i++) {
                auto start = std::chrono::steady_clock::now();
                out = session.run({{data.data(), data.size()}});
                std::cerr << "run " << i << " ms "
                          << std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count()
                          << " dsp_ms " << session.profile().timer_ticks / 19200.0 << '\n';
            }
            const auto &profile = session.profile();
            std::ofstream pf(std::string(argv[4]) + "profile.json");
            pf << "{\"cycles\":" << profile.cycles << ",\"timer_ticks\":" << profile.timer_ticks
               << ",\"timer_hz\":19200000"
               << ",\"matrix_ops\":" << profile.matrix_ops
               << ",\"blocked_fp16\":" << (profile.blocked_fp16 ? "true" : "false")
               << ",\"op_cycles\":[";
            for (size_t i = 0; i < profile.op_cycles.size(); i++)
                pf << (i ? "," : "") << profile.op_cycles[i];
            pf << "]}\n";
            session.close();
        } else {
            extern std::vector<hexecutor::Bytes> tensor_reference(
                const hexecutor::Program &, const std::vector<hexecutor::Buffer> &);
            out = tensor_reference(p, {{data.data(), data.size()}});
        }
        for (size_t i = 0; i < out.size(); i++) {
            std::ofstream f(std::string(argv[4]) + std::to_string(i) + ".f32", std::ios::binary);
            if (!f.write(reinterpret_cast<const char *>(out[i].data()), out[i].size()))
                throw std::runtime_error("cannot write output");
        }
        return interrupted ? 128 + interrupted : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
