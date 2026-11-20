// SPDX-License-Identifier: LGPL-3.0-or-later
// Multi-step runner for graphs with persistent state (conv rings, KV caches): the same session, and so the
// same state region, is fed one step after another. CPU reference by default; with the four engine files it
// runs on the CDSP. Arguments stay explicit, as in hexecutor_tensor_run.
//
//   hexecutor_decode_run IMAGE CONSTANTS STEPS OUT_PREFIX INPUT_F32... [LOADER ENGINE BOOTSTRAP CHUNK_PROBE]
//
// Each INPUT_F32 file holds STEPS consecutive inputs for one graph input (compact float32). Output k of
// every step is appended to OUT_PREFIX<k>.f32; per-step timing goes to stderr.
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

int main(int argc, char **argv) {
    try {
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        const char *usage = "usage: hexecutor_decode_run IMAGE CONSTANTS STEPS OUT_PREFIX INPUT_F32... "
                            "[LOADER ENGINE BOOTSTRAP CHUNK_PROBE]\n";
        if (argc < 6) {
            std::cerr << usage;
            return 2;
        }
        auto program = hexecutor::load_tensor_program(argv[1], argv[2]);
        const size_t n_in = program.inputs.size();
        if (argc != int(5 + n_in) && argc != int(9 + n_in)) {
            std::cerr << usage;
            return 2;
        }
        const int steps = std::stoi(argv[3]);
        if (steps < 1 || steps > 100000)
            throw std::runtime_error("STEPS must be 1..100000");
        const std::string prefix = argv[4];
        std::vector<hexecutor::Bytes> input_files;
        for (size_t i = 0; i < n_in; i++) {
            input_files.push_back(hexecutor::read_file(argv[5 + i]));
            if (input_files.back().size() != program.inputs[i].bytes() * size_t(steps))
                throw std::runtime_error("input " + std::to_string(i) + " must hold STEPS inputs of " +
                                         std::to_string(program.inputs[i].bytes()) + " bytes");
        }
        const bool device = argc == int(9 + n_in);
        std::unique_ptr<hexecutor::Session> session;
        std::unique_ptr<hexecutor::TensorReference> reference;
        if (device) {
            hexecutor::Artifacts a;
            a.loader = argv[5 + n_in];
            a.engine = argv[6 + n_in];
            a.bootstrap = argv[7 + n_in];
            a.chunk_probe = argv[8 + n_in];
            a.tensor_hmx = std::getenv("HEXECUTOR_TENSOR_PRECISE") == nullptr;
            a.tensor_fp16 = std::getenv("HEXECUTOR_TENSOR_F32") == nullptr;
            if (const char *v = std::getenv("HEXECUTOR_MAX_WAKE_SECONDS"))
                a.max_wake_seconds = std::stod(v);
            session = std::make_unique<hexecutor::Session>(program, a);
        } else {
            reference = std::make_unique<hexecutor::TensorReference>(program);
        }
        std::vector<std::ofstream> outputs;
        for (size_t k = 0; k < program.outputs.size(); k++)
            outputs.emplace_back(prefix + std::to_string(k) + ".f32", std::ios::binary);
        int done = 0;
        for (; done < steps && !interrupted; done++) {
            std::vector<hexecutor::Buffer> buffers;
            for (size_t i = 0; i < n_in; i++) {
                const size_t bytes = program.inputs[i].bytes();
                buffers.push_back({input_files[i].data() + size_t(done) * bytes, bytes});
            }
            auto start = std::chrono::steady_clock::now();
            auto out = device ? session->run(buffers) : reference->run(buffers);
            std::cerr << "step " << done << " ms "
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (device)
                std::cerr << " dsp_ms " << session->profile().timer_ticks / 19200.0;
            std::cerr << '\n';
            for (size_t k = 0; k < out.size(); k++)
                if (!outputs[k].write(reinterpret_cast<const char *>(out[k].data()), out[k].size()))
                    throw std::runtime_error("cannot write output");
        }
        if (session)
            session->close();
        std::cerr << "state_bytes " << (device ? session->state_bytes() : reference->state_bytes()) << " steps "
                  << done << '\n';
        return interrupted ? 128 + interrupted : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
