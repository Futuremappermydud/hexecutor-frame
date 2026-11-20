// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/executor.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int signal) { interrupted = signal; }
struct Graph {
    std::string prefix;
    std::vector<std::string> inputs;
};
double percentile(std::vector<double> v, double q) {
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    return v[size_t(q * double(v.size() - 1) + 0.5)];
}
} // namespace
// Runs one or more lowered Q16P programs (PREFIX.q16p / PREFIX.q16c) in one
// DSP session, round-robin, and reports host and DSP time per program.
// Every argument is explicit; nothing is looked up by name.
int main(int argc, char **argv) {
    try {
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        if (argc < 9) {
            std::cerr << "usage: hexecutor_qnn16_run LOADER ENGINE BOOTSTRAP CHUNK_PROBE OUT_PREFIX "
                         "--program PREFIX INPUT_F32... [--program PREFIX INPUT_F32...]\n"
                         "env: HEXECUTOR_RUNS (default 20), HEXECUTOR_MAX_WAKE_SECONDS\n";
            return 2;
        }
        hexecutor::Artifacts a;
        a.loader = argv[1];
        a.engine = argv[2];
        a.bootstrap = argv[3];
        a.chunk_probe = argv[4];
        const std::string out_prefix = argv[5];
        if (const char *v = std::getenv("HEXECUTOR_MAX_WAKE_SECONDS"))
            a.max_wake_seconds = std::stod(v);
        int runs = 20;
        if (const char *v = std::getenv("HEXECUTOR_RUNS"))
            runs = std::stoi(v);
        if (runs < 1 || runs > 100000)
            throw std::runtime_error("HEXECUTOR_RUNS must be 1..100000");
        std::vector<Graph> graphs;
        for (int i = 6; i < argc; i++) {
            if (!std::strcmp(argv[i], "--program")) {
                if (i + 1 >= argc)
                    throw std::runtime_error("--program needs a prefix");
                graphs.push_back({argv[++i], {}});
            } else if (graphs.empty())
                throw std::runtime_error("input before --program");
            else
                graphs.back().inputs.push_back(argv[i]);
        }
        std::vector<hexecutor::Program> programs;
        std::vector<std::vector<hexecutor::Bytes>> data;
        std::vector<std::vector<hexecutor::Buffer>> buffers;
        for (const auto &g : graphs) {
            programs.push_back(hexecutor::load_qnn16_program(g.prefix + ".q16p", g.prefix + ".q16c"));
            const auto &p = programs.back();
            if (p.inputs.size() != g.inputs.size())
                throw std::runtime_error(g.prefix + ": expected " + std::to_string(p.inputs.size()) +
                                         " inputs");
            data.emplace_back();
            for (size_t i = 0; i < g.inputs.size(); i++) {
                data.back().push_back(hexecutor::read_file(g.inputs[i]));
                if (data.back().back().size() != p.inputs[i].bytes())
                    throw std::runtime_error(g.inputs[i] + ": wrong size");
            }
        }
        for (auto &d : data) {
            buffers.emplace_back();
            for (auto &b : d)
                buffers.back().push_back({b.data(), b.size()});
        }
        auto opened = std::chrono::steady_clock::now();
        hexecutor::Session session(programs, a);
        std::cerr << "session open ms "
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - opened)
                         .count()
                  << '\n';
        std::vector<std::vector<double>> host_ms(graphs.size()), dsp_ms(graphs.size());
        std::vector<std::vector<hexecutor::Bytes>> last(graphs.size());
        std::vector<hexecutor::ExecutionProfile> profiles(graphs.size());
        for (int r = 0; r < runs && !interrupted; r++)
            for (size_t g = 0; g < graphs.size() && !interrupted; g++) {
                auto start = std::chrono::steady_clock::now();
                last[g] = session.run(buffers[g], g);
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                                .count();
                const auto &p = session.profile();
                host_ms[g].push_back(ms);
                dsp_ms[g].push_back(double(p.timer_ticks) / 19200.0);
                profiles[g] = p;
                std::cerr << "run " << r << " program " << g << " ms " << ms << " dsp_ms "
                          << dsp_ms[g].back() << '\n';
            }
        session.close();
        std::ofstream summary(out_prefix + "summary.json");
        summary << "{\"programs\":[";
        for (size_t g = 0; g < graphs.size(); g++) {
            // Drop the first run: it pays first-touch costs after the session opens.
            std::vector<double> h(host_ms[g].begin() + (host_ms[g].size() > 1),
                                  host_ms[g].end()),
                d(dsp_ms[g].begin() + (dsp_ms[g].size() > 1), dsp_ms[g].end());
            std::cerr << graphs[g].prefix << ": dsp ms p50 " << percentile(d, 0.5) << " p90 "
                      << percentile(d, 0.9) << ", host ms p50 " << percentile(h, 0.5) << " p90 "
                      << percentile(h, 0.9) << ", matrix ops " << profiles[g].matrix_ops << '\n';
            summary << (g ? "," : "") << "{\"prefix\":\"" << graphs[g].prefix << "\",\"runs\":"
                    << host_ms[g].size() << ",\"dsp_ms_p50\":" << percentile(d, 0.5)
                    << ",\"dsp_ms_p90\":" << percentile(d, 0.9)
                    << ",\"host_ms_p50\":" << percentile(h, 0.5) << ",\"host_ms_p90\":"
                    << percentile(h, 0.9) << ",\"op_cycles\":[";
            for (size_t i = 0; i < profiles[g].op_cycles.size(); i++)
                summary << (i ? "," : "") << profiles[g].op_cycles[i];
            summary << "]}";
            for (size_t i = 0; i < last[g].size(); i++) {
                std::ofstream f(out_prefix + std::to_string(g) + "_" + std::to_string(i) + ".f32",
                                std::ios::binary);
                if (!f.write(reinterpret_cast<const char *>(last[g][i].data()), last[g][i].size()))
                    throw std::runtime_error("cannot write output");
            }
        }
        summary << "]}\n";
        return interrupted ? 128 + interrupted : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
