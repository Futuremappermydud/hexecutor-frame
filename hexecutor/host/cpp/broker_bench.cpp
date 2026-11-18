// Round-robin benchmark client for hexecutord: one stream per named graph,
// fixed inputs, N rounds. Reports client-observed latency next to the broker's
// own queue/run/DSP times, so IPC overhead and graph-switch costs can be read
// directly.
// usage: hexecutor_broker_bench SOCKET ROUNDS NAME=INPUT.f32 [NAME=INPUT.f32 ...]
#include "hexecutor/broker.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace hexecutor;

int main(int argc, char **argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: hexecutor_broker_bench SOCKET ROUNDS NAME=INPUT.f32 ...\n";
            return 2;
        }
        broker::Client client(argv[1]);
        const int rounds = std::stoi(argv[2]);
        struct Graph {
            std::string name;
            Bytes input;
            broker::Client::Stream stream;
            std::vector<double> client_ms, queue_ms, run_ms, dsp_ms;
        };
        std::vector<Graph> graphs;
        for (int i = 3; i < argc; i++) {
            std::string a = argv[i];
            auto eq = a.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error("expected NAME=INPUT.f32: " + a);
            Graph g;
            g.name = a.substr(0, eq);
            g.input = read_file(a.substr(eq + 1));
            broker::StreamSpec spec;
            spec.graph = g.name;
            spec.latest_only = false;
            spec.deadline_ms = 10000;
            g.stream = client.open(spec);
            if (g.stream.input_bytes.size() != 1 || g.stream.input_bytes[0] != g.input.size())
                throw std::runtime_error("input size does not match graph " + g.name);
            graphs.push_back(std::move(g));
        }
        for (int r = 0; r < rounds; r++)
            for (auto &g : graphs) {
                auto t0 = broker::Clock::now();
                auto res = client.wait(client.submit(g.stream, {{g.input.data(), g.input.size()}}));
                double ms =
                    std::chrono::duration<double, std::milli>(broker::Clock::now() - t0).count();
                if (res.status != broker::Status::ok)
                    throw std::runtime_error(g.name + ": " + broker::to_string(res.status) + " " +
                                             res.error);
                if (ms > 20)
                    std::fprintf(stderr, "[bench] slow %s round %d: %.3f ms (queue %.3f run %.3f dsp %.3f)\n",
                                 g.name.c_str(), r, ms, res.queue_ms, res.run_ms, res.dsp_ms);
                if (r == 0)
                    continue; // first round includes session open and first staging
                g.client_ms.push_back(ms);
                g.queue_ms.push_back(res.queue_ms);
                g.run_ms.push_back(res.run_ms);
                g.dsp_ms.push_back(res.dsp_ms);
            }
        auto mean = [](const std::vector<double> &v) {
            return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
        };
        auto p95 = [](std::vector<double> v) {
            if (v.empty())
                return 0.0;
            std::sort(v.begin(), v.end());
            return v[size_t(0.95 * double(v.size() - 1))];
        };
        for (auto &g : graphs) {
            double worst = g.client_ms.empty()
                               ? 0
                               : *std::max_element(g.client_ms.begin(), g.client_ms.end());
            size_t slow = size_t(std::count_if(g.client_ms.begin(), g.client_ms.end(),
                                               [](double v) { return v > 20; }));
            std::printf("%-12s n %4zu  client %.3f ms (p95 %.3f, max %.3f, >20 ms %zu)  queue %.3f  "
                        "run %.3f  dsp %.3f  ipc %.3f\n",
                        g.name.c_str(), g.client_ms.size(), mean(g.client_ms), p95(g.client_ms),
                        worst, slow, mean(g.queue_ms), mean(g.run_ms), mean(g.dsp_ms),
                        mean(g.client_ms) - mean(g.queue_ms) - mean(g.run_ms));
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "hexecutor_broker_bench: " << e.what() << '\n';
        return 1;
    }
}
