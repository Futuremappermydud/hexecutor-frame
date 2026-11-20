// SPDX-License-Identifier: LGPL-3.0-or-later
// Greedy decode of a staged LFM3 model: one token per step through N stage graphs (CPU reference, or the CDSP when
// the four engine files are given). Embedding lookup, the tied lm_head and argmax run on the host.
//
//   hexecutor_lfm_run STAGE_DIR N_STAGES LMAX EMBED_BF16 PROMPT_IDS_CSV N_NEW OUT_PREFIX [LOADER ENGINE BOOTSTRAP CHUNK_PROBE]
//
// STAGE_DIR holds stage<i>.hxfg/.bin (export_lfm3.py). EMBED_BF16 is the raw bfloat16 embedding table [vocab, 2048].
// Writes OUT_PREFIX_hidden.f32 (the last stage's output for every position fed) and prints the generated ids.
#include "hexecutor/executor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <fstream>

namespace {
constexpr size_t D = 2048, WINDOW = 256;

float bf16(uint16_t v) {
    uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint32_t argmax_logits(const std::vector<uint8_t> &emb, size_t vocab, const float *x, float *best_logit) {
    const unsigned workers = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::pair<float, uint32_t>> best(workers, {-INFINITY, 0});
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < workers; w++)
        threads.emplace_back([&, w] {
            const size_t lo = vocab * w / workers, hi = vocab * (w + 1) / workers;
            for (size_t v = lo; v < hi; v++) {
                const uint16_t *row = reinterpret_cast<const uint16_t *>(emb.data()) + v * D;
                float acc = 0;
                for (size_t i = 0; i < D; i++)
                    acc += bf16(row[i]) * x[i];
                if (acc > best[w].first)
                    best[w] = {acc, uint32_t(v)};
            }
        });
    for (auto &t : threads)
        t.join();
    auto top = *std::max_element(best.begin(), best.end(), [](auto &a, auto &b) { return a.first < b.first; });
    if (best_logit)
        *best_logit = top.first;
    return top.second;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 8 && argc != 12) {
            std::cerr << "usage: hexecutor_lfm_run STAGE_DIR N_STAGES LMAX EMBED_BF16 PROMPT_IDS_CSV N_NEW OUT_PREFIX "
                         "[LOADER ENGINE BOOTSTRAP CHUNK_PROBE]\n";
            return 2;
        }
        const std::filesystem::path dir = argv[1];
        const size_t n_stages = std::stoul(argv[2]), lmax = std::stoul(argv[3]);
        auto emb = hexecutor::read_file(argv[4]);
        const size_t vocab = emb.size() / (D * 2);
        std::vector<uint32_t> tokens;
        {
            std::stringstream ss(argv[5]);
            std::string item;
            while (std::getline(ss, item, ','))
                tokens.push_back(uint32_t(std::stoul(item)));
        }
        const size_t prompt_len = tokens.size(), n_new = std::stoul(argv[6]);
        if (prompt_len + n_new > lmax + 1)
            throw std::runtime_error("prompt + new tokens exceed LMAX");
        const std::string prefix = argv[7];
        const bool device = argc == 12;

        std::vector<hexecutor::Program> programs;
        for (size_t i = 0; i < n_stages; i++) {
            programs.push_back(hexecutor::load_tensor_program(dir / ("stage" + std::to_string(i) + ".hxfg"),
                                                              dir / ("stage" + std::to_string(i) + ".bin")));
            const auto &in = programs.back().inputs;
            if (in.size() != 4 || in[0].bytes() != D * 4 || in[1].bytes() != 4 || in[2].bytes() != lmax * 4 ||
                in[3].bytes() != lmax * 4)
                throw std::runtime_error("stage " + std::to_string(i) + " inputs are not x, pos, mask_full, mask_slide");
        }
        std::unique_ptr<hexecutor::Session> session;
        std::vector<std::unique_ptr<hexecutor::TensorReference>> reference;
        if (device) {
            hexecutor::Artifacts a;
            a.loader = argv[8];
            a.engine = argv[9];
            a.bootstrap = argv[10];
            a.chunk_probe = argv[11];
            a.tensor_hmx = std::getenv("HEXECUTOR_TENSOR_PRECISE") == nullptr;
            a.tensor_fp16 = std::getenv("HEXECUTOR_TENSOR_F32") == nullptr;
            if (const char *v = std::getenv("HEXECUTOR_MAX_WAKE_SECONDS"))
                a.max_wake_seconds = std::stod(v);
            session = std::make_unique<hexecutor::Session>(programs, a);
        } else {
            for (auto &p : programs)
                reference.push_back(std::make_unique<hexecutor::TensorReference>(p));
        }

        std::ofstream hidden(prefix + "_hidden.f32", std::ios::binary);
        std::vector<std::ofstream> stage_dump;
        if (std::getenv("LFM_DUMP_STAGES"))
            for (size_t s = 0; s < n_stages; s++)
                stage_dump.emplace_back(prefix + "_stage" + std::to_string(s) + ".f32", std::ios::binary);
        std::vector<float> x(D), mask_full(lmax), mask_slide(lmax), pos(1);
        std::vector<uint32_t> generated;
        FILE *op_profile = std::getenv("LFM_OP_PROFILE") ? std::fopen(std::getenv("LFM_OP_PROFILE"), "w") : nullptr;
        const size_t steps = prompt_len + n_new - 1;
        for (size_t p = 0; p < steps; p++) {
            const uint32_t token = p < prompt_len ? tokens[p] : generated[p - prompt_len];
            const uint16_t *row = reinterpret_cast<const uint16_t *>(emb.data()) + size_t(token) * D;
            for (size_t i = 0; i < D; i++)
                x[i] = bf16(row[i]);
            pos[0] = float(p);
            std::fill(mask_full.begin(), mask_full.end(), -1e9f);
            std::fill(mask_slide.begin(), mask_slide.end(), -1e9f);
            for (size_t k = 0; k <= p; k++)
                mask_full[k] = 0;
            for (size_t k = p + 1 > WINDOW ? p + 1 - WINDOW : 0; k <= p; k++)
                mask_slide[k] = 0;
            auto start = std::chrono::steady_clock::now();
            double dsp_ms = 0;
            for (size_t s = 0; s < n_stages; s++) {
                std::vector<hexecutor::Buffer> in = {
                    {reinterpret_cast<const uint8_t *>(x.data()), D * 4},
                    {reinterpret_cast<const uint8_t *>(pos.data()), 4},
                    {reinterpret_cast<const uint8_t *>(mask_full.data()), lmax * 4},
                    {reinterpret_cast<const uint8_t *>(mask_slide.data()), lmax * 4}};
                auto out = device ? session->run(in, s) : reference[s]->run(in);
                if (device) {
                    const auto &pr = session->profile();
                    dsp_ms += pr.timer_ticks / 19200.0;
                    if (op_profile)
                        std::fprintf(op_profile, "P %zu S %zu cycles %llu ticks %llu matrix %u blocked %d ops %zu\n", p, s,
                                     (unsigned long long)pr.cycles, (unsigned long long)pr.timer_ticks, pr.matrix_ops,
                                     int(pr.blocked_fp16), pr.op_cycles.size());
                    if (op_profile)
                        for (size_t i = 0; i < pr.op_cycles.size(); i++)
                            std::fprintf(op_profile, "%zu %llu\n", i, (unsigned long long)pr.op_cycles[i]);
                }
                if (out.empty() || out[0].size() != D * 4)
                    throw std::runtime_error("stage output has the wrong size");
                std::memcpy(x.data(), out[0].data(), D * 4);
                if (!stage_dump.empty())
                    stage_dump[s].write(reinterpret_cast<const char *>(x.data()), D * 4);
            }
            hidden.write(reinterpret_cast<const char *>(x.data()), D * 4);
            std::cerr << "pos " << p << " token " << token << " stages_ms "
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (device)
                std::cerr << " dsp_ms " << dsp_ms;
            if (p + 1 >= prompt_len) {
                float logit = 0;
                const uint32_t next = argmax_logits(emb, vocab, x.data(), &logit);
                generated.push_back(next);
                std::cerr << " -> " << next << " (logit " << logit << ")";
            }
            std::cerr << '\n';
        }
        if (session)
            session->close();
        std::cout << "generated:";
        for (auto g : generated)
            std::cout << ' ' << g;
        std::cout << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
