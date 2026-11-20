// SPDX-License-Identifier: LGPL-3.0-or-later
// hexecutor_et_run: loads a .pte, runs its "forward" method through the
// HexecutorBackend on seeded random or file-backed float32 inputs, and prints output summaries.
//
//   hexecutor_et_run MODEL.pte [--seed N | --input INPUT_F32 ...]
//
// The backend's mode follows the environment (see README.md). For a Mac run,
// export HEXECUTOR_ET_REFERENCE=1; no device variables are read in that mode.
#include "hexecutor_backend.hpp"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::extension::FileDataLoader;
using executorch::extension::MallocMemoryAllocator;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Span;

namespace {

int fail(const char *what) {
    std::fprintf(stderr, "hexecutor_et_run: %s\n", what);
    return 1;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: hexecutor_et_run MODEL.pte [--seed N | --input INPUT_F32 ...] "
                     "[--iters N] [--warmup N]\n");
        return 2;
    }
    uint32_t seed = 1;
    int iters = 1, warmup = 0;
    std::vector<const char *> input_paths;
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_paths.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::max(0, std::atoi(argv[++i]));
        else
            return fail("unknown argument");
    }
    if (hexecutor_et::ensure_registered() != executorch::runtime::Error::Ok)
        return fail("HexecutorBackend is not registered");
    executorch::runtime::runtime_init();

    auto loader = FileDataLoader::from(argv[1]);
    if (!loader.ok())
        return fail("cannot open .pte");
    auto program = Program::load(&loader.get());
    if (!program.ok())
        return fail("cannot load .pte");
    const char *method_name = "forward";
    auto meta = program->method_meta(method_name);
    if (!meta.ok())
        return fail("no forward method");

    // Planned buffers are owned here; the method borrows them for its lifetime.
    const size_t n_planned = meta->num_memory_planned_buffers();
    std::vector<std::vector<uint8_t>> planned_bytes;
    std::vector<Span<uint8_t>> planned_spans;
    planned_bytes.reserve(n_planned);
    planned_spans.reserve(n_planned);
    for (size_t i = 0; i < n_planned; i++) {
        auto size = meta->memory_planned_buffer_size(i);
        if (!size.ok())
            return fail("bad planned buffer size");
        planned_bytes.emplace_back(size_t(size.get()));
        planned_spans.emplace_back(planned_bytes.back().data(), planned_bytes.back().size());
    }
    HierarchicalAllocator planned(Span<Span<uint8_t>>(planned_spans.data(), planned_spans.size()));
    MallocMemoryAllocator method_alloc;
    MallocMemoryAllocator temp_alloc;
    MemoryManager memory(&method_alloc, &planned, &temp_alloc);
    // Validate file-backed inputs before loading the method, which opens the DSP session.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t n_in = meta->num_inputs();
    if (!input_paths.empty() && input_paths.size() != n_in)
        return fail("supply one --input file per model input in model order");
    std::vector<std::vector<float>> input_data;
    std::vector<executorch::extension::TensorPtr> input_tensors;
    input_data.reserve(n_in);
    input_tensors.reserve(n_in);
    for (size_t i = 0; i < n_in; i++) {
        auto info = meta->input_tensor_meta(i);
        if (!info.ok() || info->scalar_type() != ScalarType::Float)
            return fail("only float32 inputs are supported");
        std::vector<SizesType> sizes(info->sizes().begin(), info->sizes().end());
        size_t n = 1;
        for (SizesType s : sizes)
            n *= size_t(s);
        auto &data = input_data.emplace_back(n);
        if (input_paths.empty()) {
            for (float &v : data)
                v = dist(rng);
        } else {
            std::ifstream file(input_paths[i], std::ios::binary | std::ios::ate);
            const auto bytes = static_cast<std::streamsize>(data.size() * sizeof(float));
            if (!file || file.tellg() != bytes || !file.seekg(0) ||
                !file.read(reinterpret_cast<char *>(data.data()), bytes)) {
                std::fprintf(stderr, "hexecutor_et_run: expected %lld bytes from input %zu (%s)\n",
                             static_cast<long long>(bytes), i, input_paths[i]);
                return 1;
            }
        }
        input_tensors.push_back(executorch::extension::from_blob(
            data.data(), sizes, ScalarType::Float));
    }
    const auto load_start = std::chrono::steady_clock::now();
    auto method = program->load_method(method_name, &memory);
    if (!method.ok())
        return fail("cannot load method");
    const double load_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - load_start)
                               .count();
    for (size_t i = 0; i < n_in; i++) {
        if (method->set_input(EValue(*input_tensors[i]), i) != executorch::runtime::Error::Ok)
            return fail("set_input failed");
    }

    // The first execute is reported separately: it stages resident records.
    std::vector<double> times;
    for (int k = 0; k < 1 + warmup + iters; k++) {
        auto start = std::chrono::steady_clock::now();
        if (method->execute() != executorch::runtime::Error::Ok)
            return fail("execute failed");
        const double t = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        if (k == 0 || k > warmup)
            times.push_back(t);
    }
    const double ms = times[0];

    const size_t n_out = meta->num_outputs();
    for (size_t i = 0; i < n_out; i++) {
        const EValue &out = method->get_output(i);
        if (!out.isTensor())
            continue;
        auto t = out.toTensor();
        const float *d = t.const_data_ptr<float>();
        const size_t n = size_t(t.numel());
        double sum = 0;
        for (size_t k = 0; k < n; k++)
            sum += d[k];
        std::printf("output %zu: dims", i);
        for (ssize_t k = 0; k < t.dim(); k++)
            std::printf(" %d", int(t.size(k)));
        std::printf(" sum=%.6f first=", sum);
        for (size_t k = 0; k < n && k < 4; k++)
            std::printf("%s%.6f", k ? "," : "", d[k]);
        std::printf("\n");
        if (const char *dump = std::getenv("HEXECUTOR_ET_DUMP")) {
            // Raw float32 of every output, appended in order, for offline comparison.
            if (FILE *fp = std::fopen(dump, i ? "ab" : "wb")) {
                std::fwrite(d, sizeof(float), n, fp);
                std::fclose(fp);
            }
        }
    }
    std::printf("load_method_ms %.3f\n", load_ms);
    std::printf("execute_ms %.3f\n", ms);
    if (times.size() > 1) {
        std::vector<double> warm(times.begin() + 1, times.end());
        std::sort(warm.begin(), warm.end());
        double sum = 0;
        for (double t : warm)
            sum += t;
        std::printf("warm_ms n=%zu min=%.3f median=%.3f p90=%.3f max=%.3f mean=%.3f\n",
                    warm.size(), warm.front(), warm[warm.size() / 2],
                    warm[std::min(warm.size() - 1, warm.size() * 9 / 10)], warm.back(),
                    sum / double(warm.size()));
    }
    return 0;
}
