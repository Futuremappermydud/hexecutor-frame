// hexecutor_et_selftest: drives the HexecutorBackend directly, without a .pte.
//
//   HEXECUTOR_ET_REFERENCE=1 hexecutor_et_selftest PROGRAM.hxfg CONSTANTS.bin \
//       [INPUT.f32 ORACLE.f32]
//
// With INPUT.f32 and ORACLE.f32 the backend output is also compared with the
// stored float32 oracle (max absolute difference is printed).
//
// It wraps an existing HXFG image in a v1 blob and checks that:
//   1. the blob initializes, executes and destroys through the registered backend;
//   2. a rank-4 input given as NCHW (layout 1) gives the same outputs as the same
//      data given as NHWC (layout 0), so the transpose path is exercised;
//   3. a wrongly shaped input is rejected with Error::InvalidArgument.
// Device mode works the same way when the HEXECUTOR_* artifact variables are set.
#include "blob.hpp"
#include "hexecutor/executor.hpp"
#include "hexecutor_backend.hpp"

#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/backend_execution_context.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::extension::from_blob;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::get_backend_class;
using executorch::runtime::Span;

namespace {

struct IoSpec {
    uint32_t layout;
    std::vector<uint32_t> dims;
};

void put32(std::vector<uint8_t> &b, uint32_t v) {
    for (int i = 0; i < 4; i++)
        b.push_back(uint8_t(v >> (8 * i)));
}

void put64(std::vector<uint8_t> &b, uint64_t v) {
    put32(b, uint32_t(v));
    put32(b, uint32_t(v >> 32));
}

std::vector<uint8_t> read_all(const char *path) {
    std::vector<uint8_t> out;
    if (FILE *f = std::fopen(path, "rb")) {
        uint8_t buf[1 << 16];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
            out.insert(out.end(), buf, buf + n);
        std::fclose(f);
    }
    return out;
}

std::vector<uint8_t> make_blob(const std::vector<uint8_t> &image,
                               const std::vector<uint8_t> &constants,
                               const std::vector<IoSpec> &ins, const std::vector<IoSpec> &outs) {
    std::vector<uint8_t> b = {'H', 'X', 'E', 'T'};
    put32(b, 1);
    put32(b, 1);
    put32(b, uint32_t(ins.size()));
    put32(b, uint32_t(outs.size()));
    for (const auto *list : {&ins, &outs})
        for (const auto &t : *list) {
            put32(b, t.layout);
            put32(b, uint32_t(t.dims.size()));
            for (uint32_t d : t.dims)
                put32(b, d);
        }
    put64(b, image.size());
    put64(b, constants.size());
    b.insert(b.end(), image.begin(), image.end());
    b.insert(b.end(), constants.begin(), constants.end());
    return b;
}

std::vector<SizesType> sizes_of(const std::vector<uint32_t> &dims) {
    std::vector<SizesType> s;
    for (uint32_t d : dims)
        s.push_back(SizesType(d));
    return s;
}

// Runs one init/execute/destroy cycle. Returns the execute error and fills `out`.
Error run_backend(const BackendInterface *backend, const std::vector<uint8_t> &blob,
                  const std::vector<SizesType> &in_sizes, float *input,
                  const std::vector<SizesType> &out_sizes, std::vector<float> *out) {
    FreeableBuffer buffer(blob.data(), blob.size(), nullptr, nullptr);
    BackendInitContext init_ctx(nullptr);
    auto handle = backend->init(init_ctx, &buffer, ArrayRef<CompileSpec>());
    if (!handle.ok()) {
        std::fprintf(stderr, "init failed: %d\n", int(handle.error()));
        return handle.error();
    }
    auto in_t = from_blob(input, in_sizes, ScalarType::Float);
    auto out_t = from_blob(out->data(), out_sizes, ScalarType::Float);
    EValue in_v(*in_t);
    EValue out_v(*out_t);
    EValue *args[2] = {&in_v, &out_v};
    BackendExecutionContext exec_ctx;
    const Error err = backend->execute(exec_ctx, handle.get(), Span<EValue *>(args, 2));
    backend->destroy(handle.get());
    return err;
}

int fail(const char *what) {
    std::fprintf(stderr, "hexecutor_et_selftest: FAIL %s\n", what);
    return 1;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5)
        return fail("usage: hexecutor_et_selftest PROGRAM.hxfg CONSTANTS.bin [INPUT.f32 ORACLE.f32]");
    if (hexecutor_et::ensure_registered() != Error::Ok)
        return fail("backend not registered");
    const BackendInterface *backend = get_backend_class("HexecutorBackend");
    if (!backend || !backend->is_available())
        return fail("HexecutorBackend unavailable");

    hexecutor::Program program;
    try {
        program = hexecutor::load_tensor_program(argv[1], argv[2]);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return fail("HXFG preflight");
    }
    if (program.inputs.size() != 1 || program.outputs.size() != 1)
        return fail("selftest expects one input and one output");
    const auto image = read_all(argv[1]);
    const auto constants = read_all(argv[2]);

    std::vector<uint32_t> in_dims, out_dims;
    for (size_t v : program.inputs[0].shape)
        in_dims.push_back(uint32_t(v));
    for (size_t v : program.outputs[0].shape)
        out_dims.push_back(uint32_t(v));
    const size_t n_in = program.inputs[0].bytes() / sizeof(float);
    const size_t n_out = program.outputs[0].bytes() / sizeof(float);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n_in);
    for (float &v : x)
        v = dist(rng);
    if (argc == 5) {
        const auto input = read_all(argv[3]);
        const auto oracle = read_all(argv[4]);
        if (input.size() != n_in * sizeof(float))
            return fail("INPUT.f32 size does not match the program input");
        std::memcpy(x.data(), input.data(), input.size());
        if (oracle.size() != n_out * sizeof(float))
            return fail("ORACLE.f32 size does not match the program output");
        std::vector<float> out(n_out);
        auto blob = make_blob(image, constants, {{0, in_dims}}, {{0, out_dims}});
        if (run_backend(backend, blob, sizes_of(in_dims), x.data(), sizes_of(out_dims), &out) !=
            Error::Ok)
            return fail("oracle execute");
        float max_abs = 0;
        for (size_t i = 0; i < n_out; i++) {
            float ref;
            std::memcpy(&ref, oracle.data() + i * sizeof(float), sizeof(float));
            max_abs = std::fmax(max_abs, std::fabs(out[i] - ref));
        }
        std::printf("oracle max_abs_diff %.9g over %zu values\n", max_abs, n_out);
        return max_abs <= 1e-3f ? (std::printf("hexecutor_et_selftest: PASS\n"), 0)
                                : fail("output differs from oracle");
    }

    // 1. As-is run.
    std::vector<float> out_a(n_out);
    auto blob_a = make_blob(image, constants, {{0, in_dims}}, {{0, out_dims}});
    if (run_backend(backend, blob_a, sizes_of(in_dims), x.data(), sizes_of(out_dims), &out_a) !=
        Error::Ok)
        return fail("as-is execute");
    for (float v : out_a)
        if (!std::isfinite(v))
            return fail("non-finite output");

    // 2. NCHW transpose check on a rank-4 NHWC input.
    if (in_dims.size() == 4) {
        const uint32_t nhwc[4] = {in_dims[0], in_dims[1], in_dims[2], in_dims[3]};
        const uint32_t nchw_dims[4] = {nhwc[0], nhwc[3], nhwc[1], nhwc[2]};
        std::vector<float> x_nchw(n_in);
        hexecutor_et::nhwc_to_nchw(x.data(), x_nchw.data(), nchw_dims);
        // The NCHW blob's input is NCHW-shaped at the PyTorch side.
        std::vector<uint32_t> pt_in = {nchw_dims[0], nchw_dims[1], nchw_dims[2], nchw_dims[3]};
        std::vector<float> out_b(n_out);
        // Layout 1 on the input: runtime shape is the NHWC shape again.
        auto blob_b = make_blob(image, constants, {{1, pt_in}}, {{0, out_dims}});
        if (run_backend(backend, blob_b, sizes_of(pt_in), x_nchw.data(), sizes_of(out_dims),
                        &out_b) != Error::Ok)
            return fail("NCHW execute");
        for (size_t i = 0; i < n_out; i++)
            if (out_a[i] != out_b[i])
                return fail("NCHW output differs from NHWC output");
        std::printf("nchw transpose: outputs match\n");
    } else {
        std::printf("nchw transpose: skipped (input rank %zu)\n", in_dims.size());
    }

    // 3. A wrongly shaped input must be rejected before any device work.
    std::vector<SizesType> flat = {SizesType(n_in)};
    std::vector<float> scratch(n_out);
    if (run_backend(backend, blob_a, flat, x.data(), sizes_of(out_dims), &scratch) !=
        Error::InvalidArgument)
        return fail("wrong input shape was not rejected with InvalidArgument");
    std::printf("shape check: rejected as expected\n");

    std::printf("output[0..3]:");
    for (size_t i = 0; i < 4 && i < n_out; i++)
        std::printf(" %.6f", out_a[i]);
    std::printf("\nhexecutor_et_selftest: PASS\n");
    return 0;
}
