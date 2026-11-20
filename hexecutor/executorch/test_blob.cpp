// SPDX-License-Identifier: LGPL-3.0-or-later
// Parser and layout tests. Needs no ExecuTorch.
#include "blob.hpp"

#include <cstdio>

using namespace hexecutor_et;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                          \
        }                                                                        \
    } while (0)

namespace {

using Bytes = std::vector<uint8_t>;

void put32(Bytes &b, uint32_t v) {
    for (int i = 0; i < 4; i++)
        b.push_back(uint8_t(v >> (8 * i)));
}
void put64(Bytes &b, uint64_t v) {
    put32(b, uint32_t(v));
    put32(b, uint32_t(v >> 32));
}

struct Tensor {
    uint32_t layout;
    std::vector<uint32_t> dims;
};

// Builds a blob with explicit fields so tests need no byte-offset arithmetic.
Bytes build(uint32_t version, uint32_t backend, const std::vector<Tensor> &ins,
            const std::vector<Tensor> &outs, uint64_t image_size, uint64_t constants_size,
            size_t image_payload, size_t constants_payload) {
    Bytes b = {'H', 'X', 'E', 'T'};
    put32(b, version);
    put32(b, backend);
    put32(b, uint32_t(ins.size()));
    put32(b, uint32_t(outs.size()));
    for (const auto *list : {&ins, &outs})
        for (const auto &t : *list) {
            put32(b, t.layout);
            put32(b, uint32_t(t.dims.size()));
            for (uint32_t d : t.dims)
                put32(b, d);
        }
    put64(b, image_size);
    put64(b, constants_size);
    for (size_t i = 0; i < image_payload; i++)
        b.push_back(uint8_t(0xA0 + i));
    for (size_t i = 0; i < constants_payload; i++)
        b.push_back(uint8_t(i));
    return b;
}

Bytes valid_blob() {
    return build(1, 1, {{1, {1, 3, 4, 5}}}, {{0, {2, 7}}}, 6, 4, 6, 4);
}

bool parses(const Bytes &b, Blob *out = nullptr) {
    Blob tmp;
    std::string err;
    return parse_blob(b.data(), b.size(), out ? out : &tmp, &err);
}

} // namespace

int main() {
    // Valid blob parses and exposes its fields.
    {
        Blob out;
        CHECK(parses(valid_blob(), &out));
        CHECK(out.backend == kBackendTensorF32);
        CHECK(out.inputs.size() == 1 && out.outputs.size() == 1);
        CHECK(out.inputs[0].layout == kLayoutNchw);
        CHECK(out.inputs[0].dims == (std::vector<uint32_t>{1, 3, 4, 5}));
        CHECK(out.inputs[0].elements == 60 && out.inputs[0].bytes == 240);
        CHECK(out.outputs[0].bytes == 56);
        CHECK(out.image.size() == 6 && out.image[0] == 0xA0 && out.image[5] == 0xA5);
        CHECK(out.constants.size() == 4 && out.constants[3] == 3);
    }
    // Empty constants section is allowed.
    CHECK(parses(build(1, 1, {{0, {4}}}, {{0, {4}}}, 1, 0, 1, 0)));
    // Bad version and backend.
    CHECK(!parses(build(2, 1, {{0, {4}}}, {{0, {4}}}, 1, 0, 1, 0)));
    CHECK(!parses(build(1, 2, {{0, {4}}}, {{0, {4}}}, 1, 0, 1, 0)));
    CHECK(!parses(build(1, 0, {{0, {4}}}, {{0, {4}}}, 1, 0, 1, 0)));
    // Bad magic.
    {
        Bytes b = valid_blob();
        b[0] = 'X';
        CHECK(!parses(b));
    }
    // Every truncation is rejected.
    {
        Bytes b = valid_blob();
        for (size_t n = 0; n < b.size(); n++) {
            Bytes t(b.begin(), b.begin() + n);
            CHECK(!parses(t));
        }
    }
    // Trailing bytes are rejected.
    {
        Bytes b = valid_blob();
        b.push_back(0);
        CHECK(!parses(b));
    }
    // Unknown layout, NCHW with rank != 4, zero dimension, rank 0.
    CHECK(!parses(build(1, 1, {{7, {4}}}, {{0, {4}}}, 1, 0, 1, 0)));
    CHECK(!parses(build(1, 1, {{1, {1, 4, 4}}}, {{0, {4}}}, 1, 0, 1, 0)));
    CHECK(!parses(build(1, 1, {{0, {4, 0}}}, {{0, {4}}}, 1, 0, 1, 0)));
    CHECK(!parses(build(1, 1, {{0, {}}}, {{0, {4}}}, 1, 0, 1, 0)));
    // Element count overflow: eight dims of 2^32-1.
    CHECK(!parses(build(1, 1, {{0, std::vector<uint32_t>(8, 0xFFFFFFFFu)}}, {{0, {4}}}, 1, 0, 1,
                        0)));
    // Too many I/O tensors.
    {
        std::vector<Tensor> many(300, Tensor{0, {4}});
        CHECK(!parses(build(1, 1, many, {{0, {4}}}, 1, 0, 1, 0)));
    }
    // Section sizes: image larger than the rest of the blob, huge constants,
    // empty image, and sizes that would overflow when added.
    CHECK(!parses(build(1, 1, {{0, {4}}}, {{0, {4}}}, 100, 0, 1, 0)));
    CHECK(!parses(build(1, 1, {{0, {4}}}, {{0, {4}}}, 1, 0xFFFFFFFFFFFFFFFFull, 1, 0)));
    CHECK(!parses(build(1, 1, {{0, {4}}}, {{0, {4}}}, 0xFFFFFFFFFFFFFFFFull, 1, 1, 1)));
    CHECK(!parses(build(1, 1, {{0, {4}}}, {{0, {4}}}, 0, 0, 0, 0)));
    // Declared constants size larger than the payload.
    CHECK(!parses(build(1, 1, {{0, {4}}}, {{0, {4}}}, 1, 8, 1, 4)));

    // Transpose round trip and an explicit element check on [2,3,4,5].
    {
        const uint32_t d[4] = {2, 3, 4, 5};
        const size_t n = 2 * 3 * 4 * 5;
        float nchw[n], nhwc[n], back[n];
        for (size_t i = 0; i < n; i++)
            nchw[i] = float(i) * 0.5f - 7.0f;
        nchw_to_nhwc(nchw, nhwc, d);
        // NCHW (n=1,c=2,h=3,w=4) is at ((1*3+2)*4+3)*5+4; NHWC (n=1,h=3,w=4,c=2)
        // is at ((1*4+3)*5+4)*3+2.
        CHECK(nhwc[((1 * 4 + 3) * 5 + 4) * 3 + 2] == nchw[((1 * 3 + 2) * 4 + 3) * 5 + 4]);
        nhwc_to_nchw(nhwc, back, d);
        bool same = true;
        for (size_t i = 0; i < n; i++)
            same = same && back[i] == nchw[i];
        CHECK(same);
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("blob tests passed");
    return 0;
}
