// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// Parser for the HexecutorBackend delegate blob (format v1, little-endian).
// Dependency-free so it can be tested without ExecuTorch.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hexecutor_et {

constexpr uint32_t kBlobVersion = 1;
constexpr uint32_t kBackendTensorF32 = 1; // HXFG image on the tensor_f32 backend.
constexpr uint32_t kLayoutAsIs = 0;
constexpr uint32_t kLayoutNchw = 1; // PyTorch NCHW <-> runtime NHWC, rank 4 only.
constexpr uint32_t kMaxIo = 256;
constexpr uint32_t kMaxRank = 8;

struct TensorDesc {
    uint32_t layout = kLayoutAsIs;
    std::vector<uint32_t> dims; // PyTorch-side dimensions.
    uint64_t elements = 0;      // Product of dims (checked).
    uint64_t bytes = 0;         // elements * 4 (checked).
};

struct Blob {
    uint32_t backend = 0;
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;
    std::vector<uint8_t> image;
    std::vector<uint8_t> constants;
};

// Returns true on success. On failure, fills `error` with a short reason and
// leaves `out` unspecified. Never reads out of bounds, never aborts.
bool parse_blob(const uint8_t *data, size_t size, Blob *out, std::string *error);

// Layout helpers. NCHW dims are {N, C, H, W}; NHWC dims are {N, H, W, C}.
// src and dst must not alias and must hold N*C*H*W floats.
void nchw_to_nhwc(const float *src, float *dst, const uint32_t nchw_dims[4]);
void nhwc_to_nchw(const float *src, float *dst, const uint32_t nchw_dims[4]);

} // namespace hexecutor_et
