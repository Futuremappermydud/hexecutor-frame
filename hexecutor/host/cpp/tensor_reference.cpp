// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/executor.hpp"
#include "tensor/graph.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
// Deliberate CPU validation entry point; Session never falls back to this.
namespace hexecutor {
struct TensorReference::Impl {
    Bytes image, constants, arena, state;
    HxGraph g{};

    explicit Impl(const Program &p) : image(read_file(p.image)), constants(read_file(p.constants)) {
        // The state region must be bound before hx_graph_open: persistent tensors are checked against
        // it, not against the arena. Its size comes from the image's own tensor table.
        const uint64_t state_bytes = persistent_state_extent(image);  // also checks the header length
        HxHeader h{};
        std::memcpy(&h, image.data(), sizeof h);
        state.assign(size_t((state_bytes + 127) & ~uint64_t(127)), 0);
        // The graph owns no memory: bind caller buffers of exactly the sizes the header declares.
        arena.assign(std::max<uint32_t>(h.arena_bytes, 128), 0);
        if (!state.empty())
            hx_graph_set_state(&g, state.data(), uint32_t(state.size()));
        uint32_t bad;
        if (hx_graph_open(&g, image.data(), uint32_t(image.size()), constants.data(),
                          uint32_t(constants.size()), arena.data(), uint32_t(arena.size()), &bad))
            throw std::runtime_error("reference preflight failed at operation " + std::to_string(bad));
    }
};

TensorReference::TensorReference(const Program &p) : impl_(std::make_unique<Impl>(p)) {}
TensorReference::~TensorReference() = default;
void TensorReference::reset_state() { std::fill(impl_->state.begin(), impl_->state.end(), 0); }
size_t TensorReference::state_bytes() const { return impl_->state.size(); }

std::vector<Bytes> TensorReference::run(const std::vector<Buffer> &inputs) {
    HxGraph &g = impl_->g;
    const auto *h = g.header;
    if (inputs.size() != h->inputs)
        throw std::runtime_error("reference input count mismatch");
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto &t = g.tensors[g.io[i]];
        if (inputs[i].bytes != size_t(hx_tensor_elements(&t)) * 4 || !inputs[i].data)
            throw std::runtime_error("reference input size mismatch");
        hx_pack(reinterpret_cast<float *>(impl_->arena.data() + t.offset),
                static_cast<const float *>(inputs[i].data), &t);
    }
    for (uint32_t i = 0; i < h->ops; i++)
        hx_execute_op(&g, i, 0, 1);
    std::vector<Bytes> out;
    for (uint32_t i = 0; i < h->outputs; i++) {
        const auto &t = g.tensors[g.io[h->inputs + i]];
        out.emplace_back(size_t(hx_tensor_elements(&t)) * 4);
        hx_unpack(reinterpret_cast<float *>(out.back().data()),
                  reinterpret_cast<const float *>(impl_->arena.data() + t.offset), &t);
    }
    return out;
}
} // namespace hexecutor

// Stateless one-shot form, kept for callers that run a program once.
std::vector<hexecutor::Bytes> tensor_reference(const hexecutor::Program &p,
                                               const std::vector<hexecutor::Buffer> &inputs) {
    hexecutor::TensorReference reference(p);
    return reference.run(inputs);
}
