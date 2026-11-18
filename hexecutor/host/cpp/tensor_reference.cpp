#include "hexecutor/executor.hpp"
#include "tensor/graph.h"
#include <stdexcept>
// Deliberate CPU validation entry point; Session never falls back to this.
std::vector<hexecutor::Bytes> tensor_reference(const hexecutor::Program &p,
                                               const std::vector<hexecutor::Buffer> &inputs) {
    auto image = hexecutor::read_file(p.image), constants = hexecutor::read_file(p.constants);
    HxGraph g{};
    uint32_t bad;
    if (hx_graph_open(&g, image.data(), image.size(), constants.data(), constants.size(), nullptr,
                      0x1000000u, &bad))
        throw std::runtime_error("reference preflight failed");
    const auto *h = g.header;
    hexecutor::Bytes arena(h->arena_bytes);
    g.arena = arena.data();
    if (inputs.size() != h->inputs)
        throw std::runtime_error("reference input count mismatch");
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto &t = g.tensors[g.io[i]];
        if (inputs[i].bytes != size_t(hx_tensor_elements(&t)) * 4 || !inputs[i].data)
            throw std::runtime_error("reference input size mismatch");
        hx_pack(reinterpret_cast<float *>(arena.data() + t.offset),
                static_cast<const float *>(inputs[i].data), &t);
    }
    for (uint32_t i = 0; i < h->ops; i++)
        hx_execute_op(&g, i, 0, 1);
    std::vector<hexecutor::Bytes> out;
    for (uint32_t i = 0; i < h->outputs; i++) {
        const auto &t = g.tensors[g.io[h->inputs + i]];
        out.emplace_back(size_t(hx_tensor_elements(&t)) * 4);
        hx_unpack(reinterpret_cast<float *>(out.back().data()),
                  reinterpret_cast<const float *>(arena.data() + t.offset), &t);
    }
    return out;
}
