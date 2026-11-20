// SPDX-License-Identifier: LGPL-3.0-or-later
#include "depth_model/deployment.hpp"
#include "hexecutor/executor.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool condition) {
    if (!condition)
        throw std::runtime_error("host assertion failed");
}
template <class F> void rejects(F f) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    check(failed);
}
template <class T> void put(hexecutor::Bytes &b, size_t at, T value) {
    std::memcpy(b.data() + at, &value, sizeof(value));
}
} // namespace
int main() {
    using namespace hexecutor;
    check(TensorSpec{"features", ScalarType::f32, {3, 4}}.bytes() == 48);
    rejects([] { TensorSpec{"x", ScalarType::f32, {0}}.bytes(); });
    rejects(
        [] { TensorSpec{"x", ScalarType::f32, {std::numeric_limits<size_t>::max(), 2}}.bytes(); });
    check(sha256(Bytes{'a', 'b', 'c'}) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    Program p;
    p.inputs = {{"a", ScalarType::u8, {2}},
                {"b", ScalarType::i16, {1}},
                {"c", ScalarType::f32, {3}},
                {"d", ScalarType::i32, {4}}};
    p.outputs = {{"first", ScalarType::f32, {5}}, {"second", ScalarType::u8, {6}}};
    p.validate();
    auto bad = p;
    bad.inputs.pop_back();
    rejects([&] { bad.validate(); });
    bad = p;
    bad.inputs[1].name = "a";
    rejects([&] { bad.validate(); });
    auto tensor = p;
    tensor.backend = Backend::tensor_f32;
    tensor.inputs = {{"rgb", ScalarType::f32, {1, 192, 192, 3}}};
    tensor.outputs = {{"joints", ScalarType::f32, {21, 3}},
                      {"presence", ScalarType::f32, {1}},
                      {"handedness", ScalarType::f32, {1}},
                      {"world", ScalarType::f32, {21, 3}}};
    tensor.validate();
    auto wrong_type = tensor;
    wrong_type.inputs[0].type = ScalarType::u8;
    rejects([&] { wrong_type.validate(); });
    tensor.inputs.clear();
    rejects([&] { tensor.validate(); });
    Bytes elf(86);
    std::memcpy(elf.data(), "\177ELF\1\1\1", 7);
    put<uint16_t>(elf, 18, 164);
    put<uint32_t>(elf, 24, 0x87606000);
    put<uint32_t>(elf, 28, 52);
    put<uint16_t>(elf, 42, 32);
    put<uint16_t>(elf, 44, 1);
    put<uint32_t>(elf, 52, 1);
    put<uint32_t>(elf, 56, 84);
    put<uint32_t>(elf, 60, 0x87606000);
    put<uint32_t>(elf, 68, 2);
    put<uint32_t>(elf, 72, 4);
    elf[84] = 7;
    elf[85] = 9;
    auto code = load_elf(elf, 0x87606000, 4096);
    check(code.size() == 4096 && code[0] == 7 && code[1] == 9 && code[2] == 0);
    auto corrupt = elf;
    put<uint32_t>(corrupt, 72, 8192);
    rejects([&] { load_elf(corrupt, 0x87606000, 4096); });
    corrupt = elf;
    put<uint32_t>(corrupt, 56, UINT32_MAX);
    rejects([&] { load_elf(corrupt, 0x87606000, 4096); });
    corrupt = elf;
    put<uint32_t>(corrupt, 28, UINT32_MAX);
    rejects([&] { load_elf(corrupt, 0x87606000, 4096); });
    rejects([] { depth_model::load_deployment("/missing", "hybrid", "primary"); });
    rejects([] { depth_model::load_deployment("/missing", "prepared-qnn", "alternate"); });
    std::cout
        << "Native tensor contract, SHA256, ELF validation, and model capability checks passed\n";
}
