// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/broker.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t token_bytes = 577u * 1024u * sizeof(float), blocks = 24, stages = blocks + 1;

size_t number(const std::string &value) {
    size_t consumed = 0;
    const size_t parsed = std::stoul(value, &consumed);
    if (consumed != value.size())
        throw std::runtime_error("invalid encoder block range");
    return parsed;
}

void write_tensor(const std::filesystem::path &path, const hexecutor::Bytes &data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(data.data()), data.size()) || !file.flush())
        throw std::runtime_error("cannot write " + path.string());
}

void run(const std::filesystem::path &socket, const std::string &encoder, size_t first,
         size_t count, const std::filesystem::path &source, const std::filesystem::path &destination) {
    if ((encoder != "pe" && encoder != "ie") || first >= stages || count == 0 || count > stages - first)
        throw std::runtime_error("invalid encoder or block range");

    const size_t patches = encoder == "pe" ? 35 : 1;
    std::vector<std::string> names;
    std::vector<hexecutor::Bytes> tokens;
    for (size_t i = 0; i < patches; ++i) {
        names.push_back(encoder == "pe" ? "pe-" + std::to_string(i) : "ie");
        auto tensor = hexecutor::read_file(source / (names.back() + ".f32"));
        if (tensor.size() != token_bytes)
            throw std::runtime_error("invalid input tensor for " + names.back());
        tokens.push_back(std::move(tensor));
    }

    std::filesystem::create_directories(destination);
    hexecutor::broker::Client client(socket);
    for (size_t block = first; block < first + count; ++block) {
        hexecutor::broker::StreamSpec spec;
        const std::string stem = "sharp-" + encoder;
        spec.graph = block == blocks ? stem + "-norm" : stem + (block < 10 ? "-block-0" : "-block-") +
                     std::to_string(block);
        spec.deadline_ms = 30000;
        const auto stream = client.open(spec);
        const std::vector<size_t> expected{token_bytes};
        if (stream.input_bytes != expected || stream.output_bytes != expected)
            throw std::runtime_error("encoder block tensor size mismatch: " + spec.graph);
        double dsp_ms = 0;
        for (size_t i = 0; i < patches; ++i) {
            auto result = client.wait(client.submit(stream, {{tokens[i].data(), tokens[i].size()}}));
            if (result.status != hexecutor::broker::Status::ok || result.outputs.size() != 1 ||
                result.outputs.front().size() != token_bytes)
                throw std::runtime_error(spec.graph + ": " + result.error);
            dsp_ms += result.dsp_ms;
            tokens[i] = std::move(result.outputs.front());
            if (encoder == "pe" && i < 25 && (block == 5 || block == 11))
                write_tensor(destination / (names[i] + (block == 5 ? "-latent0.f32" : "-latent1.f32")), tokens[i]);
        }
        client.close(stream);
        std::cout << spec.graph << " patches " << patches << " DSP " << dsp_ms << " ms\n";
    }
    for (size_t i = 0; i < patches; ++i)
        write_tensor(destination / (names[i] + ".f32"), tokens[i]);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: sharp_blocks BROKER_SOCKET pe|ie FIRST COUNT INPUT_DIRECTORY OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        run(argv[1], argv[2], number(argv[3]), number(argv[4]), argv[5], argv[6]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
