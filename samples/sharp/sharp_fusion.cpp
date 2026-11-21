// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/broker.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t tile = 8, input_channels = 1024;

struct Stage {
    size_t side;
    size_t upscale;
    size_t channels;
    size_t input_count;
};

constexpr std::array<Stage, 5> stages{{
    {96, 8, 256, 1}, {96, 4, 256, 1}, {96, 2, 512, 1}, {48, 2, 1024, 1}, {24, 2, 1024, 2},
}};

std::vector<float> read_encoder(const std::filesystem::path &path, size_t side) {
    const size_t count = side * side * input_channels;
    if (std::filesystem::file_size(path) != count * sizeof(float))
        throw std::runtime_error("unexpected encoder tensor size: " + path.string());
    std::vector<float> values(count);
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(values.data()), count * sizeof(float)))
        throw std::runtime_error("cannot read " + path.string());
    return values;
}

void extract(std::vector<float> &tile_data, const std::vector<float> &source, size_t y, size_t x, size_t side) {
    for (size_t c = 0; c < input_channels; ++c)
        for (size_t i = 0; i < tile; ++i)
            for (size_t j = 0; j < tile; ++j)
                tile_data[(i * tile + j) * input_channels + c] = source[(c * side + y + i) * side + x + j];
}

void render_stage(hexecutor::broker::Client &client, size_t index, const std::filesystem::path &capture,
                  const std::filesystem::path &output, bool first_tile) {
    const Stage &stage = stages[index];
    const std::string name = "sharp-fusion-" + std::to_string(index);
    hexecutor::broker::StreamSpec spec;
    spec.graph = name;
    spec.deadline_ms = 30000;
    const auto stream = client.open(spec);
    const size_t input_values = tile * tile * input_channels;
    const size_t tile_side = tile * stage.upscale;
    const size_t row_bytes = tile_side * stage.channels * sizeof(float);
    if (stream.input_bytes != std::vector<size_t>(stage.input_count, input_values * sizeof(float)) ||
        stream.output_bytes != std::vector<size_t>{tile_side * row_bytes})
        throw std::runtime_error("SHARP fusion stream shape mismatch: " + name);
    std::vector<std::vector<float>> inputs;
    for (size_t i = 0; i < stage.input_count; ++i)
        inputs.push_back(read_encoder(capture / ("enc-" + std::to_string(index + i) + ".f32"), stage.side));
    std::vector<std::vector<float>> tile_data(stage.input_count, std::vector<float>(input_values));
    std::vector<hexecutor::Buffer> arguments;
    for (auto &v : tile_data)
        arguments.push_back({v.data(), v.size() * sizeof(float)});

    std::ofstream out(output / ("rest_a-" + std::to_string(index) + ".f32"), std::ios::binary);
    const size_t limit = first_tile ? tile : stage.side;
    const size_t full_side = limit * stage.upscale;
    double dsp_ms = 0;
    for (size_t y = 0; y < limit; y += tile)
        for (size_t x = 0; x < limit; x += tile) {
            for (size_t i = 0; i < stage.input_count; ++i)
                extract(tile_data[i], inputs[i], y, x, stage.side);
            const auto result = client.wait(client.submit(stream, arguments));
            if (result.status != hexecutor::broker::Status::ok || result.outputs.size() != 1 ||
                result.outputs.front().size() != tile_side * row_bytes)
                throw std::runtime_error("SHARP fusion failed: " + result.error);
            dsp_ms += result.dsp_ms;
            for (size_t row = 0; row < tile_side; ++row) {
                const size_t pixel = (y * stage.upscale + row) * full_side + x * stage.upscale;
                out.seekp(pixel * stage.channels * sizeof(float));
                if (!out.write(reinterpret_cast<const char *>(result.outputs.front().data() + row * row_bytes), row_bytes))
                    throw std::runtime_error("cannot write fusion output");
            }
        }
    if (!out.flush())
        throw std::runtime_error("cannot flush fusion output");
    client.close(stream);
    std::cout << name << " DSP " << dsp_ms << " ms\n";
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: sharp_fusion BROKER_SOCKET ENCODER_TENSOR_DIRECTORY OUTPUT_DIRECTORY "
                     "[--first-tile] [--branch=0..4]\n";
        return 2;
    }
    try {
        bool first_tile = false;
        size_t start = 0, end = stages.size();
        for (int i = 4; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--first-tile") {
                first_tile = true;
            } else if (arg.size() == 10 && arg.compare(0, 9, "--branch=") == 0 &&
                       arg.back() >= '0' && static_cast<size_t>(arg.back() - '0') < stages.size()) {
                start = static_cast<size_t>(arg.back() - '0');
                end = start + 1;
            } else {
                throw std::runtime_error("invalid fusion option: " + arg);
            }
        }
        std::filesystem::create_directories(argv[3]);
        hexecutor::broker::Client client(argv[1]);
        for (size_t i = start; i < end; ++i)
            render_stage(client, i, argv[2], argv[3], first_tile);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
