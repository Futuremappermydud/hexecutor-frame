// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/broker.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array<size_t, 5> sides{768, 384, 192, 96, 48};
constexpr std::array<size_t, 5> skip_channels{256, 256, 512, 1024, 1024};

struct Crop {
    size_t y, x, height, width;
};

struct Image {
    size_t side, channels;
    std::ifstream stream;

    Image(const std::filesystem::path &path, size_t size, size_t c)
        : side(size), channels(c), stream(path, std::ios::binary) {
        if (!stream || std::filesystem::file_size(path) != side * side * channels * sizeof(float))
            throw std::runtime_error("invalid decoder input: " + path.string());
    }

    void read(Crop rect, std::vector<float> &out) {
        out.resize(rect.height * rect.width * channels);
        const size_t row_values = rect.width * channels;
        for (size_t row = 0; row < rect.height; ++row) {
            stream.seekg(((rect.y + row) * side + rect.x) * channels * sizeof(float));
            if (!stream.read(reinterpret_cast<char *>(out.data() + row * row_values), row_values * sizeof(float)))
                throw std::runtime_error("reading decoder crop failed");
        }
    }
};

std::string region_name(int row, int col) {
    return "r" + std::to_string(row) + "c" + std::to_string(col);
}

std::filesystem::path stage_path(const std::filesystem::path &out, int level, bool gaussian) {
    if (level == -1)
        return out / "disparity.f32";
    const std::string prefix = gaussian ? "gaussian-stage-" : "disparity-stage-";
    return out / (prefix + std::to_string(level) + ".f32");
}

std::filesystem::path mark(const std::filesystem::path &stage, int row, int col) {
    return stage.string() + "." + region_name(row, col) + ".ok";
}

std::vector<size_t> axis_tiles(int region, size_t side, size_t tile) {
    if (region == 0)
        return {0};
    if (region == 2)
        return {side - tile};
    std::vector<size_t> origins;
    for (size_t at = tile; at < side - tile; at += tile)
        origins.push_back(at);
    return origins;
}

void require_complete(const std::filesystem::path &path) {
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            if (!std::filesystem::exists(mark(path, row, col)))
                throw std::runtime_error("decoder input is incomplete: " + path.string());
}

void run_tiles(const std::filesystem::path &socket, const std::string &graph, const std::filesystem::path &target,
               std::vector<Image> &source, size_t channels_out, size_t side, int row, int col,
               size_t tile, size_t scale, size_t halo) {
    if (side % tile || side < 3 * tile)
        throw std::runtime_error("decoder tile grid must contain whole edge and interior tiles");
    const size_t tile_out = tile * scale, side_out = side * scale;
    const size_t row_bytes = tile_out * channels_out * sizeof(float);
    const auto marker = mark(target, row, col);
    if (std::filesystem::exists(marker))
        throw std::runtime_error("decoder region already completed: " + marker.string());
    const size_t target_bytes = side_out * side_out * channels_out * sizeof(float);
    if (!std::filesystem::exists(target)) {
        std::ofstream new_output(target, std::ios::binary);
        new_output.seekp(target_bytes - 1);
        if (!new_output.put(0) || !new_output.flush())
            throw std::runtime_error("allocating decoder output failed");
    } else if (std::filesystem::file_size(target) != target_bytes) {
        throw std::runtime_error("decoder output has an unexpected size");
    }
    std::fstream out(target, std::ios::in | std::ios::out | std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot open decoder output");

    hexecutor::broker::Client client(socket);
    hexecutor::broker::StreamSpec spec;
    spec.graph = graph + "-" + region_name(row, col);
    spec.deadline_ms = 30000;
    const auto stream = client.open(spec);
    if (stream.input_bytes.size() != source.size() || stream.output_bytes != std::vector<size_t>{tile_out * row_bytes})
        throw std::runtime_error("decoder stream shape mismatch");
    auto ys = axis_tiles(row, side, tile), xs = axis_tiles(col, side, tile);
    std::vector<std::vector<float>> buffers(source.size());
    std::vector<hexecutor::Buffer> arguments(source.size());
    double dsp_ms = 0;
    for (size_t y : ys)
        for (size_t x : xs) {
            const size_t y0 = y < halo ? 0 : y - halo, x0 = x < halo ? 0 : x - halo;
            const Crop crop{y0, x0, std::min(side, y + tile + halo) - y0,
                            std::min(side, x + tile + halo) - x0};
            for (size_t i = 0; i < source.size(); ++i) {
                const size_t factor = source[i].side / side;
                const Crop input_crop{crop.y * factor, crop.x * factor, crop.height * factor, crop.width * factor};
                source[i].read(input_crop, buffers[i]);
                const size_t bytes = buffers[i].size() * sizeof(float);
                if (stream.input_bytes[i] != bytes)
                    throw std::runtime_error("decoder input crop shape mismatch");
                arguments[i] = {buffers[i].data(), bytes};
            }
            const auto result = client.wait(client.submit(stream, arguments));
            if (result.status != hexecutor::broker::Status::ok || result.outputs.size() != 1 ||
                result.outputs.front().size() != stream.output_bytes.front())
                throw std::runtime_error(spec.graph + ": " + result.error);
            dsp_ms += result.dsp_ms;
            for (size_t r = 0; r < tile_out; ++r) {
                const size_t pixel = (y * scale + r) * side_out + x * scale;
                out.seekp(pixel * channels_out * sizeof(float));
                if (!out.write(reinterpret_cast<const char *>(result.outputs.front().data() + r * row_bytes), row_bytes))
                    throw std::runtime_error("writing decoder region failed");
            }
        }
    client.close(stream);
    if (!out.flush())
        throw std::runtime_error("flushing decoder output failed");
    std::ofstream done(marker);
    if (!(done << ys.size() * xs.size() << '\n') || !done.flush())
        throw std::runtime_error("writing decoder region marker failed");
    std::cout << spec.graph << " tiles " << ys.size() * xs.size() << " DSP " << dsp_ms << " ms\n";
}
void decode(const std::filesystem::path &socket, const std::filesystem::path &features,
            const std::filesystem::path &work, int level, int row, int col, size_t tile, bool gaussian) {
    const bool head = level == -1;
    const size_t channels = gaussian ? 128 : 256;
    const size_t side = sides[head ? 0 : level];
    const size_t halo = head ? 2 : level == 4 ? 3 : 5;
    std::vector<Image> source;
    if (level < 4) {
        const auto prior = stage_path(work, level + 1, gaussian);
        require_complete(prior);
        source.emplace_back(prior, side, channels);
    }
    if (!head)
        source.emplace_back(features / ("rest_a-" + std::to_string(level) + ".f32"), side, skip_channels[level]);
    const std::string graph = std::string(gaussian ? "sharp-gaussian-" : "sharp-disparity-") + std::to_string(level);
    run_tiles(socket, graph, stage_path(work, level, gaussian), source, head ? 2 : channels,
              side, row, col, tile, level == 0 ? 1 : 2, halo);
}

void gaussian_head(const std::filesystem::path &socket, const std::filesystem::path &features,
                   const std::filesystem::path &work, const std::string &phase, int row, int col, size_t tile) {
    constexpr size_t side = 768;
    const auto target = work / (phase + ".f32");
    const std::string graph = "sharp-gaussian-head-" + phase;
    std::vector<Image> source;
    if (phase == "fusion") {
        const auto coarse = stage_path(work, 0, true);
        require_complete(coarse);
        source.emplace_back(coarse, side, 128);
        source.emplace_back(features / "image-depth.f32", 2 * side, 5);
        run_tiles(socket, graph, target, source, 128, side, row, col, tile, 1, 4);
        return;
    }
    if (phase == "delta") {
        for (const char *branch : {"geometry", "texture"}) {
            const auto path = work / (std::string(branch) + "-1-1.f32");
            require_complete(path);
            source.emplace_back(path, side, 128);
        }
        run_tiles(socket, graph, target, source, 28, side, row, col, tile, 1, 0);
        return;
    }
    const auto split = phase.find('-');
    const std::string branch = phase.substr(0, split);
    if ((branch != "geometry" && branch != "texture") || phase.size() != split + 4 ||
        (phase[split + 1] != '0' && phase[split + 1] != '1') || phase[split + 2] != '-' ||
        (phase[split + 3] != '0' && phase[split + 3] != '1'))
        throw std::runtime_error("invalid Gaussian phase");
    const bool second_block = phase[split + 1] == '1', residual = phase[split + 3] == '1';
    source.emplace_back(work / (phase + ".normalized.f32"), side, residual ? 64 : 128);
    if (residual) {
        const auto skip = work / (second_block ? branch + "-0-1.f32" : "fusion.f32");
        require_complete(skip);
        source.emplace_back(skip, side, 128);
    }
    run_tiles(socket, graph, target, source, residual ? 128 : 64, side, row, col, tile, 1, 1);
}
} // namespace

int main(int argc, char **argv) {
    const bool head = argc == 8 && std::string(argv[7]) == "--head";
    const bool gaussian = argc == 8 && std::string(argv[7]) == "--gaussian";
    if (argc != 7 && !gaussian && !head) {
        std::cerr << "usage: sharp_disparity BROKER_SOCKET FUSION_DIRECTORY WORK_DIRECTORY LEVEL_OR_PHASE ROWCOL TILE [--gaussian|--head]\n";
        return 2;
    }
    try {
        size_t end = 0;
        const std::string region(argv[5]);
        if (region.size() != 2 || region.find_first_not_of("012") != std::string::npos)
            throw std::runtime_error("decoder region must be two 0..2 coordinates");
        const int tile = std::stoi(argv[6], &end);
        if (end != std::string(argv[6]).size() || (tile != 8 && tile != 16 && tile != 32 && !(head && tile == 64)))
            throw std::runtime_error("invalid decoder tile size");
        std::filesystem::create_directories(argv[3]);
        if (head) {
            gaussian_head(argv[1], argv[2], argv[3], argv[4], region[0] - '0', region[1] - '0', tile);
            return 0;
        }
        const int level = std::stoi(argv[4], &end);
        if (end != std::string(argv[4]).size() || level < -1 || level > 4)
            throw std::runtime_error("decoder level must be -1..4");
        if (gaussian && level == -1)
            throw std::runtime_error("the Gaussian feature decoder supports levels 4..0");
        decode(argv[1], argv[2], argv[3], level, region[0] - '0', region[1] - '0', tile, gaussian);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
