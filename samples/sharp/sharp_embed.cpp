// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/broker.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t patch_size = 384, cell_size = 16, grid = 24, token_count = 577;
constexpr size_t token_channels = 1024;

template <class T> struct Image {
    size_t width, height;
    std::vector<T> pixels;
};

std::string header_word(std::istream &file) {
    std::string word;
    for (;;) {
        const int c = file.peek();
        if (c == '#') {
            std::getline(file, word);
            word.clear();
        } else if (c != EOF && std::isspace(c)) {
            file.get();
        } else {
            break;
        }
    }
    while (file.good() && !std::isspace(file.peek()) && file.peek() != '#') {
        const int c = file.get();
        if (c == EOF)
            break;
        word.push_back(static_cast<char>(c));
    }
    return word;
}

Image<uint8_t> read_rgb(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (header_word(file) != "P6")
        throw std::runtime_error("image must be a binary PPM (P6)");
    auto dimension = [&] {
        const std::string word = header_word(file);
        size_t end = 0;
        const auto value = std::stoul(word, &end);
        if (end != word.size() || !value || value > 16384)
            throw std::runtime_error("invalid PPM dimensions");
        return static_cast<size_t>(value);
    };
    const size_t width = dimension(), height = dimension();
    const auto max = header_word(file);
    const int delimiter = file.get();
    if (max != "255" || delimiter == EOF || !std::isspace(delimiter))
        throw std::runtime_error("image must be an 8-bit RGB PPM");
    if (delimiter == '\r' && file.peek() == '\n')
        file.get();
    Image<uint8_t> image{width, height, std::vector<uint8_t>(width * height * 3)};
    if (!file.read(reinterpret_cast<char *>(image.pixels.data()), image.pixels.size()))
        throw std::runtime_error("truncated PPM pixel data");
    return image;
}

Image<float> resize_and_normalize(const Image<uint8_t> &image) {
    constexpr size_t size = 1536;
    Image<float> result{size, size, std::vector<float>(size * size * 3)};
    const float x_scale = static_cast<float>(image.width - 1) / (size - 1);
    const float y_scale = static_cast<float>(image.height - 1) / (size - 1);
    for (size_t y = 0; y < size; ++y) {
        const float fy = y * y_scale;
        const size_t y0 = std::min(static_cast<size_t>(fy), image.height - 1), y1 = std::min(y0 + 1, image.height - 1);
        for (size_t x = 0; x < size; ++x) {
            const float fx = x * x_scale;
            const size_t x0 = std::min(static_cast<size_t>(fx), image.width - 1), x1 = std::min(x0 + 1, image.width - 1);
            const float rx = fx - x0, ry = fy - y0;
            for (size_t c = 0; c < 3; ++c) {
                const float a = image.pixels[(y0 * image.width + x0) * 3 + c] / 255.0f;
                const float b = image.pixels[(y0 * image.width + x1) * 3 + c] / 255.0f;
                const float d = image.pixels[(y1 * image.width + x0) * 3 + c] / 255.0f;
                const float e = image.pixels[(y1 * image.width + x1) * 3 + c] / 255.0f;
                const float top = a * (1.0f - rx) + b * rx, bottom = d * (1.0f - rx) + e * rx;
                result.pixels[(y * size + x) * 3 + c] = 2.0f * (top * (1.0f - ry) + bottom * ry) - 1.0f;
            }
        }
    }
    return result;
}

Image<float> pyramid_level(const Image<float> &image, size_t stride) {
    const size_t size = image.width / stride;
    Image<float> result{size, size, std::vector<float>(size * size * 3)};
    for (size_t y = 0; y < size; ++y)
        for (size_t x = 0; x < size; ++x) {
            // align_corners=False at even integer scale uses the two center samples.
            const size_t start = ((y * stride + stride / 2 - 1) * image.width + x * stride + stride / 2 - 1) * 3;
            for (size_t c = 0; c < 3; ++c) {
                const size_t i = start + c;
                result.pixels[(y * size + x) * 3 + c] = 0.25f *
                    ((image.pixels[i] + image.pixels[i + 3]) +
                     (image.pixels[i + 3 * image.width] + image.pixels[i + 3 * image.width + 3]));
            }
        }
    return result;
}

std::vector<float> pack_patch(const Image<float> &image, size_t row, size_t col) {
    constexpr size_t channels = 3 * cell_size * cell_size;
    std::vector<float> result(token_count * channels);
    for (size_t y = 0; y < grid; ++y)
        for (size_t x = 0; x < grid; ++x) {
            float *token = result.data() + (1 + y * grid + x) * channels;
            for (size_t c = 0; c < 3; ++c)
                for (size_t iy = 0; iy < cell_size; ++iy)
                    for (size_t ix = 0; ix < cell_size; ++ix) {
                        const size_t src = ((row + y * cell_size + iy) * image.width + col + x * cell_size + ix) * 3;
                        *token++ = image.pixels[src + c];
                    }
        }
    return result;
}

void write_raw(const std::filesystem::path &path, const void *data, size_t size) {
    std::ofstream result(path, std::ios::binary);
    if (!result.write(reinterpret_cast<const char *>(data), size))
        throw std::runtime_error("cannot write " + path.string());
}

void run(const std::filesystem::path &image_path, const std::filesystem::path &socket,
         const std::filesystem::path &output) {
    const auto high = resize_and_normalize(read_rgb(image_path));
    const auto middle = pyramid_level(high, 2), low = pyramid_level(high, 4);
    std::filesystem::create_directories(output);
    {
        std::vector<float> rgb(high.pixels.size());
        std::transform(high.pixels.begin(), high.pixels.end(), rgb.begin(),
                       [](float value) { return (value + 1.0f) * 0.5f; });
        write_raw(output / "image.f32", rgb.data(), rgb.size() * sizeof(float));
    }
    hexecutor::broker::Client client(socket);
    auto open = [&](const std::string &name) {
        hexecutor::broker::StreamSpec spec;
        spec.graph = name;
        spec.deadline_ms = 30000;
        return client.open(spec);
    };
    auto infer = [&](const hexecutor::broker::Client::Stream &stream, const std::vector<float> &input,
                     const std::string &name) {
        const size_t bytes = input.size() * sizeof(float);
        if (stream.input_bytes != std::vector<size_t>{bytes} ||
            stream.output_bytes != std::vector<size_t>{token_count * token_channels * sizeof(float)})
            throw std::runtime_error("SHARP embedding stream shape mismatch");
        const auto result = client.wait(client.submit(stream, {{input.data(), bytes}}));
        if (result.status != hexecutor::broker::Status::ok || result.outputs.size() != 1 ||
            result.outputs.front().size() != stream.output_bytes.front())
            throw std::runtime_error("SHARP embedding failed: " + result.error);
        const auto &data = result.outputs.front();
        write_raw(output / (name + ".f32"), data.data(), data.size());
        std::cout << name << " DSP " << result.dsp_ms << " ms\n";
    };
    const auto pe = open("sharp-pe-embed");
    size_t patch = 0;
    auto run_grid = [&](const Image<float> &level, size_t stride) {
        for (size_t y = 0; y + patch_size <= level.height; y += stride)
            for (size_t x = 0; x + patch_size <= level.width; x += stride)
                infer(pe, pack_patch(level, y, x), "pe-" + std::to_string(patch++));
    };
    run_grid(high, 288);
    run_grid(middle, 192);
    infer(pe, pack_patch(low, 0, 0), "pe-" + std::to_string(patch));
    client.close(pe);
    const auto ie = open("sharp-ie-embed");
    infer(ie, pack_patch(low, 0, 0), "ie");
    client.close(ie);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: sharp_embed IMAGE.ppm BROKER_SOCKET OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        run(argv[1], argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
