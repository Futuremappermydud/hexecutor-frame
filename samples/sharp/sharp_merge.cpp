// SPDX-License-Identifier: LGPL-3.0-or-later
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t tile = 24, channels = 1024, tokens = 1 + tile * tile;

void merge(const std::filesystem::path &source, const std::filesystem::path &destination,
           size_t steps, size_t padding, size_t first, const std::string &suffix, bool image_encoder = false) {
    const size_t step = tile - 2 * padding;
    const size_t side = step * (steps - 1) + tile;
    std::vector<float> features(side * side * channels), input(tokens * channels);
    for (size_t row = 0; row < steps; ++row)
        for (size_t col = 0; col < steps; ++col) {
            const std::string name = image_encoder ? "ie" : "pe-" + std::to_string(first + row * steps + col);
            const auto path = source / (name + suffix + ".f32");
            const size_t bytes = input.size() * sizeof(float);
            if (std::filesystem::file_size(path) != bytes)
                throw std::runtime_error("invalid SHARP token tensor: " + path.string());
            std::ifstream in(path, std::ios::binary);
            if (!in.read(reinterpret_cast<char *>(input.data()), bytes))
                throw std::runtime_error("cannot read " + path.string());
            // Token zero is the class token. Only the outer edges retain the overlap.
            for (size_t y = row ? padding : 0; y < tile - (row + 1 < steps ? padding : 0); ++y)
                for (size_t x = col ? padding : 0; x < tile - (col + 1 < steps ? padding : 0); ++x)
                    for (size_t c = 0; c < channels; ++c) {
                        const size_t pixel = (row * step + y) * side + col * step + x;
                        features[c * side * side + pixel] = input[(1 + y * tile + x) * channels + c];
                    }
        }
    std::ofstream out(destination, std::ios::binary);
    if (!out.write(reinterpret_cast<const char *>(features.data()), features.size() * sizeof(float)) || !out.flush())
        throw std::runtime_error("cannot write " + destination.string());
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: sharp_merge TOKEN_DIRECTORY OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        const std::filesystem::path source = argv[1], output = argv[2];
        std::filesystem::create_directories(output);
        merge(source, output / "enc-0.f32", 5, 3, 0, "-latent0");
        merge(source, output / "enc-1.f32", 5, 3, 0, "-latent1");
        merge(source, output / "enc-2.f32", 5, 3, 0, "");
        merge(source, output / "enc-3.f32", 3, 6, 25, "");
        merge(source, output / "enc-4.f32", 1, 0, 34, "");
        merge(source, output / "enc-5.f32", 1, 0, 0, "", true);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
