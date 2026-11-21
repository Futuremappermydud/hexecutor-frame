// SPDX-License-Identifier: LGPL-3.0-or-later
// Writes the Gaussian splat interchange format from metric-space SHARP tensors.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read_float(const std::filesystem::path &path, size_t max_count) {
    const auto bytes = std::filesystem::file_size(path);
    if (bytes % sizeof(float) || bytes / sizeof(float) > max_count)
        throw std::runtime_error("invalid metric tensor size: " + path.string());
    std::vector<float> data(bytes / sizeof(float));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(data.data()), bytes))
        throw std::runtime_error("cannot read " + path.string());
    return data;
}

template <class T> void write_word(std::ostream &out, T value) {
    static_assert(sizeof(T) == sizeof(uint32_t), "PLY expects 32-bit fields");
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    std::array<unsigned char, 4> bytes{};
    for (size_t i = 0; i < bytes.size(); i++)
        bytes[i] = static_cast<unsigned char>(word >> (8u * i));
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

float color_coefficient(float linear) {
    constexpr float sh0 = 0.28209479177387814f;
    const float srgb = linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    return (srgb - 0.5f) / sh0;
}

float quantile(const std::vector<float> &sorted, float probability) {
    const float position = probability * static_cast<float>(sorted.size() - 1);
    const auto low = static_cast<size_t>(position);
    return sorted[low] + (position - low) * (sorted[std::min(low + 1, sorted.size() - 1)] - sorted[low]);
}

void export_ply(const std::filesystem::path &source, const std::filesystem::path &camera_path,
                const std::filesystem::path &target) {
    constexpr size_t max_count = 2u * 768u * 768u;
    uint32_t width = 0, height = 0;
    float focal = 0;
    std::ifstream camera(camera_path);
    if (!(camera >> width >> height >> focal) || !width || !height || !std::isfinite(focal) || focal <= 0)
        throw std::runtime_error("camera metadata must contain width, height and positive focal length in pixels");

    const auto opacity = read_float(source / "metric-4.f32", max_count);
    const size_t n = opacity.size();
    const auto position = read_float(source / "metric-0.f32", n * 3);
    const auto scales = read_float(source / "metric-1.f32", n * 3);
    const auto rotation = read_float(source / "metric-2.f32", n * 4);
    const auto color = read_float(source / "metric-3.f32", n * 3);
    if (!n || position.size() != n * 3 || scales.size() != n * 3 ||
        rotation.size() != n * 4 || color.size() != n * 3)
        throw std::runtime_error("metric Gaussian tensor lengths disagree");
    for (const auto *data : {&position, &scales, &rotation, &color, &opacity})
        if (std::any_of(data->begin(), data->end(), [](float v) { return !std::isfinite(v); }))
            throw std::runtime_error("non-finite metric Gaussian tensor");
    if (std::any_of(scales.begin(), scales.end(), [](float v) { return v <= 0; }) ||
        std::any_of(opacity.begin(), opacity.end(), [](float v) { return v <= 0 || v >= 1; }))
        throw std::runtime_error("invalid Gaussian scale or opacity");
    std::vector<float> disparity(n);
    for (size_t i = 0; i < n; i++) {
        disparity[i] = 1.0f / position[i * 3 + 2];
        if (!std::isfinite(disparity[i]))
            throw std::runtime_error("invalid Gaussian depth");
    }
    std::sort(disparity.begin(), disparity.end());

    std::ofstream out(target, std::ios::binary);
    out << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << '\n'
        << "property float x\nproperty float y\nproperty float z\n"
        << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\n"
        << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        << "element extrinsic 16\nproperty float extrinsic\nelement intrinsic 9\nproperty float intrinsic\n"
        << "element image_size 2\nproperty uint image_size\nelement frame 2\nproperty int frame\n"
        << "element disparity 2\nproperty float disparity\nelement color_space 1\nproperty uchar color_space\n"
        << "element version 3\nproperty uchar version\nend_header\n";
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < 3; j++)
            write_word(out, position[i * 3 + j]);
        for (size_t j = 0; j < 3; j++)
            write_word(out, color_coefficient(color[i * 3 + j]));
        write_word(out, std::log(opacity[i] / (1.0f - opacity[i])));
        for (size_t j = 0; j < 3; j++)
            write_word(out, std::log(scales[i * 3 + j]));
        for (size_t j = 0; j < 4; j++)
            write_word(out, rotation[i * 4 + j]);
    }
    for (size_t i = 0; i < 16; ++i)
        write_word(out, i % 5 == 0 ? 1.0f : 0.0f);
    for (float v : {focal, 0.0f, width * 0.5f, 0.0f, focal, height * 0.5f, 0.0f, 0.0f, 1.0f})
        write_word(out, v);
    for (uint32_t v : {width, height, 1u, static_cast<uint32_t>(n)})
        write_word(out, v);
    for (float p : {0.1f, 0.9f})
        write_word(out, quantile(disparity, p));
    constexpr char metadata[] = {0, 1, 5, 0}; // sRGB, version 1.5.0
    if (!out.write(metadata, sizeof(metadata)) || !out.flush())
        throw std::runtime_error("cannot write " + target.string());
    std::cout << n << " Gaussian vertices\n";
}
} // namespace

int main(int argc, char **argv) {
    static_assert(std::numeric_limits<float>::is_iec559, "PLY requires IEEE-754 floats");
    if (argc != 4) {
        std::cerr << "usage: sharp_ply METRIC_TENSOR_DIRECTORY CAMERA_TEXT OUTPUT_PLY\n";
        return 2;
    }
    try {
        export_ply(argv[1], argv[2], argv[3]);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
