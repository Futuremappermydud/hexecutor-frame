// SPDX-License-Identifier: LGPL-3.0-or-later
// Normalize HWC activation files globally between spatially tiled DSP phases.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void read_values(std::istream &input, std::vector<float> &values) {
    if (!input.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float)))
        throw std::runtime_error("could not read activation or affine parameters");
}

void groupnorm(const std::filesystem::path &source, const std::filesystem::path &affine,
               const std::filesystem::path &target, size_t side, size_t channels, size_t groups) {
    if (side == 0 || side > 1536 || channels == 0 || channels > 256 ||
        groups == 0 || channels % groups)
        throw std::runtime_error("invalid GroupNorm shape");
    if (std::filesystem::file_size(source) != side * side * channels * sizeof(float) ||
        std::filesystem::file_size(affine) != 2 * channels * sizeof(float))
        throw std::runtime_error("GroupNorm file has an unexpected size");
    if (std::filesystem::exists(target))
        throw std::runtime_error("GroupNorm output already exists");

    std::ifstream in(source, std::ios::binary), parameters(affine, std::ios::binary);
    std::vector<float> gain(channels), bias(channels), row(side * channels);
    std::vector<double> sum(groups), sum_sq(groups), mean(groups), inv_std(groups);
    read_values(parameters, gain);
    read_values(parameters, bias);
    const size_t group_width = channels / groups;
    for (size_t y = 0; y < side; ++y) {
        read_values(in, row);
        for (size_t x = 0; x < side; ++x)
            for (size_t g = 0; g < groups; ++g)
                for (size_t i = 0; i < group_width; ++i) {
                    const double value = row[x * channels + g * group_width + i];
                    sum[g] += value;
                    sum_sq[g] += value * value;
                }
    }
    const double count = side * side * group_width;
    for (size_t g = 0; g < groups; ++g) {
        mean[g] = sum[g] / count;
        const double variance = std::max(0.0, sum_sq[g] / count - mean[g] * mean[g]);
        inv_std[g] = 1.0 / std::sqrt(variance + 1e-5);
        if (!std::isfinite(mean[g]) || !std::isfinite(inv_std[g]))
            throw std::runtime_error("GroupNorm input is not finite");
    }

    const auto temporary = target.string() + ".partial";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    in.seekg(0);
    for (size_t y = 0; y < side; ++y) {
        read_values(in, row);
        for (size_t x = 0; x < side; ++x)
            for (size_t g = 0; g < groups; ++g)
                for (size_t i = 0; i < group_width; ++i) {
                    const size_t c = g * group_width + i;
                    float &value = row[x * channels + c];
                    value = static_cast<float>((value - mean[g]) * inv_std[g] * gain[c] + bias[c]);
                }
        if (!out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float)))
            throw std::runtime_error("could not write GroupNorm activation");
    }
    out.close();
    if (!out)
        throw std::runtime_error("could not close GroupNorm activation");
    std::filesystem::rename(temporary, target);
}

size_t dimension(const char *arg) {
    const std::string input(arg);
    size_t end;
    const unsigned long value = std::stoul(input, &end);
    if (end != input.size())
        throw std::runtime_error("invalid GroupNorm dimension");
    return value;
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: sharp_groupnorm INPUT_HWC AFFINE OUTPUT_HWC SIDE CHANNELS GROUPS\n";
        return 2;
    }
    try {
        groupnorm(argv[1], argv[2], argv[3], dimension(argv[4]), dimension(argv[5]), dimension(argv[6]));
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
