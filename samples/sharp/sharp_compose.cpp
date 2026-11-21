// SPDX-License-Identifier: LGPL-3.0-or-later
// Compose and unproject the output tensors of the released SHARP checkpoint.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t image_side = 1536, splat_side = 768;
constexpr size_t image_pixels = image_side * image_side, splat_pixels = splat_side * splat_side;
using Matrix = std::array<std::array<double, 3>, 3>;
using Quaternion = std::array<double, 4>;

std::vector<float> read_tensor(const std::filesystem::path &path, size_t count) {
    const size_t bytes = count * sizeof(float);
    if (std::filesystem::file_size(path) != bytes)
        throw std::runtime_error("unexpected tensor size: " + path.string());
    std::vector<float> values(count);
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(values.data()), bytes) ||
        std::any_of(values.begin(), values.end(), [](float v) { return !std::isfinite(v); }))
        throw std::runtime_error("invalid tensor: " + path.string());
    return values;
}

Matrix rotation_matrix(const Quaternion &q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double norm = w * w + x * x + y * y + z * z;
    if (!(norm > 0))
        throw std::runtime_error("zero quaternion");
    const double f = 2 / norm;
    return {{{1 - f * (y * y + z * z), f * (x * y - z * w), f * (x * z + y * w)},
             {f * (x * y + z * w), 1 - f * (x * x + z * z), f * (y * z - x * w)},
             {f * (x * z - y * w), f * (y * z + x * w), 1 - f * (x * x + y * y)}}};
}

double determinant(const Matrix &m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
           m[0][1] * (m[1][0] * m[2][2] - m[2][0] * m[1][2]) +
           m[0][2] * (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
}

Quaternion quaternion(const Matrix &r) {
    const Quaternion scaled_square{{1 + r[0][0] + r[1][1] + r[2][2], 1 + r[0][0] - r[1][1] - r[2][2],
                                    1 - r[0][0] + r[1][1] - r[2][2], 1 - r[0][0] - r[1][1] + r[2][2]}};
    const size_t which = std::max_element(scaled_square.begin(), scaled_square.end()) - scaled_square.begin();
    const double v = 0.5 * std::sqrt(scaled_square[which]), f = 0.25 / v;
    switch (which) {
    case 0: return {v, f * (r[2][1] - r[1][2]), f * (r[0][2] - r[2][0]), f * (r[1][0] - r[0][1])};
    case 1: return {f * (r[2][1] - r[1][2]), v, f * (r[0][1] + r[1][0]), f * (r[0][2] + r[2][0])};
    case 2: return {f * (r[0][2] - r[2][0]), f * (r[0][1] + r[1][0]), v, f * (r[1][2] + r[2][1])};
    default: return {f * (r[1][0] - r[0][1]), f * (r[0][2] + r[2][0]), f * (r[1][2] + r[2][1]), v};
    }
}

void project_covariance(const Quaternion &q, const std::array<double, 3> &scale,
                        const std::array<double, 3> &camera, float *sizes, float *rotation) {
    const Matrix r = rotation_matrix(q);
    Matrix covariance{};
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                covariance[i][j] += camera[i] * r[i][k] * scale[k] * scale[k] * r[j][k] * camera[j];
    Matrix vectors{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const double threshold = 1e-14 * (covariance[0][0] + covariance[1][1] + covariance[2][2]);
    for (size_t sweep = 0; sweep < 12; ++sweep)
        for (size_t a = 0; a < 2; ++a)
            for (size_t b = a + 1; b < 3; ++b) {
                const double off = covariance[a][b];
                if (std::abs(off) <= threshold)
                    continue;
                const double tau = (covariance[b][b] - covariance[a][a]) / (2 * off);
                const double t = std::copysign(1.0 / (std::abs(tau) + std::hypot(1.0, tau)), tau);
                const double c = 1 / std::sqrt(1 + t * t), s = t * c;
                covariance[a][a] -= t * off;
                covariance[b][b] += t * off;
                covariance[a][b] = covariance[b][a] = 0;
                for (size_t k = 0; k < 3; ++k) {
                    if (k != a && k != b) {
                        const double ka = covariance[k][a], kb = covariance[k][b];
                        covariance[k][a] = covariance[a][k] = c * ka - s * kb;
                        covariance[k][b] = covariance[b][k] = s * ka + c * kb;
                    }
                    const double ka = vectors[k][a], kb = vectors[k][b];
                    vectors[k][a] = c * ka - s * kb;
                    vectors[k][b] = s * ka + c * kb;
                }
            }
    std::array<size_t, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return covariance[a][a] > covariance[b][b]; });
    Matrix basis;
    for (size_t i = 0; i < 3; ++i) {
        sizes[i] = static_cast<float>(std::sqrt(std::max(covariance[order[i]][order[i]], 0.0)));
        for (size_t j = 0; j < 3; ++j)
            basis[j][i] = vectors[j][order[i]];
    }
    if (determinant(basis) < 0)
        for (auto &row : basis)
            row[2] = -row[2];
    const Quaternion metric = quaternion(basis);
    std::copy(metric.begin(), metric.end(), rotation);
}

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

double softplus(double x) {
    return std::max(0.0, x) + std::log1p(std::exp(-std::abs(x)));
}

void compose(const std::filesystem::path &image_path, const std::filesystem::path &disparity_path,
             const std::filesystem::path &delta_path, const std::filesystem::path &camera_path,
             const std::filesystem::path &output, bool delta_hwc, bool features_only) {
    size_t width = 0, height = 0;
    double focal = 0;
    std::ifstream camera_file(camera_path);
    if (!(camera_file >> width >> height >> focal) || !width || !height || !(focal > 0) || !std::isfinite(focal))
        throw std::runtime_error("invalid camera metadata");
    const auto image = read_tensor(image_path, image_pixels * 3);
    const auto disparity = read_tensor(disparity_path, image_pixels * 2);
    const float factor = static_cast<float>(focal / width);
    const auto depth = [&](float value) { return factor / std::clamp(value, 1e-4f, 1e4f); };
    const float global = depth(*std::max_element(disparity.begin(), disparity.end())) + 1e-6f;
    const float normalization = 1.0f / global;
    if (features_only) {
        std::vector<float> features(image_pixels * 5);
        for (size_t pixel = 0; pixel < image_pixels; ++pixel) {
            for (size_t c = 0; c < 3; ++c)
                features[pixel * 5 + c] = 2.0f * image[pixel * 3 + c] - 1.0f;
            for (size_t c = 0; c < 2; ++c) {
                const float normalized = std::min(depth(disparity[pixel * 2 + c]) * normalization, 100.0f);
                features[pixel * 5 + 3 + c] = 2.0f / normalized - 1.0f;
            }
        }
        std::ofstream file(output, std::ios::binary | std::ios::trunc);
        if (!file.write(reinterpret_cast<const char *>(features.data()), features.size() * sizeof(float)) || !file.flush())
            throw std::runtime_error("writing image depth features failed");
        return;
    }
    const auto delta = read_tensor(delta_path, splat_pixels * 28);
    const std::array<double, 3> camera{width / (2 * focal), height / (2 * focal), 1};
    std::array<std::vector<float>, 5> metric{
        std::vector<float>(splat_pixels * 2 * 3), std::vector<float>(splat_pixels * 2 * 3),
        std::vector<float>(splat_pixels * 2 * 4), std::vector<float>(splat_pixels * 2 * 3),
        std::vector<float>(splat_pixels * 2),
    };
    for (size_t layer = 0; layer < 2; ++layer)
        for (size_t y = 0; y < splat_side; ++y)
            for (size_t x = 0; x < splat_side; ++x) {
                const size_t pixel = y * splat_side + x, vertex = layer * splat_pixels + pixel;
                auto d = [&](size_t channel) {
                    const size_t component = channel * 2 + layer;
                    return delta[delta_hwc ? pixel * 28 + component : component * splat_pixels + pixel];
                };
                const size_t i = (2 * y) * image_side + 2 * x;
                const std::array<size_t, 4> pixels{i, i + 1, i + image_side, i + image_side + 1};
                double inverse = 0;
                for (size_t p : pixels)
                    inverse = std::max(inverse, 1.0 / std::min(depth(disparity[p * 2 + layer]) * normalization, 100.0f));
                const double z = 1 / (softplus(std::log(std::expm1(std::max(inverse, 1e-6))) + 0.001 * d(2)) + 1e-3);
                const double xx = 2.0 * (x + 0.5) / splat_side - 1 + 0.001 * d(0);
                const double yy = 2.0 * (y + 0.5) / splat_side - 1 + 0.001 * d(1);
                metric[0][vertex * 3] = static_cast<float>(camera[0] * xx * z * global);
                metric[0][vertex * 3 + 1] = static_cast<float>(camera[1] * yy * z * global);
                metric[0][vertex * 3 + 2] = static_cast<float>(z * global);
                std::array<double, 3> sizes;
                for (size_t c = 0; c < 3; ++c) {
                    sizes[c] = (4.0 / image_side) * z * global * 10 * sigmoid(d(3 + c) * (10.0 / 9) - std::log(9.0));
                    double avg = 0;
                    for (size_t p : pixels)
                        avg += 0.25 * image[p * 3 + c];
                    const double base = std::clamp(avg, 0.01, 0.99);
                    const double srgb = sigmoid(std::log(base / (1 - base)) + 0.1 * d(10 + c));
                    metric[3][vertex * 3 + c] = static_cast<float>(
                        srgb <= 0.04045 ? srgb / 12.92 : std::pow((srgb + 0.055) / 1.055, 2.4));
                }
                project_covariance({1.0 + d(6), d(7), d(8), d(9)}, sizes, camera,
                                   metric[1].data() + vertex * 3, metric[2].data() + vertex * 4);
                metric[4][vertex] = static_cast<float>(sigmoid(d(13)));
            }
    std::filesystem::create_directories(output);
    for (size_t i = 0; i < metric.size(); ++i) {
        const auto path = output / ("metric-" + std::to_string(i) + ".f32");
        std::ofstream out(path, std::ios::binary);
        if (!out.write(reinterpret_cast<const char *>(metric[i].data()), metric[i].size() * sizeof(float)) || !out.flush())
            throw std::runtime_error("cannot write " + path.string());
    }
}
} // namespace

int main(int argc, char **argv) {
    const bool features = argc == 6 && std::string(argv[1]) == "--features";
    const bool delta_hwc = argc == 7 && std::string(argv[6]) == "--delta-hwc";
    if (argc != 6 && !delta_hwc) {
        std::cerr << "usage: sharp_compose IMAGE DISPARITY DELTA CAMERA OUTPUT_DIRECTORY [--delta-hwc]\n"
                     "       sharp_compose --features IMAGE DISPARITY CAMERA IMAGE_DEPTH_HWC\n";
        return 2;
    }
    try {
        if (features)
            compose(argv[2], argv[3], {}, argv[4], argv[5], false, true);
        else
            compose(argv[1], argv[2], argv[3], argv[4], argv[5], delta_hwc, false);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
