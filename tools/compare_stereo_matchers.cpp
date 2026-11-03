#include "stereo_matcher_experiment.hpp"
#include "depth_speckle_filter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
namespace Experiment = Spatial::StereoExperiment;

namespace {
struct Image { int width = 0, height = 0; std::vector<uint8_t> pixels; };
struct Scene {
    std::string name;
    Image left, right;
    std::vector<float> truth;
    std::vector<uint8_t> evaluated, foreground, backgroundBoundary;
};
struct Options {
    std::string left, right, output, baselineDepth, baselineDepthStage = "unspecified";
    int runs = 3, warmup = 1;
    uint32_t syntheticSeed = 1709;
    bool synthetic = false, extendedControls = false, metricsOnly = false;
    bool bothRefinements = false, baselineControls = false;
    bool reuseSelection = false;
    int referenceIntegerCutoff = 120, referenceIntegerMargin = 3;
    int referenceParabolaCutoff = 120, referenceParabolaMargin = 3;
    Experiment::Config config;
    std::vector<std::pair<std::string, Experiment::Method>> methods;
};
constexpr float kDepthScale = 24.282608f;

std::string token(std::istream& stream) {
    std::string value;
    while (stream >> value) {
        if (!value.empty() && value.front() == '#') {
            stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        return value;
    }
    throw std::runtime_error("Unexpected end of PGM header");
}
Image readPgm(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path.string());
    const auto magic = token(stream);
    Image image;
    image.width = std::stoi(token(stream));
    image.height = std::stoi(token(stream));
    const int maxValue = std::stoi(token(stream));
    if ((magic != "P5" && magic != "P2") || image.width <= 0 || image.height <= 0 ||
        image.width > 8192 || image.height > 8192 || maxValue != 255)
        throw std::runtime_error("Expected 8-bit P5/P2 PGM with max value 255: " + path.string());
    image.pixels.resize(size_t(image.width) * image.height);
    if (magic == "P5") {
        const int separator = stream.get();
        if (separator == '\r' && stream.peek() == '\n') stream.get();
        if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator)))
            throw std::runtime_error("Missing PGM raster separator");
        stream.read(reinterpret_cast<char*>(image.pixels.data()), image.pixels.size());
        if (size_t(stream.gcount()) != image.pixels.size()) throw std::runtime_error("Truncated PGM raster");
    } else {
        for (auto& pixel : image.pixels) {
            const int value = std::stoi(token(stream));
            if (value < 0 || value > 255) throw std::runtime_error("Invalid PGM pixel");
            pixel = uint8_t(value);
        }
    }
    return image;
}
void writePgm(const fs::path& path, int width, int height, const std::vector<uint8_t>& pixels) {
    if (pixels.size() != size_t(width) * height) throw std::runtime_error("PGM size mismatch");
    std::ofstream stream(path, std::ios::binary);
    stream << "P5\n" << width << " " << height << "\n255\n";
    stream.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    if (!stream) throw std::runtime_error("Cannot write " + path.string());
}
void writeFloat(const fs::path& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!stream) throw std::runtime_error("Cannot write " + path.string());
}
std::vector<float> readFloat(const fs::path& path, size_t count) {
    if (fs::file_size(path) != count * sizeof(float)) throw std::runtime_error("Baseline depth size mismatch");
    std::vector<float> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    if (!stream) throw std::runtime_error("Cannot read baseline depth");
    for (auto& value : values) if (!std::isfinite(value) || value <= 0) value = 0;
    return values;
}
bool valid(float value) { return std::isfinite(value) && value > 0; }
double ratio(size_t numerator, size_t denominator) {
    return denominator ? 100.0 * numerator / denominator : std::numeric_limits<double>::quiet_NaN();
}
double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    return values[size_t(std::ceil(quantile * values.size())) - 1];
}
std::vector<uint8_t> mask(const std::vector<float>& values) {
    std::vector<uint8_t> pixels(values.size());
    for (size_t i = 0; i < values.size(); ++i) pixels[i] = valid(values[i]) ? 255 : 0;
    return pixels;
}
uint32_t hash(uint32_t value) {
    value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
    value *= 0x846ca68bu; return value ^ (value >> 16);
}
float noise(float x, float y, uint32_t seed) {
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const float fx = x - ix, fy = y - iy;
    auto sample = [&](int xx, int yy) {
        return float(hash(uint32_t(xx) * 92837111u ^ uint32_t(yy) * 689287499u ^ seed) & 65535) / 65535.f;
    };
    return (1 - fy) * ((1 - fx) * sample(ix, iy) + fx * sample(ix + 1, iy)) +
           fy * ((1 - fx) * sample(ix, iy + 1) + fx * sample(ix + 1, iy + 1));
}
uint8_t texture(float x, float y, int pattern, int layer, uint32_t seed) {
    if (pattern == 2) return 128;
    if (pattern == 3) return uint8_t(std::lround(128 + 55 * std::sin(x * float(2 * 3.141592653589793 / 8))));
    float value = 128 + (pattern == 1 ? 6 : 100) *
        (noise(x / 2.5f, y / 2.5f, seed + layer * 3571) - .5f);
    value += (pattern == 1 ? 2 : 24) * std::sin(.19f * x + .27f * y + layer);
    return uint8_t(std::clamp(std::lround(value), 0l, 255l));
}
Scene makeScene(const std::string& name, int pattern, bool step, float backgroundD, uint32_t seed,
                int silhouetteWidth = 64) {
    Scene scene;
    scene.name = name;
    scene.left.width = scene.right.width = 256;
    scene.left.height = scene.right.height = 160;
    const int width = scene.left.width, height = scene.left.height;
    const float foregroundD = 18.5f;
    const int minX = 112, maxX = minX + silhouetteWidth, minY = 36, maxY = 124;
    auto isForeground = [&](float x, int y) {
        return step && x >= minX && x < maxX && y >= minY && y < maxY;
    };
    const size_t count = size_t(width) * height;
    scene.left.pixels.resize(count); scene.right.pixels.resize(count);
    scene.truth.resize(count); scene.evaluated.resize(count);
    scene.foreground.resize(count); scene.backgroundBoundary.resize(count);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const bool foreground = isForeground(float(x), y);
        const float disparity = foreground ? foregroundD : backgroundD;
        scene.left.pixels[i] = texture(float(x), float(y), pattern, foreground ? 1 : 0, seed);
        const bool rightForeground = isForeground(x + foregroundD, y);
        const float rightWorldX = x + (rightForeground ? foregroundD : backgroundD);
        scene.right.pixels[i] = texture(rightWorldX, float(y), pattern, rightForeground ? 1 : 0, seed);
        // Independent sensor quantization noise makes the weak-texture case nontrivial.
        if (pattern == 1) scene.right.pixels[i] = uint8_t(std::clamp(int(scene.right.pixels[i]) +
            int(hash(uint32_t(i) ^ seed ^ 7351) % 3) - 1, 0, 255));
        const float rightX = x - disparity;
        const bool correspondenceVisible = isForeground(rightX + foregroundD, y) == foreground;
        scene.truth[i] = disparity;
        scene.evaluated[i] = x >= 8 && x < width - 8 && y >= 8 && y < height - 8 &&
            rightX >= 8 && rightX < width - 8 && correspondenceVisible;
        scene.foreground[i] = foreground;
        const bool nearVertical = y >= minY - 8 && y < maxY + 8 &&
            (std::abs(x - minX) <= 8 || std::abs(x - maxX) <= 8);
        const bool nearHorizontal = x >= minX - 8 && x < maxX + 8 &&
            (std::abs(y - minY) <= 8 || std::abs(y - maxY) <= 8);
        scene.backgroundBoundary[i] = step && !foreground && (nearVertical || nearHorizontal);
    }
    return scene;
}
Scene perturbScene(const std::string& name, uint32_t seed, float gain, float bias,
                   int amplitude, float vertical = 0) {
    Scene scene = makeScene(name, 0, false, 8.25f, seed);
    const auto original = scene.right.pixels;
    const int w = scene.right.width, h = scene.right.height;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const float yy = std::clamp(y + vertical, 0.f, float(h - 1));
        const int y0 = int(yy), y1 = std::min(y0 + 1, h - 1);
        const float f = yy - y0;
        const float right = (1 - f) * original[size_t(y0) * w + x] + f * original[size_t(y1) * w + x];
        auto independentNoise = [&](uint32_t salt) {
            return amplitude ? int(hash(uint32_t(i) ^ seed ^ salt) % unsigned(2 * amplitude + 1)) - amplitude : 0;
        };
        scene.left.pixels[i] = uint8_t(std::clamp(int(scene.left.pixels[i]) + independentNoise(16381), 0, 255));
        scene.right.pixels[i] = uint8_t(std::clamp(std::lround(gain * right + bias) + independentNoise(32771), 0l, 255l));
    }
    return scene;
}
void exportResult(const fs::path& output, const std::string& method, int width, int height,
                  const Experiment::Result& result) {
    writeFloat(output / (method + "-left-disparity.f32"), result.leftDisparity);
    writeFloat(output / (method + "-right-disparity.f32"), result.rightDisparity);
    writeFloat(output / (method + "-depth.f32"), result.depth);
    writeFloat(output / (method + "-pre-support-depth.f32"), result.preSupportDepth);
    writeFloat(output / (method + "-left-best-cost.f32"), result.leftBestCost);
    writeFloat(output / (method + "-right-best-cost.f32"), result.rightBestCost);
    writeFloat(output / (method + "-left-raw-cost.f32"), result.leftRawBestCost);
    writeFloat(output / (method + "-right-raw-cost.f32"), result.rightRawBestCost);
    writeFloat(output / (method + "-left-uniqueness-gap.f32"), result.leftUniquenessGap);
    writeFloat(output / (method + "-right-uniqueness-gap.f32"), result.rightUniquenessGap);
    writeFloat(output / (method + "-left-integer-disparity.f32"), result.leftBestIntegerDisparity);
    writeFloat(output / (method + "-right-integer-disparity.f32"), result.rightBestIntegerDisparity);
    writeFloat(output / (method + "-left-information.f32"), result.leftInformation);
    writeFloat(output / (method + "-right-information.f32"), result.rightInformation);
    writeFloat(output / (method + "-left-parabola-disparity.f32"), result.leftParabolaDisparity);
    writeFloat(output / (method + "-right-parabola-disparity.f32"), result.rightParabolaDisparity);
    if (!result.leftAnchors.empty()) {
        writeFloat(output / (method + "-left-anchors.f32"), result.leftAnchors);
        writeFloat(output / (method + "-right-anchors.f32"), result.rightAnchors);
        writeFloat(output / (method + "-left-support-bits.f32"), result.leftSupportBits);
        writeFloat(output / (method + "-right-support-bits.f32"), result.rightSupportBits);
        writeFloat(output / (method + "-left-anchor-stops.f32"), result.leftAnchorStops);
        writeFloat(output / (method + "-right-anchor-stops.f32"), result.rightAnchorStops);
    }
    const char* channels[] = {"census", "luma", "gradient"};
    const std::array<std::pair<const char*, const Experiment::Result::Components*>, 2> sides{{
        {"left", &result.leftComponents}, {"right", &result.rightComponents}
    }};
    for (const auto& side : sides) {
        if (side.second->competitorDisparity.empty()) continue;
        writeFloat(output / (method + "-" + side.first + "-competitor-disparity.f32"), side.second->competitorDisparity);
        for (int k = 0; k < 3; ++k) {
            writeFloat(output / (method + "-" + side.first + "-winner-" + channels[k] + ".f32"), side.second->winner[k]);
            writeFloat(output / (method + "-" + side.first + "-competitor-" + channels[k] + ".f32"), side.second->competitor[k]);
        }
    }
    writePgm(output / (method + "-left-match-rejection.pgm"), width, height, result.leftMatchRejection);
    writePgm(output / (method + "-right-match-rejection.pgm"), width, height, result.rightMatchRejection);
    writePgm(output / (method + "-lr-rejection.pgm"), width, height, result.lrRejection);
    writePgm(output / (method + "-support-rejection.pgm"), width, height, result.supportRejection);
    writePgm(output / (method + "-mask.pgm"), width, height, mask(result.depth));
    std::vector<uint8_t> disparity(result.depth.size());
    for (size_t i = 0; i < disparity.size(); ++i) if (valid(result.depth[i]))
        disparity[i] = uint8_t(std::clamp(std::lround(255 * kDepthScale / result.depth[i] / 96), 1l, 255l));
    writePgm(output / (method + "-disparity.pgm"), width, height, disparity);
}
void metrics(std::ofstream& csv, const Scene& scene, const std::string& method,
             const Experiment::Result& result, const std::vector<double>& timing, bool synthetic) {
    size_t supported = 0, leftMatched = 0, agreement = 0, evaluated = 0, missing = 0;
    size_t accepted = 0, bad = 0, foreground = 0, foregroundBad = 0, foregroundMissing = 0;
    size_t boundary = 0, leakage = 0, boundaryMissing = 0;
    size_t integerBad = 0;
    double disparityError = 0, depthError = 0;
    double infoSum = 0, acceptedInfoSum = 0, gapSum = 0, acceptedGapSum = 0;
    size_t hypotheses = 0, matchedInformation = 0, insufficient = 0;
    for (size_t i = 0; i < result.depth.size(); ++i) {
        if (!result.leftInformation.empty() && result.leftBestIntegerDisparity[i] > 0) {
            ++hypotheses; infoSum += result.leftInformation[i];
            gapSum += std::max(0.f, result.leftUniquenessGap[i]);
            insufficient += result.leftInformation[i] < 120;
            if (valid(result.leftDisparity[i])) {
                ++matchedInformation; acceptedInfoSum += result.leftInformation[i];
                acceptedGapSum += std::max(0.f, result.leftUniquenessGap[i]);
            }
        }
        supported += valid(result.depth[i]);
        if (valid(result.leftDisparity[i])) {
            ++leftMatched;
            const int x = int(i % scene.left.width), y = int(i / scene.left.width);
            const float projected = float(x) - result.leftDisparity[i];
            const float lower = std::floor(projected);
            const float fraction = projected - lower;
            const int rightX = int(lower) + (fraction > .5f ||
                (fraction == .5f && int(lower) % 2 != 0) ? 1 : 0);
            if (rightX >= 0 && rightX < scene.left.width) {
                const auto other = result.rightDisparity[size_t(y) * scene.left.width + rightX];
                agreement += other > 1.5f && std::abs(other - result.leftDisparity[i]) <= 1.25f;
            }
        }
        if (!synthetic || !scene.evaluated[i]) continue;
        ++evaluated;
        if (scene.foreground[i]) ++foreground;
        if (scene.backgroundBoundary[i]) ++boundary;
        if (!valid(result.depth[i])) {
            ++missing;
            foregroundMissing += scene.foreground[i] != 0;
            boundaryMissing += scene.backgroundBoundary[i] != 0;
            continue;
        }
        ++accepted;
        if (!result.leftBestIntegerDisparity.empty()) {
            const float integer = result.leftBestIntegerDisparity[i];
            const float low = std::floor(scene.truth[i]), high = std::ceil(scene.truth[i]);
            const float nearest = std::min(scene.truth[i] - low, high - scene.truth[i]);
            integerBad += std::abs(integer - scene.truth[i]) > nearest + 1e-6f;
        }
        const float truthDepth = kDepthScale / scene.truth[i];
        const float estimatedD = kDepthScale / result.depth[i];
        disparityError += std::abs(estimatedD - scene.truth[i]);
        depthError += std::abs(result.depth[i] - truthDepth);
        const bool badDepth = std::abs(result.depth[i] - truthDepth) > std::max(.04f, .05f * truthDepth);
        bad += badDepth;
        foregroundBad += scene.foreground[i] && badDepth;
        leakage += scene.backgroundBoundary[i] && estimatedD > (18.5f + 8.25f) * .5f;
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    csv << (synthetic ? "synthetic" : "recorded") << ',' << scene.name << ',' << method << ','
        << result.depth.size() << ',' << ratio(supported, result.depth.size()) << ','
        << ratio(leftMatched, result.depth.size()) << ',' << ratio(agreement, leftMatched) << ','
        << evaluated << ',' << accepted << ',' << (synthetic ? ratio(missing, evaluated) : nan) << ','
        << (accepted ? disparityError / accepted : nan) << ',' << (accepted ? depthError / accepted : nan) << ','
        << (synthetic ? ratio(bad, accepted) : nan) << ',' << (synthetic ? ratio(foregroundBad, foreground) : nan) << ','
        << (synthetic ? ratio(foregroundMissing, foreground) : nan) << ','
        << (synthetic ? ratio(leakage, boundary) : nan) << ',' << (synthetic ? ratio(boundaryMissing, boundary) : nan) << ','
        << percentile(timing, .5) << ',' << percentile(timing, .95) << ','
        << (synthetic ? ratio(bad, evaluated) : nan) << ','
        << (hypotheses ? infoSum / hypotheses : nan) << ','
        << (matchedInformation ? acceptedInfoSum / matchedInformation : nan) << ','
        << (hypotheses ? gapSum / hypotheses : nan) << ','
        << (matchedInformation ? acceptedGapSum / matchedInformation : nan) << ','
        << ratio(insufficient, hypotheses) << ',' << foreground << ',' << boundary << ','
        << (synthetic ? ratio(integerBad, accepted) : nan) << '\n';
}

// Attribution only: the integer-supported mask and selected basin never change.
void fixedMask(std::ofstream& csv, const Scene& scene, const std::string& method,
               const Experiment::Result& result, const std::string& maskName,
               const std::vector<float>& otherMask) {
    if (scene.truth.empty()) return;
    size_t count = 0, integerBad = 0, fitBad = 0;
    double integerError = 0, fitError = 0;
    for (size_t i = 0; i < result.depth.size(); ++i) {
        if (!scene.evaluated[i] || !valid(result.depth[i]) || !valid(otherMask[i])) continue;
        ++count;
        const float integer = result.leftBestIntegerDisparity[i], fit = result.leftParabolaDisparity[i];
        if (fit <= 0) throw std::runtime_error("Missing fixed-mask parabola fit");
        const float truthZ = kDepthScale / scene.truth[i];
        integerError += std::abs(integer - scene.truth[i]); fitError += std::abs(fit - scene.truth[i]);
        integerBad += std::abs(kDepthScale / integer - truthZ) > std::max(.04f, .05f * truthZ);
        fitBad += std::abs(kDepthScale / fit - truthZ) > std::max(.04f, .05f * truthZ);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    csv << scene.name << ',' << method << ',' << maskName << ',' << count << ','
        << (count ? integerError / count : nan) << ',' << (count ? fitError / count : nan) << ','
        << ratio(integerBad, count) << ',' << ratio(fitBad, count) << '\n';
}
void compare(std::ofstream& csv, const fs::path& output, const Scene& scene,
             const std::string& referenceName, const std::vector<float>& reference,
             const std::string& name, const std::vector<float>& depth, bool metricsOnly) {
    size_t gained = 0, lost = 0, common = 0, disagree = 0;
    double deltaSum = 0;
    size_t commonGt = 0, referenceBad = 0, candidateBad = 0;
    double referenceError = 0, candidateError = 0;
    std::vector<uint8_t> delta(depth.size());
    for (size_t i = 0; i < depth.size(); ++i) {
        if (valid(depth[i]) && !valid(reference[i])) { ++gained; delta[i] = 255; }
        else if (!valid(depth[i]) && valid(reference[i])) { ++lost; delta[i] = 64; }
        else if (valid(depth[i]) && valid(reference[i])) {
            ++common;
            const float difference = std::abs(depth[i] - reference[i]);
            deltaSum += difference;
            if (difference > std::max(.04f, .04f * reference[i])) { ++disagree; delta[i] = 160; }
            if (!scene.truth.empty() && scene.evaluated[i]) {
                ++commonGt;
                const float truthZ = kDepthScale / scene.truth[i];
                referenceError += std::abs(kDepthScale / reference[i] - scene.truth[i]);
                candidateError += std::abs(kDepthScale / depth[i] - scene.truth[i]);
                referenceBad += std::abs(reference[i] - truthZ) > std::max(.04f, .05f * truthZ);
                candidateBad += std::abs(depth[i] - truthZ) > std::max(.04f, .05f * truthZ);
            }
        }
    }
    csv << scene.name << ',' << referenceName << ',' << name << ',' << gained << ',' << lost << ',' << common << ','
        << ratio(disagree, common) << ',' << (common ? deltaSum / common : std::numeric_limits<double>::quiet_NaN()) << ','
        << commonGt << ',' << (commonGt ? referenceError / commonGt : std::numeric_limits<double>::quiet_NaN()) << ','
        << (commonGt ? candidateError / commonGt : std::numeric_limits<double>::quiet_NaN()) << ','
        << ratio(referenceBad, commonGt) << ',' << ratio(candidateBad, commonGt) << '\n';
    if (!metricsOnly) writePgm(output / (name + "-vs-" + referenceName + "-delta.pgm"), scene.left.width, scene.left.height, delta);
}
Options parse(int argc, char** argv) {
    Options options;
    std::string methods = "baseline,edge,sgm";
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto argument = [&]() -> std::string {
            if (++i == argc) throw std::runtime_error("Missing argument for " + flag);
            return argv[i];
        };
        if (flag == "--left") options.left = argument();
        else if (flag == "--right") options.right = argument();
        else if (flag == "--output") options.output = argument();
        else if (flag == "--baseline-depth") options.baselineDepth = argument();
        else if (flag == "--baseline-depth-stage") {
            options.baselineDepthStage = argument();
            if (options.baselineDepthStage != "supported" && options.baselineDepthStage != "speckle")
                throw std::runtime_error("Baseline depth stage must be supported or speckle");
        }
        else if (flag == "--synthetic-seed") {
            const auto value = std::stoull(argument());
            if (value > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Synthetic seed outside uint32 range");
            options.syntheticSeed = uint32_t(value);
        }
        else if (flag == "--methods") methods = argument();
        else if (flag == "--runs") options.runs = std::stoi(argument());
        else if (flag == "--warmup") options.warmup = std::stoi(argument());
        else if (flag == "--max-disparity") options.config.maxDisparity = std::stoi(argument());
        else if (flag == "--p1") options.config.p1 = std::stoi(argument());
        else if (flag == "--p2") options.config.p2 = std::stoi(argument());
        else if (flag == "--edge-threshold") options.config.edgeThreshold = std::stoi(argument());
        else if (flag == "--uniqueness-margin") options.config.uniquenessMargin = std::stoi(argument());
        else if (flag == "--noise-deadband") options.config.noiseDeadband = std::stoi(argument());
        else if (flag == "--min-information") options.config.minInformation = std::stoi(argument());
        else if (flag == "--composite-alpha") options.config.compositeAlpha = std::stod(argument());
        else if (flag == "--composite-beta") options.config.compositeBeta = std::stod(argument());
        else if (flag == "--luma-clip") options.config.lumaClip = std::stod(argument());
        else if (flag == "--gradient-clip") options.config.gradientClip = std::stod(argument());
        else if (flag == "--baseline-controls") options.baselineControls = true;
        else if (flag == "--reuse-selection") options.reuseSelection = true;
        else if (flag == "--reference-integer-cutoff") options.referenceIntegerCutoff = std::stoi(argument());
        else if (flag == "--reference-integer-margin") options.referenceIntegerMargin = std::stoi(argument());
        else if (flag == "--reference-parabola-cutoff") options.referenceParabolaCutoff = std::stoi(argument());
        else if (flag == "--reference-parabola-margin") options.referenceParabolaMargin = std::stoi(argument());
        else if (flag == "--refinement") {
            const auto mode = argument();
            if (mode != "integer" && mode != "parabola" && mode != "both")
                throw std::runtime_error("Refinement must be integer, parabola or both");
            options.bothRefinements = mode == "both";
            options.config.refinement = mode == "integer" ? Experiment::Refinement::Integer : Experiment::Refinement::Parabola;
        }
        else if (flag == "--support-guard") options.config.supportGuard = std::stoi(argument());
        else if (flag == "--support-exposure") options.config.supportExposure = true;
        else if (flag == "--support-radius") options.config.supportRadius = std::stoi(argument());
        else if (flag == "--cost-cutoff") options.config.costCutoff = std::stoi(argument());
        else if (flag == "--extended-controls") options.extendedControls = true;
        else if (flag == "--metrics-only") options.metricsOnly = true;
        else if (flag == "--synthetic") options.synthetic = true;
        else if (flag == "--help") {
            std::cout << "compare_stereo_matchers --output DIR [--left L.pgm --right R.pgm] [--synthetic]\n"
                "  [--methods baseline,edge,sgm,noise,composite-raw,composite-normalized,adaptive,guarded-adaptive] [--baseline-depth measured.f32]\n"
                "  [--baseline-depth-stage supported|speckle] [--synthetic-seed 1709]\n"
                "  [--runs 3 --warmup 1 --max-disparity 96 --p1 8 --p2 48]\n"
                "  [--support-guard 0 --support-exposure --support-radius 4 --edge-threshold 16 --uniqueness-margin 3 --cost-cutoff 120]\n"
                "  [--noise-deadband 2 --min-information 120 --extended-controls --metrics-only]\n"
                "  [--composite-alpha .25 --composite-beta .5 --luma-clip 16 --gradient-clip 8]\n"
                "  [--refinement parabola|integer|both --baseline-controls --reuse-selection]\n"
                "  [--reference-integer-cutoff 120 --reference-integer-margin 3]\n"
                "  [--reference-parabola-cutoff 120 --reference-parabola-margin 3]\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument " + flag);
    }
    if (options.output.empty() || (!options.synthetic && options.left.empty()) ||
        options.left.empty() != options.right.empty() || (!options.baselineDepth.empty() && options.left.empty()) ||
        options.runs < 1 || options.runs > 100 || options.warmup < 0 || options.warmup > 100)
        throw std::runtime_error("Specify --output and paired --left/--right or --synthetic; runs 1..100, warmup 0..100");
    std::stringstream stream(methods);
    std::string name;
    while (std::getline(stream, name, ',')) {
        Experiment::Method method;
        if (name == "baseline") method = Experiment::Method::Baseline;
        else if (name == "edge") method = Experiment::Method::EdgeAware;
        else if (name == "sgm") method = Experiment::Method::Sgm;
        else if (name == "noise") method = Experiment::Method::NoiseAware;
        else if (name == "guarded-adaptive") method = Experiment::Method::GuardedAdaptive;
        else if (name == "adaptive") method = Experiment::Method::Adaptive;
        else if (name == "composite-raw") method = Experiment::Method::CompositeRaw;
        else if (name == "composite-normalized") method = Experiment::Method::CompositeNormalized;
        else throw std::runtime_error("Unknown method " + name);
        for (const auto& existing : options.methods) if (existing.first == name) throw std::runtime_error("Duplicate method " + name);
        options.methods.emplace_back(name, method);
    }
    if (options.methods.empty()) throw std::runtime_error("Specify at least one method");
    return options;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const fs::path output = options.output;
        if (fs::exists(output) && !fs::is_empty(output))
            throw std::runtime_error("Refusing to overwrite nonempty output directory");
        fs::create_directories(output);
        std::vector<Scene> scenes;
        if (!options.left.empty()) {
            Scene scene;
            scene.name = "recorded";
            scene.left = readPgm(options.left); scene.right = readPgm(options.right);
            if (scene.left.width != scene.right.width || scene.left.height != scene.right.height)
                throw std::runtime_error("Stereo dimensions differ");
            scenes.push_back(std::move(scene));
        }
        if (options.synthetic) {
            scenes.push_back(makeScene("textured-integer", 0, false, 8.f, options.syntheticSeed));
            scenes.push_back(makeScene("textured-fractional", 0, false, 8.25f, options.syntheticSeed));
            scenes.push_back(makeScene("weak-texture-fractional", 1, false, 8.25f, options.syntheticSeed));
            scenes.push_back(makeScene("step-silhouette-fractional", 0, true, 8.25f, options.syntheticSeed));
            scenes.push_back(makeScene("blank-ambiguous", 2, false, 8.25f, options.syntheticSeed));
            scenes.push_back(makeScene("repetitive-ambiguous", 3, false, 8.25f, options.syntheticSeed));
            scenes.push_back(makeScene("repetitive-alias-ambiguous", 3, false, 24.25f, options.syntheticSeed));
            if (options.extendedControls) {
                for (int amplitude : {1, 3, 6})
                    scenes.push_back(perturbScene("independent-noise-" + std::to_string(amplitude), options.syntheticSeed, 1, 0, amplitude));
                scenes.push_back(perturbScene("gain-bias", options.syntheticSeed, 1.25f, 15, 0));
                scenes.push_back(perturbScene("gain-bias-noise", options.syntheticSeed, 1.25f, 15, 3));
                scenes.push_back(perturbScene("clipped-gain", options.syntheticSeed, 2, 0, 3));
                scenes.push_back(perturbScene("vertical-half-pixel", options.syntheticSeed, 1, 0, 0, .5f));
                scenes.push_back(perturbScene("vertical-one-pixel", options.syntheticSeed, 1, 0, 0, 1));
                scenes.push_back(makeScene("thin-silhouette", 0, true, 8.25f, options.syntheticSeed, 8));
                Scene unrelated = makeScene("unrelated-pair", 0, false, 8.25f, options.syntheticSeed);
                unrelated.right = makeScene("unused", 0, false, 8.25f, options.syntheticSeed ^ 60017).right;
                // No correspondences exist: no metric depth GT may be assigned.
                unrelated.truth.clear(); unrelated.evaluated.clear();
                scenes.push_back(std::move(unrelated));
            }
        }
        std::ofstream csv(output / "metrics.csv"), deltas(output / "comparison.csv"), fixed(output / "fixed-mask.csv");
        csv << std::setprecision(9);
        deltas << std::setprecision(9);
        fixed << std::setprecision(9);
        fixed << "scene,method,mask,gt_pixels,integer_mae_px,parabola_mae_px,integer_bad_pct,parabola_bad_pct\n";
        csv << "kind,scene,method,pixels,supported_pct,left_match_pct,left_right_agreement_pct,evaluated_gt_pixels,accepted_gt_pixels,missing_gt_pct,accepted_mae_disparity_px,accepted_mae_depth_m,accepted_bad_depth_pct,foreground_bad_of_all_pct,foreground_missing_pct,background_boundary_leakage_of_all_pct,background_boundary_missing_pct,cpu_wall_p50_ms,cpu_wall_p95_ms,accepted_bad_of_all_gt_pct,selected_information_mean,matched_information_mean,selected_gap_mean,matched_gap_mean,selected_information_below_120_pct,evaluated_foreground_pixels,evaluated_boundary_pixels,supported_integer_wrong_pct\n";
        deltas << "scene,reference,method,gained_pixels,lost_pixels,common_pixels,common_depth_disagreement_pct,common_mae_between_estimates_m,common_gt_pixels,common_reference_mae_px,common_candidate_mae_px,common_reference_bad_pct,common_candidate_bad_pct\n";
        for (const auto& scene : scenes) {
            const bool synthetic = !scene.truth.empty();
            const auto sceneOutput = output / scene.name;
            fs::create_directories(sceneOutput);
            if (!options.metricsOnly) {
                writePgm(sceneOutput / "left.pgm", scene.left.width, scene.left.height, scene.left.pixels);
                writePgm(sceneOutput / "right.pgm", scene.right.width, scene.right.height, scene.right.pixels);
            }
            if (synthetic && !options.metricsOnly) {
                writeFloat(sceneOutput / "truth-disparity.f32", scene.truth);
                std::vector<uint8_t> evaluated(scene.evaluated);
                for (auto& value : evaluated) value = value ? 255 : 0;
                writePgm(sceneOutput / "truth-evaluated.pgm", scene.left.width, scene.left.height, evaluated);
            }
            auto config = options.config;
            config.width = scene.left.width; config.height = scene.left.height;
            std::vector<float> reference, gpuReference;
            std::string referenceName;
            if (!synthetic && !options.baselineDepth.empty()) {
                gpuReference = readFloat(options.baselineDepth, scene.left.pixels.size());
                writePgm(sceneOutput / "external-reference-mask.pgm", config.width, config.height, mask(gpuReference));
                Experiment::Result measured;
                measured.depth = gpuReference;
                measured.leftDisparity.resize(gpuReference.size()); measured.rightDisparity.resize(gpuReference.size());
                metrics(csv, scene, "external-reference", measured, {}, false);
            }
            struct Run { std::string name; Experiment::Method method; Experiment::Config config; };
            std::vector<Run> runs;
            auto append = [&](const std::string& name, Experiment::Method method, Experiment::Config mc) {
                const auto modes = options.bothRefinements ?
                    std::vector<Experiment::Refinement>{Experiment::Refinement::Integer, Experiment::Refinement::Parabola} :
                    std::vector<Experiment::Refinement>{mc.refinement};
                for (auto mode : modes) {
                    mc.refinement = mode;
                    runs.push_back({name + (options.bothRefinements ?
                        (mode == Experiment::Refinement::Integer ? "-integer" : "-parabola") : ""), method, mc});
                }
            };
            if (options.baselineControls) {
                auto mc = config; mc.costCutoff = 120; mc.uniquenessMargin = 3;
                append("default", Experiment::Method::Baseline, mc);
                const size_t begin = runs.size();
                append("calibrated", Experiment::Method::Baseline, mc);
                for (size_t j = begin; j < runs.size(); ++j) {
                    const bool integer = runs[j].config.refinement == Experiment::Refinement::Integer;
                    runs[j].config.costCutoff = integer ? options.referenceIntegerCutoff : options.referenceParabolaCutoff;
                    runs[j].config.uniquenessMargin = integer ? options.referenceIntegerMargin : options.referenceParabolaMargin;
                }
            }
            for (const auto& method : options.methods) append(method.first, method.second, config);
            std::vector<std::pair<Run, Experiment::Result>> previous;
            for (const auto& run : runs) {
                const auto& config = run.config;
                const auto method = std::make_pair(run.name, run.method);
                std::vector<double> timings;
                Experiment::Result result;
                auto selectionConfig = [](const Experiment::Config& c) {
                    return std::make_tuple(c.width, c.height, c.maxDisparity, c.p1, c.p2, c.edgeThreshold,
                        c.uniquenessMargin, c.noiseDeadband, c.minInformation, c.costCutoff,
                        c.compositeAlpha, c.compositeBeta, c.lumaClip, c.gradientClip, c.supportRadius, c.supportGuard, c.supportExposure);
                };
                const auto cached = options.reuseSelection ? std::find_if(previous.begin(), previous.end(), [&](const auto& v) {
                    return v.first.method == run.method && selectionConfig(v.first.config) == selectionConfig(config);
                }) : previous.end();
                if (cached != previous.end()) {
                    result = cached->second;
                    for (size_t i = 0; i < result.depth.size(); ++i) {
                        if (result.leftDisparity[i] > 0) result.leftDisparity[i] = config.refinement == Experiment::Refinement::Integer ?
                            result.leftBestIntegerDisparity[i] : result.leftParabolaDisparity[i];
                        if (result.rightDisparity[i] > 0) result.rightDisparity[i] = config.refinement == Experiment::Refinement::Integer ?
                            result.rightBestIntegerDisparity[i] : result.rightParabolaDisparity[i];
                    }
                    Experiment::rerunAcceptance(result, config);
                    // A reused selection is not a full matcher timing sample.
                } else {
                    for (int i = 0; i < options.warmup; ++i)
                        Experiment::match(scene.left.pixels, scene.right.pixels, config, method.second);
                    for (int i = 0; i < options.runs; ++i) {
                        const auto start = std::chrono::steady_clock::now();
                        result = Experiment::match(scene.left.pixels, scene.right.pixels, config, method.second);
                        timings.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                    }
                }
                metrics(csv, scene, method.first, result, timings, synthetic);
                if (!options.metricsOnly) exportResult(sceneOutput, method.first, config.width, config.height, result);
                auto cleaned = result;
                FrameSpatial::removeDepthSpeckles(cleaned.depth, config.width, config.height);
                metrics(csv, scene, method.first + "-speckle", cleaned, {}, synthetic);
                if (!options.metricsOnly) {
                    writeFloat(sceneOutput / (method.first + "-speckle-depth.f32"), cleaned.depth);
                    writePgm(sceneOutput / (method.first + "-speckle-mask.pgm"), config.width, config.height, mask(cleaned.depth));
                }
                if (options.baselineControls || options.bothRefinements) {
                    for (const auto& before : previous) {
                        if (before.first.config.refinement != config.refinement) continue;
                        compare(deltas, sceneOutput, scene, before.first.name, before.second.depth,
                                method.first, result.depth, options.metricsOnly);
                        auto beforeClean = before.second.depth;
                        FrameSpatial::removeDepthSpeckles(beforeClean, config.width, config.height);
                        compare(deltas, sceneOutput, scene, before.first.name + "-speckle", beforeClean,
                                method.first + "-speckle", cleaned.depth, options.metricsOnly);
                        if (config.refinement == Experiment::Refinement::Integer) {
                            fixedMask(fixed, scene, method.first, result, "common-" + before.first.name, before.second.depth);
                            fixedMask(fixed, scene, before.first.name, before.second, "common-" + method.first, result.depth);
                        }
                    }
                    if (config.refinement == Experiment::Refinement::Integer)
                        fixedMask(fixed, scene, method.first, result, "own", result.depth);
                    previous.emplace_back(run, result);
                } else if (reference.empty()) { reference = result.depth; referenceName = method.first; }
                else compare(deltas, sceneOutput, scene, referenceName, reference, method.first, result.depth, options.metricsOnly);
                if (!gpuReference.empty()) compare(deltas, sceneOutput, scene, "external-reference", gpuReference, method.first, result.depth, options.metricsOnly);
                if (!gpuReference.empty()) compare(deltas, sceneOutput, scene, "external-reference", gpuReference,
                    method.first + "-speckle", cleaned.depth, options.metricsOnly);
                std::cout << scene.name << ' ' << method.first << " support="
                    << ratio(std::count_if(result.depth.begin(), result.depth.end(), valid), result.depth.size())
                    << "% CPU p50=" << percentile(timings, .5) << "ms\n";
            }
        }
        std::ofstream readme(output / "README.md");
        readme << "# Offline stereo matcher comparison\n\n"
            "This tool does not change the headset runtime. CPU wall times include allocations and all reference stages; "
            "they are not estimates of GPU or live pipeline performance. Warmup calls: " << options.warmup
            << "; measured calls per method and scene: " << options.runs << ".\n\n"
            "Config: maxDisparity=" << options.config.maxDisparity << ", p1=" << options.config.p1
            << ", p2=" << options.config.p2 << ", edgeThreshold=" << options.config.edgeThreshold
            << ", uniquenessMargin=" << options.config.uniquenessMargin
            << ", noiseDeadband=" << options.config.noiseDeadband << ", minInformation=" << options.config.minInformation
            << ", supportRadius=" << options.config.supportRadius
            << ", supportGuard=" << options.config.supportGuard << ", supportExposure=" << options.config.supportExposure
            << ", costCutoff=" << options.config.costCutoff << ". Depth scale=" << kDepthScale << " m*pixel.\n\n"
            "Composite alpha=" << options.config.compositeAlpha << ", beta=" << options.config.compositeBeta
            << ", lumaClip=" << options.config.lumaClip << ", gradientClip=" << options.config.gradientClip
            << "; refinement=" << (options.bothRefinements ? "both" :
                options.config.refinement == Experiment::Refinement::Integer ? "integer" : "parabola")
            << "; baseline controls=" << options.baselineControls
            << "; reuse selection=" << options.reuseSelection
            << "; calibrated integer cutoff/margin=" << options.referenceIntegerCutoff << '/' << options.referenceIntegerMargin
            << "; calibrated parabola cutoff/margin=" << options.referenceParabolaCutoff << '/' << options.referenceParabolaMargin
            << ". Reused selections still rerun both-direction LR/range and neighbor support plus speckle cleanup; "
            "their timing fields are NaN because matching was shared with an earlier arm/control. "
            "Composite definitions follow docs/stereo_composite_unary_plan.md. Component files are unweighted cross sums "
            "in 0..240 units at the selected winner and best nonadjacent competitor (ascending-disparity tie order); "
            "-1 is unavailable. Integer-mode parabola-disparity files retain the fit before acceptance reruns. "
            "Fixed-mask CSV rows use the integer-supported visible-GT mask with unchanged winners and no acceptance rerun; "
            "they are diagnostics, not candidate coverage. Integer wrong-rate accepts either nearest winner at exact half-pixel truth.\n\n"
            "Noise-aware unary uses joint reliable comparisons where both |neighbor-center| exceed the deadband. "
            "Its five-position cross is unchanged; cost=round(240*disagreements/jointBits), or 240 below minInformation. "
            "Information outputs count selected joint comparisons (0..240), including rejected hypotheses. "
            "The normalized units match the baseline range but not its confidence distribution; thresholds require separate calibration. "
            "Extended controls add bounded independent uniform noise to both views, right gain/bias and clipping, vertical resampling, "
            "an 8px thin foreground and independent-image pairing. These are synthetic stresses, not calibrated IMX616 noise. "
            "The unrelated-pair row has no GT; every supported pixel is a false correspondence.\n\n"
            "Recorded input left: " << options.left << "; right: " << options.right << "; optional external reference depth: " << options.baselineDepth
            << "; external reference stage: " << options.baselineDepthStage << ".\n\n"
            "Recorded scenes have no ground truth. Coverage, LR agreement, and differences between estimates are proxies; "
            "more coverage or agreement does not establish accuracy. Rows suffixed -speckle apply the same production 32-pixel "
            "connected-depth speckle removal after matching; unsuffixed rows stop after neighbor support. Speckle rows have "
            "no timing measurement. The external reference may be supported or speckled; its stage is supplied explicitly "
            "through --baseline-depth-stage, so compare like stages.\n\n"
            "LR agreement samples the right disparity at round-to-nearest-ties-even(x-leftDisparity), requires right disparity >1.5 px, "
            "and accepts absolute disparity difference <=1.25 px; its denominator is all accepted left matching hypotheses.\n\n"
            "Synthetic scenes are analytic rectified, noiseless pinhole pairs except independent +/-1 quantization noise on the weak-texture right image. "
            "Images are 256x160. Texture is deterministic continuous bilinear noise and a sinusoid; fractional shifts are sampled analytically. "
            "Texture and independent right-image noise seed=" << options.syntheticSeed << ". "
            "Background disparity is 8 or 8.25 px, except the periodic alias scene assigned 24.25 px; "
            "the silhouette foreground is 18.5 px in [112,176)x[36,124). "
            "Right view is rendered independently with foreground visibility. GT evaluation excludes an 8 px border and geometrically occluded correspondences. "
            "Blank and repetitive scenes have an assigned rendering disparity but lack unique photometric evidence; these are ambiguity stress tests. "
            "The periodic scenes have 8 px period and shifts differing by two periods, so they are visually indistinguishable despite different assigned depth. "
            "They do not reproduce real optics, rectification errors, camera timing, noise, or room texture.\n\n"
            "Missing GT fraction uses all evaluated pixels. MAE and bad-depth rate use accepted evaluated pixels only; "
            "bad depth means error >max(0.04 m, 5% true depth). Foreground bad-of-all uses all evaluated foreground pixels "
            "and excludes missing predictions from its numerator, so also inspect missing fraction. "
            "Background boundary leakage-of-all is the fraction of all evaluated background pixels within 8 px of the rectangle boundary "
            "given estimated disparity >13.375 px. Unsupported pixels do not count as leakage; foreground and boundary missing fractions "
            "are reported separately with those same region denominators.\n\n"
            "F32 files are native-endian float32, row-major; zero is unsupported. Disparity visualizations use a fixed 0..96 px scale. "
            "Delta masks: 255 gained support, 64 lost support, 160 common support differing by >max(0.04 m, 4% reference depth), 0 otherwise. "
            "Differences are not accuracy labels. NaN means no denominator or unavailable metric.\n";
        if (!csv || !deltas || !fixed || !readme) throw std::runtime_error("Output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "compare_stereo_matchers: " << error.what() << '\n';
        return 1;
    }
}
