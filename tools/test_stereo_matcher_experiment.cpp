#include "stereo_matcher_experiment.hpp"
#ifdef FRAME_STEREO_EXPERIMENT_VULKAN
#include "vulkan_stereo_pipeline.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace SE = Spatial::StereoExperiment;
namespace {
constexpr float focalBaseline = 24.282608f;
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
const char* name(SE::Method method) {
    switch (method) {
    case SE::Method::Baseline: return "baseline";
    case SE::Method::EdgeAware: return "edge-aware";
    case SE::Method::Sgm: return "sgm";
    case SE::Method::NoiseAware: return "noise-aware";
    case SE::Method::CompositeRaw: return "composite-raw";
    case SE::Method::CompositeNormalized: return "composite-normalized";
    case SE::Method::Adaptive: return "adaptive";
    case SE::Method::GuardedAdaptive: return "guarded-adaptive";
    }
    return "unknown";
}
std::vector<uint8_t> texture(int width, int height, unsigned seed, int offset = 20) {
    std::mt19937 random(seed);
    std::vector<uint8_t> noise(size_t(width) * height), result(noise.size());
    for (auto& value : noise) value = uint8_t(offset + random() % 160);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int sum = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
            sum += noise[size_t(std::clamp(y + dy, 0, height - 1)) * width +
                         std::clamp(x + dx, 0, width - 1)];
        result[size_t(y) * width + x] = uint8_t(sum / 9);
    }
    return result;
}
std::vector<uint8_t> shift(const std::vector<uint8_t>& left, int width, int height, float d) {
    std::vector<uint8_t> right(left.size());
    const int integer = int(d);
    const float fraction = d - integer;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const int base = std::min(x + integer, width - 1);
        const int next = std::min(base + 1, width - 1);
        right[size_t(y) * width + x] = uint8_t(std::lround(
            (1 - fraction) * left[size_t(y) * width + base] +
            fraction * left[size_t(y) * width + next]));
    }
    return right;
}
struct Score { size_t total = 0, accepted = 0, bad = 0; double error = 0; };
Score score(const std::vector<float>& depth, int width, int height,
            int x0, int x1, float disparity) {
    Score result;
    for (int y = 12; y < height - 12; ++y) for (int x = x0; x < x1; ++x) {
        ++result.total;
        const float z = depth[size_t(y) * width + x];
        require(std::isfinite(z) && z >= 0, "Depth must be finite and nonnegative");
        if (z <= 0) continue;
        ++result.accepted;
        const float error = std::abs(focalBaseline / z - disparity);
        result.error += error;
        result.bad += error > 1;
    }
    return result;
}
void printScore(const char* test, SE::Method method, const Score& result) {
    std::cout << test << ',' << name(method) << ",coverage="
              << 100.0 * result.accepted / std::max(size_t(1), result.total)
              << "%,accepted_mae=" << (result.accepted ? result.error / result.accepted : 0)
              << ",bad_gt_1px=" << result.bad << '/' << result.accepted << '\n';
}
void requireAccurate(const Score& result, const std::string& label) {
    require(result.accepted > result.total * .40, label + " textured-plane coverage below 40%");
    require(result.bad <= result.accepted * .01, label + " accepted bad disparities exceed 1%");
    require(result.error / result.accepted < .35, label + " accepted disparity MAE exceeds .35px");
}

void cpuTests() {
    SE::Config config;
    config.width = 256; config.height = 96; config.maxDisparity = 48;
    const int width = config.width, height = config.height;
    constexpr unsigned unrelatedSeeds[] = {912, 14123, 60017, 32293};
    std::vector<double> baselineUnrelatedFraction(4);
    for (SE::Method method : {SE::Method::Baseline, SE::Method::EdgeAware, SE::Method::Sgm}) {
        std::vector<uint8_t> left(size_t(width) * height, 100), right(left);
        auto result = SE::match(left, right, config, method);
        require(std::all_of(result.depth.begin(), result.depth.end(), [](float z) { return z == 0; }),
                std::string(name(method)) + " fabricated depth on constant images");
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
            left[size_t(y) * width + x] = uint8_t(30 + (x % 8) * 25);
        for (float d : {8.0f, 8.25f, 24.25f}) {
            right = shift(left, width, height, d);
            result = SE::match(left, right, config, method);
            const auto repeated = score(result.depth, width, height, 60, width - 60, d);
            const std::string label = "repeated-" + std::to_string(d);
            printScore(label.c_str(), method, repeated);
            require(repeated.accepted == 0, std::string(name(method)) + " accepted ambiguous repetitions");
        }

        left = texture(width, height, 42);
        for (float disparity : {3.35f, 5.2f, 8.1f, 18.35f}) {
            right = shift(left, width, height, disparity);
            result = SE::match(left, right, config, method);
            const auto plane = score(result.depth, width, height, 60, width - 32, disparity);
            const std::string label = std::string(name(method)) + " plane " + std::to_string(disparity);
            printScore(label.c_str(), method, plane);
            if (disparity < focalBaseline / 5) require(plane.accepted == 0, label + " exceeded 5m cutoff");
            else requireAccurate(plane, label);
        }
        right = shift(left, width, height, 12);
        for (auto& value : right) value = uint8_t(value + 30); // No saturation: census should survive an exposure offset.
        result = SE::match(left, right, config, method);
        const auto exposure = score(result.depth, width, height, 60, width - 32, 12);
        printScore("exposure-offset", method, exposure);
        requireAccurate(exposure, std::string(name(method)) + " exposure offset");

        for (size_t trial = 0; trial < 4; ++trial) {
            right = texture(width, height, unrelatedSeeds[trial]);
            result = SE::match(left, right, config, method);
            const auto unrelated = score(result.depth, width, height, 60, width - 32, 12);
            // Independent images have no known disparity. Every accepted sample
            // is a false correspondence; a nominal disparity error is meaningless.
            const double fraction = double(unrelated.accepted) / unrelated.total;
            std::cout << "unrelated-pair," << name(method) << ",seed=" << unrelatedSeeds[trial]
                      << ",false_acceptance=" << unrelated.accepted << '/' << unrelated.total
                      << ",fraction=" << fraction << '\n';
            if (method == SE::Method::Baseline) baselineUnrelatedFraction[trial] = fraction;
            else require(fraction <= baselineUnrelatedFraction[trial] + .01,
                         std::string(name(method)) + " unrelated-image false acceptance exceeds baseline by over 1 percentage point");
        }

        // Render a known disparity step by forward-projecting left pixels into
        // the right view. A nearer sample wins each collision. Visibility is
        // therefore geometric ground truth, independent of either matcher.
        constexpr int begin = 140, end = 195, backgroundD = 8, foregroundD = 28;
        left = texture(width, height, 144);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            auto& value = left[size_t(y) * width + x];
            value = uint8_t(value / 3 + (x >= begin && x < end ? 140 : 20));
        }
        right.assign(left.size(), 30);
        std::vector<int> winningDisparity(left.size(), -1), owner(left.size(), -1);
        std::vector<uint8_t> visible(left.size(), 0);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const int d = x >= begin && x < end ? foregroundD : backgroundD;
            const int targetX = x - d;
            if (targetX < 0) continue;
            const size_t target = size_t(y) * width + targetX;
            if (d > winningDisparity[target]) {
                winningDisparity[target] = d;
                owner[target] = int(size_t(y) * width + x);
                right[target] = left[size_t(y) * width + x];
            }
        }
        for (int source : owner) if (source >= 0) visible[size_t(source)] = 1;
        result = SE::match(left, right, config, method);
        const auto foreground = score(result.depth, width, height, begin + 8, end - 8, foregroundD);
        const auto background = score(result.depth, width, height, 60, begin - 28, backgroundD);
        printScore("step-foreground", method, foreground);
        printScore("step-background", method, background);
        requireAccurate(foreground, std::string(name(method)) + " step foreground");
        requireAccurate(background, std::string(name(method)) + " step background");
        size_t occluded = 0, falseAccepted = 0;
        for (int y = 12; y < height - 12; ++y)
            for (int x = begin - (foregroundD - backgroundD) + 6; x < begin - 6; ++x) {
                const size_t i = size_t(y) * width + x;
                require(!visible[i], "Synthetic occlusion ground truth unexpectedly visible");
                ++occluded; falseAccepted += result.depth[i] > 0;
            }
        std::cout << "true-occlusion," << name(method) << ",false_acceptance="
                  << falseAccepted << '/' << occluded << '\n';
        require(falseAccepted <= occluded * .05,
                std::string(name(method)) + " true-occlusion false acceptance exceeds 5%");
    }
    std::cout << "Controlled CPU matcher tests passed\n";
}

void noiseTests() {
    SE::Config config;
    config.width = 128; config.height = 64; config.maxDisparity = 32;
    auto left = texture(config.width, config.height, 983);
    auto right = shift(left, config.width, config.height, 8.25f);
    const auto baseline = SE::match(left, right, config, SE::Method::Baseline);
    config.noiseDeadband = -1; // Every descriptor bit participates: exact cost parity.
    auto masked = SE::match(left, right, config, SE::Method::NoiseAware);
    require(masked.leftDisparity == baseline.leftDisparity && masked.rightDisparity == baseline.rightDisparity &&
            masked.depth == baseline.depth && masked.leftBestCost == baseline.leftBestCost &&
            masked.leftUniquenessGap == baseline.leftUniquenessGap &&
            masked.leftMatchRejection == baseline.leftMatchRejection,
            "Full-information masked census differs from baseline");
    config.noiseDeadband = 255;
    masked = SE::match(left, right, config, SE::Method::NoiseAware);
    require(std::all_of(masked.leftDisparity.begin(), masked.leftDisparity.end(), [](float d) { return d == 0; }) &&
            std::all_of(masked.depth.begin(), masked.depth.end(), [](float z) { return z == 0; }),
            "Suppressing every bit fabricated confidence");
    config.noiseDeadband = 2;
    for (bool periodic : {false, true}) {
        for (int y = 0; y < config.height; ++y) for (int x = 0; x < config.width; ++x)
            left[size_t(y) * config.width + x] = periodic ? uint8_t(30 + (x % 8) * 25) : 100;
        right = shift(left, config.width, config.height, 8);
        masked = SE::match(left, right, config, SE::Method::NoiseAware);
        require(std::all_of(masked.depth.begin(), masked.depth.end(), [](float z) { return z == 0; }),
                "Masked census accepted flat or periodic ambiguity");
    }
    left = texture(config.width, config.height, 983);
    right = shift(left, config.width, config.height, 8);
    masked = SE::match(left, right, config, SE::Method::NoiseAware);
    for (size_t i = 0; i < left.size(); ++i) {
        require(masked.leftInformation[i] >= 0 && masked.leftInformation[i] <= 240,
                "Invalid joint information count");
        if (masked.leftDisparity[i] > 0)
            require(masked.leftInformation[i] >= config.minInformation, "Accepted insufficient descriptor information");
    }
    requireAccurate(score(masked.depth, config.width, config.height, 40, 96, 8), "masked census integer plane");
    std::cout << "Masked census parity, information and ambiguity tests passed\n";
}

void compositeTests() {
    SE::Config c;
    c.width = 96; c.height = 64; c.maxDisparity = 32;
    const auto left = texture(c.width, c.height, 983);
    const auto right = shift(left, c.width, c.height, 8.25f);
    auto sameStages = [&](const SE::Result& a, const SE::Result& b) {
        require(a.leftDisparity == b.leftDisparity && a.rightDisparity == b.rightDisparity &&
            a.preSupportDepth == b.preSupportDepth && a.depth == b.depth &&
            a.leftBestCost == b.leftBestCost && a.rightBestCost == b.rightBestCost &&
            a.leftRawBestCost == b.leftRawBestCost && a.rightRawBestCost == b.rightRawBestCost &&
            a.leftUniquenessGap == b.leftUniquenessGap && a.rightUniquenessGap == b.rightUniquenessGap &&
            a.leftBestIntegerDisparity == b.leftBestIntegerDisparity && a.rightBestIntegerDisparity == b.rightBestIntegerDisparity &&
            a.leftMatchRejection == b.leftMatchRejection && a.rightMatchRejection == b.rightMatchRejection &&
            a.lrRejection == b.lrRejection && a.supportRejection == b.supportRejection &&
            a.leftInformation == b.leftInformation && a.rightInformation == b.rightInformation &&
            a.leftParabolaDisparity == b.leftParabolaDisparity && a.rightParabolaDisparity == b.rightParabolaDisparity,
            "Composite zero-weight or OMP stage parity failed");
    };
    for (auto mode : {SE::Refinement::Integer, SE::Refinement::Parabola}) {
        c.refinement = mode; c.compositeAlpha = 0;
        const auto baseline = SE::match(left, right, c, SE::Method::Baseline);
        for (auto method : {SE::Method::CompositeRaw, SE::Method::CompositeNormalized})
            sameStages(baseline, SE::match(left, right, c, method));
    }
    // Per-hypothesis zero-weight parity in both directions, not just selected winners.
    const auto lc = SE::detail::census(left, c), rc = SE::detail::census(right, c);
    const auto lp = SE::detail::photometric(left, c, false), rp = SE::detail::photometric(right, c, false);
    for (bool reverse : {false, true}) for (int x = 5; x < c.width - 5; ++x)
        for (int d = 1; d <= SE::detail::available(x, reverse, c); ++d) {
            const auto v = SE::detail::compositeComponents(reverse ? rc : lc, reverse ? lc : rc,
                reverse ? rp : lp, reverse ? lp : rp, x, 20, d, reverse, c);
            require(SE::detail::compositeCost(v, c) == SE::detail::costAt(
                reverse ? rc : lc, reverse ? lc : rc, reverse ? right : left, x, 20, d, reverse, c, false),
                "Per-hypothesis census cost parity failed");
            for (double value : v) require(std::isfinite(value) && value >= 0 && value <= 240, "Residual out of bounds");
        }
    // Independent hand-computed raw residual: reversed ramp has negative signed gradients.
    std::vector<uint8_t> ramp(left.size()), flipped(left.size());
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        ramp[size_t(y) * c.width + x] = uint8_t(40 + x);
        flipped[size_t(y) * c.width + x] = uint8_t(180 - x);
    }
    const auto a = SE::detail::photometric(ramp, c, false), b = SE::detail::photometric(flipped, c, false);
    require(a.gx[20 * c.width + 40] == 1 && b.gx[20 * c.width + 40] == -1, "Unsigned gradient subtraction");
    const auto v = SE::detail::compositeComponents(SE::detail::census(ramp, c), SE::detail::census(flipped, c),
        a, b, 40, 20, 8, false, c);
    require(v[1] == 240 && v[2] == 30, "Hand-computed luma/gradient cross residual mismatch");
    // Positive gain/bias without clipping cancels in normalized channels above sigma floor.
    auto exposure = left;
    for (size_t i = 0; i < left.size(); ++i) {
        ramp[i] = uint8_t(20 + (i * 13 + (i / c.width) * 7) % 70);
        exposure[i] = uint8_t(2 * ramp[i] + 10);
    }
    const auto na = SE::detail::photometric(ramp, c, true), nb = SE::detail::photometric(exposure, c, true);
    for (size_t i = 0; i < left.size(); ++i)
        require(std::abs(na.luma[i] - nb.luma[i]) < 1e-10 &&
            std::abs(na.gx[i] - nb.gx[i]) < 1e-10 && std::abs(na.gy[i] - nb.gy[i]) < 1e-10,
            "Normalized channel gain/bias invariance failed");
    std::vector<uint8_t> flat(left.size(), 128);
    const auto np = SE::detail::photometric(flat, c, true);
    require(std::all_of(np.luma.begin(), np.luma.end(), [](double value) { return value == 0; }),
            "Constant image normalization is not finite zero");
    c.compositeAlpha = .25;
    for (auto method : {SE::Method::CompositeRaw, SE::Method::CompositeNormalized}) {
        const auto empty = SE::match(flat, flat, c, method);
        require(std::all_of(empty.depth.begin(), empty.depth.end(), [](float value) { return value == 0; }),
                "Composite fabricated support on flat input");
        c.refinement = SE::Refinement::Integer;
#ifdef _OPENMP
        const int oldThreads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        const auto single = SE::match(left, right, c, method);
#ifdef _OPENMP
        omp_set_num_threads(8);
#endif
        const auto parallel = SE::match(left, right, c, method);
        sameStages(single, parallel);
        require(single.leftComponents.winner == parallel.leftComponents.winner &&
            single.rightComponents.competitor == parallel.rightComponents.competitor &&
            single.leftComponents.competitorDisparity == parallel.leftComponents.competitorDisparity,
            "OMP component diagnostic mismatch");
#ifdef _OPENMP
        omp_set_num_threads(oldThreads);
#endif
        for (float disparity : parallel.leftDisparity)
            require(disparity == std::floor(disparity), "Integer arm contains subpixel offsets");
        c.refinement = SE::Refinement::Parabola;
        const auto fit = SE::match(left, right, c, method);
        auto derived = single;
        for (size_t i = 0; i < derived.leftDisparity.size(); ++i) {
            if (derived.leftDisparity[i] > 0) derived.leftDisparity[i] = derived.leftParabolaDisparity[i];
            if (derived.rightDisparity[i] > 0) derived.rightDisparity[i] = derived.rightParabolaDisparity[i];
        }
        SE::rerunAcceptance(derived, c);
        sameStages(derived, fit);
        require(single.leftParabolaDisparity == fit.leftDisparity && single.rightParabolaDisparity == fit.rightDisparity &&
            single.leftBestIntegerDisparity == fit.leftBestIntegerDisparity && single.leftMatchRejection == fit.leftMatchRejection,
            "Refinement changed selection or fixed-basin diagnostic fit");
    }
    for (double value : {-1., 2., std::numeric_limits<double>::quiet_NaN()}) {
        c.compositeAlpha = value;
        bool rejected = false;
        try { SE::match(left, right, c, SE::Method::CompositeRaw); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid composite weight accepted");
    }
    std::cout << "Composite cost, normalization, refinement and OMP parity tests passed\n";
}

#ifdef FRAME_STEREO_EXPERIMENT_VULKAN
std::vector<uint8_t> readPgm(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "Cannot read " + path);
    auto token = [&]() {
        std::string value;
        input >> std::ws;
        while (input.peek() == '#') { std::getline(input, value); input >> std::ws; }
        input >> value;
        return value;
    };
    require(token() == "P5", "Expected binary PGM " + path);
    require(std::stoi(token()) == 616 && std::stoi(token()) == 616,
            "GPU comparison requires 616x616 PGM " + path);
    require(std::stoi(token()) == 255, "Expected 8-bit PGM " + path);
    input.get();
    std::vector<uint8_t> result(616 * 616);
    input.read(reinterpret_cast<char*>(result.data()), std::streamsize(result.size()));
    require(size_t(input.gcount()) == result.size(), "Truncated PGM " + path);
    return result;
}
bool compareMap(const std::string& name, const std::vector<float>& cpu, const std::vector<float>& gpu) {
    require(cpu.size() == gpu.size(), name + " size mismatch");
    size_t maskMismatch = 0, valueMismatch = 0;
    float maximumError = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        maskMismatch += (cpu[i] > 0) != (gpu[i] > 0);
        const float error = std::abs(cpu[i] - gpu[i]);
        maximumError = std::max(maximumError, error);
        valueMismatch += error > 2e-5f;
    }
    std::cout << name << ",mask_mismatch=" << maskMismatch << ",value_mismatch_gt_2e-5="
              << valueMismatch << ",max_error=" << maximumError << '\n';
    return maskMismatch == 0 && valueMismatch == 0;
}
bool compareSupport(const std::string& label, const SE::Result& cpu,
                    const FrameSpatial::StereoDiagnostics& gpu,
                    const std::vector<float>& gpuDepth, int width, int height) {
    size_t boundaryOnly = 0, unexplained = 0, ordinaryValueMismatch = 0;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const bool maskMismatch = (cpu.depth[i] > 0) != (gpuDepth[i] > 0);
        const bool flagMismatch = cpu.supportRejection[i] != gpu.supportRejection[i];
        if (!maskMismatch && !flagMismatch) {
            ordinaryValueMismatch += std::abs(cpu.depth[i] - gpuDepth[i]) > 2e-5f;
            continue;
        }
        const float cc = cpu.preSupportDepth[i], gc = gpu.consistencyDepth[i];
        if (cc <= 0 || gc <= 0 || x == 0 || y == 0 || x == width - 1 || y == height - 1 ||
            (gpuDepth[i] > 0) != (gpu.supportRejection[i] == 0) ||
            (cpu.depth[i] > 0) != (cpu.supportRejection[i] == 0)) {
            ++unexplained;
            continue;
        }
        const float cTolerance = std::max(.04f, cc * .04f);
        const float gTolerance = std::max(.04f, gc * .04f);
        int cpuCount = 0, gpuNominalCount = 0, certainCount = 0, uncertainCount = 0;
        bool onlyThresholdDifferences = true;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const size_t j = size_t(y + dy) * width + x + dx;
            const float cn = cpu.preSupportDepth[j], gn = gpu.consistencyDepth[j];
            if ((cn > 0) != (gn > 0)) { onlyThresholdDifferences = false; continue; }
            if (cn <= 0) continue;
            const float cDistance = std::abs(cn - cc) - cTolerance;
            const float gDistance = std::abs(gn - gc) - gTolerance;
            const bool cCompatible = cDistance <= 0, gCompatible = gDistance <= 0;
            cpuCount += cCompatible; gpuNominalCount += gCompatible;
            // Propagate the observed per-sample CPU/GPU discrepancy through
            // abs(neighbor-center) and max(.04,.04*center). The extra four ulps
            // bound float arithmetic in subtraction, multiplication and comparison.
            // This applies per neighbor; there is no global mask-mismatch allowance.
            const float roundBound = 4 * std::numeric_limits<float>::epsilon() *
                std::max({cc, gc, cn, gn, .04f});
            const float bound = std::abs(cc - gc) * 1.04f + std::abs(cn - gn) + roundBound;
            const bool atThreshold = std::abs(cDistance) <= bound && std::abs(gDistance) <= bound;
            if (cCompatible != gCompatible && !atThreshold) onlyThresholdDifferences = false;
            if (atThreshold) ++uncertainCount;
            else if (cCompatible && gCompatible) ++certainCount;
        }
        const bool cpuFlagMatchesCount = (cpuCount >= 3) == (cpu.supportRejection[i] == 0);
        // At least one threshold-adjacent pair must be able to move the count
        // across the production requirement of three agreeing neighbors.
        const bool straddlesDecision = certainCount < 3 && certainCount + uncertainCount >= 3;
        if (!onlyThresholdDifferences || !cpuFlagMatchesCount || !straddlesDecision) ++unexplained;
        else {
            ++boundaryOnly;
            if (boundaryOnly <= 8)
                std::cout << label << "/support-boundary,x=" << x << ",y=" << y
                          << ",cpu_count=" << cpuCount << ",gpu_nominal_count=" << gpuNominalCount
                          << ",certain=" << certainCount << ",uncertain=" << uncertainCount << '\n';
        }
    }
    std::cout << label << "/support-parity,boundary_only=" << boundaryOnly
              << ",unexplained=" << unexplained << ",ordinary_value_mismatch=" << ordinaryValueMismatch << '\n';
    return unexplained == 0 && ordinaryValueMismatch == 0;
}
void gpuParity(const std::string& shaderDirectory, const std::string& leftPath, const std::string& rightPath) {
    FrameSpatial::VulkanStereoPipeline pipeline;
    require(pipeline.initialize(shaderDirectory, true), pipeline.lastError());
    SE::Config config;
    config.width = 616; config.height = 616; config.maxDisparity = 96;
    auto compare = [&](const char* test, const std::vector<uint8_t>& left, const std::vector<uint8_t>& right) {
        auto result = SE::match(left, right, config, SE::Method::Baseline);
        std::vector<float> depth;
        FrameSpatial::StereoDiagnostics diagnostic;
        require(pipeline.process(left, right, depth, &diagnostic), pipeline.lastError());
        bool exact = compareMap(std::string(test) + "/left-disparity", result.leftDisparity, diagnostic.leftDisparity);
        exact = compareMap(std::string(test) + "/right-disparity", result.rightDisparity, diagnostic.rightDisparity) && exact;
        exact = compareMap(std::string(test) + "/pre-support-depth", result.preSupportDepth, diagnostic.consistencyDepth) && exact;
        compareMap(std::string(test) + "/final-depth", result.depth, depth);
        exact = compareSupport(test, result, diagnostic, depth, config.width, config.height) && exact;
        size_t lrMismatch = 0, supportMismatch = 0;
        for (size_t i = 0; i < depth.size(); ++i) {
            lrMismatch += result.lrRejection[i] != diagnostic.filterRejection[i];
            supportMismatch += result.supportRejection[i] != diagnostic.supportRejection[i];
        }
        std::cout << test << "/rejection-bits,lr_mismatch=" << lrMismatch
                  << ",support_mismatch=" << supportMismatch << '\n';
        require(exact && lrMismatch == 0,
                std::string(test) + " differs from active GPU shaders");
    };
    if (!leftPath.empty()) {
        require(!rightPath.empty(), "--left requires --right");
        compare("recorded-pair", readPgm(leftPath), readPgm(rightPath));
    } else {
        std::vector<uint8_t> left(616 * 616, 100);
        compare("blank", left, left);
        for (int y = 0; y < 616; ++y) for (int x = 0; x < 616; ++x)
            left[size_t(y) * 616 + x] = uint8_t(30 + (x % 8) * 25);
        compare("repeated", left, shift(left, 616, 616, 8));
        left = texture(616, 616, 42);
        for (float disparity : {3.35f, 5.2f, 8.1f, 18.35f}) {
            const std::string label = "fractional-" + std::to_string(disparity);
            compare(label.c_str(), left, shift(left, 616, 616, disparity));
        }
        compare("unrelated", left, texture(616, 616, 912));
    }
    std::cout << "GPU baseline parity passed\n";
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        bool gpu = false;
        std::string shaderDirectory = "shaders", leftPath, rightPath;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--gpu") { gpu = true; if (i + 1 < argc && argv[i + 1][0] != '-') shaderDirectory = argv[++i]; }
            else if (flag == "--left" && i + 1 < argc) leftPath = argv[++i];
            else if (flag == "--right" && i + 1 < argc) rightPath = argv[++i];
            else throw std::runtime_error("Usage: test_stereo_matcher_experiment [--gpu [shader-directory]] [--left left.pgm --right right.pgm]");
        }
        if (gpu) {
#ifdef FRAME_STEREO_EXPERIMENT_VULKAN
            gpuParity(shaderDirectory, leftPath, rightPath);
#else
            throw std::runtime_error("GPU parity requires -DFRAME_STEREO_EXPERIMENT_VULKAN and VulkanStereoPipeline linkage");
#endif
        } else {
            require(leftPath.empty() && rightPath.empty(), "PGM input flags require --gpu");
            cpuTests();
            noiseTests();
            compositeTests();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Matcher experiment validation failed: " << error.what() << '\n';
        return 1;
    }
}
