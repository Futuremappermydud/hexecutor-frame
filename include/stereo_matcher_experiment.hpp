#pragma once

// Offline reference experiments only. No production pipeline includes this file.
// SGM recurrence: Hirschmueller, CVPR 2005, Eq. (12), four cardinal paths.
// https://www.robotic.dlr.de/fileadmin/robotic/hirschmu/cvpr05hh.pdf
// Baseline/SGM unary costs retain production's cross of five 48-bit census
// comparisons. P1/P2 therefore use its 0..240 Hamming-sum units. This isolates
// regularization, and is not the paper's MI unary or its eight/sixteen-path setup.
// NoiseAware separately masks uncertain comparisons, retaining the same cross.
// One uint8 unary volume and one uint16 accumulated volume are retained for one
// reference image at a time; each path streams two disparity rows per thread.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Spatial { namespace StereoExperiment {

enum class Method { Baseline, EdgeAware, Sgm, NoiseAware, CompositeRaw, CompositeNormalized, Adaptive, GuardedAdaptive };
enum class Refinement { Parabola, Integer };
struct Config {
    int width = 616, height = 616, maxDisparity = 96;
    int p1 = 8, p2 = 48, edgeThreshold = 16, uniquenessMargin = 3;
    // Masked census ablation: ignore |neighbor-center| <= deadband in either
    // view. Normalize joint reliable-bit disagreements to 0..240. These are
    // exploratory confidence settings, not calibrated production defaults.
    int noiseDeadband = 2, minInformation = 120, costCutoff = 120;
    double compositeAlpha = .25, compositeBeta = .5, lumaClip = 16, gradientClip = 8;
    int supportRadius = 4; // bounded census aggregation, offline only
    int supportGuard = 0; // bit 1: anchor arm stops; bit 2: descriptor boundary masks
    bool supportExposure = false; // independent global contrast-scaled luma gates
    Refinement refinement = Refinement::Parabola;
};
struct Timing {
    double censusMs = 0, matchMs = 0, filterMs = 0, supportMs = 0, totalMs = 0;
};
struct Result {
    std::vector<float> leftDisparity, rightDisparity, preSupportDepth, depth;
    // SGM best/gap are four-path averages; rawBestCost is the selected unary.
    // Baseline and EdgeAware bestCost and rawBestCost are identical.
    std::vector<float> leftBestCost, rightBestCost, leftRawBestCost, rightRawBestCost;
    std::vector<float> leftUniquenessGap, rightUniquenessGap;
    std::vector<float> leftBestIntegerDisparity, rightBestIntegerDisparity;
    // NoiseAware: joint bits; adaptive methods: selected support sample count.
    std::vector<float> leftInformation, rightInformation;
    std::vector<float> leftAnchors, rightAnchors;
    std::vector<float> leftSupportBits, rightSupportBits, leftAnchorStops, rightAnchorStops;
    // Fit at the accepted integer basin, before LR/support, even in integer mode.
    std::vector<float> leftParabolaDisparity, rightParabolaDisparity;
    struct Components {
        // Unweighted cross sums in 0..240 units; absent for noncomposite methods.
        std::array<std::vector<float>, 3> winner, competitor;
        std::vector<float> competitorDisparity;
    } leftComponents, rightComponents;
    // Match bits agree with diagnostic shader; extra32 = uninformative SGM unary
    // or insufficient masked-census comparison count,
    // extra64 = selected unary lacks independent uniqueness outside d +/- 1.
    std::vector<uint8_t> leftMatchRejection, rightMatchRejection;
    std::vector<uint8_t> lrRejection, supportRejection;
    Timing timing;
};

namespace detail {
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
inline bool interior(int x, int y, const Config& c) {
    return x >= 5 && x < c.width - 5 && y >= 5 && y < c.height - 5;
}
inline int available(int x, bool right, const Config& c) {
    return std::min(c.maxDisparity - 1, right ? c.width - 6 - x : x - 5);
}
inline std::vector<uint64_t> census(const std::vector<uint8_t>& image, const Config& c) {
    std::vector<uint64_t> output(image.size());
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) {
        for (int x = 0; x < c.width; ++x) {
            uint64_t bits = 0;
            int bit = 0;
            const int center = image[size_t(y) * c.width + x];
            for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int xx = std::clamp(x + dx, 0, c.width - 1);
                const int yy = std::clamp(y + dy, 0, c.height - 1);
                if (image[size_t(yy) * c.width + xx] > center) bits |= uint64_t(1) << bit;
                ++bit;
            }
            output[size_t(y) * c.width + x] = bits;
        }
    }
    return output;
}
inline std::vector<uint64_t> reliableBits(const std::vector<uint8_t>& image, const Config& c) {
    std::vector<uint64_t> output(image.size());
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        uint64_t bits = 0;
        int bit = 0;
        const int center = image[size_t(y) * c.width + x];
        for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int xx = std::clamp(x + dx, 0, c.width - 1);
            const int yy = std::clamp(y + dy, 0, c.height - 1);
            if (std::abs(int(image[size_t(yy) * c.width + xx]) - center) > c.noiseDeadband)
                bits |= uint64_t(1) << bit;
            ++bit;
        }
        output[size_t(y) * c.width + x] = bits;
    }
    return output;
}
inline unsigned costAt(const std::vector<uint64_t>& reference,
                       const std::vector<uint64_t>& target,
                       const std::vector<uint8_t>& luma,
                       int x, int y, int d, bool right, const Config& c, bool edge) {
    constexpr int dx[5] = {0, -2, 2, 0, 0};
    constexpr int dy[5] = {0, 0, 0, -2, 2};
    const size_t center = size_t(y) * c.width + x;
    unsigned total = 0, count = 0;
    for (int k = 0; k < 5; ++k) {
        const size_t ref = size_t(y + dy[k]) * c.width + x + dx[k];
        if (edge && std::abs(int(luma[ref]) - int(luma[center])) > c.edgeThreshold) continue;
        const size_t tgt = right ? ref + d : ref - d;
        total += unsigned(__builtin_popcountll(reference[ref] ^ target[tgt]));
        ++count;
    }
    // Round normalized cost to nearest integer; center is always retained.
    return edge ? (total * 5 + count / 2) / count : total;
}
struct Diagnostics {
    std::vector<float>& disparity;
    std::vector<float>& best;
    std::vector<float>& rawBest;
    std::vector<float>& gap;
    std::vector<float>& integerD;
    std::vector<uint8_t>& rejection;
    std::vector<float>* parabola = nullptr;
};
inline void allocate(Diagnostics o, size_t n) {
    o.disparity.assign(n, 0); o.best.assign(n, -1); o.rawBest.assign(n, -1);
    o.gap.assign(n, -1); o.integerD.assign(n, 0); o.rejection.assign(n, 0);
    if (o.parabola) o.parabola->assign(n, 0);
}
// Selection uses strict insertion and ascending d, including shader tie order.
// Four hypotheses suffice: at most three candidates occupy d +/- 1.
template<class Cost, class Raw>
inline void select(size_t index, int maxD, Cost cost, Raw raw, const Config& c,
                   Diagnostics o, bool sgm) {
    std::array<float, 4> rankedCost;
    rankedCost.fill(std::numeric_limits<float>::infinity());
    std::array<int, 4> rankedD{};
    float best = std::numeric_limits<float>::infinity();
    int bestD = 0;
    unsigned rawMin = 255, rawMax = 0;
    for (int d = 1; d <= maxD; ++d) {
        const float value = cost(d);
        if (value < best) { best = value; bestD = d; }
        const unsigned unary = raw(d);
        rawMin = std::min(rawMin, unary); rawMax = std::max(rawMax, unary);
        for (int rank = 0; rank < 4; ++rank) {
            if (value >= rankedCost[rank]) continue;
            for (int k = 3; k > rank; --k) {
                rankedCost[k] = rankedCost[k - 1]; rankedD[k] = rankedD[k - 1];
            }
            rankedCost[rank] = value; rankedD[rank] = d; break;
        }
    }
    float second = std::numeric_limits<float>::infinity();
    for (int rank = 0; rank < 4; ++rank)
        if (std::abs(rankedD[rank] - bestD) > 1) second = std::min(second, rankedCost[rank]);
    const unsigned selectedRaw = raw(bestD);
    uint8_t rejection = (bestD < 2 ? 4 : 0) | (selectedRaw >= unsigned(c.costCutoff) ? 8 : 0) |
                        (second <= best + c.uniquenessMargin ? 16 : 0);
    // A smoothness prior alone is not evidence of depth in a blank patch.
    if (sgm) {
        if (rawMax - rawMin <= unsigned(c.uniquenessMargin)) rejection |= 32;
        unsigned rawCompetitor = std::numeric_limits<unsigned>::max();
        for (int d = 1; d <= maxD; ++d)
            if (std::abs(d - bestD) > 1) rawCompetitor = std::min(rawCompetitor, raw(d));
        // The path prior must not manufacture confidence where two distinct
        // unary correspondences remain equally plausible (periodic textures).
        if (rawCompetitor <= selectedRaw + unsigned(c.uniquenessMargin)) rejection |= 64;
    }
    o.best[index] = best; o.rawBest[index] = float(selectedRaw);
    o.gap[index] = std::isfinite(second) ? second - best : -1;
    o.integerD[index] = float(bestD); o.rejection[index] = rejection;
    if (rejection) return;
    float disparity = float(bestD);
    if (bestD > 1 && bestD < maxD) {
        const float before = cost(bestD - 1), after = cost(bestD + 1);
        const float denominator = 2.0f * (before - 2.0f * best + after);
        if (std::abs(denominator) > 1e-4f)
            disparity += std::clamp((before - after) / denominator, -1.0f, 1.0f);
    }
    if (o.parabola) (*o.parabola)[index] = disparity;
    o.disparity[index] = c.refinement == Refinement::Integer ? float(bestD) : disparity;
}
struct Photometric {
    std::vector<double> luma, gx, gy;
};
inline Photometric photometric(const std::vector<uint8_t>& image, const Config& c, bool normalized) {
    Photometric p;
    p.luma.resize(image.size()); p.gx.resize(image.size()); p.gy.resize(image.size());
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        auto at = [&](int xx, int yy) {
            return double(image[size_t(std::clamp(yy, 0, c.height - 1)) * c.width +
                                std::clamp(xx, 0, c.width - 1)]);
        };
        double scale = 1, mean = 0;
        if (normalized) {
            double sum = 0, squared = 0;
            for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
                const double value = at(x + dx, y + dy);
                sum += value; squared += value * value;
            }
            mean = sum / 49;
            const double variance = std::max(0., squared / 49 - mean * mean);
            scale = 16 / std::max(std::sqrt(variance), 4.);
        }
        p.luma[i] = scale * (image[i] - mean);
        p.gx[i] = scale * (at(x + 1, y) - at(x - 1, y)) * .5;
        p.gy[i] = scale * (at(x, y + 1) - at(x, y - 1)) * .5;
    }
    return p;
}
inline std::array<double, 3> compositeComponents(
        const std::vector<uint64_t>& reference, const std::vector<uint64_t>& target,
        const Photometric& rp, const Photometric& tp,
        int x, int y, int d, bool right, const Config& c) {
    constexpr int dx[5] = {0, -2, 2, 0, 0}, dy[5] = {0, 0, 0, -2, 2};
    std::array<double, 3> sum{};
    for (int k = 0; k < 5; ++k) {
        const size_t q = size_t(y + dy[k]) * c.width + x + dx[k];
        const size_t t = right ? q + d : q - d;
        sum[0] += __builtin_popcountll(reference[q] ^ target[t]);
        sum[1] += 48 * std::min(std::abs(rp.luma[q] - tp.luma[t]) / c.lumaClip, 1.);
        sum[2] += 24 * (std::min(std::abs(rp.gx[q] - tp.gx[t]) / c.gradientClip, 1.) +
                        std::min(std::abs(rp.gy[q] - tp.gy[t]) / c.gradientClip, 1.));
    }
    return sum;
}
inline unsigned compositeCost(const std::array<double, 3>& v, const Config& c) {
    return unsigned(std::floor((1 - c.compositeAlpha) * v[0] + c.compositeAlpha *
        ((1 - c.compositeBeta) * v[1] + c.compositeBeta * v[2]) + .5));
}
inline void compositeMatch(const std::vector<uint64_t>& reference, const std::vector<uint64_t>& target,
        const Photometric& rp, const Photometric& tp, const Config& c, bool right,
        Diagnostics o, Result::Components& components) {
    allocate(o, reference.size());
    for (auto& v : components.winner) v.assign(reference.size(), -1);
    for (auto& v : components.competitor) v.assign(reference.size(), -1);
    components.competitorDisparity.assign(reference.size(), 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (!interior(x, y, c)) { o.rejection[i] = 1; continue; }
        const int maxD = available(x, right, c);
        if (maxD < 3) { o.rejection[i] = 2; continue; }
        std::array<unsigned, 256> costs{};
        for (int d = 1; d <= maxD; ++d)
            costs[d] = compositeCost(compositeComponents(reference, target, rp, tp, x, y, d, right, c), c);
        select(i, maxD, [&](int d) { return float(costs[d]); },
               [&](int d) { return costs[d]; }, c, o, false);
        const int winner = int(o.integerD[i]);
        int competitor = 0;
        for (int d = 1; d <= maxD; ++d)
            if (std::abs(d - winner) > 1 && (!competitor || costs[d] < costs[competitor])) competitor = d;
        const auto w = compositeComponents(reference, target, rp, tp, x, y, winner, right, c);
        for (int k = 0; k < 3; ++k) components.winner[k][i] = float(w[k]);
        if (competitor) {
            const auto v = compositeComponents(reference, target, rp, tp, x, y, competitor, right, c);
            for (int k = 0; k < 3; ++k) components.competitor[k][i] = float(v[k]);
            components.competitorDisparity[i] = float(competitor);
        }
    }
}
inline void localMatch(const std::vector<uint64_t>& reference,
                       const std::vector<uint64_t>& target,
                       const std::vector<uint8_t>& luma,
                       const Config& c, bool right, bool edge, Diagnostics o) {
    allocate(o, reference.size());
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (!interior(x, y, c)) { o.rejection[i] = 1; continue; }
        const int maxD = available(x, right, c);
        if (maxD < 3) { o.rejection[i] = 2; continue; }
        std::array<unsigned, 256> costs{};
        for (int d = 1; d <= maxD; ++d)
            costs[d] = costAt(reference, target, luma, x, y, d, right, c, edge);
        auto raw = [&](int d) { return costs[d]; };
        select(i, maxD, [&](int d) { return float(costs[d]); }, raw, c, o, false);
    }
}
// Intersect contiguous cross arms in BOTH images. Preserve all 48 census bits
// per sample; normalize sample count back to baseline 0..240 cost units.
// This is a bounded skeleton ablation, not full AD-Census region aggregation.
inline std::pair<unsigned, unsigned> adaptiveCost(
        const std::vector<uint64_t>& reference, const std::vector<uint64_t>& target,
        const std::vector<uint8_t>& rl, const std::vector<uint8_t>& tl,
        int x, int y, int d, bool right, const Config& c) {
    const int tx = x + (right ? d : -d);
    const size_t center = size_t(y) * c.width + x, targetCenter = size_t(y) * c.width + tx;
    unsigned sum = unsigned(__builtin_popcountll(reference[center] ^ target[targetCenter])), count = 1;
    int rmin = rl[center], rmax = rmin, tmin = tl[targetCenter], tmax = tmin;
    constexpr int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
    for (int arm = 0; arm < 4; ++arm) {
        int previousR = rl[center], previousT = tl[targetCenter];
        for (int k = 1; k <= c.supportRadius; ++k) {
            const int xx = x + dx[arm] * k, yy = y + dy[arm] * k;
            const int xt = tx + dx[arm] * k;
            // Census descriptors must have a complete 7x7 neighborhood.
            if (xx < 3 || xt < 3 || yy < 3 || xx >= c.width - 3 ||
                xt >= c.width - 3 || yy >= c.height - 3) break;
            const size_t ri = size_t(yy) * c.width + xx, ti = size_t(yy) * c.width + xt;
            const int rv = rl[ri], tv = tl[ti];
            if (std::abs(rv - int(rl[center])) > c.edgeThreshold ||
                std::abs(tv - int(tl[targetCenter])) > c.edgeThreshold ||
                std::abs(rv - previousR) > c.edgeThreshold ||
                std::abs(tv - previousT) > c.edgeThreshold) break;
            sum += unsigned(__builtin_popcountll(reference[ri] ^ target[ti])); ++count;
            rmin = std::min(rmin, rv); rmax = std::max(rmax, rv);
            tmin = std::min(tmin, tv); tmax = std::max(tmax, tv);
            previousR = rv; previousT = tv;
        }
    }
    // Similar luma alone cannot supply evidence on a uniform patch.
    if (count < 5 || rmax - rmin < 4 || tmax - tmin < 4) return {240, count};
    return {(sum * 5 + count / 2) / count, count};
}
inline void adaptiveMatch(const std::vector<uint64_t>& reference,
        const std::vector<uint64_t>& target, const std::vector<uint8_t>& rl,
        const std::vector<uint8_t>& tl, const Config& c, bool right, Diagnostics o,
        std::vector<float>& samples) {
    allocate(o, reference.size()); samples.assign(reference.size(), 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (!interior(x, y, c)) { o.rejection[i] = 1; continue; }
        const int maxD = available(x, right, c);
        if (maxD < 3) { o.rejection[i] = 2; continue; }
        std::array<unsigned, 256> costs{}, counts{};
        for (int d = 1; d <= maxD; ++d) {
            const auto v = adaptiveCost(reference, target, rl, tl, x, y, d, right, c);
            costs[d] = v.first; counts[d] = v.second;
        }
        select(i, maxD, [&](int d) { return float(costs[d]); },
               [&](int d) { return costs[d]; }, c, o, false);
        samples[i] = float(counts[int(o.integerD[i])]);
    }
}
// Fixed independent anchor evidence, never a disparity prior in selection.
inline std::vector<float> consistentAnchors(const std::vector<float>& reference,
        const std::vector<float>& target, const Config& c, bool right) {
    std::vector<float> anchors(reference.size(), 0);
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        const int d = int(reference[i]);
        if (d < 2) continue;
        const int tx = x + (right ? d : -d);
        if (tx < 0 || tx >= c.width) continue;
        const float td = target[size_t(y) * c.width + tx];
        if (td >= 2 && std::abs(float(d) - td) <= 1) anchors[i] = float(d);
    }
    return anchors;
}
inline std::vector<uint64_t> boundaryMasks(const std::vector<float>& anchors, const Config& c) {
    std::vector<uint64_t> masks(anchors.size(), (uint64_t(1) << 48) - 1);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (anchors[i] == 0) continue;
        int bit = 0;
        for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int xx = std::clamp(x + dx, 0, c.width - 1);
            const int yy = std::clamp(y + dy, 0, c.height - 1);
            const float other = anchors[size_t(yy) * c.width + xx];
            if (other > 0 && std::abs(other - anchors[i]) > 1) masks[i] &= ~(uint64_t(1) << bit);
            ++bit;
        }
    }
    return masks;
}
inline double supportContrastScale(const std::vector<uint8_t>& image) {
    std::array<size_t, 256> histogram{};
    for (uint8_t value : image) ++histogram[value];
    auto percentile = [&](size_t numerator) {
        // Nearest rank ceil(n*p); callers validate nonempty dimensions.
        const size_t rank = (image.size() * numerator + 9) / 10;
        size_t count = 0;
        for (int value = 0; value < 256; ++value) {
            count += histogram[value];
            if (count >= rank) return value;
        }
        return 255;
    };
    return std::max(.5, double(percentile(9) - percentile(1)) / 64.);
}
struct GuardedEvidence { unsigned cost = 240, samples = 0, bits = 0, stops = 0; };
inline GuardedEvidence guardedCost(const std::vector<uint64_t>& reference,
        const std::vector<uint64_t>& target, const std::vector<uint8_t>& rl,
        const std::vector<uint8_t>& tl, const std::vector<float>& ra,
        const std::vector<float>& ta, const std::vector<uint64_t>& rm,
        const std::vector<uint64_t>& tm, double rs, double ts,
        int x, int y, int d, bool right, const Config& c) {
    // Exact control reuses the original arithmetic and tie behavior.
    if (c.supportGuard == 0 && !c.supportExposure) {
        const auto v = adaptiveCost(reference, target, rl, tl, x, y, d, right, c);
        return {v.first, v.second, v.second * 48, 0};
    }
    const int tx = x + (right ? d : -d);
    const size_t center = size_t(y) * c.width + x, tc = size_t(y) * c.width + tx;
    GuardedEvidence out;
    unsigned sum = 0;
    auto add = [&](size_t ri, size_t ti) {
        const uint64_t joint = c.supportGuard & 2 ? rm[ri] & tm[ti] : (uint64_t(1) << 48) - 1;
        const unsigned bits = unsigned(__builtin_popcountll(joint));
        if (bits < 24) return false;
        const unsigned disagreements = unsigned(__builtin_popcountll((reference[ri] ^ target[ti]) & joint));
        sum += (disagreements * 48 + bits / 2) / bits;
        ++out.samples; out.bits += bits;
        return true;
    };
    if (!add(center, tc)) return out;
    int rmin = rl[center], rmax = rmin, tmin = tl[tc], tmax = tmin;
    constexpr int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
    for (int arm = 0; arm < 4; ++arm) {
        int previousR = rl[center], previousT = tl[tc];
        float labelR = c.supportGuard & 1 ? ra[center] : 0;
        float labelT = c.supportGuard & 1 ? ta[tc] : 0;
        for (int k = 1; k <= c.supportRadius; ++k) {
            const int xx = x + dx[arm] * k, yy = y + dy[arm] * k, xt = tx + dx[arm] * k;
            if (xx < 3 || xt < 3 || yy < 3 || xx >= c.width - 3 || xt >= c.width - 3 || yy >= c.height - 3) break;
            const size_t ri = size_t(yy) * c.width + xx, ti = size_t(yy) * c.width + xt;
            const int rv = rl[ri], tv = tl[ti];
            if (std::abs(rv - int(rl[center])) > c.edgeThreshold * rs ||
                std::abs(tv - int(tl[tc])) > c.edgeThreshold * ts ||
                std::abs(rv - previousR) > c.edgeThreshold * rs ||
                std::abs(tv - previousT) > c.edgeThreshold * ts) break;
            if (c.supportGuard & 1) {
                if ((labelR > 0 && ra[ri] > 0 && std::abs(labelR - ra[ri]) > 1) ||
                    (labelT > 0 && ta[ti] > 0 && std::abs(labelT - ta[ti]) > 1)) {
                    ++out.stops; break;
                }
                if (labelR == 0 && ra[ri] > 0) labelR = ra[ri];
                if (labelT == 0 && ta[ti] > 0) labelT = ta[ti];
            }
            if (!add(ri, ti)) break;
            rmin = std::min(rmin, rv); rmax = std::max(rmax, rv);
            tmin = std::min(tmin, tv); tmax = std::max(tmax, tv);
            previousR = rv; previousT = tv;
        }
    }
    if (out.samples >= 5 && rmax - rmin >= 4 * rs && tmax - tmin >= 4 * ts)
        out.cost = (sum * 5 + out.samples / 2) / out.samples;
    return out;
}
inline void guardedMatch(const std::vector<uint64_t>& reference,
        const std::vector<uint64_t>& target, const std::vector<uint8_t>& rl,
        const std::vector<uint8_t>& tl, const std::vector<float>& ra,
        const std::vector<float>& ta, const std::vector<uint64_t>& rm,
        const std::vector<uint64_t>& tm, double rs, double ts,
        const Config& c, bool right, Diagnostics o, std::vector<float>& samples,
        std::vector<float>& bits, std::vector<float>& stops) {
    allocate(o, reference.size()); samples.assign(reference.size(), 0);
    bits.assign(reference.size(), 0); stops.assign(reference.size(), 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (!interior(x, y, c)) { o.rejection[i] = 1; continue; }
        const int maxD = available(x, right, c);
        if (maxD < 3) { o.rejection[i] = 2; continue; }
        std::array<GuardedEvidence, 256> evidence{};
        for (int d = 1; d <= maxD; ++d)
            evidence[d] = guardedCost(reference, target, rl, tl, ra, ta, rm, tm, rs, ts, x, y, d, right, c);
        select(i, maxD, [&](int d) { return float(evidence[d].cost); },
               [&](int d) { return evidence[d].cost; }, c, o, false);
        const auto& selected = evidence[int(o.integerD[i])];
        samples[i] = float(selected.samples); bits[i] = float(selected.bits); stops[i] = float(selected.stops);
    }
}
inline void noiseMatch(const std::vector<uint64_t>& reference,
                       const std::vector<uint64_t>& target,
                       const std::vector<uint64_t>& referenceMask,
                       const std::vector<uint64_t>& targetMask,
                       const Config& c, bool right, Diagnostics o,
                       std::vector<float>& information) {
    allocate(o, reference.size());
    information.assign(reference.size(), 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (!interior(x, y, c)) { o.rejection[i] = 1; continue; }
        const int maxD = available(x, right, c);
        if (maxD < 3) { o.rejection[i] = 2; continue; }
        std::array<unsigned, 256> costs{}, counts{};
        constexpr int dx[5] = {0, -2, 2, 0, 0}, dy[5] = {0, 0, 0, -2, 2};
        for (int d = 1; d <= maxD; ++d) {
            unsigned disagree = 0, count = 0;
            for (int k = 0; k < 5; ++k) {
                const size_t ref = size_t(y + dy[k]) * c.width + x + dx[k];
                const size_t tgt = right ? ref + d : ref - d;
                const uint64_t joint = referenceMask[ref] & targetMask[tgt];
                count += unsigned(__builtin_popcountll(joint));
                disagree += unsigned(__builtin_popcountll((reference[ref] ^ target[tgt]) & joint));
            }
            counts[d] = count;
            // Invalid evidence must not become a zero-cost confident winner.
            costs[d] = count < unsigned(c.minInformation) ? 240 :
                (disagree * 240 + count / 2) / count;
        }
        select(i, maxD, [&](int d) { return float(costs[d]); },
               [&](int d) { return costs[d]; }, c, o, false);
        information[i] = float(counts[int(o.integerD[i])]);
        if (information[i] < c.minInformation) { o.rejection[i] |= 32; o.disparity[i] = 0; }
    }
}
inline void sgmMatch(const std::vector<uint64_t>& reference,
                     const std::vector<uint64_t>& target,
                     const std::vector<uint8_t>& luma,
                     const Config& c, bool right, Diagnostics o) {
    const size_t n = reference.size(), volumeSize = n * size_t(c.maxDisparity);
    allocate(o, n);
    std::vector<uint8_t> unary(volumeSize, 255);
    std::vector<uint16_t> sum(volumeSize, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 5; y < c.height - 5; ++y) for (int x = 5; x < c.width - 5; ++x) {
        const size_t base = (size_t(y) * c.width + x) * c.maxDisparity;
        const int maxD = available(x, right, c);
        for (int d = 1; d <= maxD; ++d)
            unary[base + d] = uint8_t(costAt(reference, target, luma, x, y, d, right, c, false));
    }
    constexpr int invalid = 4095;
    // Each direction completes before the next, so accumulation has no races.
    // Separate lines have disjoint output pixels and stream O(D) path state.
    for (int direction = 0; direction < 4; ++direction) {
        const bool horizontal = direction < 2, reverse = direction % 2 != 0;
        const int lineCount = horizontal ? c.height : c.width;
        const int lineLength = horizontal ? c.width : c.height;
        #pragma omp parallel for schedule(static)
        for (int line = 0; line < lineCount; ++line) {
            std::vector<int> previous(c.maxDisparity, invalid), current(c.maxDisparity, invalid);
            bool hasPrevious = false;
            int previousLuma = 0;
            for (int step = 0; step < lineLength; ++step) {
                const int position = reverse ? lineLength - 1 - step : step;
                const int x = horizontal ? position : line, y = horizontal ? line : position;
                const size_t pixel = size_t(y) * c.width + x;
                const size_t base = pixel * c.maxDisparity;
                const int maxD = available(x, right, c);
                if (!interior(x, y, c) || maxD < 1) { hasPrevious = false; continue; }
                const int priorMin = hasPrevious ? *std::min_element(previous.begin(), previous.end()) : 0;
                const int gradient = hasPrevious ? std::abs(int(luma[pixel]) - previousLuma) : 0;
                // Preserve small-jump P1; relax large-jump penalty at luma edges.
                const int p2 = std::max(c.p1, c.p2 / (1 + gradient / c.edgeThreshold));
                std::fill(current.begin(), current.end(), invalid);
                for (int d = 1; d <= maxD; ++d) {
                    int value = unary[base + d];
                    if (hasPrevious) {
                        const int before = d > 1 ? previous[d - 1] + c.p1 : invalid + c.p1;
                        const int after = d + 1 < c.maxDisparity ? previous[d + 1] + c.p1 : invalid + c.p1;
                        value += std::min({previous[d], before, after, priorMin + p2}) - priorMin;
                    }
                    current[d] = value;
                    sum[base + d] = uint16_t(sum[base + d] + value);
                }
                previous.swap(current); previousLuma = luma[pixel]; hasPrevious = true;
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        if (!interior(x, y, c)) { o.rejection[i] = 1; continue; }
        const int maxD = available(x, right, c);
        if (maxD < 3) { o.rejection[i] = 2; continue; }
        const size_t base = i * c.maxDisparity;
        select(i, maxD, [&](int d) { return float(sum[base + d]) * 0.25f; },
               [&](int d) { return unsigned(unary[base + d]); }, c, o, true);
    }
}
} // namespace detail

// Rerun every downstream decision after changing fractional offsets.
inline void rerunAcceptance(Result& result, const Config& c) {
    const auto filterStart = detail::Clock::now();
    if (c.width <= 0 || c.height <= 0 ||
        result.leftDisparity.size() != size_t(c.width) * size_t(c.height) ||
        result.rightDisparity.size() != result.leftDisparity.size())
        throw std::invalid_argument("Acceptance input dimensions do not match");
    const size_t n = size_t(c.width) * c.height;
    result.preSupportDepth.assign(n, 0); result.lrRejection.assign(n, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        const float ld = result.leftDisparity[i];
        result.lrRejection[i] = ld > 1.5f ? 1 : (ld > 0 ? 4 : 0);
        if (ld <= 1.5f) continue;
        // Turnip implements GLSL round's implementation-defined half ties
        // as ties-to-even. Match that rule explicitly (independent of fenv).
        const float projected = float(x) - ld;
        const float lower = std::floor(projected);
        const float fraction = projected - lower;
        const int rx = int(lower) + (fraction > 0.5f ||
            (fraction == 0.5f && int(lower) % 2 != 0) ? 1 : 0);
        if (rx < 0 || rx >= c.width) continue;
        const float rd = result.rightDisparity[size_t(y) * c.width + rx];
        if (rd <= 1.5f || std::abs(ld - rd) > 1.25f) continue;
        const float depth = 24.282608f / ld;
        result.lrRejection[i] = depth >= 0.25f && depth <= 5.0f ? 0 : 2;
        if (!result.lrRejection[i]) result.preSupportDepth[i] = depth;
    }
    result.timing.filterMs = detail::elapsed(filterStart);
    const auto supportStart = detail::Clock::now();
    result.depth.assign(n, 0); result.supportRejection.assign(n, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
        const size_t i = size_t(y) * c.width + x;
        const float center = result.preSupportDepth[i];
        if (center <= 0 || x == 0 || y == 0 || x == c.width - 1 || y == c.height - 1) {
            result.supportRejection[i] = center > 0 ? 1 : 0; continue;
        }
        const float tolerance = std::max(0.04f, center * 0.04f);
        int support = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const float neighbor = result.preSupportDepth[size_t(y + dy) * c.width + x + dx];
            if (neighbor > 0 && std::abs(neighbor - center) <= tolerance) ++support;
        }
        result.supportRejection[i] = support < 3 ? 1 : 0;
        if (support >= 3) result.depth[i] = center;
    }
    result.timing.supportMs = detail::elapsed(supportStart);
}

inline Result match(const std::vector<uint8_t>& left, const std::vector<uint8_t>& right,
                    const Config& c, Method method) {
    if (c.width <= 0 || c.height <= 0 || c.maxDisparity < 4 || c.maxDisparity > 256 ||
        c.supportGuard < 0 || c.supportGuard > 3 || c.supportRadius < 1 || c.supportRadius > 8 || c.p1 < 0 || c.p2 < c.p1 || c.p2 > 1024 || c.edgeThreshold < 1 ||
        c.uniquenessMargin < 0 || c.uniquenessMargin > 240 || c.noiseDeadband < -1 ||
        c.noiseDeadband > 255 || c.minInformation < 1 || c.minInformation > 240 ||
        c.costCutoff < 1 || c.costCutoff > 240 ||
        !std::isfinite(c.compositeAlpha) || c.compositeAlpha < 0 || c.compositeAlpha > 1 ||
        !std::isfinite(c.compositeBeta) || c.compositeBeta < 0 || c.compositeBeta > 1 ||
        !std::isfinite(c.lumaClip) || c.lumaClip <= 0 ||
        !std::isfinite(c.gradientClip) || c.gradientClip <= 0 ||
        (c.refinement != Refinement::Parabola && c.refinement != Refinement::Integer))
        throw std::invalid_argument("Invalid offline stereo matcher configuration");
    const size_t n = size_t(c.width) * size_t(c.height);
    if (left.size() != n || right.size() != n) throw std::invalid_argument("Stereo image dimensions do not match");
    if (method != Method::Baseline && method != Method::EdgeAware && method != Method::Sgm &&
        method != Method::GuardedAdaptive && method != Method::Adaptive && method != Method::NoiseAware && method != Method::CompositeRaw && method != Method::CompositeNormalized)
        throw std::invalid_argument("Unknown stereo matcher method");
    Result result;
    const auto start = detail::Clock::now(), censusStart = start;
    const auto leftCensus = detail::census(left, c), rightCensus = detail::census(right, c);
    std::vector<uint64_t> leftMask, rightMask;
    detail::Photometric leftPhoto, rightPhoto;
    const bool composite = method == Method::CompositeRaw || method == Method::CompositeNormalized;
    if (composite) {
        leftPhoto = detail::photometric(left, c, method == Method::CompositeNormalized);
        rightPhoto = detail::photometric(right, c, method == Method::CompositeNormalized);
    }
    if (method == Method::NoiseAware) {
        leftMask = detail::reliableBits(left, c); rightMask = detail::reliableBits(right, c);
    }
    result.timing.censusMs = detail::elapsed(censusStart);
    const auto matchStart = detail::Clock::now();
    detail::Diagnostics lo{result.leftDisparity, result.leftBestCost, result.leftRawBestCost,
        result.leftUniquenessGap, result.leftBestIntegerDisparity, result.leftMatchRejection, &result.leftParabolaDisparity};
    detail::Diagnostics ro{result.rightDisparity, result.rightBestCost, result.rightRawBestCost,
        result.rightUniquenessGap, result.rightBestIntegerDisparity, result.rightMatchRejection, &result.rightParabolaDisparity};
    if (composite) {
        detail::compositeMatch(leftCensus, rightCensus, leftPhoto, rightPhoto, c, false, lo, result.leftComponents);
        detail::compositeMatch(rightCensus, leftCensus, rightPhoto, leftPhoto, c, true, ro, result.rightComponents);
    } else if (method == Method::GuardedAdaptive) {
        result.leftAnchors.assign(n, 0); result.rightAnchors.assign(n, 0);
        if (c.supportGuard != 0) {
            auto ac = c; ac.costCutoff = 90; ac.uniquenessMargin = 6; ac.refinement = Refinement::Integer;
            const auto independent = match(left, right, ac, Method::Baseline);
            result.leftAnchors = detail::consistentAnchors(independent.leftDisparity, independent.rightDisparity, c, false);
            result.rightAnchors = detail::consistentAnchors(independent.rightDisparity, independent.leftDisparity, c, true);
        }
        std::vector<uint64_t> lm, rm;
        if (c.supportGuard & 2) {
            lm = detail::boundaryMasks(result.leftAnchors, c);
            rm = detail::boundaryMasks(result.rightAnchors, c);
        }
        const double ls = c.supportExposure ? detail::supportContrastScale(left) : 1.;
        const double rs = c.supportExposure ? detail::supportContrastScale(right) : 1.;
        detail::guardedMatch(leftCensus, rightCensus, left, right, result.leftAnchors, result.rightAnchors,
            lm, rm, ls, rs, c, false, lo, result.leftInformation, result.leftSupportBits, result.leftAnchorStops);
        detail::guardedMatch(rightCensus, leftCensus, right, left, result.rightAnchors, result.leftAnchors,
            rm, lm, rs, ls, c, true, ro, result.rightInformation, result.rightSupportBits, result.rightAnchorStops);
    } else if (method == Method::Adaptive) {
        detail::adaptiveMatch(leftCensus, rightCensus, left, right, c, false, lo, result.leftInformation);
        detail::adaptiveMatch(rightCensus, leftCensus, right, left, c, true, ro, result.rightInformation);
    } else if (method == Method::NoiseAware) {
        detail::noiseMatch(leftCensus, rightCensus, leftMask, rightMask, c, false, lo, result.leftInformation);
        detail::noiseMatch(rightCensus, leftCensus, rightMask, leftMask, c, true, ro, result.rightInformation);
    } else if (method == Method::Sgm) {
        detail::sgmMatch(leftCensus, rightCensus, left, c, false, lo);
        detail::sgmMatch(rightCensus, leftCensus, right, c, true, ro);
    } else {
        detail::localMatch(leftCensus, rightCensus, left, c, false, method == Method::EdgeAware, lo);
        detail::localMatch(rightCensus, leftCensus, right, c, true, method == Method::EdgeAware, ro);
    }
    if (method != Method::NoiseAware && method != Method::Adaptive && method != Method::GuardedAdaptive) {
        result.leftInformation.resize(n); result.rightInformation.resize(n);
        for (size_t i = 0; i < n; ++i) {
            result.leftInformation[i] = result.leftBestIntegerDisparity[i] > 0 ? 240.f : 0.f;
            result.rightInformation[i] = result.rightBestIntegerDisparity[i] > 0 ? 240.f : 0.f;
        }
    }
    result.timing.matchMs = detail::elapsed(matchStart);
    rerunAcceptance(result, c);
    result.timing.totalMs = detail::elapsed(start);
    return result;
}

}} // namespace Spatial::StereoExperiment
