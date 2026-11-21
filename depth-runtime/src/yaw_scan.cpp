#include "yaw_scan.hpp"
#include "nirhands/geometry.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>

namespace FrameSpatial {
namespace {

using MotionYaw::kPi;
using nirhands::Mat3;
using nirhands::Vec3;
using nirhands::operator-;
using nirhands::operator+;
using nirhands::operator*;

constexpr double kRoll = kYawScanAssumedRollDegrees * kPi / 180.0;
constexpr double kPitch = kYawScanAssumedPitchDegrees * kPi / 180.0;
constexpr double kProbeDegrees = 0.1;
constexpr double kMinRollSpread = 0.1;
// Row yaw response spread (px per degree) needed to fit it; the IR scans reach about 1.
constexpr double kMinRowYawSpread = 0.3;
constexpr double kYawProbe = -0.3 * kPi / 180.0;
constexpr double kFarDistance = 5.0;
constexpr size_t kSideSamples = 100, kSideKeyframes = 8, kBandSamples = 30, kBandKeyframes = 3;
constexpr std::array<double, 5> kBandEdges{0, 1.2, 2.0, 3.5, std::numeric_limits<double>::infinity()};

Mat3 pitchRotation(double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    return {{1, 0, 0, 0, c, -s, 0, s, c}};
}

Mat3 yawRotation(double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    return {{c, 0, s, 0, 1, 0, -s, 0, c}};
}

Mat3 rollRotation(double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    return {{c, -s, 0, s, c, 0, 0, 0, 1}};
}

IrKBSplineIntrinsics intrinsics(const nirhands::KbSplineCalibration& k) {
    return {k.fx, k.fy, k.cx, k.cy, k.k, k.spacing, k.sU, k.sV};
}

struct Observation {
    double disparity, residual, distance;
    double rollTerm;                    // yaw-equivalent degrees per degree of roll correction
    double verticalPx, verticalRoll, verticalPitch, verticalYaw;  // right-minus-left rows; px per degree
    int keyframe;
};

double huberCutoff(double medianError) {
    return std::max(.005, 1.345 * 1.4826 * medianError);
}

// Median of the errors with each keyframe weighted equally, as used for the outlier threshold.
double balancedMedian(const std::vector<double>& errors, const std::vector<double>& baseWeights, size_t keyframes) {
    std::vector<std::pair<double, double>> ranked(errors.size());
    for (size_t i = 0; i < errors.size(); ++i) ranked[i] = {errors[i], baseWeights[i]};
    std::sort(ranked.begin(), ranked.end());
    double halfway = .5 * double(keyframes);
    for (const auto& [error, baseWeight] : ranked) {
        halfway -= baseWeight;
        if (halfway <= 0) return error;
    }
    return ranked.empty() ? 0.0 : ranked.back().first;
}

template<size_t N>
struct BalancedFit {
    std::array<double, N> coefficients{};
    double scale = 0;  // robust sigma of the final errors
};

// Robust least squares of target(o) against columns(o), with equal initial weight per keyframe.
// The tracked translations set the scale, and equal keyframe weight limits the influence of a
// single view with many correlated patch matches. Empty when the columns are degenerate.
template<size_t N, class Columns, class Target>
std::optional<BalancedFit<N>> fitBalanced(const std::vector<Observation>& observations,
                                          const std::map<int, size_t>& keyframes, Columns columns, Target target) {
    if (observations.size() < N) return std::nullopt;
    std::vector<double> baseWeights, errors(observations.size());
    baseWeights.reserve(observations.size());
    for (const auto& o : observations) baseWeights.push_back(1.0 / double(keyframes.at(o.keyframe)));
    auto weights = baseWeights;
    BalancedFit<N> fit;
    for (int iteration = 0; iteration < 12; ++iteration) {
        std::array<std::array<double, N + 1>, N> normal{};
        for (size_t i = 0; i < observations.size(); ++i) {
            const auto x = columns(observations[i]);
            const double y = target(observations[i]);
            for (size_t r = 0; r < N; ++r) {
                for (size_t c = 0; c < N; ++c) normal[r][c] += weights[i] * x[r] * x[c];
                normal[r][N] += weights[i] * x[r] * y;
            }
        }
        // Gaussian elimination with partial pivoting.
        for (size_t c = 0; c < N; ++c) {
            size_t pivot = c;
            for (size_t r = c + 1; r < N; ++r)
                if (std::abs(normal[r][c]) > std::abs(normal[pivot][c])) pivot = r;
            std::swap(normal[c], normal[pivot]);
            if (!(std::abs(normal[c][c]) > 1e-12)) return std::nullopt;
            for (size_t r = 0; r < N; ++r) {
                if (r == c) continue;
                const double factor = normal[r][c] / normal[c][c];
                for (size_t k = c; k <= N; ++k) normal[r][k] -= factor * normal[c][k];
            }
        }
        for (size_t c = 0; c < N; ++c) fit.coefficients[c] = normal[c][N] / normal[c][c];
        for (size_t i = 0; i < observations.size(); ++i) {
            const auto x = columns(observations[i]);
            double predicted = 0;
            for (size_t c = 0; c < N; ++c) predicted += fit.coefficients[c] * x[c];
            errors[i] = std::abs(target(observations[i]) - predicted);
        }
        const double median = balancedMedian(errors, baseWeights, keyframes.size());
        fit.scale = 1.4826 * median;
        const double cutoff = huberCutoff(median);
        for (size_t i = 0; i < errors.size(); ++i)
            weights[i] = baseWeights[i] * cutoff / std::max(cutoff, errors[i]);
    }
    for (double c : fit.coefficients)
        if (!std::isfinite(c)) return std::nullopt;
    return fit;
}

// Keyframe-weighted standard deviation of one column, to check that a term can be estimated.
template<class Value>
double balancedSpread(const std::vector<Observation>& observations, const std::map<int, size_t>& keyframes,
                      Value value) {
    double weight = 0, mean = 0, square = 0;
    for (const auto& o : observations) {
        const double w = 1.0 / double(keyframes.at(o.keyframe)), x = value(o);
        weight += w;
        mean += w * x;
        square += w * x * x;
    }
    if (!(weight > 0)) return 0;
    mean /= weight;
    return std::sqrt(std::max(0.0, square / weight - mean * mean));
}

// Fixed-scale yaw with the roll correction held at `roll`.
double fitFixedScale(const std::vector<Observation>& observations, const std::map<int, size_t>& keyframes,
                     double roll) {
    const auto fit = fitBalanced<1>(observations, keyframes, [](const Observation&) { return std::array<double, 1>{1}; },
                                    [roll](const Observation& o) { return o.residual - roll * o.rollTerm; });
    return fit ? fit->coefficients[0] : std::numeric_limits<double>::quiet_NaN();
}

// Fixed-scale yaw of the matches in [minDistance, maxDistance), when they have enough support.
YawScanFit::Band fitBand(const std::vector<Observation>& observations, double minDistance, double maxDistance,
                         size_t minSamples, size_t minKeyframes, double roll) {
    YawScanFit::Band band;
    band.minDistance = minDistance;
    band.maxDistance = maxDistance;
    std::vector<Observation> selected;
    std::map<int, size_t> keyframes;
    for (const auto& o : observations)
        if (o.distance >= minDistance && o.distance < maxDistance) {
            selected.push_back(o);
            ++keyframes[o.keyframe];
        }
    band.samples = selected.size();
    band.keyframes = keyframes.size();
    if (band.samples >= minSamples && band.keyframes >= minKeyframes)
        band.yawDegrees = fitFixedScale(selected, keyframes, roll);
    return band;
}

void checkWritten(std::ofstream& out, const std::filesystem::path& path) {
    out.close();
    if (!out) throw std::runtime_error("could not write " + path.string());
}

}  // namespace

YawScanCalibration loadYawScanCalibration(const std::string& xrservice, const std::string& device) {
    std::array<nirhands::KbSplineCalibration, 2> splines{};
    const auto rig = nirhands::load_rig(xrservice, device, splines);
    for (const auto* camera : {&rig.left, &rig.right})
        if (camera->width != MotionYaw::kRawWidth || camera->height != MotionYaw::kRawHeight)
            throw std::runtime_error("IR calibration image dimensions do not match the capture");

    YawScanCalibration calibration;
    auto& problem = calibration.problem;
    problem.leftIntr = intrinsics(splines[0]);
    problem.rightIntr = intrinsics(splines[1]);
    // Validate both spline entries before opening the recording.
    const IrKBSplineProjector leftProjection(problem.leftIntr), rightProjection(problem.rightIntr);

    const Vec3 left = rig.left.center(), right = rig.right.center();
    const Vec3 baseline = right - left;
    problem.baseline = nirhands::norm(baseline);
    if (!std::isfinite(problem.baseline) || problem.baseline < .001)
        throw std::runtime_error("invalid IR baseline");
    const Vec3 x = nirhands::unit(baseline);
    const Vec3 forward = rig.left.R_cw.row(2) + rig.right.R_cw.row(2);
    const Vec3 zWithoutX = forward - nirhands::dot(x, forward) * x;
    if (!(nirhands::norm(zWithoutX) > .001)) throw std::runtime_error("invalid IR rectification axes");
    const Vec3 z = nirhands::unit(zWithoutX);
    const auto headFromRectified = Mat3::rows(x, nirhands::cross(z, x), z).transposed();
    calibration.headFromRectified = headFromRectified.m;
    calibration.leftOriginInHead = left;

    const auto rightBase = rig.right.R_cw * headFromRectified * pitchRotation(kPitch);
    const auto roll = rollRotation(kRoll);
    problem.rect0RL = (rig.left.R_cw * headFromRectified).m;
    problem.rect0RR = (rightBase * roll).m;
    problem.rect1RR = (rightBase * yawRotation(kYawProbe) * roll).m;
    problem.kHi = .145 * kPi / 180.0;
    problem.keyframes = 16;
    for (int i = 0; i < 23; ++i) problem.inverse.push_back(.02 + i * .008);
    for (int i = 0; i < 56; ++i) problem.inverse.push_back(.2 * std::pow(14.0, i / 55.0));
    return calibration;
}

MotionYaw::Reference yawScanReference(const YawScanCalibration& calibration, int64_t when, const HeadPose& head) {
    const double x = head.orientation[0], y = head.orientation[1];
    const double z = head.orientation[2], w = head.orientation[3];
    const Mat3 worldFromHead{{
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}};
    const Vec3 position{head.position[0], head.position[1], head.position[2]};
    return {when, (worldFromHead * Mat3{calibration.headFromRectified}).m,
            worldFromHead * calibration.leftOriginInHead + position};
}

YawScanFit fitYawScan(const std::vector<MotionYaw::Sample>& samples, double step, double midpointSeconds) {
    YawScanFit result;
    std::vector<Observation> all, halves[2];
    std::map<int, size_t> keyframes, halfKeyframes[2];
    const Mat3 roll = rollRotation(kRoll);
    const Mat3 probe = roll.transposed() * yawRotation(-kYawProbe) * roll;
    // A correction c applied like the yaw probe moves a right-image ray from probe(0) to probe(c).
    const double small = kProbeDegrees * kPi / 180.0;
    const Mat3 rollProbe = rollRotation(-small);
    const Mat3 pitchProbe = roll.transposed() * pitchRotation(-small) * roll;
    const Mat3 yawProbe = roll.transposed() * yawRotation(-small) * roll;
    auto horizontal = [](const Vec3& r) { return std::atan2(r[0], std::hypot(r[1], r[2])); };
    auto vertical = [](const Vec3& r) { return std::atan2(r[1], r[2]); };
    for (const auto& sample : samples) {
        const double theta = (sample.u - sample.d0 - MotionYaw::kCenterU) * step;
        const double phi = (sample.v - MotionYaw::kCenterV) * step;
        const Vec3 ray{std::tan(theta), std::sin(phi), std::cos(phi)};
        const double shifted = horizontal(probe * ray);
        const double sensitivity = (theta - shifted) / (step * (-.3));
        if (!(sensitivity > 0.0) || !std::isfinite(sensitivity)) continue;
        const double rollSensitivity = (theta - horizontal(rollProbe * ray)) / (step * kProbeDegrees);
        const Observation o{(sample.expected + sample.d0) / (2.0 * sensitivity),
                            (sample.expected - sample.d0) / sensitivity, sample.distance,
                            rollSensitivity / sensitivity, sample.dy0,
                            (vertical(rollProbe * ray) - phi) / (step * kProbeDegrees),
                            (vertical(pitchProbe * ray) - phi) / (step * kProbeDegrees),
                            (vertical(yawProbe * ray) - phi) / (step * kProbeDegrees), sample.i};
        if (!std::isfinite(o.disparity) || !std::isfinite(o.residual) || !std::isfinite(o.distance) ||
            !std::isfinite(o.rollTerm) || !std::isfinite(sample.tS)) continue;
        const size_t half = sample.tS < midpointSeconds ? 0 : 1;
        ++keyframes[sample.i];
        ++halfKeyframes[half][sample.i];
        if (sample.distance >= kFarDistance) ++result.farSamples;
        halves[half].push_back(o);
        all.push_back(o);
    }
    result.samples = all.size();
    result.keyframes = keyframes.size();
    if (all.size() < 120 || result.keyframes < 8 || halves[0].size() < 40 || halves[1].size() < 40 ||
        halfKeyframes[0].size() < 3 || halfKeyframes[1].size() < 3) return result;

    result.yawWithoutRollDegrees = fitFixedScale(all, keyframes, 0);
    result.yawDegrees = result.yawWithoutRollDegrees;
    // A right-camera roll error shifts horizontal disparity in proportion to image height.
    // Points high in the image are usually far, so without this term it reads as a depth trend.
    if (balancedSpread(all, keyframes, [](const Observation& o) { return o.rollTerm; }) >= kMinRollSpread) {
        const auto fit = fitBalanced<2>(all, keyframes,
            [](const Observation& o) { return std::array<double, 2>{1, o.rollTerm}; },
            [](const Observation& o) { return o.residual; });
        if (fit) {
            result.rollFitted = true;
            result.yawDegrees = fit->coefficients[0];
            result.rollDegrees = fit->coefficients[1];
            result.residualScaleDegrees = fit->scale;
        }
    }
    if (!result.rollFitted) {
        const auto fit = fitBalanced<1>(all, keyframes, [](const Observation&) { return std::array<double, 1>{1}; },
                                        [](const Observation& o) { return o.residual; });
        if (fit) result.residualScaleDegrees = fit->scale;
    }
    const double rollUsed = result.rollDegrees;

    // Rows of the stereo match give roll, pitch and yaw without the poses or motion depths.
    {
        std::vector<Observation> rows;
        std::map<int, size_t> rowKeyframes;
        for (const auto& o : all)
            if (std::isfinite(o.verticalPx)) {
                rows.push_back(o);
                ++rowKeyframes[o.keyframe];
            }
        if (rows.size() >= kSideSamples && rowKeyframes.size() >= kSideKeyframes) {
            const auto target = [](const Observation& o) { return -o.verticalPx; };
            const bool yawSpread = balancedSpread(rows, rowKeyframes,
                [](const Observation& o) { return o.verticalYaw; }) >= kMinRowYawSpread;
            if (yawSpread) {
                const auto fit = fitBalanced<3>(rows, rowKeyframes, [](const Observation& o) {
                    return std::array<double, 3>{o.verticalRoll, o.verticalPitch, o.verticalYaw};
                }, target);
                if (fit) {
                    YawScanFit::RowFit row{rows.size(), rowKeyframes.size(), fit->coefficients[0],
                                           fit->coefficients[1], fit->scale, fit->coefficients[2], {}};
                    row.calibratedYawDegrees = *row.yawDegrees + kYawScanRowYawOffsetDegrees;
                    result.rows = row;
                }
            }
            if (!result.rows) {
                const auto fit = fitBalanced<2>(rows, rowKeyframes,
                    [](const Observation& o) { return std::array<double, 2>{o.verticalRoll, o.verticalPitch}; },
                    target);
                if (fit)
                    result.rows = YawScanFit::RowFit{rows.size(), rowKeyframes.size(), fit->coefficients[0],
                                                     fit->coefficients[1], fit->scale, {}, {}};
            }
        }
    }

    if (balancedSpread(all, keyframes, [](const Observation& o) { return o.disparity; }) >= .5) {
        // A nearly constant disparity makes the yaw intercept and pose scale inseparable.
        const auto residual = [](const Observation& o) { return o.residual; };
        if (result.rollFitted) {
            const auto fit = fitBalanced<3>(all, keyframes,
                [](const Observation& o) { return std::array<double, 3>{1, o.rollTerm, o.disparity}; }, residual);
            if (fit) result.freeScale = YawScanFit::FreeScale{fit->coefficients[0], fit->coefficients[2]};
        } else {
            const auto fit = fitBalanced<2>(all, keyframes,
                [](const Observation& o) { return std::array<double, 2>{1, o.disparity}; }, residual);
            if (fit) result.freeScale = YawScanFit::FreeScale{fit->coefficients[0], fit->coefficients[1]};
        }
    }
    result.splitDifferenceDegrees = std::abs(fitFixedScale(halves[0], halfKeyframes[0], rollUsed) -
                                             fitFixedScale(halves[1], halfKeyframes[1], rollUsed));
    result.timeSplitAgrees = result.splitDifferenceDegrees <= kYawScanAgreementDegrees;
    for (size_t b = 0; b + 1 < kBandEdges.size(); ++b)
        result.bands.push_back(fitBand(all, kBandEdges[b], kBandEdges[b + 1], kBandSamples, kBandKeyframes, rollUsed));
    // The matches share one fixed scale, so a depth-dependent error moves the result with the
    // scene's depth mix. Both halves of a recording usually see the same mix; this check does not.
    result.nearSide = fitBand(all, 0, kYawScanSplitDistance, kSideSamples, kSideKeyframes, rollUsed);
    result.farSide = fitBand(all, kYawScanSplitDistance, kBandEdges.back(), kSideSamples, kSideKeyframes, rollUsed);
    if (!result.nearSide.yawDegrees || !result.farSide.yawDegrees) {
        result.state = YawScanFit::State::NeedNearAndFar;
        return result;
    }
    result.depthDifferenceDegrees = *result.farSide.yawDegrees - *result.nearSide.yawDegrees;
    result.depthSplitAgrees = std::abs(*result.depthDifferenceDegrees) <= kYawScanAgreementDegrees;
    result.state = result.timeSplitAgrees && result.depthSplitAgrees ? YawScanFit::State::Ready
                                                                      : YawScanFit::State::Unstable;
    return result;
}

std::string yawScanImageName(size_t index, bool right) {
    return "pair-" + std::to_string(index) + (right ? "-right.u8" : "-left.u8");
}

void writeYawScanProblem(const std::filesystem::path& path, const MotionYaw::Problem& problem) {
    std::ofstream out(path);
    out << std::setprecision(17) << "format frame-motion-yaw-v1\ngrid " << MotionYaw::kWidth << ' '
        << MotionYaw::kHeight << ' ' << problem.kHi << ' ' << MotionYaw::kHalf << '\n';
    auto values = [&out](const auto& row, char separator = ' ') {
        for (auto value : row) out << separator << value;
    };
    auto writeIntrinsics = [&](const char* side, const IrKBSplineIntrinsics& k) {
        out << "intrinsics " << side << ' ' << k.fx << ' ' << k.fy << ' ' << k.cx << ' ' << k.cy;
        values(k.k); values(k.nodeSpacing); values(k.sU); values(k.sV);
        out << '\n';
    };
    writeIntrinsics("left", problem.leftIntr);
    writeIntrinsics("right", problem.rightIntr);
    out << "rotation rect0.RL"; values(problem.rect0RL);
    out << "\nrotation rect0.RR"; values(problem.rect0RR);
    out << "\nrotation rect1.RR"; values(problem.rect1RR);
    out << "\nbaseline " << problem.baseline << "\nkeyframes " << problem.keyframes << "\ninverse " << problem.inverse.size();
    values(problem.inverse);
    out << "\nreferences " << problem.references.size() << '\n';
    for (size_t i = 0; i < problem.references.size(); ++i) {
        const auto& ref = problem.references[i];
        out << "ref\t" << ref.when;
        values(ref.R, '\t');
        values(ref.t, '\t');
        out << "\tpairs/" << yawScanImageName(i, false) << "\tpairs/" << yawScanImageName(i, true) << '\n';
    }
    checkWritten(out, path);
}

void writeYawScanSamples(const std::filesystem::path& path, const std::vector<MotionYaw::Sample>& samples) {
    std::ofstream out(path);
    out << std::setprecision(17) << "i,t_s,u,v,distance,expected,d0,d1,s0,sweep_score,dy0\n";
    for (const auto& s : samples)
        out << s.i << ',' << s.tS << ',' << s.u << ',' << s.v << ',' << s.distance << ',' << s.expected << ','
            << s.d0 << ',' << s.d1 << ',' << s.s0 << ',' << s.sweepScore << ',' << s.dy0 << '\n';
    checkWritten(out, path);
}

}  // namespace FrameSpatial
