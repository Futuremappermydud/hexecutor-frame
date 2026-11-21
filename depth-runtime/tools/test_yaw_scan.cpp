#include "yaw_scan.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {

using namespace FrameSpatial;

std::vector<MotionYaw::Sample> observations(double early, double late, double scale = 0, bool singleDepth = false) {
    std::vector<MotionYaw::Sample> samples;
    for (int frame = 0; frame < 12; ++frame) {
        const double yaw = frame < 6 ? early : late;
        for (int point = 0; point < 20; ++point) {
            const double measured = singleDepth ? 12.0 : 3.0 + 2.0 * point;
            // Near the centre, the known spherical response is approx 1 / .145 pixels per degree.
            const double expected = measured + (yaw / .145 + scale * measured) / (1.0 - scale / 2.0);
            const double residual = point == 7 ? 5.0 : 0.0;
            // A 0.1 m baseline: 3 px is about 13 m and 41 px about 1 m, so both sides of 2 m are covered.
            const double distance = .1 / (measured * .145 * MotionYaw::kPi / 180.0);
            samples.push_back({frame, double(frame), MotionYaw::kCenterU + measured, MotionYaw::kCenterV,
                               distance, expected + residual, measured, measured, 1, 1});
        }
    }
    return samples;
}

// Matches spread over image height with known yaw and roll corrections (degrees), built from
// the rectified-ray response to small right-camera rotations.
std::vector<MotionYaw::Sample> rolled(double yaw, double roll) {
    constexpr double step = .145 * MotionYaw::kPi / 180.0, toRadians = MotionYaw::kPi / 180.0;
    const double r0 = kYawScanAssumedRollDegrees * toRadians;
    auto rotate = [](double angle, int axis, std::array<double, 3> v) {
        const double c = std::cos(angle), s = std::sin(angle);
        if (axis == 1) return std::array<double, 3>{c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]};
        return std::array<double, 3>{c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]};
    };
    auto horizontal = [](const std::array<double, 3>& r) { return std::atan2(r[0], std::hypot(r[1], r[2])); };
    std::vector<MotionYaw::Sample> samples;
    for (int frame = 0; frame < 12; ++frame)
        for (int point = 0; point < 30; ++point) {
            // Far matches high in the image and near ones low, as with a floor in view.
            const double measured = 3.0 + 1.3 * point, v = 60.0 + 14.0 * point;
            const double u = MotionYaw::kCenterU + 60.0 - 4.0 * frame;
            const double theta = (u - measured - MotionYaw::kCenterU) * step, phi = (v - MotionYaw::kCenterV) * step;
            const std::array<double, 3> ray{std::tan(theta), std::sin(phi), std::cos(phi)};
            // Conjugate the yaw probe by the assumed roll, as the rectification applies it.
            const auto yawed = rotate(r0, 2, rotate(-yaw * toRadians, 1, rotate(-r0, 2, ray)));
            const double shift = (theta - horizontal(yawed)) + (theta - horizontal(rotate(-roll * toRadians, 2, ray)));
            const double distance = .1 / (measured * step);
            samples.push_back({frame, double(frame), u, v, distance, measured + shift / step, measured, measured, 1, 1});
        }
    return samples;
}

struct TestDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("frame-yaw-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TestDirectory() { std::filesystem::create_directory(path); }
    ~TestDirectory() { std::filesystem::remove_all(path); }
};

void testCalibration() {
    TestDirectory scratch;
    const auto rigPath = scratch.path / "rig.json", devicePath = scratch.path / "device.json";
    const std::string axes = R"("plus_x":[1,0,0],"plus_z":[0,0,1])";
    std::string grid = "[0";
    for (int i = 1; i < 25; ++i) grid += ",0";
    grid += ']';
    const std::string base = R"("cx":528,"cy":512,"fx":400,"fy":400,"k1":0,"k2":0,"k3":0)";
    auto camera = [&](const char* name, int x) {
        return std::string(R"({"sourceCamera":")") + name + R"(","width":1056,"height":1024,"extrinsics":{"__units":"mm","position":[)" +
            std::to_string(x) + ",0,0]," + axes + R"(},"intrinsics":[{"cameraModel":"kb","k4":0,)" + base +
            R"(},{"cameraModel":"kb_spline","gridSize":[5,5],"nodeSpacing":[128,128],"sU":)" + grid +
            R"(,"sV":)" + grid + ',' + base + "}]}";
    };
    std::ofstream(rigPath) << R"({"cameras":[)" << camera("slam_right", 0) << ',' << camera("slam_left", -100) << "]}";
    const std::string cad = '{' + axes + R"(,"position":[0,0,0]})";
    std::ofstream(devicePath) << R"({"head":)" << cad << R"(,"cv":{"cad_from_cal":)" << cad << "}}";
    auto calibration = loadYawScanCalibration(rigPath.string(), devicePath.string());
    assert(std::abs(calibration.problem.baseline - .1) < 1e-12);
    assert(calibration.headFromRectified[0] == 1 && calibration.headFromRectified[8] == 1);
    assert(calibration.problem.leftIntr.k[3] == 0 && calibration.problem.inverse.size() == 79);
    HeadPose head;
    head.orientation = {0, 0, 1, 0};
    head.position = {1, 2, 3};
    const auto ref = yawScanReference(calibration, 1234, head);
    assert(ref.when == 1234 && ref.R[0] == -1 && ref.R[4] == -1 && ref.R[8] == 1);
    assert(std::abs(ref.t[0] - 1.1) < 1e-12 && ref.t[1] == 2 && ref.t[2] == 3);
    calibration.problem.references.push_back(ref);
    const auto problemPath = scratch.path / "problem.txt";
    writeYawScanProblem(problemPath, calibration.problem);
    std::ifstream problem(problemPath);
    const std::string serialized{std::istreambuf_iterator<char>(problem), {}};
    assert(serialized.find("references 1\nref\t1234\t") != std::string::npos);
    assert(serialized.find("pairs/pair-0-left.u8\tpairs/pair-0-right.u8\n") != std::string::npos);
}

}  // namespace

int main() {
    constexpr double step = .145 * MotionYaw::kPi / 180.0;
    using State = YawScanFit::State;
    const auto stable = fitYawScan(observations(-.18, -.15), step, 6.0);
    assert(stable.state == State::Ready && stable.samples == 240 && stable.keyframes == 12);
    assert(std::abs(stable.yawDegrees + .165) < .005);
    assert(stable.timeSplitAgrees && stable.depthSplitAgrees && stable.depthDifferenceDegrees);
    assert(std::abs(*stable.depthDifferenceDegrees) < .01);
    assert(stable.nearSide.samples + stable.farSide.samples == 240 && stable.bands.size() == 4);
    size_t banded = 0;
    for (const auto& band : stable.bands) banded += band.samples;
    assert(banded == 240);
    assert(std::abs(stable.splitDifferenceDegrees - .03) < .005);
    assert(fitYawScan(observations(-.22, -.08), step, 6.0).state == State::Unstable);
    const auto oneDepth = fitYawScan(observations(-.22, -.21, 0, true), step, 6.0);
    assert(oneDepth.state == State::NeedNearAndFar && !oneDepth.freeScale && !oneDepth.depthDifferenceDegrees);
    const auto scaled = fitYawScan(observations(-.18, -.18, -.023), step, 6.0);
    assert(scaled.freeScale && std::abs(scaled.freeScale->yawDegrees + .18) < .005);
    assert(std::abs(scaled.freeScale->scaleTerm + .023) < .005);
    assert(std::abs(scaled.yawDegrees - scaled.freeScale->yawDegrees) > .05);
    // A depth-dependent error that both time halves share fails only the near/far check.
    const auto depthBiased = fitYawScan(observations(-.18, -.18, -.05), step, 6.0);
    assert(depthBiased.state == State::Unstable && depthBiased.timeSplitAgrees && !depthBiased.depthSplitAgrees);

    // A roll error shifts disparity with image height; fitting it recovers yaw exactly.
    const auto withRoll = fitYawScan(rolled(-.2, -.25), step, 6.0);
    assert(withRoll.rollFitted && std::abs(withRoll.rollDegrees + .25) < .01);
    assert(std::abs(withRoll.yawDegrees + .2) < .01 && std::abs(withRoll.yawWithoutRollDegrees + .2) > .02);
    assert(withRoll.depthDifferenceDegrees && std::abs(*withRoll.depthDifferenceDegrees) < .01);
    // Rows alone: inject right-minus-left row offsets for known roll, pitch and yaw corrections.
    {
        constexpr double toRadians = MotionYaw::kPi / 180.0, small = .1 * toRadians;
        const double r0 = kYawScanAssumedRollDegrees * toRadians;
        auto rot = [](int axis, double a, std::array<double, 3> v) {
            const double c = std::cos(a), s = std::sin(a);
            if (axis == 0) return std::array<double, 3>{v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]};
            if (axis == 1) return std::array<double, 3>{c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]};
            return std::array<double, 3>{c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]};
        };
        auto vertical = [](const std::array<double, 3>& r) { return std::atan2(r[1], r[2]); };
        auto samples = rolled(-.2, -.25);
        for (size_t i = 0; i < samples.size(); ++i) {
            auto& s = samples[i];
            s.u = 40.0 + double((i * 37) % 432);   // spread over all four quadrants
            const double theta = (s.u - s.d0 - MotionYaw::kCenterU) * step, phi = (s.v - MotionYaw::kCenterV) * step;
            const std::array<double, 3> ray{std::tan(theta), std::sin(phi), std::cos(phi)};
            auto response = [&](int axis) {
                const auto r = axis == 2 ? rot(2, -small, ray) : rot(2, -r0, rot(axis, -small, rot(2, r0, ray)));
                return (vertical(r) - phi) / (step * .1);
            };
            s.dy0 = -(response(2) * -.2 + response(0) * .05 + response(1) * -.03);
        }
        const auto fit = fitYawScan(samples, step, 6.0);
        assert(fit.rows && fit.rows->yawDegrees && fit.rows->calibratedYawDegrees);
        assert(std::abs(fit.rows->rollDegrees + .2) < 1e-4 && std::abs(fit.rows->pitchDegrees - .05) < 1e-4);
        assert(std::abs(*fit.rows->yawDegrees + .03) < 1e-4);
        assert(std::abs(*fit.rows->calibratedYawDegrees - (*fit.rows->yawDegrees + kYawScanRowYawOffsetDegrees)) < 1e-12);
    }

    const auto noRoll = fitYawScan(rolled(-.2, 0), step, 6.0);
    assert(noRoll.rollFitted && std::abs(noRoll.rollDegrees) < .005 && std::abs(noRoll.yawDegrees + .2) < .005);

    auto dense = observations(-.2, -.2);
    MotionYaw::Sample outlier = dense.front();
    outlier.expected += 2.;
    dense.insert(dense.end(), 300, outlier);
    const auto balanced = fitYawScan(dense, step, 6.0);
    assert(balanced.state == State::Ready && std::abs(balanced.yawDegrees + .2) < .03);
    assert(fitYawScan(observations(-.22, -.21), step, 12.0).state == State::MoreSupport);
    assert(fitYawScan({}, step, 6.0).state == State::MoreSupport);
    testCalibration();
    std::cout << "Yaw scan calibration, pose, and fit tests passed\n";
}
