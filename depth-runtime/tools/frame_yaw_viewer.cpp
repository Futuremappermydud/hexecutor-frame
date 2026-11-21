#include "ir_pair_capture.hpp"
#include "yaw_scan.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <csignal>
#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace {

using namespace FrameSpatial;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

constexpr auto kPairSpacing = 200ms;
// SteamVR keeps about 0.2 s of pose history: an older lookup returns the oldest stored pose.
// Pairs whose first lookup comes later than kMaxPoseLookupDelay after the pose time are skipped,
// and each kept pose is looked up again kPoseRequeryDelay later, still inside that history.
constexpr int64_t kMaxPoseLookupDelayNs = 100000000;
constexpr int64_t kRequeryWindowNs = 150000000;
constexpr auto kPoseRequeryDelay = 60ms;
// Poses are also looked up at these exposure offsets (ms before the left timestamp) while they
// are still in the history, so the offset can be chosen offline with fit_yaw_scan.
constexpr std::array<int, 11> kOffsetSweepMs{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
constexpr size_t kFirstFitPairs = 25, kRetryPairs = 20;
// Progress fits while recording use a subset of keyframes; the final fit uses every pair.
constexpr int kLiveKeyframes = 32;
volatile std::sig_atomic_t quitting = 0, finishEarly = 0;

void quit(int) { quitting = 1; }
void finish(int) { finishEarly = 1; }

struct Options {
    fs::path output;
    std::string xrservice = "/persist/xrservice.json", device = "/persist/device_config.json";
    int seconds = 60;
    // Best match quality in the 0-20 ms pose-offset sweep of yaw-1795224642542 (about 12.5 ms).
    double exposureOffsetMs = 12.0;
};

const char* usage =
    "usage: frame_yaw_viewer [--output NEW_DIRECTORY] [--seconds 60]\n"
    "                        [--exposure-offset-ms 12] [--xrservice FILE] [--device-config FILE]\n";

Options options(int argc, char** argv) {
    Options out;
    const auto stamp = std::chrono::system_clock::now().time_since_epoch();
    out.output = fs::path("captures") / ("yaw-" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(stamp).count()));
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (++i == argc) throw std::invalid_argument("missing value for " + name);
        const std::string value = argv[i];
        size_t end = 0;
        if (name == "--output") out.output = value;
        else if (name == "--xrservice") out.xrservice = value;
        else if (name == "--device-config") out.device = value;
        else if (name == "--seconds") {
            out.seconds = std::stoi(value, &end);
            if (end != value.size() || out.seconds < 1 || out.seconds > 300)
                throw std::invalid_argument("--seconds must be in [1, 300]");
        } else if (name == "--exposure-offset-ms") {
            out.exposureOffsetMs = std::stod(value, &end);
            if (end != value.size() || !(out.exposureOffsetMs >= 0 && out.exposureOffsetMs <= 100))
                throw std::invalid_argument("--exposure-offset-ms must be in [0, 100]");
        } else throw std::invalid_argument("unknown argument " + name);
    }
    if (out.output.empty()) throw std::invalid_argument("output path is empty");
    out.output = fs::absolute(out.output);
    return out;
}

void writeRaw(const fs::path& path, const std::vector<uint8_t>& pixels) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(pixels.data()), std::streamsize(pixels.size()));
    file.close();
    if (!file) throw std::runtime_error("could not write " + path.string());
}

std::string degrees(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << std::showpos << value << " DEG";
    return out.str();
}

// HUD line with the calibrated row yaw when the matches give one.
std::string savedPairs(const YawScanFit& fit, size_t pairs) {
    const std::string saved = std::to_string(pairs) + " PAIRS SAVED";
    if (!fit.rows || !fit.rows->calibratedYawDegrees) return saved;
    return "ROWS " + degrees(*fit.rows->calibratedYawDegrees) + " / " + saved;
}

std::string json(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string json(const std::optional<double>& value) { return value ? json(*value) : "null"; }

std::string json(const YawScanFit::Band& band) {
    std::ostringstream out;
    out << "{\"min_distance_m\": " << json(band.minDistance) << ", \"max_distance_m\": " << json(band.maxDistance)
        << ", \"samples\": " << band.samples << ", \"keyframes\": " << band.keyframes
        << ", \"yaw_correction_degrees\": " << json(band.yawDegrees) << '}';
    return out.str();
}

struct Attempt {
    MotionYaw::Problem problem;
    std::vector<MotionYaw::Sample> samples;
    YawScanFit fit;
    bool full = false;  // every recorded pair was a keyframe candidate
};

int64_t monotonicNs() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return int64_t(now.tv_sec) * 1000000000ll + now.tv_nsec;
}

// A second OpenXR lookup of a recorded pose time, made while it is still inside the runtime's
// pose history. A difference means the pose for a past time changes after it is first returned.
// Lookups later than kRequeryWindowNs after the pose time are logged but not summarized.
struct PoseRequery {
    struct Stage {
        const char* name;
        std::vector<std::optional<HeadPose>> poses;
        std::vector<double> positionMm, rotationDeg;
        size_t late = 0;
    };
    std::vector<HeadPose> originals;
    std::vector<int64_t> times;
    Stage afterDelay{"after_60ms", {}, {}, {}};
    size_t skippedLatePairs = 0;
    std::vector<double> lookupDelayMs;
    std::vector<std::array<std::optional<HeadPose>, kOffsetSweepMs.size()>> offsets;
};

// Linux thermal zones, sampled through the recording. Yaw drift with temperature would show up
// as scans that differ by when they were taken rather than by how they were fitted.
class ThermalLog {
public:
    static constexpr auto kInterval = 5s;
    // Summarized in result.json; every zone is in thermal.csv.
    static constexpr std::array<const char*, 6> kSummary{
        "camera0-thermal", "camera1-thermal", "aoss0-thermal", "cpuss0-thermal", "ddr-thermal", "max1720x_bat_7-36"};

    void open(const fs::path& path) {
        std::error_code error;
        for (const auto& entry : fs::directory_iterator("/sys/class/thermal", error)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("thermal_zone", 0) != 0) continue;
            std::ifstream type(entry.path() / "type");
            std::string label;
            if (std::getline(type, label) && !label.empty()) zones_.push_back({label, entry.path() / "temp", {}});
        }
        std::sort(zones_.begin(), zones_.end(), [](const Zone& a, const Zone& b) { return a.label < b.label; });
        out_.open(path);
        out_ << "elapsed_s";
        for (const auto& zone : zones_) out_ << ',' << zone.label;
        out_ << '\n';
        if (!out_) throw std::runtime_error("could not open thermal log");
        start_ = Clock::now();
    }

    void sampleIfDue() {
        if (Clock::now() >= next_) sample();
    }

    void sample() {
        next_ = Clock::now() + kInterval;
        out_ << std::fixed << std::setprecision(1)
             << std::chrono::duration<double>(Clock::now() - start_).count();
        for (auto& zone : zones_) {
            std::ifstream in(zone.temp);
            long milli = 0;
            out_ << ',';
            if (in >> milli) {
                zone.celsius.push_back(double(milli) / 1000.0);
                out_ << zone.celsius.back();
            }
        }
        out_ << std::defaultfloat << '\n';
    }

    // The summary covers the recording; later samples (during the final fit) stay in the CSV.
    void endRecording() {
        if (recorded_) return;
        sample();
        recorded_ = true;
        for (auto& zone : zones_) zone.recordedSamples = zone.celsius.size();
    }

    void close() {
        out_.close();
        if (!out_) throw std::runtime_error("could not write thermal log");
    }

    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(4) << '{';
        bool first = true;
        for (const char* label : kSummary)
            for (const auto& zone : zones_) {
                const size_t count = recorded_ ? zone.recordedSamples : zone.celsius.size();
                if (zone.label != label || count == 0) continue;
                const auto end = zone.celsius.begin() + std::ptrdiff_t(count);
                const auto [low, high] = std::minmax_element(zone.celsius.begin(), end);
                out << (first ? "" : ", ") << '"' << label << "\": {\"start_c\": " << zone.celsius.front()
                    << ", \"end_c\": " << *(end - 1) << ", \"min_c\": " << *low << ", \"max_c\": " << *high
                    << '}';
                first = false;
            }
        out << '}';
        return out.str();
    }

private:
    struct Zone {
        std::string label;
        fs::path temp;
        std::vector<double> celsius;
        size_t recordedSamples = 0;
    };
    std::vector<Zone> zones_;
    bool recorded_ = false;
    std::ofstream out_;
    Clock::time_point start_, next_;
};

double rotationDegrees(const HeadPose& a, const HeadPose& b) {
    double dot = 0;
    for (int i = 0; i < 4; ++i) dot += a.orientation[i] * b.orientation[i];
    return 2.0 * std::acos(std::min(1.0, std::abs(dot))) * 180.0 / MotionYaw::kPi;
}

double positionMm(const HeadPose& a, const HeadPose& b) {
    return 1000.0 * std::hypot(a.position[0] - b.position[0], a.position[1] - b.position[1],
                               a.position[2] - b.position[2]);
}

std::string json(const PoseRequery::Stage& stage, size_t recorded) {
    std::ostringstream out;
    out << "{\"located\": " << stage.positionMm.size() << ", \"recorded\": " << recorded
        << ", \"outside_history_window\": " << stage.late;
    auto stats = [&out](const char* name, std::vector<double> values) {
        std::optional<double> median, max;
        if (!values.empty()) {
            std::sort(values.begin(), values.end());
            median = values[values.size() / 2];
            max = values.back();
        }
        out << ", \"median_" << name << "\": " << json(median) << ", \"max_" << name << "\": " << json(max);
    };
    stats("position_mm", stage.positionMm);
    stats("rotation_degrees", stage.rotationDeg);
    out << '}';
    return out.str();
}

enum class Stop { None, Finish, TrackingChanged, Exit };

class Scan {
public:
    Scan(const Options& options, const YawScanCalibration& calibration, const HandViewerXr& viewer)
        : options_(options), calibration_(calibration), viewer_(viewer) {
        hud_.hud = {"IR YAW SCAN", "WAITING FOR PASSTHROUGH", ""};
    }

    ~Scan() { join(); }

    void join() {
        stop(Stop::Exit);
        if (worker_.joinable()) worker_.join();
    }

    void start() { worker_ = std::thread(&Scan::run, this); }
    void visible(bool value) { visible_.store(value); }
    void stop(Stop reason) {
        // A graceful finish can still be interrupted by tracking loss or viewer exit.
        auto current = stop_.load();
        while (int(current) < int(reason) && !stop_.compare_exchange_weak(current, reason)) {}
    }

    HandScene hud() const {
        std::lock_guard<std::mutex> lock(hudMutex_);
        return hud_;
    }

    bool failed() const { return failed_.load(); }

private:
    void hud(std::string title, std::string detail, std::string hint,
             std::array<uint8_t, 3> color = {110, 215, 255}) {
        std::lock_guard<std::mutex> lock(hudMutex_);
        hud_.hud = {std::move(title), std::move(detail), std::move(hint)};
        hud_.hud_color = color;
    }

    Attempt fit(const MotionYaw::Problem& problem) const {
        auto loader = [this](size_t index, bool right) {
            const auto path = options_.output / "pairs" / yawScanImageName(index, right);
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            constexpr size_t bytes = size_t(IrPairCapture::Width) * IrPairCapture::Height;
            if (!in || in.tellg() != std::streamoff(bytes))
                throw std::runtime_error("invalid recorded IR image " + path.string());
            std::vector<uint8_t> pixels(bytes);
            in.seekg(0);
            in.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(bytes));
            if (!in) throw std::runtime_error("could not read " + path.string());
            return pixels;
        };
        auto result = MotionYaw::runMotionYaw(problem, loader, [this] { return stop_.load() >= Stop::TrackingChanged; });
        const double midpoint = double(problem.references.back().when - problem.references.front().when) * .5e-9;
        const auto fitted = fitYawScan(result.samples, problem.kHi, midpoint);
        return {problem, std::move(result.samples), fitted,
                size_t(problem.keyframes) >= problem.references.size()};
    }

    static const char* unstableReason(const YawScanFit& fit) {
        if (!fit.depthSplitAgrees) return "NEAR AND FAR YAW DISAGREE";
        return "YAW CHANGED DURING THE SCAN";
    }

    void save(const MotionYaw::Problem& problem, const Attempt& attempt, const PoseRequery& poses,
              const ThermalLog& thermal, const char* status) const {
        const auto& root = options_.output;
        writeYawScanProblem(root / "problem.txt", problem);
        if (!attempt.problem.references.empty()) {
            if (attempt.problem.references.size() != problem.references.size() ||
                attempt.problem.keyframes != problem.keyframes)
                writeYawScanProblem(root / "fit-problem.txt", attempt.problem);
            writeYawScanSamples(root / "samples.csv", attempt.samples);
        }
        // Replaying these with the same keyframes shows how much the yaw depends on the pose lookup.
        for (const auto* stage : {&poses.afterDelay}) {
            if (stage->poses.size() != problem.references.size() || problem.references.empty() ||
                std::any_of(stage->poses.begin(), stage->poses.end(), [](const auto& pose) { return !pose; }))
                continue;
            auto requeried = problem;
            for (size_t i = 0; i < requeried.references.size(); ++i)
                requeried.references[i] = yawScanReference(calibration_, poses.times[i], *stage->poses[i]);
            writeYawScanProblem(root / (std::string("problem-requery-") + stage->name + ".txt"), requeried);
        }
        std::vector<int> offsetFiles;
        for (size_t k = 0; k < kOffsetSweepMs.size(); ++k) {
            if (poses.offsets.size() != problem.references.size() || problem.references.empty() ||
                std::any_of(poses.offsets.begin(), poses.offsets.end(), [k](const auto& row) { return !row[k]; }))
                continue;
            auto shifted = problem;
            const int64_t base = std::llround(options_.exposureOffsetMs * 1e6);
            for (size_t i = 0; i < shifted.references.size(); ++i) {
                const int64_t when = poses.times[i] + base - int64_t(kOffsetSweepMs[k]) * 1000000;
                shifted.references[i] = yawScanReference(calibration_, when, *poses.offsets[i][k]);
            }
            writeYawScanProblem(root / ("problem-offset-" + std::to_string(kOffsetSweepMs[k]) + "ms.txt"), shifted);
            offsetFiles.push_back(kOffsetSweepMs[k]);
        }
        const auto& fit = attempt.fit;
        std::ofstream report(root / "result.json");
        report << std::setprecision(17) << "{\n  \"status\": \"" << status << "\",\n"
               << "  \"yaw_method\": \"fixed_motion_scale_keyframe_balanced\",\n"
               << "  \"recorded_pairs\": " << problem.references.size() << ",\n"
               << "  \"fit_pairs\": " << attempt.problem.references.size() << ",\n"
               << "  \"fit_keyframe_target\": " << attempt.problem.keyframes << ",\n"
               << "  \"exposure_offset_ms\": " << options_.exposureOffsetMs << ",\n"
               << "  \"samples\": " << fit.samples << ",\n"
               << "  \"keyframes\": " << fit.keyframes << ",\n"
               << "  \"matches_beyond_5_m\": " << fit.farSamples;
        if (fit.state != YawScanFit::State::MoreSupport) {
            report << ",\n  \"yaw_correction_degrees\": " << fit.yawDegrees
                   << ",\n  \"pose_scale_term\": 0"
                   << ",\n  \"roll_fitted\": " << (fit.rollFitted ? "true" : "false")
                   << ",\n  \"assumed_roll_degrees\": " << kYawScanAssumedRollDegrees
                   << ",\n  \"assumed_pitch_degrees\": " << kYawScanAssumedPitchDegrees
                   << ",\n  \"roll_correction_degrees\": " << fit.rollDegrees
                   << ",\n  \"yaw_without_roll_degrees\": " << fit.yawWithoutRollDegrees
                   << ",\n  \"residual_scale_degrees\": " << fit.residualScaleDegrees
                   << ",\n  \"agreement_limit_degrees\": " << kYawScanAgreementDegrees
                   << ",\n  \"time_split_difference_degrees\": " << fit.splitDifferenceDegrees
                   << ",\n  \"time_split_agrees\": " << (fit.timeSplitAgrees ? "true" : "false")
                   << ",\n  \"depth_split_distance_m\": " << kYawScanSplitDistance
                   << ",\n  \"near_side\": " << json(fit.nearSide)
                   << ",\n  \"far_side\": " << json(fit.farSide)
                   << ",\n  \"depth_split_difference_degrees\": " << json(fit.depthDifferenceDegrees)
                   << ",\n  \"depth_split_agrees\": " << (fit.depthSplitAgrees ? "true" : "false")
                   << ",\n  \"depth_bands\": [";
            for (size_t i = 0; i < fit.bands.size(); ++i)
                report << (i ? ",\n    " : "\n    ") << json(fit.bands[i]);
            report << "\n  ]";
        }
        if (fit.rows)
            report << ",\n  \"stereo_rows\": {\"samples\": " << fit.rows->samples << ", \"keyframes\": "
                   << fit.rows->keyframes << ", \"roll_correction_degrees\": " << json(fit.rows->rollDegrees)
                   << ", \"pitch_correction_degrees\": " << json(fit.rows->pitchDegrees)
                   << ", \"residual_px\": " << json(fit.rows->residualPx)
                   << ", \"yaw_correction_degrees\": " << json(fit.rows->yawDegrees)
                   << ", \"yaw_offset_degrees\": " << kYawScanRowYawOffsetDegrees
                   << ", \"calibrated_yaw_correction_degrees\": " << json(fit.rows->calibratedYawDegrees) << '}';
        report << ",\n  \"exposure_offset_problems_ms\": [";
        for (size_t i = 0; i < offsetFiles.size(); ++i) report << (i ? ", " : "") << offsetFiles[i];
        report << ']';
        if (fit.freeScale)
            report << ",\n  \"free_scale_yaw_correction_degrees\": " << fit.freeScale->yawDegrees
                   << ",\n  \"free_scale_pose_scale_term\": " << fit.freeScale->scaleTerm;
        std::optional<double> medianDelay, maxDelay;
        if (!poses.lookupDelayMs.empty()) {
            auto delays = poses.lookupDelayMs;
            std::sort(delays.begin(), delays.end());
            medianDelay = delays[delays.size() / 2];
            maxDelay = delays.back();
        }
        report << ",\n  \"pose_lookup\": {\"median_delay_ms\": " << json(medianDelay)
               << ", \"max_delay_ms\": " << json(maxDelay)
               << ", \"max_allowed_delay_ms\": " << kMaxPoseLookupDelayNs / 1e6
               << ", \"skipped_late_pairs\": " << poses.skippedLatePairs << '}'
               << ",\n  \"pose_requery\": {\n    \"" << poses.afterDelay.name << "\": "
               << json(poses.afterDelay, problem.references.size()) << "\n  }"
               << ",\n  \"thermal\": " << thermal.json() << "\n}\n";
        report.close();
        if (!report) throw std::runtime_error("could not write scan result");
    }

    void requery(PoseRequery& poses, PoseRequery::Stage& stage, size_t index, std::ofstream& log) const {
        HeadPose pose;
        const int64_t age = monotonicNs() - poses.times[index];
        const bool located = viewer_.headPoseAt(uint64_t(poses.times[index]), pose, true);
        if (stage.poses.size() <= index) stage.poses.resize(index + 1);
        const auto& original = poses.originals[index];
        const bool late = age > kRequeryWindowNs;
        log << index << ',' << stage.name << ',' << poses.times[index] << ',' << age << ',' << (located ? 1 : 0);
        if (late) {
            ++stage.late;
            log << ",,,,,,,,,\n";
            return;
        }
        if (located) {
            stage.poses[index] = pose;
            stage.positionMm.push_back(positionMm(original, pose));
            stage.rotationDeg.push_back(rotationDegrees(original, pose));
            log << ',' << stage.positionMm.back() << ',' << stage.rotationDeg.back();
            for (double v : pose.orientation) log << ',' << v;
            for (double v : pose.position) log << ',' << v;
        } else {
            log << ",,,,,,,,,";
        }
        log << '\n';
    }

    void record() {
        IrPairCapture camera;
        hud("IR YAW SCAN", "CONNECTING TO IR CAMERAS", "");
        if (!camera.initialize()) throw std::runtime_error(camera.lastError());

        const auto& root = options_.output;
        fs::create_directories(root.parent_path());
        if (!fs::create_directory(root)) throw std::runtime_error("output already exists: " + root.string());
        fs::create_directory(root / "pairs");
        std::ofstream timing(root / "pairs.csv");
        timing << "pair,left_sequence,right_sequence,left_raw_ns,right_raw_ns,left_monotonic_ns,pose_time_ns,"
                  "pose_lookup_delay_ns\n";
        if (!timing) throw std::runtime_error("could not open pair metadata");
        std::ofstream requeryLog(root / "poses-requery.csv");
        requeryLog << std::setprecision(17)
                   << "pair,stage,pose_time_ns,lookup_age_ns,located,position_delta_mm,rotation_delta_degrees,qx,qy,qz,qw,x,y,z\n";
        if (!requeryLog) throw std::runtime_error("could not open pose requery log");
        ThermalLog thermal;
        thermal.open(root / "thermal.csv");
        thermal.sample();
        std::cout << "[Yaw] Recording to " << root << std::endl;

        auto problem = calibration_.problem;
        const auto deadline = Clock::now() + std::chrono::seconds(options_.seconds);
        const int64_t offsetNs = std::llround(options_.exposureOffsetMs * 1e6);
        std::future<Attempt> pending;
        Attempt last;
        PoseRequery poses;
        std::deque<std::pair<size_t, Clock::time_point>> requeryDue;
        size_t attemptedPairs = 0;
        bool fitFailed = false, finalStarted = false;
        auto nextPair = Clock::now();
        IrPairCapture::Pair pair;
        while (true) {
            const Stop stop = stop_.load();
            const bool recording = stop == Stop::None && Clock::now() < deadline;
            const bool canFit = stop < Stop::TrackingChanged;
            if (recording) thermal.sampleIfDue();
            else thermal.endRecording();
            while (!requeryDue.empty() && requeryDue.front().second <= Clock::now()) {
                requery(poses, poses.afterDelay, requeryDue.front().first, requeryLog);
                requeryDue.pop_front();
            }
            if (pending.valid() && pending.wait_for(0s) == std::future_status::ready) {
                try {
                    auto finished = pending.get();
                    if (canFit) {
                        last = std::move(finished);
                        fitFailed = false;
                        std::cout << "[Yaw] " << last.problem.references.size() << " fit pairs, "
                                  << last.fit.samples << " matches from " << last.fit.keyframes
                                  << " keyframes" << (last.full ? " (final)" : "") << std::endl;
                    }
                } catch (const std::exception& error) {
                    fitFailed = true;
                    std::cerr << "[Yaw] Fit failed: " << error.what() << std::endl;
                }
            }
            const size_t count = problem.references.size();
            if (canFit && !pending.valid()) {
                auto snapshot = problem;
                if (!recording && !finalStarted && count >= 8) {
                    finalStarted = true;
                    snapshot.keyframes = int(count);
                    pending = std::async(std::launch::async, [this, snapshot] { return fit(snapshot); });
                } else if (recording && count >= (attemptedPairs ? attemptedPairs + kRetryPairs : kFirstFitPairs)) {
                    attemptedPairs = count;
                    snapshot.keyframes = kLiveKeyframes;
                    pending = std::async(std::launch::async, [this, snapshot] { return fit(snapshot); });
                }
            }
            if (!recording && !pending.valid()) break;

            const std::string progress = std::to_string(count) + " IR PAIRS / " +
                std::to_string(last.fit.samples) + " MATCHES";
            if (!recording)
                hud("RECORDING STOPPED / FITTING YAW", progress, "CHECKING EVERY RECORDED PAIR");
            else if (last.fit.state == YawScanFit::State::Ready)
                hud("FIT FOUND / STILL RECORDING", progress, "KEEP MOVING TO CHECK THE FIT", {125, 240, 172});
            else if (last.fit.state == YawScanFit::State::NeedNearAndFar)
                hud("NEED NEAR AND FAR SURFACES", progress, "VIEW SURFACES CLOSER AND FARTHER THAN 2 M",
                    {255, 196, 102});
            else if (last.fit.state == YawScanFit::State::Unstable)
                hud("YAW FIT UNSTABLE", progress, unstableReason(last.fit), {255, 196, 102});
            else if (pending.valid())
                hud("RECORDING / FITTING YAW", progress, "KEEP MOVING SIDEWAYS SLOWLY");
            else
                hud("RECORDING IR SCENE", progress, "MOVE SIDEWAYS AND CHANGE DISTANCE");

            if (!recording || !visible_.load() || Clock::now() < nextPair) {
                std::this_thread::sleep_for(25ms);
                continue;
            }
            nextPair = Clock::now() + kPairSpacing;
            if (!camera.grab(pair, 100)) {
                hud("WAITING FOR IR PAIRS", progress, "KEEP THE SCENE IN VIEW", {255, 196, 102});
                continue;
            }
            if (pair.leftMonotonicNs <= offsetNs) continue;
            const int64_t when = pair.leftMonotonicNs - offsetNs;
            HeadPose head;
            if (!visible_.load() || stop_.load() != Stop::None) continue;
            const int64_t lookupDelay = monotonicNs() - when;
            if (lookupDelay > kMaxPoseLookupDelayNs) {
                ++poses.skippedLatePairs;
                std::cerr << "[Yaw] Skipped a pair: pose lookup " << lookupDelay / 1000000 << " ms after capture"
                          << std::endl;
                continue;
            }
            if (!viewer_.headPoseAt(uint64_t(when), head, true)) {
                hud("WAITING FOR TRACKED HEAD POSE", progress, "LOOK AT A WELL LIT AREA", {255, 196, 102});
                continue;
            }
            if (!problem.references.empty() && when <= problem.references.back().when) continue;

            writeRaw(root / "pairs" / yawScanImageName(count, false), pair.left);
            writeRaw(root / "pairs" / yawScanImageName(count, true), pair.right);
            timing << count << ',' << pair.leftSeq << ',' << pair.rightSeq << ',' << pair.leftTsNs << ','
                   << pair.rightTsNs << ',' << pair.leftMonotonicNs << ',' << when << ',' << lookupDelay << std::endl;
            if (!timing) throw std::runtime_error("could not write pair metadata");
            std::array<std::optional<HeadPose>, kOffsetSweepMs.size()> swept;
            for (size_t k = 0; k < kOffsetSweepMs.size(); ++k) {
                HeadPose shifted;
                if (viewer_.headPoseAt(uint64_t(pair.leftMonotonicNs - int64_t(kOffsetSweepMs[k]) * 1000000),
                                       shifted, true))
                    swept[k] = shifted;
            }
            problem.references.push_back(yawScanReference(calibration_, when, head));
            poses.offsets.push_back(swept);
            poses.originals.push_back(head);
            poses.times.push_back(when);
            poses.lookupDelayMs.push_back(lookupDelay / 1e6);
            requeryDue.emplace_back(count, Clock::now() + kPoseRequeryDelay);
        }
        timing.close();
        if (!timing) throw std::runtime_error("could not close pair metadata");
        thermal.sample();
        thermal.close();
        for (const auto& due : requeryDue) {
            std::this_thread::sleep_until(due.second);
            requery(poses, poses.afterDelay, due.first, requeryLog);
        }
        requeryLog.close();
        if (!requeryLog) throw std::runtime_error("could not write pose requery log");
        // problem.txt replays the final fit: every recorded pair is a keyframe candidate.
        problem.keyframes = std::max(1, int(problem.references.size()));

        if (stop_.load() == Stop::TrackingChanged) {
            save(problem, {}, poses, thermal, "tracking_changed");
            hud("TRACKING ORIGIN CHANGED", "RECORDING SAVED", "RELAUNCH TO START A NEW SCAN", {255, 196, 102});
            std::cout << "[Yaw] Tracking origin changed. Capture saved to " << root << std::endl;
        } else if (last.fit.state == YawScanFit::State::Ready && last.full &&
                   last.problem.references.size() == problem.references.size() && !fitFailed) {
            save(problem, last, poses, thermal, "provisional_fit");
            hud("YAW ESTIMATE READY", degrees(last.fit.yawDegrees) + " / PROVISIONAL",
                savedPairs(last.fit, problem.references.size()), {125, 240, 172});
            std::cout << "[Yaw] Provisional correction " << degrees(last.fit.yawDegrees)
                      << ", time-split difference " << last.fit.splitDifferenceDegrees
                      << " deg, far-minus-near " << *last.fit.depthDifferenceDegrees
                      << " deg. Capture saved to " << root << std::endl;
        } else {
            using State = YawScanFit::State;
            const bool finalFit = last.full && last.problem.references.size() == problem.references.size();
            const State state = finalFit ? last.fit.state : State::MoreSupport;
            const char* status = stop_.load() == Stop::Exit && !finalFit ? "interrupted" :
                fitFailed ? "fit_error" : state == State::Unstable ? "unstable_fit" :
                state == State::NeedNearAndFar ? "need_near_and_far" : "insufficient_support";
            save(problem, last, poses, thermal, status);
            const char* reason = fitFailed ? "FIT FAILED / SEE VIEWER LOG" :
                state == State::Unstable ? unstableReason(last.fit) :
                state == State::NeedNearAndFar ? "NEED MATCHES NEARER AND FARTHER THAN 2 M" :
                "NEED MORE TRACKED MATCHES";
            hud("SCAN SAVED / NO STABLE YAW", reason,
                finalFit ? savedPairs(last.fit, problem.references.size())
                         : std::to_string(problem.references.size()) + " PAIRS SAVED",
                {255, 196, 102});
            std::cout << "[Yaw] " << status << ". Capture saved to " << root << std::endl;
        }
    }

    void run() {
        try {
            record();
        } catch (const std::exception& error) {
            failed_.store(true);
            std::cerr << "[Yaw] " << error.what() << std::endl;
            hud("YAW SCAN ERROR", "SEE VIEWER LOG FOR DETAILS", "", {255, 152, 128});
        }
    }

    const Options& options_;
    const YawScanCalibration& calibration_;
    const HandViewerXr& viewer_;
    std::atomic<Stop> stop_{Stop::None};
    std::atomic<bool> visible_{false}, failed_{false};
    mutable std::mutex hudMutex_;
    HandScene hud_;
    std::thread worker_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << usage;
        return 0;
    }
    try {
        const auto config = options(argc, argv);
        const auto calibration = loadYawScanCalibration(config.xrservice, config.device);
        std::signal(SIGINT, quit);
        std::signal(SIGTERM, quit);
        std::signal(SIGUSR1, finish);
        HandViewerXr viewer;
        viewer.initialize("Frame IR Yaw Viewer");
        if (!viewer.capturePoseSupported())
            throw std::runtime_error("OpenXR cannot locate poses at IR capture times");
        Scan scan(config, calibration, viewer);
        bool started = false;
        uint64_t space = 0;
        while (!quitting && viewer.pollEvents()) {
            const bool visible = viewer.active();
            scan.visible(visible);
            if (!started && visible) {
                space = viewer.referenceSpaceChanges();
                scan.start();
                started = true;
            }
            if (started && space != viewer.referenceSpaceChanges()) scan.stop(Stop::TrackingChanged);
            if (finishEarly) scan.stop(Stop::Finish);
            viewer.render(scan.hud());
        }
        scan.join();
        return scan.failed() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "[Yaw] " << error.what() << '\n' << usage;
        return 1;
    }
}
