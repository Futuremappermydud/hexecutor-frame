#include "ir_dot_refine.hpp"
#include "calibration.hpp"
#include "model_depth_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace FrameSpatial {
namespace {
using Clock = std::chrono::steady_clock;

// A dot-confirmed surface is opaque, so RGB depth this far behind it is the model's
// dome. Beyond this it is more likely a real opening than model error.
constexpr float kMaxBehindPlane = 0.65f;
// Flat RGB this far in front of the plane is treated as dome (measured up to 450 mm).
// Anything nearer the camera than this is kept as foreground.
constexpr float kMaxDomeInFront = 0.50f;
constexpr int64_t kPlaneMaxAgeMs = 2000;
constexpr uint32_t kMinObservationPoints = 80;
constexpr int kCaptureFailuresBeforeReopen = 5;

std::string setting(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v && *v ? v : fallback;
}

bool localFlat(const std::vector<uint8_t>& luma, int x, int y) {
    uint8_t lo = 255, hi = 0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= RECT_WIDTH || yy >= RECT_HEIGHT) continue;
            const uint8_t v = luma[size_t(yy) * RECT_WIDTH + xx];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    return int(hi) - int(lo) < 12;
}

void rotate(const TsdfPose& pose, const float in[3], float out[3]) {
    for (int r = 0; r < 3; ++r)
        out[r] = pose.rotation[r][0] * in[0] + pose.rotation[r][1] * in[1] + pose.rotation[r][2] * in[2];
}

float rotationDegrees(const TsdfPose& a, const TsdfPose& b) {
    float trace = 0.f;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) trace += a.rotation[r][c] * b.rotation[r][c];
    return std::acos(std::clamp((trace - 1.f) * 0.5f, -1.f, 1.f)) * 180.f / 3.14159265f;
}

// Footprint cell of a plane point given as an offset from the centroid.
uint8_t footprintAt(const IrDotRefine::WorldPlane& wp, float dx, float dy, float dz) {
    const float u = wp.axisU[0] * dx + wp.axisU[1] * dy + wp.axisU[2] * dz;
    const float v = wp.axisV[0] * dx + wp.axisV[1] * dy + wp.axisV[2] * dz;
    const int iu = int(std::floor(u / kIrFootprintCell)) + kIrFootprintSize / 2;
    const int iv = int(std::floor(v / kIrFootprintCell)) + kIrFootprintSize / 2;
    if (iu < 0 || iv < 0 || iu >= kIrFootprintSize || iv >= kIrFootprintSize) return kIrCellUnobserved;
    return wp.footprint[size_t(iv) * kIrFootprintSize + iu];
}
}  // namespace

void IrDotRefine::WorldPlane::fillDiskFootprint() {
    footprint.assign(size_t(kIrFootprintSize) * kIrFootprintSize, kIrCellUnobserved);
    for (int v = 0; v < kIrFootprintSize; ++v)
        for (int u = 0; u < kIrFootprintSize; ++u) {
            const float du = (u - kIrFootprintSize / 2 + 0.5f) * kIrFootprintCell;
            const float dv = (v - kIrFootprintSize / 2 + 0.5f) * kIrFootprintCell;
            if (du * du + dv * dv <= radius * radius)
                footprint[size_t(v) * kIrFootprintSize + u] = kIrCellSupported;
        }
}

IrDotRefine::IrDotRefine() : bridge_(&running_) {}

IrDotRefine::~IrDotRefine() { stop(); }

void IrDotRefine::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
    bridge_.stop();
    active_ = false;
}

void IrDotRefine::waitUntil(Clock::time_point deadline) const {
    while (running_.load() && Clock::now() < deadline)
        std::this_thread::sleep_for(std::min(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - Clock::now()) + std::chrono::milliseconds(1), std::chrono::milliseconds(20)));
}

void IrDotRefine::turnEmitterOff() {
    if (pulseOffCmd_.empty()) return;
    IrPulseProcess off;
    if (!off.start(pulseOffCmd_)) return;
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    while (off.poll() == IrPulseProcess::State::Running && Clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

IrDotRefine::WorldPlane IrDotRefine::worldPlane() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return worldPlane_;
}

bool IrDotRefine::initialize() {
    if (setting("FRAME_IR_DOT", "0") != "1") return true;
    // The bridge treats the prior as spherical disparity at 0.29 deg/px.
    if (setting("FRAME_DEPTH_SPHERICAL", "0") != "1" ||
        std::abs(std::atof(setting("FRAME_DEPTH_SPHERICAL_DEG", "0.29").c_str()) - 0.29) > 1e-4) {
        error_ = "FRAME_IR_DOT needs FRAME_DEPTH_SPHERICAL=1 at 0.29 deg";
        return false;
    }
    // Sessions measured -0.12 to -0.41 deg, and error past half a dot period locks
    // the unwrap one period off (about 0.6 m), so a built-in value is not safe.
    if (setting("IR_YAW_DEG", "").empty()) {
        error_ = "FRAME_IR_DOT needs a measured session IR_YAW_DEG";
        return false;
    }
    setenv("IR_PITCH", setting("IR_PITCH", "0.25").c_str(), 0);
    setenv("IR_ROLL_DEG", setting("IR_ROLL_DEG", "0.4").c_str(), 0);
    periodMs_ = std::max(200, std::atoi(setting("FRAME_IR_DOT_PERIOD_MS", "1500").c_str()));
    minDotRatio_ = float(std::atof(setting("FRAME_IR_DOT_MIN_RATIO", "1.04").c_str()));
    maxPriorDegrees_ = float(std::atof(setting("FRAME_IR_DOT_MAX_PRIOR_DEG", "6").c_str()));
    exposureOffsetNs_ = int(1e6 * std::atof(setting("FRAME_IR_EXPOSURE_OFFSET_MS", "8").c_str()));
    if (setting("FRAME_IR_DOT_PULSE", "1") == "1") {
        const std::string secondsText = setting("FRAME_IR_DOT_PULSE_S", "0.12");
        pulseSeconds_ = float(std::atof(secondsText.c_str()));
        pulseCmd_ = setting("FRAME_IR_DOT_PULSE_CMD", "");
        if (pulseCmd_.empty()) {
            const std::string home = setting("HOME", "/home/steamos");
            const std::string py = setting("FRAME_BELICE_PYTHON", home + "/Documents/belice/.venv/bin/python");
            const std::string flash = setting("FRAME_BELICE_FLASH", home + "/Documents/belice/belice_flash.py");
            // From the belice tree so HID discovery matches the offline pulse script.
            // stderr stays visible so a missing HID device shows in the viewer log.
            const std::string run = "cd " + home + "/Documents/belice && exec " + py + " " + flash +
                                    " emitter --seconds ";
            pulseCmd_ = run + secondsText + " >/dev/null";
            pulseOffCmd_ = run + "0 >/dev/null";
        }
    }
    if (!capture_.initialize()) {
        error_ = capture_.lastError();
        std::cerr << "[IrDot] capture: " << error_ << std::endl;
        return false;
    }
    running_.store(true);
    if (!bridge_.start()) {
        error_ = bridge_.lastError();
        std::cerr << "[IrDot] bridge: " << error_ << std::endl;
        stop();
        return false;
    }
    active_ = true;
    thread_ = std::thread([this] { worker(); });
    std::cout << "[IrDot] RGB prior + IR dots, period " << periodMs_ << " ms, yaw "
              << setting("IR_YAW_DEG", "") << " deg, pulse=" << (pulseCmd_.empty() ? "off" : "on")
              << std::endl;
    return true;
}

void IrDotRefine::submit(const std::vector<float>& rgbDisparity, const std::vector<uint8_t>& rgbUncertainty,
                         const TsdfPose& rgbPose) {
    if (!active_ || !running_.load()) return;
    if (rgbDisparity.size() != size_t(MODEL_WIDTH) * MODEL_HEIGHT ||
        rgbUncertainty.size() != rgbDisparity.size())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    pendingDisp_ = rgbDisparity;
    pendingUnc_ = rgbUncertainty;
    pendingPose_ = rgbPose;
    havePending_ = true;
}

bool IrDotRefine::takeObservation(Observation& observation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!haveObservation_) return false;
    observation = std::move(observation_);
    observation_ = {};
    haveObservation_ = false;
    return true;
}

void IrDotRefine::apply(std::vector<float>& rgbDepth, const std::vector<uint8_t>& rgbLuma,
                        const TsdfPose* currentPose) {
    if (!active_ || !currentPose || rgbDepth.size() != size_t(RECT_WIDTH) * RECT_HEIGHT ||
        rgbLuma.size() != rgbDepth.size())
        return;
    WorldPlane wp;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!worldPlane_.valid) return;
        wp = worldPlane_;
    }
    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - wp.timestamp).count();
    if (age > kPlaneMaxAgeMs || wp.footprint.size() != size_t(kIrFootprintSize) * kIrFootprintSize) return;

    const auto& R = currentPose->rotation;
    const auto& t = currentPose->translation;
    float nCam[3];
    for (int i = 0; i < 3; ++i) nCam[i] = R[0][i] * wp.n[0] + R[1][i] * wp.n[1] + R[2][i] * wp.n[2];
    const float cCam = wp.c - (wp.n[0] * t[0] + wp.n[1] * t[1] + wp.n[2] * t[2]);
    if (cCam <= 0.f) return;  // the camera is behind the surface it measured

    uint32_t overwritten = 0;
    for (int y = 2; y < RECT_HEIGHT - 2; ++y) {
        const float ry = (float(y) - RECT_CY) / RECT_FOCAL;
        for (int x = 2; x < RECT_WIDTH - 2; ++x) {
            const size_t i = size_t(y) * RECT_WIDTH + x;
            const float rgb = rgbDepth[i];
            if (!(rgb > 0.f)) continue;  // holes are filled only by real IR observations
            const float rx = (float(x) - RECT_CX) / RECT_FOCAL;
            const float denom = nCam[0] * rx + nCam[1] * ry + nCam[2];
            if (denom <= 0.05f) continue;
            const float zp = cCam / denom;
            if (zp < 0.25f || zp > 5.f) continue;
            const float diff = rgb - zp;
            if (diff >= kMaxBehindPlane || diff <= -kMaxDomeInFront) continue;
            const float dx = R[0][0] * rx * zp + R[0][1] * ry * zp + R[0][2] * zp + t[0] - wp.centroid[0];
            const float dy = R[1][0] * rx * zp + R[1][1] * ry * zp + R[1][2] * zp + t[1] - wp.centroid[1];
            const float dz = R[2][0] * rx * zp + R[2][1] * ry * zp + R[2][2] * zp + t[2] - wp.centroid[2];
            if (footprintAt(wp, dx, dy, dz) != kIrCellSupported) continue;
            // In front of the plane, only textureless pixels can be dome: textured
            // RGB depth is reliable and may be a real object on the wall.
            if (diff < 0.f && !localFlat(rgbLuma, x, y)) continue;
            rgbDepth[i] = zp;
            ++overwritten;
        }
    }
    static std::atomic<int> logCount{0};
    if (overwritten && logCount.fetch_add(1) % 120 == 0)
        std::cout << "[IrDot] plane corrected " << overwritten << " RGB pixels" << std::endl;
}

// Starts a pulse and returns the pair with the most dots. Python startup means the
// emitter's on time is unknown here, and a frame lit at the end of the pulse is
// only delivered after the pulse process exits, so frames are scored until then.
bool IrDotRefine::collectDotPair(IrPairCapture::Pair& pair) {
    if (pulseCmd_.empty()) return capture_.grab(pair, 250);
    IrPairCapture::Pair baselinePair;
    if (!capture_.grab(baselinePair, 250)) {
        error_ = capture_.lastError();
        return false;
    }
    const float baseline = IrPairCapture::dotScore(baselinePair.left);
    if (!pulse_.start(pulseCmd_)) {
        error_ = "could not start dot pulse";
        return false;
    }
    float best = 0.f;
    const auto deadline = Clock::now() + std::chrono::milliseconds(int(pulseSeconds_ * 1000.f) + 3000);
    Clock::time_point stopAt = deadline;
    bool pulseDone = false, pulseOk = false;
    while (running_.load() && Clock::now() < stopAt) {
        if (!pulseDone) {
            const auto state = pulse_.poll();
            if (state != IrPulseProcess::State::Running) {
                pulseDone = true;
                pulseOk = state == IrPulseProcess::State::Succeeded;
                stopAt = Clock::now() + std::chrono::milliseconds(60);
            }
        }
        IrPairCapture::Pair candidate;
        if (!capture_.grab(candidate, 20)) continue;
        const float score = IrPairCapture::dotScore(candidate.left);
        if (score > best) {
            best = score;
            pair = std::move(candidate);
        }
    }
    if (!pulseDone) {
        if (!pulse_.abort()) {
            std::cerr << "[IrDot] pulse did not stop on SIGINT; forcing the emitter off" << std::endl;
            turnEmitterOff();
        }
        error_ = "dot pulse timed out";
        return false;
    }
    if (!pulseOk) {
        error_ = "dot pulse failed (is BELICE 04D8:00DD plugged in?)";
        return false;
    }
    if (best < baseline * minDotRatio_) {
        error_ = "no dots in the pulse frames";  // not lit or out of range: skip the CPU refine
        return false;
    }
    return true;
}

void IrDotRefine::publish(const IrBridgeReply& reply, const TsdfPose& irPose, Clock::time_point captured) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reply.kept >= kMinObservationPoints) {
        observation_.depth = std::make_shared<const std::vector<float>>(reply.depth);
        observation_.pose = irPose;
        observation_.kept = reply.kept;
        haveObservation_ = true;
    }
    if (!reply.hasPlane) {
        worldPlane_.valid = false;
        return;
    }
    rotate(irPose, reply.n, worldPlane_.n);
    rotate(irPose, reply.axisU, worldPlane_.axisU);
    rotate(irPose, reply.axisV, worldPlane_.axisV);
    rotate(irPose, reply.centroid, worldPlane_.centroid);
    float offset = 0.f;
    for (int i = 0; i < 3; ++i) {
        offset += worldPlane_.n[i] * irPose.translation[i];
        worldPlane_.centroid[i] += irPose.translation[i];
    }
    worldPlane_.c = reply.c + offset;
    worldPlane_.inliers = reply.inliers;
    worldPlane_.radius = reply.radius;
    worldPlane_.footprint = reply.footprint;
    worldPlane_.timestamp = captured;
    worldPlane_.valid = true;
}

void IrDotRefine::worker() {
    auto next = Clock::now();
    int captureFailures = 0;
    uint32_t cycles = 0, noDots = 0, refined = 0, planes = 0;
    std::vector<float> disparity;
    std::vector<uint8_t> uncertainty;
    IrBridgeReply reply;
    while (running_.load()) {
        waitUntil(next);
        next = Clock::now() + std::chrono::milliseconds(periodMs_);
        if (!running_.load()) break;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!havePending_) continue;
        }
        ++cycles;
        IrPairCapture::Pair pair;
        if (!collectDotPair(pair)) {
            if (!running_.load()) break;
            if (error_ == "no dots in the pulse frames") {
                ++noDots;
            } else {
                std::cerr << "[IrDot] " << error_ << std::endl;
                if (++captureFailures >= kCaptureFailuresBeforeReopen) {
                    captureFailures = 0;
                    std::cerr << "[IrDot] reopening IR capture" << std::endl;
                    capture_.initialize();
                }
            }
            continue;
        }
        captureFailures = 0;
        const auto captured = Clock::now();
        TsdfPose irPose;
        if (!locator_ || !locator_(pair.leftMonotonicNs - exposureOffsetNs_, irPose)) {
            std::cerr << "[IrDot] no pose at the IR exposure" << std::endl;
            continue;
        }
        // Taken after the grab so the prior is the RGB frame nearest the IR exposure.
        TsdfPose priorPose;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!havePending_) continue;
            disparity.swap(pendingDisp_);
            uncertainty.swap(pendingUnc_);
            priorPose = pendingPose_;
            havePending_ = false;
        }
        const float priorDegrees = rotationDegrees(priorPose, irPose);
        if (priorDegrees > maxPriorDegrees_) {
            std::cerr << "[IrDot] prior " << priorDegrees << " deg from the IR pose, skipped" << std::endl;
            continue;
        }
        if (!bridge_.exchange(disparity.data(), uncertainty.data(), pair.left.data(), pair.right.data(), reply)) {
            if (running_.load()) std::cerr << "[IrDot] bridge: " << bridge_.lastError() << std::endl;
            continue;
        }
        ++refined;
        planes += reply.hasPlane ? 1u : 0u;
        publish(reply, irPose, captured);
        if (cycles % 20 == 0)
            std::cout << "[IrDot] " << cycles << " cycles: " << noDots << " without dots, " << refined
                      << " refined, " << planes << " planes, last " << reply.milliseconds << " ms" << std::endl;
    }
}

}  // namespace FrameSpatial
