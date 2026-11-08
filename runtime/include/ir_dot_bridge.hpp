#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace FrameSpatial {

// Plane-aligned footprint of the fitted surface: kIrFootprintSize^2 cells of
// kIrFootprintCell metres, centred on the plane centroid.
constexpr int kIrFootprintSize = 64;
constexpr float kIrFootprintCell = 0.10f;
enum IrFootprintState : uint8_t {
    kIrCellUnobserved = 0,
    kIrCellSupported = 1,  // dots lie on the plane here
    kIrCellBlocked = 2,    // dots lie in front of the plane here: foreground
};

struct IrBridgeReply {
    uint32_t kept = 0;
    uint32_t inliers = 0;
    bool hasPlane = false;
    float milliseconds = 0.f;
    // Plane n.X = c, c > 0, in the rectified RGB left camera at IR capture time.
    float n[3]{}, c = 0.f, centroid[3]{}, axisU[3]{}, axisV[3]{};
    float radius = 0.f, rms = 0.f;
    std::vector<uint8_t> footprint;  // kIrFootprintSize^2, row = v, column = u
    std::vector<float> depth;        // RECT_WIDTH * RECT_HEIGHT, 0 = no observation
};

// Owns the Python refine process. Any failed exchange kills the process, because
// a partial transfer leaves the stream unframed. The next exchange respawns it
// after a growing backoff, so the protocol can never run desynchronised.
class IrBridgeClient {
public:
    explicit IrBridgeClient(const std::atomic<bool>* keepRunning) : keepRunning_(keepRunning) {}
    ~IrBridgeClient() { stop(); }
    IrBridgeClient(const IrBridgeClient&) = delete;
    IrBridgeClient& operator=(const IrBridgeClient&) = delete;

    bool start();
    void stop();
    bool exchange(const float* disparity, const uint8_t* uncertainty, const uint8_t* left,
                  const uint8_t* right, IrBridgeReply& reply);
    bool connected() const { return fd_ >= 0; }
    const std::string& lastError() const { return error_; }
    int transferTimeoutMs = 8000;
    int spawnCount() const { return spawns_; }

private:
    bool transfer(void* data, size_t bytes, bool sending);
    bool sendRequest(const float* disparity, const uint8_t* uncertainty, const uint8_t* left,
                     const uint8_t* right, uint32_t sequence);
    bool receiveReply(IrBridgeReply& reply, uint32_t sequence);
    void fail();

    const std::atomic<bool>* keepRunning_;
    int fd_ = -1, pid_ = -1, spawns_ = 0;
    uint32_t sequence_ = 0;
    int backoffMs_ = 0;
    std::chrono::steady_clock::time_point retryAfter_{};
    std::string error_;
};

// One emitter pulse, run asynchronously in its own process group so IR frames can
// be collected while the emitter is on.
class IrPulseProcess {
public:
    enum class State { Idle, Running, Succeeded, Failed };
    ~IrPulseProcess() { abort(); }
    bool start(const std::string& command);
    State poll();
    // SIGINT first: belice_flash.py turns the emitter off in a finally block, which
    // SIGTERM or SIGKILL would skip. Returns false if SIGKILL was needed, in which
    // case the emitter may still be on.
    bool abort(int graceMs = 1000);

private:
    int pid_ = -1;
};

}  // namespace FrameSpatial
