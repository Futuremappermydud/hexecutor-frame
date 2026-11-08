#include "ir_dot_bridge.hpp"
#include "calibration.hpp"
#include "model_depth_geometry.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace FrameSpatial {
namespace {
constexpr uint32_t kHelloMagic = 0x30444F54, kRequestMagic = 0x31444F54, kReplyMagic = 0x34444F54;
constexpr int kPollSliceMs = 100;
constexpr int kMaxBackoffMs = 30000;

std::string setting(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v && *v ? v : fallback;
}

struct Hello {
    uint32_t magic, width, height, rectWidth, footprintSize;
    float rectFocal, stepDegrees;
};
struct RequestHeader {
    uint32_t magic, width, height, rectWidth, sequence;
};
struct ReplyHeader {
    uint32_t magic, sequence, kept, hasPlane, inliers;
    float milliseconds, n[3], c, centroid[3], axisU[3], axisV[3], radius, rms;
};
static_assert(sizeof(Hello) == 28 && sizeof(RequestHeader) == 20 && sizeof(ReplyHeader) == 84,
              "bridge wire structs must match ir_dot_refine_bridge.py");

// Signal, wait up to graceMs, then SIGKILL. Returns true if the grace was enough.
bool reap(pid_t pid, int signal, int graceMs, bool group) {
    group ? killpg(pid, signal) : kill(pid, signal);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(graceMs);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid || (done < 0 && errno != EINTR)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    group ? killpg(pid, SIGKILL) : kill(pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return false;
}
}  // namespace

void IrBridgeClient::stop() {
    if (pid_ > 0) {
        reap(pid_, SIGTERM, 1000, false);
        pid_ = -1;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void IrBridgeClient::fail() {
    stop();
    backoffMs_ = backoffMs_ ? std::min(backoffMs_ * 2, kMaxBackoffMs) : 1000;
    retryAfter_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(backoffMs_);
}

bool IrBridgeClient::transfer(void* data, size_t bytes, bool sending) {
    auto* ptr = static_cast<uint8_t*>(data);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(transferTimeoutMs);
    while (bytes) {
        if (keepRunning_ && !keepRunning_->load()) {
            error_ = "stopping";
            return false;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            error_ = "IR-dot bridge timed out";
            return false;
        }
        pollfd p{fd_, short(sending ? POLLOUT : POLLIN), 0};
        const int ready = poll(&p, 1, int(std::min<long long>(remaining, kPollSliceMs)));
        if (ready == 0 || (ready < 0 && errno == EINTR)) continue;
        if (ready < 0) {
            error_ = std::strerror(errno);
            return false;
        }
        const ssize_t n = sending ? send(fd_, ptr, bytes, MSG_NOSIGNAL | MSG_DONTWAIT)
                                  : recv(fd_, ptr, bytes, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n <= 0) {
            error_ = "IR-dot bridge disconnected";
            return false;
        }
        ptr += n;
        bytes -= size_t(n);
    }
    return true;
}

bool IrBridgeClient::start() {
    stop();
    const std::string python = setting("FRAME_DEPTH_PYTHON",
        setting("HOME", "/home/steamos") + "/frame-dsp/depth-model/.venv/bin/python");
    const std::string script = setting("FRAME_IR_DOT_SCRIPT", "scripts/ir_dot_refine_bridge.py");
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) {
        error_ = std::strerror(errno);
        return false;
    }
    fcntl(sockets[0], F_SETFD, FD_CLOEXEC);
    fcntl(sockets[1], F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, sockets[0]);
    posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
    if (sockets[1] != 3) posix_spawn_file_actions_addclose(&actions, sockets[1]);
    posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
    const char* args[] = {python.c_str(), "-u", script.c_str(), "--fd", "3", nullptr};
    pid_t child = -1;
    const int result = posix_spawn(&child, python.c_str(), &actions, nullptr,
                                   const_cast<char* const*>(args), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(sockets[1]);
    if (result) {
        close(sockets[0]);
        error_ = std::strerror(result);
        return false;
    }
    fd_ = sockets[0];
    pid_ = child;
    ++spawns_;
    Hello hello{};
    if (!transfer(&hello, sizeof(hello), false)) {
        stop();
        return false;
    }
    const float step = float(std::atof(setting("FRAME_DEPTH_SPHERICAL_DEG", "0.29").c_str()));
    if (hello.magic != kHelloMagic || hello.width != uint32_t(MODEL_WIDTH) ||
        hello.height != uint32_t(MODEL_HEIGHT) || hello.rectWidth != uint32_t(RECT_WIDTH) ||
        hello.footprintSize != uint32_t(kIrFootprintSize) ||
        std::abs(hello.rectFocal - RECT_FOCAL) > 1e-3f || std::abs(hello.stepDegrees - step) > 1e-4f) {
        error_ = "incompatible IR-dot bridge";
        stop();
        return false;
    }
    return true;
}

bool IrBridgeClient::sendRequest(const float* disparity, const uint8_t* uncertainty, const uint8_t* left,
                                 const uint8_t* right, uint32_t sequence) {
    RequestHeader header{kRequestMagic, uint32_t(MODEL_WIDTH), uint32_t(MODEL_HEIGHT), uint32_t(RECT_WIDTH), sequence};
    const size_t modelPixels = size_t(MODEL_WIDTH) * MODEL_HEIGHT;
    const size_t irPixels = size_t(1056) * 1024;
    return transfer(&header, sizeof(header), true) &&
           transfer(const_cast<float*>(disparity), modelPixels * sizeof(float), true) &&
           transfer(const_cast<uint8_t*>(uncertainty), modelPixels, true) &&
           transfer(const_cast<uint8_t*>(left), irPixels, true) &&
           transfer(const_cast<uint8_t*>(right), irPixels, true);
}

bool IrBridgeClient::receiveReply(IrBridgeReply& reply, uint32_t sequence) {
    ReplyHeader header{};
    if (!transfer(&header, sizeof(header), false)) return false;
    if (header.magic != kReplyMagic || header.sequence != sequence) {
        error_ = "IR-dot bridge reply out of sequence";
        return false;
    }
    reply.footprint.resize(size_t(kIrFootprintSize) * kIrFootprintSize);
    reply.depth.resize(size_t(RECT_WIDTH) * RECT_HEIGHT);
    if (!transfer(reply.footprint.data(), reply.footprint.size(), false) ||
        !transfer(reply.depth.data(), reply.depth.size() * sizeof(float), false))
        return false;
    reply.kept = header.kept;
    reply.inliers = header.inliers;
    reply.milliseconds = header.milliseconds;
    reply.hasPlane = header.hasPlane != 0;
    std::copy(header.n, header.n + 3, reply.n);
    reply.c = header.c;
    std::copy(header.centroid, header.centroid + 3, reply.centroid);
    std::copy(header.axisU, header.axisU + 3, reply.axisU);
    std::copy(header.axisV, header.axisV + 3, reply.axisV);
    reply.radius = header.radius;
    reply.rms = header.rms;
    if (reply.hasPlane) {
        const float norm = reply.n[0] * reply.n[0] + reply.n[1] * reply.n[1] + reply.n[2] * reply.n[2];
        if (!std::isfinite(norm) || std::abs(norm - 1.f) > 1e-3f || !(reply.c > 0.f)) {
            error_ = "IR-dot bridge returned an invalid plane";
            return false;
        }
    }
    return true;
}

bool IrBridgeClient::exchange(const float* disparity, const uint8_t* uncertainty, const uint8_t* left,
                              const uint8_t* right, IrBridgeReply& reply) {
    if (fd_ < 0) {
        if (std::chrono::steady_clock::now() < retryAfter_) {
            error_ = "IR-dot bridge restart backing off";
            return false;
        }
        std::cerr << "[IrDot] restarting bridge" << std::endl;
        if (!start()) {
            fail();
            return false;
        }
    }
    const uint32_t sequence = ++sequence_;
    if (!sendRequest(disparity, uncertainty, left, right, sequence) || !receiveReply(reply, sequence)) {
        fail();
        return false;
    }
    backoffMs_ = 0;
    return true;
}

bool IrPulseProcess::start(const std::string& command) {
    if (pid_ > 0) return false;
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    const char* args[] = {"/bin/sh", "-c", command.c_str(), nullptr};
    pid_t child = -1;
    const int result = posix_spawn(&child, "/bin/sh", nullptr, &attr, const_cast<char* const*>(args), environ);
    posix_spawnattr_destroy(&attr);
    if (result) return false;
    pid_ = child;
    return true;
}

IrPulseProcess::State IrPulseProcess::poll() {
    if (pid_ <= 0) return State::Idle;
    int status = 0;
    const pid_t done = waitpid(pid_, &status, WNOHANG);
    if (done == 0 || (done < 0 && errno == EINTR)) return State::Running;
    pid_ = -1;
    return done == -1 || !WIFEXITED(status) || WEXITSTATUS(status) ? State::Failed : State::Succeeded;
}

bool IrPulseProcess::abort(int graceMs) {
    if (pid_ <= 0) return true;
    const bool clean = reap(pid_, SIGINT, graceMs, true);
    pid_ = -1;
    return clean;
}

}  // namespace FrameSpatial
