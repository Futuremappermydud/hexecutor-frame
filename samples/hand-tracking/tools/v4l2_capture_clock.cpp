#include "v4l2_capture_clock.hpp"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <linux/videodev2.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

namespace FrameSpatial {
namespace {
int64_t nowNs(clockid_t clock) {
    timespec ts{};
    clock_gettime(clock, &ts);
    return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}
// CLOCK_MONOTONIC - CLOCK_MONOTONIC_RAW from the tightest of a few paired reads.
// NTP slews MONOTONIC by tens of ppm, so it is refreshed on every new capture.
int64_t monotonicMinusRaw() {
    int64_t bestOffset = 0, bestSpan = INT64_MAX;
    for (int i = 0; i < 3; ++i) {
        const int64_t a = nowNs(CLOCK_MONOTONIC), r = nowNs(CLOCK_MONOTONIC_RAW), b = nowNs(CLOCK_MONOTONIC);
        if (b - a < bestSpan) { bestSpan = b - a; bestOffset = (a + b) / 2 - r; }
    }
    return bestOffset;
}
pid_t findXrService() {
    DIR* proc = opendir("/proc");
    if (!proc) return 0;
    pid_t found = 0;
    while (dirent* e = readdir(proc)) {
        const pid_t pid = pid_t(std::atoi(e->d_name));
        if (pid <= 0) continue;
        std::ifstream cmdline("/proc/" + std::string(e->d_name) + "/cmdline");
        std::string argv0;
        std::getline(cmdline, argv0, '\0');
        if (argv0 == "XRService" ||
            (argv0.size() > 10 && argv0.compare(argv0.size() - 10, 10, "/XRService") == 0)) {
            found = pid;
            break;
        }
    }
    closedir(proc);
    return found;
}
} // namespace

int64_t V4l2CaptureClock::monotonicNs() { return nowNs(CLOCK_MONOTONIC); }
int64_t V4l2CaptureClock::monotonicMinusRawNs() { return monotonicMinusRaw(); }
int V4l2CaptureClock::findXrServicePid() { return findXrService(); }

V4l2CaptureClock::V4l2CaptureClock(const std::string& node) {
    const pid_t pid = findXrService();
    if (pid <= 0) throw std::runtime_error("XRService not running");
    const int pidfd = int(syscall(__NR_pidfd_open, pid, 0));
    if (pidfd < 0) throw std::runtime_error("pidfd_open(XRService) failed; check ptrace_scope");
    const std::string dir = "/proc/" + std::to_string(pid) + "/fd";
    DIR* fds = opendir(dir.c_str());
    while (fds && fd_ < 0) {
        dirent* e = readdir(fds);
        if (!e) break;
        char target[256] = {};
        if (readlink((dir + "/" + e->d_name).c_str(), target, sizeof(target) - 1) <= 0 ||
            node != target) continue;
        const int fd = int(syscall(__NR_pidfd_getfd, pidfd, std::atoi(e->d_name), 0));
        if (fd < 0) continue;
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(fd, VIDIOC_G_FMT, &format) == 0 && format.fmt.pix_mp.width == 2464 &&
            format.fmt.pix_mp.height == 2464 && format.fmt.pix_mp.pixelformat == V4L2_PIX_FMT_NV12) {
            fd_ = fd;
            description_ = node + " (XRService pid " + std::to_string(pid) + ")";
        } else {
            close(fd);
        }
    }
    if (fds) closedir(fds);
    close(pidfd);
    if (fd_ < 0) throw std::runtime_error("XRService camera node " + node + " (2464x2464 NV12) not found");
}

V4l2CaptureClock::~V4l2CaptureClock() {
    if (fd_ >= 0) close(fd_);
}

void V4l2CaptureClock::poll() {
    uint32_t best = 0;
    int64_t bestTs = 0;
    for (uint32_t i = 0; i < VIDEO_MAX_FRAME; ++i) {
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_DMABUF;
        buffer.index = i;
        buffer.m.planes = planes;
        buffer.length = VIDEO_MAX_PLANES;
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buffer) != 0) break;
        // Queued buffers report no timestamp; completed ones keep theirs.
        if (buffer.sequence > best && buffer.timestamp.tv_sec != 0) {
            best = buffer.sequence;
            bestTs = int64_t(buffer.timestamp.tv_sec) * 1000000000ll + int64_t(buffer.timestamp.tv_usec) * 1000;
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (best == 0 || best == lastSequence_) return;
    lastSequence_ = best;
    history_[count_ % kHistory] = {best, bestTs + monotonicMinusRaw()};
    ++count_;
}

std::optional<V4l2Capture> V4l2CaptureClock::latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == 0) return std::nullopt;
    return history_[(count_ - 1) % kHistory];
}

std::optional<V4l2Capture> V4l2CaptureClock::atOrBefore(int64_t monotonicNs) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = count_ - 1; i >= 0 && i >= count_ - kHistory; --i)
        if (history_[i % kHistory].endOfReadoutNs <= monotonicNs) return history_[i % kHistory];
    return std::nullopt;
}

} // namespace FrameSpatial
