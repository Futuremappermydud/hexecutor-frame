#include "ir_pair_capture.hpp"
#include <cstdlib>

namespace FrameSpatial {
float IrPairCapture::dotScore(const std::vector<uint8_t>& image) {
    if (image.size() != size_t(Width) * Height) return 0.f;
    int64_t sum = 0, count = 0;
    for (int y = Height / 8; y < 7 * Height / 8; y += 4) {
        const uint8_t* row = image.data() + size_t(y) * Width;
        for (int x = Width / 8 + 2; x < 7 * Width / 8 - 2; ++x) {
            sum += std::abs(2 * int(row[x]) - int(row[x - 2]) - int(row[x + 2]));
            ++count;
        }
    }
    return count ? float(double(sum) / double(count)) : 0.f;
}
}  // namespace FrameSpatial

#ifdef __linux__
#include "v4l2_capture_clock.hpp"
#include <dirent.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sys/stat.h>
#include <thread>

namespace FrameSpatial {
namespace {
constexpr int kW = IrPairCapture::Width, kH = IrPairCapture::Height;
#ifndef SYS_pidfd_open
constexpr long SYS_pidfd_open = 434;
#endif
#ifndef SYS_pidfd_getfd
constexpr long SYS_pidfd_getfd = 438;
#endif

bool queryBuf(int fd, uint32_t index, v4l2_buffer& b, v4l2_plane* pl) {
    memset(&b, 0, sizeof b);
    memset(pl, 0, sizeof(v4l2_plane) * VIDEO_MAX_PLANES);
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_DMABUF;
    b.index = index;
    b.m.planes = pl;
    b.length = VIDEO_MAX_PLANES;
    return ioctl(fd, VIDIOC_QUERYBUF, &b) == 0;
}
}  // namespace

struct IrPairCapture::Impl {
    struct Mapping {
        const uint8_t* data = nullptr;
        size_t size = 0;
    };
    struct Seen {
        int64_t ts = 0;
        uint32_t seq = 0, index = 0;
        float mean = 0.f;
    };
    struct Cam {
        std::string node;
        int fd = -1;
        size_t stride = kW;
        int64_t lastTs = 0;
        std::deque<Seen> recent;
        Seen pending;                   // the frame most recently returned by grab
        std::map<ino_t, Mapping> maps;  // by dmabuf identity: XRService can reuse fd numbers
    };
    int pidfd = -1;
    Cam left, right;
    std::string error;

    static void reset(Cam& cam) {
        for (auto& entry : cam.maps) munmap(const_cast<uint8_t*>(entry.second.data), entry.second.size);
        cam.maps.clear();
        if (cam.fd >= 0) close(cam.fd);
        cam.fd = -1;
        cam.lastTs = 0;
        cam.recent.clear();
        cam.stride = kW;
    }
    void reset() {
        reset(left);
        reset(right);
        if (pidfd >= 0) close(pidfd);
        pidfd = -1;
    }

    bool mapBuffer(Cam& cam, int xrfd, Mapping& out) {
        const int bfd = int(syscall(SYS_pidfd_getfd, pidfd, xrfd, 0));
        if (bfd < 0) return false;
        struct stat st {};
        if (fstat(bfd, &st) != 0) {
            close(bfd);
            return false;
        }
        auto it = cam.maps.find(st.st_ino);
        if (it == cam.maps.end()) {
            if (cam.maps.size() >= 2 * VIDEO_MAX_FRAME) {  // the stream was restarted repeatedly
                for (auto& entry : cam.maps) munmap(const_cast<uint8_t*>(entry.second.data), entry.second.size);
                cam.maps.clear();
            }
            const off_t end = lseek(bfd, 0, SEEK_END);
            if (end <= 0) {
                close(bfd);
                return false;
            }
            void* m = mmap(nullptr, size_t(end), PROT_READ, MAP_SHARED, bfd, 0);
            if (m == MAP_FAILED) {
                close(bfd);
                return false;
            }
            it = cam.maps.emplace(st.st_ino, Mapping{static_cast<const uint8_t*>(m), size_t(end)}).first;
        }
        close(bfd);
        out = it->second;
        return true;
    }

    bool stillHolds(const Cam& cam, const Seen& frame) {
        v4l2_buffer b;
        v4l2_plane pl[VIDEO_MAX_PLANES];
        return queryBuf(cam.fd, frame.index, b, pl) && b.sequence == frame.seq && b.timestamp.tv_sec &&
               !(b.flags & V4L2_BUF_FLAG_QUEUED);
    }

    bool copyFrame(Cam& cam, const Seen& frame, std::vector<uint8_t>& px) {
        v4l2_buffer b;
        v4l2_plane pl[VIDEO_MAX_PLANES];
        if (!queryBuf(cam.fd, frame.index, b, pl) || b.sequence != frame.seq) return false;
        Mapping map;
        if (!mapBuffer(cam, pl[0].m.fd, map)) return false;
        const size_t lastRowEnd = size_t(pl[0].data_offset) + cam.stride * (kH - 1) + kW;
        if (lastRowEnd > map.size) {
            error = "IR buffer smaller than its format";
            return false;
        }
        px.resize(size_t(kW) * kH);
        const uint8_t* src = map.data + pl[0].data_offset;
        if (cam.stride == size_t(kW)) {
            std::memcpy(px.data(), src, px.size());
        } else {
            for (int y = 0; y < kH; ++y) std::memcpy(px.data() + size_t(y) * kW, src + cam.stride * y, kW);
        }
        return stillHolds(cam, frame);
    }

    // Records every newly finished frame with a sampled brightness.
    void poll(Cam& cam) {
        for (uint32_t i = 0; i < VIDEO_MAX_FRAME; i++) {
            v4l2_buffer b;
            v4l2_plane pl[VIDEO_MAX_PLANES];
            if (!queryBuf(cam.fd, i, b, pl)) break;
            if (!b.timestamp.tv_sec || (b.flags & V4L2_BUF_FLAG_QUEUED)) continue;
            const int64_t ts = int64_t(b.timestamp.tv_sec) * 1000000000ll + int64_t(b.timestamp.tv_usec) * 1000ll;
            if (ts <= cam.lastTs) continue;
            Mapping map;
            if (!mapBuffer(cam, pl[0].m.fd, map)) continue;
            const size_t lastRowEnd = size_t(pl[0].data_offset) + cam.stride * (kH - 1) + kW;
            if (lastRowEnd > map.size) continue;
            const uint8_t* src = map.data + pl[0].data_offset;
            int64_t sum = 0, count = 0;
            for (int y = 0; y < kH; y += 16)
                for (int x = 0; x < kW; x += 16, ++count) sum += src[cam.stride * y + x];
            Seen seen;
            seen.ts = ts;
            seen.seq = b.sequence;
            seen.index = i;
            seen.mean = float(double(sum) / double(count));
            if (!stillHolds(cam, seen)) continue;
            auto where = cam.recent.begin();
            while (where != cam.recent.end() && where->ts < ts) ++where;
            cam.recent.insert(where, seen);
            cam.lastTs = ts;
        }
        const int64_t keepAfter = cam.lastTs - 200000000ll;
        while (!cam.recent.empty() && cam.recent.front().ts < keepAfter) cam.recent.pop_front();
    }

    static bool isLong(const Cam& cam, const Seen& frame) {
        for (const Seen& other : cam.recent) {
            const int64_t apart = std::llabs(other.ts - frame.ts);
            // The alternate exposure is 8.4 ms after a long frame and 24.9 ms before the next one.
            if (apart > 4000000ll && apart < 30000000ll) return IrPairCapture::isLongExposure(frame.mean, other.mean);
        }
        return false;
    }

    int64_t lastPairTs = 0;

    bool newestLongPair() {
        poll(left);
        poll(right);
        for (auto l = left.recent.rbegin(); l != left.recent.rend(); ++l) {
            if (l->ts <= lastPairTs) return false;
            if (!isLong(left, *l)) continue;
            for (const Seen& r : right.recent) {
                if (std::llabs(l->ts - r.ts) < 1000000ll && isLong(right, r)) {
                    left.pending = *l;
                    right.pending = r;
                    lastPairTs = l->ts;
                    return true;
                }
            }
        }
        return false;
    }
};

IrPairCapture::IrPairCapture() : impl_(std::make_unique<Impl>()) {}
IrPairCapture::~IrPairCapture() {
    if (impl_) impl_->reset();
}
const std::string& IrPairCapture::lastError() const { return impl_->error; }

bool IrPairCapture::initialize() {
    impl_->reset();
    const pid_t pid = V4l2CaptureClock::findXrServicePid();
    if (!pid) {
        impl_->error = "no XRService";
        return false;
    }
    impl_->pidfd = int(syscall(SYS_pidfd_open, pid, 0));
    if (impl_->pidfd < 0) {
        impl_->error = "pidfd_open XRService failed";
        return false;
    }
    const std::string dir = "/proc/" + std::to_string(pid) + "/fd";
    DIR* d = opendir(dir.c_str());
    if (!d) {
        impl_->error = "cannot list XRService fds";
        return false;
    }
    while (dirent* e = readdir(d)) {
        char target[256] = {};
        if (readlink((dir + "/" + e->d_name).c_str(), target, sizeof(target) - 1) <= 0) continue;
        Impl::Cam* cam = nullptr;
        if (!strcmp(target, "/dev/video13")) cam = &impl_->left;
        else if (!strcmp(target, "/dev/video9")) cam = &impl_->right;
        else continue;
        cam->node = target;
        cam->fd = int(syscall(SYS_pidfd_getfd, impl_->pidfd, atoi(e->d_name), 0));
        if (cam->fd < 0) {
            closedir(d);
            impl_->error = "pidfd_getfd tracking camera failed (ptrace_scope must be 0)";
            return false;
        }
    }
    closedir(d);
    if (impl_->left.fd < 0 || impl_->right.fd < 0) {
        impl_->error = "did not find both tracking cameras";
        return false;
    }
    for (Impl::Cam* cam : {&impl_->left, &impl_->right})
        for (uint32_t i = 0; i < VIDEO_MAX_FRAME; i++) {
            v4l2_buffer b;
            v4l2_plane pl[VIDEO_MAX_PLANES];
            if (!queryBuf(cam->fd, i, b, pl)) break;
            const int64_t ts = int64_t(b.timestamp.tv_sec) * 1000000000ll + int64_t(b.timestamp.tv_usec) * 1000ll;
            if (ts > cam->lastTs) cam->lastTs = ts;
        }
    for (Impl::Cam* cam : {&impl_->left, &impl_->right}) {
        v4l2_format fmt;
        memset(&fmt, 0, sizeof fmt);
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(cam->fd, VIDIOC_G_FMT, &fmt) != 0) continue;  // keep the documented 1056 stride
        if (int(fmt.fmt.pix_mp.width) != kW || int(fmt.fmt.pix_mp.height) != kH) {
            impl_->error = cam->node + " is not 1056x1024";
            return false;
        }
        const uint32_t stride = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
        if (stride >= uint32_t(kW)) cam->stride = stride;
    }
    return true;
}

bool IrPairCapture::grab(Pair& out, int timeoutMs, bool copyPixels) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (impl_->newestLongPair()) {
            out.left.clear();
            out.right.clear();
            out.leftSeq = impl_->left.pending.seq;
            out.rightSeq = impl_->right.pending.seq;
            out.leftTsNs = impl_->left.pending.ts;
            out.rightTsNs = impl_->right.pending.ts;
            out.leftMonotonicNs = out.leftTsNs + V4l2CaptureClock::monotonicMinusRawNs();
            if (!copyPixels || fetchPixels(out)) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    impl_->error = "no matched long-exposure IR pair";
    return false;
}

bool IrPairCapture::fetchPixels(Pair& pair) {
    if (pair.leftSeq != impl_->left.pending.seq || pair.rightSeq != impl_->right.pending.seq) return false;
    return impl_->copyFrame(impl_->left, impl_->left.pending, pair.left) &&
           impl_->copyFrame(impl_->right, impl_->right.pending, pair.right);
}

}  // namespace FrameSpatial

#else

namespace FrameSpatial {

struct IrPairCapture::Impl {
    std::string error{"IrPairCapture requires Linux V4L2 and pidfd_getfd"};
};

IrPairCapture::IrPairCapture() : impl_(std::make_unique<Impl>()) {}
IrPairCapture::~IrPairCapture() = default;
bool IrPairCapture::initialize() { return false; }
const std::string& IrPairCapture::lastError() const { return impl_->error; }
bool IrPairCapture::grab(Pair&, int, bool) { return false; }
bool IrPairCapture::fetchPixels(Pair&) { return false; }

}  // namespace FrameSpatial

#endif
