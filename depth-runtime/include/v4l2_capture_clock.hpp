#pragma once
// Read-only kernel capture timestamps for one XRService camera node.
//
// XRService streams the IMX616 passthrough cameras from qcom-camss V4L2 nodes
// (/dev/video0 left, /dev/video3 right on the tested Frame). Each completed
// buffer carries the driver's end-of-readout timestamp (CLOCK_MONOTONIC_RAW)
// and a frame sequence. VIDIOC_QUERYBUF on a pidfd_getfd duplicate reads them
// without dequeuing or otherwise disturbing the stream. Method from
// april-tag-xrlib layer/v4l2_clock.cpp.
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace FrameSpatial {

struct V4l2Capture {
    uint32_t sequence = 0;
    int64_t endOfReadoutNs = 0; // CLOCK_MONOTONIC
};

class V4l2CaptureClock {
public:
    // node: "/dev/video0" (left) or "/dev/video3" (right). Throws if XRService or
    // a 2464x2464 NV12 node with that path is unavailable.
    explicit V4l2CaptureClock(const std::string& node);
    ~V4l2CaptureClock();
    V4l2CaptureClock(const V4l2CaptureClock&) = delete;
    V4l2CaptureClock& operator=(const V4l2CaptureClock&) = delete;

    void poll();                                    // a few ioctls; call often
    std::optional<V4l2Capture> latest() const;
    // Newest capture whose end of readout is at or before `monotonicNs`.
    std::optional<V4l2Capture> atOrBefore(int64_t monotonicNs) const;
    const std::string& description() const { return description_; }

    static int64_t monotonicNs();
    // CLOCK_MONOTONIC - CLOCK_MONOTONIC_RAW now, from the tightest of a few paired reads.
    static int64_t monotonicMinusRawNs();
    // 0 if XRService is not running.
    static int findXrServicePid();

private:
    int fd_ = -1;
    std::string description_;
    mutable std::mutex mutex_;
    static constexpr int kHistory = 64;
    V4l2Capture history_[kHistory]{};
    int count_ = 0;
    uint32_t lastSequence_ = 0;
};

} // namespace FrameSpatial
