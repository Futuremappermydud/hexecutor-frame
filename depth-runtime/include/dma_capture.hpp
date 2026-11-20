#pragma once

#include "spatial_types.hpp"
#include "vk_detile.hpp"
#include "timed_stereo_capture.hpp"
#include <memory>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace FrameSpatial {

class DMACapture {
public:
    DMACapture();
    ~DMACapture();

    bool initialize(pid_t compositorPid, int rightFd = -1, int leftFd = -1);
    void release();

    // RGB consumers use the default; depth-only consumers can skip chroma.
    bool captureFrame(bool includeUV = true, bool leftOnly = false);

    const uint8_t* getLeftBuffer() const;
    const uint8_t* getRightBuffer() const;
    const uint8_t* getLeftUV() const;
    const uint8_t* getRightUV() const;
    uint64_t leftHash() const;
    uint64_t rightHash() const;
    int leftSlot() const;
    int rightSlot() const;
    int width() const;
    int height() const;
    size_t getBufferSize() const { return m_bufferSize; }
    bool isReady() const { return m_isReady; }

    static bool autoDiscoverCameraGroups(pid_t compositorPid,
                                         std::vector<int>& outRightFds,
                                         std::vector<int>& outLeftFds);
    static pid_t findCompositorPid();

    // FRAME_CAPTURE_TIMING=v4l2 (E02b): pairs share one V4L2 camera sequence and carry
    // kernel end-of-readout times. Null in the legacy hash-selected mode.
    const TimedStereoPair* timedPair() const { return m_timed && m_timedPair.valid ? &m_timedPair : nullptr; }
    bool timedMode() const { return bool(m_timed); }
    TimedStereoStats timedStats() const { return m_timed ? m_timed->stats() : TimedStereoStats{}; }

private:
    pid_t m_compositorPid;
    int m_pidFd;
    std::vector<int> m_localRightFds;
    std::vector<int> m_localLeftFds;
    size_t m_bufferSize;
    bool m_isReady;
    VkYDetile m_detile;
    VkYDetile* m_activeDetile;
    std::unique_ptr<TimedStereoCapture> m_timed;
    TimedStereoPair m_timedPair;
};

} // namespace FrameSpatial
