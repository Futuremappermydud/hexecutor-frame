#pragma once

#include "spatial_types.hpp"
#include "vk_detile.hpp"
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

private:
    pid_t m_compositorPid;
    int m_pidFd;
    std::vector<int> m_localRightFds;
    std::vector<int> m_localLeftFds;
    size_t m_bufferSize;
    bool m_isReady;
    VkYDetile m_detile;
    VkYDetile* m_activeDetile;
};

} // namespace FrameSpatial
