#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace FrameSpatial {

// One compositor camera DMA-BUF: its descriptor number in the compositor and
// its dmabuf inode, which increases in allocation order.
struct CameraBuffer {
    int fd;
    uint64_t inode;
};

// Split the compositor's camera DMA-BUFs into right and left groups.
//
// The live stereo stream is one allocation of slotsPerEye right buffers
// followed by slotsPerEye left buffers. The compositor can also keep an older
// allocation that is no longer written (all zeros), for example after the
// passthrough resolution policy changes. Splitting every buffer at the
// midpoint then puts the dead group on the right and the live right camera on
// the left, and the left image comes out rotated 180 degrees. Keep only the
// newest allocation, ordered by inode.
inline bool splitCameraGroups(std::vector<CameraBuffer> buffers, size_t slotsPerEye,
                              std::vector<int>& right, std::vector<int>& left) {
    right.clear();
    left.clear();
    if (buffers.size() < 2 || slotsPerEye == 0) return false;
    std::sort(buffers.begin(), buffers.end(),
              [](const CameraBuffer& a, const CameraBuffer& b) {
                  return a.inode != b.inode ? a.inode < b.inode : a.fd < b.fd;
              });
    const size_t live = std::min(buffers.size(), 2 * slotsPerEye);
    buffers.erase(buffers.begin(), buffers.end() - live);
    const size_t mid = buffers.size() / 2;
    for (size_t i = 0; i < buffers.size(); ++i)
        (i < mid ? right : left).push_back(buffers[i].fd);
    return true;
}

} // namespace FrameSpatial
