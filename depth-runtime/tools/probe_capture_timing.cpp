// E02 capture-timing probe: snapshot every compositor camera DMA-BUF as fast as
// the production UBWC copy allows and log per-slot Y hashes with CLOCK_MONOTONIC
// brackets. No DSP, OpenXR or rectification. Output CSV on stdout:
//   group,begin_ns,end_ns,R0..R3,L0..L3,picked_left,picked_right
// Slot labels follow DMACapture's split (first half = right, second half = left).
#define _GNU_SOURCE
#include "dma_capture.hpp"
#include "vk_detile.hpp"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
using namespace FrameSpatial;
#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif
static int64_t now() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return int64_t(t.tv_sec) * 1000000000ll + t.tv_nsec; }
int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 3.0;
    const pid_t pid = DMACapture::findCompositorPid();
    std::vector<int> right, left;
    if (pid <= 0 || !DMACapture::autoDiscoverCameraGroups(pid, right, left)) return 1;
    const int pidfd = int(syscall(__NR_pidfd_open, pid, 0));
    if (pidfd < 0) { perror("pidfd_open"); return 1; }
    auto local = [&](std::vector<int>& fds) {
        for (int& fd : fds) { fd = int(syscall(__NR_pidfd_getfd, pidfd, fd, 0)); if (fd < 0) { perror("pidfd_getfd"); std::exit(1); } }
    };
    local(right); local(left);
    std::vector<std::unique_ptr<VkYDetile>> groups;
    for (size_t base = 0; base < std::max(left.size(), right.size()); base += VkYDetile::kMaxSlots) {
        auto chunk = [&](const std::vector<int>& v) {
            return std::vector<int>(v.begin() + std::min(base, v.size()),
                                    v.begin() + std::min(base + VkYDetile::kMaxSlots, v.size()));
        };
        auto g = std::make_unique<VkYDetile>();
        if (!g->importStereoGroups(chunk(left), chunk(right))) { std::cerr << "import failed at group " << groups.size() << "\n"; return 1; }
        groups.push_back(std::move(g));
    }
    std::cerr << "probe: " << right.size() << " right + " << left.size() << " left buffers, " << groups.size() << " group(s)\n";
    std::printf("group,begin_ns,end_ns");
    for (int i = 0; i < VkYDetile::kMaxSlots; i++) std::printf(",R%d", i);
    for (int i = 0; i < VkYDetile::kMaxSlots; i++) std::printf(",L%d", i);
    std::printf(",picked_left,picked_right\n");
    const int64_t stop = now() + int64_t(seconds * 1e9);
    while (now() < stop) {
        for (size_t gi = 0; gi < groups.size(); gi++) {
            auto& g = *groups[gi];
            const int64_t b = now();
            if (!g.copyYPlanes(false)) { std::cerr << "copy failed\n"; return 1; }
            const int64_t e = now();
            std::printf("%zu,%lld,%lld", gi, (long long)b, (long long)e);
            for (int i = 0; i < VkYDetile::kMaxSlots; i++) std::printf(",%016llx", i < g.rightSlotCount() ? (unsigned long long)g.slotHash(false, i) : 0ull);
            for (int i = 0; i < VkYDetile::kMaxSlots; i++) std::printf(",%016llx", i < g.leftSlotCount() ? (unsigned long long)g.slotHash(true, i) : 0ull);
            std::printf(",%d,%d\n", g.leftSlot(), g.rightSlot());
        }
    }
    for (int fd : right) close(fd);
    for (int fd : left) close(fd);
    close(pidfd);
}
