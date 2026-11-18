#define _GNU_SOURCE
#include "dma_capture.hpp"
#include "camera_groups.hpp"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <algorithm>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

namespace FrameSpatial {

DMACapture::DMACapture()
    : m_compositorPid(0)
    , m_pidFd(-1)
    , m_bufferSize(9564160)
    , m_isReady(false)
    , m_detile(2464, 2464, 9564160)
    , m_activeDetile(nullptr) {}

DMACapture::~DMACapture() {
    release();
}

pid_t DMACapture::findCompositorPid() {
    FILE *pipe = popen("pgrep vrcompositor", "r");
    if (!pipe) return 0;
    pid_t pid = 0;
    if (fscanf(pipe, "%d", &pid) != 1) pid = 0;
    pclose(pipe);
    return pid;
}

bool DMACapture::autoDiscoverCameraGroups(pid_t compositorPid,
                                         std::vector<int>& outRightFds,
                                         std::vector<int>& outLeftFds) {
    outRightFds.clear();
    outLeftFds.clear();
    char fdDir[256];
    snprintf(fdDir, sizeof(fdDir), "/proc/%d/fd", compositorPid);

    DIR *dir = opendir(fdDir);
    if (!dir) return false;

    std::vector<CameraBuffer> camBuffers;
    struct dirent *ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        char linkPath[512];
        snprintf(linkPath, sizeof(linkPath), "/proc/%d/fd/%s", compositorPid, ent->d_name);
        char target[512] = {0};
        ssize_t len = readlink(linkPath, target, sizeof(target) - 1);
        if (len <= 0) continue;
        target[len] = '\0';
        struct stat st;
        if (strstr(target, "dmabuf") && stat(linkPath, &st) == 0 && st.st_size == 9564160) {
            camBuffers.push_back({atoi(ent->d_name), uint64_t(st.st_ino)});
        }
    }
    closedir(dir);
    if (!splitCameraGroups(camBuffers, VkYDetile::kMaxSlots, outRightFds, outLeftFds)) {
        std::cerr << "[DMACapture] Failed to find camera dmabufs (found " << camBuffers.size() << ")" << std::endl;
        return false;
    }
    if (camBuffers.size() > outRightFds.size() + outLeftFds.size())
        std::cout << "[DMACapture] Ignoring " << camBuffers.size() - outRightFds.size() - outLeftFds.size()
                  << " older camera dmabufs" << std::endl;
    std::cout << "[DMACapture] Right slots:";
    for (int fd : outRightFds) std::cout << " " << fd;
    std::cout << " | Left slots:";
    for (int fd : outLeftFds) std::cout << " " << fd;
    std::cout << std::endl;
    return true;
}

bool DMACapture::initialize(pid_t compositorPid, int rightFd, int leftFd) {
    release();
    m_compositorPid = compositorPid;

    std::vector<int> rightFds, leftFds;
    if (rightFd > 0 && leftFd > 0) {
        rightFds.push_back(rightFd);
        leftFds.push_back(leftFd);
    } else if (!autoDiscoverCameraGroups(m_compositorPid, rightFds, leftFds)) {
        return false;
    }

    m_pidFd = syscall(__NR_pidfd_open, m_compositorPid, 0);
    if (m_pidFd < 0) {
        perror("[DMACapture] pidfd_open failed");
        return false;
    }

    for (int fd : rightFds) {
        int local = syscall(__NR_pidfd_getfd, m_pidFd, fd, 0);
        if (local < 0) { perror("[DMACapture] pidfd_getfd right"); release(); return false; }
        m_localRightFds.push_back(local);
    }
    for (int fd : leftFds) {
        int local = syscall(__NR_pidfd_getfd, m_pidFd, fd, 0);
        if (local < 0) { perror("[DMACapture] pidfd_getfd left"); release(); return false; }
        m_localLeftFds.push_back(local);
    }

    if (!m_detile.importStereoGroups(m_localLeftFds, m_localRightFds)) {
        std::cerr << "[DMACapture] Vulkan UBWC import/detile failed" << std::endl;
        release();
        return false;
    }
    if (!m_detile.copyYPlanes()) {
        std::cerr << "[DMACapture] Initial Y-plane detile failed" << std::endl;
        release();
        return false;
    }
    m_activeDetile = &m_detile;

    m_isReady = true;
    std::cout << "[DMACapture] Hardware cameras bound and GPU-detiled" << std::endl;
    return true;
}

bool DMACapture::captureFrame(bool includeUV, bool leftOnly) {
    if (!m_isReady) return false;
    return m_detile.copyYPlanes(includeUV, leftOnly);
}

const uint8_t* DMACapture::getLeftBuffer() const { return m_activeDetile ? m_activeDetile->leftY() : nullptr; }
const uint8_t* DMACapture::getRightBuffer() const { return m_activeDetile ? m_activeDetile->rightY() : nullptr; }
const uint8_t* DMACapture::getLeftUV() const { return m_activeDetile ? m_activeDetile->leftUV() : nullptr; }
const uint8_t* DMACapture::getRightUV() const { return m_activeDetile ? m_activeDetile->rightUV() : nullptr; }
uint64_t DMACapture::leftHash() const { return m_activeDetile ? m_activeDetile->leftHash() : 0; }
uint64_t DMACapture::rightHash() const { return m_activeDetile ? m_activeDetile->rightHash() : 0; }
int DMACapture::leftSlot() const { return m_activeDetile ? m_activeDetile->leftSlot() : -1; }
int DMACapture::rightSlot() const { return m_activeDetile ? m_activeDetile->rightSlot() : -1; }
int DMACapture::width() const { return m_activeDetile ? m_activeDetile->width() : 0; }
int DMACapture::height() const { return m_activeDetile ? m_activeDetile->height() : 0; }

void DMACapture::release() {
    m_detile.release();
    m_activeDetile = nullptr;
    m_isReady = false;
    for (int fd : m_localRightFds) if (fd >= 0) close(fd);
    for (int fd : m_localLeftFds) if (fd >= 0) close(fd);
    m_localRightFds.clear();
    m_localLeftFds.clear();
    if (m_pidFd >= 0) close(m_pidFd);
    m_pidFd = -1;
}

} // namespace FrameSpatial
