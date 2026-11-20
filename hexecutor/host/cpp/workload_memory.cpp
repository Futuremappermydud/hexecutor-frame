// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/workload_memory.hpp"
#include "hexecutor/cdsp_mem.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/ioctl.h>
#endif

namespace hexecutor {

size_t calculate_workload_size_mb(const std::vector<Program> &programs,
                                  const Artifacts &artifacts) {
    size_t base_mb = CDSP_MEM_MIN_SIZE_MB;

    if (artifacts.cdsp_memory_mb > 0)
        base_mb = std::max(base_mb, artifacts.cdsp_memory_mb);

    const char *env_mb = std::getenv("HEXECUTOR_CDSP_MEM_MB");
    if (env_mb) {
        try {
            size_t val = std::stoul(env_mb);
            base_mb = std::max(base_mb, val);
        } catch (...) {
        }
    }

    if (programs.empty())
        return std::clamp(base_mb, size_t(CDSP_MEM_MIN_SIZE_MB), size_t(CDSP_MEM_MAX_SIZE_MB));

    // Calculate required memory for tensor / QNN16 workloads
    size_t total_bytes = 0;
    // Engine ELF reservation: 1 MiB
    total_bytes += 1u << 20;

    for (const auto &p : programs) {
        if (!p.image.empty() && std::filesystem::exists(p.image)) {
            std::error_code ec;
            total_bytes += std::filesystem::file_size(p.image, ec);
        }
        if (!p.constants.empty() && std::filesystem::exists(p.constants)) {
            std::error_code ec;
            total_bytes += std::filesystem::file_size(p.constants, ec);
        }
        for (const auto &in : p.inputs)
            total_bytes += in.bytes();
        for (const auto &out : p.outputs)
            total_bytes += out.bytes();
    }

    // Overhead for arenas, stacks, L2 cache lease, near state, and canaries: 32 MiB
    total_bytes += 32u << 20;

    // Convert to MiB and align up to 16 MiB boundary (matching JTLB slot size)
    size_t calculated_mb = (total_bytes + (16u << 20) - 1) / (16u << 20) * 16;
    base_mb = std::max(base_mb, calculated_mb);

    return std::clamp(base_mb, size_t(CDSP_MEM_MIN_SIZE_MB), size_t(CDSP_MEM_MAX_SIZE_MB));
}

WorkloadMemoryManager &WorkloadMemoryManager::instance() {
    static WorkloadMemoryManager inst;
    return inst;
}

WorkloadMemoryManager::WorkloadMemoryManager() {
    current_size_mb_ = read_driver_size_mb();
}

size_t WorkloadMemoryManager::current_size_mb() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!mock_mode_)
        current_size_mb_ = read_driver_size_mb();
    return current_size_mb_;
}

size_t WorkloadMemoryManager::read_driver_size_mb() {
    if (mock_mode_)
        return current_size_mb_;

#ifdef __linux__
    std::ifstream f(CDSP_MEM_SYSFS_SIZE);
    if (f.is_open()) {
        size_t mb = 0;
        if (f >> mb && mb >= CDSP_MEM_MIN_SIZE_MB)
            return mb;
    }

    int fd = ::open(CDSP_MEM_DEV_PATH, O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
        uint32_t mb = 0;
        if (::ioctl(fd, CDSP_MEM_IOC_GET_SIZE, &mb) == 0 && mb >= CDSP_MEM_MIN_SIZE_MB) {
            ::close(fd);
            return mb;
        }
        ::close(fd);
    }
#endif

    return current_size_mb_;
}

size_t WorkloadMemoryManager::local_pinned_extent_mb_locked() const {
    size_t extent = CDSP_MEM_MIN_SIZE_MB;
    if (!active_leases_.empty()) {
        extent = std::max(extent, *std::max_element(active_leases_.begin(), active_leases_.end()));
    }
    if (persistent_state_pin_mb_ > 0) {
        extent = std::max(extent, persistent_state_pin_mb_);
    }
    return extent;
}

void WorkloadMemoryManager::set_persistent_state_pin(size_t state_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    persistent_state_pin_mb_ = state_mb;
}

size_t WorkloadMemoryManager::persistent_state_pin() const {
    return persistent_state_pin_mb_;
}

void WorkloadMemoryManager::set_mock_busy_on_shrink(bool busy) {
    std::lock_guard<std::mutex> lock(mutex_);
    mock_busy_shrink_ = busy;
}

bool WorkloadMemoryManager::request_driver_resize(size_t target_mb) {
    if (target_mb < CDSP_MEM_MIN_SIZE_MB || target_mb > CDSP_MEM_MAX_SIZE_MB || target_mb % 2 != 0)
        return false;

    if (mock_mode_) {
        if (target_mb > current_size_mb_ && mock_fail_grow_)
            return false;
        if (target_mb < current_size_mb_) {
            if (mock_busy_shrink_) {
                // Driver returns -EBUSY: treat as 'stay large'
                return true;
            }
            if (mock_fail_shrink_)
                return false;
            mock_shrink_count_++;
        } else if (target_mb > current_size_mb_) {
            mock_grow_count_++;
        }

        current_size_mb_ = target_mb;
        return true;
    }

#ifdef __linux__
    const bool is_shrink = target_mb < current_size_mb_;
    const bool is_grow = target_mb > current_size_mb_;

    // Try ioctl on /dev/cdsp_mem
    int fd = ::open(CDSP_MEM_DEV_PATH, O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
        uint32_t val = static_cast<uint32_t>(target_mb);
        int ret = ::ioctl(fd, CDSP_MEM_IOC_SET_SIZE, &val);
        int err = errno;
        ::close(fd);
        if (ret == 0) {
            size_t new_size = read_driver_size_mb();
            if (is_grow && new_size < target_mb)
                return false;
            current_size_mb_ = new_size;
            return true;
        }
        if (is_shrink && err == EBUSY) {
            // Driver refused shrink because another process or active mapping pins the window.
            // Treat -EBUSY on shrink as 'stay large'.
            current_size_mb_ = read_driver_size_mb();
            return true;
        }
        // A module without the resize ioctl: fall back to the sysfs parameter.
        // Anything else (EPERM, EBUSY on a grow, ...) is a real refusal.
        if (err != ENOTTY && err != EINVAL && err != ENOSYS)
            return false;
    }

    // Try sysfs parameter write
    int sfd = ::open(CDSP_MEM_SYSFS_SIZE, O_WRONLY | O_CLOEXEC);
    if (sfd >= 0) {
        std::string s = std::to_string(target_mb) + "\n";
        ssize_t written = ::write(sfd, s.data(), s.size());
        int err = errno;
        ::close(sfd);
        if (written == static_cast<ssize_t>(s.size())) {
            size_t new_size = read_driver_size_mb();
            if (is_grow && new_size < target_mb)
                return false;
            current_size_mb_ = new_size;
            return true;
        }
        if (is_shrink && err == EBUSY) {
            // Re-read size and stay large on EBUSY shrink
            current_size_mb_ = read_driver_size_mb();
            return true;
        }
        return false;
    }
#else
    // On non-Linux (e.g. macOS dev environment), fallback to tracked state
    current_size_mb_ = target_mb;
    return true;
#endif

    return false;
}

bool WorkloadMemoryManager::reserve(size_t target_mb) {
    if (target_mb < CDSP_MEM_MIN_SIZE_MB || target_mb > CDSP_MEM_MAX_SIZE_MB)
        return false;

    std::lock_guard<std::mutex> lock(mutex_);
    target_mb = (target_mb + 1) / 2 * 2; // align to 2 MB chunk

    if (target_mb == current_size_mb_)
        return true;

    return request_driver_resize(target_mb);
}

bool WorkloadMemoryManager::shrink(size_t target_mb) {
    if (target_mb < CDSP_MEM_MIN_SIZE_MB || target_mb > CDSP_MEM_MAX_SIZE_MB)
        return false;

    std::lock_guard<std::mutex> lock(mutex_);
    target_mb = (target_mb + 1) / 2 * 2;

    // Never shrink while this process still has a mapping or a pin (active lease or persistent state)
    size_t local_pinned = local_pinned_extent_mb_locked();
    if (target_mb < local_pinned) {
        // Refuse shrink below local pinned extent: stay large
        return true;
    }

    if (target_mb >= current_size_mb_)
        return true;

    return request_driver_resize(target_mb);
}

WorkloadLease WorkloadMemoryManager::acquire_lease(size_t required_mb, bool auto_shrink_on_release) {
    std::lock_guard<std::mutex> lock(mutex_);
    required_mb = (required_mb + 15) / 16 * 16; // align to 16 MB JTLB slot
    required_mb = std::clamp(required_mb, size_t(CDSP_MEM_MIN_SIZE_MB), size_t(CDSP_MEM_MAX_SIZE_MB));

    active_leases_.push_back(required_mb);
    size_t peak_required = *std::max_element(active_leases_.begin(), active_leases_.end());
    if (persistent_state_pin_mb_ > 0)
        peak_required = std::max(peak_required, persistent_state_pin_mb_);

    if (peak_required > current_size_mb_) {
        if (!request_driver_resize(peak_required)) {
            active_leases_.pop_back();
            throw std::runtime_error("failed to dynamically grow CDSP memory lease to " +
                                     std::to_string(peak_required) + " MB");
        }
    }

    return WorkloadLease(*this, required_mb, auto_shrink_on_release);
}

void WorkloadMemoryManager::release_lease(size_t lease_mb, bool auto_shrink) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(active_leases_.begin(), active_leases_.end(), lease_mb);
    if (it != active_leases_.end())
        active_leases_.erase(it);

    if (auto_shrink) {
        size_t target_mb = local_pinned_extent_mb_locked();
        if (target_mb < current_size_mb_)
            request_driver_resize(target_mb);
    }
}

void WorkloadMemoryManager::enable_mock_mode(bool enable, size_t initial_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    mock_mode_ = enable;
    current_size_mb_ = std::clamp(initial_mb, size_t(CDSP_MEM_MIN_SIZE_MB), size_t(CDSP_MEM_MAX_SIZE_MB));
    mock_fail_grow_ = false;
    mock_fail_shrink_ = false;
    mock_busy_shrink_ = false;
    persistent_state_pin_mb_ = 0;
    mock_grow_count_ = 0;
    mock_shrink_count_ = 0;
    active_leases_.clear();
}

void WorkloadMemoryManager::set_mock_failure(bool fail_on_grow, bool fail_on_shrink) {
    std::lock_guard<std::mutex> lock(mutex_);
    mock_fail_grow_ = fail_on_grow;
    mock_fail_shrink_ = fail_on_shrink;
}

void WorkloadMemoryManager::reset_mock_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    mock_grow_count_ = 0;
    mock_shrink_count_ = 0;
}

// WorkloadLease implementation
WorkloadLease::WorkloadLease(WorkloadMemoryManager &manager, size_t required_mb,
                             bool auto_shrink_on_release)
    : manager_(&manager), size_mb_(required_mb), auto_shrink_(auto_shrink_on_release), active_(true) {}

WorkloadLease::~WorkloadLease() {
    release();
}

WorkloadLease::WorkloadLease(WorkloadLease &&other) noexcept
    : manager_(other.manager_), size_mb_(other.size_mb_), auto_shrink_(other.auto_shrink_),
      active_(other.active_) {
    other.active_ = false;
    other.manager_ = nullptr;
}

WorkloadLease &WorkloadLease::operator=(WorkloadLease &&other) noexcept {
    if (this != &other) {
        release();
        manager_ = other.manager_;
        size_mb_ = other.size_mb_;
        auto_shrink_ = other.auto_shrink_;
        active_ = other.active_;
        other.active_ = false;
        other.manager_ = nullptr;
    }
    return *this;
}

void WorkloadLease::release() {
    if (active_ && manager_) {
        manager_->release_lease(size_mb_, auto_shrink_);
        active_ = false;
        manager_ = nullptr;
    }
}

} // namespace hexecutor
