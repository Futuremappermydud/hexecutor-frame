#pragma once
#include "hexecutor/cdsp_mem.h"
#include "hexecutor/executor.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hexecutor {

// Workload types executed on the CDSP
enum class WorkloadType {
    Depth,          // Shipped 128 MiB depth pipeline
    HandTracking,   // NIR / Mercury hand tracking (128 MiB)
    TensorGraph,    // HXFG tensor float32 models
    Qnn16,          // Lowered QNN FP16 models
    LLM,            // Liquid LFM3 / Qwen language models (up to 1.5 GiB)
    Custom          // Explicit size requirement
};

// Computes the memory requirement in MiB for a set of programs.
// The result is aligned to 16 MiB (matching Hexagon JTLB 16 MiB page slots)
// and clamped to [CDSP_MEM_MIN_SIZE_MB (128), CDSP_MEM_MAX_SIZE_MB (1536)].
size_t calculate_workload_size_mb(const std::vector<Program> &programs,
                                  const Artifacts &artifacts);

class WorkloadMemoryManager;

// RAII lease for a workload's CDSP memory reservation.
// Automatically ensures the required memory is reserved upon acquisition,
// and restores/shrinks memory when released or destroyed.
class WorkloadLease {
  public:
    WorkloadLease() = default;
    WorkloadLease(WorkloadMemoryManager &manager, size_t required_mb,
                  bool auto_shrink_on_release = true);
    ~WorkloadLease();

    WorkloadLease(const WorkloadLease &) = delete;
    WorkloadLease &operator=(const WorkloadLease &) = delete;
    WorkloadLease(WorkloadLease &&other) noexcept;
    WorkloadLease &operator=(WorkloadLease &&other) noexcept;

    size_t size_mb() const { return size_mb_; }
    size_t size_bytes() const { return size_mb_ << 20; }
    bool active() const { return active_; }
    void release();

  private:
    WorkloadMemoryManager *manager_ = nullptr;
    size_t size_mb_ = 0;
    bool auto_shrink_ = true;
    bool active_ = false;
};

// Manages dynamic CDSP memory growth and shrinking based on workloads.
class WorkloadMemoryManager {
  public:
    static WorkloadMemoryManager &instance();

    WorkloadMemoryManager();
    ~WorkloadMemoryManager() = default;

    // Current CDSP reservation in MiB (from hardware driver or mock state)
    size_t current_size_mb();

    // Dynamically reserve at least `target_mb` (grows or shrinks to match).
    // Bounds: [CDSP_MEM_MIN_SIZE_MB (128), CDSP_MEM_MAX_SIZE_MB (1536)].
    bool reserve(size_t target_mb);

    // Shrinks the CDSP reservation back down to `target_mb` (default 128 MiB baseline).
    bool shrink(size_t target_mb = CDSP_MEM_MIN_SIZE_MB);

    // Acquires an RAII lease for a workload.
    WorkloadLease acquire_lease(size_t required_mb, bool auto_shrink_on_release = true);

    // Release a lease and recalculate needed memory among active leases.
    void release_lease(size_t lease_mb, bool auto_shrink);

    // Baseline and maximum sizes
    size_t baseline_size_mb() const { return CDSP_MEM_MIN_SIZE_MB; }
    size_t max_size_mb() const { return CDSP_MEM_MAX_SIZE_MB; }

    // Persistent state pinning hook: prevents lease from shrinking below state region
    void set_persistent_state_pin(size_t state_mb);
    size_t persistent_state_pin() const;

    // Mock mode for tests (simulating kernel driver growth, shrinking, rollbacks, and IOVA)
    void enable_mock_mode(bool enable, size_t initial_mb = CDSP_MEM_MIN_SIZE_MB);
    bool is_mock_mode() const { return mock_mode_; }
    void set_mock_failure(bool fail_on_grow, bool fail_on_shrink = false);
    void set_mock_busy_on_shrink(bool busy);
    size_t mock_grow_count() const { return mock_grow_count_; }
    size_t mock_shrink_count() const { return mock_shrink_count_; }
    void reset_mock_stats();

  private:
    bool request_driver_resize(size_t target_mb);
    size_t read_driver_size_mb();
    size_t local_pinned_extent_mb_locked() const;

    std::mutex mutex_;
    size_t current_size_mb_ = CDSP_MEM_MIN_SIZE_MB;
    std::vector<size_t> active_leases_;
    size_t persistent_state_pin_mb_ = 0;

    // Mock driver state
    bool mock_mode_ = false;
    bool mock_fail_grow_ = false;
    bool mock_fail_shrink_ = false;
    bool mock_busy_shrink_ = false;
    size_t mock_grow_count_ = 0;
    size_t mock_shrink_count_ = 0;
};

} // namespace hexecutor
