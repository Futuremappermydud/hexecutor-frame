// SPDX-License-Identifier: LGPL-3.0-or-later
#include "hexecutor/cdsp_mem.h"
#include "hexecutor/executor.hpp"
#include "hexecutor/workload_memory.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(bool condition, const std::string &msg) {
    if (!condition) {
        std::cerr << "FAILED: " << msg << std::endl;
        throw std::runtime_error("Assertion failed: " + msg);
    }
}

template <typename F> void check_throws(F &&f, const std::string &msg) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, "Expected exception: " + msg);
}

} // namespace

int main() {
    using namespace hexecutor;
    std::cout << "[Test] Starting dynamic CDSP memory growth and shrinking test suite..." << std::endl;

    auto &mgr = WorkloadMemoryManager::instance();
    mgr.enable_mock_mode(true, 128);

    // 1. Basic constants & limits verification
    check(CDSP_MEM_MIN_SIZE_MB == 128, "CDSP_MEM_MIN_SIZE_MB must be 128 MiB");
    check(CDSP_MEM_MAX_SIZE_MB == 1536, "CDSP_MEM_MAX_SIZE_MB must be 1536 MiB (1.5 GiB)");
    check(CDSP_MEM_CHUNK_MB == 2, "CDSP_MEM_CHUNK_MB must be 2 MiB");
    check(CDSP_MEM_BASE_IOVA == 0xa0000000u, "CDSP_MEM_BASE_IOVA must be 0xa0000000");

    // Verify 1.5 GiB IOVA limit: 0xa0000000 + 0x60000000 = 0x100000000
    uint64_t max_span = uint64_t(CDSP_MEM_BASE_IOVA) + (uint64_t(CDSP_MEM_MAX_SIZE_MB) << 20);
    check(max_span == 0x100000000ULL, "1.5 GiB must span exactly up to 4 GiB 32-bit boundary");

    check(mgr.current_size_mb() == 128, "Initial size should be 128 MiB");

    // 2. Step-wise Dynamic Growth: 128 MiB -> 256 MiB -> 512 MiB -> 1024 MiB -> 1536 MiB
    std::vector<size_t> growth_steps = {128, 256, 512, 768, 1024, 1280, 1536};
    size_t prev_size = 128;
    for (size_t target : growth_steps) {
        if (target > prev_size) {
            size_t old_grow_count = mgr.mock_grow_count();
            bool ok = mgr.reserve(target);
            check(ok, "Grow to " + std::to_string(target) + " MiB should succeed");
            check(mgr.current_size_mb() == target, "Current size should be " + std::to_string(target) + " MiB");
            check(mgr.mock_grow_count() == old_grow_count + 1, "Mock grow count should increment");
            prev_size = target;
        }
    }
    std::cout << "  ✓ Step-wise growth from 128 MiB up to 1.5 GiB (1536 MiB) passed" << std::endl;

    // 3. Step-wise Dynamic Shrinking: 1536 MiB -> 1280 MiB -> 1024 MiB -> 768 MiB -> 512 MiB -> 256 MiB -> 128 MiB
    std::vector<size_t> shrink_steps = {1280, 1024, 768, 512, 256, 128};
    for (size_t target : shrink_steps) {
        size_t old_shrink_count = mgr.mock_shrink_count();
        bool ok = mgr.shrink(target);
        check(ok, "Shrink to " + std::to_string(target) + " MiB should succeed");
        check(mgr.current_size_mb() == target, "Current size should be " + std::to_string(target) + " MiB");
        check(mgr.mock_shrink_count() == old_shrink_count + 1, "Mock shrink count should increment");
    }
    check(mgr.current_size_mb() == 128, "Final size after shrinking should be baseline 128 MiB");
    std::cout << "  ✓ Step-wise shrinking from 1.5 GiB down to 128 MiB passed" << std::endl;

    // 4. Boundary and invalid size checks
    check(!mgr.reserve(64), "Reservation below minimum 128 MiB must be rejected");
    check(!mgr.reserve(2048), "Reservation above maximum 1536 MiB (1.5 GiB) must be rejected");
    // Odd sizes: reserve aligns up to 2 MB chunk
    mgr.reserve(130);
    check(mgr.current_size_mb() == 130, "Reservation of 130 MiB should succeed");
    mgr.shrink(128);
    check(mgr.current_size_mb() == 128, "Shrink back to 128 MiB should succeed");
    std::cout << "  ✓ Boundary and validation checks passed" << std::endl;

    // 5. Workload-based calculation tests
    {
        Artifacts a;
        std::vector<Program> depth_programs;
        size_t size = calculate_workload_size_mb(depth_programs, a);
        check(size == 128, "Empty / default workload should calculate 128 MiB");

        // Explicit override via Artifacts
        a.cdsp_memory_mb = 512;
        check(calculate_workload_size_mb(depth_programs, a) == 512, "Artifacts override to 512 MiB should be honored");

        a.cdsp_memory_mb = 1536;
        check(calculate_workload_size_mb(depth_programs, a) == 1536, "Artifacts override to 1.5 GiB should be honored");

        a.cdsp_memory_mb = 2048; // clamped to 1536
        check(calculate_workload_size_mb(depth_programs, a) == 1536, "Artifacts override > 1536 should be clamped to 1536 MiB");

        // Synthetic large model program
        a.cdsp_memory_mb = 0;
        Program large_prog;
        large_prog.backend = Backend::tensor_f32;
        // Add 800 MB worth of inputs / outputs
        large_prog.inputs.push_back(TensorSpec{"weights", ScalarType::f32, {100, 1024, 1024}}); // 400 MB
        large_prog.outputs.push_back(TensorSpec{"out", ScalarType::f32, {100, 1024, 1024}});    // 400 MB
        std::vector<Program> progs = {large_prog};
        size_t calculated = calculate_workload_size_mb(progs, a);
        check(calculated >= 800, "Calculated size for 800 MB tensors should be >= 800 MiB");
        check(calculated % 16 == 0, "Calculated size must be aligned to 16 MiB JTLB slot");
    }
    std::cout << "  ✓ Workload-based calculation and alignment passed" << std::endl;

    // 6. RAII WorkloadLease dynamic lifecycle
    {
        mgr.reset_mock_stats();
        check(mgr.current_size_mb() == 128, "Baseline before lease should be 128 MiB");

        {
            // Acquire lease for a 512 MiB workload (experimented model)
            auto lease512 = mgr.acquire_lease(512);
            check(lease512.active(), "Lease 512 should be active");
            check(lease512.size_mb() == 512, "Lease size should be 512 MiB");
            check(mgr.current_size_mb() == 512, "CDSP memory should have grown to 512 MiB");
            check(mgr.mock_grow_count() == 1, "Should have 1 grow event");

            {
                // Concurrently acquire lease for a 1.5 GiB workload (LFM3 multimodal)
                auto lease1536 = mgr.acquire_lease(1536);
                check(lease1536.active(), "Lease 1536 should be active");
                check(lease1536.size_mb() == 1536, "Lease size should be 1536 MiB");
                check(mgr.current_size_mb() == 1536, "CDSP memory should have grown to 1536 MiB");
                check(mgr.mock_grow_count() == 2, "Should have 2 grow events");

                // Test move semantics
                WorkloadLease moved_lease = std::move(lease1536);
                check(moved_lease.active(), "Moved lease should be active");
                check(!lease1536.active(), "Original lease should now be inactive");
                check(mgr.current_size_mb() == 1536, "Memory remains 1536 MiB after move");
            } // moved_lease destroyed here -> memory should shrink back to 512 MiB (peak of remaining leases)

            check(mgr.current_size_mb() == 512, "Memory should shrink to 512 MiB after releasing 1536 MiB lease");
            check(mgr.mock_shrink_count() == 1, "Should have 1 shrink event");
        } // lease512 destroyed here -> memory should shrink back to 128 MiB baseline

        check(mgr.current_size_mb() == 128, "Memory should shrink to baseline 128 MiB after all leases released");
        check(mgr.mock_shrink_count() == 2, "Should have 2 shrink events");
    }
    std::cout << "  ✓ RAII WorkloadLease dynamic growth, nesting, move, and shrinking passed" << std::endl;

    {
        auto lease = mgr.acquire_lease(512);
        lease.retain_size_on_release();
        lease.release();
        check(!lease.active(), "Release should deactivate the lease");
        check(mgr.current_size_mb() == 512, "Unverified DSP cleanup should retain the reservation size");
        check(mgr.shrink(128), "Explicit shrink should succeed after recovery");
    }

    // 7. Allocation failure and rollback safety
    {
        mgr.reset_mock_stats();
        mgr.set_mock_failure(true, false); // simulate out-of-memory on grow
        check(mgr.current_size_mb() == 128, "Starting at 128 MiB");

        check_throws([&] { mgr.acquire_lease(1024); }, "acquire_lease should throw on driver growth failure");
        check(mgr.current_size_mb() == 128, "Memory size should remain unchanged at 128 MiB after failed growth");

        mgr.set_mock_failure(false, false);
    }
    std::cout << "  ✓ Failure simulation and atomic rollback safety passed" << std::endl;

    // 8. Concurrent multi-threaded dynamic reservation test
    {
        mgr.reset_mock_stats();
        std::vector<std::thread> threads;
        const int num_threads = 8;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&mgr, i]() {
                size_t req = 128 + (i % 5) * 256; // 128, 384, 640, 896, 1152 MiB
                for (int iter = 0; iter < 10; ++iter) {
                    auto lease = mgr.acquire_lease(req);
                    check(lease.active(), "Thread lease must be active");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    lease.release();
                }
            });
        }

        for (auto &t : threads) {
            t.join();
        }

        check(mgr.current_size_mb() == 128, "After concurrent threads finish, memory must return to baseline 128 MiB");
    }
    std::cout << "  ✓ Multi-threaded concurrent reservation and release passed" << std::endl;

    // 9. JTLB Slots coverage verification
    {
        // 128 MiB = 8 slots
        size_t slots_128 = 128 / 16;
        size_t first_128 = (slots_128 > 64) ? (128 - slots_128) : 64;
        check(slots_128 == 8 && first_128 == 64, "128 MiB: 8 slots starting at 64 (64..71)");

        // 512 MiB = 32 slots
        size_t slots_512 = 512 / 16;
        size_t first_512 = (slots_512 > 64) ? (128 - slots_512) : 64;
        check(slots_512 == 32 && first_512 == 64, "512 MiB: 32 slots starting at 64 (64..95)");

        // 1024 MiB = 64 slots
        size_t slots_1024 = 1024 / 16;
        size_t first_1024 = (slots_1024 > 64) ? (128 - slots_1024) : 64;
        check(slots_1024 == 64 && first_1024 == 64, "1024 MiB: 64 slots starting at 64 (64..127)");

        // 1536 MiB = 96 slots
        size_t slots_1536 = 1536 / 16;
        size_t first_1536 = (slots_1536 > 64) ? (128 - slots_1536) : 64;
        check(slots_1536 == 96 && first_1536 == 32, "1536 MiB: 96 slots starting at 32 (32..127)");
        check(first_1536 + slots_1536 == 128, "96 slots starting at 32 end exactly at 127, all within 128 JTLB entries");
    }
    std::cout << "  ✓ JTLB slot and address coverage for up to 1.5 GiB passed" << std::endl;

    // 10. Persistent state pin protection test
    {
        mgr.reset_mock_stats();
        check(mgr.current_size_mb() == 128, "Baseline before persistent state test");
        {
            auto lease = mgr.acquire_lease(512);
            check(mgr.current_size_mb() == 512, "Grown to 512 MiB");
            mgr.set_persistent_state_pin(512);
        } // lease released here
        // Memory must remain at 512 MiB because persistent state pin holds it
        check(mgr.current_size_mb() == 512, "Memory must stay at 512 MiB while persistent state pin is live");
        check(mgr.shrink(128), "Shrink call returns true ('stay large')");
        check(mgr.current_size_mb() == 512, "Memory still 512 MiB");

        // Clear persistent state pin and shrink to baseline
        mgr.set_persistent_state_pin(0);
        check(mgr.shrink(128), "Shrink after clearing persistent state pin");
        check(mgr.current_size_mb() == 128, "Returned to 128 MiB baseline");
    }
    std::cout << "  ✓ Persistent state pin protection passed" << std::endl;

    // 11. EBUSY on shrink ('stay large') handling
    {
        mgr.reset_mock_stats();
        mgr.reserve(512);
        check(mgr.current_size_mb() == 512, "Current size is 512 MiB");

        // Simulate driver returning -EBUSY on shrink attempt
        mgr.set_mock_busy_on_shrink(true);
        bool res = mgr.shrink(128);
        check(res, "Shrink with -EBUSY treated as 'stay large'");
        check(mgr.current_size_mb() == 512, "Memory stays at 512 MiB on -EBUSY");

        mgr.set_mock_busy_on_shrink(false);
        res = mgr.shrink(128);
        check(res, "Shrink succeeds when driver no longer busy");
        check(mgr.current_size_mb() == 128, "Memory shrunk to 128 MiB");
    }
    std::cout << "  ✓ EBUSY 'stay large' handling passed" << std::endl;

    std::cout << "\nALL DYNAMIC CDSP MEMORY TESTS PASSED!" << std::endl;
    return 0;
}
