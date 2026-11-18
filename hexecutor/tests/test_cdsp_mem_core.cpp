#include "cdsp_mem_core.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
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

struct FakeOpsContext {
    std::mutex lock;
    std::set<uint32_t> allocated_chunks;
    std::set<uint64_t> mapped_iovas;
    size_t flush_count = 0;

    int fail_alloc_idx = -1;
    int fail_map_idx = -1;
};

void *test_alloc_chunk(void *ctx, uint32_t chunk_idx) {
    auto *s = static_cast<FakeOpsContext *>(ctx);
    std::lock_guard<std::mutex> guard(s->lock);
    if (s->fail_alloc_idx >= 0 && chunk_idx == static_cast<uint32_t>(s->fail_alloc_idx)) {
        return nullptr;
    }
    s->allocated_chunks.insert(chunk_idx);
    // Non-null opaque pointer handle
    return reinterpret_cast<void *>(static_cast<uintptr_t>(chunk_idx + 1));
}

void test_free_chunk(void *ctx, void *chunk_handle, uint32_t chunk_idx) {
    auto *s = static_cast<FakeOpsContext *>(ctx);
    std::lock_guard<std::mutex> guard(s->lock);
    (void)chunk_handle;
    s->allocated_chunks.erase(chunk_idx);
}

int test_iommu_map(void *ctx, uint64_t iova, void *chunk_handle, uint32_t chunk_idx) {
    auto *s = static_cast<FakeOpsContext *>(ctx);
    std::lock_guard<std::mutex> guard(s->lock);
    (void)chunk_handle;
    if (s->fail_map_idx >= 0 && chunk_idx == static_cast<uint32_t>(s->fail_map_idx)) {
        return -5; // EIO
    }
    s->mapped_iovas.insert(iova);
    return 0;
}

void test_iommu_unmap(void *ctx, uint64_t iova, uint32_t chunk_idx) {
    auto *s = static_cast<FakeOpsContext *>(ctx);
    std::lock_guard<std::mutex> guard(s->lock);
    (void)chunk_idx;
    s->mapped_iovas.erase(iova);
}

void test_flush_iotlb(void *ctx) {
    auto *s = static_cast<FakeOpsContext *>(ctx);
    std::lock_guard<std::mutex> guard(s->lock);
    s->flush_count++;
}

void *test_alloc_meta(void *ctx, size_t bytes) {
    (void)ctx;
    return std::calloc(1, bytes);
}

void test_free_meta(void *ctx, void *ptr) {
    (void)ctx;
    std::free(ptr);
}

const struct cdsp_mem_ops fake_ops = {
    test_alloc_chunk,
    test_free_chunk,
    test_iommu_map,
    test_iommu_unmap,
    test_flush_iotlb,
    test_alloc_meta,
    test_free_meta,
};

} // namespace

int main() {
    std::cout << "[Test] Starting pure C cdsp_mem_core test suite..." << std::endl;

    FakeOpsContext ctx;
    struct cdsp_mem_core core {};

    // 1. Basic initialization & constants
    int ret = cdsp_mem_core_init(&core, 128, CDSP_MEM_BASE_IOVA, &fake_ops, &ctx);
    check(ret == 0, "Core initialization at 128 MiB baseline must succeed");
    check(core.size_mb == 128, "Size must be 128 MiB");
    check(core.nchunks == 64, "128 MiB must have 64 chunks (2 MiB each)");
    check(ctx.allocated_chunks.size() == 64, "Exactly 64 chunks must be allocated");
    check(ctx.mapped_iovas.size() == 64, "Exactly 64 chunks must be mapped in IOMMU");
    check(cdsp_mem_core_get_pinned_extent_mb(&core) == 128, "Baseline pinned extent is 128 MiB");
    std::cout << "  ✓ Initialization and baseline allocation passed" << std::endl;

    // 2. Validation bounds checks
    check(cdsp_mem_core_resize(&core, 64) == -EINVAL, "Resize below 128 MiB must return -EINVAL");
    check(cdsp_mem_core_resize(&core, 2048) == -EINVAL, "Resize above 1536 MiB must return -EINVAL");
    check(cdsp_mem_core_resize(&core, 255) == -EINVAL, "Resize to odd MB must return -EINVAL");
    check(core.size_mb == 128, "Size unchanged after invalid resizes");
    std::cout << "  ✓ Validation bounds checks passed" << std::endl;

    // 3. Step-wise Growth: 128 -> 256 -> 512 -> 768 -> 1024 -> 1280 -> 1536 MiB
    std::vector<uint32_t> grow_targets = {256, 512, 768, 1024, 1280, 1536};
    for (uint32_t target : grow_targets) {
        ret = cdsp_mem_core_resize(&core, target);
        check(ret == 0, "Grow to " + std::to_string(target) + " MiB must succeed");
        check(core.size_mb == target, "core.size_mb must match target");
        check(core.nchunks == target / 2, "nchunks must match target / 2");
        check(ctx.allocated_chunks.size() == target / 2, "allocated_chunks must match target / 2");
        check(ctx.mapped_iovas.size() == target / 2, "mapped_iovas must match target / 2");
    }
    check(ctx.allocated_chunks.size() == 768, "At 1536 MiB, exactly 768 chunks allocated");
    std::cout << "  ✓ Step-wise growth from 128 MiB to 1.5 GiB passed" << std::endl;

    // 4. Step-wise Shrinking: 1536 -> 1280 -> 1024 -> 768 -> 512 -> 256 -> 128 MiB
    std::vector<uint32_t> shrink_targets = {1280, 1024, 768, 512, 256, 128};
    for (uint32_t target : shrink_targets) {
        ret = cdsp_mem_core_resize(&core, target);
        check(ret == 0, "Shrink to " + std::to_string(target) + " MiB must succeed");
        check(core.size_mb == target, "core.size_mb must match target");
        check(core.nchunks == target / 2, "nchunks must match target / 2");
        check(ctx.allocated_chunks.size() == target / 2, "allocated_chunks must match target / 2");
        check(ctx.mapped_iovas.size() == target / 2, "mapped_iovas must match target / 2");
    }
    check(ctx.allocated_chunks.size() == 64, "At 128 MiB baseline, exactly 64 chunks remaining");
    std::cout << "  ✓ Step-wise shrinking from 1.5 GiB down to 128 MiB passed" << std::endl;

    // 5. Atomic Rollback on Injected Failure at EVERY chunk index: 128 MiB -> 256 MiB
    // Target has 64 new chunks (indices 64 to 127).
    std::cout << "  ... Testing atomic rollback on failure at every chunk index (64..127)..." << std::endl;
    for (int fail_idx = 64; fail_idx < 128; fail_idx++) {
        // (a) Test failure in alloc_chunk
        ctx.fail_alloc_idx = fail_idx;
        ctx.fail_map_idx = -1;
        ret = cdsp_mem_core_resize(&core, 256);
        check(ret == -ENOMEM, "Grow must fail with -ENOMEM when alloc fails at index " + std::to_string(fail_idx));
        check(core.size_mb == 128, "Size must remain 128 MiB after failed alloc at " + std::to_string(fail_idx));
        check(core.nchunks == 64, "nchunks must remain 64");
        check(ctx.allocated_chunks.size() == 64, "No chunk leak on failed alloc at " + std::to_string(fail_idx));
        check(ctx.mapped_iovas.size() == 64, "No IOVA mapping leak on failed alloc at " + std::to_string(fail_idx));

        // (b) Test failure in iommu_map
        ctx.fail_alloc_idx = -1;
        ctx.fail_map_idx = fail_idx;
        ret = cdsp_mem_core_resize(&core, 256);
        check(ret < 0, "Grow must fail when iommu_map fails at index " + std::to_string(fail_idx));
        check(core.size_mb == 128, "Size must remain 128 MiB after failed map at " + std::to_string(fail_idx));
        check(core.nchunks == 64, "nchunks must remain 64");
        check(ctx.allocated_chunks.size() == 64, "No chunk leak on failed map at " + std::to_string(fail_idx));
        check(ctx.mapped_iovas.size() == 64, "No IOVA mapping leak on failed map at " + std::to_string(fail_idx));
    }
    ctx.fail_alloc_idx = -1;
    ctx.fail_map_idx = -1;
    std::cout << "  ✓ Atomic rollback on injected failure verified at all 64 chunk indices" << std::endl;

    // 6. Live User Mappings (VMA) and -EBUSY Rules
    {
        // Grow to 1024 MiB
        check(cdsp_mem_core_resize(&core, 1024) == 0, "Grow to 1024 MiB");

        void *vma1 = reinterpret_cast<void *>(0x1000);
        void *vma2 = reinterpret_cast<void *>(0x2000);

        // VMA 1 maps 0..512 MiB (extent 512 MiB)
        ret = cdsp_mem_core_map_add(&core, vma1, 0, 512ULL << 20);
        check(ret == 0, "Add VMA 1 mapping of 512 MiB");
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 512, "Pinned extent must now be 512 MiB");

        // Shrink below 512 MiB must be refused with -EBUSY
        check(cdsp_mem_core_resize(&core, 256) == -EBUSY, "Shrink to 256 MiB must return -EBUSY");
        check(cdsp_mem_core_resize(&core, 128) == -EBUSY, "Shrink to 128 MiB must return -EBUSY");

        // Shrinking down to or above 512 MiB must succeed
        check(cdsp_mem_core_resize(&core, 768) == 0, "Shrink down to 768 MiB (above 512) must succeed");
        check(cdsp_mem_core_resize(&core, 512) == 0, "Shrink down to exact pinned extent (512 MiB) must succeed");
        check(core.size_mb == 512, "Core size should now be 512 MiB");

        // Growing is always allowed
        check(cdsp_mem_core_resize(&core, 1024) == 0, "Grow back to 1024 MiB must succeed");

        // VMA 2 maps 0..768 MiB (extent 768 MiB) - e.g. after fork or split
        ret = cdsp_mem_core_map_add(&core, vma2, 0, 768ULL << 20);
        check(ret == 0, "Add VMA 2 mapping of 768 MiB");
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 768, "Pinned extent must now be 768 MiB");

        // Shrinking to 512 MiB now blocked by VMA 2
        check(cdsp_mem_core_resize(&core, 512) == -EBUSY, "Shrink to 512 MiB must return -EBUSY due to VMA 2");
        check(cdsp_mem_core_resize(&core, 768) == 0, "Shrink to 768 MiB must succeed");

        // Close VMA 2 -> pinned extent drops back to 512 MiB
        cdsp_mem_core_map_remove(&core, vma2);
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 512, "Pinned extent drops to 512 MiB after VMA 2 close");
        check(cdsp_mem_core_resize(&core, 512) == 0, "Shrink to 512 MiB now succeeds");

        // Close VMA 1 -> pinned extent drops to 128 MiB baseline
        cdsp_mem_core_map_remove(&core, vma1);
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 128, "Pinned extent drops to 128 MiB baseline");
        check(cdsp_mem_core_resize(&core, 128) == 0, "Shrink to 128 MiB baseline now succeeds");
    }
    std::cout << "  ✓ Live mapping extent tracking and -EBUSY rules passed" << std::endl;

    // 7. DSP-window PIN / UNPIN and -EBUSY Rules
    {
        // Grow to 1024 MiB
        check(cdsp_mem_core_resize(&core, 1024) == 0, "Grow to 1024 MiB");

        void *fileA = reinterpret_cast<void *>(0xaaaa);
        void *fileB = reinterpret_cast<void *>(0xbbbb);

        // Pin owner fileA at 1024 MiB
        ret = cdsp_mem_core_pin(&core, fileA, 1024);
        check(ret == 0, "Pin fileA at 1024 MiB");
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 1024, "Pinned extent is 1024 MiB");

        // Attempt shrink below 1024 MiB
        check(cdsp_mem_core_resize(&core, 512) == -EBUSY, "Shrink to 512 MiB must return -EBUSY while pinned at 1024");
        check(cdsp_mem_core_resize(&core, 128) == -EBUSY, "Shrink to 128 MiB must return -EBUSY while pinned at 1024");

        // Pin owner fileB at 512 MiB
        ret = cdsp_mem_core_pin(&core, fileB, 512);
        check(ret == 0, "Pin fileB at 512 MiB");

        // Unpin fileA -> fileB's 512 MiB pin is now highest
        ret = cdsp_mem_core_unpin(&core, fileA);
        check(ret == 0, "Unpin fileA");
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 512, "Pinned extent is now 512 MiB from fileB");

        // Shrink to 512 MiB succeeds, shrink to 256 MiB blocked
        check(cdsp_mem_core_resize(&core, 512) == 0, "Shrink to 512 MiB succeeds");
        check(cdsp_mem_core_resize(&core, 256) == -EBUSY, "Shrink to 256 MiB blocked by fileB");

        // Unpin fileB -> drops to baseline
        cdsp_mem_core_unpin(&core, fileB);
        check(cdsp_mem_core_get_pinned_extent_mb(&core) == 128, "Pinned extent returns to 128 MiB");
        check(cdsp_mem_core_resize(&core, 128) == 0, "Shrink to 128 MiB baseline succeeds");

        // Verify invalid pin bounds
        check(cdsp_mem_core_pin(&core, fileA, 64) == -EINVAL, "Pin < 128 MiB must return -EINVAL");
        check(cdsp_mem_core_pin(&core, fileA, 256) == -EINVAL, "Pin > current size (128 MiB) must return -EINVAL");
    }
    std::cout << "  ✓ DSP-window pin / unpin and extent protection passed" << std::endl;

    // 8. Concurrent Multi-Threaded Stress Test
    {
        std::mutex core_mutex;
        std::vector<std::thread> threads;
        const int num_threads = 8;
        const int iterations = 50;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&core, &core_mutex, i, iterations]() {
                void *thread_owner = reinterpret_cast<void *>(uintptr_t(0x10000 + i));
                for (int iter = 0; iter < iterations; ++iter) {
                    uint32_t grow_size = 256 + (i % 4) * 256; // 256, 512, 768, 1024
                    {
                        std::lock_guard<std::mutex> lock(core_mutex);
                        cdsp_mem_core_resize(&core, grow_size);
                        cdsp_mem_core_pin(&core, thread_owner, grow_size);
                        cdsp_mem_core_map_add(&core, thread_owner, 0, static_cast<uint64_t>(grow_size) << 20);
                    }

                    std::this_thread::sleep_for(std::chrono::microseconds(100));

                    {
                        std::lock_guard<std::mutex> lock(core_mutex);
                        cdsp_mem_core_map_remove(&core, thread_owner);
                        cdsp_mem_core_unpin(&core, thread_owner);
                        cdsp_mem_core_resize(&core, 128);
                    }
                }
            });
        }

        for (auto &t : threads) {
            t.join();
        }

        // After all threads finish, all pins and mappings are gone, shrink to 128 must succeed
        ret = cdsp_mem_core_resize(&core, 128);
        check(ret == 0, "Final shrink to 128 MiB after concurrent threads");
        check(core.size_mb == 128, "Final size must be 128 MiB");
        check(ctx.allocated_chunks.size() == 64, "Exactly 64 chunks allocated at end");
    }
    std::cout << "  ✓ Concurrent multi-threaded stress test passed" << std::endl;

    // 9. Destroy core
    cdsp_mem_core_destroy(&core);
    check(ctx.allocated_chunks.empty(), "All chunks must be freed on core destroy");
    check(ctx.mapped_iovas.empty(), "All IOMMU mappings must be unmapped on core destroy");
    check(core.nchunks == 0, "core.nchunks is 0 after destroy");
    std::cout << "  ✓ Clean teardown and zero resource leak passed" << std::endl;

    std::cout << "\nALL CDSP MEM CORE TESTS PASSED!" << std::endl;
    return 0;
}
