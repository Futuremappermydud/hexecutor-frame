/* SPDX-License-Identifier: GPL-2.0 */
#ifndef CDSP_MEM_CORE_H
#define CDSP_MEM_CORE_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CDSP_MEM_MIN_SIZE_MB 128u
#define CDSP_MEM_MAX_SIZE_MB 1536u /* 1.5 GiB max for 32-bit IOVA starting at 0xa0000000 */
#define CDSP_MEM_CHUNK_MB    2u
#define CDSP_MEM_CHUNK_BYTES (CDSP_MEM_CHUNK_MB * 1024u * 1024u)
#define CDSP_MEM_BASE_IOVA   0xa0000000u

/* Error codes matching standard errno */
#ifndef EBUSY
#define EBUSY 16
#endif
#ifndef EINVAL
#define EINVAL 22
#endif
#ifndef ENOMEM
#define ENOMEM 12
#endif
#ifndef EEXIST
#define EEXIST 17
#endif
#ifndef EPERM
#define EPERM 1
#endif

struct cdsp_mem_core;

/* Callbacks provided by the host environment (Linux kernel module or test harness) */
struct cdsp_mem_ops {
    /* Allocate a 2 MB physical chunk. Returns opaque handle or NULL. */
    void *(*alloc_chunk)(void *ctx, uint32_t chunk_idx);

    /* Free a 2 MB physical chunk. */
    void (*free_chunk)(void *ctx, void *chunk_handle, uint32_t chunk_idx);

    /* Map chunk into CDSP IOMMU domain at iova. Returns 0 on success, < 0 on error. */
    int (*iommu_map)(void *ctx, uint64_t iova, void *chunk_handle, uint32_t chunk_idx);

    /* Unmap chunk from CDSP IOMMU domain at iova. */
    void (*iommu_unmap)(void *ctx, uint64_t iova, uint32_t chunk_idx);

    /* Flush SMMU context / IOTLB. */
    void (*flush_iotlb)(void *ctx);

    /* Metadata memory allocator for chunk arrays and tracking nodes. */
    void *(*alloc_meta)(void *ctx, size_t bytes);
    void (*free_meta)(void *ctx, void *ptr);
};

/* Tracking node for a DSP window pin */
struct cdsp_mem_pin {
    void *owner;
    uint32_t size_mb;
    struct cdsp_mem_pin *next;
};

/* Tracking node for a live user VMA mapping */
struct cdsp_mem_mapping {
    void *owner;
    uint64_t end_bytes;
    struct cdsp_mem_mapping *next;
};

/* Pure C Core state of the CDSP memory reservation manager */
struct cdsp_mem_core {
    uint32_t size_mb;
    uint32_t nchunks;
    uint64_t iova;

    const struct cdsp_mem_ops *ops;
    void *ops_ctx;

    void **chunks;
    struct cdsp_mem_pin *pins;
    struct cdsp_mem_mapping *mappings;
    uint32_t untracked_mappings;
};

/* Initialize core manager with initial reservation (e.g. 128 MiB baseline) */
int cdsp_mem_core_init(struct cdsp_mem_core *core, uint32_t initial_size_mb,
                       uint64_t iova, const struct cdsp_mem_ops *ops, void *ops_ctx);

/* Cleanly destroy core manager, unmapping and freeing all chunks and tracking nodes */
void cdsp_mem_core_destroy(struct cdsp_mem_core *core);

/* Resize CDSP memory lease (grow or shrink).
 * Shrinking returns -EBUSY if requested size < highest pinned/mapped extent.
 * Growing is always permitted up to CDSP_MEM_MAX_SIZE_MB.
 * On growth failure, automatically rolls back newly added chunks and restores state.
 */
int cdsp_mem_core_resize(struct cdsp_mem_core *core, uint32_t new_size_mb);

/* Pin a DSP window extent for an owner (e.g. open file).
 * Returns 0 on success, -EINVAL if size exceeds current reservation or is invalid.
 */
int cdsp_mem_core_pin(struct cdsp_mem_core *core, void *owner, uint32_t size_mb);

/* Release a DSP window pin for an owner. Returns 0. */
int cdsp_mem_core_unpin(struct cdsp_mem_core *core, void *owner);

/* Register a live user mapping (e.g. VMA) extent. */
int cdsp_mem_core_map_add(struct cdsp_mem_core *core, void *owner, uint64_t offset, uint64_t size);

/* Deregister a live user mapping (e.g. VMA on close/munmap). */
int cdsp_mem_core_map_remove(struct cdsp_mem_core *core, void *owner);
/* VMA close: removes a tracked mapping, or releases one untracked count. */
int cdsp_mem_core_map_close(struct cdsp_mem_core *core, void *owner);

/* Compute the highest active pinned or mapped extent in MiB.
 * Always at least CDSP_MEM_MIN_SIZE_MB (128).
 */
uint32_t cdsp_mem_core_get_pinned_extent_mb(const struct cdsp_mem_core *core);

/* Query current reservation size in MiB */
uint32_t cdsp_mem_core_get_size_mb(const struct cdsp_mem_core *core);

/* Retrieve chunk handle at index (or NULL if index >= nchunks) */
void *cdsp_mem_core_get_chunk(const struct cdsp_mem_core *core, uint32_t chunk_idx);

#ifdef __cplusplus
}
#endif

#endif /* CDSP_MEM_CORE_H */
