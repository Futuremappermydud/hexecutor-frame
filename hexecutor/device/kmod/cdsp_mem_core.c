/* SPDX-License-Identifier: GPL-2.0 */
#include "cdsp_mem_core.h"
#ifdef __KERNEL__
#include <linux/string.h> /* the module #includes this file; <string.h> does not exist in kernel builds */
#else
#include <string.h>
#endif

int cdsp_mem_core_init(struct cdsp_mem_core *core, uint32_t initial_size_mb,
                       uint64_t iova, const struct cdsp_mem_ops *ops, void *ops_ctx)
{
    uint32_t nchunks;
    uint32_t i;

    if (!core || !ops || !ops->alloc_chunk || !ops->free_chunk ||
        !ops->iommu_map || !ops->iommu_unmap || !ops->flush_iotlb ||
        !ops->alloc_meta || !ops->free_meta)
        return -EINVAL;

    if (initial_size_mb < CDSP_MEM_MIN_SIZE_MB)
        initial_size_mb = CDSP_MEM_MIN_SIZE_MB;
    if (initial_size_mb > CDSP_MEM_MAX_SIZE_MB)
        initial_size_mb = CDSP_MEM_MAX_SIZE_MB;

    if (initial_size_mb % 2 != 0 || (iova % CDSP_MEM_CHUNK_BYTES) != 0 ||
        ((uint64_t)initial_size_mb << 20) > (1ULL << 32) - iova)
        return -EINVAL;

    nchunks = initial_size_mb / 2;

    core->size_mb = initial_size_mb;
    core->nchunks = nchunks;
    core->iova = iova;
    core->ops = ops;
    core->ops_ctx = ops_ctx;
    core->pins = NULL;
    core->mappings = NULL;

    core->chunks = (void **)ops->alloc_meta(ops_ctx, nchunks * sizeof(void *));
    if (!core->chunks)
        return -ENOMEM;
    memset(core->chunks, 0, nchunks * sizeof(void *));

    for (i = 0; i < nchunks; i++) {
        core->chunks[i] = ops->alloc_chunk(ops_ctx, i);
        if (!core->chunks[i])
            goto fail;

        int ret = ops->iommu_map(ops_ctx, iova + (uint64_t)i * CDSP_MEM_CHUNK_BYTES,
                                 core->chunks[i], i);
        if (ret != 0) {
            ops->free_chunk(ops_ctx, core->chunks[i], i);
            core->chunks[i] = NULL;
            goto fail;
        }
    }

    ops->flush_iotlb(ops_ctx);
    return 0;

fail:
    for (uint32_t j = 0; j < i; j++) {
        ops->iommu_unmap(ops_ctx, iova + (uint64_t)j * CDSP_MEM_CHUNK_BYTES, j);
        ops->free_chunk(ops_ctx, core->chunks[j], j);
        core->chunks[j] = NULL;
    }
    ops->flush_iotlb(ops_ctx);
    ops->free_meta(ops_ctx, core->chunks);
    core->chunks = NULL;
    core->nchunks = 0;
    core->size_mb = 0;
    return -ENOMEM;
}

void cdsp_mem_core_destroy(struct cdsp_mem_core *core)
{
    if (!core || !core->ops)
        return;

    const struct cdsp_mem_ops *ops = core->ops;
    void *ctx = core->ops_ctx;

    if (core->chunks) {
        for (uint32_t i = 0; i < core->nchunks; i++) {
            if (core->chunks[i]) {
                ops->iommu_unmap(ctx, core->iova + (uint64_t)i * CDSP_MEM_CHUNK_BYTES, i);
                ops->free_chunk(ctx, core->chunks[i], i);
                core->chunks[i] = NULL;
            }
        }
        ops->flush_iotlb(ctx);
        ops->free_meta(ctx, core->chunks);
        core->chunks = NULL;
    }

    /* Free all pins */
    struct cdsp_mem_pin *pin = core->pins;
    while (pin) {
        struct cdsp_mem_pin *next = pin->next;
        ops->free_meta(ctx, pin);
        pin = next;
    }
    core->pins = NULL;

    /* Free all mappings */
    struct cdsp_mem_mapping *map = core->mappings;
    while (map) {
        struct cdsp_mem_mapping *next = map->next;
        ops->free_meta(ctx, map);
        map = next;
    }
    core->mappings = NULL;

    core->nchunks = 0;
    core->size_mb = 0;
}

uint32_t cdsp_mem_core_get_pinned_extent_mb(const struct cdsp_mem_core *core)
{
    uint32_t max_mb = CDSP_MEM_MIN_SIZE_MB;
    if (!core)
        return max_mb;

    /* Check all live pins */
    for (struct cdsp_mem_pin *p = core->pins; p != NULL; p = p->next) {
        if (p->size_mb > max_mb)
            max_mb = p->size_mb;
    }

    /* Check all live mappings */
    for (struct cdsp_mem_mapping *m = core->mappings; m != NULL; m = m->next) {
        uint32_t map_mb = (uint32_t)((m->end_bytes + (1u << 20) - 1u) >> 20);
        map_mb = (map_mb + 1u) / 2u * 2u; /* Aligned to 2 MB chunk boundary */
        if (map_mb > max_mb)
            max_mb = map_mb;
    }

    return max_mb;
}

int cdsp_mem_core_resize(struct cdsp_mem_core *core, uint32_t new_size_mb)
{
    if (!core || !core->ops)
        return -EINVAL;

    const struct cdsp_mem_ops *ops = core->ops;
    void *ctx = core->ops_ctx;

    if (new_size_mb < CDSP_MEM_MIN_SIZE_MB || new_size_mb > CDSP_MEM_MAX_SIZE_MB)
        return -EINVAL;

    if (new_size_mb % 2 != 0 || ((uint64_t)new_size_mb << 20) > (1ULL << 32) - core->iova)
        return -EINVAL;

    uint32_t new_nchunks = new_size_mb / 2;
    if (new_nchunks == core->nchunks)
        return 0;

    if (new_nchunks > core->nchunks) {
        /* Growing: allocate new chunks array */
        void **new_chunks = (void **)ops->alloc_meta(ctx, new_nchunks * sizeof(void *));
        if (!new_chunks)
            return -ENOMEM;

        memset(new_chunks, 0, new_nchunks * sizeof(void *));
        if (core->chunks && core->nchunks)
            memcpy(new_chunks, core->chunks, core->nchunks * sizeof(void *));

        uint32_t i;
        for (i = core->nchunks; i < new_nchunks; i++) {
            new_chunks[i] = ops->alloc_chunk(ctx, i);
            if (!new_chunks[i])
                goto grow_rollback;

            int ret = ops->iommu_map(ctx, core->iova + (uint64_t)i * CDSP_MEM_CHUNK_BYTES,
                                     new_chunks[i], i);
            if (ret != 0) {
                ops->free_chunk(ctx, new_chunks[i], i);
                new_chunks[i] = NULL;
                goto grow_rollback;
            }
        }

        /* Success: flush SMMU and install new chunk array */
        ops->flush_iotlb(ctx);
        ops->free_meta(ctx, core->chunks);
        core->chunks = new_chunks;
        core->nchunks = new_nchunks;
        core->size_mb = new_size_mb;
        return 0;

grow_rollback:
        /* Free and unmap all chunks added in this resize attempt */
        for (uint32_t j = core->nchunks; j < i; j++) {
            ops->iommu_unmap(ctx, core->iova + (uint64_t)j * CDSP_MEM_CHUNK_BYTES, j);
            ops->free_chunk(ctx, new_chunks[j], j);
            new_chunks[j] = NULL;
        }
        if (i > core->nchunks)
            ops->flush_iotlb(ctx);
        ops->free_meta(ctx, new_chunks);
        return -ENOMEM;
    } else {
        /* Shrinking: verify against pinned and mapped extents */
        uint32_t pinned_extent = cdsp_mem_core_get_pinned_extent_mb(core);
        if (new_size_mb < pinned_extent) {
            /* Requested size would free memory in active use */
            return -EBUSY;
        }

        /* Safe to shrink: unmap and free from top down */
        for (uint32_t i = core->nchunks; i > new_nchunks; i--) {
            uint32_t idx = i - 1;
            ops->iommu_unmap(ctx, core->iova + (uint64_t)idx * CDSP_MEM_CHUNK_BYTES, idx);
            ops->free_chunk(ctx, core->chunks[idx], idx);
            core->chunks[idx] = NULL;
        }
        ops->flush_iotlb(ctx);

        /* Downsize pointer array */
        void **shrunk = (void **)ops->alloc_meta(ctx, new_nchunks * sizeof(void *));
        if (shrunk) {
            memcpy(shrunk, core->chunks, new_nchunks * sizeof(void *));
            ops->free_meta(ctx, core->chunks);
            core->chunks = shrunk;
        }

        core->nchunks = new_nchunks;
        core->size_mb = new_size_mb;
        return 0;
    }
}

int cdsp_mem_core_pin(struct cdsp_mem_core *core, void *owner, uint32_t size_mb)
{
    if (!core || !owner)
        return -EINVAL;

    if (size_mb < CDSP_MEM_MIN_SIZE_MB || size_mb > core->size_mb)
        return -EINVAL;

    /* Check if owner already has a pin */
    for (struct cdsp_mem_pin *p = core->pins; p != NULL; p = p->next) {
        if (p->owner == owner) {
            p->size_mb = size_mb;
            return 0;
        }
    }

    /* Allocate and add new pin node */
    struct cdsp_mem_pin *pin = (struct cdsp_mem_pin *)core->ops->alloc_meta(core->ops_ctx, sizeof(*pin));
    if (!pin)
        return -ENOMEM;

    pin->owner = owner;
    pin->size_mb = size_mb;
    pin->next = core->pins;
    core->pins = pin;
    return 0;
}

int cdsp_mem_core_unpin(struct cdsp_mem_core *core, void *owner)
{
    if (!core || !owner)
        return 0;

    struct cdsp_mem_pin **curr = &core->pins;
    while (*curr) {
        if ((*curr)->owner == owner) {
            struct cdsp_mem_pin *to_free = *curr;
            *curr = to_free->next;
            core->ops->free_meta(core->ops_ctx, to_free);
            return 0;
        }
        curr = &(*curr)->next;
    }
    return 0;
}

int cdsp_mem_core_map_add(struct cdsp_mem_core *core, void *owner, uint64_t offset, uint64_t size)
{
    if (!core || !owner)
        return -EINVAL;

    uint64_t end = offset + size;
    if (end > (uint64_t)core->size_mb << 20)
        return -EINVAL;

    /* Check if mapping for this owner already exists (e.g. split/update) */
    for (struct cdsp_mem_mapping *m = core->mappings; m != NULL; m = m->next) {
        if (m->owner == owner) {
            m->end_bytes = end;
            return 0;
        }
    }

    struct cdsp_mem_mapping *map = (struct cdsp_mem_mapping *)core->ops->alloc_meta(core->ops_ctx, sizeof(*map));
    if (!map)
        return -ENOMEM;

    map->owner = owner;
    map->end_bytes = end;
    map->next = core->mappings;
    core->mappings = map;
    return 0;
}

int cdsp_mem_core_map_remove(struct cdsp_mem_core *core, void *owner)
{
    if (!core || !owner)
        return 0;

    struct cdsp_mem_mapping **curr = &core->mappings;
    while (*curr) {
        if ((*curr)->owner == owner) {
            struct cdsp_mem_mapping *to_free = *curr;
            *curr = to_free->next;
            core->ops->free_meta(core->ops_ctx, to_free);
            return 0;
        }
        curr = &(*curr)->next;
    }
    return 0;
}

uint32_t cdsp_mem_core_get_size_mb(const struct cdsp_mem_core *core)
{
    return core ? core->size_mb : 0;
}

void *cdsp_mem_core_get_chunk(const struct cdsp_mem_core *core, uint32_t chunk_idx)
{
    if (!core || !core->chunks || chunk_idx >= core->nchunks)
        return NULL;
    return core->chunks[chunk_idx];
}
