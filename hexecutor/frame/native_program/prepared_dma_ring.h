#ifndef NPRG2_PREPARED_DMA_RING_H
#define NPRG2_PREPARED_DMA_RING_H

#include <stddef.h>
#include <stdint.h>

#define NPRG2_DMA_RING_DESCRIPTORS 64u
#define NPRG2_DMA_RING_SLOTS (NPRG2_DMA_RING_DESCRIPTORS-1u)
#define NPRG2_DMA_RING_EVENTS 256u

typedef struct {
    int (*begin)(void *ctx);
    int (*issue)(void *ctx,uint64_t sequence,uint32_t slot,
                 void *dst,const void *src,size_t bytes);
    /* 1 complete, 0 still in flight, -1 engine error and now idle. */
    int (*done)(void *ctx,uint64_t sequence,uint32_t slot);
    /* 1 safely released, 0 still owns resources, -1 released with error. */
    int (*finish)(void *ctx,int had_error);
    uint64_t (*clock)(void *ctx);
    void (*relax)(void *ctx);
} Nprg2DmaRingOps;

typedef struct {
    uint64_t sequence;
    uint32_t generation;
    uint8_t valid;
} Nprg2DmaRingEvent;

typedef struct {
    Nprg2DmaRingOps ops;
    void *ctx;
    uint64_t timeout_ticks,issued,completed;
    uint64_t slot_sequence[NPRG2_DMA_RING_SLOTS];
    Nprg2DmaRingEvent event[NPRG2_DMA_RING_EVENTS];
    uint64_t starts,completions,checkpoint_sets,checkpoint_waits,drains,failures;
    uint32_t failed_sequence;
    uint8_t active,failed;
} Nprg2DmaRing;

int nprg2_dma_ring_init(Nprg2DmaRing *ring,const Nprg2DmaRingOps *ops,
                        void *ctx,uint64_t timeout_ticks);
int nprg2_dma_ring_reset(Nprg2DmaRing *ring);
int nprg2_dma_ring_enqueue(Nprg2DmaRing *ring,void *dst,const void *src,size_t bytes);
int nprg2_dma_ring_checkpoint_set(Nprg2DmaRing *ring,uint32_t event_id);
int nprg2_dma_ring_checkpoint_wait(Nprg2DmaRing *ring,uint32_t event_id);
/* 1 drained successfully, 0 ownership remains in hardware, -1 drained with
 * an error. A return of 0 requires retaining every mapping and descriptor. */
int nprg2_dma_ring_drain_all(Nprg2DmaRing *ring);

#endif
