/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_dma_ring.h"

#include <string.h>

static uint64_t ring_clock(Nprg2DmaRing *r){
    return r->ops.clock?r->ops.clock(r->ctx):r->completed;
}

int nprg2_dma_ring_init(Nprg2DmaRing *r,const Nprg2DmaRingOps *ops,
                        void *ctx,uint64_t timeout_ticks){
    if(!r||!ops||!ops->begin||!ops->issue||!ops->done||!ops->finish||!ops->clock||!timeout_ticks)return 0;
    memset(r,0,sizeof *r);r->ops=*ops;r->ctx=ctx;r->timeout_ticks=timeout_ticks;return 1;
}

int nprg2_dma_ring_reset(Nprg2DmaRing *r){
    if(!r||r->active||r->issued!=r->completed)return 0;
    Nprg2DmaRingOps ops=r->ops;void *ctx=r->ctx;uint64_t timeout=r->timeout_ticks;
    memset(r,0,sizeof *r);r->ops=ops;r->ctx=ctx;r->timeout_ticks=timeout;return 1;
}

static int ring_advance(Nprg2DmaRing *r,uint64_t target){
    while(r->completed<target){
        uint64_t seq=r->completed+1u;uint32_t slot=(uint32_t)((seq-1u)%NPRG2_DMA_RING_SLOTS);
        if(r->slot_sequence[slot]!=seq){r->failed=1;r->failures++;r->failed_sequence=(uint32_t)seq;return -1;}
        int d=r->ops.done(r->ctx,seq,slot);
        if(d<0){r->failed=1;r->failures++;r->failed_sequence=(uint32_t)seq;return -1;}
        if(!d)return 0;
        r->slot_sequence[slot]=0;r->completed=seq;r->completions++;
    }
    return 1;
}

static int ring_wait_sequence(Nprg2DmaRing *r,uint64_t target){
    if(!r||target>r->issued)return -1;
    if(target<=r->completed)return 1;
    uint64_t start=ring_clock(r);
    for(;;){
        int a=ring_advance(r,target);if(a<0)return -1;
        if(target<=r->completed)return 1;
        uint64_t now=ring_clock(r);
        if(now-start>=r->timeout_ticks){r->failed=1;r->failures++;r->failed_sequence=(uint32_t)(r->completed+1u);return 0;}
        if(r->ops.relax)r->ops.relax(r->ctx);
    }
}

int nprg2_dma_ring_enqueue(Nprg2DmaRing *r,void *dst,const void *src,size_t bytes){
    if(!r||!dst||!src||!bytes||r->failed)return -1;
    if(!r->active){
        int b=r->ops.begin(r->ctx);if(b<=0){if(b<0){r->failed=1;r->failures++;}return b;}
        r->active=1;r->starts++;
    }
    if(r->issued==UINT64_MAX){r->failed=1;r->failures++;return -1;}
    while(r->issued-r->completed>=NPRG2_DMA_RING_SLOTS){
        int w=ring_wait_sequence(r,r->completed+1u);if(w!=1)return -1;
    }
    uint64_t seq=r->issued+1u;uint32_t slot=(uint32_t)((seq-1u)%NPRG2_DMA_RING_SLOTS);
    if(r->slot_sequence[slot]){r->failed=1;r->failures++;r->failed_sequence=(uint32_t)seq;return -1;}
    int q=r->ops.issue(r->ctx,seq,slot,dst,src,bytes);
    if(q<=0){
        if(q<0){r->failed=1;r->failures++;r->failed_sequence=(uint32_t)seq;return -1;}
        /* An ineligible next descriptor may use the CPU path after earlier
         * queued writes complete; never let that copy race the queue. */
        if(r->issued!=r->completed&&ring_wait_sequence(r,r->issued)!=1)return -1;
        return 0;
    }
    r->slot_sequence[slot]=seq;r->issued=seq;return 1;
}

int nprg2_dma_ring_checkpoint_set(Nprg2DmaRing *r,uint32_t id){
    if(!r||id==0||id>=NPRG2_DMA_RING_EVENTS||r->failed)return 0;
    Nprg2DmaRingEvent *e=&r->event[id];
    if(e->valid){e->generation++;if(!e->generation)e->generation=1u;}else{e->valid=1;e->generation=1u;}
    e->sequence=r->issued;r->checkpoint_sets++;return 1;
}

int nprg2_dma_ring_checkpoint_wait(Nprg2DmaRing *r,uint32_t id){
    if(!r||id==0||id>=NPRG2_DMA_RING_EVENTS||r->failed)return 0;
    r->checkpoint_waits++;
    /* A captured wait can legitimately precede the first set for a reused
     * framework ID. With no current producer snapshot there is no DMA fence. */
    if(!r->event[id].valid)return 1;
    return ring_wait_sequence(r,r->event[id].sequence)==1;
}

int nprg2_dma_ring_drain_all(Nprg2DmaRing *r){
    if(!r)return -1;
    if(!r->active)return r->failed?-1:1;
    int w=ring_wait_sequence(r,r->issued);
    if(w==0)return 0;
    int had_error=r->failed||w<0;
    int f=r->ops.finish(r->ctx,had_error);
    if(f==0)return 0;
    r->active=0;r->drains++;
    if(f<0||had_error){r->failed=1;r->failures++;return -1;}
    return 1;
}
