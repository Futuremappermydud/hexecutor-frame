#include "prepared_dma_ring.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);} } while(0)

typedef struct { uint64_t seq;uint8_t *dst;const uint8_t *src;size_t bytes;uint8_t queued,done; } MockDesc;
typedef struct {
    MockDesc d[NPRG2_DMA_RING_SLOTS];
    uint64_t clock,release_through,error_seq;
    uint32_t starts,finishes,reject_seq;
    uint8_t active,hold,error;
} MockDma;

static int mock_begin(void *ctx){MockDma *m=ctx;CHECK(!m->active);m->active=1;m->starts++;return 1;}
static int mock_issue(void *ctx,uint64_t seq,uint32_t slot,void *dst,const void *src,size_t bytes){
    MockDma *m=ctx;MockDesc *d=&m->d[slot];CHECK(m->active&&!d->queued);if(seq==m->reject_seq)return 0;*d=(MockDesc){seq,dst,src,bytes,1,0};return 1;
}
static int mock_done(void *ctx,uint64_t seq,uint32_t slot){
    MockDma *m=ctx;MockDesc *d=&m->d[slot];if(m->error)return -1;CHECK(m->active&&d->queued&&d->seq==seq);
    if(m->error_seq==seq){m->error=1;m->active=0;return -1;}
    if(m->hold&&seq>m->release_through)return 0;
    memmove(d->dst,d->src,d->bytes);d->done=1;d->queued=0;return 1;
}
static int mock_finish(void *ctx,int had_error){
    MockDma *m=ctx;m->finishes++;
    if(had_error){for(uint32_t i=0;i<NPRG2_DMA_RING_SLOTS;i++)m->d[i].queued=0;m->active=0;return -1;}
    for(uint32_t i=0;i<NPRG2_DMA_RING_SLOTS;i++)if(m->d[i].queued)return 0;
    m->active=0;return 1;
}
static uint64_t mock_clock(void *ctx){return ((MockDma *)ctx)->clock;}
static void mock_relax(void *ctx){((MockDma *)ctx)->clock++;}
static const Nprg2DmaRingOps kOps={mock_begin,mock_issue,mock_done,mock_finish,mock_clock,mock_relax};
static void init(Nprg2DmaRing *r,MockDma *m,uint64_t timeout){memset(m,0,sizeof *m);CHECK(nprg2_dma_ring_init(r,&kOps,m,timeout));}

static void test_event_snapshot_and_id_reuse(void){
    Nprg2DmaRing r;MockDma m;init(&r,&m,20);uint8_t src[5][4],dst[5][4];
    CHECK(nprg2_dma_ring_checkpoint_wait(&r,19)); /* absent producer has no queued DMA to fence */
    for(unsigned i=0;i<5;i++){memset(src[i],(int)(0x10+i),4);memset(dst[i],0,4);}
    for(unsigned i=0;i<3;i++)CHECK(nprg2_dma_ring_enqueue(&r,dst[i],src[i],4)==1);
    CHECK(nprg2_dma_ring_checkpoint_set(&r,7));uint32_t first_gen=r.event[7].generation;
    for(unsigned i=3;i<5;i++)CHECK(nprg2_dma_ring_enqueue(&r,dst[i],src[i],4)==1);
    CHECK(nprg2_dma_ring_checkpoint_wait(&r,7));
    CHECK(r.completed==3);
    for(unsigned i=0;i<3;i++)CHECK(!memcmp(dst[i],src[i],4));
    for(unsigned i=3;i<5;i++)CHECK(dst[i][0]==0);
    CHECK(nprg2_dma_ring_checkpoint_wait(&r,7)); /* repeated wait is idempotent */
    CHECK(r.completed==3);
    CHECK(nprg2_dma_ring_checkpoint_set(&r,7));
    CHECK(r.event[7].generation==first_gen+1u&&r.event[7].sequence==5u);
    CHECK(nprg2_dma_ring_checkpoint_wait(&r,7));CHECK(r.completed==5);
    CHECK(nprg2_dma_ring_drain_all(&r)==1&&!m.active);
    puts("PASS event snapshots fence only captured tail and reused IDs advance generation");
}

static void test_ring_wrap_and_fifo(void){
    Nprg2DmaRing r;MockDma m;init(&r,&m,100);enum{N=150};uint8_t src[N][8],dst[N][8];
    for(unsigned i=0;i<N;i++){memset(src[i],(int)i,sizeof src[i]);memset(dst[i],0,sizeof dst[i]);}
    for(unsigned i=0;i<N;i++)CHECK(nprg2_dma_ring_enqueue(&r,dst[i],src[i],sizeof src[i])==1);
    CHECK(nprg2_dma_ring_checkpoint_set(&r,23));CHECK(r.event[23].sequence==N);
    CHECK(nprg2_dma_ring_checkpoint_wait(&r,23));CHECK(r.completed==N);
    CHECK(nprg2_dma_ring_drain_all(&r)==1&&!m.active);
    for(unsigned i=0;i<N;i++)CHECK(!memcmp(src[i],dst[i],sizeof src[i]));
    CHECK(m.starts==1&&m.finishes==1);puts("PASS 63-slot FIFO wraps and reuses descriptors without losing sequence order");
}

static void test_error_drains_before_release(void){
    Nprg2DmaRing r;MockDma m;init(&r,&m,20);uint8_t src[3][4]={{1},{2},{3}},dst[3][4]={{0}};
    CHECK(nprg2_dma_ring_enqueue(&r,dst[0],src[0],4)==1);
    CHECK(nprg2_dma_ring_enqueue(&r,dst[1],src[1],4)==1);
    CHECK(nprg2_dma_ring_enqueue(&r,dst[2],src[2],4)==1);
    CHECK(nprg2_dma_ring_checkpoint_set(&r,4));m.error_seq=2;
    CHECK(!nprg2_dma_ring_checkpoint_wait(&r,4));
    CHECK(nprg2_dma_ring_drain_all(&r)==-1&&!r.active&&!m.active&&m.finishes==1);
    puts("PASS engine error is fatal, then drain_all releases the stopped engine");
}

static void test_timeout_retains_ownership_then_cancels(void){
    Nprg2DmaRing r;MockDma m;init(&r,&m,3);uint8_t src[4]={1,2,3,4},dst[4]={0};m.hold=1;
    CHECK(nprg2_dma_ring_enqueue(&r,dst,src,sizeof src)==1);
    CHECK(nprg2_dma_ring_checkpoint_set(&r,12));CHECK(!nprg2_dma_ring_checkpoint_wait(&r,12));
    CHECK(nprg2_dma_ring_drain_all(&r)==0&&r.active&&m.active);
    m.hold=0;CHECK(nprg2_dma_ring_drain_all(&r)==-1&&!r.active&&!m.active);
    puts("PASS timeout keeps ring/session owned; later cancellation drains before release");
}

static void test_unavailable_descriptor_drains_before_cpu_fallback(void){
    Nprg2DmaRing r;MockDma m;init(&r,&m,20);uint8_t src[3][4]={{1},{2},{3}},dst[3][4]={{0}};
    m.reject_seq=3;
    CHECK(nprg2_dma_ring_enqueue(&r,dst[0],src[0],4)==1);
    CHECK(nprg2_dma_ring_enqueue(&r,dst[1],src[1],4)==1);
    CHECK(nprg2_dma_ring_enqueue(&r,dst[2],src[2],4)==0);
    CHECK(r.completed==2&&!memcmp(dst[0],src[0],4)&&!memcmp(dst[1],src[1],4));
    memcpy(dst[2],src[2],4); /* caller's CPU fallback now cannot race DMA */
    CHECK(nprg2_dma_ring_drain_all(&r)==1&&!m.active);
    CHECK(!memcmp(dst,src,sizeof src));
    puts("PASS rejected descriptor drains earlier writes before CPU fallback");
}

int main(void){
    test_event_snapshot_and_id_reuse();test_ring_wrap_and_fifo();
    test_error_drains_before_release();test_timeout_retains_ownership_then_cancels();
    test_unavailable_descriptor_drains_before_cpu_fallback();return 0;
}
