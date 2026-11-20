/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_ASYNC_DMA_H
#define NPRG2_PREPARED_ASYNC_DMA_H

#include "prepared_executor.h"
#include "../qd_graph_dsp.h"

#include <stdint.h>
#include <stddef.h>

#ifndef QDS_PREPARED_DMA_TIMEOUT_CYCLES
#define QDS_PREPARED_DMA_TIMEOUT_CYCLES 50000000ull
#endif

typedef struct {
    uint32_t probe_ok,pending,saved_dm2,bytes;
    uint32_t probe_match,probe_target_restored,probe_dm2_restored;
    uint32_t last_status,last_command,src,dst,last_mode;
    uint32_t verified_bytes,mismatch_count,mismatch_offset,source_word,dest_word;
    uint64_t starts,drains,probe_cycles;
    Nprg2DmaRing ring;
    uint32_t ring_tail_slot,ring_mode,ring_cap;
} QdsPreparedDmaAsync;

#ifdef __hexagon__
#define QDS_DMA_ASYNC_DESC 0xf1808000u
#define QDS_DMA_ASYNC_DUMMY (QDS_DMA_ASYNC_DESC+32u)
#define QDS_DMA_ASYNC_RING_DUMMY (QDS_DMA_ASYNC_DESC+63u*32u)
#define QDS_DMA_ASYNC_RING_BYTES (64u*32u)
#define QDS_DMA_ASYNC_BYTES 128u
#define QDS_DMA_ASYNC_SRC_LO 0xa0000000u
#define QDS_DMA_ASYNC_SRC_HI 0xa8000000u
#define QDS_DMA_ASYNC_DST_LO 0x31000000u
#define QDS_DMA_ASYNC_DST_HI 0x31800000u
#define QDS_DMA_ASYNC_DST_ACTIVE_HI 0x31780000u
#define QDS_DMA_ASYNC_TEST_DST 0x31782000u

static uint32_t qds_dma_async_dm_read(uint32_t index){
    uint32_t v;__asm__ volatile("r1 = %1\n .word 0xa801c0a0\n %0 = r0":"=r"(v):"r"(index):"r0","r1","memory");return v;
}
static void qds_dma_async_dm_write(uint32_t index,uint32_t value){
    __asm__ volatile("r1 = %0\n r0 = %1\n .word 0xa801c0c0"::"r"(index),"r"(value):"r0","r1","memory");
}
static int qds_dma_async_dm_restore(uint32_t value){
    qds_dma_async_dm_write(2,value);return qds_dma_async_dm_read(2)==value;
}
static uint32_t qds_dma_async_poll(void){uint32_t s;__asm__ volatile("%0 = dmpoll":"=r"(s)::"memory");return s;}
static uint64_t qds_dma_async_cycles(void){uint64_t c;__asm__ volatile("%0 = upcycle":"=r"(c));return c;}
static uint64_t qds_dma_async_tlbr(uint32_t i){uint64_t e;__asm__ volatile("%0 = tlbr(%1)":"=r"(e):"r"(i):"memory");return e;}
static uint32_t qds_dma_async_tlbp(uint32_t va,uint32_t asid){uint32_t i;__asm__ volatile("%0 = tlbp(%1)":"=r"(i):"r"((asid<<20)|(va>>12)):"memory");return i;}
static void qds_dma_async_export(const QdsPreparedDmaAsync *s){
    volatile uint32_t *d=(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_VERIFY_DIAG_OFFSET);
    d[0]=0x564d4441u;d[1]=1u;d[2]=(uint32_t)s->starts;d[3]=(uint32_t)s->drains;
    d[4]=s->verified_bytes;d[5]=s->mismatch_count;d[6]=s->last_command;d[7]=s->bytes;
    d[8]=s->mismatch_offset;d[9]=s->source_word;d[10]=s->dest_word;d[11]=s->src;
    d[12]=s->dst;d[13]=s->last_status;d[14]=s->saved_dm2;d[15]=s->last_mode;
    __asm__ volatile("barrier" ::: "memory");
}

/* The Frame DMA-v4 stream path consumes the existing CPU-JTLB translations.
 * Verify both ends against the current ASID and rely on qdo_map/qdvtcm lease's
 * already-validated contiguous windows; do not write a DMA JTLB or change ASID. */
static int qds_dma_async_range_mapped(uintptr_t base,size_t bytes,uint32_t asid){
    if(!bytes||base>UINT32_MAX||bytes>UINT32_MAX-base)return 0;
    uintptr_t points[2]={base,(base+bytes-1u)&~(uintptr_t)0xfffu};
    for(unsigned k=0;k<2;k++){
        uint32_t va=(uint32_t)(points[k]&~(uintptr_t)0xfffu),ix=qds_dma_async_tlbp(va,asid);
        if(ix&0x80000000u)return 0; /* Hexagon tlbp high bit means miss. */
        uint64_t e=qds_dma_async_tlbr(ix&0x3ffu);if(!(e>>63))return 0;
    }
    return 1;
}
static int qds_dma_async_ranges_mapped(uintptr_t src,size_t bytes,uintptr_t dst){
    uint32_t ssr;__asm__ volatile("%0 = ssr":"=r"(ssr));uint32_t asid=(ssr>>8)&127u;
    return src>=QDS_DMA_ASYNC_SRC_LO&&src<=QDS_DMA_ASYNC_SRC_HI&&bytes<=QDS_DMA_ASYNC_SRC_HI-src&&
           dst>=QDS_DMA_ASYNC_DST_LO&&dst<=QDS_DMA_ASYNC_DST_HI&&bytes<=QDS_DMA_ASYNC_DST_HI-dst&&
           qds_dma_async_range_mapped(src,bytes,asid)&&qds_dma_async_range_mapped(dst,bytes,asid)&&
           qds_dma_async_range_mapped(QDS_DMA_ASYNC_DESC,64u,asid);
}
static void qds_dma_async_desc_init(uintptr_t src,uintptr_t dst,size_t bytes,uint32_t mode){
    volatile uint32_t *d=(volatile uint32_t *)(uintptr_t)QDS_DMA_ASYNC_DESC;
    volatile uint32_t *tail=(volatile uint32_t *)(uintptr_t)QDS_DMA_ASYNC_DUMMY;
    for(uint32_t i=0;i<8u;i++){d[i]=0;tail[i]=0;}
    d[1]=((uint32_t)bytes&0xffffffu)|(1u<<24)|(1u<<28)|((mode&2u)?(1u<<29):0u);
    d[2]=(uint32_t)src;d[3]=(uint32_t)dst;d[4]=9u;
    d[5]=((uint32_t)bytes&0xffffffu)|(1u<<24);d[6]=((uint32_t)bytes&0xffffffu)<<8;
}
static void qds_dma_async_desc_init_at(uint32_t address,uintptr_t src,uintptr_t dst,size_t bytes,uint32_t mode){
    volatile uint32_t *d=(volatile uint32_t *)(uintptr_t)address;
    for(uint32_t i=0;i<8u;i++)d[i]=0;
    d[1]=((uint32_t)bytes&0xffffffu)|(1u<<24)|(1u<<28)|((mode&2u)?(1u<<29):0u);
    d[2]=(uint32_t)src;d[3]=(uint32_t)dst;d[4]=9u;
    d[5]=((uint32_t)bytes&0xffffffu)|(1u<<24);d[6]=((uint32_t)bytes&0xffffffu)<<8;
}
/* One startup copy probe using the known Qwen sequence. Return 1 only after
 * actual source bytes match, target bytes are restored, and DM2 is restored;
 * return -1 if DMA may still own memory/state. */
static int qds_prepared_dma_async_probe(QdsPreparedDmaAsync *s,const void *src,size_t src_bytes){
    if(!s||!src||src_bytes<QDS_DMA_ASYNC_BYTES||((uintptr_t)src&127u)||
       !qds_dma_async_ranges_mapped((uintptr_t)src,QDS_DMA_ASYNC_BYTES,QDS_DMA_ASYNC_TEST_DST))return 0;
    uint32_t tid;__asm__ volatile("%0 = htid":"=r"(tid));if(tid!=0u)return 0;
    uint32_t pre=qds_dma_async_poll();if((pre&3u)!=0u||*(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_STATE_OFFSET)!=QDS_DMA_STATE_IDLE)return 0;
    volatile uint8_t *dst=(volatile uint8_t *)(uintptr_t)QDS_DMA_ASYNC_TEST_DST;
    const volatile uint8_t *source=(const volatile uint8_t *)src;uint32_t saved[32];
    for(uint32_t i=0;i<32u;i++)saved[i]=((volatile uint32_t *)dst)[i];
    uint32_t dm2=qds_dma_async_dm_read(2);s->saved_dm2=dm2;
    qds_dma_async_dm_write(2,0x2e3u);if(qds_dma_async_dm_read(2)!=0x2e3u){
        volatile uint32_t *state=(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_STATE_OFFSET);
        if(!qds_dma_async_dm_restore(dm2)){*state=QDS_DMA_STATE_PENDING_ERROR;s->pending=1;return -1;}return 0;
    }
    qds_dma_async_desc_init((uintptr_t)src,QDS_DMA_ASYNC_TEST_DST,QDS_DMA_ASYNC_BYTES,2u);
    volatile uint32_t *desc=(volatile uint32_t *)(uintptr_t)QDS_DMA_ASYNC_DESC;
    volatile uint32_t *tail=(volatile uint32_t *)(uintptr_t)QDS_DMA_ASYNC_DUMMY;
    volatile uint32_t *state=(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_STATE_OFFSET);
    *state=QDS_DMA_STATE_INFLIGHT;uint64_t start=qds_dma_async_cycles();
    __asm__ volatile("release(%0):at\n dmlink(%1,%0)"::"r"(desc),"r"(tail):"memory");
    for(;;){
        uint32_t status=qds_dma_async_poll();s->last_status=status;
        if((desc[1]>>31)&&((status&3u)==0u))break;
        if((status&3u)==2u){uint32_t ignored;__asm__ volatile("%0 = dmpause":"=r"(ignored)::"memory");status=qds_dma_async_poll();s->last_status=status;if((status&3u)==0u)break;}
        if(qds_dma_async_cycles()-start>=QDS_PREPARED_DMA_TIMEOUT_CYCLES){
            status=qds_dma_async_poll();s->last_status=status;s->probe_cycles=qds_dma_async_cycles()-start;
            if((status&3u)==1u){s->pending=1;return -1;}
            if((status&3u)==2u){uint32_t ignored;__asm__ volatile("%0 = dmpause":"=r"(ignored)::"memory");status=qds_dma_async_poll();s->last_status=status;if((status&3u)!=0u){s->pending=1;return -1;}}
            break;
        }
    }
    int match=1;for(uint32_t i=0;i<QDS_DMA_ASYNC_BYTES;i++)if(dst[i]!=source[i])match=0;
    for(uint32_t i=0;i<32u;i++)((volatile uint32_t *)dst)[i]=saved[i];
    __asm__ volatile("barrier":::"memory");int restored=1;
    for(uint32_t i=0;i<32u;i++)if(((volatile uint32_t *)dst)[i]!=saved[i])restored=0;
    int dm2_restored=qds_dma_async_dm_restore(dm2);
    if(!dm2_restored){*state=QDS_DMA_STATE_PENDING_ERROR;s->pending=1;return -1;}
    desc[1]&=0x7fffffffu;*state=QDS_DMA_STATE_IDLE;
    s->probe_match=(uint32_t)match;s->probe_target_restored=(uint32_t)restored;s->probe_dm2_restored=1u;
    s->probe_cycles=qds_dma_async_cycles()-start;
    s->probe_ok=(uint32_t)(match&&restored);return (int)s->probe_ok;
}

/* QNN-style checkpointed descriptor queue. A Set snapshots the tail sequence
 * in the portable scheduler; a Wait blocks only through that sequence. */
static uint32_t qds_dma_ring_desc(uint32_t slot){return QDS_DMA_ASYNC_DESC+slot*32u;}
static uint64_t qds_dma_ring_clock(void *ctx){(void)ctx;return qds_dma_async_cycles();}
static void qds_dma_ring_relax(void *ctx){(void)ctx;__asm__ volatile("nop" ::: "memory");}
static int qds_dma_ring_begin(void *ctx){
    QdsPreparedDmaAsync *s=ctx;if(!s||!s->probe_ok||s->pending||s->ring_mode)return 0;
    uint32_t tid;__asm__ volatile("%0 = htid":"=r"(tid));if(tid!=0u)return 0;
    uint32_t status=qds_dma_async_poll();
    volatile uint32_t *state=(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_STATE_OFFSET);
    if((status&3u)!=0u||*state!=QDS_DMA_STATE_IDLE)return 0;
    uint32_t ssr;__asm__ volatile("%0 = ssr":"=r"(ssr));uint32_t asid=(ssr>>8)&127u;
    if(!qds_dma_async_range_mapped(QDS_DMA_ASYNC_DESC,QDS_DMA_ASYNC_RING_BYTES,asid))return 0;
    uint32_t dm2=qds_dma_async_dm_read(2);qds_dma_async_dm_write(2,0x2e3u);
    if(qds_dma_async_dm_read(2)!=0x2e3u){
        if(!qds_dma_async_dm_restore(dm2)){
            *state=QDS_DMA_STATE_PENDING_ERROR;s->pending=1;s->ring_mode=1;s->ring.active=1;return -1;
        }
        return 0;
    }
    volatile uint32_t *dummy=(volatile uint32_t *)(uintptr_t)QDS_DMA_ASYNC_RING_DUMMY;
    for(uint32_t i=0;i<8u;i++)dummy[i]=0;
    s->saved_dm2=dm2;s->ring_tail_slot=63u;s->ring_mode=1u;s->pending=1u;
    *state=QDS_DMA_STATE_INFLIGHT;__asm__ volatile("barrier" ::: "memory");return 1;
}
static int qds_dma_ring_issue(void *ctx,uint64_t sequence,uint32_t slot,void *dst,const void *src,size_t bytes){
    QdsPreparedDmaAsync *s=ctx;(void)sequence;
    if(!s||!s->ring_mode||!s->probe_ok||!bytes||slot>=NPRG2_DMA_RING_SLOTS||bytes>0x00ffff00u||
       (((uintptr_t)dst|(uintptr_t)src|bytes)&127u)||
       (uintptr_t)dst<QDS_DMA_ASYNC_DST_LO||(uintptr_t)dst>QDS_DMA_ASYNC_DST_ACTIVE_HI||
       bytes>QDS_DMA_ASYNC_DST_ACTIVE_HI-(uintptr_t)dst||
       !qds_dma_async_ranges_mapped((uintptr_t)src,bytes,(uintptr_t)dst))return 0;
    uint32_t tid;__asm__ volatile("%0 = htid":"=r"(tid));if(tid!=0u)return 0;
    uint32_t mode=QDS_PREPARED_DMA_MODE,cap=QDS_PREPARED_DMA_MAX_TRANSFERS;
#if QDS_PREPARED_DMA_VERIFY || QDS_PREPARED_DMA_MAX_TRANSFERS > 0
    const volatile uint32_t *control=(const volatile uint32_t *)(uintptr_t)0xf1803e80u;
    if(control[0]==0x50414344u){cap=control[1];mode=control[2];}
#endif
    if(mode>3u)return 0;
    if(cap&&s->starts>=cap)return 0;
    if(mode&1u)dsp_clean_range(src,bytes);
    uint32_t address=qds_dma_ring_desc(slot);
    qds_dma_async_desc_init_at(address,(uintptr_t)src,(uintptr_t)dst,bytes,mode);
    volatile uint32_t *desc=(volatile uint32_t *)(uintptr_t)address;
    volatile uint32_t *tail=(volatile uint32_t *)(uintptr_t)qds_dma_ring_desc(s->ring_tail_slot);
    __asm__ volatile("barrier\n release(%0):at\n dmlink(%1,%0)"::"r"(desc),"r"(tail):"memory");
    s->ring_tail_slot=slot;s->bytes=(uint32_t)bytes;s->last_command=(uint32_t)sequence;
    s->src=(uint32_t)(uintptr_t)src;s->dst=(uint32_t)(uintptr_t)dst;s->last_mode=mode;s->starts++;
    qds_dma_async_export(s);return 1;
}
static int qds_dma_ring_done(void *ctx,uint64_t sequence,uint32_t slot){
    QdsPreparedDmaAsync *s=ctx;(void)sequence;
    if(!s||!s->ring_mode||slot>=NPRG2_DMA_RING_SLOTS)return -1;
    volatile uint32_t *desc=(volatile uint32_t *)(uintptr_t)qds_dma_ring_desc(slot);
    uint32_t status=qds_dma_async_poll();s->last_status=status;
    if((status&3u)==2u){uint32_t ignored;__asm__ volatile("%0 = dmpause":"=r"(ignored)::"memory");status=qds_dma_async_poll();s->last_status=status;return (status&3u)==0u?-1:0;}
    if(desc[1]>>31)return 1; /* this descriptor is complete even if a later one runs */
    return 0;
}
static int qds_dma_ring_finish(void *ctx,int had_error){
    QdsPreparedDmaAsync *s=ctx;if(!s||!s->ring_mode)return -1;
    uint32_t status=qds_dma_async_poll();s->last_status=status;
    if((status&3u)==1u)return 0;
    if((status&3u)==2u){uint32_t ignored;__asm__ volatile("%0 = dmpause":"=r"(ignored)::"memory");status=qds_dma_async_poll();s->last_status=status;if((status&3u)!=0u)return 0;had_error=1;}
    if(!qds_dma_async_dm_restore(s->saved_dm2)){
        *(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_STATE_OFFSET)=QDS_DMA_STATE_PENDING_ERROR;return 0;
    }
    *(volatile uint32_t *)(uintptr_t)(0xf1800000u+QDS_DMA_STATE_OFFSET)=QDS_DMA_STATE_IDLE;
    s->ring_mode=0;s->pending=0;s->drains++;qds_dma_async_export(s);return had_error?-1:1;
}
static int qds_prepared_dma_ring_init(QdsPreparedDmaAsync *s){
    static const Nprg2DmaRingOps ops={qds_dma_ring_begin,qds_dma_ring_issue,qds_dma_ring_done,
        qds_dma_ring_finish,qds_dma_ring_clock,qds_dma_ring_relax};
    return s&&nprg2_dma_ring_init(&s->ring,&ops,s,QDS_PREPARED_DMA_TIMEOUT_CYCLES);
}
#else
static int qds_prepared_dma_async_probe(QdsPreparedDmaAsync *s,const void *src,size_t src_bytes){(void)s;(void)src;(void)src_bytes;return 0;}
static int qds_prepared_dma_ring_init(QdsPreparedDmaAsync *s){(void)s;return 0;}
#endif

#endif
