/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_OWNED_MEMORY_H
#define QD_OWNED_MEMORY_H
/* Private DDR allocated by cdsp_mem, never firmware/service memory. The host
 * verifies the module size/IOVA before dispatch. Use the same 16 MB JTLB
 * encoding proven by the Qwen engine, but reject occupied entries/aliases
 * rather than evicting them. Save and restore even invalid entry contents. */
#include <stdint.h>
#define QDO_BASE 0xa0000000u
/* The mapped window is QDO_SLOTS 16 MiB JTLB entries. The shipped engines use 8 (128 MiB, the
 * cdsp_mem size they require); larger values are for window probes and dynamic reservations
 * up to 96 slots (1536 MiB = 1.5 GiB). */
#ifndef QDO_SLOTS
#define QDO_SLOTS 8u
#endif
#define QDO_BYTES (QDO_SLOTS * 0x01000000u)
/* ELF marker lets the host check the window before activating a DSP payload. */
#define QDO_WINDOW_FEATURE {'Q','D','W','I','N','D','0','1',QDO_SLOTS}
#ifndef QDO_FIRST
#define QDO_FIRST (QDO_SLOTS > 64u ? (128u - QDO_SLOTS) : 64u)
#endif
#if QDO_SLOTS > 8
#define QDO_STATE 28672u /* the map outgrows [3072, 3584) at 64+ slots; 28672.. is unused (arena is cleared to 32768) */
#else
#define QDO_STATE 3072u
#endif
#define QDO_MAGIC 0x314d4451u
#ifndef QDO_SCALAR_CACHE
#define QDO_SCALAR_CACHE 0
#endif
#if QDO_SCALAR_CACHE != 0 && QDO_SCALAR_CACHE != 7
#error QDO_SCALAR_CACHE must be 0 or 7
#endif
#if QDO_SLOTS > 8
/* Windows above the shipped 128 MiB take whichever JTLB entries are free when they are mapped, scanning down
 * from the top: the DSP runtime refills entries of its own at indices that change from run to run, so a fixed
 * index range is only free by luck. Entries 0..15 are never used. */
typedef struct { volatile uint32_t active,error; uint64_t saved[QDO_SLOTS]; uint8_t idx[QDO_SLOTS]; uint32_t clean; } QdoMap;
#define QDO_IDX(s,i) ((uint32_t)(s)->idx[i])
#else
typedef struct { volatile uint32_t active,error; uint64_t saved[QDO_SLOTS]; uint32_t clean; } QdoMap;
#define QDO_IDX(s,i) (QDO_FIRST+(i))
#endif

#ifdef __hexagon__
/* Scalar-only code/constants and heap optionally use L1 write-back. Mutable
 * graph pools, I/O and vector staging are normal shared uncached memory.
 * L2 is configured as TCM. Invalidate every private page on both transitions:
 * an earlier session may have used a different cache attribute. */
static int qdo_cached_page(uint32_t i){return i==0 || i==2 || i==3;}
static uint32_t qdo_page_attr(uint32_t i){
#if QDS_L2_CACHE
    if(i==7)return 7u;
    if(i==0 || i==2 || i==3)return QDO_SCALAR_CACHE;
    return 6u;
#else
    return qdo_cached_page(i)?QDO_SCALAR_CACHE:6u;
#endif
}
/* Uncached pages (attr 6) never allocate lines, so only the cacheable ones can hold lines that this wake
 * created or that the host's writes made stale. A full sweep is needed only when an earlier mapping might have
 * used other attributes: the host zeroes the arena (and so `clean`) at session open, and map clears it, so
 * only a wake that follows this layout's own completed unmap in the same session skips the uncached pages.
 * At 80 slots the full sweep is ~42M line operations per transition. */
#if QDS_L2_CACHE
#define QDO_CLEAN_L2 1u
#else
#define QDO_CLEAN_L2 0u
#endif
#define QDO_CLEAN_SIG (0x434c0000u|(QDO_SLOTS<<4)|(QDO_SCALAR_CACHE<<1)|QDO_CLEAN_L2)
static int qdo_page_cacheable(uint32_t i){return qdo_page_attr(i)!=6u;}
static void qdo_clean(int publish,int all) {
    for(uint32_t i=0;i<QDO_SLOTS;i++)
        if(all || qdo_page_cacheable(i))
        for(uint32_t off=0;off<0x1000000u;off+=32) {
            uint32_t p=QDO_BASE+i*0x1000000u+off;
            if(publish && (qdo_cached_page(i)
#if QDS_L2_CACHE
                           || i==7
#endif
                           ))
                __asm__ volatile("dccleaninva(%0)"::"r"(p):"memory");
            else __asm__ volatile("dcinva(%0)"::"r"(p):"memory");
        }
    __asm__ volatile("barrier":::"memory");
}
static uint64_t qdo_tlbr(uint32_t index) {
    uint64_t e;__asm__ volatile("%0=tlbr(%1)":"=r"(e):"r"(index):"memory");return e;
}
static void qdo_tlbw(uint32_t index,uint64_t e) {
    __asm__ volatile("tlbw(%0,%1)\n isync"::"r"(e),"r"(index):"memory");
}
static uint32_t qdo_map(QdoMap *s) {
    if(s->active)return 1;
    for(uint32_t i=0;i<128;i++) {
        uint64_t e=qdo_tlbr(i);
        uint32_t va=((uint32_t)(e>>32)&0xfffffu)<<12;
        if((e>>63) && va>=QDO_BASE && va<QDO_BASE+QDO_BYTES)return 2;
    }
#if QDO_SLOTS > 8
    {
        uint32_t found=0;
        for(uint32_t j=128;j>16 && found<QDO_SLOTS;j--) {
            uint64_t e=qdo_tlbr(j-1);
            if(e>>63)continue;
            s->idx[found]=(uint8_t)(j-1);s->saved[found]=e;found++;
        }
        if(found<QDO_SLOTS)return 3;
    }
#else
    for(uint32_t i=0;i<QDO_SLOTS;i++) {
        s->saved[i]=qdo_tlbr(QDO_FIRST+i);
        if(s->saved[i]>>63)return 3;
    }
#endif
    s->active=QDO_MAGIC;
    for(uint32_t i=0;i<QDO_SLOTS;i++) {
        uint32_t pn=(QDO_BASE>>12)+i*0x1000u;
        uint64_t e=((uint64_t)(0xc0000000u|pn)<<32)|
                   ((0xf0000000u|(qdo_page_attr(i)<<24))|(pn<<1)|0x40u);
        qdo_tlbw(QDO_IDX(s,i),e);
    }
    const int all=s->clean!=QDO_CLEAN_SIG;
    s->clean=0;
    qdo_clean(0,all);
    return 0;
}
static uint32_t qdo_unmap(QdoMap *s) {
    if(s->active!=QDO_MAGIC)return 0;
    qdo_clean(1,0);
    for(uint32_t i=0;i<QDO_SLOTS;i++)qdo_tlbw(QDO_IDX(s,i),s->saved[i]);
    for(uint32_t i=0;i<QDO_SLOTS;i++)if(qdo_tlbr(QDO_IDX(s,i))!=s->saved[i])return 4;
    s->clean=QDO_CLEAN_SIG;
    s->active=0;return 0;
}
#endif /* __hexagon__ */
#endif
