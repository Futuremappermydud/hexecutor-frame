#ifndef QD_OWNED_MEMORY_H
#define QD_OWNED_MEMORY_H
/* Private DDR allocated by cdsp_mem, never firmware/service memory. The host
 * verifies the module size/IOVA before dispatch. Use the same 16 MB JTLB
 * encoding proven by the Qwen engine, but reject occupied entries/aliases
 * rather than evicting them. Save and restore even invalid entry contents. */
#include <stdint.h>
#define QDO_BASE 0xa0000000u
/* The mapped window is QDO_SLOTS 16 MiB JTLB entries. The shipped engines use 8 (128 MiB, the
 * cdsp_mem size they require); larger values are for window probes and need a larger cdsp_mem. */
#ifndef QDO_SLOTS
#define QDO_SLOTS 8u
#endif
#define QDO_BYTES (QDO_SLOTS * 0x01000000u)
#define QDO_FIRST 64u
#define QDO_STATE 3072u
#define QDO_MAGIC 0x314d4451u
#ifndef QDO_SCALAR_CACHE
#define QDO_SCALAR_CACHE 0
#endif
#if QDO_SCALAR_CACHE != 0 && QDO_SCALAR_CACHE != 7
#error QDO_SCALAR_CACHE must be 0 or 7
#endif
typedef struct { volatile uint32_t active,error; uint64_t saved[QDO_SLOTS]; } QdoMap;

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
static void qdo_clean(int publish) {
    for(uint32_t i=0;i<QDO_SLOTS;i++)
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
    for(uint32_t i=0;i<QDO_SLOTS;i++) {
        s->saved[i]=qdo_tlbr(QDO_FIRST+i);
        if(s->saved[i]>>63)return 3;
    }
    s->active=QDO_MAGIC;
    for(uint32_t i=0;i<QDO_SLOTS;i++) {
        uint32_t pn=(QDO_BASE>>12)+i*0x1000u;
        uint64_t e=((uint64_t)(0xc0000000u|pn)<<32)|
                   ((0xf0000000u|(qdo_page_attr(i)<<24))|(pn<<1)|0x40u);
        qdo_tlbw(QDO_FIRST+i,e);
    }
    qdo_clean(0);
    return 0;
}
static uint32_t qdo_unmap(QdoMap *s) {
    if(s->active!=QDO_MAGIC)return 0;
    qdo_clean(1);
    for(uint32_t i=0;i<QDO_SLOTS;i++)qdo_tlbw(QDO_FIRST+i,s->saved[i]);
    for(uint32_t i=0;i<QDO_SLOTS;i++)if(qdo_tlbr(QDO_FIRST+i)!=s->saved[i])return 4;
    s->active=0;return 0;
}
#endif /* __hexagon__ */
#endif
