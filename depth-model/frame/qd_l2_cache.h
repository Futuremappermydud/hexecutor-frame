#ifndef QD_L2_CACHE_H
#define QD_L2_CACHE_H
#error "QDS_L2_CACHE executor mode is disabled until the 512 KiB TCM carve-out and worker state are handled"
/* Frame V75, verified 2026-11-01: SYSCFG.L2CFG=1 can use 512 KiB of L2 cache
 * and displace TCM [0x30080000,0x30100000) under a large DDR workload.
 * This executor path is disabled in build_resident_dsp.sh pending safe mapping.
 * Only the prepared master may enter,
 * on the private uncached DDR stack at 0xa7000000..0xa7010000. */
#include <stdint.h>
#define QDS_L2_STACK_BASE 0xa7000000u
#define QDS_L2_STACK_TOP  0xa7010000u
#define QDS_L2_BACKUP     0xa7010000u
#define QDS_L2_CARVE      0x30080000u
#define QDS_L2_CARVE_BYTES 0x10000u
static uint32_t qds_l2_syscfg(void){uint32_t v;__asm__ volatile("%0=syscfg":"=r"(v));return v;}
static uint32_t qds_l2_ccr(void){uint32_t v;__asm__ volatile("%0=ccr":"=r"(v));return v;}
static void qds_l2_setcfg(uint32_t v){
    __asm__ volatile("isync\n\tsyncht\n\tsyscfg=%0\n\tisync"::"r"(v):"memory");
}
typedef struct {uint32_t syscfg,ccr,active;} QdsL2Cache;
static uint32_t qds_l2_begin(QdsL2Cache *s){
    s->active=0;s->syscfg=qds_l2_syscfg();s->ccr=qds_l2_ccr();
    if(((s->syscfg>>16)&7u)!=0u || s->syscfg!=0x0a80a057u || s->ccr!=0x150049u)return 1;
    volatile uint32_t *from=(volatile uint32_t *)QDS_L2_CARVE;
    volatile uint32_t *to=(volatile uint32_t *)QDS_L2_BACKUP;
    for(uint32_t i=0;i<QDS_L2_CARVE_BYTES/4u;i++)to[i]=from[i];
    uint32_t cleared=s->ccr&~(15u<<16);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
    qds_l2_setcfg(s->syscfg&~(7u<<16));
    __asm__ volatile("l2kill\n\tsyncht":::"memory");
    qds_l2_setcfg(s->syscfg|(1u<<16));
    s->active=1;
    uint32_t bad=qds_l2_syscfg()!=(s->syscfg|(1u<<16));
    __asm__ volatile("ccr=%0\n\tisync"::"r"(s->ccr):"memory");
    return bad;
}
static uint32_t qds_l2_end(QdsL2Cache *s){
    if(!s->active)return 0;
    uint32_t cleared=s->ccr&~(15u<<16);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
    __asm__ volatile("syncht\n\tl2gcleaninv\n\tsyncht":::"memory");
    qds_l2_setcfg(s->syscfg);
    volatile uint32_t *from=(volatile uint32_t *)QDS_L2_BACKUP;
    volatile uint32_t *to=(volatile uint32_t *)QDS_L2_CARVE;
    for(uint32_t i=0;i<QDS_L2_CARVE_BYTES/4u;i++)to[i]=from[i];
    __asm__ volatile("ccr=%0\n\tisync"::"r"(s->ccr):"memory");
    s->active=0;
    return qds_l2_syscfg()!=s->syscfg || qds_l2_ccr()!=s->ccr;
}
#endif
