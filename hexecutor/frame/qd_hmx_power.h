#ifndef QD_HMX_POWER_H
#define QD_HMX_POWER_H
/* HMX power/clock sequences already validated by the standalone Frame GEMM.
 * Only worker 0 calls these helpers. State survives in the owned arena. */
#define cycles hmx_power_cycles
#include "bench/hmx_power.h"
#undef cycles
typedef struct { uint32_t pll[16],cfg,branch; } QdHmxClocks;
static uint32_t hmx_clock_set(uint32_t hz,QdHmxClocks *save) {
    volatile uint32_t *pll=(volatile uint32_t *)0x32341000u;
    volatile uint32_t *cc=(volatile uint32_t *)0x32348000u;
    for (uint32_t i=0;i<16;i++) save->pll[i]=pll[i];
    save->cfg=cc[0x184/4];save->branch=cc[0x1a0/4];
    if (!hz) return 0;
    /* SM8650 shipped BSP: MXU RCG 0x180, branch 0x1a0; Lucid OLE PLL.
     * Only use frequencies present in the shipped HMX operating-point table. */
    if (hz!=364800000 && hz!=499200000 && hz!=672000000 && hz!=806400000 && hz!=979200000) return 1;
    if (save->cfg!=0 || (save->pll[0]&5) || save->pll[1]!=0) return 2;
    pll[0]=0;
    pll[4]=0x44440000u | (hz/19200000);
    pll[5]=0;
    pll[6]=0x00008401;
    /* Keep this PLL's reset configuration/test values, including its
     * HMX-specific TEST_CTL_U2. Sequence matches the running kernel's
     * clk_pll_configure_lucid_ole, with bounded lock/update waits. */
    pll[1]=0;
    pll[0]=0x00800000;
    pll[0]=0x00800004;
    pll[1]=1;
    uint64_t end=cycles()+200000000;
    while (!(pll[0]&0x80000000u)) if (cycles()>end) return 3;
    pll[0]=5;
    __asm__ volatile("syncht" ::: "memory");
    cc[0x184/4]=0x200;
    cc[0x180/4]=1;
    end=cycles()+200000000;
    while (cc[0x180/4]&1) if (cycles()>end) return 4;
    __asm__ volatile("syncht" ::: "memory");
    return 0;
}
static void hmx_clock_restore(QdHmxClocks *save) {
    volatile uint32_t *pll=(volatile uint32_t *)0x32341000u;
    volatile uint32_t *cc=(volatile uint32_t *)0x32348000u;
    cc[0x184/4]=save->cfg;cc[0x180/4]=1;
    uint64_t end=cycles()+200000000;
    while (cc[0x180/4]&1) if (cycles()>end) break;
    cc[0x1a0/4]=save->branch;
    __asm__ volatile("syncht" ::: "memory");
    pll[0]=0;pll[1]=0;
    /* Only restore writable registers we changed. STATE/STATUS are read-only. */
    pll[4]=save->pll[4];pll[5]=save->pll[5];pll[6]=save->pll[6];
    pll[1]=save->pll[1];pll[0]=save->pll[0];
    __asm__ volatile("syncht" ::: "memory");
}

#define QDC_HMX_STATE_OFFSET 512u
#define QDC_HMX_ACTIVE 0x484d5831u
typedef struct {
    uint32_t active,rsc[17],trigger,address,ssr,powered,clocked;
    QdHmxClocks clocks;
    uint32_t error,restored;
} QdHmxState;
static int hmx_shutdown(QdHmxState *s) {
    if(s->active!=QDC_HMX_ACTIVE)return 0;
    volatile uint32_t *rsc=(volatile uint32_t *)0x323c0000u;
    if(s->powered) {
        __asm__ volatile("ssr=%0\n isync"::"r"(s->ssr):"memory");
        if(s->clocked)hmx_clock_restore(&s->clocks);
        if(power_trigger(rsc,0x24)){s->error=7;return 7;}
    }
    for(uint32_t i=0;i<17;i++)rsc[0x550/4+i]=s->rsc[i];
    rsc[0x464/4]=s->address;rsc[0x460/4]=s->trigger;
    __asm__ volatile("syncht":::"memory");
    s->active=0;s->restored=1;return 0;
}
static int hmx_start(QdHmxState *s) {
    if(s->active==QDC_HMX_ACTIVE)return s->error;
    volatile uint32_t *rsc=(volatile uint32_t *)0x323c0000u;
    if(rsc[0]!=0x30200 || (rsc[0x404/4]&1))return 1;
    uint32_t occupied=0;
    for(uint32_t i=0;i<17;i++){s->rsc[i]=rsc[0x550/4+i];occupied|=s->rsc[i];}
    if(occupied)return 2;
    s->trigger=rsc[0x460/4];s->address=rsc[0x464/4];
    __asm__ volatile("%0=ssr":"=r"(s->ssr));
    s->active=QDC_HMX_ACTIVE;s->error=0;s->restored=0;
    for(uint32_t i=0;i<17;i++)rsc[0x550/4+i]=hmx_power_words[i];
    __asm__ volatile("syncht":::"memory");
    if(power_trigger(rsc,0) || rsc[0x408/4]!=32){s->error=3;hmx_shutdown(s);return 3;}
    s->powered=1;
    uint32_t err=hmx_clock_set(979200000,&s->clocks);
    if(err){s->error=4;if(err>=3)s->clocked=1;hmx_shutdown(s);return 4;}
    s->clocked=1;
    uint32_t enable=s->ssr|(1u<<26);
    __asm__ volatile("ssr=%0\n isync"::"r"(enable):"memory");
    return 0;
}
#endif
