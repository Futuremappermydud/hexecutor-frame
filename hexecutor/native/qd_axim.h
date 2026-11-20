/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_AXIM_H
#define QD_AXIM_H
/* Same 750 MHz AXI-master clock sequence validated by axim_tune and Qwen.
 * Only worker 0 calls these helpers. Save state in the arena, outside packets
 * and TCM that the stock wake clears. Restore before publishing its EXIT. */
#define QDC_AXIM_CC ((volatile uint32_t *)0x32008254u)
#define QDC_CORE2_PLL ((volatile uint32_t *)0x32002000u)
#define QDC_AXIM_MAGIC 0x41584931u
#ifndef QDC_AXIM_MHZ
#define QDC_AXIM_MHZ 750
#endif
#if QDC_AXIM_MHZ == 750
#define QDC_AXIM_PLL_L 78u
#define QDC_AXIM_PLL_ALPHA 0x2000u
#elif QDC_AXIM_MHZ == 1000
/* Shipped NOM operating point; require the existing NOM rail votes. */
#define QDC_AXIM_PLL_L 104u
#define QDC_AXIM_PLL_ALPHA 0x2aaau
#else
#error Unsupported AXI operating point
#endif

static int clock_wait(volatile uint32_t *reg, uint32_t mask, uint32_t want) {
    uint64_t end=cycles()+200000000u;
    while((*reg & mask)!=want) if(cycles()>end) return 1;
    return 0;
}
static int clock_select(uint32_t cfg) {
    QDC_AXIM_CC[1]=cfg;
    QDC_AXIM_CC[0]|=1;
    int error=clock_wait(&QDC_AXIM_CC[0],1,0);
    __asm__ volatile("syncht" ::: "memory");
    return error;
}
static int clock_restore(volatile uint32_t *state) {
    if(state[0]!=QDC_AXIM_MAGIC) return 0;
    volatile uint32_t *pll=QDC_CORE2_PLL;
    /* Do not disable a PLL that an unsuccessful mux restore may still use. */
    if(clock_select(state[1])) { state[21]=5; return 5; }
    pll[0]=0; pll[1]=0;
    pll[4]=state[6]; pll[5]=state[7]; pll[6]=state[8];
    pll[1]=state[3]; pll[0]=state[2];
    __asm__ volatile("syncht" ::: "memory");
    state[20]=QDC_AXIM_CC[1];
    state[0]=0;
    return 0;
}
static int clock_raise(volatile uint32_t *state) {
    if(state[0]==QDC_AXIM_MAGIC) return 0;
    volatile uint32_t *pll=QDC_CORE2_PLL;
    state[1]=QDC_AXIM_CC[1];
    for(uint32_t i=0;i<16;i++) state[2+i]=pll[i];
    if((state[2]&5u) || state[3]!=0) { state[21]=2; return 2; }
    state[0]=QDC_AXIM_MAGIC;
    pll[0]=0;
    pll[4]=0x44440000u|QDC_AXIM_PLL_L; pll[5]=QDC_AXIM_PLL_ALPHA; pll[6]=0x00008401u;
    pll[1]=0; pll[0]=0x00800000u; pll[0]=0x00800004u; pll[1]=1;
    if(clock_wait(&pll[0],0x80000000u,0x80000000u)) {
        state[21]=3; clock_restore(state); return 3;
    }
    pll[0]=5;
    __asm__ volatile("syncht" ::: "memory");
    if(clock_select((state[1]&~0x71fu)|0x503u)) {
        state[21]=4; clock_restore(state); return 4;
    }
    state[18]=QDC_AXIM_CC[1]; state[19]=pll[0];
    return 0;
}
#endif
