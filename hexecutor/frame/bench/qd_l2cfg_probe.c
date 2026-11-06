/* Read-only L2 configuration probe (2026-11-01).
 * Thread 0 reads SYSCFG, CCR, SSR, CFGBASE and the core configuration table
 * (memw_phys at cfgbase<<5, as the standalone runtime's __rdcfg does). It
 * writes no control register, TLB entry or L2 state. */
#include <stdint.h>
#include "../qd_protocol.h"
#include "../qd_conv.h"
#define ARENA 0xf1800000u
static uint32_t rdcfg(uint32_t off) {
    uint32_t base;
    __asm__ volatile("%0=cfgbase":"=r"(base));
    __asm__ volatile("%[b]=asl(%[b],#5)\n\t%[o]=memw_phys(%[o],%[b])":[b]"+r"(base),[o]"+r"(off)::"memory");
    return off;
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;
    uint32_t seq=a[2];
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    if(tid==0) {
        uint32_t v;
        out[0]=0x47464332; /* "2CFG" */
        __asm__ volatile("%0=syscfg":"=r"(v));out[1]=v;
        __asm__ volatile("%0=ccr":"=r"(v));out[2]=v;
        __asm__ volatile("%0=ssr":"=r"(v));out[3]=v;
        __asm__ volatile("%0=cfgbase":"=r"(v));out[4]=v;
        __asm__ volatile("%0=rev":"=r"(v));out[5]=v;
        /* config table words 0x00..0x7c */
        for(uint32_t i=0;i<32;i++)out[16+i]=rdcfg(i*4u);
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=0;st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
