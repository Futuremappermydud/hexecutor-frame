/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Serialized worker-0 GEMM tuning. All tensors remain in VTCM during timing.
 * Wait for workers 1/2 to leave the stock benchmark before using their VTCM.
 */
#include <stdint.h>
#include "hmx_power.h"
#define ARENA 0xf1800000u
#define MAGIC 0x4e555448u
typedef uint32_t V __attribute__((vector_size(128)));
extern void hmx_tune_init(void *scale);
extern void hmx_tune_deep(void *a, void *b, void *c, uint32_t range, uint32_t flags);
extern void hmx_tune_finish(void *c);
extern void hmx_tune_gemm(void *a,void *b,void *c,uint32_t kb,uint32_t nt,uint32_t mt);
extern void hmx_tune_gemm_after(void *a,void *b,void *c,uint32_t kb,uint32_t nt,uint32_t mt);

typedef struct { uint32_t pll[16],cfg,branch; } ClockSaved;
static uint32_t clock_set(uint32_t hz,ClockSaved *save) {
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
static void clock_restore(ClockSaved *save) {
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

static uint64_t timer(void) {
    uint64_t value; __asm__ volatile("%0 = c31:30" : "=r"(value)); return value;
}
static void copy128(volatile V *dst, volatile V *src, uint32_t bytes) {
    for (uint32_t i=0; i<bytes/128; ++i) dst[i]=src[i];
    __asm__ volatile("syncht" ::: "memory");
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid, ssr;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid>=3) return;
    volatile uint32_t *cmd=(volatile uint32_t *)ARENA;
    volatile uint32_t *st=(volatile uint32_t *)(ARENA+128+16*tid);
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    uint32_t seq=cmd[2];
    if (tid) { st[3]=0x54495845; st[0]=seq; __asm__ volatile("barrier" ::: "memory"); return; }
    out[0]=1;
    uint64_t deadline=cycles()+2000000000u;
    while (*(volatile uint32_t *)(ARENA+144)!=seq || *(volatile uint32_t *)(ARENA+160)!=seq) {
        if (cycles()>deadline) {out[1]=1; goto done;}
    }
    uint32_t m=cmd[4], n=cmd[5], k=cmd[6], reps=cmd[7], flags=cmd[8];
    uint32_t ab=m*k*2, bb=k*n*2, cb=m*n*2;
    if (cmd[0]!=MAGIC || !m || !n || !k || m>1024 || n>1024 || k>1024 ||
        ((m|n|k)&31) || !reps || reps>100000 || ab+bb+cb+4096>0x800000) {out[1]=2; goto done;}
    volatile V *a=(volatile V *)0x31000000u;
    volatile V *b=(volatile V *)(0x31000000u+ab);
    volatile V *c=(volatile V *)(0x31000000u+ab+bb);
    volatile V *scale=(volatile V *)(0x31000000u+ab+bb+cb);
    volatile V *completion=scale+16;
    out[0]=2;
    copy128(a,(volatile V *)(ARENA+0x20000),ab);
    copy128(b,(volatile V *)(ARENA+0x20000+ab),bb);
    V zero, scales;
    for (uint32_t i=0;i<32;i++) {zero[i]=0;scales[i]=0x00003c00;}
    for (uint32_t i=0;i<cb/128;i++) c[i]=zero;
    scale[0]=scales;scale[1]=zero;
    V sentinel;
    for (uint32_t i=0;i<32;i++) sentinel[i]=0xffffffffu;
    for (uint32_t i=0;i<16;i++) completion[i]=sentinel;
    volatile uint32_t *rsc=(volatile uint32_t *)0x323c0000u;
    uint32_t saved[17], trig=rsc[0x460/4], addr=rsc[0x464/4];
    if (rsc[0]!=0x30200 || (rsc[0x404/4]&1)) {out[1]=3;goto done;}
    uint32_t occupied=0;
    for (uint32_t i=0;i<17;i++) {saved[i]=rsc[0x550/4+i];occupied|=saved[i];}
    if (occupied) {out[1]=4;goto done;}
    for (uint32_t i=0;i<17;i++) rsc[0x550/4+i]=hmx_power_words[i];
    __asm__ volatile("syncht" ::: "memory");
    out[0]=3;
    out[2]=power_trigger(rsc,0);out[3]=rsc[0x408/4];
    if (out[2] || out[3]!=32) {out[1]=5;goto restore;}
    ClockSaved clocks;
    uint32_t hz=cmd[9];
    out[11]=clock_set(hz,&clocks);
    if (out[11]) {out[1]=7;if (out[11]>=3) clock_restore(&clocks);goto power_off;}
    __asm__ volatile("%0 = ssr" : "=r"(ssr));
    uint32_t enable=ssr|(1u<<26);
    __asm__ volatile("ssr = %0\n isync" :: "r"(enable) : "memory");
    /* Read only: core/MXU PLLs and public clock-controller register window.
     * Snapshot after HMX power-on, while its domain is accessible. */
    volatile uint32_t *clklog=(volatile uint32_t *)(ARENA+0x3000);
    for (uint32_t i=0;i<32;i++) {
        out[10]=0x32340000+4*i;
        clklog[i]=*(volatile uint32_t *)(0x32340000u+4*i);
        out[10]=0x32341000+4*i;
        clklog[32+i]=*(volatile uint32_t *)(0x32341000u+4*i);
    }
    for (uint32_t i=0;i<128;i++) {
        out[10]=0x32348000+4*i;
        clklog[64+i]=*(volatile uint32_t *)(0x32348000u+4*i);
        out[10]=0x3234c000+4*i;
        clklog[192+i]=*(volatile uint32_t *)(0x3234c000u+4*i);
    }
    out[0]=4;
    hmx_tune_init((void *)scale);
    uint64_t t0=timer(), c0=cycles();
    for (uint32_t rep=0;rep<reps;rep++) {
        if (flags==4) {
            hmx_tune_gemm((void *)a,(void *)b,(void *)c,(k/32)*2048,n/32,m/32);
            continue;
        }
        if (flags==8) {
            hmx_tune_gemm_after((void *)a,(void *)b,(void *)c,(k/32)*2048,n/32,m/32);
            continue;
        }
        for (uint32_t mi=0;mi<m/32;mi++) {
            for (uint32_t ni=0;ni<n/32;ni++) {
                hmx_tune_deep((void *)(a+mi*(k/32)*16),
                              (void *)(b+ni*(k/32)*16),
                              (void *)(c+(mi*(n/32)+ni)*16),
                              (k/32)*2048-1,flags);
            }
        }
    }
    /* This address was never used for an earlier output. The ordered HMX
     * zero store followed by HVX read drains the stream without relying
     * on an output address reused by the preceding repetitions. */
    hmx_tune_finish((void *)completion);
    uint64_t elapsed=cycles()-c0, ticks=timer()-t0;
    V completed=completion[0];out[14]=completed[0];
    out[4]=(uint32_t)elapsed;out[5]=elapsed>>32;out[6]=(uint32_t)ticks;out[7]=ticks>>32;
    out[0]=5;
    __asm__ volatile("ssr = %0\n isync" :: "r"(ssr) : "memory");
    copy128((volatile V *)(ARENA+0x20000+ab+bb),c,cb);
    if (hz) clock_restore(&clocks);
    out[12]=*(volatile uint32_t *)0x32348184u;
    out[13]=*(volatile uint32_t *)0x32341000u;
    out[15]=(*(volatile uint32_t *)0x32341010u!=clocks.pll[4] ||
             *(volatile uint32_t *)0x32341014u!=clocks.pll[5] ||
             *(volatile uint32_t *)0x32341018u!=clocks.pll[6] ||
             *(volatile uint32_t *)0x32341004u!=clocks.pll[1] ||
             out[12]!=clocks.cfg ||
             *(volatile uint32_t *)0x323481a0u!=clocks.branch);
power_off:
    out[8]=power_trigger(rsc,32);out[9]=rsc[0x408/4];
    if (out[8] || out[9]!=68) out[1]=6;
restore:
    if (!(rsc[0x404/4]&1)) {
        for (uint32_t i=0;i<17;i++) rsc[0x550/4+i]=saved[i];
        rsc[0x464/4]=addr;rsc[0x460/4]=trig;
        __asm__ volatile("syncht" ::: "memory");
    }
    out[0]=6;
done:
    __asm__ volatile("barrier" ::: "memory");
    st[3]=0x54495845;st[0]=seq;
    __asm__ volatile("barrier" ::: "memory");
}
