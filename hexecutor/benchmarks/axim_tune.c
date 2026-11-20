/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Worker-0 test of the Q6 AXI-master clock (turing_cc+0x254) at 750 MHz.
 * Valve's firmware leaves the RCG on XO and the Turing core PLLs off. QuRT's
 * table (CDSP.HT.3.0-00794) reaches 750 MHz as core2 PLL 1500 MHz main output,
 * RCG source 5 divided by 2, at SVS_L1. Measures HVX DDR loads before, during
 * and after, then restores the RCG and PLL registers. Results at arena+0x10000.
 */
#include <stdint.h>

static uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile("%0 = upcycle" : "=r"(value));
    return value;
}

#define ARENA_UNCACHED 0xf1800000u
#define READ_BASE 0x88900000u  /* cached arena + 1 MB, away from the result page */
#define READ_BYTES 0x100000u
#define TURING_CC 0x32008000u
#define AXIM_CMD (0x254u / 4)
#define AXIM_CFG (0x258u / 4)
#define CORE2_PLL 0x32002000u
#define CFG_SRC_MASK 0x71fu
#define CFG_750MHZ 0x503u      /* source 5 (core2 PLL main), divide by 2 */
/* Core2 PLL: 1500 MHz (78.125 x 19.2) gives 750 MHz. Build with
 * -DAXIM_PLL_L=104 -DAXIM_PLL_ALPHA=0x2aaa for 2000 MHz -> 1000 MHz (NOM). */
#ifndef AXIM_PLL_L
#define AXIM_PLL_L 78u
#define AXIM_PLL_ALPHA 0x2000u
#endif

typedef uint32_t V __attribute__((vector_size(128)));
typedef struct { uint32_t pll[16], cfg; } Saved;

static uint64_t hvx_read(uint32_t *sink) {
    V acc = {0};
    uint64_t start = cycles();
    for (uint32_t o = 0; o < READ_BYTES; o += 1024) {
        const V *s = (const V *)(READ_BASE + o);
        acc ^= s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7];
    }
    *sink ^= acc[0];
    return cycles() - start;
}

/* Eight independent accumulators so loads can overlap: a throughput figure. */
static uint64_t hvx_read_overlap(uint32_t *sink) {
    V a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0}, a4 = {0}, a5 = {0}, a6 = {0}, a7 = {0};
    uint64_t start = cycles();
    for (uint32_t o = 0; o < READ_BYTES; o += 1024) {
        const V *s = (const V *)(READ_BASE + o);
        a0 += s[0]; a1 += s[1]; a2 += s[2]; a3 += s[3];
        a4 += s[4]; a5 += s[5]; a6 += s[6]; a7 += s[7];
    }
    *sink ^= (a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7)[0];
    return cycles() - start;
}

static int wait_clear(volatile uint32_t *reg, uint32_t bit) {
    uint64_t end = cycles() + 200000000;
    while (*reg & bit) if (cycles() > end) return -1;
    return 0;
}

/* Same Lucid OLE bring-up as hmx_tune.c's clock_set, with bounded waits. */
static int pll_start(volatile uint32_t *pll, Saved *save) {
    for (uint32_t i = 0; i < 16; i++) save->pll[i] = pll[i];
    if ((save->pll[0] & 5) || save->pll[1] != 0) return 2;  /* expect it unused */
    pll[0] = 0;
    pll[4] = 0x44440000u | AXIM_PLL_L;
    pll[5] = AXIM_PLL_ALPHA;
    pll[6] = 0x00008401;
    pll[1] = 0;
    pll[0] = 0x00800000;
    pll[0] = 0x00800004;
    pll[1] = 1;
    uint64_t end = cycles() + 200000000;
    while (!(pll[0] & 0x80000000u)) if (cycles() > end) return 3;
    pll[0] = 5;
    __asm__ volatile("syncht" ::: "memory");
    return 0;
}

static void pll_restore(volatile uint32_t *pll, const Saved *save) {
    pll[0] = 0; pll[1] = 0;
    pll[4] = save->pll[4]; pll[5] = save->pll[5]; pll[6] = save->pll[6];
    pll[1] = save->pll[1]; pll[0] = save->pll[0];
    __asm__ volatile("syncht" ::: "memory");
}

static int rcg_set(volatile uint32_t *cc, uint32_t cfg) {
    cc[AXIM_CFG] = cfg;
    cc[AXIM_CMD] |= 1;
    int ret = wait_clear(&cc[AXIM_CMD], 1);
    __asm__ volatile("syncht" ::: "memory");
    return ret;
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    uint32_t seq = *(volatile uint32_t *)(ARENA_UNCACHED + 8);
    if (tid == 0) {
        /* Order must match FIELDS in run_axim_tune.py. */
        volatile uint32_t *out = (volatile uint32_t *)(ARENA_UNCACHED + 0x10000);
        volatile uint32_t *cc = (volatile uint32_t *)TURING_CC;
        volatile uint32_t *pll = (volatile uint32_t *)CORE2_PLL;
        Saved save;
        uint32_t sink = 0;
        out[0] = 1;                                   /* progress */
        hvx_read(&sink);                              /* warm TLB */
        out[2] = (uint32_t)hvx_read(&sink);           /* XO cycles */
        out[12] = (uint32_t)hvx_read_overlap(&sink);  /* XO, overlapped */
        save.cfg = cc[AXIM_CFG];
        out[3] = save.cfg;
        int err = pll_start(pll, &save);
        out[4] = pll[0];
        if (!err) {
            out[0] = 2;
            if (rcg_set(cc, (save.cfg & ~CFG_SRC_MASK) | CFG_750MHZ)) err = 4;
            out[5] = cc[AXIM_CFG];
            out[6] = cc[AXIM_CMD];
            if (!err) {
                out[0] = 3;
                out[7] = (uint32_t)hvx_read(&sink);   /* 750 MHz cycles */
                out[13] = (uint32_t)hvx_read_overlap(&sink);
            }
            if (rcg_set(cc, save.cfg) && !err) err = 5;
            out[0] = 4;
            pll_restore(pll, &save);
        }
        out[8] = cc[AXIM_CFG];
        out[9] = pll[0];
        out[10] = (uint32_t)hvx_read(&sink);          /* restored cycles */
        out[14] = (uint32_t)hvx_read_overlap(&sink);
        out[11] = sink;
        out[1] = (uint32_t)err;
        out[0] = 0xffffffffu;                         /* finished */
    }
    __asm__ volatile("barrier" ::: "memory");
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16 + 12) = 0x54495845u;
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
