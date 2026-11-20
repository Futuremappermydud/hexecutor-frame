/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Read-only dump of the Turing clock controllers feeding the CDSP's memory path,
 * into arena+0x10000 (worker 0 only). Offsets come from QuRT's clock tables
 * (CDSP.HT.3.0, see bench/qurt_clock_map.py): turing_cc at 0x32008000, the
 * Q6/MXU clock controller at 0x32348000. Registers Valve's driver already
 * writes are read first, so a stall on a later one is identifiable. */
#include <stdint.h>

#define ARENA_UNCACHED 0xf1800000u
#define TURING_CC 0x32008000u
#define Q6_CC 0x32348000u
#define TURING_CC_PLL 0x32000000u  /* QuRT HWIO TURING_CORE0..3_CM_PLL_LUCID_OLE */

/* out[0] holds the address about to be read, so a stalled read is identifiable. */
static void dump(volatile uint32_t *out, uint32_t *n, uint32_t base, uint32_t off, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        out[0] = base + off + 4 * i;
        __asm__ volatile("barrier" ::: "memory");
        out[1 + (*n)++] = *(volatile uint32_t *)(base + off + 4 * i);
    }
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    uint32_t seq = *(volatile uint32_t *)(ARENA_UNCACHED + 8);
    if (tid == 0) {
        volatile uint32_t *out = (volatile uint32_t *)(ARENA_UNCACHED + 0x10000);
        uint32_t n = 0;
        /* Order and count must match FIELDS in run_clk_dump.py. */
        dump(out, &n, TURING_CC, 0x404, 1);  /* q6ss_q6_axim branch (Valve enables) */
        dump(out, &n, TURING_CC, 0x22c, 1);  /* Valve's "NSPNOC_CBCR" */
        dump(out, &n, TURING_CC, 0x414, 1);  /* Valve's "AHBS_AON"; QuRT: alt_reset_aon */
        dump(out, &n, Q6_CC, 0x20, 2);       /* Q6 core RCG CMD/CFG */
        dump(out, &n, Q6_CC, 0x180, 2);      /* HMX RCG CMD/CFG */
        dump(out, &n, TURING_CC, 0x248, 1);  /* nspnoc branch per QuRT */
        dump(out, &n, TURING_CC, 0x410, 1);  /* q6ss_ahbs_aon per QuRT */
        dump(out, &n, TURING_CC, 0x254, 2);  /* ceng/q6_axim RCG CMD/CFG */
        dump(out, &n, TURING_CC, 0x25c, 1);  /* ceng_nsp branch */
        dump(out, &n, TURING_CC, 0x264, 1);  /* ceng_proc branch */
        dump(out, &n, TURING_CC, 0x124, 2);  /* dma_nspnoc/vapss_q6_axi RCG CMD/CFG */
        dump(out, &n, TURING_CC, 0x270, 1);  /* dma_nspnoc branch */
        dump(out, &n, TURING_CC, 0x130, 1);  /* vapss_q6_axi branch */
        /* Raw words, labelled in RAW in run_clk_dump.py. */
        dump(out, &n, TURING_CC, 0x254 + 0x14, 1);  /* q6_axim RCG CMD_DFSR */
        dump(out, &n, TURING_CC, 0x254 + 0x1c, 8);  /* q6_axim DFS perf-level CFGs */
        for (uint32_t pll = 0; pll < 4; pll++)      /* Turing core0-3 Lucid OLE PLLs */
            dump(out, &n, TURING_CC_PLL + pll * 0x1000u, 0x0, 8);  /* MODE..USER_CTL_U */
        out[0] = 0xffffffffu;                /* finished */
    }
    __asm__ volatile("barrier" ::: "memory");
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16 + 12) = 0x54495845u;
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
