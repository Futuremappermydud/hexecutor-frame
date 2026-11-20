/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Stage-1 HMX probe: privileged register reads only, no writes to hardware.
 * Results at arena+2048 (uncached): per worker 16 words. */
#include <stdint.h>

#define ARENA_UNCACHED 0xf1800000u
#define QDSP6SS 0x32300000u /* identity-mapped uncached by the firmware TLB */

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid, ssr, cfg;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    volatile uint32_t *out = (volatile uint32_t *)(ARENA_UNCACHED + 2048 + tid * 64);
    uint32_t seq = *(volatile uint32_t *)(ARENA_UNCACHED + 8);
    __asm__ volatile("%0 = ssr" : "=r"(ssr));
    __asm__ volatile("%0 = cfgbase" : "=r"(cfg));
    out[0] = ssr; out[1] = cfg;
    if (tid == 0) {
        uint32_t table = cfg << 16;
        out[15] = 1;                        /* progress: before config table */
        if (table >= 0x32000000u && table + 0xc0 <= 0x33000000u) {
            volatile uint32_t *t = (volatile uint32_t *)table;
            out[2] = t[0x30 / 4];           /* coproc type */
            out[3] = t[0x34 / 4];           /* coproc contexts */
            out[4] = t[0x38 / 4];           /* VTCM base */
            out[5] = t[0x3c / 4];           /* VTCM size */
            out[6] = t[0x78 / 4];           /* HMX int8 rate */
            out[7] = t[0x68 / 4];           /* DMA version */
            out[8] = t[0x08 / 4];           /* subsystem base */
            out[9] = t[0x70 / 4];           /* core id */
        } else {
            out[2] = 0xdeadbeef;            /* table outside mapped window: skipped */
        }
        out[15] = 2;                        /* progress: before RSC sequencer */
        out[10] = *(volatile uint32_t *)(QDSP6SS + 0x30404); /* RSC_SEQ_BUSY_DRV0 */
        out[11] = *(volatile uint32_t *)(QDSP6SS + 0x30460); /* OVERRIDE_TRIGGER_DRV0 */
        out[12] = *(volatile uint32_t *)(QDSP6SS + 0x30464); /* OVERRIDE_TRIGGER_START_ADDR */
        out[15] = 3;
    }
    __asm__ volatile("barrier" ::: "memory");
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16 + 12) = 0x54495845u; /* exited */
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
