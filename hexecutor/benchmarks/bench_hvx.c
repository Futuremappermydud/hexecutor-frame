/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Microbenchmarks for the Frame CDSP worker callback (runs behind qd_loader).
 * Command at arena+0 (uncached view); per-thread results at arena+2048. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define ARENA_UNCACHED 0xf1800000u
typedef struct { uint32_t magic, sequence, mode, iters, addr, bytes, stride, threads; } Cmd;
typedef struct { uint32_t done, cyc_lo, cyc_hi, result, pk_lo, pk_hi, r0, r1; } Out;

static inline uint64_t cycles(void) {
    uint64_t c; __asm__ volatile("%0 = upcycle" : "=r"(c)); return c;
}
static inline uint64_t packets(void) {
    uint64_t c; __asm__ volatile("%0 = c19:18" : "=r"(c)); return c; /* PKTCOUNT */
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid; __asm__ volatile("%0 = htid" : "=r"(tid));
    volatile Cmd *cmd = (volatile Cmd *)ARENA_UNCACHED;
    volatile Out *out = (volatile Out *)(ARENA_UNCACHED + 2048 + tid * 32);
    if (tid >= 3) return;
    uint32_t seq = cmd->sequence, mode = cmd->mode, iters = cmd->iters;
    uint32_t addr = cmd->addr, bytes = cmd->bytes, stride = cmd->stride;
    uint32_t result = 0;
    uint64_t c0 = cycles(), p0 = packets();
    if (cmd->magic != 0x48434e42u || tid >= cmd->threads) {
        /* idle */
    } else if (mode == 1) { /* 4 independent vmpyiacc chains, registers only */
        HVX_Vector a0 = Q6_V_vsplat_R(1), a1 = a0, a2 = a0, a3 = a0, w = Q6_V_vsplat_R(3);
        for (uint32_t i = 0; i < iters; i++) {
            a0 = Q6_Vw_vmpyiacc_VwVwRh(a0, w, 0x10001u * i);
            a1 = Q6_Vw_vmpyiacc_VwVwRh(a1, w, 0x20002u);
            a2 = Q6_Vw_vmpyiacc_VwVwRh(a2, w, 0x30003u);
            a3 = Q6_Vw_vmpyiacc_VwVwRh(a3, w, 0x40004u);
        }
        HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1), Q6_Vw_vadd_VwVw(a2, a3));
        result = Q6_R_vextract_VR(s, 0);
    } else if (mode == 2) { /* streaming vector loads: iters passes over [addr, addr+bytes) */
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t p = 0; p < iters; p++)
            for (uint32_t o = 0; o < bytes; o += stride)
                acc = Q6_Vw_vadd_VwVw(acc, *(const HVX_Vector *)(addr + o));
        result = Q6_R_vextract_VR(acc, 0);
    } else if (mode == 3) { /* scalar word loads */
        uint32_t acc = 0;
        for (uint32_t p = 0; p < iters; p++)
            for (uint32_t o = 0; o < bytes; o += stride) acc += *(const volatile uint32_t *)(addr + o);
        result = acc;
    } else if (mode == 4) { /* 4 independent vrmpy u8*s8 chains (128 MAC each) */
        HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0;
        HVX_Vector x = Q6_V_vsplat_R(0x01020304), w0 = Q6_V_vsplat_R(0x05060708);
        for (uint32_t i = 0; i < iters; i++) {
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, x, w0);
            a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, x, w0);
            a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, x, w0);
            a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, x, w0);
        }
        HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1), Q6_Vw_vadd_VwVw(a2, a3));
        result = Q6_R_vextract_VR(s, 0);
    } else if (mode == 6) { /* copy addr -> dst (stride), 8 loads in flight per group */
        uint32_t dst = 0x31000000u + tid * 0xF0000u;
        for (uint32_t p = 0; p < iters; p++)
            for (uint32_t o = 0; o < bytes; o += 1024) {
                const HVX_Vector *s8 = (const HVX_Vector *)(addr + tid * bytes + o);
                HVX_Vector *d8 = (HVX_Vector *)(dst + (o & 0x7ffff));
                HVX_Vector v0 = s8[0], v1 = s8[1], v2 = s8[2], v3 = s8[3];
                HVX_Vector v4 = s8[4], v5 = s8[5], v6 = s8[6], v7 = s8[7];
                d8[0] = v0; d8[1] = v1; d8[2] = v2; d8[3] = v3;
                d8[4] = v4; d8[5] = v5; d8[6] = v6; d8[7] = v7;
            }
        result = Q6_R_vextract_VR(*(HVX_Vector *)dst, 0);
    } else if (mode == 7) { /* per-thread offset streaming loads (addr + tid*bytes) */
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t p = 0; p < iters; p++)
            for (uint32_t o = 0; o < bytes; o += stride)
                acc = Q6_Vw_vadd_VwVw(acc, *(const HVX_Vector *)(addr + tid * bytes + o));
        result = Q6_R_vextract_VR(acc, 0);
    } else if (mode == 5) { /* scalar ALU loop, one packet per iteration */
        uint32_t x = 0;
        for (uint32_t i = 0; i < iters; i++) __asm__ volatile("%0 = add(%0, #1)" : "+r"(x));
        result = x;
    }
    uint64_t c1 = cycles(), p1 = packets();
    out->cyc_lo = (uint32_t)(c1 - c0); out->cyc_hi = (uint32_t)((c1 - c0) >> 32);
    out->pk_lo = (uint32_t)(p1 - p0); out->pk_hi = (uint32_t)((p1 - p0) >> 32);
    out->result = result;
    __asm__ volatile("barrier" ::: "memory");
    out->done = seq;
    /* Generic QdStatus.done_sequence, which the transport waits on. */
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
