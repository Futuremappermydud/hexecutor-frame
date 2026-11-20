/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Staged, single-owner HMX bring-up. Configuration accesses use memw_phys:
 * cfgbase is a physical address, not a firmware virtual mapping.
 * Only thread 0 touches configuration/hardware. Other workers exit normally.
 */
#include <stdint.h>
#define ARENA 0xf1800000u
#ifdef HMX_EXEC
extern void hmx_clear(void);
extern void hmx_dot(void *, void *, void *, void *);
typedef uint32_t vector128 __attribute__((vector_size(128)));
#endif

static uint32_t physread(uint32_t lo, uint32_t hi) {
    uint32_t value;
    __asm__ volatile("%0 = memw_phys(%1,%2)" : "=r"(value) : "r"(lo), "r"(hi) : "memory");
    return value;
}

static uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile("%0 = upcycle" : "=r"(value));
    return value;
}

/* Exact HMX subsequences recovered from this device's stock QuRT ELF.
 * Source/provenance is recorded in build/hmx-qurt-sequence.json.
 * These do not include the neighboring whole-core sleep sequences. */
static const uint32_t hmx_power_words[17] = {
    0x503d502du,0x70275024u,0x50364024u,0x403d7020u,
    0x5039503cu,0x7025403cu,0x503a4030u,0x10005026u,
    0x70285025u,0x503c503bu,0x403c5038u,0x40255037u,
    0x4038403bu,0x60204037u,0x70265023u,0x502e4023u,0x1000402eu
};

static uint32_t power_trigger(volatile uint32_t *rsc, uint32_t address) {
    rsc[0x410/4] = (rsc[0x410/4] & 1u) | 0x80000000u;
    rsc[0x464/4] = address;
    rsc[0x460/4] &= ~1u;
    __asm__ volatile("syncht" ::: "memory");
    rsc[0x460/4] |= 1u;
    __asm__ volatile("syncht" ::: "memory");
    uint64_t start = cycles();
    /* QuRT waits for the completion latch, then clears it via bit 31.
     * Merely observing BUSY=0 can race the start of the sequence. */
    while (!(rsc[0x410/4] & 1u)) {
        if (cycles() - start > 200000000u) return 1;
    }
    rsc[0x460/4] &= ~1u;
    rsc[0x410/4] = (rsc[0x410/4] & 1u) | 0x80000000u;
    __asm__ volatile("syncht" ::: "memory");
    return 0;
}

#ifdef HMX_EXEC
static void hmx_test(uint32_t ssr, uint32_t stage, volatile uint32_t *out) {
        uint32_t enabled = ssr | (1u << 26);
        out[15] = 9;
        __asm__ volatile("ssr = %0\n isync" :: "r"(enabled) : "memory");
        hmx_clear();
        out[15] = 10;
        if (stage != 9) {
            vector128 ones, zero, sentinel, scales;
            for (uint32_t i = 0; i < 32; i++) {
                ones[i] = 0x3c003c00u;
                zero[i] = 0;
                sentinel[i] = 0xffffffffu;
                scales[i] = 0x00003c00u; /* FP16 scale=1, bias=0 per column */
            }
            volatile vector128 *act = (volatile vector128 *)0x31000000u;
            volatile vector128 *weight = (volatile vector128 *)0x31001000u;
            volatile vector128 *output = (volatile vector128 *)0x31002000u;
            volatile vector128 *scale = (volatile vector128 *)0x31003000u;
            for (uint32_t i = 0; i < 16; i++) {
                if (stage == 13 || stage == 14) {
                    act[i] = ((volatile vector128 *)(ARENA + 0x20000))[i];
                    weight[i] = ((volatile vector128 *)(ARENA + 0x21000))[i];
                } else { act[i] = ones; weight[i] = ones; }
                output[i] = sentinel;
            }
            scale[0] = scales; scale[1] = zero;
            __asm__ volatile("barrier" ::: "memory");
            out[15] = 11;
            uint32_t iterations = stage == 14 ? *(volatile uint32_t *)(ARENA + 16) : 1;
            if (iterations == 0 || iterations > 1000000) iterations = 1;
            uint64_t begin = cycles();
            for (uint32_t i = 0; i < iterations; i++) {
                hmx_dot((void *)act, (void *)weight, (void *)output, (void *)scale);
                /* A VTCM read forces the preceding HMX store to finish. */
                vector128 completed = output[0];
                __asm__ volatile("" :: "v"(completed) : "memory");
            }
            uint64_t elapsed = cycles() - begin;
            out[9] = (uint32_t)elapsed; out[10] = elapsed >> 32;
            out[11] = iterations;
            out[15] = 12;
            volatile vector128 *host = (volatile vector128 *)(ARENA + 0x18000);
            for (uint32_t i = 0; i < 16; i++) host[i] = output[i];
            __asm__ volatile("barrier" ::: "memory");
        }
        __asm__ volatile("ssr = %0\n isync" :: "r"(ssr) : "memory");
    }
#endif

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid, ssr, cfg;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    volatile uint32_t *a = (volatile uint32_t *)ARENA;
    volatile uint32_t *out = (volatile uint32_t *)(ARENA + 2048 + tid * 64);
    uint32_t seq = a[2], stage = a[3];
    __asm__ volatile("%0 = ssr" : "=r"(ssr));
    __asm__ volatile("%0 = cfgbase" : "=r"(cfg));
    out[0] = ssr; out[1] = cfg; out[15] = 1;
    if (tid == 0 && (stage == 11 || stage == 12 || stage == 13 || stage == 14)) {
        volatile uint32_t *rsc = (volatile uint32_t *)0x323c0000u;
        uint32_t saved[17], trigger = rsc[0x460/4], address = rsc[0x464/4];
        out[15] = 20;
        if (rsc[0] != 0x30200u || (rsc[0x404/4] & 1u)) {
            out[2] = 0xbad00001u;
        } else {
            uint32_t occupied = 0;
            for (uint32_t i = 0; i < 17; i++) {
                saved[i] = rsc[0x550/4+i]; occupied |= saved[i];
            }
            if (occupied) out[2] = 0xbad00002u;
            else {
                for (uint32_t i = 0; i < 17; i++) rsc[0x550/4+i] = hmx_power_words[i];
                __asm__ volatile("syncht" ::: "memory");
                uint32_t mismatch = 0;
                for (uint32_t i = 0; i < 17; i++) mismatch |= rsc[0x550/4+i] ^ hmx_power_words[i];
                out[15] = 21;
                out[2] = mismatch ? 0xbad00003u : power_trigger(rsc, 0);
                out[3] = rsc[0x404/4]; out[4] = rsc[0x460/4];
                out[7] = rsc[0x408/4];
                out[15] = 22;
                if (!out[2]) {
#ifdef HMX_EXEC
                    if (stage >= 12) hmx_test(ssr, stage == 12 ? 10 : stage, out);
#endif
                    out[5] = power_trigger(rsc, 32);
                    out[6] = rsc[0x404/4];
                    out[8] = rsc[0x408/4];
                    out[15] = 23;
                }
                if (!(rsc[0x404/4] & 1u)) {
                    for (uint32_t i = 0; i < 17; i++) rsc[0x550/4+i] = saved[i];
                    rsc[0x464/4] = address; rsc[0x460/4] = trigger;
                    __asm__ volatile("syncht" ::: "memory");
                    out[15] = 24;
                }
            }
        }
        volatile uint32_t *dump = (volatile uint32_t *)(ARENA + 0x12000);
        for (uint32_t i = 0; i < 0x1000/4; i++) dump[i] = rsc[i];
    }
#ifdef HMX_EXEC
    if (tid == 0 && (stage == 9 || stage == 10)) hmx_test(ssr, stage, out);
#endif
    if (tid == 0 && stage == 7) {
        static const uint32_t addresses[] = {
            0x32300000, 0x32300408, 0x32330404, 0x323b0404,
            0x323c0000, 0x323c0404, 0x30180008, 0x30180078
        };
        for (uint32_t i = 0; i < 8; i++) {
            uint32_t address = addresses[i];
            out[14] = address;
            __asm__ volatile("barrier" ::: "memory");
            out[2 + i] = physread(address & 0x7fc, address >> 11);
        }
        out[15] = 7;
    }
    if (tid == 0 && stage >= 1) {
        volatile uint32_t *table = (volatile uint32_t *)(ARENA + 0x10000);
        for (uint32_t off = 0; off < 0xc0; off += 4) {
            out[14] = off;
            __asm__ volatile("barrier" ::: "memory");
            /* memw_phys uses an 11-bit byte offset and PA >> 11,
             * rather than the usual low/high 32-bit address pair. */
            table[off / 4] = physread(off, cfg << 5);
        }
        if (stage != 7 && stage < 9) out[15] = 2;
        if (stage == 4 || stage == 5 || stage == 6 || stage == 8) {
            /* cfg SSBASE is already the private base. H2 subtracts
             * 0x80000 in boot.ref.S, then adds it back in setup.ref.c. */
            uint32_t base = (table[2] << 16);
            /* SM8650's shipped QuRT HWIO table places RSCC at public
             * base + 0xc0000, i.e. private base + 0x40000. */
            if (stage == 6 || stage == 8) base += 0x10000;
            out[2] = base;
            out[14] = 0x30404;
            __asm__ volatile("barrier" ::: "memory");
            out[3] = *(volatile uint32_t *)(base + 0x30404);
            out[14] = 0x30460;
            __asm__ volatile("barrier" ::: "memory");
            out[4] = *(volatile uint32_t *)(base + 0x30460);
            out[14] = 0x30464;
            __asm__ volatile("barrier" ::: "memory");
            out[5] = *(volatile uint32_t *)(base + 0x30464);
            out[15] = 3;
            if (stage == 5 || stage == 6 || stage == 8) {
                volatile uint32_t *dump = (volatile uint32_t *)(ARENA + 0x12000);
                /* RSCC identification and sequence memory window. */
                for (uint32_t off = 0; off < (stage == 8 ? 0x4000u : 0x1000u); off += 4) {
                    out[14] = 0x30000 + off;
                    __asm__ volatile("barrier" ::: "memory");
                    dump[off / 4] = *(volatile uint32_t *)(base + 0x30000 + off);
                }
                out[15] = 5;
            }
        }
        if (stage == 3) {
            volatile uint64_t *tlbs = (volatile uint64_t *)(ARENA + 0x11000);
            for (uint32_t index = 0; index < 128; index++) {
                uint64_t entry;
                __asm__ volatile("%0 = tlbr(%1)" : "=r"(entry) : "r"(index));
                tlbs[index] = entry;
            }
            out[15] = 4;
        }
    }
    __asm__ volatile("barrier" ::: "memory");
    volatile uint32_t *status = (volatile uint32_t *)(ARENA + 128 + tid * 16);
    status[3] = 0x54495845u;
    status[0] = seq;
    __asm__ volatile("barrier" ::: "memory");
}
