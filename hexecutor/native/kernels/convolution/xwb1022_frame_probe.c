/* One-shot Frame probe for the simulator-proven generic V75 Xwb 3x3 leaf.
 * This does not call the loader and is not a performance benchmark. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
static uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile("%0=upcycle" : "=r"(value));
    return value;
}
#ifdef XWB1022_SIM
#include <stdio.h>
#include <string.h>
#endif
#include "../../qd_protocol.h"
#include "../../qd_conv.h"
#ifndef XWB1022_SIM
#include "../../qd_axim.h"
#include "../../qd_hmx_power.h"
#endif

#ifndef ARENA
#define ARENA 0xf1800000u
#endif
#define MAGIC 0x334b4851u
#define FIXTURE 0x10000u
#define FIXTURE_HEADER 204u
#define OUTPUT_COPY 0x120000u
#define INPUT_ECHO 0x130000u
#define TABLES 0x170000u
#define DONE_OFFSET 0x210000u
#define BACKUP 0x300000u
#define WORKSPACE_OFFSET 0x700000u
#define VTCM_BYTES 0x80000u
#define VT_INPUT 0x00000u
#define VT_INPUT_GUARD 0x33000u
#define VT_OUTPUT 0x38000u
#define VT_OUTPUT_GUARD 0x48000u
#define VT_WEIGHTS 0x4a000u
#define VT_FI 0x4d000u
#define VT_ZERO 0x4e000u
#define VT_TABLES 0x4f000u
#define VT_DONE 0x4f800u
#define IN_BYTES 208896u
#define OUT_BYTES 65536u
#define W_BYTES 9216u
#define FI_BYTES 256u

extern void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);
extern void qd_set_hmx_params_stride1(uint32_t *, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t);
extern void qd_hmx_finish(void *, void *);
extern void qds_on_stack(void (*entry)(uint32_t), uint32_t tid, uint32_t stack);

static uint8_t *live_vtcm;
#ifndef XWB1022_SIM
void *memset(void *dst, int value, uint32_t bytes) {
    uint8_t *p = dst;
    while (bytes--) *p++ = (uint8_t)value;
    return dst;
}
#endif
static void copybytes(void *dst, const void *src, uint32_t bytes) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    while (bytes >= 128u) {
        *(HVX_UVector *)d = *(const HVX_UVector *)s;
        d += 128u; s += 128u; bytes -= 128u;
    }
    while (bytes--) *d++ = *s++;
}

static void copy128(void *dst, const void *src, uint32_t bytes) {
    for (uint32_t i = 0; i < bytes; i += 128u)
        *(HVX_Vector *)((uint8_t *)dst + i) =
            *(const HVX_Vector *)((const uint8_t *)src + i);
}

static void fill128(void *dst, uint32_t bytes, uint32_t value) {
    HVX_Vector fill = Q6_V_vsplat_R(value);
    for (uint32_t i = 0; i < bytes; i += 128u)
        *(HVX_Vector *)((uint8_t *)dst + i) = fill;
}

static uint32_t fnv32(const uint8_t *data, uint32_t bytes) {
    uint32_t hash = 2166136261u;
    while (bytes--) hash = (hash ^ *data++) * 16777619u;
    return hash;
}
#ifndef XWB1022_SIM
static int same128(const uint8_t *a, const uint8_t *b, uint32_t bytes) {
    HVX_Vector diff = Q6_V_vzero();
    for (uint32_t i = 0; i < bytes; i += 128)
        diff = Q6_V_vor_VV(diff, Q6_V_vxor_VV(*(const HVX_Vector *)(a + i),
                                              *(const HVX_Vector *)(b + i)));
    uint32_t any = 0;
    for (uint32_t i = 0; i < 32; ++i)
        any |= (uint32_t)Q6_R_vextract_VR(diff, i * 4u);
    return any == 0;
}
#endif

static int drain(int32_t *zero_words, volatile uint8_t *done) {
    *(volatile uint32_t *)done = 0xffffffffu;
    qd_hmx_finish(zero_words, (void *)done);
    uint64_t deadline = cycles() + 200000000u;
    while (*(volatile uint32_t *)done)
        if (cycles() > deadline) return 1;
    return 0;
}

#ifndef XWB1022_SIM
static int tlb(volatile uint32_t *result, uint32_t before_restore) {
    uint64_t e8, e9;
    __asm__ volatile("%0=tlbr(%1)" : "=r"(e8) : "r"(8));
    __asm__ volatile("%0=tlbr(%1)" : "=r"(e9) : "r"(9));
    uint32_t at = before_restore ? 23u : 19u;
    result[at] = (uint32_t)e8;
    result[at + 1u] = (uint32_t)(e8 >> 32);
    result[at + 2u] = (uint32_t)e9;
    result[at + 3u] = (uint32_t)(e9 >> 32);
    return e8 == 0xc0031000f7062020ull && e9 == 0xc0031400f7062820ull;
}
#endif

static int in_vtcm_window(const uint8_t *base, const uint8_t *ptr, uint32_t bytes) {
    uintptr_t start = (uintptr_t)base;
    uintptr_t address = (uintptr_t)ptr;
    return address >= start && address + bytes >= address &&
           address + bytes <= start + VTCM_BYTES;
}

static void run_kernel(uint32_t unused) {
    (void)unused;
    volatile uint32_t *result = (volatile uint32_t *)(ARENA + 2048u);
    uint8_t *arena = (uint8_t *)ARENA;
    const uint8_t *fixture = arena + FIXTURE;
    const uint32_t *header = (const uint32_t *)fixture;
    const uint8_t *input_source = fixture + FIXTURE_HEADER;
    const uint8_t *weight_source = input_source + IN_BYTES;
    const uint8_t *fi_source = weight_source + W_BYTES;

    if (header[0] != MAGIC || header[1] != 1u || header[2] != 1022u ||
        header[3] != 836u || header[4] != 102u || header[5] != 32u ||
        header[6] != IN_BYTES || header[7] != W_BYTES || header[8] != FI_BYTES ||
        header[9] != OUT_BYTES || header[10] != OUT_BYTES) {
        result[1] = 21;
        return;
    }

    if (FIXTURE + FIXTURE_HEADER + IN_BYTES + W_BYTES + FI_BYTES + 2u * OUT_BYTES > TABLES ||
        OUTPUT_COPY + OUT_BYTES > INPUT_ECHO || INPUT_ECHO + IN_BYTES > TABLES ||
        TABLES + 4096u > DONE_OFFSET || DONE_OFFSET + 128u > BACKUP ||
        BACKUP + VTCM_BYTES > 0x800000u || WORKSPACE_OFFSET + VTCM_BYTES > 0x800000u) {
        result[1] = 22;
        return;
    }

    uint8_t *vt = live_vtcm;
    if (vt == 0) {
        result[1] = 28;
        return;
    }
    uint8_t *input = vt + VT_INPUT;
    uint8_t *input_guard = vt + VT_INPUT_GUARD;
    uint8_t *output = vt + VT_OUTPUT;
    uint8_t *output_guard = vt + VT_OUTPUT_GUARD;
    uint8_t *weights = vt + VT_WEIGHTS;
    uint8_t *fi = vt + VT_FI;
    int32_t *zero_words = (int32_t *)(vt + VT_ZERO);
    volatile uint8_t *done = (volatile uint8_t *)(vt + VT_DONE);
    uint32_t *input_table = (uint32_t *)(arena + TABLES);
    uint32_t *output_table = input_table + 256u;
    uint32_t *input_desc = (uint32_t *)(arena + TABLES + 2048u);
    uint32_t *output_desc = input_desc + 8u;
    uint32_t *cfg = output_desc + 8u;
    uint32_t *mode = cfg + 16u;

    if (!in_vtcm_window(vt, input, IN_BYTES) ||
        !in_vtcm_window(vt, input_guard, 0x5000u) ||
        !in_vtcm_window(vt, output, OUT_BYTES) ||
        !in_vtcm_window(vt, output_guard, 0x2000u) ||
        !in_vtcm_window(vt, weights, W_BYTES) ||
        !in_vtcm_window(vt, fi, FI_BYTES) ||
        !in_vtcm_window(vt, (const uint8_t *)zero_words, 256u) ||
        !in_vtcm_window(vt, (const uint8_t *)done, 2048u) ||
        VT_INPUT + IN_BYTES > VT_INPUT_GUARD ||
        VT_INPUT_GUARD + 0x5000u > VT_OUTPUT ||
        VT_OUTPUT + OUT_BYTES > VT_OUTPUT_GUARD ||
        VT_OUTPUT_GUARD + 0x2000u > VT_WEIGHTS ||
        VT_WEIGHTS + W_BYTES > VT_FI || VT_FI + FI_BYTES > VT_ZERO ||
        VT_ZERO + 256u > VT_TABLES || VT_TABLES + 0x800u > VT_DONE ||
        VT_DONE + 2048u > VTCM_BYTES) {
        result[1] = 28;
        return;
    }

    uint64_t prepare_started = cycles();
    copybytes(input, input_source, IN_BYTES);
    copybytes(weights, weight_source, W_BYTES);
    copybytes(fi, fi_source, FI_BYTES);
    fill128(output, OUT_BYTES, 0xa5a5a5a5u);
    fill128(input_guard, 0x5000u, 0xa5a5a5a5u);
    fill128(output_guard, 0x2000u, 0xa5a5a5a5u);
    for (uint32_t i = 0; i < 64u; ++i) zero_words[i] = i < 32u ? 0x404000 : 0;
    for (uint32_t i = 0; i < 256u; ++i) {
        input_table[i] = (uint32_t)input_guard;
        output_table[i] = (uint32_t)output_guard;
    }
    for (uint32_t i = 0; i < 102u; ++i)
        input_table[i] = (uint32_t)(input + i * 2048u);
    for (uint32_t i = 0; i < 32u; ++i)
        output_table[i] = (uint32_t)(output + i * 2048u);
    for (uint32_t i = 0; i < 102u; ++i) {
        if (!in_vtcm_window(vt, (const uint8_t *)input_table[i], 2048u)) {
            result[1] = 29;
            return;
        }
    }
    for (uint32_t i = 0; i < 32u; ++i) {
        if (!in_vtcm_window(vt, (const uint8_t *)output_table[i], 2048u)) {
            result[1] = 29;
            return;
        }
    }

    result[12] = (uint32_t)(cycles() - prepare_started);
    uint64_t hash_started = cycles();
    result[28] = fnv32(input, IN_BYTES);
    result[29] = fnv32(weights, W_BYTES);
    result[30] = fnv32(fi, FI_BYTES);
    result[13] = (uint32_t)(cycles() - hash_started);
    uint64_t publish_started = cycles();

    for (uintptr_t p = (uintptr_t)input; p < (uintptr_t)input + IN_BYTES; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)output; p < (uintptr_t)output + OUT_BYTES; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)weights; p < (uintptr_t)weights + W_BYTES; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)fi; p < (uintptr_t)fi + FI_BYTES; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)input_guard; p < (uintptr_t)input_guard + 0x5000u; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)output_guard; p < (uintptr_t)output_guard + 0x2000u; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)zero_words; p < (uintptr_t)zero_words + 256u; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)input_table; p < (uintptr_t)input_table + 2048u; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)input_desc; p < (uintptr_t)mode + 8u; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    __asm__ volatile("barrier" ::: "memory");

    input_desc[0] = (uint32_t)input_table;
    input_desc[1] = 1u; input_desc[2] = 34u;
    input_desc[3] = 265u; input_desc[4] = 17u; input_desc[5] = 32u;
    output_desc[0] = (uint32_t)output_table;
    output_desc[1] = 1u; output_desc[2] = 32u;
    output_desc[3] = 256u; output_desc[4] = 8u; output_desc[5] = 32u;
    for (uint32_t i = 0; i < 16u; ++i) cfg[i] = 0u;
    mode[0] = 1u; mode[1] = 0u;
    qd_set_hmx_params_stride1(cfg, 0x700u, 3u, 3u, 32u, 7u, 1u, 12u);
    for (uintptr_t p = (uintptr_t)input_desc; p < (uintptr_t)mode + 8u; p += 32u)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    __asm__ volatile("barrier" ::: "memory");

    const uint32_t expected_cfg[13] = {
        0xe0, 0x700, 0x140, 0x71f, 0x740, 0x71f, 0x17ff,
        0xbff, 0, 0, 0, 0, 0xc
    };
    for (uint32_t i = 0; i < 13u; ++i) {
        if (cfg[i] != expected_cfg[i]) {
            result[1] = 23;
            return;
        }
    }

    *done = 0u;
    __asm__ volatile("dccleaninva(%0)\n barrier" :: "r"((uintptr_t)done) : "memory");
    result[14] = (uint32_t)(cycles() - publish_started);
    uint64_t prepared = cycles() - prepare_started;
    result[6] = (uint32_t)prepared; result[7] = (uint32_t)(prepared >> 32);
    uint64_t leaf_started = cycles();
    __asm__ volatile("mxclracc" ::: "memory");
    qd_qnn_hmx_stride1_generic(output_desc, input_desc, weights, fi, cfg, mode);
    if (drain(zero_words, done)) {
        result[1] = 24;
        result[31] = *(volatile uint32_t *)done;
        return;
    }

    uint64_t leaf_ticks = cycles() - leaf_started;
    result[4] = (uint32_t)leaf_ticks; result[5] = (uint32_t)(leaf_ticks >> 32);
    copy128(arena + OUTPUT_COPY, output, OUT_BYTES);
    copy128(arena + INPUT_ECHO, input, IN_BYTES);
    for (uint32_t i = 0; i < 0x5000u; ++i) {
        if (input_guard[i] != 0xa5u) { result[1] = 25; return; }
    }
    for (uint32_t i = 0; i < 0x2000u; ++i) {
        if (output_guard[i] != 0xa5u) { result[1] = 25; return; }
    }
    for (uint32_t i = 102u; i < 256u; ++i) {
        if (input_table[i] != (uint32_t)input_guard) { result[1] = 26; return; }
    }
    for (uint32_t i = 32u; i < 256u; ++i) {
        if (output_table[i] != (uint32_t)output_guard) { result[1] = 27; return; }
    }
    result[8] = OUT_BYTES;
    result[9] = IN_BYTES;
    result[2] = 3;
}

#ifndef XWB1022_SIM
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0=htid" : "=r"(tid));
    if (tid >= 3u) return;

    volatile uint32_t *command = (volatile uint32_t *)ARENA;
    volatile uint32_t *result = (volatile uint32_t *)(ARENA + 2048u);
    uint32_t sequence = command[2];

    if (tid == 0u) {
        for (uint32_t i = 0; i < 32u; ++i) result[i] = 0u;
        result[0] = MAGIC;
        result[2] = 1u;
        uint64_t deadline = cycles() + 2000000000u;
        while (*(volatile uint32_t *)(ARENA + 144u) != sequence ||
               *(volatile uint32_t *)(ARENA + 160u) != sequence) {
            if (cycles() > deadline) {
                result[1] = 1u;
                goto done;
            }
        }

        if (command[0] != MAGIC || command[1] != 1u || sequence == 0u ||
            command[3] != FIXTURE || command[4] != WORKSPACE_OFFSET ||
            command[5] != 1u) {
            result[1] = 2u;
            goto done;
        }

        volatile uint32_t *clocks = (volatile uint32_t *)(ARENA + QDC_CLOCK_OFFSET);
        QdHmxState *hmx = (QdHmxState *)(ARENA + QDC_HMX_STATE_OFFSET);
        result[1] = clock_raise(clocks);
        if (!result[1]) result[1] = hmx_start(hmx);

        if (!result[1]) {
            if (!tlb(result, 0u)) {
                result[1] = 6u;
            } else {
                uint8_t *vtcm = (uint8_t *)(QDC_VTCM_BASE + WORKSPACE_OFFSET);
                copy128((void *)(ARENA + BACKUP), vtcm, VTCM_BYTES);
                __asm__ volatile("syncht" ::: "memory");
                result[18] = 1u;
                live_vtcm = vtcm;
                qds_on_stack(run_kernel, 0u, 0x30090000u);
            }
        }

        result[10] = hmx_shutdown(hmx);
        if (result[18] && !result[10]) {
            if (!tlb(result, 1u)) {
                result[1] = 6u;
            } else {
                if (result[1] == 24u) {
                    /* Capture diagnostics only after HMX is powered down. */
                    copy128((void *)(ARENA + OUTPUT_COPY),
                            (void *)(QDC_VTCM_BASE + WORKSPACE_OFFSET + VT_OUTPUT), OUT_BYTES);
                    copy128((void *)(ARENA + INPUT_ECHO),
                            (void *)(QDC_VTCM_BASE + WORKSPACE_OFFSET + VT_INPUT), IN_BYTES);
                }
                copy128((void *)(QDC_VTCM_BASE + WORKSPACE_OFFSET),
                        (void *)(ARENA + BACKUP), VTCM_BYTES);
                __asm__ volatile("syncht" ::: "memory");
                result[17] = same128((const uint8_t *)(QDC_VTCM_BASE + WORKSPACE_OFFSET),
                                     (const uint8_t *)(ARENA + BACKUP), VTCM_BYTES);
                if (!result[17]) result[1] = 4u;
            }
        }

        result[11] = clock_restore(clocks);
        if (result[10] || result[11]) result[1] = 5u;
        result[2] = 4u;
    }

done:
    __asm__ volatile("barrier" ::: "memory");
    volatile QdStatus *status =
        (volatile QdStatus *)(ARENA + QD_STATUS_OFFSET) + tid;
    status->error = tid ? 0u : result[1];
    status->rows = 0u;
    status->reserved = QDC_EXITED;
    __asm__ volatile("barrier" ::: "memory");
    status->done_sequence = sequence;
}
#endif
