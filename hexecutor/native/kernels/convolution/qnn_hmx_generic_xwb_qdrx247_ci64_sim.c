/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Real CPU-prefix Xwb/FI fixture for QDRX sequence 247 (CI64/CO64). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

extern void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);
extern void qd_hmx_finish(void *, void *);
extern void qd_set_hmx_params_stride1(uint32_t *, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t);

#define PAGE 2048u
#define FIXTURE "/src/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin"
#define MAGIC 0x324b5748u
#define MAX_FIXTURE (600u * 1024u)
#define INPUT_BLOCKS 108u
#define OUTPUT_BLOCKS 32u
#define OUTPUT_TABLE_SLICE 100u
#define INPUT_BYTES (INPUT_BLOCKS * PAGE)
#define OUTPUT_BYTES (OUTPUT_BLOCKS * PAGE)
#define WEIGHT_BYTES 36864u
#define FI_BYTES 512u

static uint8_t payload[MAX_FIXTURE] __attribute__((aligned(128)));
static uint8_t output_ref[OUTPUT_BYTES] __attribute__((aligned(128)));
static uint32_t in_table[256] __attribute__((aligned(128)));
static uint32_t out_table[256] __attribute__((aligned(128)));

static uint64_t fnv64(const uint8_t *p, uint32_t n) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (uint32_t i = 0; i < n; ++i) h = (h ^ p[i]) * UINT64_C(0x100000001b3);
    return h;
}

int main(void) {
    FILE *f = fopen(FIXTURE, "rb");
    if (!f) { puts("FAIL fixture open"); return 1; }
    uint32_t h[32];
    if (fread(h, sizeof h, 1, f) != 1 || h[0] != MAGIC || h[1] != 247u) {
        puts("FAIL fixture header"); fclose(f); return 2;
    }
    uint32_t xbytes = h[7] * PAGE, ybytes = h[13] * PAGE;
    uint32_t wbytes = h[17], fibytes = h[18];
    uint64_t total = (uint64_t)xbytes + ybytes + wbytes + fibytes;
    if (h[7] != INPUT_BLOCKS || h[13] != 136u || h[12] != OUTPUT_TABLE_SLICE ||
        h[14] != 64u || h[15] != 64u || h[16] != 2u || wbytes != WEIGHT_BYTES ||
        fibytes != FI_BYTES || total > sizeof payload || h[6] != 0u ||
        h[12] + OUTPUT_BLOCKS > h[13]) {
        puts("FAIL unsupported fixture geometry"); fclose(f); return 3;
    }
    if (fread(payload, (size_t)total, 1, f) != 1 || fgetc(f) != EOF) {
        puts("FAIL fixture payload length"); fclose(f); return 4;
    }
    fclose(f);
    const uint8_t *xdata = payload;
    const uint8_t *ydata = xdata + xbytes;
    const uint8_t *weights_data = ydata + ybytes;
    const uint8_t *fi_data = weights_data + wbytes;
    memcpy(output_ref, ydata + OUTPUT_TABLE_SLICE * PAGE, OUTPUT_BYTES);

    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x80000u; i += 4096u)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");
    memset((void *)base, 0xa5, 0x80000u);

    uint8_t *input = (uint8_t *)base;
    uint8_t *input_guard = (uint8_t *)(base + 0x36000u);
    uint8_t *output = (uint8_t *)(base + 0x38000u);
    uint8_t *output_guard = (uint8_t *)(base + 0x48000u);
    uint8_t *weights = (uint8_t *)(base + 0x4a000u);
    int32_t *fi = (int32_t *)(base + 0x54000u);
    int32_t *zero_fi = (int32_t *)(base + 0x55000u);
    volatile uint32_t *done = (volatile uint32_t *)(base + 0x56000u);
    uint8_t *table_guard = output_guard + PAGE;
    if (input + INPUT_BYTES > input_guard || output + OUTPUT_BYTES > output_guard ||
        weights + WEIGHT_BYTES > (uint8_t *)fi || (uint8_t *)(fi + 128) > (uint8_t *)zero_fi ||
        (uint8_t *)(zero_fi + 64) > (uint8_t *)done || table_guard + PAGE > (uint8_t *)(base + 0x80000u)) {
        puts("FAIL VTCM region bounds"); return 5;
    }
    memcpy(input, xdata, INPUT_BYTES);
    memcpy(weights, weights_data, WEIGHT_BYTES);
    memcpy(fi, fi_data, FI_BYTES);
    memset(input_guard, 0xa5, 0x2000u);
    memset(output, 0xa5, OUTPUT_BYTES);
    memset(output_guard, 0xa5, PAGE);
    memset(zero_fi, 0, 64u * sizeof(uint32_t));
    for (uint32_t c = 0; c < 32u; ++c) zero_fi[c] = 0x404000;
    for (uint32_t i = 0; i < 256u; ++i) {
        in_table[i] = (uint32_t)input_guard;
        out_table[i] = (uint32_t)table_guard;
    }
    for (uint32_t b = 0; b < INPUT_BLOCKS; ++b) in_table[b] = (uint32_t)(input + b * PAGE);
    for (uint32_t b = 0; b < OUTPUT_BLOCKS; ++b) out_table[b] = (uint32_t)(output + b * PAGE);
    for (uintptr_t p = (uintptr_t)input; p < (uintptr_t)input + INPUT_BYTES; p += 32)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)weights; p < (uintptr_t)weights + WEIGHT_BYTES; p += 32)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)fi; p < (uintptr_t)fi + FI_BYTES; p += 32)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    __asm__ volatile("barrier" ::: "memory");

    uint32_t input_desc[] = {(uint32_t)in_table, 2u, 36u, 137u, 17u, 64u};
    uint32_t output_desc[] = {(uint32_t)out_table, 2u, 32u,
                              128u, 8u, 64u};
    uint32_t cfg[16] = {0}, mode[] = {1u, 0u};
    qd_set_hmx_params_stride1(cfg, 0x700u, 3u, 3u, 64u, 7u, 1u, 12u);
    printf("QDRX247 input=%ux%u max=%ux%u slice=%u blocks=%u output=%ux%u max=%ux%u slice=%u active=%u cfg=",
           h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11], h[12], OUTPUT_BLOCKS);
    for (uint32_t i = 0; i < 13u; ++i) printf("%s%x", i ? "," : "", cfg[i]);
    printf("\nstaged_fnv64 X=%016llx W=%016llx FI=%016llx Yref=%016llx\n",
           (unsigned long long)fnv64(input, INPUT_BYTES),
           (unsigned long long)fnv64(weights, WEIGHT_BYTES),
           (unsigned long long)fnv64((const uint8_t *)fi, FI_BYTES),
           (unsigned long long)fnv64(output_ref, OUTPUT_BYTES));

    *done = 0xffffffffu;
    __asm__ volatile("mxclracc" ::: "memory");
    qd_qnn_hmx_stride1_generic(output_desc, input_desc, weights, fi, cfg, mode);
    qd_hmx_finish(zero_fi, (void *)done);
    uint32_t spins = 0;
    while (*done && spins++ < 100001u) __asm__ volatile("nop");
    if (*done) { puts("FAIL completion"); return 6; }

    uint32_t bad = 0, plus1 = 0, minus1 = 0, other = 0;
    for (uint32_t b = 0; b < OUTPUT_BLOCKS; ++b)
        for (uint32_t i = 0; i < PAGE; ++i) {
            int delta = (int)output[b * PAGE + i] - (int)output_ref[b * PAGE + i];
            if (delta) {
                ++bad; if (delta == 1) ++plus1; else if (delta == -1) ++minus1; else ++other;
                if (bad == 1u) printf("FIRST_DIFF page=%u byte=%u got=%u ref=%u delta=%d\n",
                                      b, i, output[b * PAGE + i], output_ref[b * PAGE + i], delta);
            }
        }
    printf("CPU physical output compare mismatches=%u/%u +1=%u -1=%u other=%u spins=%u\n",
           bad, OUTPUT_BYTES, plus1, minus1, other, spins);
    for (uint32_t i = 0; i < 0x2000u; ++i)
        if (input_guard[i] != 0xa5u) { printf("FAIL input guard=%u\n", i); return 7; }
    for (uint32_t i = 0; i < PAGE; ++i)
        if (output_guard[i] != 0xa5u) { printf("FAIL output guard=%u\n", i); return 8; }
    for (uint32_t i = INPUT_BLOCKS; i < 256u; ++i)
        if (in_table[i] != (uint32_t)input_guard) { printf("FAIL input table tail=%u\n", i); return 9; }
    for (uint32_t i = OUTPUT_BLOCKS; i < 256u; ++i)
        if (out_table[i] != (uint32_t)table_guard) { printf("FAIL output table tail=%u\n", i); return 10; }
    puts("RAW_OUTPUT_BEGIN");
    for (uint32_t i = 0; i < OUTPUT_BYTES; ++i) {
        printf("%02x", output[i]);
        if ((i & 255u) == 255u) putchar('\n');
    }
    puts("RAW_OUTPUT_END");
    if (minus1 || other) { puts("FAIL numeric difference outside 0/+1"); return 11; }
    puts("PASS QDRX247 CI64/CO64 physical outputs are CPU-equal or +1; guards and table tails intact.");
    return 0;
}
