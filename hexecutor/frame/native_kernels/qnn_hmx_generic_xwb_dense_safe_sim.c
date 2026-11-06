/* Generic V75 spatial leaf with CPU-prefix Xwb/FI/input snapshot for QDRX 1022. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "xwb1022_actual_data.h"

extern void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);
extern void qd_hmx_finish(void *, void *);
extern void qd_set_hmx_params_stride1(uint32_t *, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t);

#define PAGE_BYTES 2048u
#define INPUT_BLOCKS 102u
#define OUTPUT_BLOCKS 32u
#define INPUT_BYTES (INPUT_BLOCKS * PAGE_BYTES)
#define OUTPUT_BYTES (OUTPUT_BLOCKS * PAGE_BYTES)
#define IH 10u
#define IW 258u
#define OH 8u
#define OW 256u
#define CI 32u
#define CO 32u
#define W_BLOCKS 34u
#define H_BLOCKS 3u

int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x80000u; i += 4096u)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");
    memset((void *)base, 0, 0x80000u);

    uint8_t *input = (uint8_t *)base;
    uint8_t *input_guard = (uint8_t *)(base + INPUT_BYTES);
    uint8_t *output = (uint8_t *)(base + 0x38000u);
    uint8_t *output_guard = (uint8_t *)(base + 0x48000u);
    uint8_t *weights = (uint8_t *)(base + 0x4a000u);
    int32_t *fi = (int32_t *)(base + 0x4d000u);
    int32_t *zero = (int32_t *)(base + 0x4e000u);
    uint32_t *inblocks = (uint32_t *)(base + 0x4f000u);
    uint32_t *outblocks = (uint32_t *)(base + 0x4f400u);
    volatile uint32_t *done = (volatile uint32_t *)(base + 0x4f800u);
    uint8_t *table_guard = output_guard + PAGE_BYTES;
    if (sizeof(dense_safe_oracle) != 65536u || sizeof(actual_weights) != 9216u ||
        sizeof(actual_fi) != 64u * sizeof(int32_t) || sizeof(oracle) != 65536u ||
        (uint32_t)(output + OUTPUT_BYTES) > (uint32_t)output_guard ||
        (uint32_t)(table_guard + PAGE_BYTES) > base + 0x80000u) {
        puts("FAIL static bounds/fixture sizes");
        return 10;
    }
    memset(input, 0, INPUT_BYTES);
    for (uint32_t y = 0; y < IH; ++y)
        for (uint32_t x = 0; x < IW; ++x)
            for (uint32_t c = 0; c < CI; ++c) {
                uint32_t py = y + 7u, px = x + 7u;
                uint32_t b = (py / 8u) * W_BLOCKS + px / 8u;
                uint32_t off = (py % 8u) * 256u + (px % 8u) * CI + c;
                input[b * PAGE_BYTES + off] = 1u;
            }
    memset(input_guard, 0xa5, 0x5000u);
    memset(output, 0xa5, OUTPUT_BYTES);
    memset(output_guard, 0xa5, PAGE_BYTES);
    for (uint32_t i = 0; i < 256u; ++i) {
        inblocks[i] = (uint32_t)input_guard;
        outblocks[i] = (uint32_t)table_guard;
    }
    for (uint32_t b = 0; b < INPUT_BLOCKS; ++b)
        inblocks[b] = (uint32_t)(input + b * PAGE_BYTES);
    for (uint32_t b = 0; b < OUTPUT_BLOCKS; ++b)
        outblocks[b] = (uint32_t)(output + b * PAGE_BYTES);
    memcpy(weights, actual_weights, sizeof(actual_weights));
    for (uint32_t c = 0; c < 32u; ++c) { fi[c] = 0x403c00; fi[32u + c] = 65536; }
    zero[0] = 0x404000;

    uint32_t input_desc[] = {(uint32_t)inblocks, 1u, W_BLOCKS,
                             IW + 7u, IH + 7u, CI};
    uint32_t output_desc[] = {(uint32_t)outblocks, 1u, OW / 8u, OW, OH, CO};
    uint32_t cfg[16] = {0};
    uint32_t mode[] = {1u, 0u};
    qd_set_hmx_params_stride1(cfg, 0x700u, 3u, H_BLOCKS, CI, 7u, 1u, 12u);
    printf("qdrx sequence=1022 command_index=836 Xwb/FI actual CPU prefix; input=%ux%u max=272x24 origin=7,7 blocks=34x3; output=%ux%u; cfg=",
           IW, IH, OW, OH);
    for (uint32_t i = 0; i < 13u; ++i) printf("%s%x", i ? "," : "", cfg[i]);
    puts("");

    *done = 0xffffffffu;
    __asm__ volatile("mxclracc" ::: "memory");
    qd_qnn_hmx_stride1_generic(output_desc, input_desc, weights, fi, cfg, mode);
    qd_hmx_finish(zero, (void *)done);
    uint32_t spins = 0;
    while (*done && spins++ < 100001u) __asm__ volatile("nop");
    if (*done) { puts("FAIL completion timeout"); return 1; }
    printf("completion spins=%u\n", spins);

    uint32_t mismatch = 0, plus1 = 0, minus1 = 0, other = 0;
    for (uint32_t y = 0; y < OH; ++y)
        for (uint32_t x = 0; x < OW; ++x)
            for (uint32_t c = 0; c < CO; ++c) {
                uint32_t b = (y / 8u) * (OW / 8u) + x / 8u;
                uint32_t off = (y % 8u) * 256u + (x % 8u) * CO + c;
                uint32_t idx = (y * OW + x) * CO + c;
                uint32_t got = output[b * PAGE_BYTES + off];
                int delta = (int)got - (int)dense_safe_oracle[idx];
                if (delta) {
                    ++mismatch;
                    if (delta == 1) ++plus1;
                    else if (delta == -1) ++minus1;
                    else ++other;
                    if (mismatch == 1u)
                        printf("FIRST_MISMATCH y=%u x=%u co=%u got=%u ref=%u delta=%d\n", y, x, c, got, dense_safe_oracle[idx], delta);
                }
            }
    printf("dense_safe_compare mismatches=%u/%u plus1=%u minus1=%u other=%u\n",
           mismatch, OUTPUT_BYTES, plus1, minus1, other);
    for (uint32_t i = 0; i < 0x5000u; ++i)
        if (input_guard[i] != 0xa5u) { printf("FAIL input guard byte=%u\n", i); return 2; }
    for (uint32_t i = 0; i < PAGE_BYTES; ++i)
        if (output_guard[i] != 0xa5u) { printf("FAIL output guard byte=%u\n", i); return 3; }
    for (uint32_t i = INPUT_BLOCKS; i < 256u; ++i)
        if (inblocks[i] != (uint32_t)input_guard) { printf("FAIL input table tail=%u\n", i); return 4; }
    for (uint32_t i = OUTPUT_BLOCKS; i < 256u; ++i)
        if (outblocks[i] != (uint32_t)table_guard) { printf("FAIL output table tail=%u\n", i); return 5; }
    if (minus1 || other) {
        printf("FAIL numerical deltas: minus1=%u other=%u\n", minus1, other);
        return 6;
    }
    printf("PASS numerical output is exact or +1; completion and all guards/table tails intact.\n");
    return 0;
}
