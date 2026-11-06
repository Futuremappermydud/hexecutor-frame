/* Bounded V75 generic HMX one-hot probe using a recovered CB*2.Xwb graph shape. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

extern void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);
extern void qd_hmx_finish(void *, void *);
extern void qd_set_hmx_params_stride1(uint32_t *, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t);

#define IH 10u
#define IW 258u
#define IMH 24u
#define IMW 272u
#define OH 8u
#define OW 256u
#define CI 32u
#define CO 32u
#define H_BLOCKS (IMH / 8u)
#define W_BLOCKS (IMW / 8u)
#define I_BLOCKS (H_BLOCKS * W_BLOCKS)
#define O_BLOCKS (OH / 8u) * (OW / 8u)
#define PAGE_BYTES 2048u
#define INPUT_BYTES (I_BLOCKS * PAGE_BYTES)
#define OUTPUT_BYTES (O_BLOCKS * PAGE_BYTES)
#define INPUT_ORIGIN_H 7u
#define INPUT_ORIGIN_W 7u

static uint32_t xwb_offset(uint32_t y, uint32_t x, uint32_t ci, uint32_t co) {
    /* tools/export_constants.py Xwb layout: [O/32,I/32,H,W,I/4,O%32,I%4]. */
    uint32_t outer = (((co / 32u * (CI / 32u) + ci / 32u) * 3u + y) * 3u + x);
    return outer * 1024u + ((ci / 4u) % 8u) * 128u + (co % 32u) * 4u + ci % 4u;
}

int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x80000u; i += 4096u)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");
    memset((void *)base, 0, 0x80000);

    /* Bounds: input [0,0x33000), output [0x38000,0x48000), guard [0x48000,0x48800),
     * Xwb [0x4a000,0x4d000), FI [0x4d000,0x4d100), zero [0x4e000,0x4e100),
     * tables [0x4f000,0x4f800), completion [0x4f800,0x4f804). */
    uint8_t *input = (uint8_t *)base;
    uint8_t *output = (uint8_t *)(base + 0x38000u);
    uint8_t *output_guard = (uint8_t *)(base + 0x48000u);
    uint8_t *input_tail = (uint8_t *)(base + 0x33000u);
    uint8_t *weights = (uint8_t *)(base + 0x4a000u);
    int32_t *fi = (int32_t *)(base + 0x4d000u);
    int32_t *zero = (int32_t *)(base + 0x4e000u);
    uint32_t *inblocks = (uint32_t *)(base + 0x4f000u);
    uint32_t *outblocks = (uint32_t *)(base + 0x4f400u);
    volatile uint32_t *done = (volatile uint32_t *)(base + 0x4f800u);
    uint8_t *table_guard = output_guard + 0x800u;

    if (INPUT_BYTES != 208896u || OUTPUT_BYTES != 65536u ||
        (uint32_t)(input + INPUT_BYTES) > (uint32_t)output ||
        (uint32_t)(output + OUTPUT_BYTES) > (uint32_t)output_guard ||
        (uint32_t)(weights + 9u * 1024u) > (uint32_t)fi ||
        (uint32_t)(table_guard + PAGE_BYTES) > base + 0x80000u) {
        puts("FAIL static region bounds");
        return 10;
    }

    memset(output, 0xa5, OUTPUT_BYTES);
    memset(output_guard, 0xa5, PAGE_BYTES);
    for (uint32_t i = 0; i < 256u; ++i) {
        inblocks[i] = (uint32_t)input_tail;
        outblocks[i] = (uint32_t)table_guard;
    }
    for (uint32_t hy = 0; hy < H_BLOCKS; ++hy)
        for (uint32_t wx = 0; wx < W_BLOCKS; ++wx) {
            uint32_t b = hy * W_BLOCKS + wx;
            inblocks[b] = (uint32_t)(input + b * PAGE_BYTES);
        }
    for (uint32_t b = 0; b < O_BLOCKS; ++b)
        outblocks[b] = (uint32_t)(output + b * PAGE_BYTES);

    /* Unique values for the first 10x10 logical window, shifted into the tensor's
     * 7x7 padding origin. All remaining physical activation bytes stay zero. */
    for (uint32_t y = 0; y < 10u; ++y)
        for (uint32_t x = 0; x < 10u; ++x)
            for (uint32_t c = 0; c < CI; ++c) {
                uint32_t ay = y + INPUT_ORIGIN_H, ax = x + INPUT_ORIGIN_W;
                uint32_t b = (ay / 8u) * W_BLOCKS + ax / 8u;
                uint32_t off = (ay % 8u) * 256u + (ax % 8u) * CI + c;
                input[(uint32_t)(inblocks[b] - base) + off] = (uint8_t)(1u + y * 10u + x);
            }

    memset(weights, 0, 9u * 1024u);
    memset(fi, 0, 64u * sizeof(*fi));
    fi[0] = 0x406000;
    zero[0] = 0x404000;
    uint32_t input_desc[] = {(uint32_t)inblocks, 1u, W_BLOCKS, IW, IH, CI};
    uint32_t output_desc[] = {(uint32_t)outblocks, 1u, OW / 8u, OW, OH, CO};
    uint32_t cfg[16] = {0};
    uint32_t mode[] = {1u, 0u};
    qd_set_hmx_params_stride1(cfg, 0x700u, 3u, H_BLOCKS, CI, 7u, 1u, 12u);
    printf("graphshape input=%ux%u max=%ux%u origin=%u,%u blocks=%ux%u rowstride=%u; output=%ux%u blocks=%u; cfg=",
           IW, IH, IMW, IMH, INPUT_ORIGIN_W, INPUT_ORIGIN_H, W_BLOCKS, H_BLOCKS, W_BLOCKS,
           OW, OH, O_BLOCKS);
    for (uint32_t i = 0; i < 13u; ++i) printf("%s%x", i ? "," : "", cfg[i]);
    puts("");

    for (uint32_t tap = 0; tap < 9u; ++tap) {
        uint32_t ky = tap / 3u, kx = tap % 3u;
        memset(weights, 0, 9u * 1024u);
        weights[xwb_offset(ky, kx, 0u, 0u)] = 1;
        memset(output, 0xa5, OUTPUT_BYTES);
        memset(output_guard, 0xa5, PAGE_BYTES);
        *done = 0xffffffffu;
        __asm__ volatile("mxclracc" ::: "memory");
        qd_qnn_hmx_stride1_generic(output_desc, input_desc, weights, fi, cfg, mode);
        qd_hmx_finish(zero, (void *)done);
        uint32_t spins = 0;
        while (*done && spins++ < 100001u) __asm__ volatile("nop");
        uint32_t got = output[0];
        uint32_t want = tap + 1u;
        printf("tap=%u Xwb-byte=%u got00=%u want00=%u spins=%u\n",
               tap, xwb_offset(ky, kx, 0u, 0u), got, want, spins);
        if (*done) { puts("FAIL completion"); return 1; }
        if (got != want) printf("MISMATCH tap=%u got=%u want=%u\n", tap, got, want);
        for (uint32_t i = 0; i < PAGE_BYTES; ++i)
            if (output_guard[i] != 0xa5u) { printf("FAIL output guard byte=%u\n", i); return 2; }
        for (uint32_t i = O_BLOCKS; i < 256u; ++i)
            if (outblocks[i] != (uint32_t)table_guard) { printf("FAIL output table tail=%u\n", i); return 3; }
        for (uint32_t i = I_BLOCKS; i < 256u; ++i)
            if (inblocks[i] != (uint32_t)input_tail) { printf("FAIL input table tail=%u\n", i); return 4; }
    }
    puts("Completed all Xwb one-hot taps with output/table guards intact; mismatches are mapping evidence, not support proof.");
    return 0;
}
