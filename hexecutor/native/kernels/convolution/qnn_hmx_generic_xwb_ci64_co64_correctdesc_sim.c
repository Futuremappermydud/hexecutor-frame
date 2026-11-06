/* Generic V75 CI64/CO64 Xwb descriptor proof with exact physical-layout oracle. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

extern void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *,
                                       uint32_t *, uint32_t *);
extern void qd_hmx_finish(void *, void *);
extern void qd_set_hmx_params_stride1(uint32_t *, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t);

#define PAGE 2048u
#define IH 10u
#define IW 10u
#define OH 8u
#define OW 8u
#define CI 64u
#define CO 64u
#define WB 3u
#define DB 2u
#define ROWSTRIDE (WB * DB)
#define IB (3u * ROWSTRIDE)
#define OB (2u * (OW / 8u))
#define INBYTES (IB * PAGE)
#define OUTBYTES (OB * PAGE)

static uint32_t xwb_offset(uint32_t ky, uint32_t kx, uint32_t ci, uint32_t co) {
    uint32_t outer = ((co / 32u) * (CI / 32u) + ci / 32u) * 9u + ky * 3u + kx;
    uint32_t lane = ((ci / 4u) % 8u) * 128u + (co % 32u) * 4u + ci % 4u;
    return outer * 1024u + lane;
}

static uint8_t get_input(const uint8_t *input, uint32_t y, uint32_t x, uint32_t ci) {
    uint32_t py = y + 7u, px = x + 7u;
    uint32_t block = ((py / 8u) * WB + px / 8u) * DB + ci / 32u;
    uint32_t offset = (py % 8u) * 256u + (px % 8u) * 32u + ci % 32u;
    return input[block * PAGE + offset];
}

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
    uint8_t *input_guard = (uint8_t *)(base + INBYTES);
    uint8_t *output = (uint8_t *)(base + 0x10000u);
    uint8_t *output_guard = (uint8_t *)(base + 0x12000u);
    uint8_t *weights = (uint8_t *)(base + 0x14000u);
    int32_t *fi = (int32_t *)(base + 0x1d000u);
    int32_t *zero_fi = (int32_t *)(base + 0x1e000u);
    uint32_t *input_table = (uint32_t *)(base + 0x1f000u);
    uint32_t *output_table = (uint32_t *)(base + 0x1f400u);
    volatile uint32_t *done = (volatile uint32_t *)(base + 0x1f800u);
    uint8_t *table_guard = output_guard + PAGE;

    if (IB != 18u || INBYTES != 36864u || OB != 2u || OUTBYTES != 4096u ||
        9u * CI * CO != 36864u ||
        (uint32_t)(input + INBYTES) > (uint32_t)output ||
        (uint32_t)(output + OUTBYTES) > (uint32_t)output_guard ||
        (uint32_t)(weights + 9u * CI * CO) > (uint32_t)fi ||
        (uint32_t)(fi + 128) > (uint32_t)zero_fi ||
        (uint32_t)(zero_fi + 64) > (uint32_t)input_table) {
        puts("FAIL region bounds");
        return 10;
    }

    memset(input_guard, 0xa5, 0x1000u);
    memset(output_guard, 0xa5, PAGE);
    for (uint32_t i = 0; i < 256u; ++i) {
        input_table[i] = (uint32_t)input_guard;
        output_table[i] = (uint32_t)table_guard;
    }
    for (uint32_t b = 0; b < IB; ++b)
        input_table[b] = (uint32_t)(input + b * PAGE);
    for (uint32_t b = 0; b < OB; ++b)
        output_table[b] = (uint32_t)(output + b * PAGE);

    for (uint32_t y = 0; y < IH; ++y)
        for (uint32_t x = 0; x < IW; ++x)
            for (uint32_t ci = 0; ci < CI; ++ci) {
                uint8_t value = (uint8_t)(1u + ((y * 29u + x * 13u + ci * 7u) % 250u));
                uint32_t py = y + 7u, px = x + 7u;
                uint32_t block = ((py / 8u) * WB + px / 8u) * DB + ci / 32u;
                uint32_t offset = (py % 8u) * 256u + (px % 8u) * 32u + ci % 32u;
                input[block * PAGE + offset] = value;
            }

    for (uint32_t block = 0; block < CO / 32u; ++block)
        for (uint32_t c = 0; c < 32u; ++c) {
            fi[block * 64u + c] = 0x406000;
            fi[block * 64u + 32u + c] = 0;
        }
    for (uint32_t c = 0; c < 64u; ++c)
        zero_fi[c] = c < 32u ? 0x404000 : 0;

    uint32_t input_desc[] = {(uint32_t)input_table, DB, ROWSTRIDE,
                             IW + 7u, IH + 7u, CI};
    uint32_t output_desc[] = {(uint32_t)output_table, DB, 2u,
                              OW, OH, CO};
    uint32_t cfg[16] = {0}, mode[] = {1u, 0u};
    qd_set_hmx_params_stride1(cfg, 0x700u, 3u, 3u, CI, 7u, 1u, 12u);
    printf("CI64/CO64 input depth-blocks=%u rowstride=%u; output depth-blocks=%u cfg=",
           DB, ROWSTRIDE, DB);
    for (uint32_t i = 0; i < 13u; ++i)
        printf("%s%x", i ? "," : "", cfg[i]);
    putchar('\n');

    uint32_t total_bad = 0;
    for (uint32_t ky = 0; ky < 3u; ++ky)
        for (uint32_t kx = 0; kx < 3u; ++kx) {
            memset(weights, 0, 9u * CI * CO);
            for (uint32_t ci = 0; ci < CI; ++ci) {
                uint32_t co = (ci + 17u) % CO;
                weights[xwb_offset(ky, kx, ci, co)] = 1;
            }
            memset(output, 0xa5, OUTBYTES);
            *done = 0xffffffffu;
            __asm__ volatile("mxclracc" ::: "memory");
            qd_qnn_hmx_stride1_generic(output_desc, input_desc, weights, fi, cfg, mode);
            qd_hmx_finish(zero_fi, (void *)done);
            uint32_t spins = 0;
            while (*done && spins++ < 100001u)
                __asm__ volatile("nop");
            if (*done) {
                printf("FAIL completion tap=%u,%u\n", ky, kx);
                return 1;
            }

            uint32_t bad = 0;
            for (uint32_t y = 0; y < OH; ++y)
                for (uint32_t x = 0; x < OW; ++x)
                    for (uint32_t co = 0; co < CO; ++co) {
                        uint32_t block = (y / 8u) * 2u + co / 32u;
                        uint32_t offset = (y % 8u) * 256u + (x % 8u) * 32u + co % 32u;
                        uint8_t got = output[block * PAGE + offset];
                        uint32_t ci = (co + CO - 17u) % CO;
                        uint8_t want = get_input(input, y + ky, x + 2u - kx, ci);
                        if (got != want) {
                            if (bad < 4u)
                                printf("MISMATCH tap=%u,%u y=%u x=%u co=%u got=%u want=%u\n",
                                       ky, kx, y, x, co, got, want);
                            ++bad;
                        }
                    }
            printf("tap=%u,%u mismatches=%u/%u completion_spins=%u\n",
                   ky, kx, bad, OH * OW * CO, spins);
            total_bad += bad;
        }
    for (uint32_t i = 0; i < 0x1000u; ++i)
        if (input_guard[i] != 0xa5u) { printf("FAIL input guard %u\n", i); return 2; }
    for (uint32_t i = 0; i < PAGE; ++i)
        if (output_guard[i] != 0xa5u) { printf("FAIL output guard %u\n", i); return 3; }
    for (uint32_t i = IB; i < 256u; ++i)
        if (input_table[i] != (uint32_t)input_guard) { printf("FAIL input table tail %u\n", i); return 4; }
    for (uint32_t i = OB; i < 256u; ++i)
        if (output_table[i] != (uint32_t)table_guard) { printf("FAIL output table tail %u\n", i); return 5; }
    if (total_bad) {
        printf("FAIL CI64/CO64 total mismatches=%u/%u\n", total_bad, 9u * OH * OW * CO);
        return 6;
    }
    puts("PASS CI64/CO64 cross-depth Xwb permutation, all 9 taps, full outputs, guards, and table tails.");
    return 0;
}
