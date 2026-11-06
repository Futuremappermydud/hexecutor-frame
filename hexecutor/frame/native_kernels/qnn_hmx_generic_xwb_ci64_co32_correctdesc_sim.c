/* Generic V75 CI64/CO32 Xwb depth-block and spatial-tap proof. */
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
#define IW 130u
#define OH 8u
#define OW 128u
#define CI 64u
#define CO 32u
#define WB 18u
#define DB 2u
#define ROWSTRIDE (WB * DB)
#define IB (3u * ROWSTRIDE)
#define OB (OW / 8u)
#define INBYTES (IB * PAGE)
#define OUTBYTES (OB * PAGE)

static uint32_t xwb_offset(uint32_t ky, uint32_t kx, uint32_t ci, uint32_t co) {
    uint32_t depth_block = ci / 32u;
    uint32_t lane = ((ci / 4u) % 8u) * 128u + (co % 32u) * 4u + ci % 4u;
    return ((depth_block * 9u + ky * 3u + kx) * 1024u) + lane;
}

static uint8_t get_input(const uint8_t *in, uint32_t y, uint32_t x, uint32_t ci) {
    uint32_t py = y + 7u, px = x + 7u;
    uint32_t block = ((py / 8u) * WB + px / 8u) * DB + ci / 32u;
    uint32_t offset = (py % 8u) * 256u + (px % 8u) * 32u + ci % 32u;
    return in[block * PAGE + offset];
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

    uint8_t *in = (uint8_t *)base;
    uint8_t *input_guard = (uint8_t *)(base + INBYTES);
    uint8_t *out = (uint8_t *)(base + 0x39000u);
    uint8_t *output_guard = (uint8_t *)(base + 0x41000u);
    uint8_t *weights = (uint8_t *)(base + 0x43000u);
    int32_t *fi = (int32_t *)(base + 0x48000u);
    int32_t *zero_fi = (int32_t *)(base + 0x49000u);
    uint32_t *in_table = (uint32_t *)(base + 0x4a000u);
    uint32_t *out_table = (uint32_t *)(base + 0x4a400u);
    volatile uint32_t *done = (volatile uint32_t *)(base + 0x4a800u);
    uint8_t *table_guard = output_guard + PAGE;

    if (INBYTES != 221184u || OUTBYTES != 32768u ||
        9u * CI * CO != 18432u ||
        (uint32_t)(in + INBYTES) > (uint32_t)out ||
        (uint32_t)(out + OUTBYTES) > (uint32_t)output_guard ||
        (uint32_t)(weights + 9u * CI * CO) > (uint32_t)fi ||
        (uint32_t)(fi + 64) > (uint32_t)zero_fi ||
        (uint32_t)(zero_fi + 64) > (uint32_t)in_table) {
        puts("FAIL region bounds");
        return 10;
    }

    memset(input_guard, 0xa5, 0x3000u);
    memset(output_guard, 0xa5, PAGE);
    for (uint32_t i = 0; i < 256u; ++i) {
        in_table[i] = (uint32_t)input_guard;
        out_table[i] = (uint32_t)table_guard;
    }
    for (uint32_t b = 0; b < IB; ++b)
        in_table[b] = (uint32_t)(in + b * PAGE);
    for (uint32_t b = 0; b < OB; ++b)
        out_table[b] = (uint32_t)(out + b * PAGE);

    for (uint32_t y = 0; y < IH; ++y)
        for (uint32_t x = 0; x < IW; ++x)
            for (uint32_t ci = 0; ci < CI; ++ci) {
                uint8_t value = (uint8_t)(1u + ((y * 29u + x * 13u + ci * 7u) % 250u));
                uint32_t py = y + 7u, px = x + 7u;
                uint32_t block = ((py / 8u) * WB + px / 8u) * DB + ci / 32u;
                uint32_t offset = (py % 8u) * 256u + (px % 8u) * 32u + ci % 32u;
                in[block * PAGE + offset] = value;
            }

    for (uint32_t c = 0; c < CO; ++c) {
        fi[c] = 0x406000;
        fi[CO + c] = 0;
    }
    for (uint32_t c = 0; c < 64u; ++c)
        zero_fi[c] = c < 32u ? 0x404000 : 0;

    uint32_t in_desc[] = {(uint32_t)in_table, DB, ROWSTRIDE,
                          IW + 7u, IH + 7u, CI};
    uint32_t out_desc[] = {(uint32_t)out_table, 1u, OW / 8u,
                           OW, OH, CO};
    uint32_t cfg[16] = {0}, mode[] = {1u, 0u};
    qd_set_hmx_params_stride1(cfg, 0x700u, 3u, 3u, CI, 7u, 1u, 12u);
    printf("CI64/CO32 Xwb: input depth-blocks=%u, rowstride=%u; output depth-blocks=1; cfg=",
           DB, ROWSTRIDE);
    for (uint32_t i = 0; i < 13u; ++i)
        printf("%s%x", i ? "," : "", cfg[i]);
    putchar('\n');

    uint32_t total_bad = 0;
    for (uint32_t ky = 0; ky < 3u; ++ky)
        for (uint32_t kx = 0; kx < 3u; ++kx) {
            memset(weights, 0, 9u * CI * CO);
            for (uint32_t co = 0; co < CO; ++co)
                weights[xwb_offset(ky, kx, co + 32u, co)] = 1;
            memset(out, 0xa5, OUTBYTES);
            *done = 0xffffffffu;
            __asm__ volatile("mxclracc" ::: "memory");
            qd_qnn_hmx_stride1_generic(out_desc, in_desc, weights, fi, cfg, mode);
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
                        uint32_t block = (y / 8u) * (OW / 8u) + x / 8u;
                        uint32_t offset = (y % 8u) * 256u + (x % 8u) * CO + co;
                        uint8_t got = out[block * PAGE + offset];
                        uint8_t want = get_input(in, y + ky, x + 2u - kx, co + 32u);
                        if (got != want) {
                            if (bad < 2u)
                                printf("MISMATCH tap=%u,%u y=%u x=%u co=%u got=%u want=%u\n",
                                       ky, kx, y, x, co, got, want);
                            ++bad;
                        }
                    }
            printf("tap=%u,%u mismatches=%u/%u completion_spins=%u\n",
                   ky, kx, bad, OH * OW * CO, spins);
            total_bad += bad;
        }

    for (uint32_t i = 0; i < 0x3000u; ++i)
        if (input_guard[i] != 0xa5u) { printf("FAIL input guard %u\n", i); return 2; }
    for (uint32_t i = 0; i < PAGE; ++i)
        if (output_guard[i] != 0xa5u) { printf("FAIL output guard %u\n", i); return 3; }
    for (uint32_t i = IB; i < 256u; ++i)
        if (in_table[i] != (uint32_t)input_guard) { printf("FAIL input table tail %u\n", i); return 4; }
    for (uint32_t i = OB; i < 256u; ++i)
        if (out_table[i] != (uint32_t)table_guard) { printf("FAIL output table tail %u\n", i); return 5; }
    if (total_bad) {
        printf("FAIL total mismatches=%u/%u\n", total_bad, 9u * OH * OW * CO);
        return 6;
    }
    puts("PASS CI64/CO32: raw Xwb off-diagonal depth tile, all 9 taps, full logical outputs, guards, and table tails.");
    return 0;
}
