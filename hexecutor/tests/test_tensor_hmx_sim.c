/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Regression: K=1152 in the hand landmark model exceeds one deep load. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void hx_hmx_init(void *);
extern void hx_hmx_gemm(void *, void *, void *, uint32_t, uint32_t, uint32_t);
extern void hx_hmx_gemm_bias(void *, void *, void *, uint32_t, uint32_t, void *);
extern void hx_hmx_finish(void *);
int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x100000u; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    uint16_t *a = (uint16_t *)base, *b = (uint16_t *)(base + 0x40000u),
             *c = (uint16_t *)(base + 0x80000u);
    uint32_t *scale = (uint32_t *)(base + 0x81000u);
    void *done = (void *)(base + 0x82000u);
    for (uint32_t i = 0; i < 32; i++) {
        scale[i] = 0x3c00u;
        scale[32 + i] = 0;
    }
    const uint32_t ks[] = {32, 1024, 1056, 1152, 4096};
    for (uint32_t test = 0; test < sizeof(ks) / sizeof(ks[0]); test++) {
        uint32_t k = ks[test], tiles = k / 32u;
        for (uint32_t kb = 0; kb < tiles; kb++)
            for (uint32_t i = 0; i < 1024; i++) {
                a[kb * 1024u + i] = (kb == 0 || kb == tiles - 1) ? 0x3c00u : 0;
                b[kb * 1024u + i] = 0x3c00u;
            }
        memset(c, 0xa5, 2048);
        hx_hmx_init(scale);
        hx_hmx_gemm(a, b, c, tiles * 2048u, 1, 1);
        hx_hmx_finish(done);
        uint16_t expected = k == 32 ? 0x5000u : 0x5400u;
        for (uint32_t i = 0; i < 1024; i++)
            if (c[i] != expected) {
                printf("FAIL K=%u lane=%u got=%04x expected=%04x\n", (unsigned)k, (unsigned)i, c[i],
                       expected);
                return 1;
            }
    }
    /* Per-column packed FP16 scale/bias, including a second N tile. */
    memset(a, 0, 2048);
    memset(b, 0, 4096);
    for (uint32_t n = 0; n < 2; n++)
        for (uint32_t i = 0; i < 32; i++) {
            float bias = (float)((int)(n * 32 + i) - 31) * .03125f;
            union {
                float f;
                uint32_t u;
            } bits = {bias};
            uint32_t mag = bits.u & 0x7fffffff;
            uint16_t h = (uint16_t)((bits.u >> 16) & 0x8000);
            if (mag)
                h |= (uint16_t)((mag >> 13) - 0x1c000);
            scale[n * 64 + i] = 0x3c00u | ((uint32_t)h << 16);
            scale[n * 64 + 32 + i] = 0;
        }
    hx_hmx_init(scale);
    hx_hmx_gemm_bias(a, b, c, 2048, 2, scale);
    hx_hmx_finish(done);
    for (uint32_t n = 0; n < 2; n++)
        for (uint32_t r = 0; r < 32; r++)
            for (uint32_t col = 0; col < 32; col++) {
                float bias = (float)((int)(n * 32 + col) - 31) * .03125f;
                union {
                    float f;
                    uint32_t u;
                } bits = {bias};
                uint32_t mag = bits.u & 0x7fffffff;
                uint16_t expected = (uint16_t)((bits.u >> 16) & 0x8000);
                if (mag)
                    expected |= (uint16_t)((mag >> 13) - 0x1c000);
                uint32_t index = n * 1024 + (r / 2) * 64 + col * 2 + r % 2;
                if (c[index] != expected) {
                    printf("FAIL matrix FP16 bias tile%u row%u col%u got%x expected%x\n",
                           (unsigned)n, (unsigned)r, (unsigned)col, c[index], expected);
                    return 1;
                }
            }
    puts("PASS fp16 HMX deep-load boundaries K=32,1024,1056,1152,4096");
    return 0;
}
