/* Isolate generic aligned spatial leaf setup with a 1x1 identity convolution. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

extern void qd_qnn_hmx_3x3(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);
extern void qd_hmx_finish(void *, void *);
extern void qd_set_hmx_params_stride1(uint32_t *,uint32_t,uint32_t,uint32_t,
                                      uint32_t,uint32_t,uint32_t,uint32_t);
#ifndef MANUAL_CFG
#define MANUAL_CFG 0
#endif

int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x80000; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");

    memset((void *)base, 0, 0x80000);
    uint8_t *output = (uint8_t *)base;
    uint8_t *input = output + 0x10000;
    uint8_t *weights = input + 0x10000;
    int32_t *fi = (int32_t *)(input + 0x20000);
    int32_t *zero = (int32_t *)(input + 0x21000);
    uint32_t *inblocks = (uint32_t *)(input + 0x22000);
    uint32_t *outblocks = (uint32_t *)(input + 0x22400);
    uint8_t *guard = output + 0x4000;
    uint32_t *done = (uint32_t *)(input + 0x23000);

    memset(output, 0xa5, 2048);
    memset(guard, 0xa5, 2048);
    for (uint32_t i = 0; i < 256; ++i) {
        inblocks[i] = (uint32_t)(input + 0x24000);
        outblocks[i] = (uint32_t)guard;
    }
    inblocks[0] = (uint32_t)input;
    outblocks[0] = (uint32_t)output;
    memset(weights, 0, 9 * 1024); /* Explicitly zero the entire 3x3 weight envelope. */
    memset(fi, 0, 64 * sizeof(*fi));
    for (uint32_t y = 0; y < 8; ++y)
        for (uint32_t x = 0; x < 8; ++x)
            for (uint32_t c = 0; c < 32; ++c)
                input[(y * 256) + (x * 32) + c] = (uint8_t)(1 + y * 11 + x * 7 + c % 5);
    for (uint32_t c = 0; c < 32; ++c) {
        uint32_t k = c;
        weights[(((k / 32) * 8 + (k % 32) / 4) * 32 + c) * 4 + k % 4] = 1;
        fi[c] = 0x406000;
    }
    for (uint32_t c = 0; c < 32; ++c) zero[c] = 0x404000;

    uint32_t in[] = {(uint32_t)inblocks, 1, 1, 8, 8, 32};
    uint32_t out[] = {(uint32_t)outblocks, 1, 1, 8, 8, 32};
    uint32_t cfg[16] = {0};
    uint32_t mode[] = {1, 0};
#if MANUAL_CFG
    cfg[3] = 2047;
    cfg[6] = 1023;
#else
    qd_set_hmx_params_stride1(cfg, 0x7e0, 1, 1, 32, 128, 0, 0);
#endif
    printf("baseline manual=%u cfg=", (unsigned)MANUAL_CFG);
    for (uint32_t i = 0; i < 13; ++i) printf("%s%x", i ? "," : "", cfg[i]);
    puts("");
    __asm__ volatile("mxclracc" ::: "memory");
    qd_qnn_hmx_3x3(out, in, weights, fi, cfg, mode);
    *done = 0xffffffff;
    qd_hmx_finish(zero, done);
    uint32_t tries = 0;
    while (*done && tries++ < 100001) __asm__ volatile("nop");
    if (*done) {
        printf("FAIL completion=%x tries=%u output=%u\n", *done, (unsigned)tries, output[0]);
        return 1;
    }
    for (uint32_t y = 0; y < 8; ++y)
        for (uint32_t x = 0; x < 8; ++x)
            for (uint32_t c = 0; c < 32; ++c) {
                uint32_t got = output[y * 256 + x * 32 + c];
                uint32_t want = (uint8_t)(1 + y * 11 + x * 7 + c % 5);
                if (got != want) {
                    printf("FAIL output y=%u x=%u c=%u got=%u want=%u\n",
                           (unsigned)y, (unsigned)x, (unsigned)c, got, want);
                    return 2;
                }
            }
    for (uint32_t i = 0; i < 2048; ++i)
        if (guard[i] != 0xa5) { printf("FAIL guard byte=%u\n", (unsigned)i); return 3; }
    for (uint32_t i = 1; i < 256; ++i)
        if (inblocks[i] != (uint32_t)(input + 0x24000) || outblocks[i] != (uint32_t)guard) {
            printf("FAIL table tail index=%u\n", (unsigned)i); return 4;
        }
    puts("PASS generic aligned 1x1 center identity, output and table guards");
    return 0;
}
