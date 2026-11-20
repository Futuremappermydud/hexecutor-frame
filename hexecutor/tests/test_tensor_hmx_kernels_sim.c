/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hexagon_standalone.h>
#include "../native/tensor/hmx.c"
static int conversion(void) {
    const float values[] = {
        0,   -0.0f,    1,        -1,         65504,    65520,        INFINITY,     -INFINITY,
        NAN, 0x1p-24f, 0x1p-25f, 0x1.8p-24f, 0x1p-14f, 0x1.ffcp-15f, 1 + 0x1p-11f, 1 + 0x1.8p-10f};
    const uint16_t expected[] = {0,      0x8000, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0x7c00, 0xfc00,
                                 0x7e00, 1,      0,      2,      0x400,  0x400,  0x3c00, 0x3c02};
    float in[32] __attribute__((aligned(128)));
    uint32_t out[32] __attribute__((aligned(128)));
    for (unsigned i = 0; i < 32; i++)
        in[i] = values[i % 16];
    *(HVX_Vector *)out = half_words(*(HVX_Vector *)in);
    for (unsigned i = 0; i < 32; i++)
        if (out[i] != expected[i % 16]) {
            printf("FAIL half lane %u got %x expected %x\n", i, (unsigned)out[i], expected[i % 16]);
            return 1;
        }
    return 0;
}
static int rgb(uint32_t base, uint32_t kernel) {
    const unsigned ih = 7, iw = 9, co = 35, stride = 64, oh = 4, ow = 5;
    const unsigned k = kernel * kernel * 3, kb = (k + 31) / 32, nb = stride / 32;
    uint8_t *constants = calloc(1, 512 + kb * nb * 2048);
    if (!constants)
        return 1;
    uint16_t *weights = (uint16_t *)(constants + 512);
    HxHeader h = {0};
    HxTensor tensors[2] = {{0, ih, iw, 3, 3, {0}}, {768, oh, ow, co, stride, {0}}};
    HxOp op = {0};
    op.kind = HX_CONV;
    op.x = 0;
    op.out = 1;
    op.y = op.z = HXFG_NONE;
    op.p[0] = op.p[1] = kernel;
    op.p[2] = op.p[3] = 2;
    op.p[4] = op.p[5] = kernel / 2;
    op.p[7] = 0;
    op.p[9] = 512;
    op.p[10] = kb;
    HxGraph g = {&h, tensors, &op, NULL, constants, (uint8_t *)base};
    float *x = (float *)base, *out = (float *)(base + 768);
    for (unsigned i = 0; i < ih * iw * 3; i++)
        x[i] = (float)((int)(i % 9) - 4) * 0.25f;
    for (unsigned channel = 0; channel < k; channel++)
        for (unsigned n = 0; n < co; n++) {
            unsigned at = (n / 32) * kb * 1024 + (channel / 32) * 1024 + ((channel % 32) / 2) * 64 +
                          (n % 32) * 2 + channel % 2;
            // Signed +/-1/16, exact fp16 weights; zero-padded channels/K.
            weights[at] = (channel + n) % 2 ? 0xac00 : 0x2c00;
        }
    hx_hmx_conv(&g, 0, (uint8_t *)(base + 0x700000));
    for (unsigned y = 0; y < oh; y++)
        for (unsigned xx = 0; xx < ow; xx++)
            for (unsigned n = 0; n < stride; n++) {
                float expected = 0;
                if (n < co)
                    for (unsigned ky = 0; ky < kernel; ky++)
                        for (unsigned kx = 0; kx < kernel; kx++) {
                            int yy = (int)(y * 2 + ky) - (int)(kernel / 2),
                                px = (int)(xx * 2 + kx) - (int)(kernel / 2);
                            if (yy < 0 || px < 0 || yy >= ih || px >= iw)
                                continue;
                            for (unsigned c = 0; c < 3; c++) {
                                unsigned channel = (ky * kernel + kx) * 3 + c;
                                expected += x[(yy * iw + px) * 3 + c] *
                                            ((channel + n) % 2 ? -0.0625f : 0.0625f);
                            }
                        }
                float got = out[(y * ow + xx) * stride + n];
                if (fabsf(got - expected) > 1e-6f) {
                    printf("FAIL RGB %ux%u y%u x%u c%u got %f expected %f\n", (unsigned)kernel,
                           (unsigned)kernel, y, xx, n, got, expected);
                    free(constants);
                    return 1;
                }
            }
    free(constants);
    return 0;
}
int main(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x800000u; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    if (conversion() || rgb(base, 3) || rgb(base, 5))
        return 1;
    puts("PASS fp16 conversion boundaries and RGB 3x3/5x5 gather, padding, and channel tails");
    return 0;
}
