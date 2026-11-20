/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* im2col_tile_any (HVX) must produce the same K-tiles as the scalar hx_im2col_tile for any dense
 * convolution shape: random channels, kernel, stride, padding, dilation and odd widths. */
#include "../native/tensor/fp16.c"
#include <stdio.h>
#include <stdlib.h>
static uint32_t seed = 12345u;
static uint32_t rnd(uint32_t n) {
    seed = seed * 1664525u + 1013904223u;
    return (seed >> 8) % n;
}
int main(void) {
    static uint16_t src[1u << 20] __attribute__((aligned(128)));
    static uint16_t want[64u * 1024u] __attribute__((aligned(128)));
    static uint16_t got[64u * 1024u] __attribute__((aligned(128)));
    uint32_t cases = 0;
    for (uint32_t n = 0; n < 400u; n++) {
        HxTensor a = {0}, b = {0};
        HxOp o = {0};
        a.c = 1u + rnd(80u);
        a.stride = (a.c + 31u) & ~31u;
        a.h = 1u + rnd(24u);
        a.w = 1u + rnd(24u);
        const uint32_t kh = 1u + rnd(4u), kw = n & 1u ? kh : 1u + rnd(4u), sh = 1u + rnd(2u),
                       sw = n & 2u ? sh : 1u + rnd(2u), dh = n % 7u ? 1u : 1u + rnd(2u),
                       dw = n % 7u ? 1u : 1u + rnd(2u);
        const uint32_t pt = rnd(kh + 1u), pl = rnd(kw + 1u);
        const int32_t oh = ((int32_t)(a.h + 2u * pt) - (int32_t)((kh - 1u) * dh + 1u)) / (int32_t)sh + 1,
                      ow = ((int32_t)(a.w + 2u * pl) - (int32_t)((kw - 1u) * dw + 1u)) / (int32_t)sw + 1;
        if (oh <= 0 || ow <= 0)
            continue;
        b.h = (uint32_t)oh;
        b.w = (uint32_t)ow;
        b.c = 32u;
        b.stride = 32u;
        const uint32_t K = kh * kw * a.c, kb = (K + 31u) / 32u;
        if (kb > 32u || plane_bytes(&a) / 2u * (a.stride / 32u) > (1u << 20))
            continue;
        o.kind = HX_CONV;
        o.z = dh == 1u && dw == 1u ? HXFG_NONE : (dh << 8) | dw;
        o.p[0] = kh; o.p[1] = kw; o.p[2] = sh; o.p[3] = sw; o.p[4] = pt; o.p[5] = pl; o.p[10] = kb;
        for (uint32_t i = 0; i < plane_bytes(&a) / 2u * (a.stride / 32u); i++)
            src[i] = (uint16_t)(0x3c00u + rnd(0x400u) + (rnd(2u) << 15));
        for (uint32_t t = 0; t < tiles(&b); t++) {
            memset(want, 0xa5, kb * 2048u);
            memset(got, 0x5a, kb * 2048u);
            hx_im2col_tile(want, src, a.h, a.w, a.c, plane_bytes(&a) / 2u, b.h, b.w, o.p, kb, t,
                           hx_dilation_h(&o), hx_dilation_w(&o));
            im2col_tile_any((HVX_Vector *)got, src, &a, &b, &o, t);
            if (memcmp(want, got, kb * 2048u)) {
                printf("FAIL case %u a %ux%ux%u k %ux%u s %ux%u pad %u,%u dil %u,%u tile %u\n",
                       (unsigned)n, (unsigned)a.h, (unsigned)a.w, (unsigned)a.c, (unsigned)kh,
                       (unsigned)kw, (unsigned)sh, (unsigned)sw, (unsigned)pt, (unsigned)pl,
                       (unsigned)dh, (unsigned)dw, (unsigned)t);
                return 1;
            }
        }
        cases++;
    }
    printf("PASS im2col any %u cases\n", (unsigned)cases);
    return 0;
}
