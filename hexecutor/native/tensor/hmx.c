/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "hmx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>
extern void hx_hmx_init(void *scale);
extern void hx_hmx_gemm(void *a, void *b, void *c, uint32_t kbytes, uint32_t nt, uint32_t mt);
extern void hx_hmx_finish(void *c);
/* All matrix operands are in the bootstrap's preserved VTCM lease. */

#include "fp16_hvx.h"
/* Place a contiguous RGB row fragment in an HMX K tile. The full-width
 * predicate is explicit: vsetq(128) is empty on this ISA. */
static inline HVX_Vector rgb_piece(HVX_Vector row, uint32_t src, uint32_t dst, uint32_t count) {
    HVX_Vector v = Q6_V_vror_VR(row, (int32_t)(src * 4u) - (int32_t)(dst * 4u));
    HVX_VectorPred mask = Q6_Q_not_Q(Q6_Q_vsetq_R(dst * 4u));
    if (dst + count < 32u)
        mask = Q6_Q_and_QQ(mask, Q6_Q_vsetq_R((dst + count) * 4u));
    return Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}
void hx_hmx_conv(const HxGraph *g, uint32_t index, uint8_t *workspace) {
    uint8_t *const scratch = workspace;
    const HxOp *o = &g->ops[index];
    const HxTensor *a = &g->tensors[o->x], *b = &g->tensors[o->out];
    const uint32_t *p = o->p;
    uint32_t kb = p[10], nt = b->stride / 32u, kbytes = kb * 2048u, wbytes = kbytes * nt;
    uint16_t *ap = (uint16_t *)scratch, *wp = (uint16_t *)(scratch + kbytes),
             *cp = (uint16_t *)(scratch + kbytes + wbytes);
    uint32_t *scale = (uint32_t *)((uint8_t *)cp + nt * 2048u);
    /* Matrix completion stores require a full tile-aligned region. They
     * must not round down into conversion parameters reused by the next tile. */
    uint8_t *completion =
        (uint8_t *)(((uintptr_t)scale + 256u + 2047u) & ~(uintptr_t)2047u);
    memcpy(wp, g->constants + p[9], wbytes);
    for (uint32_t i = 0; i < 32; i++) {
        scale[i] = 0x3c00u;
        scale[32 + i] = 0;
    }
    hx_hmx_init(scale);
    const float *x = (const float *)(g->arena + a->offset),
                *bias = (const float *)(g->constants + p[7]);
    float *out = (float *)(g->arena + b->offset);
    uint32_t pixels = b->h * b->w;
    const float *spatial = x;
    uint32_t spatial_stride = a->stride;
    uint8_t *extra = completion + 2048u;
    /* Compact RGB once into TCM. A scalar DDR load per im2col tap dominated
     * the first convolution; subsequent taps now reuse the local image. */
    uint32_t compact_bytes = a->h * a->w * 3u * 4u;
    int staged_rgb = 0;
    if (a->c == 3 && compact_bytes <= 0x100000u &&
        (uint32_t)(extra - scratch) + ((compact_bytes + 127u) & ~127u) + kb * 256u <= 0x100000u) {
        float *compact = (float *)extra;
        float tmp[32] __attribute__((aligned(128)));
        if (a->stride == 3)
            memcpy(compact, x, compact_bytes);
        else
            for (uint32_t i = 0; i < a->h * a->w; i++) {
                *(HVX_Vector *)tmp = *(const HVX_Vector *)(x + i * a->stride);
                compact[i * 3u] = tmp[0];
                compact[i * 3u + 1u] = tmp[1];
                compact[i * 3u + 2u] = tmp[2];
            }
        staged_rgb = 1;
        spatial = compact;
        spatial_stride = 3;
        extra += ((compact_bytes + 127u) & ~127u);
    }
    float *patch0 = (float *)extra, *patch1 = patch0 + kb * 32u;
    int rgb_gather = staged_rgb && p[0] <= 5 && p[1] <= 5 && a->w <= 224;
    HVX_Vector tap_offset[3], tap_x[3], tap_y[3];
    if (rgb_gather) {
        /* Every tap reuses this RGB image. Convert once instead of doing the
         * same exact FP32->FP16 conversion for every overlapping window. */
        uint32_t words = (compact_bytes + 127u) / 128u;
        for (uint32_t v = 0; v < words; v++)
            ((HVX_Vector *)spatial)[v] = half_words(((const HVX_Vector *)spatial)[v]);
    }
    if (rgb_gather)
        for (uint32_t k = 0; k < kb; k++) {
            uint32_t offsets[32] __attribute__((aligned(128))),
                xs[32] __attribute__((aligned(128))), ys[32] __attribute__((aligned(128)));
            for (uint32_t n = 0; n < 32; n++) {
                uint32_t channel = k * 32u + n, ky = channel / (p[1] * 3u),
                         kx = (channel / 3u) % p[1];
                offsets[n] = ((ky * a->w + kx) * 3u + channel % 3u) * 4u;
                xs[n] = kx;
                ys[n] = ky;
            }
            tap_offset[k] = *(HVX_Vector *)offsets;
            tap_x[k] = *(HVX_Vector *)xs;
            tap_y[k] = *(HVX_Vector *)ys;
        }
    for (uint32_t base = 0; base < pixels; base += 32u) {
        /* A: [K/32][16 pixel pairs][32 input channels][2 pixels].
         * The pointwise path converts/interleaves two complete rows per HVX
         * instruction. Spatial convolutions use explicit zero-padded im2col. */
        if (p[0] == 1 && p[1] == 1 && p[2] == 1 && p[3] == 1 && !p[4] && !p[5] &&
            a->stride >= kb * 32u) {
            for (uint32_t k = 0; k < kb; k++)
                for (uint32_t r = 0; r < 32; r += 2) {
                    HVX_Vector v0 =
                        base + r < pixels
                            ? *(const HVX_Vector *)(x + (base + r) * a->stride + k * 32u)
                            : Q6_V_vzero();
                    HVX_Vector v1 =
                        base + r + 1u < pixels
                            ? *(const HVX_Vector *)(x + (base + r + 1u) * a->stride + k * 32u)
                            : Q6_V_vzero();
                    *(HVX_Vector *)(ap + k * 1024u + (r / 2u) * 64u) =
                        Q6_Vh_vshuffe_VhVh(half_words(v1), half_words(v0));
                }
        } else if (rgb_gather) {
            for (uint32_t r = 0; r < 32u; r += 2) {
                uint32_t pixel = base + r;
                int32_t iy0 = (int32_t)((pixel / b->w) * p[2]) - (int32_t)p[4];
                int32_t ix0 = (int32_t)((pixel % b->w) * p[3]) - (int32_t)p[5];
                if (pixel + 1 < pixels && pixel / b->w == (pixel + 1) / b->w && p[0] == p[1] &&
                    (p[0] == 3 || p[0] == 5) && iy0 >= 0 && ix0 >= 0 &&
                    iy0 + (int32_t)p[0] <= (int32_t)a->h &&
                    ix0 + (int32_t)p[3] + (int32_t)p[1] <= (int32_t)a->w) {
                    HVX_Vector row[5];
                    for (uint32_t ky = 0; ky < p[0]; ky++) {
                        const float *src = spatial + ((iy0 + ky) * a->w + ix0) * 3u;
                        HVX_Vector v0 = *(const HVX_UVector *)src;
                        HVX_Vector v1 = *(const HVX_UVector *)(src + p[3] * 3u);
                        row[ky] = Q6_Vh_vshuffe_VhVh(v1, v0);
                    }
                    if (p[0] == 3) {
                        *(HVX_Vector *)(ap + (r / 2u) * 64u) = Q6_V_vor_VV(
                            rgb_piece(row[0], 0, 0, 9),
                            Q6_V_vor_VV(rgb_piece(row[1], 0, 9, 9), rgb_piece(row[2], 0, 18, 9)));
                    } else {
                        *(HVX_Vector *)(ap + (r / 2u) * 64u) = Q6_V_vor_VV(
                            rgb_piece(row[0], 0, 0, 15),
                            Q6_V_vor_VV(rgb_piece(row[1], 0, 15, 15), rgb_piece(row[2], 0, 30, 2)));
                        *(HVX_Vector *)(ap + 1024u + (r / 2u) * 64u) = Q6_V_vor_VV(
                            rgb_piece(row[2], 2, 0, 13),
                            Q6_V_vor_VV(rgb_piece(row[3], 0, 13, 15), rgb_piece(row[4], 0, 28, 4)));
                        *(HVX_Vector *)(ap + 2048u + (r / 2u) * 64u) = rgb_piece(row[4], 4, 0, 11);
                    }
                    continue;
                }
                for (uint32_t k = 0; k < kb; k++) {
                    HVX_Vector pair[2] = {Q6_V_vzero(), Q6_V_vzero()};
                    for (uint32_t q = 0; q < 2u && base + r + q < pixels; q++) {
                        int32_t iy = (int32_t)(((base + r + q) / b->w) * p[2]) - (int32_t)p[4];
                        int32_t ix = (int32_t)(((base + r + q) % b->w) * p[3]) - (int32_t)p[5];
                        uint32_t row = iy < 0 ? 0u : (uint32_t)iy;
                        HVX_Vector xx = Q6_Vw_vadd_VwVw(tap_x[k], Q6_V_vsplat_R(ix));
                        HVX_Vector yy = Q6_Vw_vadd_VwVw(tap_y[k], Q6_V_vsplat_R(iy));
                        HVX_VectorPred mask =
                            Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(xx, Q6_V_vsplat_R(-1)),
                                        Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(a->w), xx));
                        mask = Q6_Q_and_QQ(mask,
                                           Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(yy, Q6_V_vsplat_R(-1)),
                                                       Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(a->h), yy)));
                        uint32_t valid = p[0] * p[1] * 3u - k * 32u;
                        if (valid > 32u)
                            valid = 32u;
                        if (valid < 32u)
                            mask = Q6_Q_and_QQ(mask, Q6_Q_vsetq_R(valid * 4u));
                        int32_t origin = ((iy - (int32_t)row) * (int32_t)a->w + ix) * 12;
                        HVX_Vector offsets = Q6_Vw_vadd_VwVw(tap_offset[k], Q6_V_vsplat_R(origin));
                        uint32_t rows = a->h - row;
                        if (rows > p[0])
                            rows = p[0];
                        Q6_vgather_AQRMVw((HVX_Vector *)patch0, mask,
                                          (uint32_t)(spatial + row * a->w * 3u),
                                          rows * a->w * 12u - 1u, offsets);
                        pair[q] = Q6_V_vmux_QVV(mask, *(volatile HVX_Vector *)patch0, Q6_V_vzero());
                    }
                    *(HVX_Vector *)(ap + k * 1024u + (r / 2u) * 64u) =
                        Q6_Vh_vshuffe_VhVh(pair[1], pair[0]);
                }
            }
        } else {
            for (uint32_t r = 0; r < 32u; r += 2) {
                memset(patch0, 0, kb * 256u);
                for (uint32_t q = 0; q < 2u && base + r + q < pixels; q++) {
                    float *patch = q ? patch1 : patch0;
                    uint32_t oy = (base + r + q) / b->w, ox = (base + r + q) % b->w;
                    for (uint32_t ky = 0; ky < p[0]; ky++)
                        for (uint32_t kx = 0; kx < p[1]; kx++) {
                            int32_t iy = (int32_t)(oy * p[2] + ky) - (int32_t)p[4],
                                    ix = (int32_t)(ox * p[3] + kx) - (int32_t)p[5];
                            if (iy < 0 || ix < 0 || iy >= (int32_t)a->h || ix >= (int32_t)a->w)
                                continue;
                            const float *src =
                                spatial + ((uint32_t)iy * a->w + (uint32_t)ix) * spatial_stride;
                            float *dest = patch + (ky * p[1] + kx) * a->c;
                            if (a->c == 3) {
                                dest[0] = src[0];
                                dest[1] = src[1];
                                dest[2] = src[2];
                            } else
                                memcpy(dest, src, a->c * 4u);
                        }
                }
                for (uint32_t k = 0; k < kb; k++) {
                    HVX_Vector v0 = *(const HVX_Vector *)(patch0 + k * 32u),
                               v1 = *(const HVX_Vector *)(patch1 + k * 32u);
                    *(HVX_Vector *)(ap + k * 1024u + (r / 2u) * 64u) =
                        Q6_Vh_vshuffe_VhVh(half_words(v1), half_words(v0));
                }
            }
        }
        hx_hmx_gemm(ap, wp, cp, kbytes, nt, 1);
        /* Ordered completion at a fresh address drains matrix stores before
         * any worker reads the result or overwrites the activation tiles. */
        hx_hmx_finish(completion);
        for (uint32_t n = 0; n < nt; n++) {
            HVX_Vector bv = *(const HVX_Vector *)(bias + n * 32u);
            for (uint32_t r = 0; r < 32u && base + r < pixels; r += 2) {
                HVX_Vector hv = *(const HVX_Vector *)(cp + n * 1024u + (r / 2u) * 64u);
                HVX_VectorPair f = Q6_Wqf32_vmpy_VhfVhf(hv, Q6_Vh_vsplat_R(0x3c00));
                HVX_Vector v0 = Q6_Vsf_equals_Vqf32(
                    Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(f)), bv));
                HVX_Vector v1 = Q6_Vsf_equals_Vqf32(
                    Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(f)), bv));
                *(HVX_Vector *)(out + (base + r) * b->stride + n * 32u) = activate(v0, p[8]);
                if (base + r + 1u < pixels)
                    *(HVX_Vector *)(out + (base + r + 1u) * b->stride + n * 32u) =
                        activate(v1, p[8]);
            }
        }
    }
}
