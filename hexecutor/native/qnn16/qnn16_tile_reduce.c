/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): reduce. */
#include "qnn16_tile_priv.h"

/* Pattern 1: out [1,1,1,C] = sum over every row and column, per channel.
 * Accumulates the even (lo) and odd (hi) pixel words of every pair in qf32;
 * word j of the sum is channel j of the group. */
static int reduce_hw(const Qnn16Tensor *x, const Qnn16Tensor *out) {
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t bx = x->padding.b, bo = out->padding.b;
    const uint32_t groups = (x->dims.c + 31u) / 32u;
    const int32_t p0 = (int32_t)(x->padding.w >> 1);
    const int32_t p1 = (int32_t)((x->padding.w + x->dims.w + 1u) >> 1); /* exclusive */
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    for (uint32_t g = 0; g < groups; g++) {
        HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero();
        for (uint32_t y = 0; y < x->dims.h; y++) {
            const uint32_t fy = y + x->padding.h;
            for (int32_t p = p0; p < p1; p++) {
                const uint32_t lanes = lanes_of(x, (uint32_t)p);
                if (!lanes)
                    continue;
                const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, bx, fy, p, g), one);
                if (lanes & 1u)
                    acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_V_lo_W(w));
                if (lanes & 2u)
                    acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_V_hi_W(w));
            }
        }
        const HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, acc1);
        /* Even lane of channel c = sum of channel c; odd lanes are zero and not stored. */
        const HVX_Vector value = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_V_vzero(), s));
        store_live(vec(&mo, bo, out->padding.h, (int32_t)(out->padding.w >> 1), g), value, 1u, channel_mask(out, g));
    }
    return 1;
}

/* Pattern 2: out [1,H,1,1] = sum over all columns and channels, per row. */
static int reduce_w(const Qnn16Tensor *x, const Qnn16Tensor *out) {
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t bx = x->padding.b, bo = out->padding.b;
    const uint32_t groups = (x->dims.c + 31u) / 32u;
    const int32_t p0 = (int32_t)(x->padding.w >> 1);
    const int32_t p1 = (int32_t)((x->padding.w + x->dims.w + 1u) >> 1); /* exclusive */
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00), zero = Q6_V_vzero();
    for (uint32_t y = 0; y < x->dims.h; y++) {
        const uint32_t fy = y + x->padding.h;
        HVX_Vector acc0 = zero, acc1 = zero;
        for (uint32_t g = 0; g < groups; g++) {
            const HVX_VectorPred live = channel_mask(x, g);
            const int partial = x->dims.c - g * 32u < 32u;
            for (int32_t p = p0; p < p1; p++) {
                const uint32_t lanes = lanes_of(x, (uint32_t)p);
                if (!lanes)
                    continue;
                HVX_Vector v = ld(&mx, bx, fy, p, g);
                if (partial)
                    v = Q6_V_vmux_QVV(live, v, zero);
                const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(v, one);
                if (lanes & 1u)
                    acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_V_lo_W(w));
                if (lanes & 2u)
                    acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_V_hi_W(w));
            }
        }
        /* 32 partial sums (one per channel lane) added in scalar C on the stack. */
        float part[32] __attribute__((aligned(128)));
        *(HVX_Vector *)part = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc0, acc1));
        float sum = 0.0f;
        for (uint32_t i = 0; i < 32u; i++)
            sum += part[i];
        const HVX_Vector *dst = vec(&mo, bo, y + out->padding.h, (int32_t)(out->padding.w >> 1), 0);
        if (dst) /* halfword 0 = pixel 0 (even lane), channel 0 */
            Q6_vmem_QRIV(Q6_Q_vsetq_R(2), (HVX_Vector *)dst, Q6_V_vsplat_R((int)fh(sum)));
    }
    return 1;
}

int qnn16_tile_reduce_sum(const Qnn16Tensor *x, const Qnn16Tensor *out) {
    if (!blocked_any(x) || !blocked_any(out) || x->dims.b != 1u || out->dims.b != 1u)
        return 0;
    if ((x->padding.w & 1u) || (out->padding.w & 1u))
        return 0;
    if (out->dims.h == 1u && out->dims.w == 1u && out->dims.c == x->dims.c)
        return reduce_hw(x, out);
    if (out->dims.h == x->dims.h && out->dims.w == 1u && out->dims.c == 1u)
        return reduce_w(x, out);
    return 0;
}

/* Pattern 3: out [1,1,W,C] = mean over every row, per pixel and channel. Each
 * pixel pair and channel group sums its rows in row order (the generic path's
 * order) in qf32, then multiplies by the float32 mean and converts exactly
 * (f2h_words). Pair p of out covers the same pixels as pair p of x, so the
 * parts split by pair. */
int qnn16_tile_reduce_mean_h(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    if (!blocked_any(x) || !blocked_any(out) || x->dims.b != 1u || out->dims.b != 1u)
        return 0;
    if (out->dims.h != 1u || out->dims.w != x->dims.w || out->dims.c != x->dims.c || out->padding.w != x->padding.w ||
        (x->padding.w & 1u))
        return 0;
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t bx = x->padding.b, bo = out->padding.b;
    const uint32_t groups = (x->dims.c + 31u) / 32u;
    const int32_t p0 = (int32_t)(x->padding.w >> 1);
    const int32_t p1 = (int32_t)((x->padding.w + x->dims.w + 1u) >> 1); /* exclusive */
    const uint32_t n_in = x->dims.b * x->dims.h * x->dims.w * x->dims.c;
    const uint32_t n_out = out->dims.b * out->dims.h * out->dims.w * out->dims.c;
    const float mean = (float)n_out / (float)n_in;
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00), m = splat_sf(mean);
    for (uint32_t g = 0; g < groups; g++) {
        const HVX_VectorPred qc = channel_mask(out, g);
        for (int32_t p = p0; p < p1; p++) {
            if ((uint32_t)(p - p0) % parts != part)
                continue;
            const uint32_t lanes = lanes_of(x, (uint32_t)p);
            if (!lanes)
                continue;
            HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero(); /* even and odd pixel */
            for (uint32_t y = 0; y < x->dims.h; y++) {
                const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, bx, y + x->padding.h, p, g), one);
                if (lanes & 1u)
                    acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_V_lo_W(w));
                if (lanes & 2u)
                    acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_V_hi_W(w));
            }
            /* Scale in float32, then exact round to nearest even: even pixel in each word's
             * low half, odd pixel in the high half. */
            const HVX_Vector lo = f2h_words(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc0), m)));
            const HVX_Vector hi = f2h_words(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc1), m)));
            store_live(vec(&mo, bo, out->padding.h, p, g), Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(hi, 16)), lanes, qc);
        }
    }
    return 1;
}
