/* HVX tile form (qnn16_tile.h): L2NORM over axis 1 (H) of [b,h,w,c] tiles. */
#include "qnn16_tile_priv.h"
#include <math.h>

/* out = x / sqrt(max(sum of squares over h, epsilon)) per (b, w, c) column.
 * A unit is (b, pixel pair, channel group): its even and odd pixel lanes are
 * the columns (w, c) of that pair and group, so units share no data and split
 * over the parts with no barrier. Pass 1 sums the squares of the unit's rows in
 * qf32 (the rows in the generic path's order); the 64 lane sums are reduced to
 * their reciprocal square roots in scalar code on a local buffer, with the
 * generic path's expression; pass 2 scales each row and converts to FP16. Input
 * and output may be the same tensor: a unit reads and writes only its own vectors. */
static void l2norm_scalar(const Qnn16Tensor *x, const Qnn16Tensor *out, float epsilon, uint32_t part, uint32_t parts);

int qnn16_tile_l2norm(const Qnn16Tensor *x, const Qnn16Tensor *out, float epsilon, uint32_t part, uint32_t parts) {
    if (!whole_pairs(x) || !whole_pairs(out))
        return 0;
    if (x->dims.b != out->dims.b || x->dims.h != out->dims.h || x->dims.w != out->dims.w ||
        x->dims.c != out->dims.c || x->padding.b != out->padding.b || x->padding.h != out->padding.h ||
        x->padding.w != out->padding.w)
        return 0;
    if (x->dims.c % 32u) { /* partial last channel group, e.g. the skeleton head's c = 24 */
        l2norm_scalar(x, out, epsilon, part, parts);
        return 1;
    }
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t bx = x->padding.b, bo = out->padding.b;
    const int32_t p0 = (int32_t)(x->padding.w >> 1);
    const uint32_t pairs = x->dims.w >> 1, groups = x->dims.c / 32u;
    const uint32_t units = x->dims.b * pairs * groups;
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00), zero = Q6_V_vzero();
    for (uint32_t u = part; u < units; u += parts) {
        const uint32_t b = u / (pairs * groups), r = u % (pairs * groups), g = r % groups;
        const int32_t pair = p0 + (int32_t)(r / groups);
        HVX_Vector acc0 = zero, acc1 = zero;
        for (uint32_t y = 0; y < x->dims.h; y++) {
            const HVX_Vector v = ld(&mx, bx + b, y + x->padding.h, pair, g);
            const HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
            acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_V_lo_W(sq));
            acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_V_hi_W(sq));
        }
        /* Lanes 0..31 of s: even pixel per channel; 32..63: odd pixel. */
        float s[64] __attribute__((aligned(128)));
        *(HVX_Vector *)s = Q6_Vsf_equals_Vqf32(acc0);
        *(HVX_Vector *)(s + 32) = Q6_Vsf_equals_Vqf32(acc1);
        for (uint32_t i = 0; i < 64u; i++)
            s[i] = 1.f / sqrtf(s[i] > epsilon ? s[i] : epsilon);
        const HVX_Vector f0 = qf_of_sf(*(HVX_Vector *)s), f1 = qf_of_sf(*(HVX_Vector *)(s + 32));
        for (uint32_t y = 0; y < x->dims.h; y++) {
            const uint32_t fy = y + x->padding.h;
            const HVX_Vector v = ld(&mx, bx + b, fy, pair, g);
            const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(v, one);
            const HVX_Vector lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(w), f0);
            const HVX_Vector hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(w), f1);
            HVX_Vector *dst = vec(&mo, bo + b, fy, pair, g);
            if (dst)
                *dst = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
        }
    }
    return 1;
}

/* Channel counts that are not a multiple of 32: the generic path's float32 arithmetic in its order (sum of
 * squares over rows top to bottom, 1 / sqrtf(max(sum, epsilon)), one rounding to FP16 per output), so the result
 * is bit-identical to it. What makes the generic path slow is its per-element gather and scatter, not the
 * arithmetic: here a unit (b, pixel pair, channel group) addresses its row vectors directly and touches only the
 * live channels, so the padding channels of the output keep their contents, as on the generic path. Channel c
 * of pixel (x & 1) is FP16 lane 2c + (x & 1) of a pair vector (qnn16_hvx.c tile_byte). hf/fh are the inline forms
 * of the generic path's qnn16_half_to_float/qnn16_float_to_half. Absent blocks read as 0 and are not written.
 * Scalar because the engine is built without -mhvx-ieee-fp and qf32 lanes would not match the generic rounding. */
static void l2norm_scalar(const Qnn16Tensor *x, const Qnn16Tensor *out, float epsilon, uint32_t part, uint32_t parts) {
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t bx = x->padding.b, bo = out->padding.b;
    const int32_t p0 = (int32_t)(x->padding.w >> 1);
    const uint32_t pairs = x->dims.w >> 1, groups = (x->dims.c + 31u) / 32u;
    const uint32_t units = x->dims.b * pairs * groups;
    for (uint32_t u = part; u < units; u += parts) {
        const uint32_t b = u / (pairs * groups), r = u % (pairs * groups), g = r % groups;
        const int32_t pair = p0 + (int32_t)(r / groups);
        const uint32_t live = x->dims.c - g * 32u < 32u ? x->dims.c - g * 32u : 32u, lanes = 2u * live;
        float acc[64];
        for (uint32_t i = 0; i < lanes; i++)
            acc[i] = 0.f;
        for (uint32_t y = 0; y < x->dims.h; y++) {
            const uint16_t *v = (const uint16_t *)vec(&mx, bx + b, y + x->padding.h, pair, g);
            if (!v)
                continue; /* adds 0 * 0 */
            for (uint32_t i = 0; i < lanes; i++) {
                const float f = hf(v[i]);
                acc[i] += f * f;
            }
        }
        for (uint32_t i = 0; i < lanes; i++)
            acc[i] = 1.f / sqrtf(acc[i] > epsilon ? acc[i] : epsilon);
        for (uint32_t y = 0; y < x->dims.h; y++) {
            const uint32_t fy = y + x->padding.h;
            const uint16_t *v = (const uint16_t *)vec(&mx, bx + b, fy, pair, g);
            uint16_t *dst = (uint16_t *)vec(&mo, bo + b, fy, pair, g);
            if (!dst)
                continue;
            for (uint32_t i = 0; i < lanes; i++)
                dst[i] = fh((v ? hf(v[i]) : 0.f) * acc[i]);
        }
    }
}
