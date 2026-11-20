/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): f32. */
#include "qnn16_tile_priv.h"

int qnn16_tile_slice_f32(float *dst, const float *src, uint32_t rows, uint32_t xc, uint32_t oc, uint32_t start) {
    if (rows == 0 || oc == 0 || start + oc > xc)
        return 0;
    for (uint32_t r = 0; r < rows; r++) {
        const float *s = src + (uint32_t)r * xc + start;
        float *d = dst + (uint32_t)r * oc;
        for (uint32_t c = 0; c < oc; c += 32u) {
            const uint32_t n = oc - c < 32u ? oc - c : 32u;
            const HVX_Vector v = *(const HVX_UVector *)(s + c);
            store_bytes((uint8_t *)(d + c), v, 4u * n);
        }
    }
    return 1;
}

#define WELFORD_MAX_C 256u
#define WELFORD_MAX_COUNT 16u
int qnn16_tile_welford_merge(float *out, const float *const *partials, uint32_t count, const float *factor,
                             const float *ratio, uint32_t c) {
    if (count == 0 || count > WELFORD_MAX_COUNT || c == 0 || c > WELFORD_MAX_C)
        return 0;
    /* Results of every chunk: res[2k] = mean, res[2k + 1] = m2. Stored only at the end, since out may alias partials[0]. */
    HVX_Vector res[2 * (WELFORD_MAX_C / 32u)];
    const uint32_t chunks = (c + 31u) / 32u;
    for (uint32_t k = 0; k < chunks; k++) {
        const uint32_t j = 32u * k;
        HVX_Vector mean = qf_of_sf(*(const HVX_UVector *)(partials[0] + j));
        HVX_Vector m2 = qf_of_sf(*(const HVX_UVector *)(partials[0] + c + j));
        for (uint32_t i = 1; i < count; i++) {
            const float *p = partials[i];
            const HVX_Vector nm = qf_of_sf(*(const HVX_UVector *)(p + j));
            const HVX_Vector nq = qf_of_sf(*(const HVX_UVector *)(p + c + j));
            const HVX_Vector f = qf_of_sf(*(const HVX_UVector *)(factor + (i - 1) * 32u));
            const HVX_Vector r = qf_of_sf(*(const HVX_UVector *)(ratio + (i - 1) * 32u));
            const HVX_Vector delta = Q6_Vqf32_vsub_Vqf32Vqf32(nm, mean);
            m2 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m2, nq),
                                          Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(delta, delta), f));
            mean = Q6_Vqf32_vadd_Vqf32Vqf32(mean, Q6_Vqf32_vmpy_Vqf32Vqf32(delta, r));
        }
        if (count == 1u) { /* a plain copy: keep the bits */
            res[2u * k] = *(const HVX_UVector *)(partials[0] + j);
            res[2u * k + 1u] = *(const HVX_UVector *)(partials[0] + c + j);
            continue;
        }
        res[2u * k] = Q6_Vsf_equals_Vqf32(mean);
        res[2u * k + 1u] = Q6_Vsf_equals_Vqf32(m2);
    }
    for (uint32_t k = 0; k < chunks; k++) {
        const uint32_t j = 32u * k;
        const uint32_t n = c - j < 32u ? c - j : 32u;
        store_bytes((uint8_t *)(out + j), res[2u * k], 4u * n);
        store_bytes((uint8_t *)(out + c + j), res[2u * k + 1u], 4u * n);
    }
    return 1;
}
