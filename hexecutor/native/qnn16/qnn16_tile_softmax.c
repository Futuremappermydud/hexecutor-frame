/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): softmax over channels. */
#include "qnn16_tile_priv.h"

/* Order-preserving signed key of an IEEE float's bits: float order equals
 * signed-integer order of the keys. Involutive: applying it twice is identity. */
static inline HVX_Vector softmax_key(HVX_Vector f) {
    const HVX_Vector sgn = Q6_Vw_vasr_VwR(f, 31);
    return Q6_V_vxor_VV(f, Q6_V_vand_VV(sgn, Q6_V_vsplat_R(0x7fffffff)));
}
static inline float softmax_unkey(int32_t k) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = (uint32_t)k;
    if (k < 0)
        c.u ^= 0x7fffffffu;
    return c.f;
}
/* exp(v - m) in qf32 for one word lane: vq = v (qf32), msf = m (IEEE splat).
 * Same shape as swish_core: E = (poly(t / 16))^16 with t = min(m - v, 16). */
static inline __attribute__((always_inline)) HVX_Vector softmax_exp(HVX_Vector vq, HVX_Vector msf, const SwishK *k) {
    const HVX_Vector d = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(vq, msf)); /* v - m <= 0 */
    const HVX_Vector t = Q6_Vw_vmin_VwVw(Q6_V_vand_VV(d, Q6_V_vsplat_R(0x7fffffff)), Q6_V_vsplat_R(0x41800000));
    const HVX_Vector u = Q6_Vqf32_vmpy_VsfVsf(t, splat_sf(0.0625f));
    HVX_Vector e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(k->c5, u), k->c[4]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[3]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[2]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[1]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[0]);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    return e;
}

int qnn16_tile_softmax(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    if (!blocked_any(x) || !blocked_any(out) || x->dims.b != out->dims.b || x->dims.h != out->dims.h ||
        x->dims.w != out->dims.w || x->dims.c != out->dims.c || x->padding.w != out->padding.w ||
        (x->dims.c & 31u) || x->dims.c > 2048u)
        return 0;
    /* Pixel pair p holds logical pixels 2p and 2p + 1; lanes_of() decides which are stored. */
    const Map mx = map_of(x), mo = map_of(out);
    const SwishK k = swish_k();
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    const uint32_t ng = x->dims.c / 32u;
    const uint32_t p0 = x->padding.w / 2u, pend = (x->padding.w + x->dims.w + 1u) / 2u;
    const HVX_Vector zero = Q6_V_vzero();
    uint32_t slot = 0;
    for (uint32_t b = 0; b < x->dims.b; b++)
        for (uint32_t y = 0; y < x->dims.h; y++)
            for (uint32_t pp = p0; pp < pend; pp++) {
                PART_ROW(slot);
                const uint32_t lanes = lanes_of(out, pp);
                if (!lanes)
                    continue;
                const uint32_t fi = y + x->padding.h, fo = y + out->padding.h;
                const uint32_t bi = b + x->padding.b, bo = b + out->padding.b;
                const int32_t ip = (int32_t)pp;

                /* Pass 1: per-lane running max over the groups (as keys), then reduce the 32 lanes. */
                HVX_Vector kl = Q6_V_vsplat_R((int)0x80000000u), kh = kl;
                for (uint32_t g = 0; g < ng; g++) {
                    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, bi, fi, ip, g), one);
                    kl = Q6_Vw_vmax_VwVw(kl, softmax_key(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w))));
                    kh = Q6_Vw_vmax_VwVw(kh, softmax_key(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w))));
                }
                int32_t kb[32] __attribute__((aligned(128)));
                *(HVX_Vector *)kb = kl;
                int32_t bl = (int32_t)0x80000000u;
                for (uint32_t i = 0; i < 32u; i++)
                    if (kb[i] > bl)
                        bl = kb[i];
                *(HVX_Vector *)kb = kh;
                int32_t bh = (int32_t)0x80000000u;
                for (uint32_t i = 0; i < 32u; i++)
                    if (kb[i] > bh)
                        bh = kb[i];
                const float me = softmax_unkey(bl), mo_f = softmax_unkey(bh);
                const HVX_Vector me_sf = splat_sf(me), mo_sf = splat_sf(mo_f);

                /* Pass 2: per-lane sums of exp(v - m) over the groups, then reduce. */
                HVX_Vector sl = zero, sh = zero;
                for (uint32_t g = 0; g < ng; g++) {
                    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, bi, fi, ip, g), one);
                    sl = Q6_Vqf32_vadd_Vqf32Vqf32(sl, softmax_exp(Q6_V_lo_W(w), me_sf, &k));
                    sh = Q6_Vqf32_vadd_Vqf32Vqf32(sh, softmax_exp(Q6_V_hi_W(w), mo_sf, &k));
                }
                float fl[32] __attribute__((aligned(128))), fh[32] __attribute__((aligned(128)));
                *(HVX_Vector *)fl = Q6_Vsf_equals_Vqf32(sl);
                *(HVX_Vector *)fh = Q6_Vsf_equals_Vqf32(sh);
                float s_e = 0.0f, s_o = 0.0f;
                for (uint32_t i = 0; i < 32u; i++) {
                    s_e += fl[i];
                    s_o += fh[i];
                }
                const float r_e = 1.0f / s_e, r_o = 1.0f / s_o;
                const HVX_Vector re_qf = qf_of_sf(splat_sf(r_e)), ro_qf = qf_of_sf(splat_sf(r_o));

                /* Pass 3: recompute exp per group, scale, store. Each group is read before its store. */
                for (uint32_t g = 0; g < ng; g++) {
                    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, bi, fi, ip, g), one);
                    const HVX_Vector ol = Q6_Vqf32_vmpy_Vqf32Vqf32(softmax_exp(Q6_V_lo_W(w), me_sf, &k), re_qf);
                    const HVX_Vector oh = Q6_Vqf32_vmpy_Vqf32Vqf32(softmax_exp(Q6_V_hi_W(w), mo_sf, &k), ro_qf);
                    put(vec(&mo, bo, fo, ip, g), Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(oh, ol)), lanes);
                }
            }
    return 1;
}
