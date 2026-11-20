/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef HEXECUTOR_TENSOR_FP16_HVX_H
#define HEXECUTOR_TENSOR_FP16_HVX_H
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
/* IEEE binary16 round-to-nearest-even, including subnormals. The V75
 * compiler does not lower sf/hf vcvt, so convert the bit fields with HVX. */
static inline HVX_Vector half_words(HVX_Vector v) {
    HVX_Vector one = Q6_V_vsplat_R(1), zero = Q6_V_vzero();
    HVX_Vector a = Q6_V_vand_VV(v, Q6_V_vsplat_R(0x7fffffff));
    HVX_Vector e = Q6_Vuw_vlsr_VuwR(a, 23);
    HVX_Vector round =
        Q6_Vw_vadd_VwVw(Q6_V_vsplat_R(0xfff), Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 13), one));
    HVX_Vector normal =
        Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(a, round), 13), Q6_V_vsplat_R(0x1c000));
    HVX_Vector sh =
        Q6_Vw_vmin_VwVw(Q6_V_vsplat_R(24),
                        Q6_Vw_vmax_VwVw(Q6_V_vsplat_R(14), Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(126), e)));
    HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(a, Q6_V_vsplat_R(0x7fffff)), Q6_V_vsplat_R(0x800000));
    HVX_Vector base = Q6_Vw_vlsr_VwVw(m, sh);
    HVX_Vector rb =
        Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(sh, one)), one),
                        Q6_V_vand_VV(base, one));
    HVX_Vector sub = Q6_Vw_vlsr_VwVw(Q6_Vw_vadd_VwVw(m, rb), sh);
    sub = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, Q6_V_vsplat_R(101)), sub, zero);
    HVX_Vector h = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, Q6_V_vsplat_R(112)), normal, sub);
    h = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, Q6_V_vsplat_R(142)), Q6_V_vsplat_R(0x7c00), h);
    h = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(a, Q6_V_vsplat_R(0x7f800000)), Q6_V_vsplat_R(0x7e00), h);
    HVX_Vector sign = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v, 16), Q6_V_vsplat_R(0x8000));
    return Q6_V_vor_VV(h, sign);
}
/* Exact binary16 bit expansion, including zero/subnormal/Inf/NaN. QF32
 * multiplication by one can introduce an FP32 ulp during conversion. */
static inline HVX_Vector float_words(HVX_Vector v) {
    HVX_Vector a = Q6_V_vand_VV(v, Q6_V_vsplat_R(0x7fff));
    HVX_Vector n = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(a, 13), Q6_V_vsplat_R(0x38000000));
    HVX_Vector sh = Q6_Vw_vsub_VwVw(Q6_Vuw_vcl0_Vuw(a), Q6_V_vsplat_R(21));
    HVX_Vector m = Q6_Vw_vasl_VwVw(a, sh);
    HVX_Vector sub = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(113), sh), 23),
                                 Q6_Vw_vasl_VwR(Q6_V_vand_VV(m, Q6_V_vsplat_R(1023)), 13));
    n = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x400), a), sub, n);
    HVX_Vector special = Q6_V_vor_VV(Q6_V_vsplat_R(0x7f800000),
                                     Q6_Vw_vasl_VwR(Q6_V_vand_VV(a, Q6_V_vsplat_R(1023)), 13));
    n = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(a, Q6_V_vsplat_R(0x7bff)), special, n);
    n = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(a, Q6_V_vzero()), n, Q6_V_vzero());
    return Q6_V_vor_VV(n, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v, Q6_V_vsplat_R(0x8000)), 16));
}
static inline HVX_Vector splat(float f) {
    union {
        float f;
        uint32_t u;
    } v = {f};
    return Q6_V_vsplat_R(v.u);
}
static inline HVX_Vector activate(HVX_Vector v, uint32_t a) {
    if (a == 1 || a == 3)
        v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vzero());
    if (a == 2)
        v = Q6_Vsf_vmax_VsfVsf(splat(-1), Q6_Vsf_vmin_VsfVsf(splat(1), v));
    if (a == 3)
        v = Q6_Vsf_vmin_VsfVsf(splat(6), v);
    return v;
}

#endif
