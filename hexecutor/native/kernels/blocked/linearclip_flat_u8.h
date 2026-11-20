/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_LINEARCLIP_FLAT_U8_H
#define QD_LINEARCLIP_FLAT_U8_H
/* Flat (dense, direct) u8 -> u8 LinearClip, the LINEARCLIP/69 profile of
 * prepared_reference.c k_linearclip:
 *   y = clip(((((x*m + 128) >> 8) + 32) >> 6), 0, 255),  0 <= m <= 65535.
 * x*m <= 255*65535 fits in 32 bits, every shift is of a non-negative value,
 * and the only clip that can act is the upper one. The HVX form widens 128
 * bytes to four word vectors, multiplies by the low halfword of a splat
 * (vmpyie), shifts/adds exactly as above and packs back in order; the scalar
 * loop (host build, and the tail) is the same expression. src and dst may be
 * identical (in place) or disjoint; no other overlap is allowed. */
#include <stddef.h>
#include <stdint.h>
#if defined(__HVX__) && __HVX_LENGTH__==128
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static inline uint8_t qd_linearclip_flat_u8_lane(uint8_t x,uint32_t m) {
    uint32_t v=(((x*m+128u)>>8)+32u)>>6;
    return (uint8_t)(v>255u?255u:v);
}

static inline void qd_linearclip_flat_u8(const uint8_t *src,uint8_t *dst,size_t n,uint32_t m) {
    size_t i=0;
#if defined(__HVX__) && __HVX_LENGTH__==128
    const HVX_Vector mv=Q6_V_vsplat_R((int32_t)m),c128=Q6_V_vsplat_R(128),c32=Q6_V_vsplat_R(32),cmax=Q6_V_vsplat_R(255);
    for(;i+128<=n;i+=128) {
        HVX_Vector x=*(const HVX_UVector *)(src+i);
        HVX_VectorPair h=Q6_Wuh_vunpack_Vub(x);
        HVX_VectorPair w0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(h)),w1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(h));
        HVX_Vector w[4]={Q6_V_lo_W(w0),Q6_V_hi_W(w0),Q6_V_lo_W(w1),Q6_V_hi_W(w1)};
        for(int k=0;k<4;k++) {
            HVX_Vector a=Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(w[k],mv),c128);
            a=Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(a,8),c32);
            w[k]=Q6_Vw_vmin_VwVw(Q6_Vw_vasr_VwR(a,6),cmax);
        }
        HVX_Vector h0=Q6_Vh_vpack_VwVw_sat(w[1],w[0]),h1=Q6_Vh_vpack_VwVw_sat(w[3],w[2]);
        *(HVX_UVector *)(dst+i)=Q6_Vub_vpack_VhVh_sat(h1,h0);
    }
#endif
    for(;i<n;i++)dst[i]=qd_linearclip_flat_u8_lane(src[i],m);
}
#endif
