/* Integer forms of the reference's float conversions (k_dequantize,
 * k_quantize), bit-equal to their double formulas; test_exact_math.c checks
 * them exhaustively/randomly against those formulas. */
#ifndef PREPARED_EXACT_MATH_H
#define PREPARED_EXACT_MATH_H
#include <stdint.h>

/* k_dequantize: exact = d*sc (an exact double: |d| < 2^31... in practice
 * < 2^17, sc a float), result = the float nearest exact, a tie going to the
 * neighbour of larger magnitude (RNE, then the "away" test of the
 * reference). sc = m24 * 2^e with m24 in [2^23, 2^24), sc > 0.
 * Returns 0 when the result is not a normal float (the caller then uses the
 * reference formula). |d| must be < 2^31. */
static inline int dq_exact(int32_t d, uint32_t m24, int32_t e, uint32_t *bits) {
    if(!d) {*bits=0;return 1;}
    uint32_t sign=d<0?0x80000000u:0u;
    uint64_t P=(uint64_t)(d<0?-(int64_t)d:(int64_t)d)*m24;
    int32_t nb=64-__builtin_clzll(P);
    uint32_t mant;
    if(nb<=24) mant=(uint32_t)(P<<(24-nb));
    else {
        int32_t sh=nb-24;
        uint64_t half=1ull<<(sh-1);
        mant=(uint32_t)(P>>sh);
        if((P&((half<<1)-1))>=half) {
            mant++;
            if(mant==(1u<<24)){mant>>=1;nb++;}
        }
    }
    int32_t ex=e+nb+126; /* value = 1.f * 2^(e+nb-1) */
    if(ex<1 || ex>254) return 0;
    *bits=sign|(uint32_t)ex<<23|(mant&0x7fffffu);
    return 1;
}
/* sc (float bits) as m24 * 2^e; 0 unless sc is a positive normal float. */
static inline int dq_scale(uint32_t b, uint32_t *m24, int32_t *e) {
    uint32_t x=(b>>23)&255u;
    if((b>>31) || !x || x==255u) return 0;
    *m24=(b&0x7fffffu)|0x800000u;*e=(int32_t)x-150;
    return 1;
}

/* k_quantize: v = floor(s*inv + 0.5) + zp clipped to [0, 65535], with s a
 * float and inv = (double)(float)(1/scale) = mi * 2^ei (mi in [2^23,2^24)).
 * s*inv is an exact 48-bit product times a power of two, and adding 0.5 in
 * double is exact wherever it can change the floor, so the integer
 * floor((P + 2^(sh-1)) >> sh) is equal. Returns 0 for s zero-exponent
 * (zero/subnormal) or inf/NaN: the caller uses the reference formula. */
static inline int q_exact(uint32_t sb, uint32_t mi, int32_t ei, int32_t zp, uint16_t *out) {
    uint32_t x=(sb>>23)&255u;
    if(!x || x==255u) return 0;
    uint64_t P=(uint64_t)((sb&0x7fffffu)|0x800000u)*mi;
    int32_t k=(int32_t)x-150+ei;
    int neg=(int)(sb>>31);
    int64_t r;
    if(k>=0) {*out=neg?0:65535;return 1;} /* |s*inv| >= 2^46 */
    int32_t sh=-k;
    if(sh>=63) r=0;
    else {
        int64_t S=neg?-(int64_t)P:(int64_t)P;
        r=(S+((int64_t)1<<(sh-1)))>>sh;
    }
    r+=zp;
    *out=(uint16_t)(r<0?0:r>65535?65535:r);
    return 1;
}
#endif
