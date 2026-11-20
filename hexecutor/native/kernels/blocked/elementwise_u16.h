/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_BLOCKED_ELEMENTWISE_U16_H
#define QD_BLOCKED_ELEMENTWISE_U16_H
/* Exact HVX elementwise kernels over QNN u16 crouton ("CH", 8x4x32, 2 KiB)
 * tensors, for the prepared executor's five u16 elementwise ops:
 *   REQUANT_U16 (variant 71), ADD1111_U16 (59), MUL_U16 (74),
 *   ADD_U16 (14), ADDSUB_U16 (60).
 * Every op is lane-independent, so the CH interleave of adjacent W values
 * does not matter: a 128-byte half block (2 pixels x 32 channels) is
 * processed as 64 u16 lanes and stored under a pixel/channel mask.
 *
 * The scalar reference is int64 arithmetic in prepared_reference.c. This
 * header holds the derivation of its 32-bit lane form: the make_* functions
 * take the raw call arguments exactly as the reference reads them and return
 * 0 unless every int32/uint32 intermediate of the lane form provably cannot
 * overflow for ANY u16 inputs (so the lane form equals the int64 form). The
 * scalar *_lane functions are the lane form in plain C (used by the host
 * exhaustive test); the HVX code in elementwise_u16.c mirrors them op for op.
 */
#include "linearclip_u8_u16.h"
#include <stdint.h>

typedef QdBlockedU16Out QdBlockedU16;
enum { QD_EW16_REQUANT=1, QD_EW16_ADD1111=2, QD_EW16_MUL=3, QD_EW16_ADD=4, QD_EW16_ADDSUB=5,
       QD_EW16_ADDSUBU=6 /* ADDSUB whose sum range is non-negative and needs the full uint32 */ };

typedef struct {
    uint32_t kind;
    uint32_t shift;       /* final shift, 1..31 */
    uint32_t k0, k1;      /* multipliers (uint32; two's complement where signed) */
    int32_t  c;           /* REQUANT zin; ADD1111 c=s*k1+k3 (set per run); ADD round; ADDSUB k3; MUL K */
    int32_t  c2;          /* REQUANT zout; MUL p2 */
    int32_t  hi;          /* MUL hi */
    uint32_t lo;          /* MUL lo */
} QdEw16Params;

#define QD_EW16_AI (-0x7fffffff-1)

/* ---- derivation helpers (portable C, used by the validator and host test) */
static inline int qd_ew16_fits_i32(int64_t v) { return v>=-2147483647ll-1 && v<=2147483647ll; }

/* REQUANT_U16. Reference: clip((((v-zin)*m + 2^(s-1)) >> s) + zout, 0, 65535),
 * v in [0,65535], m=r3&0xffff, s=15-(r3>>24), zin=r4&0xffff, zout=r4>>16.
 * Lane form: p=(v-zin)*m+2^(s-1) in int32 (checked at both extremes of v,
 * m>=0 so p is monotone in v), q=(p>>s)+zout in int32, saturating pack. */
static inline int qd_ew16_make_requant(int64_t r3,int64_t r4,QdEw16Params *p) {
    if ((uint64_t)r3>=(1ull<<40)||(uint64_t)r4>=(1ull<<40)) return 0;
    int64_t m=r3&0xffff,s=15-(r3>>24),zin=r4&0xffff,zout=r4>>16;
    if (s<1||s>31) return 0;
    int64_t rnd=1ll<<(s-1),pmin=(0-zin)*m+rnd,pmax=(65535-zin)*m+rnd;
    if (!qd_ew16_fits_i32(pmin)||!qd_ew16_fits_i32(pmax)) return 0;
    if (!qd_ew16_fits_i32((pmin>>s)+zout)||!qd_ew16_fits_i32((pmax>>s)+zout)) return 0;
    p->kind=QD_EW16_REQUANT;p->shift=(uint32_t)s;p->k0=(uint32_t)m;p->k1=0;
    p->c=(int32_t)zin;p->c2=(int32_t)zout;p->hi=0;p->lo=0;return 1;
}
static inline uint16_t qd_ew16_requant_lane(const QdEw16Params *p,uint16_t v) {
    int32_t d=(int32_t)v-p->c;
    int32_t pr=(int32_t)((uint32_t)d*p->k0+(1u<<(p->shift-1)));
    int32_t q=(pr>>p->shift)+p->c2;
    return q<0?0:q>65535?65535:(uint16_t)q;
}

/* ADD1111_U16. Reference: clip((x*k0 + s*k1 + k3) >> k2, 0, 65535) with
 * k0=r4,k1=r5 (<2^40), k2=r6+1, k3=2*s32(r7)+2^r6, s the scalar u16 operand.
 * Lane form: c=s*k1+k3 (int32, set per run), t=(x*k0+c)>>k2 in int32. All four
 * corners (x,s in {0,65535}) and c itself are checked to fit int32. */
static inline int qd_ew16_make_add1111(int64_t r4,int64_t r5,int64_t r6,int64_t r7,QdEw16Params *p) {
    if ((uint64_t)r4>=(1ull<<40)||(uint64_t)r5>=(1ull<<40)||(uint64_t)r6>=(1ull<<40)) return 0;
    if (r6+1>31) return 0;
    int64_t k3=2*(int64_t)(int32_t)(uint32_t)(uint64_t)r7+(1ll<<r6);
    for (int xs=0;xs<2;xs++) for (int ss=0;ss<2;ss++)
        if (!qd_ew16_fits_i32((xs?65535*r4:0)+(ss?65535*r5:0)+k3)) return 0;
    if (!qd_ew16_fits_i32(k3)||!qd_ew16_fits_i32(65535*r5+k3)) return 0;
    p->kind=QD_EW16_ADD1111;p->shift=(uint32_t)(r6+1);p->k0=(uint32_t)r4;p->k1=(uint32_t)r5;
    p->c=(int32_t)k3; /* scalar term s*k1 is added per run by qd_ew16_add1111_set_scalar */
    p->c2=0;p->hi=0;p->lo=0;return 1;
}
/* k3 is kept in p->c until a scalar is applied; this returns the lane constant. */
static inline int32_t qd_ew16_add1111_const(const QdEw16Params *p,uint32_t s) {
    return (int32_t)((uint32_t)p->c+s*p->k1);
}
static inline uint16_t qd_ew16_add1111_lane(const QdEw16Params *p,int32_t cc,uint16_t x) {
    int32_t t=(int32_t)((uint32_t)x*p->k0+(uint32_t)cc)>>p->shift;
    return t<0?0:t>65535?65535:(uint16_t)t;
}

/* ADD_U16. Reference: clip((a*k0 + b*k1 + 2^k2) >> (k2+1), 0, 65535),
 * k0=r4,k1=r5,k2=r6. Lane form: unsigned 32-bit sum, checked to be <2^32,
 * logical shift (k2+1<=31 so the result is <2^31), signed saturating pack. */
static inline int qd_ew16_make_add(int64_t r4,int64_t r5,int64_t r6,QdEw16Params *p) {
    if ((uint64_t)r4>=(1ull<<40)||(uint64_t)r5>=(1ull<<40)||(uint64_t)r6>=(1ull<<40)) return 0;
    if (r6+1>31) return 0;
    if (65535*(r4+r5)+(1ll<<r6)>4294967295ll) return 0;
    p->kind=QD_EW16_ADD;p->shift=(uint32_t)(r6+1);p->k0=(uint32_t)r4;p->k1=(uint32_t)r5;
    p->c=(int32_t)(1u<<r6);p->c2=0;p->hi=0;p->lo=0;return 1;
}
static inline uint16_t qd_ew16_add_lane(const QdEw16Params *p,uint16_t a,uint16_t b) {
    uint32_t t=((uint32_t)a*p->k0+(uint32_t)b*p->k1+(uint32_t)p->c)>>p->shift;
    return t>65535?65535:(uint16_t)t;
}

/* ADDSUB_U16. Reference: clip((a*k0 - b*k1 + k3) >> k2, 0, 65535) with
 * k0=r4,k1=r5,k2=r6+1,k3=zp*2^k2+2^r6+1 (zp the output tensor zero point).
 * Lane form: 32-bit sum over [k3-65535*k1, k3+65535*k0] (signed or unsigned, see make). */
static inline int qd_ew16_make_addsub(int64_t r4,int64_t r5,int64_t r6,int32_t zp,QdEw16Params *p) {
    if ((uint64_t)r4>=(1ull<<40)||(uint64_t)r5>=(1ull<<40)||(uint64_t)r6>=(1ull<<40)) return 0;
    if (r6+1>31) return 0;
    int64_t k3=(int64_t)zp*(1ll<<(r6+1))+(1ll<<r6)+1,lo=k3-65535*r5,hi=k3+65535*r4;
    /* The sum is linear in (a,b), so lo/hi are its exact extremes. Signed
     * int32 if it fits, else unsigned uint32 (non-negative, logical shift). */
    uint32_t kind;
    if (qd_ew16_fits_i32(lo)&&qd_ew16_fits_i32(hi)) kind=QD_EW16_ADDSUB;
    else if (lo>=0&&hi<=4294967295ll) kind=QD_EW16_ADDSUBU;
    else return 0;
    p->kind=kind;p->shift=(uint32_t)(r6+1);p->k0=(uint32_t)r4;p->k1=(uint32_t)r5;
    p->c=(int32_t)(uint32_t)k3;p->c2=0;p->hi=0;p->lo=0;return 1;
}
static inline uint16_t qd_ew16_addsub_lane(const QdEw16Params *p,uint16_t a,uint16_t b) {
    uint32_t t=(uint32_t)a*p->k0-(uint32_t)b*p->k1+(uint32_t)p->c;
    if (p->kind==QD_EW16_ADDSUBU) { t>>=p->shift; return t>65535?65535:(uint16_t)t; }
    int32_t v=(int32_t)t>>p->shift;
    return v<0?0:v>65535?65535:(uint16_t)v;
}

/* MUL_U16 (MUL1111 shares it with a splat b). Reference, zb=za=0:
 *   D=s32(a*b+K); lo=P1&0xffff; hi=s32(P1)>>16; e=(D*lo)>>16;
 *   q=clip32((2e+2*D*hi)>>16); s=clip32(q+s32(P2)); out=clip((s+4)>>3,0,65535).
 * With D=65536*Dh+Dl (Dh=D>>16 signed, Dl=D&0xffff):
 *   e   = Dh*lo + ((Dl*lo)>>16)                        (fits int32: |e|<2^31)
 *   q   = 2*hi*Dh + (e>>15) + (((e&0x7fff) + hi*Dl) >> 15)
 *   (no clip: |q| <= 2^16*(1+|hi|) < 2^31 for |hi|<=16384)
 *   s   = sat32(q+P2);  out = pack_sat(((s>>2)+1)>>1)
 * since (s+4)>>3 = floor((floor(s/4)+1)/2). Every step is exact (see above);
 * the host test additionally checks it exhaustively. */
static inline int qd_ew16_make_mul(int64_t P1,int64_t zb,int64_t za,int64_t K,int64_t P2,QdEw16Params *p) {
    if ((zb&0xffff)||(za&0xffff)) return 0;
    int32_t hi=(int32_t)(uint32_t)(uint64_t)P1>>16;
    if (hi>16384||hi<-16384) return 0;
    p->kind=QD_EW16_MUL;p->shift=0;p->k0=0;p->k1=0;
    p->c=(int32_t)(uint32_t)(uint64_t)K;p->c2=(int32_t)(uint32_t)(uint64_t)P2;
    p->hi=hi;p->lo=(uint32_t)(P1&0xffff);return 1;
}
/* Second stage as a function of the 32-bit D (the host test runs it over all 2^32 D). */
static inline uint16_t qd_ew16_mul_stage2(const QdEw16Params *p,uint32_t D) {
    int32_t Dh=(int32_t)D>>16;uint32_t Dl=D&0xffffu;
    int32_t e=(int32_t)((uint32_t)Dh*p->lo+((Dl*p->lo)>>16));
    int32_t q=(int32_t)((uint32_t)Dh*(uint32_t)(2*p->hi)+(uint32_t)(e>>15)+
              (uint32_t)(((int32_t)(((uint32_t)e&0x7fffu)+(uint32_t)p->hi*Dl))>>15));
    int64_t s64=(int64_t)q+p->c2;
    int32_t s=s64>2147483647ll?2147483647:s64<-2147483647ll-1?(int32_t)(-2147483647-1):(int32_t)s64;
    int32_t r=((s>>2)+1)>>1;
    return r<0?0:r>65535?65535:(uint16_t)r;
}
static inline uint16_t qd_ew16_mul_lane(const QdEw16Params *p,uint16_t a,uint16_t b) {
    return qd_ew16_mul_stage2(p,(uint32_t)a*(uint32_t)b+(uint32_t)p->c);
}

/* ---- 16-bit "fast forms" (HVX, whole-block path). Same results as the lane
 * forms above, proven for the parameter ranges their make functions accept;
 * outside them the whole path falls back to the 32-bit lane forms.
 *
 * MUL fast form. With D = a*b+K (wrapped int32), the reference chain
 *   e=(D*lo)>>16, q=(2e+2*D*hi)>>16, s=sat32(q+P2), out=clip((s+4)>>3)
 * equals out = clip(floor((D*P1 + (P2+4)*2^31) / 2^34)) (nested floors merge;
 * the q and s clips cannot change a result that is clipped to u16 anyway)
 * for 0 <= hi <= 16384. Put D' = D + 2^31 (mod 2^32; Dh' = D'>>16 and Dl =
 * D'&0xffff are unsigned, Dh = Dh' - 2^15), Qa = (P2+4)>>1, Qb = ((P2+4)&1)<<15:
 *   t1 = (Dl*lo)>>16
 *   X  = Dh'*lo + Dl*hi + t1 + (Qb - 2^15*lo)           (== Dh*lo + Dl*hi + t1 + Qb)
 *   Y  = Dh'*hi + (Qa - 2^15*hi) + (X>>16)              (== Dh*hi + Qa + (X>>16))
 *   out = clip(Y>>2, 0, 65535)
 * with X and Y proven to fit int32 over the whole (Dh, Dl) range. */
typedef struct { uint32_t kk,hi,lo,cx,cy; } QdEw16MulFast;
static inline int qd_ew16_mul_fast_make(const QdEw16Params *p,QdEw16MulFast *f) {
    if (p->kind!=QD_EW16_MUL||p->hi<0||p->hi>16384) return 0;
    int64_t H=p->hi,lo=p->lo,P2=p->c2+4ll,Qa=P2>>1,Qb=(P2&1)<<15;
    int64_t xmin=-32768*lo+Qb,xmax=32767*lo+65535*H+((65535*lo)>>16)+Qb;
    if (!qd_ew16_fits_i32(xmin)||!qd_ew16_fits_i32(xmax)) return 0;
    int64_t ymin=-32768*H+Qa+(xmin>>16),ymax=32767*H+Qa+(xmax>>16);
    if (!qd_ew16_fits_i32(ymin)||!qd_ew16_fits_i32(ymax)) return 0;
    f->kk=(uint32_t)p->c+0x80000000u;f->hi=(uint32_t)H;f->lo=(uint32_t)lo;
    f->cx=(uint32_t)(Qb-32768*lo);f->cy=(uint32_t)(Qa-32768*H);return 1;
}
/* D' = D + 2^31 (mod 2^32), the word the HVX code forms as a*b + kk. */
static inline uint16_t qd_ew16_mul_fast_stage2(const QdEw16MulFast *f,uint32_t dp) {
    uint32_t dh=dp>>16,dl=dp&0xffffu,t1=(dl*f->lo)>>16;
    int32_t x=(int32_t)(dh*f->lo+dl*f->hi+t1+f->cx);
    int32_t y=(int32_t)(dh*f->hi+f->cy+(uint32_t)(x>>16));
    int32_t r=y>>2;
    return r<0?0:r>65535?65535:(uint16_t)r;
}

/* Applies the op over the logical region (dims/origin) of a, b and out.
 * b is NULL for REQUANT and ADD1111. bscalar>=0 supplies b as a splat u16
 * (MUL1111; b NULL); for ADD1111 the scalar is passed in bscalar (0..65535).
 * Same-origin rules as the other blocked elementwise glue: dims equal,
 * origin[2]%4 and origin[3]%32 equal across tensors. Absent blocks read as
 * zero and are not stored. Returns 0 on bad arguments. */
int qd_blocked_ew16_apply(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *out,
                          const QdEw16Params *p,int32_t bscalar);
/* Whole-block form: a, b, out are all whole 8x4x32 blocks (origins and dims
 * multiples of 8, 4, 32 in H, W, C) with equal dims. Returns 1 if so. The
 * apply processes blocks i = part, part+parts, ... of the region in (B,H,W,C)
 * order, 16 full vectors each (absent inputs read zero, absent outputs are
 * not stored), using the 16-bit fast forms where the parameters allow and the
 * 32-bit lane forms otherwise. Returns 0 on bad arguments or a non-whole region. */
int qd_blocked_ew16_whole(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *out);
int qd_blocked_ew16_apply_whole(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *out,
                                const QdEw16Params *p,int32_t bscalar,uint32_t part,uint32_t parts);
#endif
