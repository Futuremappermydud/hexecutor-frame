/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "elementwise_u16.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <limits.h>

/* Word-lane forms of the ops in elementwise_u16.h; each mirrors the
 * qd_ew16_*_lane scalar function operation for operation. A "word vector" is
 * 32 zero-extended u16 lanes. */
typedef struct { HVX_Vector lo,hi; int has_hi; } Mc; /* x * uint32 constant, mod 2^32 */
typedef struct {
    uint32_t kind,shift;
    Mc m0,m1,mlo,m2hi,mhi;
    HVX_Vector c,c2,ffff,x7fff,one,rnd;
} Ctx;

static Mc mc_make(uint32_t m) {
    Mc r;r.lo=Q6_V_vsplat_R(m&65535u);r.hi=Q6_V_vsplat_R(m>>16);r.has_hi=(m>>16)!=0;return r;
}
static inline HVX_Vector mulc(HVX_Vector x,const Mc *m) {
    HVX_Vector r=Q6_Vw_vmpyie_VwVuh(x,m->lo);
    if(m->has_hi)r=Q6_Vw_vadd_VwVw(r,Q6_Vw_vasl_VwR(Q6_Vw_vmpyie_VwVuh(x,m->hi),16));
    return r;
}
static void ctx_make(Ctx *c,const QdEw16Params *p,int32_t bscalar) {
    c->kind=p->kind;c->shift=p->shift;
    c->m0=mc_make(p->k0);c->m1=mc_make(p->k1);
    c->mlo=mc_make(p->lo);c->m2hi=mc_make((uint32_t)(2*p->hi));c->mhi=mc_make((uint32_t)p->hi);
    c->c=Q6_V_vsplat_R((uint32_t)p->c);c->c2=Q6_V_vsplat_R((uint32_t)p->c2);
    if(p->kind==QD_EW16_ADD1111)c->c=Q6_V_vsplat_R((uint32_t)qd_ew16_add1111_const(p,(uint32_t)bscalar));
    c->ffff=Q6_V_vsplat_R(0xffffu);c->x7fff=Q6_V_vsplat_R(0x7fffu);c->one=Q6_V_vsplat_R(1u);
    c->rnd=Q6_V_vsplat_R(p->kind==QD_EW16_REQUANT?(1u<<(p->shift-1)):0u);
}
static inline HVX_Vector lane_w(const Ctx *c,HVX_Vector a,HVX_Vector b) {
    switch(c->kind) {
    case QD_EW16_REQUANT: {
        HVX_Vector d=Q6_Vw_vsub_VwVw(a,c->c);
        HVX_Vector pr=Q6_Vw_vadd_VwVw(mulc(d,&c->m0),c->rnd);
        return Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(pr,(int)c->shift),c->c2);
    }
    case QD_EW16_ADD1111:
        return Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(mulc(a,&c->m0),c->c),(int)c->shift);
    case QD_EW16_ADD: {
        HVX_Vector s=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(mulc(a,&c->m0),mulc(b,&c->m1)),c->c);
        return Q6_Vuw_vlsr_VuwR(s,(int)c->shift);
    }
    case QD_EW16_ADDSUB: {
        HVX_Vector s=Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(mulc(a,&c->m0),mulc(b,&c->m1)),c->c);
        return Q6_Vw_vasr_VwR(s,(int)c->shift);
    }
    case QD_EW16_ADDSUBU: {
        HVX_Vector s=Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(mulc(a,&c->m0),mulc(b,&c->m1)),c->c);
        return Q6_Vuw_vlsr_VuwR(s,(int)c->shift);
    }
    default: { /* QD_EW16_MUL */
        HVX_Vector D=Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(a,b),c->c);
        HVX_Vector Dh=Q6_Vw_vasr_VwR(D,16),Dl=Q6_V_vand_VV(D,c->ffff);
        HVX_Vector e=Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(Dh,c->mlo.lo),
                                     Q6_Vuw_vlsr_VuwR(Q6_Vw_vmpyie_VwVuh(Dl,c->mlo.lo),16));
        HVX_Vector t=Q6_Vw_vadd_VwVw(Q6_V_vand_VV(e,c->x7fff),mulc(Dl,&c->mhi));
        HVX_Vector q=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(mulc(Dh,&c->m2hi),Q6_Vw_vasr_VwR(e,15)),
                                     Q6_Vw_vasr_VwR(t,15));
        HVX_Vector s=Q6_Vw_vadd_VwVw_sat(q,c->c2);
        return Q6_Vw_vavg_VwVw(Q6_Vw_vasr_VwR(s,2),c->one);
    }
    }
}
/* 64 u16 lanes in, 64 out. */
static inline HVX_Vector lane_h(const Ctx *c,HVX_Vector a,HVX_Vector b) {
    HVX_VectorPair wa=Q6_Wuw_vunpack_Vuh(a),wb=Q6_Wuw_vunpack_Vuh(b);
    HVX_Vector r0=lane_w(c,Q6_V_lo_W(wa),Q6_V_lo_W(wb)),r1=lane_w(c,Q6_V_hi_W(wa),Q6_V_hi_W(wb));
    return Q6_Vuh_vpack_VwVw_sat(r1,r0);
}


/* ---- 16-bit fast forms for whole blocks (see elementwise_u16.h for MUL).
 * One-product ops (REQUANT, ADD1111):  y = clip_u16((x*k0 + A) >> sh)
 * Two-product ops (ADD, ADDSUB):       y = clip_u16((a*k0 + b'*k1 + A) >> sh),
 *                                      b' = b (ADD) or 65535-b (ADDSUB)
 * The sum is formed in 32-bit word lanes by uh x uh -> uw products accumulated
 * onto A (mod 2^32), then one narrowing vasr (shift mod 16, so sh in 1..15)
 * with unsigned saturation: the signed form reads the words as int32 (floor
 * shift, negative sums clip to 0), the unsigned form as uint32. Both equal the
 * exact int64 expression when the true sum lies in int32 (resp. uint32),
 * which fx_set proves from the constants (smin/smax are its exact extremes). */
enum { FX_NONE=0, FX_1, FX_2, FX_2INV, FX_MUL };
typedef struct {
    int mode,uns;uint32_t sh,rk0,rk1,r1,r2;
    HVX_Vector acc,kk,cx,cy;
} Fx;

static int fx_set(Fx *f,int mode,int uns,uint32_t k0,uint32_t k1,uint32_t sh,int64_t A,int64_t smin,int64_t smax) {
    if (k0>65535u||(mode!=FX_1&&k1>65535u)||sh<1||sh>15) return 0;
    if (uns?(smin<0||smax>4294967295ll):(!qd_ew16_fits_i32(smin)||!qd_ew16_fits_i32(smax))) return 0;
    f->mode=mode;f->uns=uns;f->sh=sh;f->rk0=k0|k0<<16;f->rk1=k1|k1<<16;
    f->acc=Q6_V_vsplat_R((uint32_t)A);
    return 1;
}
static void fx_make(Fx *f,const QdEw16Params *p,int32_t bscalar) {
    f->mode=FX_NONE;
    switch (p->kind) {
    case QD_EW16_REQUANT: {
        uint32_t s=p->shift;int64_t m=p->k0,zin=p->c,zout=p->c2;
        if (s<1||s>15||zin<0||zin>65535||zout<0||zout>65535) return;
        int64_t A=(1ll<<(s-1))-zin*m+(zout<<s);
        fx_set(f,FX_1,0,p->k0,0,s,A,A,65535*m+A);
        return;
    }
    case QD_EW16_ADD1111: {
        int64_t C=(int32_t)qd_ew16_add1111_const(p,(uint32_t)bscalar);
        fx_set(f,FX_1,0,p->k0,0,p->shift,C,C,65535ll*p->k0+C);
        return;
    }
    case QD_EW16_ADD: {
        int64_t A=(int64_t)(uint32_t)p->c;
        fx_set(f,FX_2,1,p->k0,p->k1,p->shift,A,A,65535ll*((int64_t)p->k0+p->k1)+A);
        return;
    }
    case QD_EW16_ADDSUB: case QD_EW16_ADDSUBU: {
        int u=p->kind==QD_EW16_ADDSUBU;
        int64_t k3=u?(int64_t)(uint32_t)p->c:(int64_t)(int32_t)p->c,Kc=k3-65535ll*p->k1;
        fx_set(f,FX_2INV,u,p->k0,p->k1,p->shift,Kc,Kc,65535ll*((int64_t)p->k0+p->k1)+Kc);
        return;
    }
    default: {
        QdEw16MulFast m;
        if (!qd_ew16_mul_fast_make(p,&m)) return;
        f->mode=FX_MUL;f->r1=m.lo|m.hi<<16;f->r2=m.hi|m.lo<<16;
        f->kk=Q6_V_vsplat_R(m.kk);f->cx=Q6_V_vsplat_R(m.cx);f->cy=Q6_V_vsplat_R(m.cy);
        return;
    }
    }
}
#define FX_INL static inline __attribute__((always_inline))
/* One small non-inlined function per form: the constants are copied into
 * locals first so the 16 unrolled vectors keep them in registers. Sources
 * advance by stride sa/sb vectors per step (0 = one constant vector). */
#define FX_NARROW(w,UNS) ((UNS)?Q6_Vuh_vasr_VuwVuwR_sat(Q6_V_hi_W(w),Q6_V_lo_W(w),sh):Q6_Vuh_vasr_VwVwR_sat(Q6_V_hi_W(w),Q6_V_lo_W(w),sh))
#define FX_BLK1(NAME,UNS) \
static __attribute__((noinline)) void NAME(const Fx *f,const HVX_Vector *pa,int sa,HVX_Vector *po) { \
    const HVX_Vector acc=f->acc;const uint32_t rk=f->rk0,sh=f->sh; \
    const HVX_VectorPair a0=Q6_W_vcombine_VV(acc,acc); \
    _Pragma("clang loop unroll(full)") \
    for(int v=0;v<16;v++){HVX_VectorPair w=Q6_Wuw_vmpyacc_WuwVuhRuh(a0,pa[v*sa],rk);po[v]=FX_NARROW(w,UNS);} }
#define FX_BLK2(NAME,INV,UNS) \
static __attribute__((noinline)) void NAME(const Fx *f,const HVX_Vector *pa,int sa,const HVX_Vector *pb,int sb,HVX_Vector *po) { \
    const HVX_Vector acc=f->acc;const uint32_t rk0=f->rk0,rk1=f->rk1,sh=f->sh; \
    const HVX_VectorPair a0=Q6_W_vcombine_VV(acc,acc); \
    _Pragma("clang loop unroll(full)") \
    for(int v=0;v<16;v++){HVX_Vector b=pb[v*sb];HVX_VectorPair w=Q6_Wuw_vmpyacc_WuwVuhRuh(a0,pa[v*sa],rk0); \
        if(INV)b=Q6_V_vnot_V(b); w=Q6_Wuw_vmpyacc_WuwVuhRuh(w,b,rk1);po[v]=FX_NARROW(w,UNS);} }
FX_BLK1(blk1_s,0)
FX_BLK2(blk2_u,0,1) FX_BLK2(blk2i_s,1,0) FX_BLK2(blk2i_u,1,1)
/* MUL: stage on one word vector of D' (32 lanes). */
FX_INL HVX_Vector fx_mul_words(HVX_Vector d,HVX_Vector cx,HVX_Vector cy,uint32_t r1,uint32_t r2) {
    HVX_VectorPair pa=Q6_Wuw_vmpy_VuhRuh(d,r1);
    HVX_VectorPair pb=Q6_Wuw_vmpy_VuhRuh(d,r2);
    HVX_Vector t1=Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(pa),16);
    HVX_Vector x=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(pb),cx),Q6_Vw_vadd_VwVw(Q6_V_hi_W(pb),t1));
    return Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_V_hi_W(pa),cy),Q6_Vw_vasr_VwR(x,16));
}
static __attribute__((noinline)) void blk_mul(const Fx *f,const HVX_Vector *pa,int sa,const HVX_Vector *pb,int sb,HVX_Vector *po) {
    const HVX_VectorPair kk=Q6_W_vcombine_VV(f->kk,f->kk);
    const HVX_Vector cx=f->cx,cy=f->cy;
    const uint32_t r1=f->r1,r2=f->r2;
    #pragma clang loop unroll(full)
    for (int v=0;v<16;v++) {
        HVX_VectorPair pr=Q6_Wuw_vmpyacc_WuwVuhVuh(kk,pa[v*sa],pb[v*sb]);
        HVX_Vector ye=fx_mul_words(Q6_V_lo_W(pr),cx,cy,r1,r2),yo=fx_mul_words(Q6_V_hi_W(pr),cx,cy,r1,r2);
        po[v]=Q6_Vuh_vasr_VwVwR_sat(yo,ye,2);
    }
}
static void fx_block(const Fx *f,const HVX_Vector *pa,int sa,const HVX_Vector *pb,int sb,HVX_Vector *po) {
    switch (f->mode) {
    case FX_1: blk1_s(f,pa,sa,po);break;
    case FX_2: blk2_u(f,pa,sa,pb,sb,po);break;
    case FX_2INV: if (f->uns) blk2i_u(f,pa,sa,pb,sb,po);else blk2i_s(f,pa,sa,pb,sb,po);break;
    default: blk_mul(f,pa,sa,pb,sb,po);break;
    }
}

static int valid4(const QdBlockedU16 *t) {
    if(!t||!t->pools||!t->blocks||t->meta[1]>INT32_MAX-7||t->meta[2]>INT32_MAX-3||t->meta[3]>INT32_MAX-31)return 0;
    for(int i=0;i<4;i++)if(t->meta[i]<=0||t->dims[i]<0||t->origin[i]<0||(int64_t)t->origin[i]+t->dims[i]>t->meta[i])return 0;
    /* slots = meta0 * ceil(meta1/8) * ceil(meta2/4) * ceil(meta3/32) must fit u32 (no 64-bit divides: each step <2^61) */
    uint64_t slots=(uint32_t)t->meta[0];
    slots*=(uint32_t)(t->meta[1]+7)>>3;if(slots>UINT32_MAX)return 0;
    slots*=(uint32_t)(t->meta[2]+3)>>2;if(slots>UINT32_MAX)return 0;
    slots*=(uint32_t)(t->meta[3]+31)>>5;
    return slots<=UINT32_MAX;
}
/* Row `h` (logical, 0-based) of the 128-byte-pair block holding pixel W
 * (W%4==0, tensor coordinates incl. origin) and depth block db. */
static uint8_t *row_ptr(const QdBlockedU16 *t,int b,int h,int W,int db) {
    int H=h+t->origin[1],B=b+t->origin[0],nbh=(t->meta[1]+7)/8,nbw=(t->meta[2]+3)/4,nbd=(t->meta[3]+31)/32;
    int64_t ix=(((int64_t)B*nbh+H/8)*nbw+W/4)*nbd+t->slice+db;
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u)+(H%8)*256;
}
static HVX_VectorPred prefix(unsigned n) {return n>=128?Q6_Q_not_Q(Q6_Q_vsetq_R(0)):Q6_Q_vsetq_R(n);}
static HVX_VectorPred store_mask(int first,int last,int even_valid,int odd_valid,HVX_VectorPred even) {
    HVX_VectorPred ch=Q6_Q_and_QQn(prefix((unsigned)last*4),prefix((unsigned)first*4));
    HVX_VectorPred q=Q6_Q_and_QQ(ch,even_valid?even:Q6_Q_vsetq_R(0));
    if(odd_valid)q=Q6_Q_or_QQ(q,Q6_Q_and_QQ(ch,Q6_Q_not_Q(even)));
    return q;
}


/* ---- whole-block entry points */
static int whole4(const QdBlockedU16 *t) {
    return !(t->origin[1]&7)&&!(t->origin[2]&3)&&!(t->origin[3]&31)&&!(t->dims[1]&7)&&!(t->dims[2]&3)&&!(t->dims[3]&31);
}
static int whole_nv(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *o) {
    for(int i=0;i<4;i++)if(a->dims[i]!=o->dims[i]||(b&&a->dims[i]!=b->dims[i]))return 0;
    return whole4(a)&&whole4(o)&&(!b||whole4(b));
}
int qd_blocked_ew16_whole(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *o) {
    if(!valid4(a)||!valid4(o)||(b&&!valid4(b)))return 0;
    return whole_nv(a,b,o);
}
typedef struct { int64_t base,sB,sH,sW; } Bix;
static Bix bix_make(const QdBlockedU16 *t) {
    int64_t nbh=(t->meta[1]+7)>>3,nbw=(t->meta[2]+3)>>2,nbd=(t->meta[3]+31)>>5;
    Bix x;x.sW=nbd;x.sH=nbw*nbd;x.sB=nbh*nbw*nbd;
    x.base=(int64_t)t->origin[0]*x.sB+(int64_t)(t->origin[1]>>3)*x.sH+(int64_t)(t->origin[2]>>2)*x.sW+t->slice+(t->origin[3]>>5);
    return x;
}
static const uint8_t *bix_ptr(const QdBlockedU16 *t,int64_t ix) {
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u);
}
int qd_blocked_ew16_apply_whole(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *o,
                                const QdEw16Params *p,int32_t bscalar,uint32_t part,uint32_t parts) {
    if(!p||!parts||part>=parts||!valid4(a)||!valid4(o)||(b&&!valid4(b))||!whole_nv(a,b,o))return 0;
    int two=p->kind==QD_EW16_MUL||p->kind==QD_EW16_ADD||p->kind==QD_EW16_ADDSUB||p->kind==QD_EW16_ADDSUBU;
    int splat_b=two&&!b;
    if(splat_b&&(p->kind!=QD_EW16_MUL||bscalar<0||bscalar>65535))return 0;
    if(p->kind==QD_EW16_ADD1111&&(bscalar<0||bscalar>65535))return 0;
    if(p->kind<QD_EW16_REQUANT||p->kind>QD_EW16_ADDSUBU)return 0;
    if(p->kind!=QD_EW16_MUL&&(p->shift<1||p->shift>31))return 0;
    for(int i=0;i<4;i++)if(!a->dims[i])return 1;
    Fx fx;fx_make(&fx,p,bscalar);
    Ctx cx;ctx_make(&cx,p,bscalar);
    HVX_Vector one_a[1],one_b[1];
    one_a[0]=Q6_V_vzero();one_b[0]=splat_b?Q6_V_vsplat_R((uint32_t)bscalar*0x10001u):Q6_V_vzero();
    Bix ia=bix_make(a),ib,io=bix_make(o);
    if(b)ib=bix_make(b);
    const int nB=a->dims[0],nH=a->dims[1]>>3,nW=a->dims[2]>>2,nD=a->dims[3]>>5;
    /* blocks part, part+parts, ... of the (B,H,W,D) enumeration */
    uint32_t ph=0;
    for(int B=0;B<nB;B++)for(int H=0;H<nH;H++)for(int W=0;W<nW;W++) {
        const int64_t ra=ia.base+B*ia.sB+H*ia.sH+W*ia.sW,ro=io.base+B*io.sB+H*io.sH+W*io.sW,
                      rb=b?ib.base+B*ib.sB+H*ib.sH+W*ib.sW:0;
        for(int D=0;D<nD;D++) {
            const uint32_t cur=ph;ph=(ph+1==parts)?0:ph+1;
            if(cur!=part)continue;
            HVX_Vector *po=(HVX_Vector *)bix_ptr(o,ro+D);
            if(!po)continue;
            const HVX_Vector *pa=(const HVX_Vector *)bix_ptr(a,ra+D);
            const HVX_Vector *pb=b?(const HVX_Vector *)bix_ptr(b,rb+D):0;
            int sa=pa?1:0,sb=0;
            if(!pa)pa=one_a;
            if(b&&pb)sb=1;else pb=one_b;
            if(fx.mode!=FX_NONE)fx_block(&fx,pa,sa,pb,sb,po);
            else for(int v=0;v<16;v++)po[v]=lane_h(&cx,pa[v*sa],pb[v*sb]);
        }
    }
    return 1;
}

int qd_blocked_ew16_apply(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *o,
                          const QdEw16Params *p,int32_t bscalar) {
    if(!p||!valid4(a)||!valid4(o))return 0;
    if(whole_nv(a,b,o)&&qd_blocked_ew16_apply_whole(a,b,o,p,bscalar,0,1))return 1;
    int two=p->kind==QD_EW16_MUL||p->kind==QD_EW16_ADD||p->kind==QD_EW16_ADDSUB||p->kind==QD_EW16_ADDSUBU;
    int splat_b=two&&!b;
    if(two&&!b&&(p->kind!=QD_EW16_MUL||bscalar<0||bscalar>65535))return 0;
    if(two&&b&&!valid4(b))return 0;
    if(p->kind==QD_EW16_ADD1111&&(bscalar<0||bscalar>65535))return 0;
    if(p->kind<QD_EW16_REQUANT||p->kind>QD_EW16_ADDSUBU)return 0;
    if(p->kind!=QD_EW16_MUL&&(p->shift<1||p->shift>31))return 0;
    for(int i=0;i<4;i++){
        if(a->dims[i]!=o->dims[i]||(b&&a->dims[i]!=b->dims[i]))return 0;
    }
    if((a->origin[3]&31)!=(o->origin[3]&31)||(a->origin[2]&3)!=(o->origin[2]&3))return 0;
    if(b&&((a->origin[3]&31)!=(b->origin[3]&31)||(a->origin[2]&3)!=(b->origin[2]&3)))return 0;
    for(int i=0;i<4;i++)if(!a->dims[i])return 1;
    Ctx cx;ctx_make(&cx,p,bscalar);
    HVX_Vector bsplat=splat_b?Q6_V_vsplat_R((uint32_t)bscalar*0x10001u):Q6_V_vzero();
    HVX_VectorPred even=Q6_Q_vcmp_eq_VhVh(Q6_V_vsplat_R(0xffff0000u),Q6_V_vzero());
    int w0=a->origin[2]&~3,w1=a->origin[2]+a->dims[2];
    int d0=a->origin[3]/32,d1=(a->origin[3]+a->dims[3]+31)/32;
    for(int B=0;B<a->dims[0];B++)for(int h=0;h<a->dims[1];h++)
      for(int W=w0;W<w1;W+=4)for(int da=d0;da<d1;da++) {
        int wo=W-a->origin[2]+o->origin[2],doo=da-d0+o->origin[3]/32;
        uint8_t *po=row_ptr(o,B,h,wo,doo);
        if(!po)continue;
        uint8_t *pa=row_ptr(a,B,h,W,da);
        uint8_t *pb=b?row_ptr(b,B,h,W-a->origin[2]+b->origin[2],da-d0+b->origin[3]/32):0;
        int first=a->origin[3]-da*32,last=first+a->dims[3];
        if(first<0)first=0;if(first>32)first=32;if(last<0)last=0;if(last>32)last=32;
        if(first>=last)continue;
        for(int pair=0;pair<2;pair++) {
            int opw=wo+pair*2;
            int ev=opw>=o->origin[2]&&opw<o->origin[2]+o->dims[2];
            int od=opw+1>=o->origin[2]&&opw+1<o->origin[2]+o->dims[2];
            if(!ev&&!od)continue;
            HVX_Vector va=pa?*(const HVX_UVector *)(pa+pair*128):Q6_V_vzero();
            HVX_Vector vb=splat_b?bsplat:(pb?*(const HVX_UVector *)(pb+pair*128):Q6_V_vzero());
            HVX_Vector vr=lane_h(&cx,va,vb);
            Q6_vmem_QRIV(store_mask(first,last,ev,od,even),(HVX_Vector *)(po+pair*128),vr);
        }
      }
    return 1;
}
