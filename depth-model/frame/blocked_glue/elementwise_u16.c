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

static int valid4(const QdBlockedU16 *t) {
    if(!t||!t->pools||!t->blocks||t->meta[1]>INT32_MAX-7||t->meta[2]>INT32_MAX-3||t->meta[3]>INT32_MAX-31)return 0;
    for(int i=0;i<4;i++)if(t->meta[i]<=0||t->dims[i]<0||t->origin[i]<0||(int64_t)t->origin[i]+t->dims[i]>t->meta[i])return 0;
    uint64_t f[3]={(uint64_t)(t->meta[1]+7)/8,(uint64_t)(t->meta[2]+3)/4,(uint64_t)(t->meta[3]+31)/32},slots=(uint32_t)t->meta[0];
    for(int i=0;i<3;i++){if(f[i]&&slots>UINT32_MAX/f[i])return 0;slots*=f[i];}
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

int qd_blocked_ew16_apply(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *o,
                          const QdEw16Params *p,int32_t bscalar) {
    if(!p||!valid4(a)||!valid4(o))return 0;
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
