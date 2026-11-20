/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "add_u8.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int valid(const QdBlockedU8 *t) {
    if(!t||!t->pools||!t->blocks||t->meta[1]>INT32_MAX-7||
       t->meta[2]>INT32_MAX-7||t->meta[3]>INT32_MAX-31)return 0;
    for(int i=0;i<4;i++)if(t->meta[i]<=0||t->dims[i]<0||t->origin[i]<0||
       (int64_t)t->origin[i]+t->dims[i]>t->meta[i])return 0;
    uint64_t f[3]={(uint64_t)(t->meta[1]+7)/8,(uint64_t)(t->meta[2]+7)/8,
                   (uint64_t)(t->meta[3]+31)/32},slots=(uint32_t)t->meta[0];
    for(int i=0;i<3;i++){if(f[i]&&slots>UINT32_MAX/f[i])return 0;slots*=f[i];}
    return slots+t->slice<=UINT32_MAX;
}
static uintptr_t block_start(const QdBlockedU8 *t,int64_t ix) {
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    return pool&&t->pools[pool]?(uintptr_t)t->pools[pool]+(e&~127u):0;
}
static int same_dims(const QdBlockedU8 *a,const QdBlockedU8 *b) {
    for(int i=0;i<4;i++)if(a->dims[i]!=b->dims[i])return 0;return 1;
}
/* Count/allocate exactly block_count slots; the nested crop loops visit each
 * metadata table index at most once, so missing entries cannot overrun. */
static int collect(const QdBlockedU8 *t,uintptr_t *arr,size_t cap,size_t *n) {
    *n=0;if(!t->dims[0]||!t->dims[1]||!t->dims[2]||!t->dims[3])return 1;
    int nh=(t->meta[1]+7)/8,nw=(t->meta[2]+7)/8,nd=(t->meta[3]+31)/32;
    for(int B=t->origin[0];B<t->origin[0]+t->dims[0];B++)
      for(int H=t->origin[1]/8;H<=(t->origin[1]+t->dims[1]-1)/8;H++)
       for(int W=t->origin[2]/8;W<=(t->origin[2]+t->dims[2]-1)/8;W++)
        for(int D=t->origin[3]/32;D<=(t->origin[3]+t->dims[3]-1)/32;D++) {
            int64_t ix=(((int64_t)B*nh+H)*nw+W)*nd+t->slice+D;
            uintptr_t a=block_start(t,ix);if(a){if(*n>=cap)return 0;arr[(*n)++]=a;}
        }
    return 1;
}
static int overlap(uintptr_t a,uintptr_t b) {return a&&b&&a<b+2048&&b<a+2048;}
static int alias_safe(const QdBlockedU8 *a,const QdBlockedU8 *b,const QdBlockedU8 *o) {
    const QdBlockedU8 *t[3]={a,b,o};uintptr_t *v[3]={0};size_t n[3]={0};int ok=0;
    for(int i=0;i<3;i++) {
#if UINTPTR_MAX == UINT32_MAX
        if(t[i]->block_count>SIZE_MAX/sizeof(uintptr_t))goto done;
#endif
        v[i]=malloc((t[i]->block_count?t[i]->block_count:1)*sizeof(uintptr_t));if(!v[i])goto done;
        if(!collect(t[i],v[i],t[i]->block_count,&n[i]))goto done;
    }
    /* Output entries must be injective across distinct touched table indices. */
    for(size_t i=0;i<n[2];i++)for(size_t j=0;j<i;j++)if(overlap(v[2][i],v[2][j]))goto done;
    for(size_t i=0;i<n[2];i++)for(int k=0;k<2;k++)for(size_t j=0;j<n[k];j++)
        if(overlap(v[2][i],v[k][j]))goto done;
    ok=1;
done:
    for(int i=0;i<3;i++)free(v[i]);return ok;
}

int qd_blocked_add_u8_prepare(const QdBlockedU8 *a,const QdBlockedU8 *b,
        const QdBlockedU8 *out,QdBlockedAddU8 *prepared) {
    if(!prepared||!valid(a)||!valid(b)||!valid(out)||!same_dims(a,b)||!same_dims(a,out))return 0;
    if((a->origin[3]&31)!=(b->origin[3]&31)||(a->origin[3]&31)!=(out->origin[3]&31)||
       (a->origin[2]&3)!=(b->origin[2]&3)||(a->origin[2]&3)!=(out->origin[2]&3))return 0;
    int empty=0;for(int i=0;i<4;i++)if(!a->dims[i])empty=1;
    if(!empty&&!alias_safe(a,b,out))return 0;
    prepared->a=*a;prepared->b=*b;prepared->out=*out;prepared->cookie=0x51444138u;return 1;
}

uint8_t qd_add_u8_oracle(uint8_t a,uint8_t b,const QdAddU8Params *p) {
    uint64_t v=(uint64_t)a*p->ma+(uint64_t)b*p->mb+(1ull<<(p->shift-1));
    v>>=p->shift;return (uint8_t)(v>255?255:v);
}
static int params_ok(const QdAddU8Params *p) {
    if(!p||p->shift<1||p->shift>31)return 0;
    return (uint64_t)255*p->ma+(uint64_t)255*p->mb+(1ull<<(p->shift-1))<=UINT32_MAX;
}
static uint8_t *vector_ptr(const QdBlockedU8 *t,int b,int h,int w,int db) {
    int H=h+t->origin[1],B=b+t->origin[0],W=w,nh=(t->meta[1]+7)/8,
        nw=(t->meta[2]+7)/8,nd=(t->meta[3]+31)/32;
    int64_t ix=(((int64_t)B*nh+H/8)*nw+W/8)*nd+t->slice+db;
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u)+(H%8)*256+(W%8)*32;
}
static HVX_Vector mul_u32(HVX_Vector x,uint32_t m) {
    HVX_Vector lo=Q6_V_vsplat_R(m&65535u);
    if(!(m>>16))return Q6_Vw_vmpyie_VwVuh(x,lo); /* the high product is 0 */
    HVX_Vector hi=Q6_V_vsplat_R(m>>16);
    return Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(x,lo),Q6_Vw_vasl_VwR(Q6_Vw_vmpyie_VwVuh(x,hi),16));
}
static HVX_Vector add_words(HVX_Vector a,HVX_Vector b,const QdAddU8Params *p) {
    HVX_Vector s=Q6_Vw_vadd_VwVw(mul_u32(a,p->ma),mul_u32(b,p->mb));
    s=Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(s,Q6_V_vsplat_R(1u<<(p->shift-1))),p->shift);
    return Q6_Vw_vmin_VwVw(s,Q6_V_vsplat_R(255));
}
static HVX_Vector add_vec(HVX_Vector a,HVX_Vector b,const QdAddU8Params *p) {
    HVX_VectorPair ah=Q6_Wuh_vunpack_Vub(a),bh=Q6_Wuh_vunpack_Vub(b);
    HVX_VectorPair a0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(ah)),a1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(ah));
    HVX_VectorPair b0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(bh)),b1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(bh));
    HVX_Vector q0=add_words(Q6_V_lo_W(a0),Q6_V_lo_W(b0),p),q1=add_words(Q6_V_hi_W(a0),Q6_V_hi_W(b0),p);
    HVX_Vector q2=add_words(Q6_V_lo_W(a1),Q6_V_lo_W(b1),p),q3=add_words(Q6_V_hi_W(a1),Q6_V_hi_W(b1),p);
    HVX_Vector lo=Q6_Vh_vpack_VwVw_sat(q1,q0),hi=Q6_Vh_vpack_VwVw_sat(q3,q2);
    return Q6_Vub_vpack_VhVh_sat(hi,lo);
}
/* Same arithmetic with 16-bit multipliers (ma, mb < 2^16, shift 1..15; vasr takes the shift mod 16):
 * a*ma + b*mb fits u32 (< 2^25), vmpy/vmpyacc uh x uh -> uw computes it
 * exactly, and vasr uw->uh :rnd:sat adds 2^(shift-1) before the shift. The
 * 255 clamp follows the u16 saturation. Even/odd word lanes from vmpy are
 * re-interleaved by the narrowing vasr (hi = odd, lo = even). */
static int params16(const QdAddU8Params *p) {return p->ma<65536u&&p->mb<65536u&&p->shift>=1&&p->shift<=15;}
static HVX_Vector add_half16(HVX_Vector ah,HVX_Vector bh,uint32_t ma,uint32_t mb,uint32_t sh) {
    HVX_VectorPair w=Q6_Wuw_vmpy_VuhRuh(ah,ma|ma<<16);
    w=Q6_Wuw_vmpyacc_WuwVuhRuh(w,bh,mb|mb<<16);
    return Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(w),Q6_V_lo_W(w),sh),Q6_Vh_vsplat_R(255));
}
static HVX_Vector add_vec16(HVX_Vector a,HVX_Vector b,const QdAddU8Params *p) {
    HVX_VectorPair ah=Q6_Wuh_vunpack_Vub(a),bh=Q6_Wuh_vunpack_Vub(b);
    HVX_Vector lo=add_half16(Q6_V_lo_W(ah),Q6_V_lo_W(bh),p->ma,p->mb,p->shift);
    HVX_Vector hi=add_half16(Q6_V_hi_W(ah),Q6_V_hi_W(bh),p->ma,p->mb,p->shift);
    return Q6_Vub_vpack_VhVh_sat(hi,lo);
}
static HVX_VectorPred prefix(unsigned n) {return n>=128?Q6_Q_not_Q(Q6_Q_vsetq_R(0)):Q6_Q_vsetq_R(n);}

int qd_blocked_add_u8_apply(const QdBlockedAddU8 *prepared,const QdAddU8Params *p) {
    if(!prepared||prepared->cookie!=0x51444138u||!params_ok(p))return 0;
    const QdBlockedU8 *a=&prepared->a,*b=&prepared->b,*o=&prepared->out;
    for(int i=0;i<4;i++)if(!a->dims[i])return 1;
    int d0=a->origin[3]/32,d1=(a->origin[3]+a->dims[3]+31)/32;
    int w0=a->origin[2]&~3,w1=a->origin[2]+a->dims[2];
    for(int B=0;B<a->dims[0];B++)for(int H=0;H<a->dims[1];H++)
      for(int W=w0;W<w1;W+=4)for(int da=d0;da<d1;da++) {
        int wb=W-a->origin[2]+b->origin[2],wo=W-a->origin[2]+o->origin[2];
        int db=da-a->origin[3]/32+b->origin[3]/32,doo=da-a->origin[3]/32+o->origin[3]/32;
        uint8_t *pa=vector_ptr(a,B,H,W,da),*pb=vector_ptr(b,B,H,wb,db),*po=vector_ptr(o,B,H,wo,doo);
        if(!po)continue;
        HVX_Vector va=pa?*(const HVX_UVector *)pa:Q6_V_vzero();
        HVX_Vector vb=pb?*(const HVX_UVector *)pb:Q6_V_vzero();
        HVX_Vector vr=add_vec(va,vb,p);
        int first=a->origin[3]-da*32,last=first+a->dims[3];
        if(first<0)first=0;if(first>32)first=32;if(last<0)last=0;if(last>32)last=32;
        HVX_VectorPred mask=Q6_Q_vsetq_R(0);
        for(int j=0;j<4;j++)if(W+j>=a->origin[2]&&W+j<w1)
            mask=Q6_Q_or_QQ(mask,Q6_Q_and_QQn(prefix((unsigned)(j*32+last)),prefix((unsigned)(j*32+first))));
        if(first<last)Q6_vmem_QRIV(mask,(HVX_Vector *)po,vr);
      }
    return 1;
}

/* Whole-block form: every tensor's region is whole 8x8x32 blocks (origins and
 * extents multiples of 8, 8, 32), so each output block is 16 full vectors
 * computed from the matching input blocks; same lanes and the same absent-
 * block rules (absent input = zeros, absent output = no store) as apply.
 * Blocks i = part, part+parts, ... of the region in (B, H, W, D) order. */
static int whole(const QdBlockedU8 *t) {
    return !(t->origin[1]&7)&&!(t->origin[2]&7)&&!(t->origin[3]&31)&&!(t->dims[1]&7)&&!(t->dims[2]&7)&&!(t->dims[3]&31);
}
int qd_blocked_add_u8_whole(const QdBlockedAddU8 *prepared) {
    if(!prepared||prepared->cookie!=0x51444138u)return 0;
    return whole(&prepared->a)&&whole(&prepared->b)&&whole(&prepared->out);
}
/* Block-table position of a tensor's region: 32-bit, no divide per block. */
typedef struct { const uint32_t *blocks; uint8_t *const *pools; uint32_t count, base, nh, nw, nd; } BlkIx;
static inline BlkIx blk_ix(const QdBlockedU8 *t) {
    BlkIx x;
    x.blocks=t->blocks;x.pools=t->pools;x.count=t->block_count;
    x.nh=(uint32_t)((t->meta[1]+7)/8);x.nw=(uint32_t)((t->meta[2]+7)/8);x.nd=(uint32_t)((t->meta[3]+31)/32);
    /* ((B0*nh + H0)*nw + W0)*nd + slice + D0 for the region's origin */
    x.base=((((uint32_t)t->origin[0]*x.nh+(uint32_t)(t->origin[1]/8))*x.nw+(uint32_t)(t->origin[2]/8))*x.nd)+t->slice+(uint32_t)(t->origin[3]/32);
    return x;
}
static inline const uint8_t *blk32(const BlkIx *x,uint32_t ix) {
    if(ix>=x->count)return NULL;
    const uint32_t e=x->blocks[ix],pool=e&127u;
    return pool&&x->pools[pool]?x->pools[pool]+(e&~127u):NULL;
}
int qd_blocked_add_u8_apply_whole(const QdBlockedAddU8 *prepared,const QdAddU8Params *p,uint32_t part,uint32_t parts) {
    if(!qd_blocked_add_u8_whole(prepared)||!params_ok(p)||!parts)return 0;
    const QdBlockedU8 *a=&prepared->a,*b=&prepared->b,*o=&prepared->out;
    const int nB=a->dims[0],nH=a->dims[1]/8,nW=a->dims[2]/8,nD=a->dims[3]/32;
    const int same=a==b||(a->blocks==b->blocks&&a->slice==b->slice&&!memcmp(a->origin,b->origin,sizeof a->origin)&&
                          !memcmp(a->meta,b->meta,sizeof a->meta));
    const int f16=params16(p);
    const BlkIx xa=blk_ix(a),xb=blk_ix(b),xo=blk_ix(o);
    uint32_t i=0,next=part;
    for(int B=0;B<nB;B++)for(int H=0;H<nH;H++)for(int W=0;W<nW;W++) {
        const uint32_t ra=xa.base+(((uint32_t)B*xa.nh+(uint32_t)H)*xa.nw+(uint32_t)W)*xa.nd,
                       rb=xb.base+(((uint32_t)B*xb.nh+(uint32_t)H)*xb.nw+(uint32_t)W)*xb.nd,
                       ro=xo.base+(((uint32_t)B*xo.nh+(uint32_t)H)*xo.nw+(uint32_t)W)*xo.nd;
        for(int D=0;D<nD;D++,i++) {
            if(i!=next)continue;
            next+=parts;
            uint8_t *po=(uint8_t *)blk32(&xo,ro+(uint32_t)D);if(!po)continue;
            const uint8_t *pa=blk32(&xa,ra+(uint32_t)D),*pb=same?pa:blk32(&xb,rb+(uint32_t)D);
            if(f16)for(int v=0;v<16;v++) {
                HVX_Vector va=pa?((const HVX_Vector *)pa)[v]:Q6_V_vzero();
                HVX_Vector vb=pb?((const HVX_Vector *)pb)[v]:Q6_V_vzero();
                ((HVX_Vector *)po)[v]=add_vec16(va,vb,p);
            }
            else for(int v=0;v<16;v++) {
                HVX_Vector va=pa?((const HVX_Vector *)pa)[v]:Q6_V_vzero();
                HVX_Vector vb=pb?((const HVX_Vector *)pb)[v]:Q6_V_vzero();
                ((HVX_Vector *)po)[v]=add_vec(va,vb,p);
            }
        }
    }
    return 1;
}
