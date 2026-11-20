/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "linearclip_u16.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

static int valid(const QdBlockedU16 *t) {
    if (!t || !t->pools || !t->blocks) return 0;
    if(t->meta[1]>INT32_MAX-7||t->meta[2]>INT32_MAX-3||t->meta[3]>INT32_MAX-31)return 0;
    for (unsigned i=0;i<4;i++) if (t->meta[i]<=0 || t->dims[i]<0 || t->origin[i]<0 ||
            (int64_t)t->origin[i]+t->dims[i]>t->meta[i]) return 0;
    /* Ensure every metadata-derived table index and slice addition fits. */
    uint64_t nb[3]={(uint64_t)(t->meta[1]+7)/8,(uint64_t)(t->meta[2]+3)/4,
                    (uint64_t)(t->meta[3]+31)/32};
    uint64_t slots=(uint32_t)t->meta[0];
    for(unsigned i=0;i<3;i++) {
        if(nb[i] && slots>UINT32_MAX/nb[i])return 0;
        slots*=nb[i];
    }
    if(slots+t->slice>UINT32_MAX)return 0;
    return 1;
}

static uintptr_t block_start(const QdBlockedU16 *t,uint32_t i) {
    uint32_t e=t->blocks[i],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return (uintptr_t)t->pools[pool]+(e&~127u);
}

static int same_mapping(const QdBlockedU16 *in,const QdBlockedU16 *out) {
    for(int i=0;i<4;i++)
        if(in->meta[i]!=out->meta[i]||in->dims[i]!=out->dims[i]||in->origin[i]!=out->origin[i])return 0;
    if(in->slice!=out->slice)return 0;
    if(in->block_count!=out->block_count)return 0;
    for(uint32_t i=0;i<in->block_count;i++)
        if(block_start(in,i)!=block_start(out,i))return 0;
    return 1;
}

/* Conservative proof over whole tables. Disjoint tables are safe. Overlap is
 * accepted only for the exact same, injective physical mapping (in-place). */
static int safe_aliasing(const QdBlockedU16 *in,const QdBlockedU16 *out) {
    int injective=1,cross_overlap=0;
    for(uint32_t i=0;i<out->block_count;i++) {
        uintptr_t a=block_start(out,i);if(!a)continue;
        for(uint32_t j=0;j<i;j++) {
            uintptr_t b=block_start(out,j);
            if(b && a<b+2048 && b<a+2048)injective=0;
        }
        for(uint32_t j=0;j<in->block_count;j++) {
            uintptr_t b=block_start(in,j);
            if(b && a<b+2048 && b<a+2048)cross_overlap=1;
        }
    }
    if(!injective)return 0;
    return !cross_overlap||same_mapping(in,out);
}

uint32_t qd_linearclip_u16_oracle(uint16_t x,const QdLinearclipParams *p) {
    int64_t n=((int64_t)x-p->input_zero)*p->multiplier+
              ((int64_t)1<<(p->shift-1));
    int64_t v=(n>>p->shift)+p->output_zero,hi=65535;
    return (uint32_t)(v<0?0:v>hi?hi:v);
}

static uint8_t *physical_vector(const QdBlockedU16 *t,int32_t b,int32_t h,
                               int32_t pair_w,int32_t depth_block) {
    int32_t H=h+t->origin[1],B=b+t->origin[0],
        nbh=(t->meta[1]+7)/8,nbw=(t->meta[2]+3)/4,nbd=(t->meta[3]+31)/32;
    int64_t index=(((int64_t)B*nbh+H/8)*nbw+pair_w/4)*nbd+t->slice+depth_block;
    if(index<0||(uint64_t)index>=t->block_count)return 0;
    uint32_t e=t->blocks[index],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u)+(H%8)*256+((pair_w%4)/2)*128;
}

static HVX_Vector mul_u32(HVX_Vector a,uint32_t m) {
    HVX_Vector lo=Q6_V_vsplat_R(m&65535u),hi=Q6_V_vsplat_R(m>>16);
    return Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(a,lo),
            Q6_Vw_vasl_VwR(Q6_Vw_vmpyie_VwVuh(a,hi),16));
}
static HVX_Vector affine_vec(HVX_Vector x,const QdLinearclipParams *p) {
    HVX_VectorPair u=Q6_Wuw_vunpack_Vuh(x);
    HVX_Vector z=Q6_V_vsplat_R((uint32_t)p->input_zero);
    HVX_Vector rd=Q6_V_vsplat_R(1u<<(p->shift-1));
    HVX_Vector a=Q6_Vw_vadd_VwVw(mul_u32(Q6_Vw_vsub_VwVw(Q6_V_lo_W(u),z),(uint32_t)p->multiplier),rd);
    HVX_Vector b=Q6_Vw_vadd_VwVw(mul_u32(Q6_Vw_vsub_VwVw(Q6_V_hi_W(u),z),(uint32_t)p->multiplier),rd);
    a=Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(a,p->shift),Q6_V_vsplat_R((uint32_t)p->output_zero));
    b=Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(b,p->shift),Q6_V_vsplat_R((uint32_t)p->output_zero));
    return Q6_Vuh_vpack_VwVw_sat(b,a);
}
static HVX_VectorPred prefix_bytes(unsigned n) {
    return n>=128?Q6_Q_not_Q(Q6_Q_vsetq_R(0)):Q6_Q_vsetq_R(n);
}
static int affine_fits(const QdLinearclipParams *p) {
    if(p->shift>31 || p->input_zero<0 || p->input_zero>65535 ||
       p->output_zero<0 || p->output_zero>65535)return 0;
    int64_t r=1ll<<(p->shift-1),a=-(int64_t)p->input_zero*p->multiplier+r;
    int64_t b=(65535-(int64_t)p->input_zero)*p->multiplier+r;
    if(a<INT32_MIN||a>INT32_MAX||b<INT32_MIN||b>INT32_MAX)return 0;
    a=(a>>p->shift)+p->output_zero;b=(b>>p->shift)+p->output_zero;
    return a>=INT32_MIN&&a<=INT32_MAX&&b>=INT32_MIN&&b<=INT32_MAX;
}


/* ---- Whole-block form: every region is whole 8x4x32 blocks, so one table
 * lookup serves 16 vectors and no store masks are needed. Absent input blocks
 * read as zeros, absent output blocks discard their stores. Blocks are
 * visited in (B,H,W,D) order; in-place (same blocks) is safe because each
 * vector is read before it is written.
 *
 * Fast forms (exact; verified exhaustively over all u16 by linearclip_u16_sim):
 *   multiplier == -(1<<shift): v = (oz+iz) - x          (floor(-(x-iz)+1/2) = iz-x)
 *   multiplier ==  (1<<shift): v = x + (oz-iz)
 * both clipped to [0,65535] by saturating u16 add/sub. Everything else uses
 * the generic 32-bit affine. */
static int whole_region(const QdBlockedU16 *t) {
    return !(t->origin[1]&7)&&!(t->origin[2]&3)&&!(t->origin[3]&31)&&
           !(t->dims[1]&7)&&!(t->dims[2]&3)&&!(t->dims[3]&31);
}
static uint8_t *block_ptr_u16(const QdBlockedU16 *t,int64_t ix) {
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    return pool&&t->pools[pool]?t->pools[pool]+(e&~127u):0;
}
enum { LC_GENERIC, LC_NEG, LC_POS };
static int apply_whole(const QdBlockedU16 *in,const QdBlockedU16 *out,const QdLinearclipParams *p) {
    const int nB=in->dims[0],nH=in->dims[1]/8,nW=in->dims[2]/4,nD=in->dims[3]/32;
    const int iH=(in->meta[1]+7)/8,iW=(in->meta[2]+3)/4,iD=(in->meta[3]+31)/32;
    const int oH=(out->meta[1]+7)/8,oW=(out->meta[2]+3)/4,oD=(out->meta[3]+31)/32;
    int mode=LC_GENERIC;
    if(p->multiplier==-(int64_t)(1ll<<p->shift)&&p->shift<=31)mode=LC_NEG;
    else if(p->multiplier>0&&(int64_t)p->multiplier==(1ll<<p->shift))mode=LC_POS;
    const int c=p->output_zero+p->input_zero,d=p->output_zero-p->input_zero;
    HVX_Vector vc=Q6_Vh_vsplat_R(c&65535),vd=Q6_Vh_vsplat_R((d<0?-d:d)&65535),vx=Q6_Vh_vsplat_R(c>65535?c-65535:0);
    HVX_Vector all1=Q6_Vh_vsplat_R(65535);
    for(int B=0;B<nB;B++)for(int H=0;H<nH;H++)for(int W=0;W<nW;W++)for(int D=0;D<nD;D++) {
        int64_t ix=((((int64_t)(B+in->origin[0])*iH+H+in->origin[1]/8)*iW+W+in->origin[2]/4)*iD)+in->slice+D+in->origin[3]/32;
        int64_t ox=((((int64_t)(B+out->origin[0])*oH+H+out->origin[1]/8)*oW+W+out->origin[2]/4)*oD)+out->slice+D+out->origin[3]/32;
        HVX_Vector *po=(HVX_Vector *)block_ptr_u16(out,ox);
        if(!po)continue;
        const HVX_Vector *pi=(const HVX_Vector *)block_ptr_u16(in,ix);
        if(mode==LC_NEG) {
            if(c<=65535) {
                #pragma unroll
                for(int v=0;v<16;v++)po[v]=Q6_Vuh_vsub_VuhVuh_sat(vc,pi?pi[v]:Q6_V_vzero());
            } else {
                #pragma unroll
                for(int v=0;v<16;v++)po[v]=Q6_Vuh_vadd_VuhVuh_sat(Q6_Vh_vsub_VhVh(all1,pi?pi[v]:Q6_V_vzero()),vx);
            }
        } else if(mode==LC_POS) {
            if(d>=0) {
                #pragma unroll
                for(int v=0;v<16;v++)po[v]=Q6_Vuh_vadd_VuhVuh_sat(pi?pi[v]:Q6_V_vzero(),vd);
            } else {
                #pragma unroll
                for(int v=0;v<16;v++)po[v]=Q6_Vuh_vsub_VuhVuh_sat(pi?pi[v]:Q6_V_vzero(),vd);
            }
        } else {
            #pragma unroll
            for(int v=0;v<16;v++)po[v]=affine_vec(pi?pi[v]:Q6_V_vzero(),p);
        }
    }
    return 1;
}

int qd_blocked_linearclip_u16_prepare(const QdBlockedU16 *in,const QdBlockedU16 *out,
        const QdLinearclipParams *p,QdBlockedU16Linearclip *prepared) {
    if(!prepared||!valid(in)||!valid(out)||!p||p->reserved||p->shift<1||p->shift>31||!affine_fits(p)||
       in->dims[0]!=out->dims[0]||in->dims[1]!=out->dims[1]||
       in->dims[2]!=out->dims[2]||in->dims[3]!=out->dims[3])return 0;
    if((in->origin[2]&1)!=(out->origin[2]&1)||
       (in->origin[3]&31)!=(out->origin[3]&31))return 0;
    int empty=0;for(int i=0;i<4;i++)if(!in->dims[i])empty=1;
    if(!empty&&!safe_aliasing(in,out))return 0;
    prepared->in=*in;prepared->out=*out;prepared->params=*p;prepared->cookie=0x51444c43u;
    return 1;
}

int qd_blocked_linearclip_u16_apply_params(const QdBlockedU16Linearclip *prepared,
        const QdLinearclipParams *p) {
    if(!prepared||prepared->cookie!=0x51444c43u||!p||p->reserved||
       p->shift<1||p->shift>31||!affine_fits(p))return 0;
    const QdBlockedU16 *in=&prepared->in,*out=&prepared->out;
    for(int i=0;i<4;i++)if(!in->dims[i])return 1;
    if(whole_region(in)&&whole_region(out))return apply_whole(in,out,p);
    /* One 128-byte vector is 32 depth positions x two interleaved W parity
     * lanes. Compute the affine over physical lanes, but store only the crop. */
    int32_t w0=in->origin[2]&~1,w1=in->origin[2]+in->dims[2];
    int32_t d0=in->origin[3]/32,d1=(in->origin[3]+in->dims[3]+31)/32;
    HVX_VectorPred even=Q6_Q_vcmp_eq_VhVh(Q6_V_vsplat_R(0xffff0000u),Q6_V_vzero());
    for(int32_t b=0;b<in->dims[0];b++)for(int32_t h=0;h<in->dims[1];h++)
      for(int32_t W=w0;W<w1;W+=2)for(int32_t db=d0;db<d1;db++) {
        int32_t outW=W-in->origin[2]+out->origin[2];
        int32_t outDb=db-in->origin[3]/32+out->origin[3]/32;
        uint8_t *src=physical_vector(in,b,h,W,db),*dst=physical_vector(out,b,h,outW,outDb);
        if(!dst)continue;
        HVX_Vector x=src?*(const HVX_UVector *)src:Q6_V_vzero();
        HVX_Vector y=affine_vec(x,p);
        int32_t first=in->origin[3]-db*32,last=first+in->dims[3];
        if(first<0)first=0;if(first>32)first=32;
        if(last<0)last=0;if(last>32)last=32;
        HVX_VectorPred channels=Q6_Q_and_QQn(prefix_bytes((unsigned)last*4),prefix_bytes((unsigned)first*4));
        int even_valid=(W>=in->origin[2]&&W<w1),odd_valid=(W+1>=in->origin[2]&&W+1<w1);
        HVX_VectorPred mask=Q6_Q_and_QQ(channels,
                (even_valid?even:Q6_Q_vsetq_R(0)));
        if(odd_valid)mask=Q6_Q_or_QQ(mask,Q6_Q_and_QQ(channels,Q6_Q_not_Q(even)));
        if(even_valid||odd_valid)Q6_vmem_QRIV(mask,(HVX_Vector *)dst,y);
      }
    return 1;
}

int qd_blocked_linearclip_u16_apply(const QdBlockedU16Linearclip *prepared) {
    return prepared&&qd_blocked_linearclip_u16_apply_params(prepared,&prepared->params);
}

int qd_blocked_linearclip_u16(const QdBlockedU16 *in,const QdBlockedU16 *out,
                              const QdLinearclipParams *p) {
    QdBlockedU16Linearclip prepared;
    return qd_blocked_linearclip_u16_prepare(in,out,p,&prepared)&&
           qd_blocked_linearclip_u16_apply(&prepared);
}
