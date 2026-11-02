#include "linearclip_u8_u16.h"
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static int valid4(const uint8_t *const *pools,const uint32_t *blocks,
                  uint32_t count,const int32_t *m,const int32_t *d,const int32_t *o,
                  int width_block) {
    (void)count;
    if(!pools||!blocks||m[1]>INT32_MAX-7||
       m[2]>INT32_MAX-(width_block-1)||m[3]>INT32_MAX-31)return 0;
    for(int i=0;i<4;i++)if(m[i]<=0||d[i]<0||o[i]<0||(int64_t)o[i]+d[i]>m[i])return 0;
    uint64_t nbh=(uint64_t)(m[1]+7)/8,nbw=(uint64_t)(m[2]+width_block-1)/width_block,
             nbd=(uint64_t)(m[3]+31)/32,slots=(uint32_t)m[0];
    uint64_t f[3]={nbh,nbw,nbd};
    for(int i=0;i<3;i++){if(f[i]&&slots>UINT32_MAX/f[i])return 0;slots*=f[i];}
    return slots<=UINT32_MAX;
}
static uintptr_t start4(const uint8_t *const *pools,const uint32_t *blocks,uint32_t i) {
    uint32_t e=blocks[i],pool=e&127u;
    return pool&&pools[pool]?(uintptr_t)pools[pool]+(e&~127u):0;
}
static int collect_touched(const uint8_t *const *pools,const uint32_t *blocks,uint32_t count,
        uint32_t slice,const int32_t *m,const int32_t *d,const int32_t *o,int width,
        uintptr_t *starts,size_t *nstarts,int reject_alias) {
    int nbh=(m[1]+7)/8,nbw=(m[2]+width-1)/width,nbd=(m[3]+31)/32;
    size_t n=0;
    for(int B=o[0];B<o[0]+d[0];B++)
      for(int H=o[1]/8;H<=(o[1]+d[1]-1)/8;H++)
       for(int W=o[2]/width;W<=(o[2]+d[2]-1)/width;W++)
        for(int D=o[3]/32;D<=(o[3]+d[3]-1)/32;D++) {
            int64_t index=(((int64_t)B*nbh+H)*nbw+W)*nbd+slice+D;
            uintptr_t a=0;
            if(index>=0&&(uint64_t)index<count)a=start4(pools,blocks,(uint32_t)index);
            if(!a)continue;
            if(reject_alias)for(size_t i=0;i<n;i++)
                if(a<starts[i]+2048&&starts[i]<a+2048)return 0;
            if(n>=count)return 0;
            starts[n++]=a;
        }
    *nstarts=n;return 1;
}
static int touched_storage_disjoint(const QdBlockedU8 *in,const QdBlockedU16Out *out) {
    size_t ni=0,no=0,ic=in->block_count?in->block_count:1,oc=out->block_count?out->block_count:1;
    if(ic>SIZE_MAX/sizeof(uintptr_t)||oc>SIZE_MAX/sizeof(uintptr_t))return 0;
    uintptr_t *is=malloc(ic*sizeof *is),*os=malloc(oc*sizeof *os);
    if(!is||!os){free(is);free(os);return 0;}
    int ok=collect_touched((const uint8_t *const *)in->pools,in->blocks,in->block_count,
            in->slice,in->meta,in->dims,in->origin,8,is,&ni,0)&&
           collect_touched((const uint8_t *const *)out->pools,out->blocks,out->block_count,
            out->slice,out->meta,out->dims,out->origin,4,os,&no,1);
    if(ok)for(size_t i=0;i<no&&ok;i++)for(size_t j=0;j<ni;j++)
        if(os[i]<is[j]+2048&&is[j]<os[i]+2048){ok=0;break;}
    free(is);free(os);return ok;
}
static uint8_t *u8_group(const QdBlockedU8 *t,int b,int h,int w4,int db) {
    int H=h+t->origin[1],B=b+t->origin[0],nbh=(t->meta[1]+7)/8,
        nbw=(t->meta[2]+7)/8,nbd=(t->meta[3]+31)/32;
    int64_t index=(((int64_t)B*nbh+H/8)*nbw+w4/8)*nbd+t->slice+db;
    if(index<0||(uint64_t)index>=t->block_count)return 0;
    uint32_t e=t->blocks[index],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u)+(H%8)*256+(w4%8)*32;
}
static uint8_t *u16_pair(const QdBlockedU16Out *t,int b,int h,int pair_w,int db) {
    int H=h+t->origin[1],B=b+t->origin[0],nbh=(t->meta[1]+7)/8,
        nbw=(t->meta[2]+3)/4,nbd=(t->meta[3]+31)/32;
    int64_t index=(((int64_t)B*nbh+H/8)*nbw+pair_w/4)*nbd+t->slice+db;
    if(index<0||(uint64_t)index>=t->block_count)return 0;
    uint32_t e=t->blocks[index],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u)+(H%8)*256+((pair_w%4)/2)*128;
}
uint32_t qd_linearclip_u8_u16_oracle(uint8_t x,const QdLinearclipU8U16Params *p) {
    uint64_t y=((uint64_t)x*p->multiplier+(1ull<<(p->shift-1)))>>p->shift;
    return y>65535?65535:(uint32_t)y;
}
static HVX_Vector mul_u32(HVX_Vector x,uint32_t m) {
    HVX_Vector lo=Q6_V_vsplat_R(m&65535u),hi=Q6_V_vsplat_R(m>>16);
    return Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(x,lo),Q6_Vw_vasl_VwR(Q6_Vw_vmpyie_VwVuh(x,hi),16));
}
static HVX_Vector affine(HVX_Vector x,const QdLinearclipU8U16Params *p) {
    HVX_VectorPair w=Q6_Wuw_vunpack_Vuh(x);
    HVX_Vector rd=Q6_V_vsplat_R(1u<<(p->shift-1));
    HVX_Vector a=Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(mul_u32(Q6_V_lo_W(w),p->multiplier),rd),p->shift);
    HVX_Vector b=Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(mul_u32(Q6_V_hi_W(w),p->multiplier),rd),p->shift);
    return Q6_Vuh_vpack_VwVw_sat(b,a);
}
static HVX_VectorPred prefix(unsigned n) {return n>=128?Q6_Q_not_Q(Q6_Q_vsetq_R(0)):Q6_Q_vsetq_R(n);}
static HVX_VectorPred store_mask(int first,int last,int even_valid,int odd_valid,
                                 HVX_VectorPred even) {
    HVX_VectorPred ch=Q6_Q_and_QQn(prefix((unsigned)last*4),prefix((unsigned)first*4));
    HVX_VectorPred q=Q6_Q_and_QQ(ch,even_valid?even:Q6_Q_vsetq_R(0));
    if(odd_valid)q=Q6_Q_or_QQ(q,Q6_Q_and_QQ(ch,Q6_Q_not_Q(even)));
    return q;
}
int qd_blocked_linearclip_u8_u16_prepare(const QdBlockedU8 *in,const QdBlockedU16Out *out,
                                        QdBlockedLinearclipU8U16 *prepared) {
    if(!prepared||!in||!out||!valid4((const uint8_t *const *)in->pools,in->blocks,in->block_count,in->meta,in->dims,in->origin,8)||
       !valid4((const uint8_t *const *)out->pools,out->blocks,out->block_count,out->meta,out->dims,out->origin,4))return 0;
    for(int i=0;i<4;i++)if(in->dims[i]!=out->dims[i])return 0;
    if(in->origin[2]%4!=out->origin[2]%4||in->origin[3]%32!=out->origin[3]%32)return 0;
    int empty=0;for(int i=0;i<4;i++)if(!in->dims[i])empty=1;
    if(!empty&&!touched_storage_disjoint(in,out))return 0;
    prepared->in=*in;prepared->out=*out;prepared->cookie=0x51445538u;return 1;
}
int qd_blocked_linearclip_u8_u16_apply(const QdBlockedLinearclipU8U16 *prepared,
        const QdLinearclipU8U16Params *p) {
    if(!prepared||prepared->cookie!=0x51445538u||!p||p->multiplier>65535||p->shift<1||p->shift>31)return 0;
    const QdBlockedU8 *in=&prepared->in;const QdBlockedU16Out *out=&prepared->out;
    for(int i=0;i<4;i++)if(!in->dims[i])return 1;
    int w0=in->origin[2]&~3,w1=in->origin[2]+in->dims[2];
    int d0=in->origin[3]/32,d1=(in->origin[3]+in->dims[3]+31)/32;
    HVX_VectorPred even=Q6_Q_vcmp_eq_VhVh(Q6_V_vsplat_R(0xffff0000u),Q6_V_vzero());
    for(int b=0;b<in->dims[0];b++)for(int h=0;h<in->dims[1];h++)
      for(int W=w0;W<w1;W+=4)for(int db=d0;db<d1;db++) {
        uint8_t *src=u8_group(in,b,h,W,db);HVX_Vector x=src?*(const HVX_UVector *)src:Q6_V_vzero();
        HVX_VectorPair half=Q6_Wuh_vunpack_Vub(x);
        HVX_Vector y0=Q6_Vh_vshuff_Vh(affine(Q6_V_lo_W(half),p));
        HVX_Vector y1=Q6_Vh_vshuff_Vh(affine(Q6_V_hi_W(half),p));
        int first=in->origin[3]-db*32,last=first+in->dims[3];
        if(first<0)first=0;if(first>32)first=32;if(last<0)last=0;if(last>32)last=32;
        int outW=W-in->origin[2]+out->origin[2];
        int outDb=db-in->origin[3]/32+out->origin[3]/32;
        for(int pair=0;pair<2;pair++) {
            int pw=W+pair*2,opw=outW+pair*2;
            uint8_t *dst=u16_pair(out,b,h,opw,outDb);if(!dst)continue;
            int ev=opw>=out->origin[2]&&opw<out->origin[2]+out->dims[2];
            int od=opw+1>=out->origin[2]&&opw+1<out->origin[2]+out->dims[2];
            HVX_VectorPred mask=store_mask(first,last,ev,od,even);
            if(ev||od)Q6_vmem_QRIV(mask,(HVX_Vector *)dst,pair?y1:y0);
            (void)pw;
        }
      }
    return 1;
}

/* ---- Whole-block form --------------------------------------------------
 * Every region is whole blocks (input 8x8x32 u8, output 8x4x32 u16: origins
 * and extents multiples of 8/8/32 for the input and 8/4/32 for the output),
 * so one table lookup per block serves 16 vectors and no store masks are
 * needed: each input block (rows x 8 W x 32 c) feeds two output blocks (W
 * 0..3 and 4..7). Absent input blocks read as zeros; absent output blocks
 * discard their stores. Blocks i = part, part+parts, ... of the region in
 * (B, H, W8, D) order.
 *
 * Fast arithmetic (multiplier < 2^16 and 1 <= shift <= 15): vmpy uh x uh -> uw
 * is exact (x*m < 2^24), and vasr uw -> uh :rnd:sat adds 2^(shift-1), shifts
 * and clips to u16, re-interleaving the even/odd product lanes. The vasr
 * shift is taken mod 16, so shift 16..31 uses the generic 32-bit path. */
static int whole_in(const QdBlockedU8 *t) {
    return !(t->origin[1]&7)&&!(t->origin[2]&7)&&!(t->origin[3]&31)&&
           !(t->dims[1]&7)&&!(t->dims[2]&7)&&!(t->dims[3]&31);
}
static int whole_out(const QdBlockedU16Out *t) {
    return !(t->origin[1]&7)&&!(t->origin[2]&3)&&!(t->origin[3]&31)&&
           !(t->dims[1]&7)&&!(t->dims[2]&7)&&!(t->dims[3]&31);
}
int qd_blocked_linearclip_u8_u16_whole(const QdBlockedLinearclipU8U16 *prepared) {
    if(!prepared||prepared->cookie!=0x51445538u)return 0;
    return whole_in(&prepared->in)&&whole_out(&prepared->out);
}
static uint8_t *block_ptr(const uint8_t *const *pools,const uint32_t *blocks,uint32_t count,int64_t ix) {
    if(ix<0||(uint64_t)ix>=count)return 0;
    uint32_t e=blocks[ix],pool=e&127u;
    return pool&&pools[pool]?(uint8_t *)pools[pool]+(e&~127u):0;
}
static HVX_Vector aff16(HVX_Vector h,uint32_t m2,uint32_t sh) {
    HVX_VectorPair w=Q6_Wuw_vmpy_VuhRuh(h,m2);
    return Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(w),Q6_V_lo_W(w),sh);
}
static void block16(const HVX_Vector *pi,HVX_Vector *pa,HVX_Vector *pb,uint32_t m2,uint32_t sh) {
    #pragma unroll
    for(int v=0;v<16;v++) {
        HVX_Vector x=pi?pi[v]:Q6_V_vzero();
        HVX_VectorPair h=Q6_Wuh_vunpack_Vub(x);
        HVX_Vector y0=Q6_Vh_vshuff_Vh(aff16(Q6_V_lo_W(h),m2,sh));
        HVX_Vector y1=Q6_Vh_vshuff_Vh(aff16(Q6_V_hi_W(h),m2,sh));
        HVX_Vector *po=(v&1)?pb:pa;
        if(po){po[(v>>1)*2]=y0;po[(v>>1)*2+1]=y1;}
    }
}
static void block32(const HVX_Vector *pi,HVX_Vector *pa,HVX_Vector *pb,const QdLinearclipU8U16Params *p) {
    #pragma unroll
    for(int v=0;v<16;v++) {
        HVX_Vector x=pi?pi[v]:Q6_V_vzero();
        HVX_VectorPair half=Q6_Wuh_vunpack_Vub(x);
        HVX_Vector y0=Q6_Vh_vshuff_Vh(affine(Q6_V_lo_W(half),p));
        HVX_Vector y1=Q6_Vh_vshuff_Vh(affine(Q6_V_hi_W(half),p));
        HVX_Vector *po=(v&1)?pb:pa;
        if(po){po[(v>>1)*2]=y0;po[(v>>1)*2+1]=y1;}
    }
}
int qd_blocked_linearclip_u8_u16_apply_whole(const QdBlockedLinearclipU8U16 *prepared,
        const QdLinearclipU8U16Params *p,uint32_t part,uint32_t parts) {
    if(!qd_blocked_linearclip_u8_u16_whole(prepared)||!p||p->multiplier>65535||p->shift<1||p->shift>31||!parts)return 0;
    const QdBlockedU8 *in=&prepared->in;const QdBlockedU16Out *out=&prepared->out;
    for(int i=0;i<4;i++)if(!in->dims[i])return 1;
    const int nB=in->dims[0],nH=in->dims[1]/8,nW=in->dims[2]/8,nD=in->dims[3]/32;
    const int inH=(in->meta[1]+7)/8,inW=(in->meta[2]+7)/8,inD=(in->meta[3]+31)/32;
    const int oH=(out->meta[1]+7)/8,oW=(out->meta[2]+3)/4,oD=(out->meta[3]+31)/32;
    const int f16=p->shift<=15;
    const uint32_t m2=p->multiplier|p->multiplier<<16;
    uint32_t i=0;
    for(int B=0;B<nB;B++)for(int H=0;H<nH;H++)for(int W=0;W<nW;W++)for(int D=0;D<nD;D++,i++) {
        if(i%parts!=part)continue;
        int64_t ix=((((int64_t)(B+in->origin[0])*inH+H+in->origin[1]/8)*inW+W+in->origin[2]/8)*inD)+in->slice+D+in->origin[3]/32;
        int64_t ox=((((int64_t)(B+out->origin[0])*oH+H+out->origin[1]/8)*oW+2*W+out->origin[2]/4)*oD)+out->slice+D+out->origin[3]/32;
        const HVX_Vector *pi=(const HVX_Vector *)block_ptr((const uint8_t *const *)in->pools,in->blocks,in->block_count,ix);
        HVX_Vector *pa=(HVX_Vector *)block_ptr((const uint8_t *const *)out->pools,out->blocks,out->block_count,ox);
        HVX_Vector *pb=(HVX_Vector *)block_ptr((const uint8_t *const *)out->pools,out->blocks,out->block_count,ox+oD);
        if(!pa&&!pb)continue;
        if(f16)block16(pi,pa,pb,m2,p->shift);else block32(pi,pa,pb,p);
    }
    return 1;
}
