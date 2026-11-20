/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "sign_blocked.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <limits.h>
#include <stdlib.h>

static int valid(const QdBlockedU16 *t) {
    if(!t||!t->pools||!t->blocks||t->meta[1]>INT32_MAX-7||
       t->meta[2]>INT32_MAX-3||t->meta[3]>INT32_MAX-31)return 0;
    for(int i=0;i<4;i++)if(t->meta[i]<=0||t->dims[i]<0||t->origin[i]<0||
       (int64_t)t->origin[i]+t->dims[i]>t->meta[i])return 0;
    uint64_t f[3]={(uint64_t)(t->meta[1]+7)/8,(uint64_t)(t->meta[2]+3)/4,
                   (uint64_t)(t->meta[3]+31)/32},slots=(uint32_t)t->meta[0];
    for(int i=0;i<3;i++){if(f[i]&&slots>UINT32_MAX/f[i])return 0;slots*=f[i];}
    return slots+t->slice<=UINT32_MAX;
}
static uintptr_t block_start(const QdBlockedU16 *t,int64_t index) {
    if(index<0||(uint64_t)index>=t->block_count)return 0;
    uint32_t e=t->blocks[index],pool=e&127u;
    return pool&&t->pools[pool]?(uintptr_t)t->pools[pool]+(e&~127u):0;
}
static int same_descriptor(const QdBlockedU16 *a,const QdBlockedU16 *b) {
    if(a->pools!=b->pools||a->blocks!=b->blocks||a->block_count!=b->block_count||a->slice!=b->slice)return 0;
    for(int i=0;i<4;i++)if(a->meta[i]!=b->meta[i]||a->dims[i]!=b->dims[i]||a->origin[i]!=b->origin[i])return 0;
    return 1;
}
static int touched_safe(const QdBlockedU16 *in,const QdBlockedU16 *out) {
    if(in->block_count>SIZE_MAX/sizeof(uintptr_t)||out->block_count>SIZE_MAX/sizeof(uintptr_t))return 0;
    uintptr_t *is=malloc((in->block_count?in->block_count:1)*sizeof *is);
    uintptr_t *os=malloc((out->block_count?out->block_count:1)*sizeof *os);
    if(!is||!os){free(is);free(os);return 0;}
    size_t ni=0,no=0;
    for(int B=in->origin[0];B<in->origin[0]+in->dims[0];B++)
      for(int H=in->origin[1]/8;H<=(in->origin[1]+in->dims[1]-1)/8;H++)
       for(int W=in->origin[2]/4;W<=(in->origin[2]+in->dims[2]-1)/4;W++)
        for(int D=in->origin[3]/32;D<=(in->origin[3]+in->dims[3]-1)/32;D++) {
            int64_t ix=(((int64_t)B*((in->meta[1]+7)/8)+H)*((in->meta[2]+3)/4)+W)*((in->meta[3]+31)/32)+in->slice+D;
            uintptr_t a=block_start(in,ix);if(a)is[ni++]=a;
        }
    int ok=1,identical=same_descriptor(in,out);
    for(int B=out->origin[0];B<out->origin[0]+out->dims[0]&&ok;B++)
      for(int H=out->origin[1]/8;H<=(out->origin[1]+out->dims[1]-1)/8&&ok;H++)
       for(int W=out->origin[2]/4;W<=(out->origin[2]+out->dims[2]-1)/4&&ok;W++)
        for(int D=out->origin[3]/32;D<=(out->origin[3]+out->dims[3]-1)/32&&ok;D++) {
            int64_t ix=(((int64_t)B*((out->meta[1]+7)/8)+H)*((out->meta[2]+3)/4)+W)*((out->meta[3]+31)/32)+out->slice+D;
            uintptr_t a=block_start(out,ix);if(!a)continue;
            for(size_t j=0;j<no;j++)if(a<os[j]+2048&&os[j]<a+2048){ok=0;break;}
            if(!identical)for(size_t j=0;j<ni&&ok;j++)if(a<is[j]+2048&&is[j]<a+2048){ok=0;break;}
            if(ok)os[no++]=a;
        }
    free(is);free(os);return ok;
}
static uint8_t *vector_ptr(const QdBlockedU16 *t,int b,int h,int pair_w,int db) {
    int H=h+t->origin[1],B=b+t->origin[0],nbh=(t->meta[1]+7)/8,
        nbw=(t->meta[2]+3)/4,nbd=(t->meta[3]+31)/32;
    int64_t ix=(((int64_t)B*nbh+H/8)*nbw+pair_w/4)*nbd+t->slice+db;
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    if(!pool||!t->pools[pool])return 0;
    return t->pools[pool]+(e&~127u)+(H%8)*256+((pair_w%4)/2)*128;
}
static HVX_Vector splat(int64_t value) {
    uint32_t x=(uint16_t)value;return Q6_V_vsplat_R(x|(x<<16));
}
static HVX_VectorPred prefix(unsigned n) {return n>=128?Q6_Q_not_Q(Q6_Q_vsetq_R(0)):Q6_Q_vsetq_R(n);}
/* Whole-block form: every region is whole 8x4x32 blocks, so a single table
 * lookup serves 16 vectors and no store masks are needed. Absent input blocks
 * read as zeros (the selector then sees x=0), absent output blocks discard
 * their stores. In-place is safe: each vector is read before it is written. */
static int whole_region(const QdBlockedU16 *t) {
    return !(t->origin[1]&7)&&!(t->origin[2]&3)&&!(t->origin[3]&31)&&
           !(t->dims[1]&7)&&!(t->dims[2]&3)&&!(t->dims[3]&31);
}
static uint8_t *block_ptr(const QdBlockedU16 *t,int64_t ix) {
    if(ix<0||(uint64_t)ix>=t->block_count)return 0;
    uint32_t e=t->blocks[ix],pool=e&127u;
    return pool&&t->pools[pool]?t->pools[pool]+(e&~127u):0;
}
static int apply_whole(const QdBlockedU16 *in,const QdBlockedU16 *out,int32_t input_zero,
        const QdSignLevels *levels) {
    const int nB=in->dims[0],nH=in->dims[1]/8,nW=in->dims[2]/4,nD=in->dims[3]/32;
    const int iH=(in->meta[1]+7)/8,iW=(in->meta[2]+3)/4,iD=(in->meta[3]+31)/32;
    const int oH=(out->meta[1]+7)/8,oW=(out->meta[2]+3)/4,oD=(out->meta[3]+31)/32;
    HVX_Vector zp=splat(input_zero),neg=splat(levels->neg),zero=splat(levels->zero),pos=splat(levels->pos);
    for(int B=0;B<nB;B++)for(int H=0;H<nH;H++)for(int W=0;W<nW;W++)for(int D=0;D<nD;D++) {
        int64_t ix=((((int64_t)(B+in->origin[0])*iH+H+in->origin[1]/8)*iW+W+in->origin[2]/4)*iD)+in->slice+D+in->origin[3]/32;
        int64_t ox=((((int64_t)(B+out->origin[0])*oH+H+out->origin[1]/8)*oW+W+out->origin[2]/4)*oD)+out->slice+D+out->origin[3]/32;
        HVX_Vector *po=(HVX_Vector *)block_ptr(out,ox);
        if(!po)continue;
        const HVX_Vector *pi=(const HVX_Vector *)block_ptr(in,ix);
        #pragma unroll
        for(int v=0;v<16;v++) {
            HVX_Vector x=pi?pi[v]:Q6_V_vzero();
            HVX_Vector y=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(x,zp),pos,neg);
            po[v]=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(x,zp),zero,y);
        }
    }
    return 1;
}
int qd_blocked_sign_u16_prepare(const QdBlockedU16 *in,const QdBlockedU16 *out,
        QdBlockedSignU16 *prepared) {
    if(!prepared||!valid(in)||!valid(out))return 0;
    for(int i=0;i<4;i++)if(in->dims[i]!=out->dims[i])return 0;
    if((in->origin[2]&1)!=(out->origin[2]&1)||(in->origin[3]&31)!=(out->origin[3]&31))return 0;
    int empty=0;for(int i=0;i<4;i++)if(!in->dims[i])empty=1;
    if(!empty&&!touched_safe(in,out))return 0;
    prepared->in=*in;prepared->out=*out;prepared->cookie=0x51445347u;return 1;
}
int qd_blocked_sign_u16_apply(const QdBlockedSignU16 *prepared,int32_t input_zero,
        const QdSignLevels *levels) {
    if(!prepared||prepared->cookie!=0x51445347u||!levels||input_zero<0||input_zero>65535||
       levels->neg<0||levels->neg>65535||levels->zero<0||levels->zero>65535||
       levels->pos<0||levels->pos>65535)return 0;
    const QdBlockedU16 *in=&prepared->in,*out=&prepared->out;
    for(int i=0;i<4;i++)if(!in->dims[i])return 1;
    if(whole_region(in)&&whole_region(out))return apply_whole(in,out,input_zero,levels);
    int w0=in->origin[2]&~1,w1=in->origin[2]+in->dims[2],
        d0=in->origin[3]/32,d1=(in->origin[3]+in->dims[3]+31)/32;
    uint32_t z=(uint16_t)input_zero;HVX_Vector zp=Q6_V_vsplat_R(z|(z<<16));
    HVX_Vector neg=splat(levels->neg),zero=splat(levels->zero),pos=splat(levels->pos);
    HVX_VectorPred even=Q6_Q_vcmp_eq_VhVh(Q6_V_vsplat_R(0xffff0000u),Q6_V_vzero());
    for(int b=0;b<in->dims[0];b++)for(int h=0;h<in->dims[1];h++)
      for(int W=w0;W<w1;W+=2)for(int db=d0;db<d1;db++) {
        int outW=W-in->origin[2]+out->origin[2],outDb=db-in->origin[3]/32+out->origin[3]/32;
        uint8_t *src=vector_ptr(in,b,h,W,db),*dst=vector_ptr(out,b,h,outW,outDb);
        if(!dst)continue;
        HVX_Vector x=src?*(const HVX_UVector *)src:Q6_V_vzero();
        HVX_Vector y=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(x,zp),pos,neg);
        y=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(x,zp),zero,y);
        int first=in->origin[3]-db*32,last=first+in->dims[3];
        if(first<0)first=0;if(first>32)first=32;if(last<0)last=0;if(last>32)last=32;
        HVX_VectorPred ch=Q6_Q_and_QQn(prefix((unsigned)last*4),prefix((unsigned)first*4));
        int ev=W>=in->origin[2]&&W<w1,od=W+1>=in->origin[2]&&W+1<w1;
        HVX_VectorPred mask=Q6_Q_and_QQ(ch,ev?even:Q6_Q_vsetq_R(0));
        if(od)mask=Q6_Q_or_QQ(mask,Q6_Q_and_QQ(ch,Q6_Q_not_Q(even)));
        if(ev||od)Q6_vmem_QRIV(mask,(HVX_Vector *)dst,y);
      }
    return 1;
}
