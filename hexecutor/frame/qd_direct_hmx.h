#ifndef QD_DIRECT_HMX_H
#define QD_DIRECT_HMX_H
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
/* Fused graph croutons -> exact wide HMX A tiles, including kernel taps.
 * h,w are output extents (multiples of eight); c is a multiple of 32.
 * A four-pixel group may cross a graph block: two bounded aligned loads and
 * valign join that boundary. Pixel lanes outside the input view are zero. */
#ifdef QDG_DSP
static uint8_t *dsp_cache_block(uint32_t e,int store);
#endif
static inline uint8_t *dsp_hmx_source(uint8_t *const *pools,uint32_t e,int cached) {
#ifdef QDG_DSP
    if(cached)return dsp_cache_block(e,0);
#else
    (void)cached;
#endif
    return pools[e&127]+(e&~127u);
}
static HVX_Vector dsp_hmx_fragment(uint8_t *const *pools,const uint32_t *blocks,uint32_t count,
        uint32_t slice,uint32_t nbh,uint32_t nbw,uint32_t nbd,const int32_t *origin,
        uint32_t H,uint32_t W,uint32_t cb,uint32_t valid,int cached) {
    HVX_Vector zero=Q6_V_vzero();
    if(!valid)return zero;
    uint32_t row=(origin[0]*nbh+H/8)*nbw,index=(row+W/8)*nbd+slice+origin[3]/32+cb;
    uint32_t e=index<count?blocks[index]:0,pool=e&127,part=W%8;
    const uint8_t *ptr=pool && pools[pool]?dsp_hmx_source(pools,e,cached)+(H%8)*256:NULL;
    HVX_Vector v;
    if(part<=4)v=ptr?*(const HVX_UVector *)(ptr+part*32):zero;
    else {
        HVX_Vector first=ptr?*(const HVX_Vector *)(ptr+128):zero,next=zero;
        if(valid>8-part && W/8+1<nbw) {
            index+=nbd;e=index<count?blocks[index]:0;pool=e&127;
            if(pool && pools[pool])next=*(const HVX_Vector *)(dsp_hmx_source(pools,e,cached)+(H%8)*256);
        }
        v=Q6_V_valign_VVR(next,first,(part-4)*32);
    }
    return valid==4?v:Q6_V_vmux_QVV(Q6_Q_vsetq_R(valid*32),v,zero);
}
static void dsp_direct_hmx_a(uint8_t *dest,uint8_t *const *pools,const uint32_t *blocks,
        uint32_t count,uint32_t slice,const int32_t *m,const int32_t *origin,
        uint32_t h,uint32_t w,uint32_t c,uint32_t kh,uint32_t kw,uint32_t owp,int flatten,int cached) {
    uint32_t nbh=(m[1]+7)/8,nbw=(m[2]+7)/8,nbd=(m[3]+31)/32,cbs=c/32,kb=kh*kw*cbs;
    uint32_t mbs=flatten?h*w/32:h*(owp/32),input_w=w+kw-1;
    HVX_Vector zero=Q6_V_vzero();
    for(uint32_t mb=0;mb<mbs;mb++)for(uint32_t ky=0;ky<kh;ky++)for(uint32_t kx=0;kx<kw;kx++)for(uint32_t cb=0;cb<cbs;cb++)for(uint32_t q=0;q<8;q++) {
        uint32_t pixel=mb*32+q*4,y=flatten?pixel/w:mb/(owp/32),x=flatten?pixel%w:(mb%(owp/32))*32+q*4;
        uint32_t valid=x+kx<input_w?(input_w-x-kx<4?input_w-x-kx:4):0;
        HVX_Vector input=dsp_hmx_fragment(pools,blocks,count,slice,nbh,nbw,nbd,origin,
                                         origin[1]+y+ky,origin[2]+x+kx,cb,valid,cached);
        HVX_VectorPair wide=Q6_W_vshuff_VVR(input,zero,-1);
        uint32_t k=(ky*kw+kx)*cbs+cb;
        HVX_Vector *out=(HVX_Vector *)(dest+((size_t)mb*kb+k)*2048+q*256);
        out[0]=Q6_Vh_vshuff_Vh(Q6_V_lo_W(wide));out[1]=Q6_Vh_vshuff_Vh(Q6_V_hi_W(wide));
    }
}
#endif
