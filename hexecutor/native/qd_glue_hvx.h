#ifndef QD_GLUE_HVX_H
#define QD_GLUE_HVX_H
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>
/* Sources may be unaligned or scalar-cached. The general copy owns those
 * checks; the packet transposer below uses explicit vector transfers in TCM. */
static inline void dsp_copy_small(void *d,const void *s,size_t n) { memcpy(d,s,n); }
/* Dense [h][w][c] -> [h][c/32][wp][32]. Source has 128-byte padded storage.
 * Four 32-byte fragments are assembled in registers for each output vector. */
static void dsp_pack_x32(uint8_t *packet,const uint8_t *src,uint32_t h,uint32_t w,uint32_t c,uint32_t wp) {
    for(uint32_t y=0;y<h;y++)for(uint32_t cb=0;cb<c/32;cb++) {
        uint8_t *dst=packet+((size_t)y*(c/32)+cb)*wp*32;
        for(uint32_t x=0;x<wp;x+=4) {
            HVX_Vector v=Q6_V_vzero();
            for(uint32_t j=0;j<4 && x+j<w;j++) {
                const uint8_t *s=src+((size_t)y*w+x+j)*c+cb*32;
                uint32_t part=(uintptr_t)s&127,q=j*32;
                HVX_Vector a=*(const HVX_Vector *)((uintptr_t)s&~127u);
                HVX_VectorPred mask=j==3?Q6_Q_not_Q(Q6_Q_vsetq_R(96)):
                    Q6_Q_and_QQn(Q6_Q_vsetq_R(q+32),Q6_Q_vsetq_R(q));
                v=Q6_V_vmux_QVV(mask,Q6_V_vror_VR(a,(part-q)&127),v);
            }
            uint32_t take=wp-x<4?wp-x:4;
            if(take==4)*(HVX_UVector *)(dst+x*32)=v;
            else {uint8_t tail[128] __attribute__((aligned(128)));*(HVX_Vector *)tail=v;memcpy(dst+x*32,tail,take*32);}
        }
    }
}
#ifdef QDG_DSP
static uint8_t *dsp_cache_block(uint32_t e,int store);
#define DSP_BLOCK_PTR(pools,e,store) dsp_cache_block(e,store)
#else
#define DSP_BLOCK_PTR(pools,e,store) ((pools)[(e)&127]+((e)&~127u))
#endif
/* Direct blocked u8 transfer. One vector load supplies each 32-lane fragment;
 * predicated stores preserve all padding and aliased bytes. Every vector read
 * stays inside its addressed 2 KB block. Dense tails use a bounded local copy. */
static void dsp_block_u8(uint8_t *const *pools,const uint32_t *blocks,uint32_t count,uint32_t slice,
        const int32_t *m,const int32_t *dims,const int32_t *origin,uint8_t *values,int store) {
    int32_t nbh=(m[1]+7)/8,nbw=(m[2]+7)/8,nbd=(m[3]+31)/32;
    size_t total=(size_t)dims[0]*dims[1]*dims[2]*dims[3],pos=0;
    HVX_Vector v=Q6_V_vzero();
    uint8_t tail[128] __attribute__((aligned(128)));
    for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++) {
        int32_t H=h+origin[1],row=((b+origin[0])*nbh+H/8)*nbw;
        for(int32_t w=0;w<dims[2];w++) {
            int32_t W=w+origin[2],base=(row+W/8)*nbd+slice+origin[3]/32;
            uint32_t within=(H%8)*256+(W%8)*32;
            for(int32_t d=0;d<dims[3];d+=32,pos+=32) {
                int32_t index=base+d/32;
                uint32_t e=index>=0 && (uint32_t)index<count?blocks[index]:0,pool=e&127;
                uint8_t *p=pool && pools[pool]?DSP_BLOCK_PTR(pools,e,store)+within:NULL;
                uint32_t quarter=pos&127,part=p?((uintptr_t)p&127):0;
                if(store) {
                    if(!quarter) {
                        if(total-pos>=128)v=*(const HVX_UVector *)(values+pos);
                        else {memset(tail,0,128);memcpy(tail,values+pos,total-pos);v=*(const HVX_Vector *)tail;}
                    }
                    if(p) {
                        HVX_VectorPred mask=part==96?Q6_Q_not_Q(Q6_Q_vsetq_R(96)):
                            Q6_Q_and_QQn(Q6_Q_vsetq_R(part+32),Q6_Q_vsetq_R(part));
                        HVX_Vector moved=Q6_V_vror_VR(v,(quarter-part)&127);
                        Q6_vmem_QRIV(mask,(HVX_Vector *)((uintptr_t)p&~127u),moved);
                    }
                } else {
                    HVX_Vector src=p?*(const HVX_Vector *)((uintptr_t)p&~127u):Q6_V_vzero();
                    HVX_VectorPred mask=quarter==96?Q6_Q_not_Q(Q6_Q_vsetq_R(96)):
                        Q6_Q_and_QQn(Q6_Q_vsetq_R(quarter+32),Q6_Q_vsetq_R(quarter));
                    v=Q6_V_vmux_QVV(mask,Q6_V_vror_VR(src,(part-quarter)&127),v);
                    if(quarter==96)*(HVX_UVector *)(values+pos-96)=v;
                }
            }
        }
    }
    if(!store && (total&127)) {*(HVX_Vector *)tail=v;memcpy(values+(total&~127u),tail,total&127);}
}
static void dsp_block_u16(uint8_t *const *pools,const uint32_t *blocks,uint32_t count,uint32_t slice,
        const int32_t *m,const int32_t *dims,const int32_t *origin,uint8_t *values,int store) {
    int32_t nbh=(m[1]+7)/8,nbw=(m[2]+3)/4,nbd=(m[3]+31)/32;
    size_t total=(size_t)dims[0]*dims[1]*dims[2]*dims[3]*2,pos=0;
    HVX_Vector v=Q6_V_vzero();uint8_t tail[128] __attribute__((aligned(128)));
    HVX_VectorPred even=Q6_Q_vcmp_eq_VhVh(Q6_V_vsplat_R(0xffff0000u),Q6_V_vzero());
    for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++) {
        int32_t H=h+origin[1],row=((b+origin[0])*nbh+H/8)*nbw;
        for(int32_t w=0;w<dims[2];w++) {
            int32_t W=w+origin[2],base=(row+W/4)*nbd+slice+origin[3]/32;
            uint32_t within=(H%8)*256+((W%4)/2)*128;
            for(int32_t d=0;d<dims[3];d+=32,pos+=64) {
                int32_t index=base+d/32;uint32_t e=index>=0 && (uint32_t)index<count?blocks[index]:0,pool=e&127;
                HVX_Vector *p=pool && pools[pool]?(HVX_Vector *)(DSP_BLOCK_PTR(pools,e,store)+within):NULL;
                uint32_t part=pos&127;
                if(store) {
                    if(!part){if(total-pos>=128)v=*(const HVX_UVector *)(values+pos);else{memset(tail,0,128);memcpy(tail,values+pos,total-pos);v=*(const HVX_Vector *)tail;}}
                    if(p) {
                        HVX_Vector shuffled=Q6_Vh_vshuff_Vh(Q6_V_vror_VR(v,part));
                        if(W&1)shuffled=Q6_V_vror_VR(shuffled,126);
                        Q6_vmem_QRIV((W&1)?Q6_Q_not_Q(even):even,p,shuffled);
                    }
                } else {
                    HVX_Vector src=p?Q6_Vh_vdeal_Vh(*p):Q6_V_vzero();
                    HVX_VectorPred mask=part?Q6_Q_not_Q(Q6_Q_vsetq_R(64)):Q6_Q_vsetq_R(64);
                    v=Q6_V_vmux_QVV(mask,Q6_V_vror_VR(src,((W&1)*64-part)&127),v);
                    if(part)*(HVX_UVector *)(values+pos-64)=v;
                }
            }
        }
    }
    if(!store && (total&127)){*(HVX_Vector *)tail=v;memcpy(values+(total&~127u),tail,total&127);}
}
/* Low 32 bits of word * unsigned word. Splitting the coefficient into two
 * unsigned halfwords avoids signed-halfword overflow. The caller proves the
 * complete u8 epilogue fits uint32, exactly as the ARM fast path does. */
static HVX_Vector dsp_mul_u32(HVX_Vector a,uint32_t m) {
    HVX_Vector lo=Q6_V_vsplat_R(m&65535),hi=Q6_V_vsplat_R(m>>16);
    return Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(a,lo),
            Q6_Vw_vasl_VwR(Q6_Vw_vmpyie_VwVuh(a,hi),16));
}
static HVX_Vector dsp_add_words(HVX_Vector a,HVX_Vector b,uint32_t ma,uint32_t mb,uint32_t rd,uint32_t shift) {
    HVX_Vector sum=Q6_Vw_vadd_VwVw(dsp_mul_u32(a,ma),dsp_mul_u32(b,mb));
    sum=Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(sum,Q6_V_vsplat_R(rd)),shift);
    /* shift >= 1 leaves a nonnegative signed word after the logical shift. */
    return Q6_Vw_vmin_VwVw(sum,Q6_V_vsplat_R(255));
}
static void dsp_add_u8(uint8_t *out,const uint8_t *a,const uint8_t *b,size_t count,uint32_t ma,uint32_t mb,uint32_t shift) {
    uint32_t rd=1u<<(shift-1);size_t i=0;
    for(;i+128<=count;i+=128) {
        HVX_VectorPair ah=Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)(a+i));
        HVX_VectorPair bh=Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)(b+i));
        HVX_VectorPair a0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(ah)),a1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(ah));
        HVX_VectorPair b0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(bh)),b1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(bh));
        HVX_Vector q0=dsp_add_words(Q6_V_lo_W(a0),Q6_V_lo_W(b0),ma,mb,rd,shift);
        HVX_Vector q1=dsp_add_words(Q6_V_hi_W(a0),Q6_V_hi_W(b0),ma,mb,rd,shift);
        HVX_Vector q2=dsp_add_words(Q6_V_lo_W(a1),Q6_V_lo_W(b1),ma,mb,rd,shift);
        HVX_Vector q3=dsp_add_words(Q6_V_hi_W(a1),Q6_V_hi_W(b1),ma,mb,rd,shift);
        HVX_Vector lo=Q6_Vh_vpack_VwVw_sat(q1,q0),hi=Q6_Vh_vpack_VwVw_sat(q3,q2);
        *(HVX_UVector *)(out+i)=Q6_Vub_vpack_VhVh_sat(hi,lo);
    }
    for(;i<count;i++){uint32_t v=(a[i]*ma+b[i]*mb+rd)>>shift;out[i]=v>255?255:v;}
}

static int dsp_affine_fits(int64_t m,int64_t zin,int64_t zout,int64_t shift) {
    if(shift<1 || shift>31 || m<INT32_MIN || m>INT32_MAX || zin<0 || zin>65535 || zout<0 || zout>65535)return 0;
    int64_t a=-zin*m+(1ll<<(shift-1)),b=(65535-zin)*m+(1ll<<(shift-1));
    return a>=INT32_MIN && a<=INT32_MAX && b>=INT32_MIN && b<=INT32_MAX;
}
static HVX_Vector dsp_affine_vec(HVX_Vector input,int32_t m,int32_t zin,int32_t zout,uint32_t shift) {
    HVX_VectorPair w=Q6_Wuw_vunpack_Vuh(input);HVX_Vector z=Q6_V_vsplat_R(zin),rd=Q6_V_vsplat_R(1u<<(shift-1));
    HVX_Vector a=Q6_Vw_vadd_VwVw(dsp_mul_u32(Q6_Vw_vsub_VwVw(Q6_V_lo_W(w),z),(uint32_t)m),rd);
    HVX_Vector b=Q6_Vw_vadd_VwVw(dsp_mul_u32(Q6_Vw_vsub_VwVw(Q6_V_hi_W(w),z),(uint32_t)m),rd);
    a=Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(a,shift),Q6_V_vsplat_R(zout));
    b=Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(b,shift),Q6_V_vsplat_R(zout));
    return Q6_Vuh_vpack_VwVw_sat(b,a);
}
/* The caller proves both extreme inputs fit signed 32 bits before rounding.
 * The fallback tail uses the original signed 64-bit expression. */
static void dsp_affine_u16(void *out,const uint16_t *input,size_t count,int32_t m,int32_t zin,int32_t zout,uint32_t shift,int to8) {
    size_t i=0;
    if(to8) {
        for(;i+128<=count;i+=128) {
            HVX_Vector a=dsp_affine_vec(*(const HVX_UVector *)(input+i),m,zin,zout,shift);
            HVX_Vector b=dsp_affine_vec(*(const HVX_UVector *)(input+i+64),m,zin,zout,shift);
            a=Q6_Vuh_vmin_VuhVuh(a,Q6_V_vsplat_R(0x00ff00ff));b=Q6_Vuh_vmin_VuhVuh(b,Q6_V_vsplat_R(0x00ff00ff));
            *(HVX_UVector *)((uint8_t *)out+i)=Q6_Vub_vpack_VhVh_sat(b,a);
        }
    } else for(;i+64<=count;i+=64)
        *(HVX_UVector *)((uint16_t *)out+i)=dsp_affine_vec(*(const HVX_UVector *)(input+i),m,zin,zout,shift);
    for(;i<count;i++) {
        int64_t v=((((int64_t)input[i]-zin)*m+(1ll<<(shift-1)))>>shift)+zout,hi=to8?255:65535;
        v=v<0?0:v>hi?hi:v;
        if(to8)((uint8_t *)out)[i]=(uint8_t)v;else ((uint16_t *)out)[i]=(uint16_t)v;
    }
}
static HVX_Vector dsp_add16_words(HVX_Vector a,HVX_Vector b,uint32_t ma,uint32_t mb,uint32_t shift) {
    HVX_Vector v=Q6_Vw_vadd_VwVw(dsp_mul_u32(a,ma),dsp_mul_u32(b,mb));
    v=Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(v,Q6_V_vsplat_R(1u<<(shift-1))),shift);
    return Q6_Vw_vmin_VwVw(v,Q6_V_vsplat_R(65535));
}
static void dsp_add_u16(uint16_t *out,const uint16_t *a,const uint16_t *b,size_t count,uint32_t ma,uint32_t mb,uint32_t shift) {
    size_t i=0;
    for(;i+64<=count;i+=64) {
        HVX_VectorPair av=Q6_Wuw_vunpack_Vuh(*(const HVX_UVector *)(a+i)),bv=Q6_Wuw_vunpack_Vuh(*(const HVX_UVector *)(b+i));
        HVX_Vector lo=dsp_add16_words(Q6_V_lo_W(av),Q6_V_lo_W(bv),ma,mb,shift);
        HVX_Vector hi=dsp_add16_words(Q6_V_hi_W(av),Q6_V_hi_W(bv),ma,mb,shift);
        *(HVX_UVector *)(out+i)=Q6_Vuh_vpack_VwVw_sat(hi,lo);
    }
    for(;i<count;i++){uint32_t v=(a[i]*ma+b[i]*mb+(1u<<(shift-1)))>>shift;out[i]=v>65535?65535:v;}
}
#endif
