#ifndef QD_PACK_HVX_H
#define QD_PACK_HVX_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
/* Source and the two temporary vectors are VTCM, c is a multiple of four.
 * Gather each pixel's two pairs of u16 values, deinterleave byte planes, then
 * interleave the pairs into the convolution's four-channel packet layout. */
static void dsp_pack_u16_gather(uint8_t *packet,size_t plane,const uint16_t *source,
        uint32_t h,uint32_t w,uint32_t c,uint32_t wp,HVX_Vector *temporary) {
    int32_t offsets[32] __attribute__((aligned(128)));
    for(uint32_t i=0;i<32;i++)offsets[i]=i*c*2;
    HVX_Vector index=*(HVX_Vector *)offsets,zero=Q6_V_vzero();
    for(uint32_t y=0;y<h;y++)for(uint32_t g=0;g<c/4;g++)for(uint32_t x=0;x<wp;x+=32) {
        uint32_t valid=x<w?(w-x<32?w-x:32):0,take=wp-x<32?wp-x:32;
        HVX_Vector low=zero,high=zero;
        if(valid) {
            uint32_t base=(uint32_t)(source+((size_t)y*w+x)*c+g*4),range=(valid-1)*c*2+7;
            if(valid==32) {
                Q6_vgather_ARMVw(temporary,base,range,index);
                Q6_vgather_ARMVw(temporary+1,base+4,range-4,index);
            } else {
                HVX_VectorPred mask=Q6_Q_vsetq_R(valid*4);
                Q6_vgather_AQRMVw(temporary,mask,base,range,index);
                Q6_vgather_AQRMVw(temporary+1,mask,base+4,range-4,index);
            }
            HVX_Vector a=*(volatile HVX_Vector *)temporary,b=*(volatile HVX_Vector *)(temporary+1);
            if(valid!=32) {
                HVX_VectorPred mask=Q6_Q_vsetq_R(valid*4);
                a=Q6_V_vmux_QVV(mask,a,zero);b=Q6_V_vmux_QVV(mask,b,zero);
            }
            HVX_VectorPair v=Q6_W_vshuff_VVR(Q6_Vb_vdeal_Vb(b),Q6_Vb_vdeal_Vb(a),-2);
            low=Q6_V_lo_W(v);high=Q6_V_hi_W(v);
        }
        uint8_t *dst=packet+(((size_t)y*(c/4)+g)*wp+x)*4;
        if(take==32){*(HVX_UVector *)dst=low;*(HVX_UVector *)(dst+plane)=high;}
        else {
            uint8_t tail[128] __attribute__((aligned(128)));
            *(HVX_Vector *)tail=low;memcpy(dst,tail,take*4);
            *(HVX_Vector *)tail=high;memcpy(dst+plane,tail,take*4);
        }
    }
}
#endif
