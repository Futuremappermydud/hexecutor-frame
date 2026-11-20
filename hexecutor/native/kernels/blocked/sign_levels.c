/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "sign_levels.h"
#include <math.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static int64_t old_sign_round(int32_t sign,double scale,int32_t zero) {
    double y=floor((double)sign/scale+zero+0.5);
    if(y<0)return 0;
    if(y>65535)return 65535;
    return (int64_t)y;
}

int qd_sign_prepare_levels(double output_scale,int32_t output_zero,QdSignLevels *levels) {
    if(!levels||!isfinite(output_scale)||output_scale<=0)return 0;
    levels->neg=old_sign_round(-1,output_scale,output_zero);
    levels->zero=old_sign_round(0,output_scale,output_zero);
    levels->pos=old_sign_round(1,output_scale,output_zero);
    return 1;
}

int64_t qd_sign_select_u16(uint16_t input,int32_t input_zero,const QdSignLevels *levels) {
    int64_t delta=(int64_t)input-input_zero;
    return delta<0?levels->neg:delta>0?levels->pos:levels->zero;
}

#if defined(__hexagon__)
static HVX_Vector splat_u16(int64_t value) {
    uint32_t x=(uint16_t)value;
    return Q6_V_vsplat_R(x|(x<<16));
}
#endif

int qd_sign_apply_u16_u16(const uint16_t *input,uint16_t *output,size_t count,
        int32_t input_zero,const QdSignLevels *levels) {
    if(!levels||((!input||!output)&&count))return 0;
#if !defined(__hexagon__)
    for(size_t i=0;i<count;i++)output[i]=(uint16_t)qd_sign_select_u16(input[i],input_zero,levels);
    return 1;
#else
    if(input_zero<0||input_zero>65535) {
        for(size_t i=0;i<count;i++)output[i]=(uint16_t)qd_sign_select_u16(input[i],input_zero,levels);
        return 1;
    }
    if((uintptr_t)input%128||(uintptr_t)output%128) {
        for(size_t i=0;i<count;i++)output[i]=(uint16_t)qd_sign_select_u16(input[i],input_zero,levels);
        return 1;
    }
    size_t i=0;uint32_t z=(uint16_t)input_zero;HVX_Vector zp=Q6_V_vsplat_R(z|(z<<16));
    HVX_Vector neg=splat_u16(levels->neg),zero=splat_u16(levels->zero),pos=splat_u16(levels->pos);
    for(;i+64<=count;i+=64) {
        HVX_Vector x=*(const HVX_UVector *)(input+i);
        HVX_VectorPred gt=Q6_Q_vcmp_gt_VuhVuh(x,zp);
        HVX_VectorPred eq=Q6_Q_vcmp_eq_VhVh(x,zp);
        HVX_Vector y=Q6_V_vmux_QVV(gt,pos,neg);
        y=Q6_V_vmux_QVV(eq,zero,y);
        *(HVX_UVector *)(output+i)=y;
    }
    for(;i<count;i++)output[i]=(uint16_t)qd_sign_select_u16(input[i],input_zero,levels);
    return 1;
#endif
}
