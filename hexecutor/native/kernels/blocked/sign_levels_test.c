/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "sign_levels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int64_t oracle(uint16_t input,int32_t input_zero,double output_scale,int32_t output_zero) {
    int64_t delta=(int64_t)input-input_zero;
    int64_t sign=delta>0?1:delta<0?-1:0;
    double y=floor((double)sign/output_scale+output_zero+0.5);
    return y<0?0:y>65535?65535:(int64_t)y;
}

int main(void) {
    static const int32_t input_zeroes[]={65432,9196,10701,61378};
    static const double output_scales[]={3.0518509447574615e-05};
    static const int32_t output_zeroes[]={32768};
    uint64_t checked=0;
    for(unsigned q=0;q<sizeof output_scales/sizeof output_scales[0];q++) {
        QdSignLevels levels;
        if(!qd_sign_prepare_levels(output_scales[q],output_zeroes[q],&levels))return 1;
        if(levels.neg!=1||levels.zero!=32768||levels.pos!=65535) {
            printf("unexpected levels: %lld %lld %lld\n",(long long)levels.neg,
                   (long long)levels.zero,(long long)levels.pos);return 2;
        }
        for(unsigned z=0;z<sizeof input_zeroes/sizeof input_zeroes[0];z++)
          for(uint32_t x=0;x<=65535;x++) {
            int64_t expected=oracle((uint16_t)x,input_zeroes[z],output_scales[q],output_zeroes[q]);
            int64_t got=qd_sign_select_u16((uint16_t)x,input_zeroes[z],&levels);
            if(got!=expected) {
                printf("mismatch x%u input_zp%d got%lld want%lld\n",x,input_zeroes[z],
                       (long long)got,(long long)expected);return 3;
            }
            /* Preserve from_i64's cast-after-u16-clipping behavior for u8 too. */
            if((uint8_t)got!=(uint8_t)expected)return 4;
            checked++;
          }
    }
    printf("PASS: %llu exhaustive u16 sign selections, equality and endpoint cases\n",
           (unsigned long long)checked);
    return 0;
}
