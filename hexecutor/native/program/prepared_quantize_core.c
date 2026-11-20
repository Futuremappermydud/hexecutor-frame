/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_quantize_core.h"
#include <stdint.h>


static int finite_f32(float x){union{float f;uint32_t u;}bits;bits.f=x;return (bits.u&0x7f800000u)!=0x7f800000u;}
int nprg2_quantize_f32_to_u16(const float *src, size_t src_bytes,
                              uint16_t *dst, size_t dst_bytes,
                              uint32_t count, uint16_t zero_point,
                              float output_scale) {
    if (!src || !dst || count != 32u || src_bytes != 128u || dst_bytes != 64u ||
        !finite_f32(output_scale) || output_scale <= 0.0f) return 0;
    uintptr_t s=(uintptr_t)src, d=(uintptr_t)dst;
    if (src_bytes > UINTPTR_MAX-s || dst_bytes > UINTPTR_MAX-d ||
        (s < d+dst_bytes && d < s+src_bytes)) return 0;
    /* Preserve the CPU reference's float32-rounded reciprocal. */
    const float inv_f32=(float)(1.0/(double)output_scale);
    if(!finite_f32(inv_f32))return 0;
    const double inv=(double)inv_f32;
    for(uint32_t i=0;i<count;i++)if(!finite_f32(src[i]))return 0;
    /* For positive q, C's conversion truncates toward zero and therefore
     * equals floor(q). Clamp before conversion to keep the cast in range. */
    for(uint32_t i=0;i<count;i++){
        double q=(double)src[i]*inv+0.5+(double)zero_point;
        dst[i]=(uint16_t)(q<=0.0?0.0:q>=65535.0?65535.0:q);
    }
    return 1;
}
