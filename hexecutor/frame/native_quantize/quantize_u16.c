#include "quantize_u16.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

int nq_quantize_f32_to_u16(const float *src, size_t src_bytes,
                           uint16_t *dst, size_t dst_bytes,
                           uint32_t count, uint16_t zero_point,
                           float output_scale) {
    if (!src || !dst || count != 32u || src_bytes != 128u || dst_bytes != 64u ||
        !isfinite(output_scale) || output_scale <= 0.0f) return 0;
    uintptr_t s=(uintptr_t)src, d=(uintptr_t)dst;
    if (src_bytes > UINTPTR_MAX-s || dst_bytes > UINTPTR_MAX-d ||
        (s < d+dst_bytes && d < s+src_bytes)) return 0;
    /* Match the CPU reference: reciprocal is rounded to float32 once, then
     * the product, +0.5, floor, zero point and clamp are evaluated in double. */
    const double inv=(double)(float)(1.0/(double)output_scale);
    for (uint32_t i=0;i<count;i++) if (!isfinite(src[i])) return 0;
    for (uint32_t i=0;i<count;i++) {
        double q=floor((double)src[i]*inv+0.5)+(double)zero_point;
        dst[i]=(uint16_t)(q<0.0?0.0:q>65535.0?65535.0:q);
    }
    return 1;
}
