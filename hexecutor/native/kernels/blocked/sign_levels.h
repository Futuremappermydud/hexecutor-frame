/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_SIGN_LEVELS_H
#define QD_SIGN_LEVELS_H
#include <stddef.h>
#include <stdint.h>

/* Three exact precomputed outputs for k_sign's s=-1,0,+1 cases. Values are
 * clipped to the old kernel's u16 range but intentionally not cast to the
 * output dtype; the caller must still use its normal from_i64 conversion. */
typedef struct { int64_t neg, zero, pos; } QdSignLevels;

int qd_sign_prepare_levels(double output_scale,int32_t output_zero,QdSignLevels *levels);
int64_t qd_sign_select_u16(uint16_t input,int32_t input_zero,const QdSignLevels *levels);
/* Dense u16 selector. Exact in-place is supported; partial overlap is not.
 * Unaligned buffers use the scalar selector. */
int qd_sign_apply_u16_u16(const uint16_t *input,uint16_t *output,size_t count,
        int32_t input_zero,const QdSignLevels *levels);

#endif
