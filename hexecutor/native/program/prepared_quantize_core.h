/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_QUANTIZE_CORE_H
#define NPRG2_PREPARED_QUANTIZE_CORE_H
#include <stddef.h>
#include <stdint.h>

/* Numerical core extracted from the recovered CPU executor's Quantize rule.
 * It intentionally has no graph/runtime dependency. */
int nprg2_quantize_f32_to_u16(const float *src, size_t src_bytes,
                           uint16_t *dst, size_t dst_bytes,
                           uint32_t count, uint16_t zero_point,
                           float output_scale);
#endif
