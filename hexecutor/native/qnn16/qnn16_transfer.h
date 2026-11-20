/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Synchronous QNN FP16 VTCM transfer helpers.
 *
 * The run-list engine supplies exact byte counts from decoded tensor
 * placements. These helpers do not infer logical tensor sizes: packed Xpe/Xwe
 * weights and Fi tables can reserve more bytes than their logical dimensions.
 */
#ifndef QNN16_TRANSFER_H
#define QNN16_TRANSFER_H
#include <stdint.h>

/* Copy `bytes` from each source chunk to the corresponding VTCM destination.
 * Source chunks begin `src_stride` bytes apart; a zero stride means `bytes`.
 * Returns 1 on success, 0 for an invalid descriptor. */
int qnn16_copy_chunks(const void *src, uint32_t src_stride,
                      uint8_t *const *dst, uint32_t count, uint32_t bytes);

/* Clear each destination chunk. Returns 1 on success, 0 for invalid input. */
int qnn16_zero_chunks(uint8_t *const *dst, uint32_t count, uint32_t bytes);

/* A synchronous helper has completed all stores when it returns. This fence
 * is for callers that mix it with DSP DMA or HMX work before a checkpoint. */
void qnn16_transfer_fence(void);

#endif
