/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "qnn16_transfer.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static int range_fits(uintptr_t base, size_t bytes) {
    return bytes == 0 || base <= UINTPTR_MAX - (bytes - 1);
}

int qnn16_copy_chunks(const void *src, uint32_t src_stride,
                      uint8_t *const *dst, uint32_t count, uint32_t bytes) {
    if (!src || !dst || count == 0 || bytes == 0) return 0;
    if (src_stride == 0) src_stride = bytes;
    if (src_stride < bytes) return 0;

    const uintptr_t source = (uintptr_t)src;
    if (!range_fits(source, bytes) ||
        (count > 1 && src_stride > (UINTPTR_MAX - source - (bytes - 1)) / (count - 1)))
        return 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!dst[i] || !range_fits((uintptr_t)dst[i], bytes)) return 0;
    }

    const uint8_t *in = (const uint8_t *)src;
    for (uint32_t i = 0; i < count; ++i)
        memcpy(dst[i], in + (size_t)i * src_stride, bytes);
    return 1;
}

int qnn16_zero_chunks(uint8_t *const *dst, uint32_t count, uint32_t bytes) {
    if (!dst || count == 0 || bytes == 0) return 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!dst[i] || !range_fits((uintptr_t)dst[i], bytes)) return 0;
    }
    for (uint32_t i = 0; i < count; ++i) memset(dst[i], 0, bytes);
    return 1;
}

void qnn16_transfer_fence(void) { __sync_synchronize(); }
