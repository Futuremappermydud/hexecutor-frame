/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_LAYOUT_AUDIT_INPUT_SLICE_PAD_DIRECT_H
#define QD_LAYOUT_AUDIT_INPUT_SLICE_PAD_DIRECT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Exact direct-output implementation of the resident k_input_slice_pad copy.
 * Returns 1 when it handled the operation, 0 when the caller must use its
 * existing fallback. All rejection checks happen before the first store.
 *
 * The output is contiguous NHWC u8, as required by QDR_DIRECT. Pad values and
 * starts follow qd_graph.c: lo = start - pad_before; the intersection with
 * the source is copied and all other output bytes receive `fill`.
 */
static inline int qd_layout_input_slice_pad_direct_u8(
        uint8_t *dst, size_t dst_bytes,
        const uint8_t *src, size_t src_bytes,
        const int32_t in[4], const int32_t out[4],
        const int32_t start[4], const int32_t pad_before[4], uint8_t fill) {
    size_t in_count = 1, out_count = 1;
    int32_t lo[4], s0[4], s1[4];
    if (!dst || !src || !in || !out || !start || !pad_before) return 0;
    for (int i = 0; i < 4; ++i) {
        if (in[i] < 0 || out[i] < 0) return 0;
        if ((size_t)in[i] && in_count > SIZE_MAX / (size_t)in[i]) return 0;
        if ((size_t)out[i] && out_count > SIZE_MAX / (size_t)out[i]) return 0;
        in_count *= (size_t)in[i];
        out_count *= (size_t)out[i];
        int64_t l = (int64_t)start[i] - pad_before[i];
        int64_t end = l + out[i];
        if (l < INT32_MIN || l > INT32_MAX) return 0;
        lo[i] = (int32_t)l;
        s0[i] = lo[i] > 0 ? lo[i] : 0;
        s1[i] = end < in[i] ? (int32_t)end : in[i];
        if (s1[i] < s0[i]) s1[i] = s0[i];
    }
    if (in_count != src_bytes || out_count != dst_bytes) return 0;
    /* The resident fallback materializes before t_write; aliasing must fall
     * back because the direct fill could otherwise destroy unread input. */
    uintptr_t db = (uintptr_t)dst, sb = (uintptr_t)src;
    if (dst_bytes > UINTPTR_MAX - db || src_bytes > UINTPTR_MAX - sb) return 0;
    if (db < sb + src_bytes && sb < db + dst_bytes) return 0;

    memset(dst, fill, dst_bytes);
    for (int i = 0; i < 4; ++i) if (s1[i] <= s0[i]) return 1;
    const size_t in_row = (size_t)in[2] * (size_t)in[3];
    const size_t span = (size_t)(s1[3] - s0[3]);
    const size_t dst_channel = (size_t)(s0[3] - lo[3]);
    const int full_channel_row = lo[3] == 0 && s0[3] == 0
                              && s1[3] == in[3] && in[3] == out[3];
    if (full_channel_row) {
        const size_t row_bytes = (size_t)(s1[2] - s0[2]) * (size_t)in[3];
        for (int32_t b = s0[0]; b < s1[0]; ++b)
            for (int32_t h = s0[1]; h < s1[1]; ++h) {
                size_t si = ((size_t)b * (size_t)in[1] + (size_t)h) * in_row
                          + (size_t)s0[2] * (size_t)in[3];
                size_t di = (((size_t)(b - lo[0]) * (size_t)out[1]
                              + (size_t)(h - lo[1])) * (size_t)out[2]
                              + (size_t)(s0[2] - lo[2])) * (size_t)out[3];
                memcpy(dst + di, src + si, row_bytes);
            }
        return 1;
    }
    for (int32_t b = s0[0]; b < s1[0]; ++b)
        for (int32_t h = s0[1]; h < s1[1]; ++h)
            for (int32_t w = s0[2]; w < s1[2]; ++w) {
                size_t si = (((size_t)b * (size_t)in[1] + (size_t)h) * in_row)
                          + (size_t)w * (size_t)in[3] + (size_t)s0[3];
                size_t di = ((((size_t)(b - lo[0]) * (size_t)out[1]
                               + (size_t)(h - lo[1])) * (size_t)out[2]
                               + (size_t)(w - lo[2])) * (size_t)out[3]
                              + dst_channel);
                memcpy(dst + di, src + si, span);
            }
    return 1;
}

#endif
