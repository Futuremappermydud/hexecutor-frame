/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): concatenation along H of tile tensors.
 *
 * Output row y is row r of one input; with the same width, channels and column parity on both sides the row's
 * vectors are copied pair by pair, masked to the logical output. Rows are shared out as part::parts. */
#include "qnn16_tile_priv.h"

int qnn16_tile_concat_h_ok(const Qnn16Tensor *in, uint32_t count, const Qnn16Tensor *out) {
    uint32_t used[Q16_TILE_VTCM_BLOCKS / 32u];
    uint64_t rows = 0;
    if (!count || !blocked_any(out) || (out->padding.w & 1u) || out->base_bytes > Q16_TILE_VTCM_BLOCKS * 2048u)
        return 0;
    /* The copy is not alias safe: no input may share a block with the output. */
    for (uint32_t i = 0; i < Q16_TILE_VTCM_BLOCKS / 32u; i++)
        used[i] = 0;
    for (uint32_t i = 0; i < out->block_count; i++) {
        const uint32_t off = out->block_offsets[i];
        if (off != NONE)
            used[off >> 16] |= 1u << ((off >> 11) & 31u);
    }
    for (uint32_t k = 0; k < count; k++) {
        const Qnn16Tensor *x = &in[k];
        if (!blocked_any(x) || x->base != out->base || x->dims.b != out->dims.b || x->dims.w != out->dims.w ||
            x->dims.c != out->dims.c || (x->padding.w & 1u))
            return 0;
        for (uint32_t i = 0; i < x->block_count; i++) {
            const uint32_t off = x->block_offsets[i];
            if (off != NONE && (off >= Q16_TILE_VTCM_BLOCKS * 2048u || (used[off >> 16] >> ((off >> 11) & 31u)) & 1u))
                return 0;
        }
        rows += x->dims.h;
    }
    return rows == out->dims.h;
}

void qnn16_tile_concat_h(const Qnn16Tensor *in, uint32_t count, const Qnn16Tensor *out, uint32_t part,
                         uint32_t parts) {
    const Map mo = map_of(out);
    const uint32_t groups = (out->dims.c + 31u) / 32u, pairs = (out->dims.w + 1u) / 2u, po = out->padding.w / 2u;
    uint32_t slot = 0, first = 0;
    for (uint32_t k = 0; k < count; k++) {
        const Qnn16Tensor *x = &in[k];
        const Map mx = map_of(x);
        const uint32_t px = x->padding.w / 2u;
        for (uint32_t b = 0; b < out->dims.b; b++)
            for (uint32_t r = 0; r < x->dims.h; r++) {
                PART_ROW(slot);
                const uint32_t fx = r + x->padding.h, fo = first + r + out->padding.h;
                for (uint32_t g = 0; g < groups; g++) {
                    const HVX_VectorPred qc = channel_mask(out, g);
                    for (uint32_t p = 0; p < pairs; p++)
                        store_live(vec(&mo, b + out->padding.b, fo, (int32_t)(po + p), g),
                                   ld(&mx, b + x->padding.b, fx, (int32_t)(px + p), g), lanes_of(out, po + p), qc);
                }
            }
        first += x->dims.h;
    }
}
