/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): nearest. */
#include "qnn16_tile_priv.h"

/* out(b, y, x, c) = in(b, oy + y / 2, ox + x / 2, c). Output pair p (output
 * pixels 2p, 2p + 1) comes from the single input pixel ox + p, duplicated into
 * both lanes of the output pair. */
int qnn16_tile_resize_nearest(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t oy, uint32_t ox, uint32_t part,
                              uint32_t parts) {
    uint32_t slot = 0;
    if (!blocked_any(x) || !blocked_any(out) || x->dims.b != out->dims.b || x->dims.c != out->dims.c ||
        (out->padding.w & 1u) || oy >= x->dims.h || ox >= x->dims.w || (out->dims.h - 1u) / 2u >= x->dims.h - oy ||
        (out->dims.w - 1u) / 2u >= x->dims.w - ox)
        return 0;
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t npairs = (out->dims.w + 1u) / 2u, pout = out->padding.w / 2u;
    for (uint32_t g = 0; g < (out->dims.c + 31u) / 32u; g++) {
        const HVX_VectorPred qc = channel_mask(out, g);
        for (uint32_t b = 0; b < out->dims.b; b++)
            for (uint32_t y = 0; y < out->dims.h; y++) {
                PART_ROW(slot);
                const uint32_t iy = oy + y / 2u + x->padding.h;
                for (uint32_t p = 0; p < npairs; p++) {
                    const uint32_t fx = ox + p + x->padding.w;
                    const HVX_Vector v = ld(&mx, b + x->padding.b, iy, (int32_t)(fx / 2u), g);
                    const HVX_Vector w = (fx & 1u) ? Q6_Vh_vshuffo_VhVh(v, v) : Q6_Vh_vshuffe_VhVh(v, v);
                    const uint32_t op = p + pout;
                    store_live(vec(&mo, b + out->padding.b, y + out->padding.h, (int32_t)op, g), w, lanes_of(out, op),
                               qc);
                }
            }
    }
    return 1;
}
