/* HVX tile form (qnn16_tile.h): decimate. */
#include "qnn16_tile_priv.h"

/* out(b, y, x, c) = in(b, y * 2, x * 2, c) for sy = sx = 2. Output pair p
 * (output pixels 2p, 2p + 1) takes input pixels 4p and 4p + 2, which are the
 * even lanes of input pairs 2p and 2p + 1 (plus the input column padding). */
int qnn16_tile_decimate(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t sy, uint32_t sx, uint32_t part,
                        uint32_t parts) {
    uint32_t slot = 0;
    if (sy != 2u || sx != 2u || !blocked_any(x) || !blocked_any(out) || x->dims.b != out->dims.b ||
        x->dims.c != out->dims.c || out->dims.h != (x->dims.h + 1u) / 2u || out->dims.w != (x->dims.w + 1u) / 2u ||
        (x->padding.w & 1u) || (out->padding.w & 1u))
        return 0;
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t npairs = (out->dims.w + 1u) / 2u;
    const uint32_t pin = x->padding.w / 2u, pout = out->padding.w / 2u;
    for (uint32_t g = 0; g < (out->dims.c + 31u) / 32u; g++) {
        const HVX_VectorPred qc = channel_mask(out, g);
        for (uint32_t b = 0; b < out->dims.b; b++)
            for (uint32_t y = 0; y < out->dims.h; y++) {
                PART_ROW(slot);
                const uint32_t iy = 2u * y + x->padding.h;
                for (uint32_t p = 0; p < npairs; p++) {
                    const int32_t ip = (int32_t)(2u * p + pin);
                    const HVX_Vector a = ld(&mx, b + x->padding.b, iy, ip, g);
                    const HVX_Vector c = ld(&mx, b + x->padding.b, iy, ip + 1, g);
                    const HVX_Vector v = Q6_Vh_vshuffe_VhVh(c, a);
                    const uint32_t op = p + pout;
                    store_live(vec(&mo, b + out->padding.b, y + out->padding.h, (int32_t)op, g), v,
                               lanes_of(out, op), qc);
                }
            }
    }
    return 1;
}
