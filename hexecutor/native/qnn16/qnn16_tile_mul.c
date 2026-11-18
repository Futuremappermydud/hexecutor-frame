/* HVX tile form (qnn16_tile.h): mul. */
#include "qnn16_tile_priv.h"

static inline HVX_Vector mul_hf(HVX_Vector x, HVX_Vector y) {
    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(x, y));
}
static inline int same_dims(const Qnn16Shape p, const Qnn16Shape q) {
    return p.b == q.b && p.h == q.h && p.w == q.w && p.c == q.c;
}
/* b's factors are few (one per column or row) and b may live in the cached
 * scratch arena, where HVX must not read: copy them with scalar loads. */
static inline void factors(uint16_t *dst, const Qnn16Tensor *b, uint32_t n) {
    const uint16_t *src = (const uint16_t *)(b->base + b->direct_offset);
    for (uint32_t i = 0; i < n; i++)
        dst[i] = src[i];
    for (uint32_t i = n; i < 64u; i++)
        dst[i] = 0;
}

int qnn16_tile_mul(const Qnn16Tensor *a, const Qnn16Tensor *b, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    if (!a || !b || !out || !blocked_any(out) || !same_dims(a->dims, out->dims) || a->padding.w != out->padding.w)
        return 0;
    const Qnn16Shape d = out->dims;
    const uint32_t P = out->padding.w;
    if (P & 1u)
        return 0;
    const uint32_t p0 = P / 2u, p1 = (P + d.w + 1u) / 2u, ng = (d.c + 31u) / 32u;
    const Map mo = map_of(out);
    uint32_t slot = 0;

    /* 1. Same grid for all three: b has the shape of a, in tiles. */
    if (blocked_any(a) && blocked_any(b) && same_dims(b->dims, d) && b->padding.w == P) {
        const Map ma = map_of(a), mb = map_of(b);
        for (uint32_t g = 0; g < ng; g++) {
            const HVX_VectorPred qc = channel_mask(out, g);
            for (uint32_t n = 0; n < d.b; n++)
                for (uint32_t y = 0; y < d.h; y++) {
                    PART_ROW(slot);
                    for (uint32_t pp = p0; pp < p1; pp++) {
                        const HVX_Vector x = ld(&ma, n + a->padding.b, y + a->padding.h, (int32_t)pp, g);
                        const HVX_Vector z = ld(&mb, n + b->padding.b, y + b->padding.h, (int32_t)pp, g);
                        store_live(vec(&mo, n + out->padding.b, y + out->padding.h, (int32_t)pp, g), mul_hf(x, z),
                                   lanes_of(out, pp), qc);
                    }
                }
        }
        return 1;
    }

    /* 2. b direct [1,1,W,1]: one factor per column x. */
    if (blocked_any(a) && b->direct && b->dims.b == 1 && b->dims.h == 1 && b->dims.c == 1 && b->dims.w == d.w &&
        b->full.c == 1 && b->full.w == d.w && b->full.h == 1 && b->full.b == 1 && d.w <= 64u) {
        uint16_t hd[64];
        factors(hd, b, d.w);
        const Map ma = map_of(a);
        for (uint32_t g = 0; g < ng; g++) {
            const HVX_VectorPred qc = channel_mask(out, g);
            for (uint32_t n = 0; n < d.b; n++)
                for (uint32_t y = 0; y < d.h; y++) {
                    PART_ROW(slot);
                    for (uint32_t pp = p0; pp < p1; pp++) {
                        const int32_t q = (int32_t)pp - (int32_t)(P / 2u);
                        const uint32_t word = (q >= 0 && q < 32) ? (uint32_t)hd[2 * q] | (uint32_t)hd[2 * q + 1] << 16 : 0u;
                        const HVX_Vector m = Q6_V_vsplat_R((int)word);
                        const HVX_Vector x = ld(&ma, n + a->padding.b, y + a->padding.h, (int32_t)pp, g);
                        store_live(vec(&mo, n + out->padding.b, y + out->padding.h, (int32_t)pp, g), mul_hf(x, m),
                                   lanes_of(out, pp), qc);
                    }
                }
        }
        return 1;
    }

    /* 3. b direct [1,H,1,1]: one factor per row y. */
    if (blocked_any(a) && b->direct && b->dims.b == 1 && b->dims.w == 1 && b->dims.c == 1 && b->dims.h == d.h &&
        b->full.w == 1 && b->full.c == 1 && b->full.h == d.h && b->full.b == 1 && d.h <= 64u) {
        uint16_t hd[64];
        factors(hd, b, d.h);
        const Map ma = map_of(a);
        for (uint32_t g = 0; g < ng; g++) {
            const HVX_VectorPred qc = channel_mask(out, g);
            for (uint32_t n = 0; n < d.b; n++)
                for (uint32_t y = 0; y < d.h; y++) {
                    PART_ROW(slot);
                    const HVX_Vector m = Q6_V_vsplat_R((int)((uint32_t)hd[y] * 0x10001u));
                    for (uint32_t pp = p0; pp < p1; pp++) {
                        const HVX_Vector x = ld(&ma, n + a->padding.b, y + a->padding.h, (int32_t)pp, g);
                        store_live(vec(&mo, n + out->padding.b, y + out->padding.h, (int32_t)pp, g), mul_hf(x, m),
                                   lanes_of(out, pp), qc);
                    }
                }
        }
        return 1;
    }

    return 0;
}
