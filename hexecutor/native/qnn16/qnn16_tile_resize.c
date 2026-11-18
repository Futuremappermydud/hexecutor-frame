/* HVX tile form (qnn16_tile.h): resize. */
#include "qnn16_tile_priv.h"

/* Two qf32 vectors holding the even (lo) and odd (hi) pixel of an input pair,
 * one word per channel. */
typedef struct {
    HVX_Vector lo, hi;
} QPairV;

/* Vertical blend of the widened pair (a, b) of two input rows: a * w0 + b * w1
 * when mix is set, else a alone. */
static inline QPairV vblend(HVX_Vector a, HVX_Vector b, int mix, HVX_Vector w0, HVX_Vector w1) {
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    const HVX_VectorPair wa = Q6_Wqf32_vmpy_VhfVhf(a, one);
    QPairV r;
    if (!mix) {
        r.lo = Q6_V_lo_W(wa);
        r.hi = Q6_V_hi_W(wa);
        return r;
    }
    const HVX_VectorPair wb = Q6_Wqf32_vmpy_VhfVhf(b, one);
    r.lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(wa), w0),
                                    Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(wb), w1));
    r.hi = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(wa), w0),
                                    Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(wb), w1));
    return r;
}

/* p * wp + q * wq in qf32. */
static inline HVX_Vector lerp(HVX_Vector p, HVX_Vector q, HVX_Vector wp, HVX_Vector wq) {
    return Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(p, wp), Q6_Vqf32_vmpy_Vqf32Vqf32(q, wq));
}

/* Input pair `pair` of the two rows, blended vertically. */
static inline QPairV load_blend(const Map *mx, uint32_t bx, uint32_t ry0, uint32_t ry1, int32_t pair, uint32_t g,
                                int mix, HVX_Vector w0, HVX_Vector w1) {
    const HVX_Vector a = ld(mx, bx, ry0, pair, g);
    const HVX_Vector b = mix ? ld(mx, bx, ry1, pair, g) : Q6_V_vzero();
    return vblend(a, b, mix, w0, w1);
}

/* Exact 2x horizontal upsample (sx = 0.5, ox = -0.25) with a vertical 0.5 scale
 * and offset oy, clamped to the input rows as in qnn16_tensor_resize.
 *
 * Output pixel 2k   = 0.25 * in[max(k-1,0)] + 0.75 * in[k]
 * Output pixel 2k+1 = 0.75 * in[k] + 0.25 * in[min(k+1,W-1)]
 * Input pair j holds pixels (2j, 2j+1) = (lo_j, hi_j); output pair 2j covers
 * pixels 4j, 4j+1 and pair 2j+1 covers pixels 4j+2, 4j+3. */
int qnn16_tile_resize2x(const Qnn16Tensor *x, const Qnn16Tensor *out, const float v[4], uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    /* v = {sy, oy, sx, ox}. */
    if (v[0] != 0.5f || v[2] != 0.5f || v[3] != -0.25f)
        return 0;
    if (v[1] != v[1] || v[1] > 3.0e38f || v[1] < -3.0e38f)
        return 0;
    if (!blocked(x) || !blocked(out) || x->dims.b != out->dims.b || x->dims.c != out->dims.c ||
        out->dims.w != 2u * x->dims.w || (x->dims.w & 1u) || (x->padding.w & 1u) || (out->padding.w & 1u))
        return 0;

    const uint32_t H = x->dims.h, W = x->dims.w, n = W / 2u, nc = x->dims.c / 32u;
    const uint32_t px = x->padding.w / 2u, po = out->padding.w / 2u;
    const Map mx = map_of(x), my = map_of(out);
    const HVX_Vector q25 = qf_of_sf(splat_sf(0.25f)), q75 = qf_of_sf(splat_sf(0.75f));

    for (uint32_t g = 0; g < nc; g++)
        for (uint32_t b = 0; b < out->dims.b; b++)
            for (uint32_t y = 0; y < out->dims.h; y++) {
                PART_ROW(slot);
                /* Vertical source rows, as in the reference (float32 arithmetic). */
                float fy = (float)y * v[0] + v[1];
                if (fy < 0.0f)
                    fy = 0.0f;
                if (fy > (float)(H - 1u))
                    fy = (float)(H - 1u);
                const uint32_t y0 = (uint32_t)fy, y1 = y0 + 1u < H ? y0 + 1u : y0;
                const float wy = fy - (float)y0;
                const int mix = wy != 0.0f;
                const HVX_Vector w0 = qf_of_sf(splat_sf(1.0f - wy)), w1 = qf_of_sf(splat_sf(wy));
                const uint32_t bx = b + x->padding.b, ry0 = y0 + x->padding.h, ry1 = y1 + x->padding.h;
                const uint32_t bo = b + out->padding.b, fyo = y + out->padding.h;

                /* Software pipeline over input pairs: cur = pair j, nx = pair j + 1. */
                QPairV cur = load_blend(&mx, bx, ry0, ry1, (int32_t)px, g, mix, w0, w1);
                HVX_Vector prev_hi = cur.lo; /* hi_{-1} := lo_0 */
                for (uint32_t j = 0; j < n; j++) {
                    QPairV nx = cur;
                    HVX_Vector nlo = cur.hi; /* lo_{n} := hi_{n-1} */
                    if (j + 1u < n) {
                        nx = load_blend(&mx, bx, ry0, ry1, (int32_t)(px + j + 1u), g, mix, w0, w1);
                        nlo = nx.lo;
                    }
                    const HVX_Vector e0 = lerp(prev_hi, cur.lo, q25, q75);  /* pixel 4j   */
                    const HVX_Vector o0 = lerp(cur.lo, cur.hi, q75, q25);   /* pixel 4j+1 */
                    const HVX_Vector e1 = lerp(cur.lo, cur.hi, q25, q75);   /* pixel 4j+2 */
                    const HVX_Vector o1 = lerp(cur.hi, nlo, q75, q25);      /* pixel 4j+3 */
                    const uint32_t p0 = po + 2u * j, p1 = p0 + 1u;
                    put(vec(&my, bo, fyo, (int32_t)p0, g), Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(o0, e0)),
                        lanes_of(out, p0));
                    put(vec(&my, bo, fyo, (int32_t)p1, g), Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(o1, e1)),
                        lanes_of(out, p1));
                    prev_hi = cur.hi;
                    cur = nx;
                }
            }
    return 1;
}
