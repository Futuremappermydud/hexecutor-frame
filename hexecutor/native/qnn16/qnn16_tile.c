/* HVX depthwise and space-to-depth leaves on FP16 crouton tiles (qnn16_tile.h). */
#include "qnn16_tile_priv.h"


typedef struct {
    HVX_Vector scale, bias, hi; /* qf32 per channel, qf32 per channel, FP16 pairs */
    int clamp;
} Ep;
static Ep epilogue(const uint8_t *fi, uint32_t tables, uint32_t g) {
    /* Word c of the table's first vector holds scale c in its low half; word c
     * of the second, bias c in its high half; a second table's first vector,
     * the clamp in its high half. vshuffe/vshuffo duplicate those halves. */
    const uint8_t *t = fi + g * tables * 256u;
    const HVX_Vector t0 = *(const HVX_UVector *)t, t1 = *(const HVX_UVector *)(t + 128u);
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    Ep e;
    e.scale = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuffe_VhVh(t0, t0), one));
    e.bias = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuffo_VhVh(t1, t1), one));
    if (tables == 2) {
        const HVX_Vector t2 = *(const HVX_UVector *)(t + 256u);
        e.hi = Q6_Vh_vshuffo_VhVh(t2, t2);
    } else
        e.hi = Q6_V_vzero();
    e.clamp = tables == 2;
    return e;
}
static inline HVX_Vector finish(HVX_Vector lo, HVX_Vector hi, const Ep *e) {
    lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(lo, e->scale), e->bias);
    hi = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hi, e->scale), e->bias);
    HVX_Vector y = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
    if (e->clamp) {
        /* Signed halfword order is FP16 order against 0 and a positive bound. */
        y = Q6_Vh_vmax_VhVh(y, Q6_V_vzero());
        y = Q6_Vh_vmin_VhVh(y, e->hi);
    }
    return y;
}
static inline void mac(HVX_Vector *lo, HVX_Vector *hi, HVX_Vector a, HVX_Vector w) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(a, w);
    *lo = Q6_Vqf32_vadd_Vqf32Vqf32(*lo, Q6_V_lo_W(p));
    *hi = Q6_Vqf32_vadd_Vqf32Vqf32(*hi, Q6_V_hi_W(p));
}
/* Store into a group: a partial last group (partial set) also masks its padding channels. */
static inline __attribute__((always_inline)) void put_c(HVX_Vector *p, HVX_Vector y, uint32_t lanes, int partial,
                                                        HVX_VectorPred qc) {
    if (partial)
        store_live(p, y, lanes, qc);
    else
        put(p, y, lanes);
}
static inline __attribute__((always_inline)) void put_row_c(HVX_Vector *const *o, uint32_t i, const Qnn16Tensor *Y,
                                                            uint32_t pp, HVX_Vector y, int partial, HVX_VectorPred qc) {
    put_c(o[i], y, lanes_of(Y, pp), partial, qc);
}

/* Tile operands whose channel count may be a partial last group (dims.c % 32 != 0). */
static int pair_any_ok(const Qnn16Tensor *x, const Qnn16Tensor *y) {
    return blocked_any(x) && blocked_any(y) && x->dims.b == y->dims.b && x->padding.w >= y->padding.w;
}
static uint32_t up32(uint32_t c) { return (c + 31u) / 32u * 32u; }
int qnn16_tile_dw3x3_ok(const Qnn16Tensor *x, const Qnn16Tensor *y) {
    return pair_any_ok(x, y) && up32(x->dims.c) == up32(y->dims.c) && x->dims.h == y->dims.h + 2u &&
           x->dims.w == y->dims.w + 2u;
}
int qnn16_tile_dw3x3_s2_ok(const Qnn16Tensor *x, const Qnn16Tensor *y) {
    return pair_any_ok(x, y) && x->dims.c == 4u * up32(y->dims.c) && x->dims.h >= y->dims.h + 1u &&
           x->dims.w >= y->dims.w + 1u;
}
int qnn16_tile_dw1x1_ok(const Qnn16Tensor *x, const Qnn16Tensor *y) {
    return pair_any_ok(x, y) && up32(x->dims.c) >= up32(y->dims.c) + 32u && x->dims.h == y->dims.h &&
           x->dims.w == y->dims.w;
}
int qnn16_tile_s2d_ok(const Qnn16Tensor *x, const Qnn16Tensor *y) {
    return blocked_any(x) && blocked(y) && x->dims.b == y->dims.b && y->dims.c == 4u * ((x->dims.c + 31u) / 32u * 32u);
}

/* One output row from three input rows' pointer tables (starting at the first
 * pair the row needs). Odd column offset: taps are mid(Pi, Pi+1), Pi+1,
 * mid(Pi+1, Pi+2); even: Pi, mid(Pi, Pi+1), Pi+1. The last tap of a step is
 * the first of the next, so each row costs one load and one shuffle pair. */
static inline __attribute__((always_inline)) void dw3x3_row(HVX_Vector *const *r0, HVX_Vector *const *r1,
                                                            HVX_Vector *const *r2, HVX_Vector *const *o,
                                                            const HVX_Vector *w, const Ep *e, const Qnn16Tensor *Y,
                                                            uint32_t p0, uint32_t np, const int odd, const int clamp,
                                                            const int partial, HVX_VectorPred qc) {
    const HVX_Vector w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7], w8 = w[8];
    const HVX_Vector scale = e->scale, bias = e->bias, top = e->hi, z = Q6_V_vzero();
    /* Only the last pair of a row can be partial (blocked outputs start on an even column). */
    const uint32_t last = lanes_of(Y, p0 + np - 1u), full = np - (last != 3u);
    HVX_Vector a0, a1, b0, b1, c0, c1;
    a1 = *r0[0]; b1 = *r1[0]; c1 = *r2[0];
    if (odd) {
        a0 = *r0[1]; b0 = *r1[1]; c0 = *r2[1];
        a1 = mid(a1, a0); b1 = mid(b1, b0); c1 = mid(c1, c0);
    } else {
        a0 = a1; b0 = b1; c0 = c1;
    }
    for (uint32_t i = 0; i < np; i++) {
        HVX_Vector t1a, t2a, t1b, t2b, t1c, t2c;
        if (odd) {
            HVX_Vector na = *r0[i + 2u], nb = *r1[i + 2u], nc = *r2[i + 2u];
            t1a = a0; t2a = mid(a0, na); a0 = na;
            t1b = b0; t2b = mid(b0, nb); b0 = nb;
            t1c = c0; t2c = mid(c0, nc); c0 = nc;
        } else {
            HVX_Vector na = *r0[i + 1u], nb = *r1[i + 1u], nc = *r2[i + 1u];
            t1a = mid(a0, na); t2a = na; a0 = na;
            t1b = mid(b0, nb); t2b = nb; b0 = nb;
            t1c = mid(c0, nc); t2c = nc; c0 = nc;
        }
        HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(a1, w0);
        HVX_Vector lo = Q6_V_lo_W(m), hi = Q6_V_hi_W(m);
        mac(&lo, &hi, t1a, w1);
        mac(&lo, &hi, t2a, w2);
        mac(&lo, &hi, b1, w3);
        mac(&lo, &hi, t1b, w4);
        mac(&lo, &hi, t2b, w5);
        mac(&lo, &hi, c1, w6);
        mac(&lo, &hi, t1c, w7);
        mac(&lo, &hi, t2c, w8);
        a1 = t2a; b1 = t2b; c1 = t2c;
        lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(lo, scale), bias);
        hi = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hi, scale), bias);
        HVX_Vector y = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
        if (clamp)
            y = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(y, z), top);
        if (i < full && !partial)
            *o[i] = y;
        else
            put_c(o[i], y, i < full ? 3u : last, partial, qc);
    }
}
void qnn16_tile_dw3x3(const Qnn16TileJob *j, uint32_t part, uint32_t parts) {
    const HVX_VectorPred qev = even_words(splat_base());
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t oh = Y->dims.h, d = X->padding.w - Y->padding.w;
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u, np = p1 - p0;
    const int32_t first = (int32_t)((2u * p0 + d) >> 1);
    const uint32_t n = ((2u * p1 + d + 1u) >> 1) - (uint32_t)first + 1u;
    /* Contiguous (group, row) items per worker: setup once per group touched. */
    const uint32_t rows_ = Y->dims.b * oh, groups_ = (Y->dims.c + 31u) / 32u;
    const uint32_t i0_ = groups_ * rows_ * part / parts, i1_ = groups_ * rows_ * (part + 1u) / parts;
    uint32_t it_ = i0_, r = i0_ % rows_;
    for (uint32_t g = i0_ / rows_; it_ < i1_; g++, r = 0) {
        HVX_Vector w[9];
        const int partial = g == groups_ - 1u && (Y->dims.c % 32u) != 0u;
        const HVX_VectorPred qc = channel_mask(Y, g);
        /* t=514 tap vector: word 2j = { w[2j], 0 }, word 2j + 1 = { 0, w[2j + 1] }. */
        for (uint32_t k = 0; k < 9; k++) {
            uint32_t ky = k / 3u, kx = k % 3u, q = ky * 3u + 2u - kx;
            const HVX_Vector v = *(const HVX_UVector *)(j->weights + g * 576u + q * 64u);
            w[k] = Q6_V_vmux_QVV(qev, Q6_Vh_vshuffe_VhVh(v, v), Q6_Vh_vshuffo_VhVh(v, v));
        }
        const Ep e = epilogue(j->fi, j->tables, g);
        for (; r < rows_ && it_ < i1_; r++, it_++) {
            uint32_t b = r / oh, oy = r % oh;
            uint32_t bx = b + X->padding.b, by = b + Y->padding.b;
            uint32_t fyo = oy + Y->padding.h, fyi = oy + X->padding.h;
            HVX_Vector *r0[MAXP], *r1[MAXP], *r2[MAXP], *o[MAXP];
            if (!(Y->padding.w & 1u) && row_ptrs(&mx, bx, fyi, first, n, g, r0) &&
                row_ptrs(&mx, bx, fyi + 1u, first, n, g, r1) && row_ptrs(&mx, bx, fyi + 2u, first, n, g, r2) &&
                row_ptrs(&my, by, fyo, (int32_t)p0, np, g, o)) {
                if (d & 1u) {
                    if (e.clamp)
                        dw3x3_row(r0, r1, r2, o, w, &e, Y, p0, np, 1, 1, partial, qc);
                    else
                        dw3x3_row(r0, r1, r2, o, w, &e, Y, p0, np, 1, 0, partial, qc);
                } else if (e.clamp)
                    dw3x3_row(r0, r1, r2, o, w, &e, Y, p0, np, 0, 1, partial, qc);
                else
                    dw3x3_row(r0, r1, r2, o, w, &e, Y, p0, np, 0, 0, partial, qc);
                continue;
            }
            for (uint32_t pp = p0; pp < p1; pp++) {
                HVX_Vector lo = Q6_V_vzero(), hi = Q6_V_vzero();
                int32_t col = (int32_t)(2u * pp + d);
                for (uint32_t ky = 0; ky < 3; ky++)
                    for (uint32_t kx = 0; kx < 3; kx++)
                        mac(&lo, &hi, tap(&mx, bx, fyi + ky, col + (int32_t)kx, g), w[ky * 3u + kx]);
                put_c(vec(&my, by, fyo, (int32_t)pp, g), finish(lo, hi, &e), lanes_of(Y, pp), partial, qc);
            }
        }
    }
}

void qnn16_tile_dw3x3_s2(const Qnn16TileJob *j, uint32_t part, uint32_t parts) {
    /* Space-to-depth input: tap (ky, kx) reads phase (ky % 2) * 2 + (kx == 1)
     * of row y + (ky == 2), column x + (kx > 0). */
    static const uint8_t tile_of_tap[9] = {1, 4, 0, 7, 8, 6, 3, 5, 2};
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t oh = Y->dims.h, d = X->padding.w - Y->padding.w;
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u, np = p1 - p0;
    const int32_t first = (int32_t)((2u * p0 + d) >> 1);
    const uint32_t n = ((2u * p1 + d + 1u) >> 1) - (uint32_t)first + 1u;
    const HVX_VectorPred qev = even_words(splat_base());
    /* Contiguous (group, row) items per worker: setup once per group touched. */
    const uint32_t rows_ = Y->dims.b * oh, groups_ = (Y->dims.c + 31u) / 32u;
    const uint32_t i0_ = groups_ * rows_ * part / parts, i1_ = groups_ * rows_ * (part + 1u) / parts;
    uint32_t it_ = i0_, r = i0_ % rows_;
    for (uint32_t g = i0_ / rows_; it_ < i1_; g++, r = 0) {
        HVX_Vector w[9];
        const int partial = g == groups_ - 1u && (Y->dims.c % 32u) != 0u;
        const HVX_VectorPred qc = channel_mask(Y, g);
        /* Xwe tap tile: row j (128 bytes) holds channel 2j's weight in word 2j's
         * low half and 2j + 1's in word 2j + 1's high half (the diagonal). Keep
         * bytes [8j, 8j + 8) of each row, then duplicate as for t=514. */
        for (uint32_t k = 0; k < 9; k++) {
            if (j->compact) {
                const HVX_Vector v = *(const HVX_UVector *)((const uint8_t *)j->weights + (g * 9u + k) * 128u);
                w[k] = Q6_V_vmux_QVV(qev, Q6_Vh_vshuffe_VhVh(v, v), Q6_Vh_vshuffo_VhVh(v, v));
                continue;
            }
            const uint8_t *t = (const uint8_t *)(j->weights + (g * 9u + tile_of_tap[k]) * 1024u);
            HVX_Vector acc = Q6_V_vzero();
            HVX_Vector mask = Q6_V_vmux_QVV(Q6_Q_vsetq_R(8), Q6_V_vsplat_R(-1), Q6_V_vzero());
            for (uint32_t r = 0; r < 16u; r++) {
                acc = Q6_V_vor_VV(acc, Q6_V_vand_VV(*(const HVX_UVector *)(t + r * 128u), mask));
                mask = Q6_V_vror_VR(mask, 120);
            }
            w[k] = Q6_V_vmux_QVV(qev, Q6_Vh_vshuffe_VhVh(acc, acc), Q6_Vh_vshuffo_VhVh(acc, acc));
        }
        const Ep e = epilogue(j->fi, j->tables, g);
        for (; r < rows_ && it_ < i1_; r++, it_++) {
            uint32_t b = r / oh, oy = r % oh;
            uint32_t bx = b + X->padding.b, by = b + Y->padding.b;
            uint32_t fyo = oy + Y->padding.h, fyi = oy + X->padding.h;
            HVX_Vector *a0[MAXP], *a1[MAXP], *a2[MAXP], *a3[MAXP], *b0[MAXP], *b1[MAXP], *o[MAXP];
            if (row_ptrs(&mx, bx, fyi, first, n, g * 4u, a0) && row_ptrs(&mx, bx, fyi, first, n, g * 4u + 1u, a1) &&
                row_ptrs(&mx, bx, fyi, first, n, g * 4u + 2u, a2) && row_ptrs(&mx, bx, fyi, first, n, g * 4u + 3u, a3) &&
                row_ptrs(&mx, bx, fyi + 1u, first, n, g * 4u, b0) &&
                row_ptrs(&mx, bx, fyi + 1u, first, n, g * 4u + 1u, b1) &&
                row_ptrs(&my, by, fyo, (int32_t)p0, np, g, o)) {
                if (!(d & 1u)) {
                    /* Even offset: tap c is pair i, tap c + 1 is mid(pair i, pair i + 1);
                     * each array loads one new pair per step. */
                    const HVX_Vector w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5], w6 = w[6],
                                     w7 = w[7], w8 = w[8];
                    HVX_Vector A0 = *a0[0], A1 = *a1[0], A2 = *a2[0], A3 = *a3[0], B0 = *b0[0], B1 = *b1[0];
                    for (uint32_t i = 0; i < np; i++) {
                        const HVX_Vector N0 = *a0[i + 1u], N1 = *a1[i + 1u], N2 = *a2[i + 1u], N3 = *a3[i + 1u];
                        const HVX_Vector M0 = *b0[i + 1u], M1 = *b1[i + 1u];
                        HVX_VectorPair ma = Q6_Wqf32_vmpy_VhfVhf(A0, w0), mb = Q6_Wqf32_vmpy_VhfVhf(A2, w3);
                        HVX_Vector loa = Q6_V_lo_W(ma), hia = Q6_V_hi_W(ma), lob = Q6_V_lo_W(mb), hib = Q6_V_hi_W(mb);
                        mac(&loa, &hia, mid(A1, N1), w1);
                        mac(&lob, &hib, mid(A3, N3), w4);
                        mac(&loa, &hia, mid(A0, N0), w2);
                        mac(&lob, &hib, mid(A2, N2), w5);
                        mac(&loa, &hia, B0, w6);
                        mac(&lob, &hib, mid(B1, M1), w7);
                        mac(&loa, &hia, mid(B0, M0), w8);
                        A0 = N0; A1 = N1; A2 = N2; A3 = N3; B0 = M0; B1 = M1;
                        put_row_c(o, i, Y, p0 + i,
                                  finish(Q6_Vqf32_vadd_Vqf32Vqf32(loa, lob), Q6_Vqf32_vadd_Vqf32Vqf32(hia, hib), &e),
                                  partial, qc);
                    }
                    continue;
                }
                for (uint32_t pp = p0; pp < p1; pp++) {
                    int32_t c = (int32_t)(2u * pp + d);
                    HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(tapr(a0, first, c), w[0]);
                    HVX_Vector lo = Q6_V_lo_W(m), hi = Q6_V_hi_W(m);
                    mac(&lo, &hi, tapr(a1, first, c + 1), w[1]);
                    mac(&lo, &hi, tapr(a0, first, c + 1), w[2]);
                    mac(&lo, &hi, tapr(a2, first, c), w[3]);
                    mac(&lo, &hi, tapr(a3, first, c + 1), w[4]);
                    mac(&lo, &hi, tapr(a2, first, c + 1), w[5]);
                    mac(&lo, &hi, tapr(b0, first, c), w[6]);
                    mac(&lo, &hi, tapr(b1, first, c + 1), w[7]);
                    mac(&lo, &hi, tapr(b0, first, c + 1), w[8]);
                    put_row_c(o, pp - p0, Y, pp, finish(lo, hi, &e), partial, qc);
                }
                continue;
            }
            for (uint32_t pp = p0; pp < p1; pp++) {
                HVX_Vector lo = Q6_V_vzero(), hi = Q6_V_vzero();
                int32_t col = (int32_t)(2u * pp + d);
                for (uint32_t ky = 0; ky < 3; ky++)
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        uint32_t q = (ky % 2u) * 2u + (kx == 1u);
                        mac(&lo, &hi, tap(&mx, bx, fyi + (ky == 2u), col + (kx > 0u), g * 4u + q),
                            w[ky * 3u + kx]);
                    }
                put_c(vec(&my, by, fyo, (int32_t)pp, g), finish(lo, hi, &e), lanes_of(Y, pp), partial, qc);
            }
        }
    }
}

void qnn16_tile_dw1x1(const Qnn16TileJob *j, uint32_t part, uint32_t parts) {
    /* t=544: output channel c combines input channels c and c + 32. */
    const HVX_Vector bit4 = Q6_V_vand_VV(splat_base(), Q6_V_vsplat_R(0x04040404));
    const HVX_Vector clear4 = bit4, set4 = Q6_V_vxor_VV(bit4, Q6_V_vsplat_R(0x04040404));
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t oh = Y->dims.h, d = X->padding.w - Y->padding.w;
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u, np = p1 - p0;
    const int32_t first = (int32_t)((2u * p0 + d) >> 1);
    const uint32_t n = ((2u * p1 + d - 1u) >> 1) - (uint32_t)first + 1u;
    /* Contiguous (group, row) items per worker: setup once per group touched. */
    const uint32_t rows_ = Y->dims.b * oh, groups_ = (Y->dims.c + 31u) / 32u;
    const uint32_t i0_ = groups_ * rows_ * part / parts, i1_ = groups_ * rows_ * (part + 1u) / parts;
    uint32_t it_ = i0_, r = i0_ % rows_;
    for (uint32_t g = i0_ / rows_; it_ < i1_; g++, r = 0) {
        const int partial = g == groups_ - 1u && (Y->dims.c % 32u) != 0u;
        const HVX_VectorPred qc = channel_mask(Y, g);
        /* t=544: both channels of pair j use half 4j (word 2j low) and half
         * 4j + 3 (word 2j + 1 high); vdelta copies the word across the pair. */
        const HVX_Vector v = *(const HVX_UVector *)(j->weights + g * 1024u);
        const HVX_Vector w0 = Q6_V_vdelta_VV(Q6_Vh_vshuffe_VhVh(v, v), clear4);
        const HVX_Vector w1 = Q6_V_vdelta_VV(Q6_Vh_vshuffo_VhVh(v, v), set4);
        const Ep e = epilogue(j->fi, j->tables, g);
        for (; r < rows_ && it_ < i1_; r++, it_++) {
            uint32_t b = r / oh, oy = r % oh;
            uint32_t bx = b + X->padding.b, by = b + Y->padding.b;
            uint32_t fyo = oy + Y->padding.h, fyi = oy + X->padding.h;
            HVX_Vector *x0[MAXP], *x1[MAXP], *o[MAXP];
            if (row_ptrs(&mx, bx, fyi, first, n, g, x0) && row_ptrs(&mx, bx, fyi, first, n, g + 1u, x1) &&
                row_ptrs(&my, by, fyo, (int32_t)p0, np, g, o)) {
                for (uint32_t pp = p0; pp < p1; pp++) {
                    int32_t c = (int32_t)(2u * pp + d);
                    HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(tapr(x0, first, c), w0);
                    HVX_Vector lo = Q6_V_lo_W(m), hi = Q6_V_hi_W(m);
                    mac(&lo, &hi, tapr(x1, first, c), w1);
                    put_row_c(o, pp - p0, Y, pp, finish(lo, hi, &e), partial, qc);
                }
                continue;
            }
            for (uint32_t pp = p0; pp < p1; pp++) {
                HVX_Vector lo = Q6_V_vzero(), hi = Q6_V_vzero();
                int32_t col = (int32_t)(2u * pp + d);
                mac(&lo, &hi, tap(&mx, bx, fyi, col, g), w0);
                mac(&lo, &hi, tap(&mx, bx, fyi, col, g + 1u), w1);
                put_c(vec(&my, by, fyo, (int32_t)pp, g), finish(lo, hi, &e), lanes_of(Y, pp), partial, qc);
            }
        }
    }
}

void qnn16_tile_s2d(const Qnn16TileJob *j, uint32_t part, uint32_t parts) {
    /* Output channel ((c / 32) * 4 + q) * 32 + c % 32 at (y, x) is input
     * channel c at row 2y + q / 2, column 2x - q % 2, zero outside. */
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t oh = Y->dims.h;
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u;
    const HVX_VectorPred qe = even_lanes(), qo = odd_lanes();
    /* Contiguous (group, row) items per worker: setup once per group touched. */
    const uint32_t rows_ = Y->dims.b * oh, groups_ = Y->dims.c / 32u;
    const uint32_t i0_ = groups_ * rows_ * part / parts, i1_ = groups_ * rows_ * (part + 1u) / parts;
    uint32_t it_ = i0_, r = i0_ % rows_;
    for (uint32_t G = i0_ / rows_; it_ < i1_; G++, r = 0) {
        uint32_t g = G / 4u, q = G % 4u;
        /* Source column of the even lane of output pair pp: s(pp) = 4 pp + s0. */
        const int32_t s0 = -2 * (int32_t)Y->padding.w - (int32_t)(q & 1u) + (int32_t)X->padding.w;
        /* Interior pairs [ia, ib): both source columns inside the input and
         * both output lanes inside the output (depends on q only). */
        uint32_t ia = p1, ib = p1;
        for (uint32_t pp = p0; pp < p1; pp++) {
            const int32_t ix = 2 * ((int32_t)(2u * pp) - (int32_t)Y->padding.w) - (int32_t)(q & 1u);
            const int in_ = ix >= 0 && ix + 2 < (int32_t)X->dims.w && lanes_of(Y, pp) == 3u;
            if (in_ && ia == p1)
                ia = pp;
            if (!in_ && ia != p1 && ib == p1)
                ib = pp;
        }
        const int even_s = !(s0 & 1);
        /* A partial last input group: lanes past the logical channels read as zero. */
        const uint32_t live = X->dims.c - g * 32u < 32u ? X->dims.c - g * 32u : 32u;
        const HVX_VectorPred qlive = Q6_Q_vsetq_R((int)(live * 4u));
        for (; r < rows_ && it_ < i1_; r++, it_++) {
            uint32_t b = r / oh, oy = r % oh;
            uint32_t bx = b + X->padding.b, by = b + Y->padding.b;
            uint32_t iy = 2u * oy + q / 2u, fyo = oy + Y->padding.h, fyi = iy + X->padding.h;
            HVX_Vector *in[MAXP], *o[MAXP];
            /* Interior [ia, ib) through the rows' pointer tables; the rest are edge pairs. */
            const int32_t first = (4 * (int32_t)ia + s0) >> 1;
            const uint32_t n = ia < ib ? 2u * (ib - ia) : 0u;
            const int done = ia < ib && iy < X->dims.h && first >= 0 &&
                             row_ptrs(&mx, bx, fyi, first, n, g, in) &&
                             row_ptrs(&my, by, fyo, (int32_t)ia, ib - ia, G, o);
            if (done && live != 32u) {
                for (uint32_t i = 0; i < ib - ia; i++) {
                    const HVX_Vector v = even_s ? Q6_Vh_vshuffe_VhVh(*in[2u * i + 1u], *in[2u * i])
                                                : Q6_Vh_vshuffo_VhVh(*in[2u * i + 1u], *in[2u * i]);
                    *o[i] = Q6_V_vmux_QVV(qlive, v, Q6_V_vzero());
                }
            } else if (done) {
                if (even_s)
                    for (uint32_t i = 0; i < ib - ia; i++)
                        *o[i] = Q6_Vh_vshuffe_VhVh(*in[2u * i + 1u], *in[2u * i]);
                else
                    for (uint32_t i = 0; i < ib - ia; i++)
                        *o[i] = Q6_Vh_vshuffo_VhVh(*in[2u * i + 1u], *in[2u * i]);
            }
            for (uint32_t pp = p0; pp < p1; pp++) {
                if (done && pp == ia) {
                    pp = ib - 1u; /* skip the interior done above */
                    continue;
                }
                HVX_Vector y = Q6_V_vzero();
                if (iy < X->dims.h) {
                    int32_t ix = 2 * ((int32_t)(2u * pp) - (int32_t)Y->padding.w) - (int32_t)(q & 1u);
                    int32_t s = ix + (int32_t)X->padding.w;
                    HVX_Vector lo_pair = ld(&mx, bx, fyi, s >> 1, g), hi_pair = ld(&mx, bx, fyi, (s >> 1) + 1, g);
                    y = !(s & 1) ? Q6_Vh_vshuffe_VhVh(hi_pair, lo_pair) : Q6_Vh_vshuffo_VhVh(hi_pair, lo_pair);
                    int ve = ix >= 0 && ix < (int32_t)X->dims.w, vo = ix + 2 >= 0 && ix + 2 < (int32_t)X->dims.w;
                    if (!ve && !vo)
                        y = Q6_V_vzero();
                    else if (!ve)
                        y = Q6_V_vmux_QVV(qo, y, Q6_V_vzero());
                    else if (!vo)
                        y = Q6_V_vmux_QVV(qe, y, Q6_V_vzero());
                    if (live != 32u)
                        y = Q6_V_vmux_QVV(qlive, y, Q6_V_vzero());
                }
                put(vec(&my, by, fyo, (int32_t)pp, G), y, lanes_of(Y, pp));
            }
        }
    }
}

int qnn16_tile_conv3x3_ok(const Qnn16Tensor *x, const Qnn16Tensor *y, uint32_t packed_ic) {
    return blocked_any(x) && blocked_any(y) && x->dims.b == y->dims.b && x->padding.w >= y->padding.w &&
           x->dims.h == y->dims.h + 2u && x->dims.w == y->dims.w + 2u && x->dims.c <= packed_ic &&
           packed_ic <= x->full.c;
}
int qnn16_tile_pw_ok(const Qnn16Tensor *x, const Qnn16Tensor *y, uint32_t packed_ic) {
    return blocked_any(x) && blocked_any(y) && x->dims.b == y->dims.b && x->padding.w >= y->padding.w &&
           x->dims.h == y->dims.h && x->dims.w == y->dims.w && x->dims.c <= packed_ic && packed_ic <= x->full.c;
}
/* Dense K x K (K = 3 Xwe, K = 1 packed Xpe) with any channel counts. The
 * input word for channel ci at pixels (c, c + 1) is splatted against the
 * weight row holding ci's pair for all 32 outputs. */
#define MAXG 5u
static inline const uint8_t *dense_row(const Qnn16TileJob *j, uint32_t k, uint32_t g, uint32_t ci, uint32_t q) {
    if (k == 1u)
        return (const uint8_t *)j->weights + (g * ((j->packed_ic + 1u) / 2u) + ci / 2u) * 128u;
    const uint32_t kg = (j->packed_ic + 31u) / 32u;
    return (const uint8_t *)j->weights + (((g * kg + ci / 32u) * 9u + q) * 16u + (ci % 32u) / 2u) * 128u;
}
/* Word w of v in every word: vdelta with byte i control (i ^ 4w) & 0x7c (the
 * splat_base vector xor 4w) moves every byte to offset (i & 3) of word w. */
static void dense(const Qnn16TileJob *j, uint32_t k, uint32_t part, uint32_t parts) {
    const HVX_Vector base = splat_base();
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t oh = Y->dims.h, d = X->padding.w - Y->padding.w, ic = X->dims.c, ng = (ic + 31u) / 32u;
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u;
    const int32_t first = (int32_t)((2u * p0 + d) >> 1);
    const uint32_t n = ((2u * p1 + d + k - 2u) >> 1) - (uint32_t)first + 1u;
    const HVX_VectorPred qe = even_lanes(), qo = odd_lanes();
    /* Contiguous (group, row) items per worker: setup once per group touched. */
    const uint32_t rows_ = Y->dims.b * oh, groups_ = (Y->dims.c + 31u) / 32u;
    const uint32_t i0_ = groups_ * rows_ * part / parts, i1_ = groups_ * rows_ * (part + 1u) / parts;
    uint32_t it_ = i0_, r = i0_ % rows_;
    for (uint32_t g = i0_ / rows_; it_ < i1_; g++, r = 0) {
        const Ep e = epilogue(j->fi, j->tables, g);
        const uint32_t live = Y->dims.c - g * 32u;
        const HVX_VectorPred qc = live >= 32u ? Q6_Q_vand_VR(Q6_V_vsplat_R(-1), 0xffffffffu)
                                              : Q6_Q_vsetq_R((int)(live * 4u));
        for (; r < rows_ && it_ < i1_; r++, it_++) {
            uint32_t b = r / oh, oy = r % oh;
            uint32_t bx = b + X->padding.b, by = b + Y->padding.b;
            uint32_t fyo = oy + Y->padding.h, fyi = oy + X->padding.h;
            HVX_Vector *rows[3][MAXG][MAXP];
            int fast = ng <= MAXG;
            for (uint32_t ky = 0; fast && ky < k; ky++)
                for (uint32_t ig = 0; fast && ig < ng; ig++)
                    fast = row_ptrs(&mx, bx, fyi + ky, first, n, ig, rows[ky][ig]);
            /* Four output pairs at a time share each weight row and splat control. */
            for (uint32_t pp = p0; pp < p1; pp += 4u) {
                const uint32_t nb = p1 - pp < 4u ? p1 - pp : 4u;
                HVX_Vector lo[4], hi[4];
#pragma unroll
                for (uint32_t t = 0; t < 4u; t++)
                    lo[t] = hi[t] = Q6_V_vzero();
                for (uint32_t ky = 0; ky < k; ky++)
                    for (uint32_t kx = 0; kx < k; kx++) {
                        const uint32_t q = ky * 3u + 2u - kx;
                        for (uint32_t ig = 0; ig < ng; ig++) {
                            HVX_Vector v[4];
#pragma unroll
                            for (uint32_t t = 0; t < 4u; t++) {
                                const int32_t cc = (int32_t)(2u * (pp + (t < nb ? t : 0u)) + d + kx);
                                v[t] = fast ? tapr(rows[ky][ig], first, cc) : tap(&mx, bx, fyi + ky, cc, ig);
                            }
                            const uint32_t end = ic - ig * 32u < 32u ? ic - ig * 32u : 32u;
                            for (uint32_t w = 0; w < end; w += 2u) {
                                const HVX_Vector row = *(const HVX_UVector *)dense_row(j, k, g, ig * 32u + w, q);
                                const HVX_Vector we = Q6_Vh_vshuffe_VhVh(row, row), wo = Q6_Vh_vshuffo_VhVh(row, row);
                                const HVX_Vector ce = Q6_V_vxor_VV(base, Q6_V_vsplat_R((int)(w * 4u * 0x01010101u)));
                                const HVX_Vector co = Q6_V_vxor_VV(base, Q6_V_vsplat_R((int)((w + 1u) * 4u * 0x01010101u)));
                                const int pair = w + 1u < end;
#pragma unroll
                                for (uint32_t t = 0; t < 4u; t++) {
                                    mac(&lo[t], &hi[t], Q6_V_vdelta_VV(v[t], ce), we);
                                    if (pair)
                                        mac(&lo[t], &hi[t], Q6_V_vdelta_VV(v[t], co), wo);
                                }
                            }
                        }
                    }
                for (uint32_t t = 0; t < nb; t++) {
                    HVX_Vector y = finish(lo[t], hi[t], &e);
                    HVX_Vector *p = vec(&my, by, fyo, (int32_t)(pp + t), g);
                    uint32_t lanes = lanes_of(Y, pp + t);
                    if (!p || !lanes)
                        continue;
                    HVX_VectorPred qm = lanes == 3u ? qc : Q6_Q_and_QQ(qc, lanes == 1u ? qe : qo);
                    Q6_vmem_QRIV(qm, p, y);
                }
            }
        }
    }
}
void qnn16_tile_conv3x3(const Qnn16TileJob *j, uint32_t part, uint32_t parts) { dense(j, 3u, part, parts); }
void qnn16_tile_pw(const Qnn16TileJob *j, uint32_t part, uint32_t parts) { dense(j, 1u, part, parts); }

int qnn16_tile_stage3x3_ok(const Qnn16Tensor *x, const Qnn16Tensor *y, uint32_t stride2, uint32_t kg) {
    /* A partial last tile row or column (96 px crops: 4x12, 6x6 outputs) is staged whole; hmx_conv3
     * restores the output lanes past the tensor, which may belong to another view of the same blocks. */
    if (!blocked_any(x) || !blocked(y) || x->dims.b != y->dims.b ||
        y->padding.b || y->padding.h || y->padding.w || y->full.b != y->dims.b ||
        (y->full.h + 7u) / 8u != (y->dims.h + 7u) / 8u || (y->full.w + 3u) / 4u != (y->dims.w + 3u) / 4u ||
        y->full.c != y->dims.c || x->padding.b || !kg)
        return 0;
    for (uint32_t i = 0; i < y->block_count; i++)
        if (y->block_offsets[i] == NONE)
            return 0;
    if (stride2)
        return x->dims.c == kg * 128u && x->dims.h >= y->dims.h + 1u && x->dims.w >= y->dims.w + 1u;
    return x->dims.c <= kg * 32u && x->dims.c + 31u >= kg * 32u && x->dims.h == y->dims.h + 2u &&
           x->dims.w == y->dims.w + 2u;
}
void qnn16_tile_stage3x3(const Qnn16StageJob *j, uint32_t part, uint32_t parts) {
    /* Stride 2 reads the space-to-depth input: tap (ky, kx) is phase
     * (ky % 2) * 2 + (kx == 1) at row y + (ky == 2), column x + (kx > 0), and
     * its weight tile is tile_of_tap[ky * 3 + kx]. Stride 1: tile ky * 3 + 2 - kx. */
    static const uint8_t tap_of_tile[9] = {2, 0, 8, 6, 1, 7, 5, 3, 4};
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X);
    const uint32_t tw = (Y->dims.w + 3u) / 4u, th = (Y->dims.h + 7u) / 8u, K = j->kg * 9u;
    for (uint32_t i = part; i < j->tiles; i += parts) {
        const uint32_t t = j->first + i, b = t / (th * tw), ty = (t / tw) % th, tx = t % tw;
        for (uint32_t cig = 0; cig < j->kg; cig++) {
            const uint32_t live = j->stride2 || X->dims.c >= (cig + 1u) * 32u ? 32u : X->dims.c - cig * 32u;
            const HVX_VectorPred qc = Q6_Q_vsetq_R((int)(live * 4u));
            for (uint32_t q = 0; q < 9u; q++) {
                uint32_t g, fy0, fx0;
                if (j->stride2) {
                    const uint32_t k = tap_of_tile[q], ky = k / 3u, kx = k % 3u;
                    g = cig * 4u + (ky % 2u) * 2u + (kx == 1u);
                    fy0 = ty * 8u + (ky == 2u) + X->padding.h;
                    fx0 = tx * 4u + (kx > 0u) + X->padding.w;
                } else {
                    g = cig;
                    fy0 = ty * 8u + q / 3u + X->padding.h;
                    fx0 = tx * 4u + (2u - q % 3u) + X->padding.w;
                }
                const uint32_t slot = i * K + cig * 9u + q;
                uint8_t *dst = j->scratch[slot];
                if (!(fy0 & 7u) && !(fx0 & 3u) && live == 32u) {
                    const HVX_Vector *p = vec(&mx, b, fy0, (int32_t)(fx0 >> 1), g);
                    if (p) {
                        j->a[slot] = (const uint8_t *)p;
                        continue;
                    }
                }
                j->a[slot] = dst;
                const int32_t pair = (int32_t)(fx0 >> 1);
                for (uint32_t r = 0; r < 8u; r++) {
                    HVX_Vector v0, v1;
                    const HVX_Vector p0 = ld(&mx, b, fy0 + r, pair, g), p1 = ld(&mx, b, fy0 + r, pair + 1, g);
                    if (fx0 & 1u) {
                        v0 = mid(p0, p1);
                        v1 = mid(p1, ld(&mx, b, fy0 + r, pair + 2, g));
                    } else {
                        v0 = p0;
                        v1 = p1;
                    }
                    if (live != 32u) {
                        v0 = Q6_V_vmux_QVV(qc, v0, Q6_V_vzero());
                        v1 = Q6_V_vmux_QVV(qc, v1, Q6_V_vzero());
                    }
                    *(HVX_Vector *)(dst + r * 256u) = v0;
                    *(HVX_Vector *)(dst + r * 256u + 128u) = v1;
                }
            }
        }
    }
}

int qnn16_tile_zap(uint8_t *const *blocks, uint32_t count, uint32_t row_start, uint32_t col_start) {
    if (!blocks || row_start > 8u || col_start > 4u)
        return -1;
    const HVX_Vector z = Q6_V_vzero();
    const HVX_VectorPred qo = odd_lanes();
    for (uint32_t i = 0; i < count; i++) {
        if (!blocks[i])
            return -1;
        for (uint32_t y = row_start; y < 8u; y++)
            for (uint32_t k = 0; k < 2u; k++) {
                HVX_Vector *p = (HVX_Vector *)(blocks[i] + y * 256u + k * 128u);
                if (2u * k >= col_start)
                    *p = z;
                else if (2u * k + 1u >= col_start)
                    Q6_vmem_QRIV(qo, p, z);
            }
    }
    return 0;
}

/* ---- Elementwise and data movement on tiles (serial; small tensors). ---- */
static void map_tiles(const Qnn16Tensor *X, const Qnn16Tensor *Y, MapFn fn, const void *ctx, uint32_t part,
                      uint32_t parts) {
    uint32_t slot = 0;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u;
    const int32_t dp = ((int32_t)X->padding.w - (int32_t)Y->padding.w) / 2;
    for (uint32_t g = 0; g < (Y->dims.c + 31u) / 32u; g++) {
        const HVX_VectorPred qc = channel_mask(Y, g);
        for (uint32_t b = 0; b < Y->dims.b; b++)
            for (uint32_t y = 0; y < Y->dims.h; y++) {
                PART_ROW(slot);
                for (uint32_t pp = p0; pp < p1; pp++)
                    store_live(vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)pp, g),
                               fn(ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)pp + dp, g), ctx),
                               lanes_of(Y, pp), qc);
            }
    }
}
typedef struct {
    HVX_Vector hi;
} ClampCtx;
static HVX_Vector relu_clamp(HVX_Vector v, const void *ctx) {
    /* max(v, 0) keeps -0 (0x8000), as the float comparison does. */
    const HVX_Vector neg0 = Q6_V_vsplat_R((int)0x80008000u);
    HVX_Vector y = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(v, neg0), v, Q6_Vh_vmax_VhVh(v, Q6_V_vzero()));
    return Q6_Vh_vmin_VhVh(y, ((const ClampCtx *)ctx)->hi);
}
int qnn16_tile_minmax(const Qnn16Tensor *x, const Qnn16Tensor *y, float lo, float hi, uint32_t part, uint32_t parts) {
    if (!same_grid(x, y) || lo != 0.0f || !(hi > 0.0f && hi < 65504.0f))
        return 0;
    ClampCtx c;
    c.hi = Q6_V_vsplat_R((int)((uint32_t)fh(hi) * 0x10001u));
    if (hf(fh(hi)) != hi)
        return 0;
    map_tiles(x, y, relu_clamp, &c, part, parts);
    return 1;
}
typedef struct {
    HVX_Vector k;
} ScaleCtx;
static HVX_Vector times(HVX_Vector v, const void *ctx) {
    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(v, ((const ScaleCtx *)ctx)->k));
}
int qnn16_tile_mul_scalar(const Qnn16Tensor *x, const Qnn16Tensor *k, const Qnn16Tensor *y, uint32_t part,
                          uint32_t parts) {
    if (!same_grid(x, y) || !blocked_any(k) || k->dims.b != 1 || k->dims.h != 1 || k->dims.w != 1 || k->dims.c != 1)
        return 0;
    const Map mk = map_of(k);
    const HVX_Vector *p = vec(&mk, k->padding.b, k->padding.h, (int32_t)(k->padding.w / 2u), 0);
    uint16_t h = p ? ((const uint16_t *)p)[k->padding.w & 1u] : 0;
    ScaleCtx c;
    c.k = Q6_V_vsplat_R((int)((uint32_t)h * 0x10001u));
    map_tiles(x, y, times, &c, part, parts);
    return 1;
}
int qnn16_tile_x2s(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    /* Depth-to-space, DCR: out(2y + q / 2, 2x + q % 2, c) = in(y, x, q * C + c). */
    if (!blocked(X) || !blocked(Y) || X->dims.b != Y->dims.b || Y->dims.h != 2u * X->dims.h ||
        Y->dims.w != 2u * X->dims.w || X->dims.c != 4u * Y->dims.c || (Y->padding.w & 1u))
        return 0;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t cg = Y->dims.c / 32u;
    for (uint32_t g = 0; g < cg; g++)
        for (uint32_t b = 0; b < Y->dims.b; b++)
            for (uint32_t oy = 0; oy < Y->dims.h; oy++) {
                PART_ROW(slot);
                const uint32_t dy = oy & 1u, iy = oy / 2u + X->padding.h, ga = 2u * dy * cg + g, gb = ga + cg;
                for (uint32_t x = 0; x < X->dims.w; x++) {
                    const uint32_t fx = x + X->padding.w;
                    HVX_Vector a = ld(&mx, b + X->padding.b, iy, (int32_t)(fx / 2u), ga);
                    HVX_Vector c = ld(&mx, b + X->padding.b, iy, (int32_t)(fx / 2u), gb);
                    HVX_Vector v = (fx & 1u) ? Q6_Vh_vshuffo_VhVh(c, a) : Q6_Vh_vshuffe_VhVh(c, a);
                    const uint32_t pp = x + Y->padding.w / 2u;
                    put(vec(&my, b + Y->padding.b, oy + Y->padding.h, (int32_t)pp, g), v, lanes_of(Y, pp));
                }
            }
    return 1;
}
int qnn16_tile_transpose_hw(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    /* out(b, y, x, c) = in(b, x, y, c). */
    if (!blocked_any(X) || !blocked_any(Y) || X->dims.b != Y->dims.b || X->dims.h != Y->dims.w ||
        X->dims.w != Y->dims.h || X->dims.c != Y->dims.c || (Y->padding.w & 1u))
        return 0;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u;
    for (uint32_t g = 0; g < (Y->dims.c + 31u) / 32u; g++) {
        const HVX_VectorPred qc = channel_mask(Y, g);
        for (uint32_t b = 0; b < Y->dims.b; b++)
            for (uint32_t y = 0; y < Y->dims.h; y++) {
                PART_ROW(slot);
                const uint32_t fx = y + X->padding.w;
                for (uint32_t pp = p0; pp < p1; pp++) {
                    const uint32_t x0 = 2u * pp - Y->padding.w, fy = x0 + X->padding.h;
                    HVX_Vector a = ld(&mx, b + X->padding.b, fy, (int32_t)(fx / 2u), g);
                    HVX_Vector c = ld(&mx, b + X->padding.b, fy + 1u, (int32_t)(fx / 2u), g);
                    HVX_Vector v = (fx & 1u) ? Q6_Vh_vshuffo_VhVh(c, a) : Q6_Vh_vshuffe_VhVh(c, a);
                    store_live(vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)pp, g), v, lanes_of(Y, pp), qc);
                }
            }
    }
    return 1;
}
/* out(b, y, x, c) = in(b, y, c, x) for in width 32 (16 pairs) and at most 32
 * channels, without padding: each row is a 16 x 16 transpose of 8-byte groups
 * (two channels x two pixels) after swapping the middle halves of each group. */
static int transpose_wc_vector(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    if (X->dims.w != 32u || Y->dims.c != 32u || X->dims.c > 32u || (X->dims.c & 1u) || X->padding.w ||
        Y->padding.w || X->padding.c || Y->padding.c || Y->dims.w != X->dims.c)
        return 0;
    const Map mx = map_of(X), my = map_of(Y);
    const HVX_VectorPred qev = even_words(splat_base());
    const uint32_t np = (Y->dims.w + 1u) / 2u;
    for (uint32_t b = 0; b < Y->dims.b; b++)
        for (uint32_t y = 0; y < Y->dims.h; y++) {
            PART_ROW(slot);
            HVX_Vector r[16];
            for (uint32_t j = 0; j < 16u; j++) {
                const HVX_Vector p = ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)j, 0);
                /* [a0 a1 | b0 b1] -> [a0 b0 | a1 b1] in each 8-byte group. */
                const HVX_Vector e = Q6_Vh_vshuffe_VhVh(Q6_V_vror_VR(p, 4), p);
                const HVX_Vector o = Q6_Vh_vshuffo_VhVh(p, Q6_V_vror_VR(p, 124));
                r[j] = Q6_V_vmux_QVV(qev, e, o);
            }
            for (uint32_t round = 0; round < 4u; round++) {
                HVX_Vector t[16];
                for (uint32_t i = 0; i < 8u; i++) {
                    HVX_VectorPair z = Q6_W_vshuff_VVR(r[i + 8u], r[i], -8);
                    t[2u * i] = Q6_V_lo_W(z);
                    t[2u * i + 1u] = Q6_V_hi_W(z);
                }
                for (uint32_t i = 0; i < 16u; i++)
                    r[i] = t[i];
            }
            for (uint32_t pp = 0; pp < np; pp++)
                put(vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)pp, 0), r[pp], lanes_of(Y, pp));
        }
    return 1;
}
int qnn16_tile_transpose_wc(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    /* out(b, y, x, c) = in(b, y, c, x): scalar halfword moves, tile to tile,
     * with the output row's pair pointers looked up once per channel group. */
    if (!blocked_any(X) || !blocked_any(Y) || X->dims.b != Y->dims.b || X->dims.h != Y->dims.h ||
        X->dims.w != Y->dims.c || X->dims.c != Y->dims.w)
        return 0;
    if (transpose_wc_vector(X, Y, part, parts))
        return 1;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t p0 = Y->padding.w / 2u, np = (Y->padding.w + Y->dims.w + 1u) / 2u - p0;
    for (uint32_t b = 0; b < Y->dims.b; b++)
        for (uint32_t y = 0; y < Y->dims.h; y++) {
            PART_ROW(slot);
            const uint32_t fyi = y + X->padding.h, fyo = y + Y->padding.h;
            HVX_Vector *o[MAXP];
            uint32_t og = UINT32_MAX;
            int fast = 0;
            for (uint32_t w = 0; w < X->dims.w; w++) { /* = output channel */
                const uint32_t fw = w + X->padding.w, oc = w + Y->padding.c;
                if (oc / 32u != og) {
                    og = oc / 32u;
                    fast = np <= MAXP && row_ptrs(&my, b + Y->padding.b, fyo, (int32_t)p0, np, og, o);
                }
                const uint16_t *src = 0;
                uint32_t sg = UINT32_MAX;
                for (uint32_t c = 0; c < X->dims.c; c++) { /* = output column */
                    const uint32_t fc = c + X->padding.c, fx = c + Y->padding.w;
                    if (fc / 32u != sg) {
                        sg = fc / 32u;
                        src = (const uint16_t *)vec(&mx, b + X->padding.b, fyi, (int32_t)(fw / 2u), sg);
                    }
                    uint16_t *dst = fast ? (uint16_t *)o[fx / 2u - p0]
                                         : (uint16_t *)vec(&my, b + Y->padding.b, fyo, (int32_t)(fx / 2u), og);
                    if (dst)
                        dst[(oc % 32u) * 2u + (fx & 1u)] = src ? src[(fc % 32u) * 2u + (fw & 1u)] : 0;
                }
            }
        }
    return 1;
}

/* ---- Direct (contiguous NHWC in VTCM) <-> tiles. ---- */
int qnn16_tile_format(const Qnn16Tensor *X, const Qnn16Tensor *Y) {
    const Qnn16Shape d = X->dims;
    if (d.b != Y->dims.b || d.h != Y->dims.h || d.w != Y->dims.w || d.c != Y->dims.c)
        return 0;
    if (blocked_any(X) && direct_ok(Y, 2u)) {
        /* Tile pixel -> 32 contiguous halves: vdeal puts even pixels' halves in
         * bytes 0-63 and odd pixels' in 64-127. */
        const Map mx = map_of(X);
        for (uint32_t b = 0; b < d.b; b++)
            for (uint32_t y = 0; y < d.h; y++)
                for (uint32_t x = 0; x < d.w; x++) {
                    const uint32_t fx = x + X->padding.w;
                    for (uint32_t g = 0; g * 32u < d.c; g++) {
                        HVX_Vector v = Q6_Vh_vdeal_Vh(ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)(fx / 2u), g));
                        if (fx & 1u)
                            v = Q6_V_vror_VR(v, 64);
                        const uint32_t n = d.c - g * 32u < 32u ? d.c - g * 32u : 32u;
                        store_bytes(direct_at(Y, b, y, x, g * 32u, 2u), v, n * 2u);
                    }
                }
        return 1;
    }
    if (direct_ok(X, 2u) && blocked_any(Y)) {
        /* Two pixels' contiguous halves -> one tile pair: vshuff interleaves them. */
        const Map my = map_of(Y);
        const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + d.w + 1u) / 2u;
        for (uint32_t g = 0; g * 32u < d.c; g++) {
            const HVX_VectorPred qc = channel_mask(Y, g);
            for (uint32_t b = 0; b < d.b; b++)
                for (uint32_t y = 0; y < d.h; y++)
                    for (uint32_t pp = p0; pp < p1; pp++) {
                        const uint32_t lanes = lanes_of(Y, pp);
                        const uint32_t x0 = 2u * pp - Y->padding.w; /* may wrap when lane 0 is padding */
                        HVX_Vector a = Q6_V_vzero(), c = Q6_V_vzero();
                        if (lanes & 1u)
                            a = *(const HVX_UVector *)direct_at(X, b, y, x0, g * 32u, 2u);
                        if (lanes & 2u)
                            c = *(const HVX_UVector *)direct_at(X, b, y, x0 + 1u, g * 32u, 2u);
                        store_live(vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)pp, g),
                                   Q6_V_lo_W(Q6_W_vshuff_VVR(c, a, -2)), lanes, qc);
                    }
        }
        return 1;
    }
    return 0;
}
int qnn16_tile_cast_f32(const Qnn16Tensor *X, const Qnn16Tensor *Y) {
    /* FP16 tiles -> float32 direct: even and odd pixels are each word's low
     * and high half. */
    const Qnn16Shape d = X->dims;
    if (!blocked_any(X) || !direct_ok(Y, 4u) || d.b != Y->dims.b || d.h != Y->dims.h || d.w != Y->dims.w ||
        d.c != Y->dims.c)
        return 0;
    const Map mx = map_of(X);
    const uint32_t p0 = X->padding.w / 2u, p1 = (X->padding.w + d.w + 1u) / 2u;
    for (uint32_t g = 0; g * 32u < d.c; g++) {
        const uint32_t n = (d.c - g * 32u < 32u ? d.c - g * 32u : 32u) * 4u;
        for (uint32_t b = 0; b < d.b; b++)
            for (uint32_t y = 0; y < d.h; y++)
                for (uint32_t pp = p0; pp < p1; pp++) {
                    const uint32_t lanes = lanes_of(X, pp), x0 = 2u * pp - X->padding.w;
                    const HVX_Vector v = ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)pp, g);
                    if (lanes & 1u)
                        store_bytes(direct_at(Y, b, y, x0, g * 32u, 4u),
                                    h2f_words(Q6_V_vand_VV(v, Q6_V_vsplat_R(0xffff))), n);
                    if (lanes & 2u)
                        store_bytes(direct_at(Y, b, y, x0 + 1u, g * 32u, 4u), h2f_words(Q6_Vuw_vlsr_VuwR(v, 16)), n);
                }
    }
    return 1;
}
int qnn16_tile_avgpool(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t k, uint32_t stride) {
    /* Valid k x k average: window sums in qf32, then times 1 / k^2. */
    if (!blocked_any(X) || !blocked_any(Y) || !k || !stride || X->dims.b != Y->dims.b || X->dims.c != Y->dims.c ||
        X->dims.h < k || X->dims.w < k || Y->dims.h != (X->dims.h - k) / stride + 1u ||
        Y->dims.w != (X->dims.w - k) / stride + 1u || (Y->padding.w & 1u))
        return 0;
    const Map mx = map_of(X), my = map_of(Y);
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    float inv = 1.0f / (float)(k * k);
    uint32_t inv_bits;
    __builtin_memcpy(&inv_bits, &inv, 4);
    const HVX_Vector scale = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R((int)inv_bits), Q6_V_vzero());
    for (uint32_t g = 0; g * 32u < Y->dims.c; g++) {
        const HVX_VectorPred qc = channel_mask(Y, g);
        for (uint32_t b = 0; b < Y->dims.b; b++)
            for (uint32_t oy = 0; oy < Y->dims.h; oy++)
                for (uint32_t ox = 0; ox < Y->dims.w; ox += 2u) {
                    HVX_Vector acc[2] = {Q6_V_vzero(), Q6_V_vzero()};
                    for (uint32_t t = 0; t < 2u && ox + t < Y->dims.w; t++)
                        for (uint32_t i = 0; i < k; i++)
                            for (uint32_t jx = 0; jx < k; jx++) {
                                const uint32_t fx = (ox + t) * stride + jx + X->padding.w;
                                HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(
                                    ld(&mx, b + X->padding.b, (oy * stride + i) + X->padding.h, (int32_t)(fx / 2u), g), one);
                                acc[t] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[t], (fx & 1u) ? Q6_V_hi_W(m) : Q6_V_lo_W(m));
                            }
                    HVX_Vector lo = Q6_Vqf32_vmpy_Vqf32Vqf32(acc[0], scale), hi = Q6_Vqf32_vmpy_Vqf32Vqf32(acc[1], scale);
                    const uint32_t pp = (ox + Y->padding.w) / 2u;
                    store_live(vec(&my, b + Y->padding.b, oy + Y->padding.h, (int32_t)pp, g),
                               Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo)), lanes_of(Y, pp), qc);
                }
    }
    return 1;
}

int qnn16_tile_im2col_ok(const Qnn16Tensor *X, const Qnn16Tensor *Y) {
    return direct_ok(X, 4u) && X->dims.c == 1u && X->dims.h >= 3u && X->dims.w >= 3u && blocked_any(Y) &&
           Y->dims.c == 9u && !(Y->padding.w & 1u) && Y->dims.b == X->dims.b &&
           Y->dims.h == (X->dims.h - 3u) / 2u + 1u && Y->dims.w == (X->dims.w - 3u) / 2u + 1u;
}
void qnn16_tile_im2col(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    /* out(y, x, ky * 3 + kx) = in(2y + ky, 2x + kx). For a chunk of 7 output
     * pairs from pp0, each input row's 32 floats from column 4 pp0 become
     * words { h(col t), h(col t + 2) }; rotating that row by (4 (pp - pp0) - 3 ky)
     * words lines up channels 3 ky .. 3 ky + 2 for pair pp. */
    const Map my = map_of(Y);
    const uint32_t np = (Y->dims.w + 1u) / 2u, p0 = Y->padding.w / 2u;
    const HVX_VectorPred qc = channel_mask(Y, 0);
    const HVX_Vector ones = Q6_V_vsplat_R(-1);
    const HVX_Vector m0 = Q6_V_vmux_QVV(Q6_Q_vsetq_R(12), ones, Q6_V_vzero());
    const HVX_Vector m1 = Q6_V_vmux_QVV(Q6_Q_and_QQn(Q6_Q_vsetq_R(24), Q6_Q_vsetq_R(12)), ones, Q6_V_vzero());
    const HVX_Vector m2 = Q6_V_vmux_QVV(Q6_Q_and_QQn(Q6_Q_vsetq_R(36), Q6_Q_vsetq_R(24)), ones, Q6_V_vzero());
    for (uint32_t r = part; r < Y->dims.b * Y->dims.h; r += parts) {
        const uint32_t b = r / Y->dims.h, y = r % Y->dims.h;
        for (uint32_t c0 = 0; c0 < np; c0 += 7u) {
            HVX_Vector v[3];
            for (uint32_t ky = 0; ky < 3u; ky++) {
                const HVX_Vector f = *(const HVX_UVector *)direct_at(X, b, 2u * y + ky, 4u * c0, 0, 4u);
                const HVX_Vector hw = f2h_words(f);
                v[ky] = Q6_V_vor_VV(hw, Q6_Vw_vasl_VwR(Q6_V_vror_VR(hw, 8), 16));
            }
            for (uint32_t i = c0; i < np && i < c0 + 7u; i++) {
                const uint32_t t0 = 4u * (i - c0);
                const HVX_Vector o =
                    Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(v[0], (int)((4u * t0) & 127u)), m0),
                                Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(v[1], (int)((4u * (t0 - 3u)) & 127u)), m1),
                                            Q6_V_vand_VV(Q6_V_vror_VR(v[2], (int)((4u * (t0 - 6u)) & 127u)), m2)));
                const uint32_t pp = p0 + i;
                store_live(vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)pp, 0), o, lanes_of(Y, pp), qc);
            }
        }
    }
}

int qnn16_tile_shuffle(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t groups, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    /* Channel shuffle: out[c] = in[(c % g) * (C / g) + c / g]. Every output word
     * is the source word broadcast by vdelta, kept under a rotating word mask. */
    const uint32_t C = X->dims.c;
    if (!same_grid(X, Y) || !groups || C % groups)
        return 0;
    const uint32_t per = C / groups;
    const Map mx = map_of(X), my = map_of(Y);
    const HVX_Vector base = splat_base();
    const HVX_Vector m0 = Q6_V_vmux_QVV(Q6_Q_vsetq_R(4), Q6_V_vsplat_R(-1), Q6_V_vzero());
    const uint32_t p0 = Y->padding.w / 2u, p1 = (Y->padding.w + Y->dims.w + 1u) / 2u;
    const int32_t dp = ((int32_t)X->padding.w - (int32_t)Y->padding.w) / 2;
    const uint32_t ng = (C + 31u) / 32u;
    if (ng > 64u)
        return 0;
    for (uint32_t b = 0; b < Y->dims.b; b++)
        for (uint32_t y = 0; y < Y->dims.h; y++)
            for (uint32_t pp = p0; pp < p1; pp++) {
                const HVX_Vector *src_at[64];
                int all = 1;
                for (uint32_t gi = 0; gi < ng; gi++) {
                    src_at[gi] = vec(&mx, b + X->padding.b, y + X->padding.h, (int32_t)pp + dp, gi);
                    all &= src_at[gi] != 0;
                }
                if (!all)
                    return 0; /* absent source block: the generic path rewrites the whole output */
                uint32_t rem = 0, quo = 0; /* c % groups, c / groups without division */
                for (uint32_t G = 0; G * 32u < C; G++) {
                    /* Split over (pixel, output group) items: single-pixel shuffles are common. */
                    if ((slot++ % parts) != part) {
                        const uint32_t skip = C - G * 32u < 32u ? C - G * 32u : 32u;
                        for (uint32_t w = 0; w < skip; w++)
                            if (++rem == groups) {
                                rem = 0;
                                quo++;
                            }
                        continue;
                    }
                    HVX_Vector out = Q6_V_vzero(), mask = m0;
                    const uint32_t live = C - G * 32u < 32u ? C - G * 32u : 32u;
                    for (uint32_t w = 0; w < live; w++) {
                        const uint32_t src = rem * per + quo;
                        if (++rem == groups) {
                            rem = 0;
                            quo++;
                        }
                        const HVX_Vector v = *src_at[src >> 5];
                        const HVX_Vector sp =
                            Q6_V_vdelta_VV(v, Q6_V_vxor_VV(base, Q6_V_vsplat_R((int)((src & 31u) * 4u * 0x01010101u))));
                        out = Q6_V_vor_VV(out, Q6_V_vand_VV(sp, mask));
                        mask = Q6_V_vror_VR(mask, 124);
                    }
                    store_live(vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)pp, G), out, lanes_of(Y, pp),
                               channel_mask(Y, G));
                }
            }
    return 1;
}
int qnn16_tile_to_compact(const Qnn16Tensor *X, const Qnn16Tensor *Y) {
    /* Blocked FP16 -> compact direct FP16 in X's NHWC order (RESHAPE). */
    const Qnn16Shape d = X->dims, f = Y->full;
    if (!blocked_any(X) || !direct_ok(Y, 2u) || Y->padding.b || Y->padding.h || Y->padding.w || Y->padding.c ||
        f.b != Y->dims.b || f.h != Y->dims.h || f.w != Y->dims.w || f.c != Y->dims.c ||
        (uint64_t)d.b * d.h * d.w * d.c != (uint64_t)f.b * f.h * f.w * f.c)
        return 0;
    const Map mx = map_of(X);
    uint8_t *dst = Y->base + Y->direct_offset;
    for (uint32_t b = 0; b < d.b; b++)
        for (uint32_t y = 0; y < d.h; y++)
            for (uint32_t x = 0; x < d.w; x++) {
                const uint32_t fx = x + X->padding.w;
                for (uint32_t g = 0; g * 32u < d.c; g++) {
                    HVX_Vector v = Q6_Vh_vdeal_Vh(ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)(fx / 2u), g));
                    if (fx & 1u)
                        v = Q6_V_vror_VR(v, 64);
                    const uint32_t n = d.c - g * 32u < 32u ? d.c - g * 32u : 32u;
                    store_bytes(dst + 2u * ((((b * d.h) + y) * d.w + x) * d.c + g * 32u), v, n * 2u);
                }
            }
    return 1;
}

/* ---- GroupNorm pieces and SiLU on tiles (QNN's Welford/InstanceNorm/Swish split). ---- */

int qnn16_tile_welford(const Qnn16Tensor *X, float *out) {
    /* Per channel over the tensor's pixels: mean and sum of squared deviations
     * (two passes; equal to the sequential reciprocal-count recurrence up to
     * float32 rounding). */
    if (!whole_pairs(X) || X->dims.b != 1u)
        return 0;
    const Map mx = map_of(X);
    const uint32_t p0 = X->padding.w / 2u, p1 = p0 + X->dims.w / 2u, c = X->dims.c;
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    const HVX_Vector inv = splat_sf(1.0f / (float)(X->dims.h * X->dims.w));
    for (uint32_t g = 0; g < (c + 31u) / 32u; g++) {
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        for (uint32_t y = 0; y < X->dims.h; y++)
            for (uint32_t pp = p0; pp < p1; pp++) {
                const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, X->padding.b, y + X->padding.h, (int32_t)pp, g), one);
                s0 = Q6_Vqf32_vadd_Vqf32Vqf32(s0, Q6_V_lo_W(w));
                s1 = Q6_Vqf32_vadd_Vqf32Vqf32(s1, Q6_V_hi_W(w));
            }
        const HVX_Vector mean_sf =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(s0, s1), qf_of_sf(inv)));
        HVX_Vector m0 = Q6_V_vzero(), m1 = Q6_V_vzero();
        for (uint32_t y = 0; y < X->dims.h; y++)
            for (uint32_t pp = p0; pp < p1; pp++) {
                const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, X->padding.b, y + X->padding.h, (int32_t)pp, g), one);
                const HVX_Vector d0 = Q6_Vqf32_vsub_Vqf32Vsf(Q6_V_lo_W(w), mean_sf);
                const HVX_Vector d1 = Q6_Vqf32_vsub_Vqf32Vsf(Q6_V_hi_W(w), mean_sf);
                m0 = Q6_Vqf32_vadd_Vqf32Vqf32(m0, Q6_Vqf32_vmpy_Vqf32Vqf32(d0, d0));
                m1 = Q6_Vqf32_vadd_Vqf32Vqf32(m1, Q6_Vqf32_vmpy_Vqf32Vqf32(d1, d1));
            }
        const HVX_Vector m2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m0, m1));
        const uint32_t live = c - g * 32u < 32u ? c - g * 32u : 32u;
        HVX_Vector buf[2];
        /* The destination is contiguous float32 [mean c][m2 c]; store live lanes only. */
        buf[0] = mean_sf;
        buf[1] = m2;
        if (live == 32u) {
            *(HVX_UVector *)(out + g * 32u) = mean_sf;
            *(HVX_UVector *)(out + c + g * 32u) = m2;
        } else {
            for (uint32_t i = 0; i < live; i++) {
                out[g * 32u + i] = ((const float *)&buf[0])[i];
                out[c + g * 32u + i] = ((const float *)&buf[1])[i];
            }
        }
    }
    return 1;
}

int qnn16_tile_instance_norm(const Qnn16Tensor *X, const Qnn16Tensor *Y, const float *moments, const uint16_t *scale,
                             const float *bias, uint32_t part, uint32_t parts) {
    /* y = (x * scale[c] - moments[c]) * moments[C + c] + bias[c]; scale holds
     * channel c's FP16 factor in half 2c. */
    if (!same_grid(X, Y) || !whole_pairs(X) || !whole_pairs(Y) || (X->dims.c & 31u))
        return 0;
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t p0 = Y->padding.w / 2u, np = Y->dims.w / 2u, c = X->dims.c;
    const int32_t dp = ((int32_t)X->padding.w - (int32_t)Y->padding.w) / 2;
    uint32_t slot = 0;
    for (uint32_t g = 0; g < c / 32u; g++) {
        const HVX_Vector sv = *(const HVX_UVector *)(scale + g * 64u);
        const HVX_Vector s = Q6_Vh_vshuffe_VhVh(sv, sv);
        const HVX_Vector mean = *(const HVX_UVector *)(moments + g * 32u);
        const HVX_Vector inv = qf_of_sf(*(const HVX_UVector *)(moments + c + g * 32u));
        const HVX_Vector add = *(const HVX_UVector *)(bias + g * 32u);
        for (uint32_t b = 0; b < Y->dims.b; b++)
            for (uint32_t y = 0; y < Y->dims.h; y++) {
                PART_ROW(slot);
                for (uint32_t i = 0; i < np; i++) {
                    HVX_Vector *o = vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)(p0 + i), g);
                    if (!o)
                        continue;
                    const HVX_VectorPair w =
                        Q6_Wqf32_vmpy_VhfVhf(ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)(p0 + i) + dp, g), s);
                    HVX_Vector lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(Q6_V_lo_W(w), mean), inv);
                    HVX_Vector hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(Q6_V_hi_W(w), mean), inv);
                    lo = Q6_Vqf32_vadd_Vqf32Vsf(lo, add);
                    hi = Q6_Vqf32_vadd_Vqf32Vsf(hi, add);
                    *o = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
                }
            }
    }
    return 1;
}

int qnn16_tile_swish(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    if (!same_grid(X, Y) || !whole_pairs(X) || !whole_pairs(Y) || (X->dims.c & 31u))
        return 0;
    const SwishK k = swish_k();
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t p0 = Y->padding.w / 2u, np = Y->dims.w / 2u;
    const int32_t dp = ((int32_t)X->padding.w - (int32_t)Y->padding.w) / 2;
    uint32_t slot = 0;
    for (uint32_t g = 0; g < X->dims.c / 32u; g++)
        for (uint32_t b = 0; b < Y->dims.b; b++)
            for (uint32_t y = 0; y < Y->dims.h; y++) {
                PART_ROW(slot);
                for (uint32_t i = 0; i < np; i++) {
                    HVX_Vector *o = vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)(p0 + i), g);
                    if (o)
                        *o = swish_hf(ld(&mx, b + X->padding.b, y + X->padding.h, (int32_t)(p0 + i) + dp, g), &k);
                }
            }
    return 1;
}
int qnn16_tile_add_swish(const Qnn16Tensor *X, const Qnn16Tensor *Y, uint32_t part, uint32_t parts) {
    /* out c = SiLU(in c + in (c + 32)), the sum kept in float32 precision. */
    if (!whole_pairs(X) || !whole_pairs(Y) || X->dims.b != Y->dims.b || X->dims.h != Y->dims.h ||
        X->dims.w != Y->dims.w || X->dims.c != 64u || Y->dims.c != 32u)
        return 0;
    const SwishK k = swish_k();
    const Map mx = map_of(X), my = map_of(Y);
    const uint32_t p0 = Y->padding.w / 2u, np = Y->dims.w / 2u;
    const int32_t dp = ((int32_t)X->padding.w - (int32_t)Y->padding.w) / 2;
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
    uint32_t slot = 0;
    for (uint32_t b = 0; b < Y->dims.b; b++)
        for (uint32_t y = 0; y < Y->dims.h; y++) {
            PART_ROW(slot);
            for (uint32_t i = 0; i < np; i++) {
                HVX_Vector *o = vec(&my, b + Y->padding.b, y + Y->padding.h, (int32_t)(p0 + i), 0);
                if (!o)
                    continue;
                const int32_t pi = (int32_t)(p0 + i) + dp;
                const HVX_VectorPair a = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, b + X->padding.b, y + X->padding.h, pi, 0), one);
                const HVX_VectorPair c = Q6_Wqf32_vmpy_VhfVhf(ld(&mx, b + X->padding.b, y + X->padding.h, pi, 1), one);
                const HVX_Vector lo = swish_qf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(a), Q6_V_lo_W(c)), &k);
                const HVX_Vector hi = swish_qf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(a), Q6_V_hi_W(c)), &k);
                *o = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
            }
        }
    return 1;
}
