/* Shared static helpers of the HVX tile leaves (qnn16_tile*.c): crouton block
 * maps, masked stores, direct-tensor addressing, exact FP16/float32 word
 * conversions and the qf32 SiLU core. Include from tile sources only. */
#ifndef QNN16_TILE_PRIV_H
#define QNN16_TILE_PRIV_H
#include "qnn16_tile.h"
#include "qnn16_half.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define NONE 0xffffffffu

typedef struct {
    uint8_t *base;
    const uint32_t *blocks;
    uint32_t nh, nw, nc, pairs, rows;
} Map;

static inline Map map_of(const Qnn16Tensor *t) {
    Map m;
    m.base = t->base;
    m.blocks = t->block_offsets;
    m.nh = (t->full.h + 7u) / 8u;
    m.nw = (t->full.w + 3u) / 4u;
    m.nc = (t->full.c + 31u) / 32u;
    m.pairs = m.nw * 2u;
    m.rows = m.nh * 8u;
    return m;
}
/* Vector of full-tensor row fy, pixel pair `pair`, channel group g; 0 if absent. */
static inline HVX_Vector *vec(const Map *m, uint32_t b, uint32_t fy, int32_t pair, uint32_t g) {
    if (pair < 0 || fy >= m->rows || (uint32_t)pair >= m->pairs || g >= m->nc)
        return 0;
    uint32_t off = m->blocks[((b * m->nh + fy / 8u) * m->nw + (uint32_t)pair / 2u) * m->nc + g];
    if (off == NONE)
        return 0;
    return (HVX_Vector *)(m->base + off + (fy & 7u) * 256u + ((uint32_t)pair & 1u) * 128u);
}
static inline HVX_Vector ld(const Map *m, uint32_t b, uint32_t fy, int32_t pair, uint32_t g) {
    HVX_Vector *p = vec(m, b, fy, pair, g);
    return p ? *p : Q6_V_vzero();
}
/* Lanes (even, odd) = pixels (col, col + 1) of row fy, group g. */
static inline HVX_Vector tap(const Map *m, uint32_t b, uint32_t fy, int32_t col, uint32_t g) {
    if (!(col & 1))
        return ld(m, b, fy, col >> 1, g);
    HVX_Vector a = ld(m, b, fy, (col - 1) >> 1, g), c = ld(m, b, fy, (col + 1) >> 1, g);
    return Q6_Vh_vshuffe_VhVh(c, Q6_Vh_vshuffo_VhVh(a, a));
}

/* Byte i = i & 0x7c (the word index times 4, in every byte of the word).
 * Bit 6 of the byte index is "i >= 64"; each lower bit k follows from bit
 * k + 1 as m ^ vror(m, 2^k), so five masks build the vector without a chain. */
static inline HVX_Vector splat_base(void) {
    const HVX_Vector ones = Q6_V_vsplat_R(-1), z = Q6_V_vzero();
    HVX_Vector m = Q6_V_vmux_QVV(Q6_Q_vsetq_R(64), z, ones); /* bit 6 */
    HVX_Vector v = Q6_V_vand_VV(m, Q6_V_vsplat_R(0x40404040));
    for (uint32_t bit = 32u; bit >= 4u; bit >>= 1) {
        m = Q6_V_vxor_VV(m, Q6_V_vror_VR(m, (int)bit));
        v = Q6_V_vor_VV(v, Q6_V_vand_VV(m, Q6_V_vsplat_R((int)(bit * 0x01010101u))));
    }
    return v;
}
static inline HVX_VectorPred even_words(HVX_Vector base) {
    return Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(base, Q6_V_vsplat_R(0x04040404)), Q6_V_vzero());
}

static inline HVX_VectorPred even_lanes(void) { return Q6_Q_vand_VR(Q6_V_vsplat_R(0x0000ffff), 0xffffffffu); }
static inline HVX_VectorPred odd_lanes(void) { return Q6_Q_vand_VR(Q6_V_vsplat_R((int)0xffff0000u), 0xffffffffu); }
/* lanes: bit 0 even pixel valid, bit 1 odd pixel valid. */
static inline uint32_t lanes_of(const Qnn16Tensor *t, uint32_t pair) {
    uint32_t lo = t->padding.w, hi = t->padding.w + t->dims.w, f = pair * 2u;
    return (f >= lo && f < hi ? 1u : 0u) | (f + 1u >= lo && f + 1u < hi ? 2u : 0u);
}
static inline void put(HVX_Vector *p, HVX_Vector y, uint32_t lanes) {
    if (!p)
        return;
    if (lanes == 3u)
        *p = y;
    else if (lanes == 1u)
        Q6_vmem_QRIV(even_lanes(), p, y);
    else if (lanes == 2u)
        Q6_vmem_QRIV(odd_lanes(), p, y);
}

static inline int blocked_any(const Qnn16Tensor *t) {
    const Qnn16Shape f = t->full;
    uint32_t n = f.b * ((f.h + 7u) / 8u) * ((f.w + 3u) / 4u) * ((f.c + 31u) / 32u);
    return !t->direct && t->block_offsets && t->block_count == n && !(f.c % 32u) && !t->padding.c && t->dims.b &&
           t->dims.h && t->dims.w && t->dims.c;
}
static inline int blocked(const Qnn16Tensor *t) { return blocked_any(t) && !(t->dims.c % 32u); }

#define MAXP 48u
/* Vector pointers of pairs [first, first + n) in row fy, group g. 0 when any
 * is outside the tensor or absent; the caller then takes the checked path. */
static inline int row_ptrs(const Map *m, uint32_t b, uint32_t fy, int32_t first, uint32_t n, uint32_t g, HVX_Vector **out) {
    if (n > MAXP || first < 0 || fy >= m->rows || (uint32_t)first + n > m->pairs || g >= m->nc)
        return 0;
    const uint32_t *bl = m->blocks + (b * m->nh + fy / 8u) * m->nw * m->nc + g;
    uint8_t *row = m->base + (fy & 7u) * 256u;
    /* One block-table load per two pairs. */
    uint32_t i = 0, pair = (uint32_t)first;
    if (pair & 1u) {
        const uint32_t off = bl[(pair / 2u) * m->nc];
        if (off == NONE)
            return 0;
        out[i++] = (HVX_Vector *)(row + off + 128u);
        pair++;
    }
    for (; i + 1u < n; i += 2u, pair += 2u) {
        const uint32_t off = bl[(pair / 2u) * m->nc];
        if (off == NONE)
            return 0;
        out[i] = (HVX_Vector *)(row + off);
        out[i + 1u] = (HVX_Vector *)(row + off + 128u);
    }
    if (i < n) {
        const uint32_t off = bl[(pair / 2u) * m->nc];
        if (off == NONE)
            return 0;
        out[i] = (HVX_Vector *)(row + off);
    }
    return 1;
}
/* Lanes = pixels (c, c + 1) from a row's pointer table starting at pair `first`. */
static inline HVX_Vector tapr(HVX_Vector *const *r, int32_t first, int32_t c) {
    int32_t i = (c >> 1) - first;
    if (!(c & 1))
        return *r[i];
    HVX_Vector a = *r[i];
    return Q6_Vh_vshuffe_VhVh(*r[i + 1], Q6_Vh_vshuffo_VhVh(a, a));
}
static inline void put_row(HVX_Vector *const *o, uint32_t i, const Qnn16Tensor *Y, uint32_t pp, HVX_Vector y) {
    put(o[i], y, lanes_of(Y, pp));
}

static inline HVX_Vector mid(HVX_Vector a, HVX_Vector b) { return Q6_Vh_vshuffe_VhVh(b, Q6_Vh_vshuffo_VhVh(a, a)); }

static inline HVX_VectorPred channel_mask(const Qnn16Tensor *Y, uint32_t g) {
    uint32_t live = Y->dims.c - g * 32u;
    return live >= 32u ? Q6_Q_vand_VR(Q6_V_vsplat_R(-1), 0xffffffffu) : Q6_Q_vsetq_R((int)(live * 4u));
}
static inline void store_live(HVX_Vector *p, HVX_Vector y, uint32_t lanes, HVX_VectorPred qc) {
    if (!p || !lanes)
        return;
    HVX_VectorPred q = lanes == 3u ? qc : Q6_Q_and_QQ(qc, lanes == 1u ? even_lanes() : odd_lanes());
    Q6_vmem_QRIV(q, p, y);
}
/* Same logical grid with pixel pairs aligned between x and y. */
static inline int same_grid(const Qnn16Tensor *x, const Qnn16Tensor *y) {
    return blocked_any(x) && blocked_any(y) && x->dims.b == y->dims.b && x->dims.h == y->dims.h &&
           x->dims.w == y->dims.w && x->dims.c == y->dims.c && !((x->padding.w ^ y->padding.w) & 1u);
}
typedef HVX_Vector (*MapFn)(HVX_Vector, const void *);
/* Rows (in loop order) part, part + parts, ... */
#define PART_ROW(slot)                                                                                                \
    if (((slot)++ % parts) != part)                                                                                     \
    continue

/* Store the first n (<= 128) bytes of v at any VTCM address. */
static inline void store_bytes(uint8_t *dst, HVX_Vector v, uint32_t n) {
    const uint32_t off = (uint32_t)(uintptr_t)dst & 127u;
    HVX_Vector *a = (HVX_Vector *)(dst - off);
    const HVX_Vector r = Q6_V_vror_VR(v, (int)((128u - off) & 127u));
    const uint32_t end = off + n;
    Q6_vmem_QRIV(Q6_Q_and_QQn(Q6_Q_vsetq2_R((int)(end < 128u ? end : 128u)), Q6_Q_vsetq_R((int)off)), a, r);
    if (end > 128u)
        Q6_vmem_QRIV(Q6_Q_vsetq_R((int)(end - 128u)), a + 1, r);
}
static inline uint8_t *direct_at(const Qnn16Tensor *t, uint32_t b, uint32_t y, uint32_t x, uint32_t c, uint32_t esz) {
    const Qnn16Shape f = t->full, p = t->padding;
    return t->base + t->direct_offset +
           esz * ((((b + p.b) * f.h + y + p.h) * f.w + x + p.w) * f.c + c + p.c);
}
/* Unaligned 128-byte loads must stay inside the window below the completion tile. */
static inline int direct_ok(const Qnn16Tensor *t, uint32_t esz) {
    const Qnn16Shape f = t->full;
    uint64_t bytes = (uint64_t)f.b * f.h * f.w * f.c * esz;
    return t->direct && (uint64_t)t->direct_offset + bytes + 128u <= t->base_bytes;
}

/* Exact FP16 -> IEEE single for words holding one half in their low 16 bits
 * (as hf() does: normals rebias, subnormals normalize via count leading zeros). */
static inline HVX_Vector h2f_words(HVX_Vector u) {
    const HVX_Vector z = Q6_V_vzero();
    const HVX_Vector sign = Q6_Vw_vasl_VwR(Q6_V_vand_VV(u, Q6_V_vsplat_R(0x8000)), 16);
    const HVX_Vector a = Q6_V_vand_VV(u, Q6_V_vsplat_R(0x7fff));
    const HVX_Vector e = Q6_Vuw_vlsr_VuwR(a, 10), m = Q6_V_vand_VV(a, Q6_V_vsplat_R(0x3ff));
    const HVX_Vector normal = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(a, 13), Q6_V_vsplat_R(112 << 23));
    const HVX_Vector clz = Q6_Vuw_vcl0_Vuw(m);
    const HVX_Vector sub = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(134), clz), 23),
                                       Q6_V_vand_VV(Q6_Vw_vasl_VwVw(m, Q6_Vw_vsub_VwVw(clz, Q6_V_vsplat_R(8))),
                                                    Q6_V_vsplat_R(0x7fffff)));
    const HVX_Vector inf = Q6_V_vor_VV(Q6_V_vsplat_R(0x7f800000), Q6_Vw_vasl_VwR(m, 13));
    HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(e, Q6_V_vsplat_R(31)), inf, normal);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(e, z), sub, r);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(a, z), z, r);
    return Q6_V_vor_VV(r, sign);
}

/* Exact float32 -> FP16 (round to nearest even, as fh()) into each word's low
 * half: normals rebias with the rounding increment; values below 2^-14 shift
 * their significand right with RNE; overflow gives inf, NaN 0x7e00. */
static inline HVX_Vector f2h_words(HVX_Vector x) {
    const HVX_Vector z = Q6_V_vzero(), one = Q6_V_vsplat_R(1);
    const HVX_Vector sign = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(x, Q6_V_vsplat_R((int)0x80000000u)), 16);
    const HVX_Vector a = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector odd = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 13), one);
    const HVX_Vector normal = Q6_Vuw_vlsr_VuwR(
        Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a, Q6_V_vsplat_R((int)0xc8000fffu)), odd), 13);
    const HVX_Vector e = Q6_Vuw_vlsr_VuwR(a, 23);
    const HVX_VectorPred has_e = Q6_Q_vcmp_gt_VwVw(e, z);
    const HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(a, Q6_V_vsplat_R(0x7fffff)),
                                     Q6_V_vmux_QVV(has_e, Q6_V_vsplat_R(0x800000), z));
    HVX_Vector sh = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(126), Q6_V_vmux_QVV(has_e, e, one));
    sh = Q6_Vw_vmin_VwVw(sh, Q6_V_vsplat_R(31));
    const HVX_Vector half = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(sh, one)), one);
    const HVX_Vector lsb = Q6_V_vand_VV(Q6_Vw_vlsr_VwVw(m, sh), one);
    const HVX_Vector sub = Q6_Vw_vlsr_VwVw(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(m, half), lsb), sh);
    HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x38800000), a), sub, normal);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(a, Q6_V_vsplat_R(0x477fffff)), Q6_V_vsplat_R(0x7c00), r);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(a, Q6_V_vsplat_R(0x7f800000)), Q6_V_vsplat_R(0x7e00), r);
    return Q6_V_vor_VV(r, sign);
}

static inline HVX_Vector splat_sf(float v) {
    union {
        float f;
        int32_t i;
    } u;
    u.f = v;
    return Q6_V_vsplat_R(u.i);
}
static inline HVX_Vector qf_of_sf(HVX_Vector v) { return Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vzero()); }
/* Whole pairs only: even column padding and even width, same batch grid. */
static inline int whole_pairs(const Qnn16Tensor *t) { return blocked_any(t) && !(t->padding.w & 1u) && !(t->dims.w & 1u); }

/* SiLU y = x * sigmoid(x) in qf32. With t = min(|x|, 16) and E = exp(-t) =
 * (exp(-t / 16))^16 (degree-5 polynomial, four squarings) and r = 1 / (1 + E)
 * (quadratic start, two Newton steps): y = x * r for x >= 0, x * E * r below.
 * Within one FP16 unit of the exactly rounded result; x below -16 is treated
 * as -16 (|y| < 2e-6 there). */
typedef struct {
    HVX_Vector c5, c[5], q0, q1, q2, one;
} SwishK;
static inline SwishK swish_k(void) {
    SwishK k;
    k.c5 = qf_of_sf(splat_sf(-0.005009557285471203f));
    k.c[4] = splat_sf(0.03810793567484634f);
    k.c[3] = splat_sf(-0.1648041336439795f);
    k.c[2] = splat_sf(0.49954315020760026f);
    k.c[1] = splat_sf(-0.999957528772394f);
    k.c[0] = splat_sf(0.9999993273432366f);
    k.q0 = qf_of_sf(splat_sf(0.32740728f));
    k.q1 = splat_sf(-1.45887808f);
    k.q2 = splat_sf(2.11751422f);
    k.one = splat_sf(1.0f);
    return k;
}
static inline __attribute__((always_inline)) HVX_Vector swish_core(HVX_Vector x, HVX_Vector u, HVX_VectorPred neg,
                                                                   const SwishK *k) {
    HVX_Vector e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(k->c5, u), k->c[4]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[3]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[2]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[1]);
    e = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(e, u), k->c[0]);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    e = Q6_Vqf32_vmpy_Vqf32Vqf32(e, e);
    const HVX_Vector d = Q6_Vqf32_vadd_Vqf32Vsf(e, k->one);
    HVX_Vector r = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(k->q0, d), k->q1);
    r = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(r, d), k->q2);
    r = Q6_Vqf32_vsub_Vqf32Vqf32(
        r, Q6_Vqf32_vmpy_Vqf32Vqf32(r, Q6_Vqf32_vsub_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(d, r), k->one)));
    r = Q6_Vqf32_vsub_Vqf32Vqf32(
        r, Q6_Vqf32_vmpy_Vqf32Vqf32(r, Q6_Vqf32_vsub_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(d, r), k->one)));
    return Q6_Vqf32_vmpy_Vqf32Vqf32(x, Q6_V_vmux_QVV(neg, Q6_Vqf32_vmpy_Vqf32Vqf32(e, r), r));
}
static inline __attribute__((always_inline)) HVX_Vector swish_hf(HVX_Vector x, const SwishK *k) {
    const HVX_Vector z = Q6_V_vzero(), sign = Q6_V_vsplat_R((int)0x80008000u);
    const HVX_Vector t = Q6_Vuh_vmin_VuhVuh(Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fff7fff)), Q6_V_vsplat_R(0x4c004c00));
    const HVX_Vector xc = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(z, x), Q6_V_vor_VV(t, sign), x);
    const HVX_VectorPair xw = Q6_Wqf32_vmpy_VhfVhf(xc, Q6_V_vsplat_R(0x3c003c00));
    const HVX_VectorPair uw = Q6_Wqf32_vmpy_VhfVhf(t, Q6_V_vsplat_R(0x2c002c00)); /* 1/16 */
    const HVX_VectorPred n0 = Q6_Q_vcmp_gt_VwVw(z, Q6_Vw_vasl_VwR(x, 16)), n1 = Q6_Q_vcmp_gt_VwVw(z, x);
    const HVX_Vector lo = swish_core(Q6_V_lo_W(xw), Q6_V_lo_W(uw), n0, k);
    const HVX_Vector hi = swish_core(Q6_V_hi_W(xw), Q6_V_hi_W(uw), n1, k);
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
}
/* SiLU of a float32-precision sum held in qf32. */
static inline __attribute__((always_inline)) HVX_Vector swish_qf(HVX_Vector s, const SwishK *k) {
    const HVX_Vector z = Q6_V_vzero(), v = Q6_Vsf_equals_Vqf32(s);
    const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(z, v);
    const HVX_Vector t = Q6_Vw_vmin_VwVw(Q6_V_vand_VV(v, Q6_V_vsplat_R(0x7fffffff)), Q6_V_vsplat_R(0x41800000));
    const HVX_Vector xc = Q6_V_vmux_QVV(neg, Q6_V_vor_VV(t, Q6_V_vsplat_R((int)0x80000000u)), v);
    return swish_core(qf_of_sf(xc), Q6_Vqf32_vmpy_VsfVsf(t, splat_sf(0.0625f)), neg, k);
}
#endif
