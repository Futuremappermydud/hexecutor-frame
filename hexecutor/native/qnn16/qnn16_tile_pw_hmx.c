/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* PW1X1 (packed Xpe weights) on the HMX 1x1 leaf, qnn16_tile_pw_hmx.c.
 *
 * The leaf (qnn16_conv1x1_wt, qnn16_hmx.S) accumulates the kg activation
 * blocks of each output block against ng * kg weight tiles, each tile 16 input
 * pairs x 32 outputs x 2 in HMX order. The packed weights hold ceil(ic / 2)
 * pairs per output group, so the driver first copies them into tiles, with
 * zero rows past the last pair and zero columns past the last output channel.
 * Activation lanes past the logical input channels are zeroed in place, so a
 * zero weight row multiplies a finite value. The leaf writes whole blocks, so
 * results go to temporary blocks and only the logical lanes are copied out,
 * which keeps every padding lane as it was.
 *
 * Callers (exec.c) run the parts in this order: tails, weights, then for each
 * chunk of rows: rows (serial), leaf, finish, store. */
#include "qnn16_tile_priv.h"

int qnn16_tile_pw_hmx_ok(const Qnn16Tensor *x, const Qnn16Tensor *y, uint32_t packed_ic, uint32_t *kg, uint32_t *ng) {
    if (!blocked_any(x) || !blocked_any(y) || !packed_ic || packed_ic > x->full.c || x->dims.c > packed_ic)
        return 0;
    if (x->dims.b != y->dims.b || x->dims.h != y->dims.h || x->dims.w != y->dims.w || x->full.b != y->full.b ||
        x->full.h != y->full.h || x->full.w != y->full.w)
        return 0;
    if (x->padding.b || x->padding.h || x->padding.w || y->padding.b || y->padding.h || y->padding.w)
        return 0;
    *kg = (packed_ic + 31u) / 32u;
    *ng = y->full.c / 32u;
    return *kg <= x->full.c / 32u && *ng;
}

static uint8_t *block_at(const Qnn16Tensor *t, uint32_t index) {
    const uint32_t off = t->block_offsets[index];
    return off == NONE ? 0 : t->base + off;
}

/* Zero the activation lanes at or past the logical channels of groups [0, kg), in place. */
void qnn16_tile_pw_hmx_tails(const Qnn16Tensor *x, uint32_t kg, uint32_t part, uint32_t parts) {
    const Map mx = map_of(x);
    const uint32_t blocks = x->full.b * mx.nh * mx.nw;
    const HVX_Vector z = Q6_V_vzero();
    for (uint32_t m = part; m < blocks; m += parts)
        for (uint32_t k = 0; k < kg; k++) {
            const uint32_t live = x->dims.c > 32u * k ? x->dims.c - 32u * k : 0u;
            if (live >= 32u)
                continue;
            uint8_t *blk = block_at(x, m * mx.nc + k);
            if (!blk)
                continue;
            const HVX_VectorPred keep = Q6_Q_vsetq_R((int)(live * 4u));
            for (uint32_t v = 0; v < 16u; v++) {
                HVX_Vector *p = (HVX_Vector *)(blk + v * 128u);
                *p = live ? Q6_V_vmux_QVV(keep, *p, z) : z;
            }
        }
}

/* Tile t = n * kg + k: 16 pair rows at pairs 16k.., output group n; zero past the packed pairs and outputs. */
void qnn16_tile_pw_hmx_weights(const uint16_t *w, const Qnn16Tensor *y, uint32_t packed_ic, uint32_t kg, uint32_t ng,
                               uint8_t *const *tiles, uint32_t part, uint32_t parts) {
    const uint32_t pairs = (packed_ic + 1u) / 2u;
    const HVX_Vector z = Q6_V_vzero();
    for (uint32_t t = part; t < kg * ng; t += parts) {
        const uint32_t n = t / kg, k = t % kg;
        const HVX_VectorPred keep = channel_mask(y, n);
        uint8_t *dst = tiles[t];
        for (uint32_t r = 0; r < 16u; r++) {
            const uint32_t pair = 16u * k + r;
            HVX_Vector v = z;
            if (pair < pairs)
                v = Q6_V_vmux_QVV(keep, *(const HVX_UVector *)((const uint8_t *)w + (n * pairs + pair) * 128u), z);
            *(HVX_Vector *)(dst + r * 128u) = v;
        }
    }
}

/* a[mi * kg + k] = activation block k of output block first + mi (serial). */
void qnn16_tile_pw_hmx_rows(const Qnn16Tensor *x, uint32_t kg, uint32_t first, uint32_t rows, const uint8_t **a) {
    const uint32_t nc = map_of(x).nc;
    for (uint32_t mi = 0; mi < rows; mi++)
        for (uint32_t k = 0; k < kg; k++)
            a[mi * kg + k] = block_at(x, (first + mi) * nc + k);
}

/* Output blocks first..first+rows (temps[mi * ng + n] holds block (first + mi, n)): logical lanes only. */
void qnn16_tile_pw_hmx_store(const Qnn16Tensor *y, uint32_t ng, uint8_t *const *temps, uint32_t first, uint32_t rows,
                             uint32_t part, uint32_t parts) {
    const Map my = map_of(y);
    for (uint32_t it = part; it < rows * ng; it += parts) {
        const uint32_t mi = it / ng, n = it % ng, m = first + mi;
        const uint32_t wb = m % my.nw, hb = (m / my.nw) % my.nh;
        uint8_t *dst = block_at(y, m * my.nc + n);
        const uint8_t *src = temps[it];
        if (!dst)
            continue;
        const HVX_VectorPred qc = channel_mask(y, n);
        for (uint32_t r = 0; r < 8u && hb * 8u + r < y->dims.h; r++)
            for (uint32_t h = 0; h < 2u; h++) {
                const uint32_t lanes = lanes_of(y, wb * 2u + h);
                if (!lanes)
                    continue;
                const uint32_t at = r * 256u + h * 128u;
                store_live((HVX_Vector *)(dst + at), *(const HVX_Vector *)(src + at), lanes, qc);
            }
    }
}
