/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): stage. */
#include "qnn16_tile_priv.h"

/* Stride 2 (space-to-depth input, four phase groups per 128 channels). Tap
 * (ky, kx) reads phase plane pl = (ky % 2) * 2 + (kx == 1) at rows R + (ky == 2)
 * and columns C + (kx > 0), where R = fyb and C = fxb (padding included). Per
 * tile and input group each plane is loaded once as a 9 x 3 pixel-pair window
 * at pairs P..P+2 (P = C / 2), with its mids m[r][k] = mid(w[r][k], w[r][k+1]);
 * tap offset o = C + (kx > 0) - 2P is 0..2, and the taps copy exactly the
 * vectors the plain form loads, so the staged bytes are identical. */
static int stage3x3_s2_fast(const Qnn16StageJob *j, uint32_t part, uint32_t parts) {
    static const uint8_t tap_of_tile[9] = {2, 0, 8, 6, 1, 7, 5, 3, 4};
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X);
    const uint32_t tw = (Y->dims.w + 3u) / 4u, th = (Y->dims.h + 7u) / 8u, K = j->kg * 9u;
    for (uint32_t i = part; i < j->tiles; i += parts) {
        const uint32_t t = j->first + i, b = t / (th * tw), ty = (t / tw) % th, tx = t % tw;
        const uint32_t R = ty * 8u + X->padding.h, C = tx * 4u + X->padding.w;
        const int32_t P = (int32_t)(C >> 1);
        for (uint32_t cig = 0; cig < j->kg; cig++) {
            HVX_Vector w[4][9][3], m[4][9][2];
            for (uint32_t pl = 0; pl < 4u; pl++)
                for (uint32_t r = 0; r < 9u; r++) {
                    for (uint32_t k = 0; k < 3u; k++)
                        w[pl][r][k] = ld(&mx, b, R + r, P + (int32_t)k, cig * 4u + pl);
                    for (uint32_t k = 0; k < 2u; k++)
                        m[pl][r][k] = mid(w[pl][r][k], w[pl][r][k + 1u]);
                }
            for (uint32_t q = 0; q < 9u; q++) {
                const uint32_t kt = tap_of_tile[q], ky = kt / 3u, kx = kt % 3u;
                const uint32_t pl = (ky % 2u) * 2u + (kx == 1u), dr = ky == 2u, dc = kx > 0u;
                const uint32_t fy0 = R + dr, fx0 = C + dc, g = cig * 4u + pl, slot = i * K + cig * 9u + q;
                uint8_t *dst = j->scratch[slot];
                if (!(fy0 & 7u) && !(fx0 & 3u)) {
                    const HVX_Vector *p = vec(&mx, b, fy0, (int32_t)(fx0 >> 1), g);
                    if (p) {
                        j->a[slot] = (const uint8_t *)p;
                        continue;
                    }
                }
                j->a[slot] = dst;
                const uint32_t o = fx0 - 2u * (uint32_t)P, k = o >> 1;
                for (uint32_t r = 0; r < 8u; r++) {
                    HVX_Vector v0, v1;
                    if (o & 1u) {
                        v0 = m[pl][dr + r][k];
                        v1 = m[pl][dr + r][k + 1u];
                    } else {
                        v0 = w[pl][dr + r][k];
                        v1 = w[pl][dr + r][k + 1u];
                    }
                    *(HVX_Vector *)(dst + r * 256u) = v0;
                    *(HVX_Vector *)(dst + r * 256u + 128u) = v1;
                }
            }
        }
    }
    return 1;
}

/* Stride 1 (stride 2 goes to stage3x3_s2_fast). For one tile and input group,
 * the nine taps read the 10 x 4 pixel-pair window at pairs P..P+3
 * (P = fxb / 2) and rows fyb..fyb+9, so the window is loaded once. Tap q reads
 * rows dy + r (dy = q / 3) and columns col = fxb + 2 - q % 3, which lie at
 * window offset o = col - 2P in 0..3: even o takes window pairs k, k + 1 and odd
 * o takes the precomputed mids M[k], M[k + 1] (identical to the plain form's
 * mid(p0, p1), mid(p1, p2)). Channel masking is word-local, so masking the
 * window before the mids equals masking the plain form's outputs. */
int qnn16_tile_stage3x3_fast(const Qnn16StageJob *j, uint32_t part, uint32_t parts) {
    if (j->stride2)
        return stage3x3_s2_fast(j, part, parts);
    const Qnn16Tensor *X = j->x, *Y = j->out;
    const Map mx = map_of(X);
    const uint32_t tw = (Y->dims.w + 3u) / 4u, th = (Y->dims.h + 7u) / 8u, K = j->kg * 9u;
    for (uint32_t i = part; i < j->tiles; i += parts) {
        const uint32_t t = j->first + i, b = t / (th * tw), ty = (t / tw) % th, tx = t % tw;
        const uint32_t fyb = ty * 8u + X->padding.h, fxb = tx * 4u + X->padding.w;
        const uint32_t P = fxb >> 1;
        for (uint32_t cig = 0; cig < j->kg; cig++) {
            const uint32_t live = X->dims.c >= (cig + 1u) * 32u ? 32u : X->dims.c - cig * 32u;
            const HVX_VectorPred qc = Q6_Q_vsetq_R((int)(live * 4u));
            HVX_Vector w[10][4], m[10][3];
            for (uint32_t r = 0; r < 10u; r++)
                for (uint32_t k = 0; k < 4u; k++)
                    w[r][k] = ld(&mx, b, fyb + r, (int32_t)(P + k), cig);
            if (live != 32u)
                for (uint32_t r = 0; r < 10u; r++)
                    for (uint32_t k = 0; k < 4u; k++)
                        w[r][k] = Q6_V_vmux_QVV(qc, w[r][k], Q6_V_vzero());
            for (uint32_t r = 0; r < 10u; r++)
                for (uint32_t k = 0; k < 3u; k++)
                    m[r][k] = mid(w[r][k], w[r][k + 1]);
            for (uint32_t q = 0; q < 9u; q++) {
                const uint32_t dy = q / 3u, col = fxb + (2u - q % 3u), slot = i * K + cig * 9u + q;
                uint8_t *dst = j->scratch[slot];
                if (!((fyb + dy) & 7u) && !(col & 3u) && live == 32u) {
                    const HVX_Vector *p = vec(&mx, b, fyb + dy, (int32_t)(col >> 1), cig);
                    if (p) {
                        j->a[slot] = (const uint8_t *)p;
                        continue;
                    }
                }
                j->a[slot] = dst;
                const uint32_t o = col - 2u * P, k = o >> 1;
                for (uint32_t r = 0; r < 8u; r++) {
                    HVX_Vector v0, v1;
                    if (o & 1u) {
                        v0 = m[dy + r][k];
                        v1 = m[dy + r][k + 1u];
                    } else {
                        v0 = w[dy + r][k];
                        v1 = w[dy + r][k + 1u];
                    }
                    *(HVX_Vector *)(dst + r * 256u) = v0;
                    *(HVX_Vector *)(dst + r * 256u + 128u) = v1;
                }
            }
        }
    }
    return 1;
}
