/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX tile form (qnn16_tile.h): transpose. */
#include "qnn16_tile_priv.h"

/* One 32-pixel x 32-channel block (16 pixel pairs x 16 channel pairs of 8-byte
 * groups): r[j] = pixel pair j of the input group, returns r[i] = output pair i
 * (output pixel = input channel, output channel = input pixel). Same network
 * as transpose_wc_vector in qnn16_tile.c. */
static void transpose_block(HVX_Vector *r, HVX_VectorPred qev) {
    for (uint32_t j = 0; j < 16u; j++) {
        const HVX_Vector p = r[j];
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
}

/* out(b, y, x, c) = in(b, y, c, x) on tiles, zero channel padding, even width
 * padding on both sides. Each (input channel group gi, input pixel chunk gj)
 * of 32 x 32 is one block: input pairs 16 gj + i of group gi become output
 * channel group gj, and the block's output pairs 16 gi + i (of group gj) are
 * stored with lanes and channels masked to the logical output. */
static int transpose_wc_blocks(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    if (!blocked_any(x) || !blocked_any(out) || x->dims.b != out->dims.b || x->dims.h != out->dims.h ||
        x->dims.w != out->dims.c || x->dims.c != out->dims.w || (x->padding.w & 1u) || (out->padding.w & 1u))
        return 0;
    const Map mx = map_of(x), my = map_of(out);
    const HVX_VectorPred qev = even_words(splat_base());
    const uint32_t px = x->padding.w / 2u, pw = out->padding.w / 2u;
    const uint32_t ngi = (x->dims.c + 31u) / 32u, ngj = (x->dims.w + 31u) / 32u;
    for (uint32_t b = 0; b < out->dims.b; b++)
        for (uint32_t y = 0; y < out->dims.h; y++) {
            PART_ROW(slot);
            const uint32_t fy = y + x->padding.h, fo = y + out->padding.h;
            for (uint32_t gi = 0; gi < ngi; gi++)
                for (uint32_t gj = 0; gj < ngj; gj++) {
                    HVX_Vector r[16];
                    for (uint32_t i = 0; i < 16u; i++)
                        r[i] = ld(&mx, b + x->padding.b, fy, (int32_t)(px + 16u * gj + i), gi);
                    transpose_block(r, qev);
                    const HVX_VectorPred qc = channel_mask(out, gj);
                    for (uint32_t i = 0; i < 16u; i++) {
                        const uint32_t P = pw + 16u * gi + i;
                        store_live(vec(&my, b + out->padding.b, fo, (int32_t)P, gj), r[i], lanes_of(out, P), qc);
                    }
                }
        }
    return 1;
}

/* out(b, y, x, 0) = in(0, y, x, b) for direct tensors: x [1,h,w,2] -> out
 * [2,h,w,1]. A row of x is 2 * w halfwords; 64 pixels (256 bytes) of it are
 * two vectors v0, v1 holding (c0, c1) pairs. vpacke/vpacko take the even
 * (channel 0) and odd (channel 1) halves of the 32 words of each vector, so
 * the channel-0 values of the 64 pixels land in v0 then v1, and likewise for
 * channel 1. */
static int transpose_direct_b(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    if (!direct_ok(x, 2) || !direct_ok(out, 2) || x->dims.b != 1 || x->dims.c != 2 || out->dims.b != 2 ||
        out->dims.c != 1 || x->dims.h != out->dims.h || x->dims.w != out->dims.w ||
        x->padding.b || x->padding.h || x->padding.w || x->padding.c || out->padding.b || out->padding.h ||
        out->padding.w || out->padding.c || x->full.b != x->dims.b || x->full.h != x->dims.h ||
        x->full.w != x->dims.w || x->full.c != x->dims.c || out->full.b != out->dims.b ||
        out->full.h != out->dims.h || out->full.w != out->dims.w || out->full.c != out->dims.c)
        return 0;
    for (uint32_t y = 0; y < x->dims.h; y++) {
        PART_ROW(slot);
        for (uint32_t k = 0; k < x->dims.w / 64u; k++) {
            const uint8_t *s = direct_at(x, 0, y, 64u * k, 0, 2);
            const HVX_Vector v0 = *(const HVX_UVector *)s;
            const HVX_Vector v1 = *(const HVX_UVector *)(s + 128);
            store_bytes(direct_at(out, 0, y, 64u * k, 0, 2), Q6_Vh_vpacke_VwVw(v1, v0), 128);
            store_bytes(direct_at(out, 1, y, 64u * k, 0, 2), Q6_Vh_vpacko_VwVw(v1, v0), 128);
        }
        /* Width not a multiple of 64 (the 96 px skeleton crop): the last pixels one at a time. */
        const uint16_t *s = (const uint16_t *)direct_at(x, 0, y, 0, 0, 2);
        uint16_t *d0 = (uint16_t *)direct_at(out, 0, y, 0, 0, 2), *d1 = (uint16_t *)direct_at(out, 1, y, 0, 0, 2);
        for (uint32_t p = x->dims.w & ~63u; p < x->dims.w; p++) {
            d0[p] = s[2u * p];
            d1[p] = s[2u * p + 1u];
        }
    }
    return 1;
}

/* out(0, w, c, h) = in(0, h, w, c) for direct tensors without padding: in is an H x K matrix of halfwords
 * (K = W * C) and out its K x H transpose, out[k * H + h] = in[h * K + k]. Blocks of 32 x 32 keep the reads and
 * writes of a block within a few rows; parts take output row blocks. */
static int transpose_direct_hwc_whc(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts) {
    uint32_t slot = 0;
    const Qnn16Shape a = x->dims, z = out->dims;
    if (!direct_ok(x, 2) || !direct_ok(out, 2) || a.b != 1 || z.b != 1 || z.h != a.w || z.w != a.c || z.c != a.h)
        return 0;
    const Qnn16Shape fa = x->full, fz = out->full, pa = x->padding, pz = out->padding;
    if (pa.b || pa.h || pa.w || pa.c || pz.b || pz.h || pz.w || pz.c || fa.b != a.b || fa.h != a.h || fa.w != a.w ||
        fa.c != a.c || fz.b != z.b || fz.h != z.h || fz.w != z.w || fz.c != z.c)
        return 0;
    const uint32_t H = a.h, K = a.w * a.c;
    const uint16_t *src = (const uint16_t *)direct_at(x, 0, 0, 0, 0, 2);
    uint16_t *dst = (uint16_t *)direct_at(out, 0, 0, 0, 0, 2);
    for (uint32_t k0 = 0; k0 < K; k0 += 32u) {
        PART_ROW(slot);
        const uint32_t k1 = k0 + 32u < K ? k0 + 32u : K;
        for (uint32_t h0 = 0; h0 < H; h0 += 32u) {
            const uint32_t h1 = h0 + 32u < H ? h0 + 32u : H;
            for (uint32_t k = k0; k < k1; k++)
                for (uint32_t h = h0; h < h1; h++)
                    dst[k * H + h] = src[h * K + k];
        }
    }
    return 1;
}

int qnn16_tile_transpose(const Qnn16Tensor *x, const Qnn16Tensor *out, const uint32_t perm[4], uint32_t part,
                         uint32_t parts) {
    if (!perm)
        return 0;
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2)
        return transpose_wc_blocks(x, out, part, parts);
    if (perm[0] == 3 && perm[1] == 1 && perm[2] == 2 && perm[3] == 0)
        return transpose_direct_b(x, out, part, parts);
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1)
        return transpose_direct_hwc_whc(x, out, part, parts);
    return 0;
}
