/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_abs.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    BLOCK_BYTES = 2048,
    GEOMETRY_CROUTON = 64,
    GEOMETRY_SHAPE = 1,
    DTYPE_U16 = 3
};

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int32_t rdi16(const uint8_t *p) {
    return (int16_t)rd16(p);
}

static int32_t rd32(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static double rd_scale(const uint8_t *p) {
    double d;
    memcpy(&d, p, sizeof(double));
    return d;
}

static uint8_t *crouton_addr(const Nprg2ExecCall *c, const Nprg2TensorView *v,
                             uint32_t b, uint32_t h, uint32_t w, uint32_t d_idx) {
    const uint8_t *g = v->geometry_record;
    int32_t mh = rdi16(g + 16), mw = rdi16(g + 18), md = rdi16(g + 20);
    int32_t oh = rdi16(g + 24), ow = rdi16(g + 26), od = rdi16(g + 28);
    uint32_t H = h + (uint32_t)oh;
    uint32_t W = w + (uint32_t)ow;
    uint32_t D = d_idx + (uint32_t)od;

    uint32_t nbh = ((uint32_t)mh + 7u) / 8u;
    uint32_t nbw = ((uint32_t)mw + 3u) / 4u;
    uint32_t nbd = ((uint32_t)md + 31u) / 32u;

    uint32_t block_idx = ((b * nbh + (H / 8u)) * nbw + (W / 4u)) * nbd + v->slice_origin + (D / 32u);
    if (block_idx >= v->block_count || !v->blocks) return NULL;

    uint32_t entry = v->blocks[block_idx];
    uint32_t pool = entry & 127u;
    uint32_t offset = entry & ~127u;
    if (pool >= 7 || !c->pools[pool].base || offset + BLOCK_BYTES > c->pools[pool].bytes) return NULL;

    uint32_t byte_in_page = (H % 8u) * 256u + ((W % 4u) / 2u) * 128u + (D % 32u) * 4u + (W % 2u) * 2u;
    return c->pools[pool].base + offset + byte_in_page;
}

int nprg2_exec_abs_validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode != QDR_ABS ||
        c->command.input_count != 1 || c->command.output_count != 1) return 0;
    if (c->command.variant != 12 && c->command.variant != 11) return 0;

    uint32_t in_idx, out_idx;
    Nprg2TensorView in, out;
    if (nprg2_exec_input(c, 0, &in_idx, &in) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &out_idx, &out) != NPRG2_EXEC_OK) return 0;

    const uint8_t *ig = in.geometry_record, *og = out.geometry_record;
    if (!ig || !og || ig[0] != DTYPE_U16 || og[0] != DTYPE_U16) return 0;

    for (int i = 0; i < 4; i++) {
        if (rdi16(ig + 6 + i * 2) != rdi16(og + 6 + i * 2)) return 0;
    }

    if (c->command.variant == 12) {
        /* Abs@FH*2: Direct flat halfword */
        if (in.storage_kind != 1 || out.storage_kind != 1 ||
            in.pool >= 7 || out.pool >= 7 ||
            !c->pools[in.pool].base || !c->pools[out.pool].base) return 0;
        uint32_t count = (uint32_t)rdi16(ig + 6) * (uint32_t)rdi16(ig + 8) *
                         (uint32_t)rdi16(ig + 10) * (uint32_t)rdi16(ig + 12);
        uint32_t bytes = count * 2u;
        if (in.offset + bytes > c->pools[in.pool].bytes ||
            out.offset + bytes > c->pools[out.pool].bytes) return 0;
        if (out.pool == 1 && out.offset + bytes > c->pool1_writable_bytes) return 0;
        if (in.pool == out.pool && in.offset < out.offset + bytes && out.offset < in.offset + bytes) return 0;
    } else {
        /* Abs@CH*2: Crouton CH u16 */
        if (in.storage_kind != 2 || out.storage_kind != 2 ||
            !(ig[1] & GEOMETRY_CROUTON) || !(og[1] & GEOMETRY_CROUTON) ||
            memcmp(ig + 2, "CH", 2) || memcmp(og + 2, "CH", 2) ||
            !in.blocks || !out.blocks) return 0;
    }
    return 1;
}

int nprg2_exec_abs_run(const Nprg2ExecCall *c, void *ctx) {
    if (!nprg2_exec_abs_validate(c, ctx)) return 0;

    uint32_t in_idx, out_idx;
    Nprg2TensorView in, out;
    nprg2_exec_input(c, 0, &in_idx, &in);
    nprg2_exec_output(c, 0, &out_idx, &out);

    const uint8_t *ig = in.geometry_record, *og = out.geometry_record;
    int32_t i_zp = rd32(ig + 30);
    double i_scale = rd_scale(ig + 34);
    int32_t o_zp = rd32(og + 30);
    double o_scale = rd_scale(og + 34);

    if (o_scale == 0.0) return 0;
    float ratio = (float)i_scale * (float)(1.0 / o_scale);
    int64_t mult = (int64_t)rintf(ratio * 262144.0f);
    volatile float a = ratio * 32768.0f;
    volatile float b = a + (float)o_zp;
    volatile float e = b * 8.0f;
    int64_t offset = (int64_t)rintf(e);

    if (c->command.variant == 12) {
        /* Flat FH direct run */
        uint32_t b_dim = (uint32_t)rdi16(ig + 6);
        uint32_t h_dim = (uint32_t)rdi16(ig + 8);
        uint32_t w_dim = (uint32_t)rdi16(ig + 10);
        uint32_t d_dim = (uint32_t)rdi16(ig + 12);
        uint32_t total = b_dim * h_dim * w_dim * d_dim;

        const uint16_t *src = (const uint16_t *)(c->pools[in.pool].base + in.offset);
        uint16_t *dst = (uint16_t *)(c->pools[out.pool].base + out.offset);

        for (uint32_t k = 0; k < total; k++) {
            int64_t lane = (k % 2u == 0);
            int64_t centered = (int64_t)llabs((int64_t)src[k] - i_zp) - 32768;
            int64_t y = ((centered * mult + lane * 16384) >> 15) + offset;
            int64_t v = (y + 4) >> 3;
            if (v < 0) v = 0;
            if (v > 65535) v = 65535;
            dst[k] = (uint16_t)v;
        }
        return 1;
    }

    if (c->command.variant == 11) {
        /* Blocked Crouton CH run */
        uint32_t b_dim = (uint32_t)rdi16(ig + 6);
        uint32_t h_dim = (uint32_t)rdi16(ig + 8);
        uint32_t w_dim = (uint32_t)rdi16(ig + 10);
        uint32_t d_dim = (uint32_t)rdi16(ig + 12);

        /* Fast form: one address per run of channels inside one block of
         * both tensors (channel stride 4 bytes in a CH crouton), and
         * 32x32 -> 64 products when mult fits 32 bits. Same arithmetic. */
        if (mult >= INT32_MIN && mult <= INT32_MAX) {
            uint32_t iod = (uint32_t)rdi16(ig + 28), ood = (uint32_t)rdi16(og + 28);
            int32_t m32 = (int32_t)mult;
            for (uint32_t bb = 0; bb < b_dim; bb++)
            for (uint32_t h = 0; h < h_dim; h++)
            for (uint32_t w = 0; w < w_dim; w++) {
                int64_t lane16 = (w % 2u == 0) ? 16384 : 0;
                for (uint32_t d = 0; d < d_dim;) {
                    uint32_t n = 32u - (d + iod) % 32u, n2 = 32u - (d + ood) % 32u;
                    if (n2 < n) n = n2;
                    if (d_dim - d < n) n = d_dim - d;
                    const uint8_t *ip = crouton_addr(c, &in, bb, h, w, d);
                    uint8_t *op = crouton_addr(c, &out, bb, h, w, d);
                    if (!ip || !op) return 0;
                    for (uint32_t k = 0; k < n; k++) {
                        int32_t val = (int32_t)rd16(ip + 4u * k) - i_zp;
                        int32_t centered = (val < 0 ? -val : val) - 32768;
                        int64_t y = (((int64_t)centered * m32 + lane16) >> 15) + offset;
                        int64_t v = (y + 4) >> 3;
                        v = v < 0 ? 0 : v > 65535 ? 65535 : v;
                        op[4u * k] = (uint8_t)(v & 0xff);
                        op[4u * k + 1] = (uint8_t)(v >> 8);
                    }
                    d += n;
                }
            }
            return 1;
        }
        for (uint32_t bb = 0; bb < b_dim; bb++) {
            for (uint32_t h = 0; h < h_dim; h++) {
                for (uint32_t w = 0; w < w_dim; w++) {
                    int64_t lane = (w % 2u == 0);
                    for (uint32_t d = 0; d < d_dim; d++) {
                        uint8_t *in_p = crouton_addr(c, &in, bb, h, w, d);
                        uint8_t *out_p = crouton_addr(c, &out, bb, h, w, d);
                        if (!in_p || !out_p) return 0;

                        uint16_t val = rd16(in_p);
                        int64_t centered = (int64_t)llabs((int64_t)val - i_zp) - 32768;
                        int64_t y = ((centered * mult + lane * 16384) >> 15) + offset;
                        int64_t v = (y + 4) >> 3;
                        if (v < 0) v = 0;
                        if (v > 65535) v = 65535;

                        out_p[0] = (uint8_t)(v & 0xff);
                        out_p[1] = (uint8_t)(v >> 8);
                    }
                }
            }
        }
        return 1;
    }
    return 0;
}

int nprg2_executor_register_abs(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    int rc = nprg2_executor_register(e, QDR_ABS, 12, nprg2_exec_abs_validate, nprg2_exec_abs_run, NULL);
    if (rc != NPRG2_EXEC_OK) return rc;
    return nprg2_executor_register(e, QDR_ABS, 11, nprg2_exec_abs_validate, nprg2_exec_abs_run, NULL);
}
