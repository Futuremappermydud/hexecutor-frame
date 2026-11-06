#include "prepared_conv_s2.h"

#include <limits.h>
#include <string.h>

enum {
    BLOCK_BYTES = 2048,
    GEOMETRY_CROUTON = 64,
    GEOMETRY_SHAPE = 1,
    DTYPE_U8 = 1,
    DTYPE_I8 = 2,
    DTYPE_I32 = 5,
    QDR_OPCODE_HMX_CONV = 6,
    QDR_VARIANT_S2 = 36
};

/* Tap index mapping from (ky, kx) to stored Xwb tap index k in 0..8:
 * TAPS order: ((0, 2), (0, 0), (2, 2), (2, 0), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1)) */
static const int8_t TAP_INDEX[3][3] = {
    {1, 4, 0},
    {7, 8, 6},
    {3, 5, 2}
};

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int32_t rdi16(const uint8_t *p) {
    return (int16_t)rd16(p);
}

static int get_dims(const uint8_t *g, uint32_t d[4], uint32_t m[4], uint32_t p[4]) {
    if (!g || !(g[1] & GEOMETRY_SHAPE)) return 0;
    for (int i = 0; i < 4; i++) {
        int32_t dv = rdi16(g + 6 + i * 2);
        int32_t mv = rdi16(g + 14 + i * 2);
        int32_t pv = rdi16(g + 22 + i * 2);
        if (dv < 0 || mv < 0 || pv < 0) return 0;
        d[i] = (uint32_t)dv;
        m[i] = (uint32_t)mv;
        p[i] = (uint32_t)pv;
    }
    return 1;
}

static uint8_t *crouton_addr_u8(const Nprg2ExecCall *c, const Nprg2TensorView *v,
                                uint32_t b, uint32_t h, uint32_t w, uint32_t d_idx) {
    const uint8_t *g = v->geometry_record;
    int32_t mh = rdi16(g + 16), mw = rdi16(g + 18), md = rdi16(g + 20);
    int32_t oh = rdi16(g + 24), ow = rdi16(g + 26), od = rdi16(g + 28);
    uint32_t H = h + (uint32_t)oh;
    uint32_t W = w + (uint32_t)ow;
    uint32_t D = d_idx + (uint32_t)od;

    uint32_t nbh = ((uint32_t)mh + 7u) / 8u;
    uint32_t nbw = ((uint32_t)mw + 7u) / 8u;
    uint32_t nbd = ((uint32_t)md + 31u) / 32u;

    uint32_t block_idx = ((b * nbh + (H / 8u)) * nbw + (W / 8u)) * nbd + v->slice_origin + (D / 32u);
    if (block_idx >= v->block_count || !v->blocks) return NULL;

    uint32_t entry = v->blocks[block_idx];
    uint32_t pool = entry & 127u;
    uint32_t offset = entry & ~127u;
    if (pool >= 7 || !c->pools[pool].base || offset + BLOCK_BYTES > c->pools[pool].bytes) return NULL;

    uint32_t byte_in_page = (H % 8u) * 256u + (W % 8u) * 32u + (D % 32u);
    return c->pools[pool].base + offset + byte_in_page;
}

int nprg2_conv_s2_setup_descriptors(const Nprg2ExecCall *c, Nprg2ConvS2Descriptors *desc) {
    if (!c || !desc) return 0;
    memset(desc, 0, sizeof(*desc));

    uint32_t xi, wi, fi, yi;
    Nprg2TensorView x, w, f, y;
    if (nprg2_exec_input(c, 0, &xi, &x) != NPRG2_EXEC_OK ||
        nprg2_exec_input(c, 1, &wi, &w) != NPRG2_EXEC_OK ||
        nprg2_exec_input(c, 2, &fi, &f) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &yi, &y) != NPRG2_EXEC_OK) return 0;

    uint32_t xd[4], xm[4], xp[4];
    uint32_t yd[4], ym[4], yp[4];
    uint32_t wd[4], wm[4], wp[4];
    uint32_t fd[4], fm[4], fp[4];
    if (!get_dims(x.geometry_record, xd, xm, xp) ||
        !get_dims(y.geometry_record, yd, ym, yp) ||
        !get_dims(w.geometry_record, wd, wm, wp) ||
        !get_dims(f.geometry_record, fd, fm, fp)) return 0;

    memcpy(desc->input_shape.dims, xd, sizeof(xd));
    memcpy(desc->input_shape.max_dims, xm, sizeof(xm));
    memcpy(desc->output_shape.dims, yd, sizeof(yd));
    memcpy(desc->output_shape.max_dims, ym, sizeof(ym));
    memcpy(desc->weight_shape.dims, wd, sizeof(wd));
    memcpy(desc->weight_shape.max_dims, wm, sizeof(wm));
    memcpy(desc->fi_shape.dims, fd, sizeof(fd));
    memcpy(desc->fi_shape.max_dims, fm, sizeof(fm));

    uint32_t n_in = x.block_count > 256 ? 256 : x.block_count;
    for (uint32_t i = 0; i < n_in; i++) {
        uint32_t entry = x.blocks[i];
        uint32_t pool = entry & 127u;
        uint32_t offset = entry & ~127u;
        if (pool < 7 && c->pools[pool].base && offset + BLOCK_BYTES <= c->pools[pool].bytes) {
            desc->input_block_table[i] = (uint32_t)(uintptr_t)(c->pools[pool].base + offset);
        }
    }

    uint32_t n_out = y.block_count > 256 ? 256 : y.block_count;
    for (uint32_t i = 0; i < n_out; i++) {
        uint32_t entry = y.blocks[i];
        uint32_t pool = entry & 127u;
        uint32_t offset = entry & ~127u;
        if (pool < 7 && c->pools[pool].base && offset + BLOCK_BYTES <= c->pools[pool].bytes) {
            desc->output_block_table[i] = (uint32_t)(uintptr_t)(c->pools[pool].base + offset);
        }
    }

    desc->input_obj.shape = (uint32_t)(uintptr_t)&desc->input_shape;
    desc->input_obj.data = (uint32_t)(uintptr_t)desc->input_block_table;

    desc->output_obj.shape = (uint32_t)(uintptr_t)&desc->output_shape;
    desc->output_obj.data = (uint32_t)(uintptr_t)desc->output_block_table;

    desc->weight_obj.shape = (uint32_t)(uintptr_t)&desc->weight_shape;
    desc->weight_obj.data = (uint32_t)(uintptr_t)(c->pools[w.pool].base + w.offset);

    desc->fi_obj.shape = (uint32_t)(uintptr_t)&desc->fi_shape;
    desc->fi_obj.data = (uint32_t)(uintptr_t)(c->pools[f.pool].base + f.offset);

    return 1;
}

int nprg2_exec_conv_s2_validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode != QDR_OPCODE_HMX_CONV ||
        c->command.variant != QDR_VARIANT_S2 ||
        c->command.input_count < 3 || c->command.output_count != 1) return 0;

    uint32_t xi, wi, fi, yi;
    Nprg2TensorView x, w, f, y;
    if (nprg2_exec_input(c, 0, &xi, &x) != NPRG2_EXEC_OK ||
        nprg2_exec_input(c, 1, &wi, &w) != NPRG2_EXEC_OK ||
        nprg2_exec_input(c, 2, &fi, &f) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &yi, &y) != NPRG2_EXEC_OK) return 0;

    const uint8_t *xg = x.geometry_record, *wg = w.geometry_record;
    const uint8_t *fg = f.geometry_record, *yg = y.geometry_record;

    if (!xg || !wg || !fg || !yg) return 0;

    if (xg[0] != DTYPE_U8 || yg[0] != DTYPE_U8 ||
        wg[0] != DTYPE_I8 || fg[0] != DTYPE_I32) return 0;

    if (!(xg[1] & GEOMETRY_CROUTON) || !(yg[1] & GEOMETRY_CROUTON) ||
        !(wg[1] & GEOMETRY_SHAPE) || !(fg[1] & GEOMETRY_SHAPE)) return 0;

    if (memcmp(xg + 2, "CB", 2) && memcmp(xg + 2, "cb", 2)) return 0;
    if (memcmp(yg + 2, "CB", 2) && memcmp(yg + 2, "cb", 2)) return 0;
    if (memcmp(wg + 2, "Xwb", 3) && memcmp(wg + 2, "xwb", 3)) return 0;
    if (memcmp(fg + 2, "Fi", 2) && memcmp(fg + 2, "fi", 2)) return 0;

    if (x.storage_kind != 2 || y.storage_kind != 2 ||
        w.storage_kind != 1 || f.storage_kind != 1) return 0;

    if (!x.blocks || !y.blocks || !x.block_count || !y.block_count) return 0;
    if (w.pool >= 7 || f.pool >= 7 || !c->pools[w.pool].base || !c->pools[f.pool].base) return 0;

    uint32_t xd[4], xm[4], xp[4];
    uint32_t yd[4], ym[4], yp[4];
    uint32_t wd[4], wm[4], wp[4];
    uint32_t fd[4], fm[4], fp[4];
    if (!get_dims(xg, xd, xm, xp) || !get_dims(yg, yd, ym, yp) ||
        !get_dims(wg, wd, wm, wp) || !get_dims(fg, fd, fm, fp)) return 0;

    if (xd[0] != 1 || yd[0] != 1 || wd[0] != 3 || wd[1] != 3) return 0;
    if (fd[0] != 1 || fd[2] != 1 || fd[3] != 64) return 0;

    uint32_t cin = xd[3];
    uint32_t cout = yd[3];
    uint32_t ci_w = wd[2];

    if (cin != 4u * ci_w) return 0;
    if (!((cin == 128u && ci_w == 32u && cout == 64u && fd[1] == 2u) ||
          (cin == 256u && ci_w == 64u && cout == 128u && fd[1] == 4u) ||
          (cin == 512u && ci_w == 128u && cout == 128u && fd[1] == 4u))) return 0;

    if (yd[1] != xd[1] - 1u || yd[2] != xd[2] - 2u) return 0;

    size_t w_bytes = (size_t)3 * 3 * ci_w * cout;
    size_t fi_bytes = (size_t)cout * 8;
    if (w.offset > c->pools[w.pool].bytes || w_bytes > c->pools[w.pool].bytes - w.offset) return 0;
    if (f.offset > c->pools[f.pool].bytes || fi_bytes > c->pools[f.pool].bytes - f.offset) return 0;

    uint32_t nbh_out = (ym[1] + 7u) / 8u;
    uint32_t nbw_out = (ym[2] + 7u) / 8u;
    uint32_t nbd_out = (ym[3] + 31u) / 32u;
    uint32_t n_out_blocks = nbh_out * nbw_out * nbd_out;

    if (y.slice_origin > y.block_count || n_out_blocks > y.block_count - y.slice_origin) return 0;

    for (uint32_t k = 0; k < n_out_blocks; k++) {
        uint32_t entry = y.blocks[y.slice_origin + k];
        uint32_t pool = entry & 127u;
        uint32_t offset = entry & ~127u;
        if (pool != 1 || !c->pools[1].base) return 0;
        if (offset + BLOCK_BYTES > c->pool1_writable_bytes) return 0;
    }

    for (uint32_t k = 0; k < n_out_blocks; k++) {
        uint32_t out_off = y.blocks[y.slice_origin + k] & ~127u;

        if (w.pool == 1 && out_off < w.offset + w_bytes && w.offset < out_off + BLOCK_BYTES) return 0;
        if (f.pool == 1 && out_off < f.offset + fi_bytes && f.offset < out_off + BLOCK_BYTES) return 0;

        for (uint32_t ik = 0; ik < x.block_count; ik++) {
            uint32_t in_entry = x.blocks[ik];
            if ((in_entry & 127u) == 1u) {
                uint32_t in_off = in_entry & ~127u;
                if (out_off == in_off) return 0;
            }
        }

        for (uint32_t j = k + 1; j < n_out_blocks; j++) {
            uint32_t other_off = y.blocks[y.slice_origin + j] & ~127u;
            if (out_off == other_off) return 0;
        }
    }

    return 1;
}

int nprg2_exec_conv_s2_run(const Nprg2ExecCall *c, void *ctx) {
    if (!nprg2_exec_conv_s2_validate(c, ctx)) return 0;

    Nprg2ConvS2Descriptors desc;
    if (!nprg2_conv_s2_setup_descriptors(c, &desc)) return 0;

    uint32_t xi, wi, fi, yi;
    Nprg2TensorView x, w, f, y;
    nprg2_exec_input(c, 0, &xi, &x);
    nprg2_exec_input(c, 1, &wi, &w);
    nprg2_exec_input(c, 2, &fi, &f);
    nprg2_exec_output(c, 0, &yi, &y);

    const int8_t *w_ptr = (const int8_t *)(c->pools[w.pool].base + w.offset);
    const int32_t *fi_ptr = (const int32_t *)(c->pools[f.pool].base + f.offset);

    uint32_t cin = desc.input_shape.dims[3];
    uint32_t cout = desc.output_shape.dims[3];
    uint32_t ci_w = desc.weight_shape.dims[2];
    uint32_t out_h = desc.output_shape.dims[1];
    uint32_t out_w = desc.output_shape.dims[2];

    uint32_t ocb = cout / 32u;
    uint32_t icb = cin / 32u;

    uint32_t nbh_out = (desc.output_shape.max_dims[1] + 7u) / 8u;
    uint32_t nbw_out = (desc.output_shape.max_dims[2] + 7u) / 8u;

    for (uint32_t ob = 0; ob < ocb; ob++) {
        int32_t mults[32], biases[32];
        for (int c_idx = 0; c_idx < 32; c_idx++) {
            mults[c_idx] = fi_ptr[ob * 64 + c_idx];
            biases[c_idx] = fi_ptr[ob * 64 + 32 + c_idx];
        }

        for (uint32_t by = 0; by < nbh_out; by++) {
            for (uint32_t bx = 0; bx < nbw_out; bx++) {
                uint8_t *out_block = crouton_addr_u8(c, &y, 0, by * 8u, bx * 8u, ob * 32u);
                if (!out_block) continue;

                for (uint32_t r = 0; r < 8u; r++) {
                    uint32_t pix_y = by * 8u + r;
                    if (pix_y >= out_h) continue;

                    for (uint32_t col = 0; col < 8u; col++) {
                        uint32_t pix_x = bx * 8u + col;
                        if (pix_x >= out_w) continue;

                        int64_t acc[32];
                        memset(acc, 0, sizeof(acc));

                        for (uint32_t a = 0; a < 2u; a++) {
                            for (uint32_t b = 0; b < 2u; b++) {
                                uint32_t in_y = pix_y + a;
                                uint32_t in_x = pix_x + b;

                                for (uint32_t ib = 0; ib < icb; ib++) {
                                    uint32_t cb = ib / 4u;
                                    uint32_t rem_ib = ib % 4u;
                                    uint32_t dy = rem_ib / 2u;
                                    uint32_t dx = rem_ib % 2u;

                                    uint32_t ky = 2u * a + dy;
                                    int32_t kx = (int32_t)(2u * b) - (int32_t)dx;
                                    if (ky >= 3u || kx < 0 || kx >= 3) continue;

                                    int8_t k = TAP_INDEX[ky][kx];
                                    uint32_t th = (uint32_t)k / 3u;
                                    uint32_t tw = (uint32_t)k % 3u;

                                    uint8_t *in_page = crouton_addr_u8(c, &x, 0, in_y, in_x, ib * 32u);
                                    if (!in_page) continue;

                                    size_t base_group = (((size_t)ob * (ci_w / 32u) + cb) * 3u + th) * 3u + tw;
                                    size_t group_off = base_group * 1024u;

                                    for (uint32_t sub_c = 0; sub_c < 32u; sub_c++) {
                                        uint8_t xv = in_page[sub_c];
                                        if (xv == 0) continue;

                                        size_t off = group_off + ((sub_c / 4u) % 8u) * 128u + (sub_c % 4u);
                                        const int8_t *w_row = w_ptr + off;

                                        for (int o_sub = 0; o_sub < 32; o_sub++) {
                                            acc[o_sub] += (int64_t)xv * (int64_t)w_row[o_sub * 4];
                                        }
                                    }
                                }
                            }
                        }

                        uint8_t *out_p = out_block + r * 256u + col * 32u;
                        for (int o_sub = 0; o_sub < 32; o_sub++) {
                            uint32_t hword = (uint32_t)mults[o_sub] & 0xffffu;
                            uint32_t shift = 34u - ((hword >> 10u) & 31u);
                            uint32_t pre = shift - 14u;
                            uint32_t post = 14u;
                            int64_t a_val = acc[o_sub] + (int64_t)biases[o_sub];
                            int64_t q = (((a_val >> pre) * (1024 + (hword & 1023u)) + ((int64_t)1 << (post - 1u))) >> post);
                            if (q < 0) q = 0;
                            if (q > 255) q = 255;
                            out_p[o_sub] = (uint8_t)q;
                        }
                    }
                }
            }
        }
    }

    return 1;
}

int nprg2_executor_register_conv_s2(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    return nprg2_executor_register(e, QDR_OPCODE_HMX_CONV, QDR_VARIANT_S2,
                                  nprg2_exec_conv_s2_validate,
                                  nprg2_exec_conv_s2_run, NULL);
}
