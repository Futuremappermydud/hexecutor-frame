#include "prepared_conv_u16.h"

#include <limits.h>
#include <string.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

enum {
    BLOCK_BYTES = 2048,
    GEOMETRY_CROUTON = 64,
    GEOMETRY_SHAPE = 1,
    DTYPE_I8 = 2,
    DTYPE_U16 = 3,
    DTYPE_I32 = 5
};

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int32_t rdi16(const uint8_t *p) {
    return (int16_t)rd16(p);
}

static uint8_t *crouton_vec(const Nprg2ExecCall *c, const Nprg2TensorView *v,
                            uint32_t b, uint32_t h, uint32_t w, uint32_t db) {
    const uint8_t *g = v->geometry_record;
    int32_t mh = rdi16(g + 16), mw = rdi16(g + 18), md = rdi16(g + 20);
    int32_t oh = rdi16(g + 24), ow = rdi16(g + 26), od = rdi16(g + 28);
    uint32_t H = h + (uint32_t)oh;
    uint32_t W = w + (uint32_t)ow;
    uint32_t D = db + (uint32_t)(od / 32);

    uint32_t nbh = ((uint32_t)mh + 7u) / 8u;
    uint32_t nbw = ((uint32_t)mw + 3u) / 4u;
    uint32_t nbd = ((uint32_t)md + 31u) / 32u;

    uint32_t block_idx = ((b * nbh + (H / 8u)) * nbw + (W / 4u)) * nbd + v->slice_origin + D;
    if (block_idx >= v->block_count || !v->blocks) return NULL;

    uint32_t entry = v->blocks[block_idx];
    uint32_t pool = entry & 127u;
    uint32_t offset = entry & ~127u;
    if (pool >= 7 || !c->pools[pool].base || offset + BLOCK_BYTES > c->pools[pool].bytes) return NULL;

    uint32_t byte_in_page = (H % 8u) * 256u + ((W % 4u) / 2u) * 128u;
    return c->pools[pool].base + offset + byte_in_page;
}

#if defined(__hexagon__)

static void decode_fi(const int32_t *fi_words, HVX_Vector *v_m, HVX_Vector *v_s,
                      HVX_Vector *v_rd, HVX_Vector *v_bias, HVX_Vector *v_pmax) {
    int32_t m_arr[32], s_arr[32], rd_arr[32], bias_arr[32], pmax_arr[32];
    for (int k = 0; k < 32; k++) {
        uint32_t h = (uint32_t)fi_words[k] & 0xffffu;
        uint32_t e = (h >> 10) & 31u;
        uint32_t mant = h & 1023u;
        int32_t m = 1024 + (int32_t)mant;
        int32_t s = 26 - (int32_t)e;
        int32_t rd = 1 << (s - 1);
        int64_t top = ((int64_t)65535 << s) - rd;
        m_arr[k] = m;
        s_arr[k] = s;
        rd_arr[k] = rd;
        bias_arr[k] = fi_words[32 + k];
        pmax_arr[k] = (int32_t)((top + m - 1) / m);
    }
    *v_m = *(const HVX_Vector *)m_arr;
    *v_s = *(const HVX_Vector *)s_arr;
    *v_rd = *(const HVX_Vector *)rd_arr;
    *v_bias = *(const HVX_Vector *)bias_arr;
    *v_pmax = *(const HVX_Vector *)pmax_arr;
}

static inline HVX_Vector requant_vec(HVX_Vector a, HVX_Vector v_m, HVX_Vector v_s,
                                     HVX_Vector v_rd, HVX_Vector v_pmax, HVX_Vector v_qmax) {
    HVX_Vector a_clamped = Q6_Vw_vmin_VwVw(a, v_pmax);
    HVX_Vector prod = Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(a_clamped, v_m), v_rd);
    HVX_Vector q = Q6_Vw_vasr_VwVw(prod, v_s);
    q = Q6_Vw_vmax_VwVw(q, Q6_V_vzero());
    return Q6_Vw_vmin_VwVw(q, v_qmax);
}

static void conv_1x1_hvx(const Nprg2ExecCall *call, const Nprg2TensorView *vx,
                         const Nprg2TensorView *vy, const int8_t *w_ptr, const int32_t *fi_ptr) {
    HVX_Vector v_m, v_s, v_rd, v_bias, v_pmax;
    decode_fi(fi_ptr, &v_m, &v_s, &v_rd, &v_bias, &v_pmax);
    HVX_Vector v_qmax = Q6_V_vsplat_R(65535u);
    HVX_Vector v_mask16 = Q6_V_vsplat_R(0x0000ffffu);

    HVX_Vector w_vec[8];
    for (int g = 0; g < 8; g++) {
        w_vec[g] = *(const HVX_Vector *)(w_ptr + g * 128);
    }

    for (uint32_t bx = 0; bx < 16; bx++) {
        for (uint32_t r = 0; r < 8; r++) {
            for (uint32_t pair = 0; pair < 2; pair++) {
                uint32_t w = bx * 4u + pair * 2u;
                uint8_t *in_ptr = crouton_vec(call, vx, 0, r, w, 0);
                uint8_t *out_ptr = crouton_vec(call, vy, 0, r, w, 0);
                if (!in_ptr || !out_ptr) continue;

                HVX_Vector v_in = *(const HVX_Vector *)in_ptr;
                HVX_Vector v_c0 = Q6_V_vand_VV(v_in, v_mask16);
                HVX_Vector v_c1 = Q6_Vuw_vlsr_VuwR(v_in, 16u);

                uint32_t c0[32], c1[32];
                *(HVX_Vector *)c0 = v_c0;
                *(HVX_Vector *)c1 = v_c1;

                HVX_Vector acc_lo0 = Q6_V_vzero(), acc_hi0 = Q6_V_vzero();
                HVX_Vector acc_lo1 = Q6_V_vzero(), acc_hi1 = Q6_V_vzero();

                for (int g = 0; g < 8; g++) {
                    uint32_t x0_0 = c0[4*g+0], x0_1 = c0[4*g+1], x0_2 = c0[4*g+2], x0_3 = c0[4*g+3];
                    uint32_t lo0 = (x0_0 & 0xff) | ((x0_1 & 0xff) << 8) | ((x0_2 & 0xff) << 16) | ((x0_3 & 0xff) << 24);
                    uint32_t hi0 = ((x0_0 >> 8) & 0xff) | (((x0_1 >> 8) & 0xff) << 8) | (((x0_2 >> 8) & 0xff) << 16) | (((x0_3 >> 8) & 0xff) << 24);

                    uint32_t x1_0 = c1[4*g+0], x1_1 = c1[4*g+1], x1_2 = c1[4*g+2], x1_3 = c1[4*g+3];
                    uint32_t lo1 = (x1_0 & 0xff) | ((x1_1 & 0xff) << 8) | ((x1_2 & 0xff) << 16) | ((x1_3 & 0xff) << 24);
                    uint32_t hi1 = ((x1_0 >> 8) & 0xff) | (((x1_1 >> 8) & 0xff) << 8) | (((x1_2 >> 8) & 0xff) << 16) | (((x1_3 >> 8) & 0xff) << 24);

                    acc_lo0 = Q6_Vw_vrmpyacc_VwVubVb(acc_lo0, Q6_V_vsplat_R(lo0), w_vec[g]);
                    acc_hi0 = Q6_Vw_vrmpyacc_VwVubVb(acc_hi0, Q6_V_vsplat_R(hi0), w_vec[g]);

                    acc_lo1 = Q6_Vw_vrmpyacc_VwVubVb(acc_lo1, Q6_V_vsplat_R(lo1), w_vec[g]);
                    acc_hi1 = Q6_Vw_vrmpyacc_VwVubVb(acc_hi1, Q6_V_vsplat_R(hi1), w_vec[g]);
                }

                HVX_Vector a0 = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(acc_lo0, 8), acc_hi0), v_bias);
                HVX_Vector a1 = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(acc_lo1, 8), acc_hi1), v_bias);

                HVX_Vector q0 = requant_vec(a0, v_m, v_s, v_rd, v_pmax, v_qmax);
                HVX_Vector q1 = requant_vec(a1, v_m, v_s, v_rd, v_pmax, v_qmax);

                HVX_Vector v_out = Q6_V_vor_VV(q0, Q6_Vw_vasl_VwR(q1, 16u));
                *(HVX_Vector *)out_ptr = v_out;
            }
        }
    }
}

static void conv_3x3_hvx(const Nprg2ExecCall *call, const Nprg2TensorView *vx,
                         const Nprg2TensorView *vy, const int8_t *w_ptr, const int32_t *fi_ptr,
                         uint32_t ci, uint32_t co) {
    uint32_t ocb = co / 32u;
    uint32_t icb = ci / 32u;
    HVX_Vector v_qmax = Q6_V_vsplat_R(65535u);
    HVX_Vector v_mask16 = Q6_V_vsplat_R(0x0000ffffu);

    uint32_t x_ow = (uint32_t)rdi16(vx->geometry_record + 26);

    for (uint32_t ob = 0; ob < ocb; ob++) {
        HVX_Vector v_m, v_s, v_rd, v_bias, v_pmax;
        decode_fi(fi_ptr + ob * 64, &v_m, &v_s, &v_rd, &v_bias, &v_pmax);

        for (uint32_t bx = 0; bx < 64; bx++) {
            for (uint32_t r = 0; r < 8; r++) {
                HVX_Vector acc_lo[4], acc_hi[4];
                for (int c = 0; c < 4; c++) {
                    acc_lo[c] = Q6_V_vzero();
                    acc_hi[c] = Q6_V_vzero();
                }

                for (uint32_t ib = 0; ib < icb; ib++) {
                    for (uint32_t ky = 0; ky < 3; ky++) {
                        uint32_t in_h = r + ky;
                        uint32_t col[6][32];
                        for (uint32_t k = 0; k < 6; k++) {
                            uint32_t in_w = bx * 4u + k;
                            uint8_t *p = crouton_vec(call, vx, 0, in_h, in_w, ib);
                            if (p) {
                                HVX_Vector vec = *(const HVX_Vector *)p;
                                uint32_t W = in_w + x_ow;
                                HVX_Vector ch = (W % 2u == 0) ? Q6_V_vand_VV(vec, v_mask16) : Q6_Vuw_vlsr_VuwR(vec, 16u);
                                *(HVX_Vector *)col[k] = ch;
                            } else {
                                *(HVX_Vector *)col[k] = Q6_V_vzero();
                            }
                        }

                        for (uint32_t kx = 0; kx < 3; kx++) {
                            uint32_t stored_kx = 2u - kx;
                            const int8_t *w_tap = w_ptr + ((((ob * icb + ib) * 3u + ky) * 3u + stored_kx) * 8u) * 128;

                            for (int g = 0; g < 8; g++) {
                                HVX_Vector vw = *(const HVX_Vector *)(w_tap + g * 128);

                                for (int c = 0; c < 4; c++) {
                                    uint32_t *t = col[c + kx];
                                    uint32_t x0 = t[4*g+0], x1 = t[4*g+1], x2 = t[4*g+2], x3 = t[4*g+3];
                                    uint32_t lo = (x0 & 0xff) | ((x1 & 0xff) << 8) | ((x2 & 0xff) << 16) | ((x3 & 0xff) << 24);
                                    uint32_t hi = ((x0 >> 8) & 0xff) | (((x1 >> 8) & 0xff) << 8) | (((x2 >> 8) & 0xff) << 16) | (((x3 >> 8) & 0xff) << 24);

                                    acc_lo[c] = Q6_Vw_vrmpyacc_VwVubVb(acc_lo[c], Q6_V_vsplat_R(lo), vw);
                                    acc_hi[c] = Q6_Vw_vrmpyacc_VwVubVb(acc_hi[c], Q6_V_vsplat_R(hi), vw);
                                }
                            }
                        }
                    }
                }

                HVX_Vector q[4];
                for (int c = 0; c < 4; c++) {
                    HVX_Vector a = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(acc_lo[c], 8), acc_hi[c]), v_bias);
                    q[c] = requant_vec(a, v_m, v_s, v_rd, v_pmax, v_qmax);
                }

                uint8_t *out0 = crouton_vec(call, vy, 0, r, bx * 4u + 0u, ob);
                uint8_t *out1 = crouton_vec(call, vy, 0, r, bx * 4u + 2u, ob);
                if (out0) *(HVX_Vector *)out0 = Q6_V_vor_VV(q[0], Q6_Vw_vasl_VwR(q[1], 16u));
                if (out1) *(HVX_Vector *)out1 = Q6_V_vor_VV(q[2], Q6_Vw_vasl_VwR(q[3], 16u));
            }
        }
    }
}

#else /* Portable scalar fallback for host build/verification */

static uint16_t requant_scalar(int64_t a, int32_t mult, int32_t shift, int32_t round) {
    int64_t prod = a * (int64_t)mult + round;
    int64_t q = prod >> shift;
    if (q < 0) q = 0;
    if (q > 65535) q = 65535;
    return (uint16_t)q;
}

static void conv_1x1_host(const Nprg2ExecCall *call, const Nprg2TensorView *vx,
                          const Nprg2TensorView *vy, const int8_t *w_ptr, const int32_t *fi_ptr) {
    int32_t m[32], s[32], rd[32], bias[32];
    for (int k = 0; k < 32; k++) {
        uint32_t h = (uint32_t)fi_ptr[k] & 0xffffu;
        uint32_t e = (h >> 10) & 31u;
        m[k] = 1024 + (int32_t)(h & 1023u);
        s[k] = 26 - (int32_t)e;
        rd[k] = 1 << (s[k] - 1);
        bias[k] = fi_ptr[32 + k];
    }

    for (uint32_t bx = 0; bx < 16; bx++) {
        for (uint32_t r = 0; r < 8; r++) {
            for (uint32_t w = bx * 4u; w < bx * 4u + 4u; w++) {
                uint8_t *in_page = crouton_vec(call, vx, 0, r, w, 0);
                uint8_t *out_page = crouton_vec(call, vy, 0, r, w, 0);
                if (!in_page || !out_page) continue;

                const uint8_t *xg = vx->geometry_record;
                uint32_t x_ow = (uint32_t)rdi16(xg + 26);
                const uint8_t *yg = vy->geometry_record;
                uint32_t y_ow = (uint32_t)rdi16(yg + 26);
                uint32_t in_parity = (w + x_ow) % 2u;
                uint32_t col_parity = (w + y_ow) % 2u;
                uint16_t in_ch[32];
                for (int c = 0; c < 32; c++) {
                    const uint8_t *p = in_page + c * 4u + in_parity * 2u;
                    in_ch[c] = rd16(p);
                }

                for (int co = 0; co < 32; co++) {
                    int64_t acc = 0;
                    for (int ci = 0; ci < 32; ci++) {
                        int32_t g = ci / 4;
                        int32_t sub = ci % 4;
                        int8_t weight = w_ptr[g * 128 + co * 4 + sub];
                        acc += (int64_t)in_ch[ci] * weight;
                    }
                    int64_t a = (acc >> 8) + bias[co];
                    uint16_t q = requant_scalar(a, m[co], s[co], rd[co]);
                    uint8_t *out_p = out_page + co * 4u + col_parity * 2u;
                    out_p[0] = (uint8_t)(q & 0xff);
                    out_p[1] = (uint8_t)(q >> 8);
                }
            }
        }
    }
}

static void conv_3x3_host(const Nprg2ExecCall *call, const Nprg2TensorView *vx,
                          const Nprg2TensorView *vy, const int8_t *w_ptr, const int32_t *fi_ptr,
                          uint32_t ci, uint32_t co) {
    uint32_t ocb = co / 32u;
    uint32_t icb = ci / 32u;

    for (uint32_t ob = 0; ob < ocb; ob++) {
        const int32_t *fi_ob = fi_ptr + ob * 64;
        int32_t m[32], s[32], rd[32], bias[32];
        for (int k = 0; k < 32; k++) {
            uint32_t h = (uint32_t)fi_ob[k] & 0xffffu;
            uint32_t e = (h >> 10) & 31u;
            m[k] = 1024 + (int32_t)(h & 1023u);
            s[k] = 26 - (int32_t)e;
            rd[k] = 1 << (s[k] - 1);
            bias[k] = fi_ob[32 + k];
        }

        for (uint32_t bx = 0; bx < 64; bx++) {
            for (uint32_t r = 0; r < 8; r++) {
                for (uint32_t w = bx * 4u; w < bx * 4u + 4u; w++) {
                    uint8_t *out_page = crouton_vec(call, vy, 0, r, w, ob);
                    if (!out_page) continue;

                    const uint8_t *xg = vx->geometry_record;
                    uint32_t x_ow = (uint32_t)rdi16(xg + 26);
                    const uint8_t *yg = vy->geometry_record;
                    uint32_t y_ow = (uint32_t)rdi16(yg + 26);
                    uint32_t col_parity = (w + y_ow) % 2u;

                    for (int co_idx = 0; co_idx < 32; co_idx++) {
                        int64_t acc = 0;
                        for (uint32_t ib = 0; ib < icb; ib++) {
                            for (uint32_t ky = 0; ky < 3; ky++) {
                                uint32_t in_h = r + ky;
                                for (uint32_t kx = 0; kx < 3; kx++) {
                                    uint32_t in_w = w + kx;
                                    uint8_t *in_page = crouton_vec(call, vx, 0, in_h, in_w, ib);
                                    uint32_t in_parity = (in_w + x_ow) % 2u;
                                    uint32_t stored_kx = 2u - kx;
                                    const int8_t *w_tap = w_ptr + ((((ob * icb + ib) * 3u + ky) * 3u + stored_kx) * 8u) * 128;

                                    for (int ci_idx = 0; ci_idx < 32; ci_idx++) {
                                        uint16_t x_val = 0;
                                        if (in_page) {
                                            const uint8_t *p = in_page + ci_idx * 4u + in_parity * 2u;
                                            x_val = rd16(p);
                                        }
                                        int32_t g = ci_idx / 4;
                                        int32_t sub = ci_idx % 4;
                                        int8_t weight = w_tap[g * 128 + co_idx * 4 + sub];
                                        acc += (int64_t)x_val * weight;
                                    }
                                }
                            }
                        }
                        int64_t a = (acc >> 8) + bias[co_idx];
                        uint16_t q = requant_scalar(a, m[co_idx], s[co_idx], rd[co_idx]);
                        uint8_t *out_p = out_page + co_idx * 4u + col_parity * 2u;
                        out_p[0] = (uint8_t)(q & 0xff);
                        out_p[1] = (uint8_t)(q >> 8);
                    }
                }
            }
        }
    }
}

#endif /* __hexagon__ */

int nprg2_exec_conv_u16_validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode != QDR_HMX_CONV ||
        c->command.input_count < 3 || c->command.output_count != 1) return 0;
    if (c->command.variant != 34 && c->command.variant != 35) return 0;

    uint32_t xi, wi, fi, yi;
    Nprg2TensorView x, w, f, y;
    if (nprg2_exec_input(c, 0, &xi, &x) != NPRG2_EXEC_OK ||
        nprg2_exec_input(c, 1, &wi, &w) != NPRG2_EXEC_OK ||
        nprg2_exec_input(c, 2, &fi, &f) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &yi, &y) != NPRG2_EXEC_OK) return 0;

    const uint8_t *xg = x.geometry_record, *wg = w.geometry_record;
    const uint8_t *fg = f.geometry_record, *yg = y.geometry_record;

    if (!(xg[1] & GEOMETRY_CROUTON) || !(yg[1] & GEOMETRY_CROUTON) ||
        xg[0] != DTYPE_U16 || yg[0] != DTYPE_U16 ||
        memcmp(xg + 2, "CH", 2) || memcmp(yg + 2, "CH", 2) ||
        x.storage_kind != 2 || y.storage_kind != 2 ||
        !x.blocks || !y.blocks) return 0;

    if (w.storage_kind != 1 || f.storage_kind != 1 ||
        wg[0] != DTYPE_I8 || fg[0] != DTYPE_I32 ||
        w.pool != 1 || f.pool != 1) return 0;

    if ((w.offset & 127u) || (f.offset & 127u)) return 0;
    if (w.offset > c->pools[1].bytes || f.offset > c->pools[1].bytes) return 0;

    for (uint32_t i = 0; i < y.block_count; i++) {
        uint32_t entry = y.blocks[i];
        uint32_t pool = entry & 127u;
        uint32_t offset = entry & ~127u;
        if (pool != 1 || (offset & (BLOCK_BYTES - 1u)) ||
            offset > c->pool1_writable_bytes ||
            BLOCK_BYTES > c->pool1_writable_bytes - offset) return 0;
    }
    for (uint32_t i = 0; i < x.block_count; i++) {
        uint32_t entry = x.blocks[i];
        uint32_t pool = entry & 127u;
        uint32_t offset = entry & ~127u;
        if (pool >= 7 || !c->pools[pool].base || (offset & (BLOCK_BYTES - 1u)) ||
            offset > c->pools[pool].bytes ||
            BLOCK_BYTES > c->pools[pool].bytes - offset) return 0;
    }

    if (c->command.variant == 34) {
        if (memcmp(wg + 2, "Xpb", 3) && memcmp(wg + 2, "xpb", 3)) return 0;
    } else {
        if (memcmp(wg + 2, "Xwb", 3) && memcmp(wg + 2, "xwb", 3)) return 0;
    }
    return 1;
}

int nprg2_exec_conv_u16_run(const Nprg2ExecCall *c, void *ctx) {
    if (!nprg2_exec_conv_u16_validate(c, ctx)) return 0;

    uint32_t xi, wi, fi, yi;
    Nprg2TensorView x, w, f, y;
    nprg2_exec_input(c, 0, &xi, &x);
    nprg2_exec_input(c, 1, &wi, &w);
    nprg2_exec_input(c, 2, &fi, &f);
    nprg2_exec_output(c, 0, &yi, &y);

    const int8_t *w_ptr = (const int8_t *)(c->pools[w.pool].base + w.offset);
    const int32_t *fi_ptr = (const int32_t *)(c->pools[f.pool].base + f.offset);

    const uint8_t *wg = w.geometry_record;
    uint32_t kh = (uint32_t)rdi16(wg + 14);
    uint32_t kw = (uint32_t)rdi16(wg + 16);
    uint32_t ci = (uint32_t)rdi16(wg + 18);
    uint32_t co = (uint32_t)rdi16(wg + 20);

#if defined(__hexagon__)
    if (c->command.variant == 34 && kh == 1 && kw == 1) {
        conv_1x1_hvx(c, &x, &y, w_ptr, fi_ptr);
        return 1;
    }
    if (c->command.variant == 35 && kh == 3 && kw == 3) {
        conv_3x3_hvx(c, &x, &y, w_ptr, fi_ptr, ci, co);
        return 1;
    }
#else
    if (c->command.variant == 34 && kh == 1 && kw == 1) {
        conv_1x1_host(c, &x, &y, w_ptr, fi_ptr);
        return 1;
    }
    if (c->command.variant == 35 && kh == 3 && kw == 3) {
        conv_3x3_host(c, &x, &y, w_ptr, fi_ptr, ci, co);
        return 1;
    }
#endif
    return 0;
}

int nprg2_executor_register_conv_u16(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    int rc = nprg2_executor_register(e, QDR_HMX_CONV, 34,
                                     nprg2_exec_conv_u16_validate,
                                     nprg2_exec_conv_u16_run, NULL);
    if (rc != NPRG2_EXEC_OK) return rc;

    return nprg2_executor_register(e, QDR_HMX_CONV, 35,
                                  nprg2_exec_conv_u16_validate,
                                  nprg2_exec_conv_u16_run, NULL);
}
