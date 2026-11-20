/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX LINEAR_Q row kernel (graph.c, FP32 path) against a double reference on the Hexagon simulator. */
#include "graph.c"
#include <stdio.h>

static uint64_t hx_cycles_now(void) { uint64_t c; __asm__ volatile("%0 = pcycle" : "=r"(c)); return c; }
static uint32_t rng = 12345u;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
static uint16_t to_half(float f) { /* normal positive/negative values only */
    union { float f; uint32_t u; } v = {f};
    const uint32_t sign = (v.u >> 16) & 0x8000u, exp = (v.u >> 23) & 255u, mant = (v.u >> 13) & 1023u;
    return (uint16_t)(sign | ((exp - 112u) << 10) | mant);
}
static double ref_row(const uint8_t *w, const float *x, uint32_t blocks, uint32_t fmt) {
    double sum = 0;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * (fmt == 0 ? 18u : 34u);
        const uint32_t sa = fmt == 0 ? 16u : 32u;
        const double s = hx_half_to_float((uint16_t)(blk[sa] | (blk[sa + 1] << 8)));
        double a = 0;
        for (uint32_t j = 0; j < 32; j++) {
            const int q = fmt == 0 ? ((j & 1) ? (blk[j / 2] >> 4) : (blk[j / 2] & 15)) - 8 : (int8_t)blk[j];
            a += (double)x[b * 32 + j] * q;
        }
        sum += a * s;
    }
    return sum;
}
static uint8_t wbuf[34 * 256 * 4 + 256] __attribute__((aligned(128)));
static float xbuf[8192 + 64] __attribute__((aligned(128)));
int main(void) {
    int fail = 0;
    for (uint32_t fmt = 0; fmt < 2; fmt++)
        for (uint32_t blocks = 1; blocks <= 240; blocks += 47) {
            for (uint32_t off = 0; off < 3; off++) {
                uint8_t *w = wbuf + off * 7; /* misaligned rows */
                const uint32_t pitch = fmt == 0 ? 18u : 34u, sa = fmt == 0 ? 16u : 32u;
                for (uint32_t b = 0; b < blocks; b++) {
                    for (uint32_t j = 0; j < sa; j++)
                        w[b * pitch + j] = (uint8_t)next();
                    const uint16_t h = to_half(((float)(next() % 1000) + 100.0f) * 1e-5f * ((next() & 1) ? -1.0f : 1.0f));
                    w[b * pitch + sa] = (uint8_t)h;
                    w[b * pitch + sa + 1] = (uint8_t)(h >> 8);
                }
                float *x = xbuf + off;
                for (uint32_t i = 0; i < blocks * 32; i++)
                    x[i] = ((float)(next() % 20001) - 10000.0f) * 1e-4f;
                const double want = ref_row(w, x, blocks, fmt);
                const float got = hx_linear_q_row_hvx(w, x, blocks, fmt);
                double mag = 0;
                for (uint32_t i = 0; i < blocks * 32; i++) mag += fabs(x[i]) * 0.01;
                const double err = fabs((double)got - want);
                if (err > 1e-5 * mag + 1e-6) {
                    printf("FAIL fmt %u blocks %u off %u got %.8g want %.8g\n", fmt, blocks, off, got, want);
                    fail = 1;
                }
            }
        }
    /* Four-row kernel: contiguous rows, misaligned starts. */
    for (uint32_t fmt = 0; fmt < 2; fmt++)
        for (uint32_t blocks = 1; blocks <= 240; blocks += 47)
            for (uint32_t off = 0; off < 3; off++) {
                const uint32_t pitch = fmt == 0 ? 18u : 34u, sa = fmt == 0 ? 16u : 32u, rb = blocks * pitch;
                uint8_t *w = wbuf + off * 7;
                if (4u * rb + off * 7 + 256u > sizeof wbuf) continue;
                for (uint32_t b = 0; b < 4u * blocks; b++) {
                    for (uint32_t j = 0; j < sa; j++)
                        w[b * pitch + j] = (uint8_t)next();
                    const uint16_t h = to_half(((float)(next() % 1000) + 100.0f) * 1e-5f * ((next() & 1) ? -1.0f : 1.0f));
                    w[b * pitch + sa] = (uint8_t)h;
                    w[b * pitch + sa + 1] = (uint8_t)(h >> 8);
                }
                float *x = xbuf + off;
                for (uint32_t i = 0; i < blocks * 32; i++)
                    x[i] = ((float)(next() % 20001) - 10000.0f) * 1e-4f;
                float got[4];
                hx_linear_q_rows4_hvx(w, rb, x, blocks, fmt, got);
                double mag = 0;
                for (uint32_t i = 0; i < blocks * 32; i++) mag += fabs(x[i]) * 0.01;
                for (uint32_t r = 0; r < 4; r++) {
                    const double want = ref_row(w + r * rb, x, blocks, fmt);
                    if (fabs((double)got[r] - want) > 1e-5 * mag + 1e-6) {
                        printf("FAIL rows4 fmt %u blocks %u off %u row %u got %.8g want %.8g\n", fmt, blocks, off, r, got[r], want);
                        fail = 1;
                    }
                }
            }
    /* Grouped layouts: repack 64 rows in place, read back every code and scale, then the HVX group kernel. */
    for (uint32_t fmt = 0; fmt < 2; fmt++)
        for (uint32_t blocks = 2; blocks <= 120; blocks += 118) {
            static uint8_t orig[64 * 120 * 34], grp_w[64 * 120 * 34 + 256] __attribute__((aligned(128)));
            static uint8_t scratch[32 * 120 * 34];
            const uint32_t pitch = fmt == 0 ? 18u : 34u, sa = fmt == 0 ? 16u : 32u, rb = blocks * pitch;
            for (uint32_t b = 0; b < 64u * blocks; b++) {
                for (uint32_t j = 0; j < sa; j++)
                    orig[b * pitch + j] = (uint8_t)next();
                const uint16_t h = to_half(((float)(next() % 1000) + 100.0f) * 1e-5f * ((next() & 1) ? -1.0f : 1.0f));
                orig[b * pitch + sa] = (uint8_t)h;
                orig[b * pitch + sa + 1] = (uint8_t)(h >> 8);
            }
            memcpy(grp_w, orig, 64u * rb);
            hx_lq_group_layout(grp_w, fmt, blocks * 32u, 64u, scratch);
            const uint32_t q4 = fmt == 0;
            uint32_t bad = 0;
            for (uint32_t r = 0; r < 64; r++)
                for (uint32_t bk = 0; bk < blocks; bk++) {
                    const uint8_t *src = orig + r * rb + bk * pitch, *gp = grp_w + (r / 32u) * 32u * rb;
                    const uint32_t at = hx_lq_group_scale_at(q4, r % 32u, bk);
                    if (gp[at] != src[sa] || gp[at + 1] != src[sa + 1]) bad++;
                    for (uint32_t kk = 0; kk < 32; kk++) {
                        const int want = q4 ? ((kk & 1) ? (src[kk / 2] >> 4) : (src[kk / 2] & 15)) - 8 : (int8_t)src[kk];
                        if (hx_lq_group_code(gp, q4, r % 32u, bk, kk) != want) bad++;
                    }
                }
            if (bad) { printf("FAIL grouped layout fmt %u blocks %u: %u mismatches\n", fmt, blocks, bad); fail = 1; }
            float *x = xbuf + 1;
            for (uint32_t i = 0; i < blocks * 32; i++)
                x[i] = ((float)(next() % 20001) - 10000.0f) * 1e-4f;
            x[5] = 0.0f;
            static int16_t xq[128 * 32 + 64] __attribute__((aligned(128)));
            static float sx[128];
            static int32_t corr[128];
            hx_lq_quantize_x(x, blocks, xq, sx, corr);
            double qerr = 0; /* in units of the block step sx */
            for (uint32_t i = 0; i < blocks * 32; i++) {
                const double e = sx[i / 32] > 0 ? fabs((double)xq[i] * sx[i / 32] - x[i]) / sx[i / 32] : fabs(x[i]);
                if (e > qerr) qerr = e;
            }
            if (qerr > 1.01) { printf("FAIL quantize err %g\n", qerr); fail = 1; }
            double mag = 0;
            for (uint32_t i = 0; i < blocks * 32; i++) mag += fabs(x[i]) * 0.01;
            for (uint32_t gi = 0; gi < 2; gi++) {
                float dots[32] __attribute__((aligned(128)));
                hx_lq_group_hvx(grp_w + gi * 32u * rb, xq, sx, corr, blocks, q4, dots);
                for (uint32_t i = 0; i < 32; i++) {
                    const uint8_t *row = orig + (gi * 32u + i) * rb;
                    const double want = ref_row(row, x, blocks, fmt);
                    double qb = 0; /* activation step bound: one unit of sx per element */
                    for (uint32_t bk = 0; bk < blocks; bk++) {
                        const uint8_t *src = row + bk * pitch;
                        const double sc = fabs(hx_half_to_float((uint16_t)(src[sa] | (src[sa + 1] << 8))));
                        for (uint32_t kk = 0; kk < 32; kk++) {
                            const int c = q4 ? ((kk & 1) ? (src[kk / 2] >> 4) : (src[kk / 2] & 15)) - 8 : (int8_t)src[kk];
                            qb += fabs((double)c) * sc * sx[bk];
                        }
                    }
                    if (fabs((double)dots[i] - want) > qb + 1e-5 * mag + 1e-6) {
                        printf("FAIL grouped fmt %u blocks %u row %u got %.8g want %.8g\n", fmt, blocks, gi * 32 + i, dots[i], want);
                        fail = 1;
                        break;
                    }
                }
            }
            if (blocks == 120) {
                float dots[32] __attribute__((aligned(128)));
                const uint64_t t0 = hx_cycles_now();
                for (uint32_t rep = 0; rep < 16; rep++)
                    hx_lq_group_hvx(grp_w + (rep & 1) * 32u * rb, xq, sx, corr, blocks, q4, dots);
                const uint64_t t1 = hx_cycles_now();
                printf("fmt %u grouped: %.2f cycles per row-block (%.1f per 32-row block)\n", fmt,
                       (double)(t1 - t0) / (16.0 * 32 * blocks), (double)(t1 - t0) / (16.0 * blocks));
            }
        }
    /* Vector SiLU against expf over [-100, 100]. */
    {
        float v[32] __attribute__((aligned(128))), r[32] __attribute__((aligned(128)));
        double worst = 0, worst_x = 0;
        for (int it = 0; it < 200; it++) {
            for (int i = 0; i < 32; i++) v[i] = (float)(((int)(next() % 200001) - 100000) * 1e-3) * (it < 100 ? 0.1f : 1.0f);
            *(HVX_Vector *)r = hx_vsilu(*(HVX_Vector *)v);
            for (int i = 0; i < 32; i++) {
                const double want = v[i] / (1.0 + exp(-(double)v[i]));
                const double e = fabs(r[i] - want) / (fabs(want) + 1e-6);
                if (e > worst) { worst = e; worst_x = v[i]; }
            }
        }
        printf("silu worst rel err %g at x %g\n", worst, worst_x);
        if (worst > 1e-5) { printf("FAIL silu\n"); fail = 1; }
    }
    /* SDPA row: masked (sliding window) and causal keys, unaligned rows. */
    {
        static float kb[256 * 72 + 64], vb[256 * 72 + 64], qr[64 + 32], mask[256], dst[64 + 32];
        for (uint32_t i = 0; i < sizeof kb / 4; i++) { kb[i] = ((float)(next() % 2001) - 1000.0f) * 1e-3f; vb[i] = ((float)(next() % 2001) - 1000.0f) * 1e-3f; }
        for (uint32_t i = 0; i < 64; i++) qr[i + 1] = ((float)(next() % 2001) - 1000.0f) * 1e-3f;
        for (uint32_t pos = 0; pos < 256; pos += 37)
            for (uint32_t win = 0; win < 2; win++) {
                for (uint32_t k = 0; k < 256; k++)
                    mask[k] = (k <= pos && (!win || k + 100u > pos)) ? (k == pos ? 0.5f : 0.0f) : -1e9f;
                const float sc = 0.125f;
                const uint32_t last = win ? 0xffffffffu : pos;
                if (!hx_sdpa_row_hvx(qr + 1, kb + 3, 72, vb + 5, 72, 256, 64, 64, sc, last, mask, 0, dst + 1)) {
                    printf("FAIL sdpa returned 0\n"); fail = 1; continue;
                }
                double sw[256], peak = -1e300, tot = 0;
                for (uint32_t k = 0; k < 256; k++) {
                    double d = 0;
                    for (uint32_t j = 0; j < 64; j++) d += (double)qr[1 + j] * kb[3 + k * 72 + j];
                    sw[k] = (last != 0xffffffffu && k > last) ? -1e300 : d * sc + mask[k];
                    if (sw[k] > peak) peak = sw[k];
                }
                for (uint32_t k = 0; k < 256; k++) { sw[k] = sw[k] < -1e299 ? 0 : exp(sw[k] - peak); tot += sw[k]; }
                double worst = 0;
                for (uint32_t j = 0; j < 64; j++) {
                    double o = 0;
                    for (uint32_t k = 0; k < 256; k++) o += sw[k] / tot * vb[5 + k * 72 + j];
                    if (fabs(o - dst[1 + j]) > worst) worst = fabs(o - dst[1 + j]);
                }
                if (worst > 2e-5) { printf("FAIL sdpa pos %u win %u err %g\n", pos, win, worst); fail = 1; }
            }
        for (uint32_t k = 0; k < 256; k++) mask[k] = -1e9f;
        if (hx_sdpa_row_hvx(qr + 1, kb, 72, vb, 72, 256, 64, 64, 0.125f, 0xffffffffu, mask, 0, dst)) {
            printf("FAIL sdpa all-masked row must fall back\n"); fail = 1;
        }
    }
    /* Cycle count for one 2048 x 512 Q8 slice and Q4 slice. */
    for (uint32_t fmt = 0; fmt < 2; fmt++) {
        const uint32_t blocks = 64, pitch = fmt == 0 ? 18u : 34u;
        static uint8_t big[512 * 64 * 34 + 256];
        for (uint32_t i = 0; i < sizeof big; i++) big[i] = (uint8_t)next() & 0x3b; /* small scales */
        const uint64_t t0 = hx_cycles_now();
        float s = 0;
        for (uint32_t r = 0; r < 512; r++)
            s += hx_linear_q_row_hvx(big + r * blocks * pitch, xbuf, blocks, fmt);
        const uint64_t t1 = hx_cycles_now();
        printf("fmt %u: %.2f cycles per block (sum %g)\n", fmt, (double)(t1 - t0) / (512.0 * blocks), s);
        float s4 = 0, o4[4];
        const uint64_t t2 = hx_cycles_now();
        for (uint32_t r = 0; r < 512; r += 4) {
            hx_linear_q_rows4_hvx(big + r * blocks * pitch, blocks * pitch, xbuf, blocks, fmt, o4);
            s4 += o4[0] + o4[1] + o4[2] + o4[3];
        }
        const uint64_t t3 = hx_cycles_now();
        printf("fmt %u rows4: %.2f cycles per block (sum %g)\n", fmt, (double)(t3 - t2) / (512.0 * blocks), s4);
    }
    printf(fail ? "FAIL linear_q hvx\n" : "PASS linear_q hvx\n");
    return fail;
}
