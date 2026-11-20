/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* DSP unit test: llm/qwen_dsp.c's HVX per-element phases against their scalar
 * references, on synthetic data in L2-TCM (worker 0 only). Results at
 * arena+0x10000; decode them with bench/run_qwen_ops_test.py. */
#define QWEN_OPS_TEST
#include "../examples/llm/qwen_dsp.c"

#define T_X     (L2 + 0x00000u)   /* 2 x FFN floats (gate, up) */
#define T_W     (L2 + 0x06000u)   /* DIM floats */
#define T_REF   (L2 + 0x07000u)   /* FFN floats */
#define T_HVX   (L2 + 0x0a000u)   /* FFN floats */
#define T_QREF  (L2 + 0x10000u)   /* 2 QRows */
#define T_QHVX  (L2 + 0x12000u)   /* 2 QRows */
#define T_ROW   (L2 + 0x20000u)   /* argmax row, 64 vectors */

static uint32_t rng = 0x12345678u;
static float frand(float lo, float hi) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return lo + (hi - lo) * (float)(rng >> 8) * (1.0f / 16777216.0f);
}
static float fabsf_(float v) { return v < 0.0f ? -v : v; }
static uint32_t fbits(float v) { union { float f; uint32_t u; } b; b.f = v; return b.u; }

/* Relative error against max(|ref|, floor), and the count of inexact floats. */
static void compare(const float *ref, const float *got, uint32_t n, float floor,
                    volatile uint32_t *out) {
    float worst = 0.0f;
    uint32_t inexact = 0;
    for (uint32_t i = 0; i < n; i++) {
        float m = fabsf_(ref[i]) > floor ? fabsf_(ref[i]) : floor;
        float e = fabsf_(ref[i] - got[i]) / m;
        if (e > worst) worst = e;
        inexact += fbits(ref[i]) != fbits(got[i]);
    }
    out[0] = inexact; out[1] = fbits(worst);
}

static void test_rmsnorm(volatile uint32_t *out) {
    float *x = (float *)T_X, *w = (float *)T_W, *r = (float *)T_REF, *h = (float *)T_HVX;
    for (uint32_t i = 0; i < DIM; i++) { x[i] = frand(-4.0f, 4.0f); w[i] = frand(0.5f, 1.5f); }
    x[7] = 60.0f;  /* an outlier channel, as in real residual streams */
    rmsnorm(r, x, w, DIM, 1u);
    rmsnorm_hvx(h, x, w, DIM);
    compare(r, h, DIM, 1e-3f, out);
}

static void test_add(volatile uint32_t *out) {
    float *x = (float *)T_X, *r = (float *)T_REF, *h = (float *)T_HVX;
    for (uint32_t i = 0; i < DIM; i++) { r[i] = h[i] = frand(-8.0f, 8.0f); x[i] = frand(-8.0f, 8.0f); }
    for (uint32_t i = 0; i < DIM; i++) r[i] += x[i];
    add_hvx(h, x, DIM);
    compare(r, h, DIM, 1e-6f, out);
}

/* Rows 0 (aligned) and 1 (QRow is not a multiple of 128 bytes). */
static void test_quantize(volatile uint32_t *out) {
    float *x = (float *)T_X;
    for (uint32_t b = 0; b < 2u * FFN / 32u; b++) {
        float scale = b % 7u == 0 ? 0.0f : frand(0.001f, 20.0f);  /* includes all-zero blocks */
        for (uint32_t i = 0; i < 32u; i++) x[b * 32u + i] = scale * frand(-1.0f, 1.0f);
        if (b % 5u == 1u) x[b * 32u + 3u] = -scale;                 /* exact extremes */
    }
    QRow *qr = (QRow *)T_QREF, *qh = (QRow *)T_QHVX;
    for (uint32_t i = 0; i < 2u * sizeof(QRow) / 4u; i++) ((uint32_t *)qr)[i] = ((uint32_t *)qh)[i] = 0xa5a5a5a5u;
    for (uint32_t t = 0; t < 2u; t++) quantize(qr, x, FFN, FFN, t, t + 1u);
    for (uint32_t tid = 0; tid < 3u; tid++) quantize_split(qh, x, FFN, FFN, 2u, tid);
    uint32_t q = 0, s8 = 0, dxp = 0, first = 0xffffffffu;
    for (uint32_t t = 0; t < 2u; t++) {
        for (uint32_t i = 0; i < FFN; i++)
            if (qr[t].q[i] != qh[t].q[i]) { q++; if (first == 0xffffffffu) first = t * FFN + i; }
        for (uint32_t i = 0; i < FFN / 32u; i++) s8 += qr[t].s8[i] != qh[t].s8[i];
        for (uint32_t i = 0; i < FFN / 64u; i++) dxp += qr[t].dxp[i] != qh[t].dxp[i];
    }
    out[0] = q; out[1] = s8; out[2] = dxp; out[3] = first;
    if (first != 0xffffffffu) {
        uint32_t t = first / FFN, i = first % FFN;
        out[4] = (uint32_t)(int32_t)qr[t].q[i]; out[5] = (uint32_t)(int32_t)qh[t].q[i];
        out[6] = fbits(x[t * FFN + i]);
    }
}

static void test_silu(volatile uint32_t *out) {
    float *gu = (float *)T_X, *r = (float *)T_REF, *h = (float *)T_HVX;
    for (uint32_t i = 0; i < FFN; i++) { gu[i] = frand(-12.0f, 12.0f); gu[FFN + i] = frand(-3.0f, 3.0f); }
    gu[1] = 95.0f; gu[2] = -95.0f; gu[3] = 0.0f; gu[4] = 1e-6f;
    for (uint32_t i = 0; i < FFN; i++) {
        float g = gu[i];
        r[i] = g / (1.0f + exp_approx(-g)) * gu[FFN + i];
    }
    silu_hvx(h, gu, 0, FFN / 32u);
    compare(r, h, FFN, 1e-3f, out);
}

static void test_rope(volatile uint32_t *out) {
    float *head = (float *)T_X, *w = (float *)T_W, *r = (float *)T_REF, *h = (float *)T_HVX;
    float *cs = (float *)(T_X + 0x1000u);
    for (uint32_t i = 0; i < 128u; i++) { head[i] = frand(-3.0f, 3.0f); w[i] = frand(0.5f, 2.0f); }
    for (uint32_t i = 0; i < 64u; i++) {
        float c = frand(-1.0f, 1.0f), s = frand(-1.0f, 1.0f);
        cs[2u * i] = c; cs[2u * i + 1u] = s;
    }
    for (uint32_t i = 0; i < 128u; i++) h[i] = head[i];
    rmsnorm(r, head, w, 128u, 1u);
    for (uint32_t i = 0; i < 64u; i++) {
        float a = r[i], b = r[64u + i], c = cs[2u * i], s = cs[2u * i + 1u];
        r[i] = a * c - b * s;
        r[64u + i] = a * s + b * c;
    }
    rope_head_hvx(h, w, cs);
    compare(r, h, 128u, 1e-3f, out);
}

static void test_softmax(volatile uint32_t *out) {
    float *r = (float *)T_REF, *h = (float *)T_HVX;
    uint32_t len = 77;  /* a partial last vector */
    for (uint32_t i = 0; i < 96u; i++) r[i] = h[i] = frand(-40.0f, 40.0f);
    float m = -1.0e30f, d = 0.0f;
    for (uint32_t i = 0; i < len; i++) { r[i] *= 0.08838834764831845f; if (r[i] > m) m = r[i]; }
    for (uint32_t i = 0; i < len; i++) { r[i] = exp_approx(r[i] - m); d += r[i]; }
    float ir = 1.0f / d, ih = softmax_hvx(h, len);
    for (uint32_t i = 0; i < len; i++) { r[i] *= ir; h[i] *= ih; }
    compare(r, h, len, 1e-3f, out);
    uint32_t tail = 0;
    for (uint32_t i = len; i < 96u; i++) tail += fbits(h[i]) != 0;
    out[2] = tail;
}

/* Patterns: random, all negative, ties (first index must win), max in last lane. */
static void test_argmax(volatile uint32_t *out) {
    float *row = (float *)T_ROW;
    uint32_t bad = 0;
    for (uint32_t pattern = 0; pattern < 4u; pattern++) {
        for (uint32_t i = 0; i < 64u * 32u; i++)
            row[i] = pattern == 1u ? frand(-50.0f, -1.0f) : frand(-20.0f, 20.0f);
        if (pattern == 2u) { row[700] = 99.0f; row[701] = 99.0f; row[1500] = 99.0f; }
        if (pattern == 3u) row[64u * 32u - 1u] = 99.0f;
        for (uint32_t range = 0; range < 2u; range++) {
            uint32_t a = range ? 5u : 0u, b = range ? 50u : 64u;
            float best = -1.0e30f, got;
            uint32_t arg = 0;
            for (uint32_t i = a * 32u; i < b * 32u; i++)
                if (row[i] > best) { best = row[i]; arg = i; }
            uint32_t hv = argmax_hvx(row, a, b, &got);
            bad += hv != arg || fbits(got) != fbits(best);
        }
    }
    out[0] = bad;
}

/* Cycles for the HVX and scalar versions at decode sizes. */
static void timing(volatile uint32_t *out) {
    float *x = (float *)T_X, *w = (float *)T_W, *h = (float *)T_HVX;
    QRow *q = (QRow *)T_QHVX;
    uint64_t t0 = cycles(); quantize(q, x, FFN, FFN, 0, 1u);
    uint64_t t1 = cycles(); quantize_split(q, x, FFN, FFN, 1u, 0); /* one worker's third */
    uint64_t t2 = cycles(); rmsnorm(h, x, w, DIM, 1u);
    uint64_t t3 = cycles(); rmsnorm_hvx(h, x, w, DIM);
    uint64_t t4 = cycles();
    for (uint32_t i = 0; i < FFN; i++) h[i] = x[i] / (1.0f + exp_approx(-x[i])) * x[FFN + i];
    uint64_t t5 = cycles(); silu_hvx(h, x, 0, FFN / 32u);
    uint64_t t6 = cycles();
    out[0] = (uint32_t)(t1 - t0); out[1] = (uint32_t)(t2 - t1);
    out[2] = (uint32_t)(t3 - t2); out[3] = (uint32_t)(t4 - t3);
    out[4] = (uint32_t)(t5 - t4); out[5] = (uint32_t)(t6 - t5);
}

/* attention()'s Q.K inner loop over one 64-position block (128 K^T vectors). */
static __attribute__((noinline)) HVX_Vector qk_block(const HVX_Vector *col, const uint16_t *q0, const uint16_t *q1) {
    HVX_Vector e0 = Q6_V_vzero(), o0 = e0, e1 = e0, o1 = e0;
    for (uint32_t d = 0; d < 128u; d++) {
        HVX_Vector k = col[d];
        HVX_VectorPair p0 = Q6_Wqf32_vmpy_VhfVhf(k, Q6_Vh_vsplat_R(q0[d]));
        HVX_VectorPair p1 = Q6_Wqf32_vmpy_VhfVhf(k, Q6_Vh_vsplat_R(q1[d]));
        e0 = Q6_Vqf32_vadd_Vqf32Vqf32(e0, Q6_V_lo_W(p0));
        o0 = Q6_Vqf32_vadd_Vqf32Vqf32(o0, Q6_V_hi_W(p0));
        e1 = Q6_Vqf32_vadd_Vqf32Vqf32(e1, Q6_V_lo_W(p1));
        o1 = Q6_Vqf32_vadd_Vqf32Vqf32(o1, Q6_V_hi_W(p1));
    }
    return Q6_V_vxor_VV(Q6_V_vxor_VV(e0, o0), Q6_V_vxor_VV(e1, o1));
}

/* Cycles for 32 blocks (512 KB of K^T) from DDR (cdsp_mem), L2-TCM and VTCM,
 * and for staging one block from DDR into L2 with copy128. */
static void qk_location(volatile uint32_t *out) {
    const uint16_t *q = (const uint16_t *)(L2 + 0x3f000u);
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t srcs[3] = {BIG + 0x1000000u, L2 + 0x40000u, VTCM + 0x600000u};
    if (install_tlb()) { out[0] = 0xbad; return; }
    for (uint32_t w = 0; w < 3u; w++) {
        uint32_t blocks = w == 1u ? 8u : 32u;   /* L2 holds 128 KB of it */
        uint64_t t0 = cycles();
        for (uint32_t b = 0; b < blocks; b++)
            acc = Q6_V_vxor_VV(acc, qk_block((const HVX_Vector *)(srcs[w] + b * 16384u), q, q + 128u));
        out[w] = (uint32_t)((cycles() - t0) / blocks);
    }
    uint64_t t0 = cycles();
    for (uint32_t b = 0; b < 8u; b++) copy128((void *)(L2 + 0x40000u), (const void *)(BIG + 0x1000000u + b * 16384u), 16384u);
    out[3] = (uint32_t)((cycles() - t0) / 8u);
    remove_tlb();
    out[4] = Q6_R_vextract_VR(acc, 0);
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    uint32_t seq = *(volatile uint32_t *)(ARENA + 8);
    if (tid == 0) {
        volatile uint32_t *out = (volatile uint32_t *)(ARENA + 0x10000u);
        out[0] = 1;
        test_rmsnorm(out + 8);
        test_add(out + 16);
        test_quantize(out + 24);
        test_silu(out + 32);
        test_rope(out + 40);
        test_argmax(out + 48);
        test_softmax(out + 52);
        /* Time at the AXI-master clock the Qwen payload runs at; at XO every
         * instruction-cache miss to DDR dominates. */
        out[7] = axim_raise();
        timing(out + 56);
        qk_location(out + 80);
        axim_restore();
        out[0] = 0xffffffffu;
    }
    fence();
    *(volatile uint32_t *)(ARENA + 128 + tid * 16 + 12) = 0x54495845u;
    *(volatile uint32_t *)(ARENA + 128 + tid * 16) = seq;
    fence();
}
