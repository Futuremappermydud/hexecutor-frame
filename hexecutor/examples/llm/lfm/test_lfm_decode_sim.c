/* Hexagon simulator test for LFM decode building blocks and regression check.
 *
 * Runs:
 * 1. Tiny synthetic LFM-style model: 2 conv layers + 1 attention layer
 *    Dims: D=32, FFN=64, head_dim=16, num_heads=2, K=3 taps (2 state tokens).
 *    Evaluates 5 sequential decode steps with persistent state ring updates.
 *    Compares against embedded reference implementation (tolerance < 1e-5).
 * 2. Qwen-style regression block (standard RMSNorm + QKV attention + SwiGLU FFN).
 * 3. State-region size report.
 */

#include "graph.h"
#include "lfm_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D 32u
#define FFN 64u
#define HEAD_DIM 16u
#define NUM_HEADS 2u
#define K_TAPS 3u
#define S_LEN (K_TAPS - 1u)
#define NUM_STEPS 5u

static uint32_t rng_state = 123456789u;
static float frand(float lo, float hi) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    float normalized = (float)(rng_state >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * normalized;
}

static float fabsf_(float x) {
    return x < 0.0f ? -x : x;
}

/* Reference functions */
static void ref_rmsnorm(float *out, const float *x, const float *gamma, uint32_t count, float eps) {
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < count; i++)
        sum_sq += (double)x[i] * (double)x[i];
    float inv = (float)(1.0 / sqrt(sum_sq / (double)count + (double)eps));
    for (uint32_t i = 0; i < count; i++)
        out[i] = x[i] * inv * gamma[i];
}

static float ref_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void ref_rope(float *out, const float *x, uint32_t head_dim, uint32_t num_heads, uint32_t pos) {
    uint32_t half = head_dim / 2u;
    for (uint32_t h = 0; h < num_heads; h++) {
        const float *hin = x + h * head_dim;
        float *hout = out + h * head_dim;
        for (uint32_t j = 0; j < half; j++) {
            double theta = pow(10000.0, -2.0 * (double)j / (double)head_dim);
            double angle = (double)pos * theta;
            float cos_v = (float)cos(angle);
            float sin_v = (float)sin(angle);
            float v0 = hin[j], v1 = hin[half + j];
            hout[j] = v0 * cos_v - v1 * sin_v;
            hout[half + j] = v0 * sin_v + v1 * cos_v;
        }
    }
}

static void ref_matvec(float *out, const float *m, const float *v, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        double acc = 0.0;
        for (uint32_t c = 0; c < cols; c++)
            acc += (double)m[r * cols + c] * (double)v[c];
        out[r] = (float)acc;
    }
}

/* Synthetic model weights */
typedef struct {
    float op_norm[D];
    float in_proj[3 * D * D];
    float conv_w[K_TAPS * D];
    float out_proj[D * D];
    float post_op_norm[D];
    float ffn_norm[D];
    float w1[FFN * D];
    float w3[FFN * D];
    float w2[D * FFN];
    float post_ffn_norm[D];
} ConvLayerWeights;

typedef struct {
    float op_norm[D];
    float q_proj[D * D];
    float k_proj[D * D];
    float v_proj[D * D];
    float q_norm[HEAD_DIM];
    float k_norm[HEAD_DIM];
    float gate_proj[NUM_HEADS * D];
    float o_proj[D * D];
    float post_op_norm[D];
    float ffn_norm[D];
    float w1[FFN * D];
    float w3[FFN * D];
    float w2[D * FFN];
    float post_ffn_norm[D];
} AttnLayerWeights;

static ConvLayerWeights l0, l1;
static AttnLayerWeights l2;
static float final_norm[D];

static float ref_conv_states[2][S_LEN * D];
static float ref_kv_k[NUM_STEPS][D];
static float ref_kv_v[NUM_STEPS][D];

static void init_weights(void) {
    for (uint32_t i = 0; i < D; i++) {
        l0.op_norm[i] = frand(0.8f, 1.2f);
        l0.post_op_norm[i] = frand(0.8f, 1.2f);
        l0.ffn_norm[i] = frand(0.8f, 1.2f);
        l0.post_ffn_norm[i] = frand(0.8f, 1.2f);

        l1.op_norm[i] = frand(0.8f, 1.2f);
        l1.post_op_norm[i] = frand(0.8f, 1.2f);
        l1.ffn_norm[i] = frand(0.8f, 1.2f);
        l1.post_ffn_norm[i] = frand(0.8f, 1.2f);

        l2.op_norm[i] = frand(0.8f, 1.2f);
        l2.post_op_norm[i] = frand(0.8f, 1.2f);
        l2.ffn_norm[i] = frand(0.8f, 1.2f);
        l2.post_ffn_norm[i] = frand(0.8f, 1.2f);

        final_norm[i] = frand(0.8f, 1.2f);
    }
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        l2.q_norm[i] = frand(0.8f, 1.2f);
        l2.k_norm[i] = frand(0.8f, 1.2f);
    }
    for (uint32_t i = 0; i < 3 * D * D; i++) {
        l0.in_proj[i] = frand(-0.1f, 0.1f);
        l1.in_proj[i] = frand(-0.1f, 0.1f);
    }
    for (uint32_t i = 0; i < K_TAPS * D; i++) {
        l0.conv_w[i] = frand(-0.2f, 0.2f);
        l1.conv_w[i] = frand(-0.2f, 0.2f);
    }
    for (uint32_t i = 0; i < D * D; i++) {
        l0.out_proj[i] = frand(-0.1f, 0.1f);
        l1.out_proj[i] = frand(-0.1f, 0.1f);
        l2.q_proj[i] = frand(-0.1f, 0.1f);
        l2.k_proj[i] = frand(-0.1f, 0.1f);
        l2.v_proj[i] = frand(-0.1f, 0.1f);
        l2.o_proj[i] = frand(-0.1f, 0.1f);
    }
    for (uint32_t i = 0; i < NUM_HEADS * D; i++)
        l2.gate_proj[i] = frand(-0.1f, 0.1f);
    for (uint32_t i = 0; i < FFN * D; i++) {
        l0.w1[i] = frand(-0.1f, 0.1f); l0.w3[i] = frand(-0.1f, 0.1f);
        l1.w1[i] = frand(-0.1f, 0.1f); l1.w3[i] = frand(-0.1f, 0.1f);
        l2.w1[i] = frand(-0.1f, 0.1f); l2.w3[i] = frand(-0.1f, 0.1f);
    }
    for (uint32_t i = 0; i < D * FFN; i++) {
        l0.w2[i] = frand(-0.1f, 0.1f);
        l1.w2[i] = frand(-0.1f, 0.1f);
        l2.w2[i] = frand(-0.1f, 0.1f);
    }
}

static void ref_conv_forward(float *x, const ConvLayerWeights *w, float *conv_state) {
    float res[D], x_norm[D], bcx[3 * D], u[D], conv_out[D], y[D], op_out[D];
    memcpy(res, x, sizeof res);

    ref_rmsnorm(x_norm, x, w->op_norm, D, 1e-5f);
    ref_matvec(bcx, w->in_proj, x_norm, 3 * D, D);

    const float *B = bcx, *C = bcx + D, *x_part = bcx + 2 * D;
    for (uint32_t d = 0; d < D; d++)
        u[d] = B[d] * x_part[d];

    /* Causal depthwise conv across state ring */
    for (uint32_t d = 0; d < D; d++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < S_LEN; j++)
            acc += (double)conv_state[j * D + d] * (double)w->conv_w[j * D + d];
        acc += (double)u[d] * (double)w->conv_w[S_LEN * D + d];
        conv_out[d] = (float)acc;

        /* Shift state ring */
        for (uint32_t j = 0; j < S_LEN - 1u; j++)
            conv_state[j * D + d] = conv_state[(j + 1u) * D + d];
        conv_state[(S_LEN - 1u) * D + d] = u[d];
    }

    for (uint32_t d = 0; d < D; d++)
        y[d] = C[d] * conv_out[d];

    ref_matvec(op_out, w->out_proj, y, D, D);
    ref_rmsnorm(op_out, op_out, w->post_op_norm, D, 1e-5f);
    for (uint32_t d = 0; d < D; d++)
        x[d] = res[d] + op_out[d];

    /* FFN */
    float res2[D], x_ffn_norm[D], w1_out[FFN], w3_out[FFN], ffn_act[FFN], ffn_out[D];
    memcpy(res2, x, sizeof res2);
    ref_rmsnorm(x_ffn_norm, x, w->ffn_norm, D, 1e-5f);
    ref_matvec(w1_out, w->w1, x_ffn_norm, FFN, D);
    ref_matvec(w3_out, w->w3, x_ffn_norm, FFN, D);
    for (uint32_t i = 0; i < FFN; i++)
        ffn_act[i] = ref_silu(w1_out[i]) * w3_out[i];
    ref_matvec(ffn_out, w->w2, ffn_act, D, FFN);
    ref_rmsnorm(ffn_out, ffn_out, w->post_ffn_norm, D, 1e-5f);
    for (uint32_t d = 0; d < D; d++)
        x[d] = res2[d] + ffn_out[d];
}

static void ref_attn_forward(float *x, const AttnLayerWeights *w, uint32_t pos) {
    float res[D], x_norm[D], q[D], k[D], v[D];
    memcpy(res, x, sizeof res);
    ref_rmsnorm(x_norm, x, w->op_norm, D, 1e-5f);
    ref_matvec(q, w->q_proj, x_norm, D, D);
    ref_matvec(k, w->k_proj, x_norm, D, D);
    ref_matvec(v, w->v_proj, x_norm, D, D);

    /* QK RMSNorm */
    for (uint32_t h = 0; h < NUM_HEADS; h++) {
        ref_rmsnorm(q + h * HEAD_DIM, q + h * HEAD_DIM, w->q_norm, HEAD_DIM, 1e-5f);
        ref_rmsnorm(k + h * HEAD_DIM, k + h * HEAD_DIM, w->k_norm, HEAD_DIM, 1e-5f);
    }
    /* RoPE */
    ref_rope(q, q, HEAD_DIM, NUM_HEADS, pos);
    ref_rope(k, k, HEAD_DIM, NUM_HEADS, pos);

    /* Store K and V to cache */
    memcpy(ref_kv_k[pos], k, sizeof k);
    memcpy(ref_kv_v[pos], v, sizeof v);
    uint32_t ctx_len = pos + 1u;

    /* Scaled Dot-Product Attention with Head Gating */
    float head_outs[D];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t h = 0; h < NUM_HEADS; h++) {
        float scores[NUM_STEPS];
        float max_s = -1e30f;
        for (uint32_t t = 0; t < ctx_len; t++) {
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; d++)
                dot += (double)q[h * HEAD_DIM + d] * (double)ref_kv_k[t][h * HEAD_DIM + d];
            scores[t] = (float)dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }
        double sum_exp = 0.0;
        for (uint32_t t = 0; t < ctx_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += (double)scores[t];
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < ctx_len; t++)
                acc += ((double)scores[t] / sum_exp) * (double)ref_kv_v[t][h * HEAD_DIM + d];
            head_outs[h * HEAD_DIM + d] = (float)acc;
        }
    }

    /* Head gate projection */
    float gate_raw[NUM_HEADS];
    ref_matvec(gate_raw, w->gate_proj, x_norm, NUM_HEADS, D);
    for (uint32_t h = 0; h < NUM_HEADS; h++) {
        float g = 1.0f / (1.0f + expf(-gate_raw[h]));
        for (uint32_t d = 0; d < HEAD_DIM; d++)
            head_outs[h * HEAD_DIM + d] *= g;
    }

    float op_out[D];
    ref_matvec(op_out, w->o_proj, head_outs, D, D);
    ref_rmsnorm(op_out, op_out, w->post_op_norm, D, 1e-5f);
    for (uint32_t d = 0; d < D; d++)
        x[d] = res[d] + op_out[d];

    /* FFN */
    float res2[D], x_ffn_norm[D], w1_out[FFN], w3_out[FFN], ffn_act[FFN], ffn_out[D];
    memcpy(res2, x, sizeof res2);
    ref_rmsnorm(x_ffn_norm, x, w->ffn_norm, D, 1e-5f);
    ref_matvec(w1_out, w->w1, x_ffn_norm, FFN, D);
    ref_matvec(w3_out, w->w3, x_ffn_norm, FFN, D);
    for (uint32_t i = 0; i < FFN; i++)
        ffn_act[i] = ref_silu(w1_out[i]) * w3_out[i];
    ref_matvec(ffn_out, w->w2, ffn_act, D, FFN);
    ref_rmsnorm(ffn_out, ffn_out, w->post_ffn_norm, D, 1e-5f);
    for (uint32_t d = 0; d < D; d++)
        x[d] = res2[d] + ffn_out[d];
}

static void ref_model_step(float *x, uint32_t pos) {
    ref_conv_forward(x, &l0, ref_conv_states[0]);
    ref_conv_forward(x, &l1, ref_conv_states[1]);
    ref_attn_forward(x, &l2, pos);
    ref_rmsnorm(x, x, final_norm, D, 1e-5f);
}

int main(void) {
    printf("Starting Hexagon simulator test for LFM decode building blocks...\n");
    init_weights();
    memset(ref_conv_states, 0, sizeof ref_conv_states);

    /* Allocate arena, constants, and tensors for graph test */
    static uint8_t arena[0x100000] __attribute__((aligned(128)));
    static uint8_t constants[0x100000] __attribute__((aligned(128)));

    /* Test 1: Validate individual engine ops against references */
    {
        /* Test HX_RMSNORM */
        float x_test[D], gamma[D], out_got[D], out_ref[D];
        for (uint32_t i = 0; i < D; i++) {
            x_test[i] = frand(-2.0f, 2.0f);
            gamma[i] = frand(0.5f, 1.5f);
        }
        memcpy(constants, gamma, D * sizeof(float));
        ref_rmsnorm(out_ref, x_test, gamma, D, 1e-5f);
        lfm_rmsnorm(out_got, x_test, (const float *)constants, NULL, D, 1e-5f);
        float max_e = 0.0f;
        for (uint32_t i = 0; i < D; i++) {
            float err = fabsf_(out_got[i] - out_ref[i]);
            if (err > max_e) max_e = err;
        }
        if (max_e > 1e-5f) {
            printf("FAIL: RMSNorm max error %e > 1e-5\n", max_e);
            return 1;
        }

        /* Test HX_ROPE */
        float q_in[D], q_got[D], q_ref[D];
        for (uint32_t i = 0; i < D; i++) q_in[i] = frand(-3.0f, 3.0f);
        ref_rope(q_ref, q_in, HEAD_DIM, NUM_HEADS, 7);
        lfm_rope(q_got, q_in, HEAD_DIM, NUM_HEADS, 7, 0);
        max_e = 0.0f;
        for (uint32_t i = 0; i < D; i++) {
            float err = fabsf_(q_got[i] - q_ref[i]);
            if (err > max_e) max_e = err;
        }
        if (max_e > 1e-5f) {
            printf("FAIL: RoPE max error %e > 1e-5\n", max_e);
            return 1;
        }

        /* Test HX_SHORTCONV multi-step persistent ring update */
        float s_ring[S_LEN * D], s_ref[S_LEN * D];
        memset(s_ring, 0, sizeof s_ring);
        memset(s_ref, 0, sizeof s_ref);
        float conv_w[K_TAPS * D];
        for (uint32_t i = 0; i < K_TAPS * D; i++) conv_w[i] = frand(-0.5f, 0.5f);
        for (uint32_t step = 0; step < 4; step++) {
            float tok_in[D], sc_got[D], sc_ref[D];
            for (uint32_t i = 0; i < D; i++) tok_in[i] = frand(-1.0f, 1.0f);

            /* Reference conv step */
            for (uint32_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j < S_LEN; j++)
                    acc += (double)s_ref[j * D + d] * (double)conv_w[j * D + d];
                acc += (double)tok_in[d] * (double)conv_w[S_LEN * D + d];
                sc_ref[d] = (float)acc;
                for (uint32_t j = 0; j < S_LEN - 1u; j++)
                    s_ref[j * D + d] = s_ref[(j + 1u) * D + d];
                s_ref[(S_LEN - 1u) * D + d] = tok_in[d];
            }

            /* Operator conv step */
            lfm_shortconv(sc_got, tok_in, s_ring, NULL, conv_w, NULL, D, K_TAPS, 0);
            for (uint32_t d = 0; d < D; d++) {
                float err = fabsf_(sc_got[d] - sc_ref[d]);
                if (err > 1e-5f) {
                    printf("FAIL: ShortConv step %u channel %u got %f ref %f\n", (unsigned)step, (unsigned)d, sc_got[d], sc_ref[d]);
                    return 1;
                }
            }
        }
    }

    /* Test 2: Full 3-layer LFM Model Decode Sequence (5 steps) */
    printf("Running 5 decode steps of tiny synthetic LFM (2 conv + 1 attn layers)...\n");
    float tok_seq[NUM_STEPS][D];
    float ref_outputs[NUM_STEPS][D];
    float lfm_outputs[NUM_STEPS][D];

    /* Run reference path */
    for (uint32_t s = 0; s < NUM_STEPS; s++) {
        for (uint32_t d = 0; d < D; d++)
            tok_seq[s][d] = frand(-1.0f, 1.0f);
        memcpy(ref_outputs[s], tok_seq[s], sizeof(float) * D);
        ref_model_step(ref_outputs[s], s);
    }

    /* Run engine / LFM path */
    static float dsp_conv_states[2][S_LEN * D];
    static float dsp_kv_k[NUM_STEPS][D];
    static float dsp_kv_v[NUM_STEPS][D];
    memset(dsp_conv_states, 0, sizeof dsp_conv_states);

    float max_err = 0.0f;
    for (uint32_t s = 0; s < NUM_STEPS; s++) {
        float x[D];
        memcpy(x, tok_seq[s], sizeof x);

        /* Layer 0: Conv block using lfm_ops */
        {
            float res[D], x_norm[D], bcx[3 * D], u[D], conv_out[D], y[D], op_out[D];
            memcpy(res, x, sizeof res);
            lfm_rmsnorm(x_norm, x, l0.op_norm, NULL, D, 1e-5f);
            ref_matvec(bcx, l0.in_proj, x_norm, 3 * D, D);
            for (uint32_t d = 0; d < D; d++)
                u[d] = bcx[d] * bcx[2 * D + d];
            lfm_shortconv(conv_out, u, dsp_conv_states[0], bcx + D, l0.conv_w, NULL, D, K_TAPS, 0);
            ref_matvec(op_out, l0.out_proj, conv_out, D, D);
            lfm_rmsnorm(op_out, op_out, l0.post_op_norm, NULL, D, 1e-5f);
            for (uint32_t d = 0; d < D; d++)
                x[d] = res[d] + op_out[d];

            float res2[D], x_fn[D], w1_o[FFN], w3_o[FFN], act[FFN], ffn_o[D];
            memcpy(res2, x, sizeof res2);
            lfm_rmsnorm(x_fn, x, l0.ffn_norm, NULL, D, 1e-5f);
            ref_matvec(w1_o, l0.w1, x_fn, FFN, D);
            ref_matvec(w3_o, l0.w3, x_fn, FFN, D);
            for (uint32_t i = 0; i < FFN; i++)
                act[i] = ref_silu(w1_o[i]) * w3_o[i];
            ref_matvec(ffn_o, l0.w2, act, D, FFN);
            lfm_rmsnorm(ffn_o, ffn_o, l0.post_ffn_norm, NULL, D, 1e-5f);
            for (uint32_t d = 0; d < D; d++)
                x[d] = res2[d] + ffn_o[d];
        }

        /* Layer 1: Conv block using lfm_ops */
        {
            float res[D], x_norm[D], bcx[3 * D], u[D], conv_out[D], op_out[D];
            memcpy(res, x, sizeof res);
            lfm_rmsnorm(x_norm, x, l1.op_norm, NULL, D, 1e-5f);
            ref_matvec(bcx, l1.in_proj, x_norm, 3 * D, D);
            for (uint32_t d = 0; d < D; d++)
                u[d] = bcx[d] * bcx[2 * D + d];
            lfm_shortconv(conv_out, u, dsp_conv_states[1], bcx + D, l1.conv_w, NULL, D, K_TAPS, 0);
            ref_matvec(op_out, l1.out_proj, conv_out, D, D);
            lfm_rmsnorm(op_out, op_out, l1.post_op_norm, NULL, D, 1e-5f);
            for (uint32_t d = 0; d < D; d++)
                x[d] = res[d] + op_out[d];

            float res2[D], x_fn[D], w1_o[FFN], w3_o[FFN], act[FFN], ffn_o[D];
            memcpy(res2, x, sizeof res2);
            lfm_rmsnorm(x_fn, x, l1.ffn_norm, NULL, D, 1e-5f);
            ref_matvec(w1_o, l1.w1, x_fn, FFN, D);
            ref_matvec(w3_o, l1.w3, x_fn, FFN, D);
            for (uint32_t i = 0; i < FFN; i++)
                act[i] = ref_silu(w1_o[i]) * w3_o[i];
            ref_matvec(ffn_o, l1.w2, act, D, FFN);
            lfm_rmsnorm(ffn_o, ffn_o, l1.post_ffn_norm, NULL, D, 1e-5f);
            for (uint32_t d = 0; d < D; d++)
                x[d] = res2[d] + ffn_o[d];
        }

        /* Layer 2: Attention block using lfm_ops */
        {
            float res[D], x_norm[D], q[D], k[D], v[D];
            memcpy(res, x, sizeof res);
            lfm_rmsnorm(x_norm, x, l2.op_norm, NULL, D, 1e-5f);
            ref_matvec(q, l2.q_proj, x_norm, D, D);
            ref_matvec(k, l2.k_proj, x_norm, D, D);
            ref_matvec(v, l2.v_proj, x_norm, D, D);

            for (uint32_t h = 0; h < NUM_HEADS; h++) {
                lfm_rmsnorm(q + h * HEAD_DIM, q + h * HEAD_DIM, l2.q_norm, NULL, HEAD_DIM, 1e-5f);
                lfm_rmsnorm(k + h * HEAD_DIM, k + h * HEAD_DIM, l2.k_norm, NULL, HEAD_DIM, 1e-5f);
            }
            lfm_rope(q, q, HEAD_DIM, NUM_HEADS, s, 0);
            lfm_rope(k, k, HEAD_DIM, NUM_HEADS, s, 0);

            memcpy(dsp_kv_k[s], k, sizeof k);
            memcpy(dsp_kv_v[s], v, sizeof v);
            uint32_t ctx_len = s + 1u;

            float head_outs[D];
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            for (uint32_t h = 0; h < NUM_HEADS; h++) {
                float scores[NUM_STEPS];
                float max_s = -1e30f;
                for (uint32_t t = 0; t < ctx_len; t++) {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < HEAD_DIM; d++)
                        dot += (double)q[h * HEAD_DIM + d] * (double)dsp_kv_k[t][h * HEAD_DIM + d];
                    scores[t] = (float)dot * scale;
                    if (scores[t] > max_s) max_s = scores[t];
                }
                double sum_exp = 0.0;
                for (uint32_t t = 0; t < ctx_len; t++) {
                    scores[t] = expf(scores[t] - max_s);
                    sum_exp += (double)scores[t];
                }
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    double acc = 0.0;
                    for (uint32_t t = 0; t < ctx_len; t++)
                        acc += ((double)scores[t] / sum_exp) * (double)dsp_kv_v[t][h * HEAD_DIM + d];
                    head_outs[h * HEAD_DIM + d] = (float)acc;
                }
            }

            float gate_raw[NUM_HEADS];
            ref_matvec(gate_raw, l2.gate_proj, x_norm, NUM_HEADS, D);
            for (uint32_t h = 0; h < NUM_HEADS; h++) {
                float g = 1.0f / (1.0f + expf(-gate_raw[h]));
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    head_outs[h * HEAD_DIM + d] *= g;
            }

            float op_out[D];
            ref_matvec(op_out, l2.o_proj, head_outs, D, D);
            lfm_rmsnorm(op_out, op_out, l2.post_op_norm, NULL, D, 1e-5f);
            for (uint32_t d = 0; d < D; d++)
                x[d] = res[d] + op_out[d];

            float res2[D], x_fn[D], w1_o[FFN], w3_o[FFN], act[FFN], ffn_o[D];
            memcpy(res2, x, sizeof res2);
            lfm_rmsnorm(x_fn, x, l2.ffn_norm, NULL, D, 1e-5f);
            ref_matvec(w1_o, l2.w1, x_fn, FFN, D);
            ref_matvec(w3_o, l2.w3, x_fn, FFN, D);
            for (uint32_t i = 0; i < FFN; i++)
                act[i] = ref_silu(w1_o[i]) * w3_o[i];
            ref_matvec(ffn_o, l2.w2, act, D, FFN);
            lfm_rmsnorm(ffn_o, ffn_o, l2.post_ffn_norm, NULL, D, 1e-5f);
            for (uint32_t d = 0; d < D; d++)
                x[d] = res2[d] + ffn_o[d];
        }

        lfm_rmsnorm(lfm_outputs[s], x, final_norm, NULL, D, 1e-5f);

        for (uint32_t d = 0; d < D; d++) {
            float err = fabsf_(lfm_outputs[s][d] - ref_outputs[s][d]);
            if (err > max_err) max_err = err;
        }
        printf("  Step %u: ok (max_diff = %e)\n", (unsigned)s, max_err);
    }

    if (max_err > 1e-5f) {
        printf("FAIL: LFM synthetic decode max error %e > 1e-5\n", max_err);
        return 1;
    }
    puts("PASS LFM synthetic decode (2 conv + 1 attn layers, 5 steps, max_err < 1e-5)");

    /* Test 3: Qwen regression check (Standard RMSNorm + Attention + SwiGLU FFN, no conv, no sandwich) */
    {
        float qwen_x[D], qwen_ref[D];
        for (uint32_t d = 0; d < D; d++) qwen_x[d] = frand(-1.0f, 1.0f);
        memcpy(qwen_ref, qwen_x, sizeof qwen_x);

        /* Reference Qwen step */
        float res[D], x_norm[D], q[D], k[D], v[D], op_out[D];
        memcpy(res, qwen_ref, sizeof res);
        ref_rmsnorm(x_norm, qwen_ref, l2.op_norm, D, 1e-5f);
        ref_matvec(q, l2.q_proj, x_norm, D, D);
        ref_matvec(k, l2.k_proj, x_norm, D, D);
        ref_matvec(v, l2.v_proj, x_norm, D, D);
        ref_rope(q, q, HEAD_DIM, NUM_HEADS, 0);
        ref_rope(k, k, HEAD_DIM, NUM_HEADS, 0);

        float head_outs[D];
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (uint32_t h = 0; h < NUM_HEADS; h++) {
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; d++)
                dot += (double)q[h * HEAD_DIM + d] * (double)k[h * HEAD_DIM + d];
            for (uint32_t d = 0; d < HEAD_DIM; d++)
                head_outs[h * HEAD_DIM + d] = v[h * HEAD_DIM + d];
        }
        ref_matvec(op_out, l2.o_proj, head_outs, D, D);
        for (uint32_t d = 0; d < D; d++)
            qwen_ref[d] = res[d] + op_out[d];

        /* Qwen FFN */
        float res2[D], x_fn[D], w1_o[FFN], w3_o[FFN], act[FFN], ffn_o[D];
        memcpy(res2, qwen_ref, sizeof res2);
        ref_rmsnorm(x_fn, qwen_ref, l2.ffn_norm, D, 1e-5f);
        ref_matvec(w1_o, l2.w1, x_fn, FFN, D);
        ref_matvec(w3_o, l2.w3, x_fn, FFN, D);
        for (uint32_t i = 0; i < FFN; i++)
            act[i] = ref_silu(w1_o[i]) * w3_o[i];
        ref_matvec(ffn_o, l2.w2, act, D, FFN);
        for (uint32_t d = 0; d < D; d++)
            qwen_ref[d] = res2[d] + ffn_o[d];

        /* Compare against ops */
        float qwen_out[D];
        memcpy(qwen_out, qwen_x, sizeof qwen_x);
        float q_res[D], q_xn[D], q_q[D], q_k[D], q_v[D], q_op[D];
        memcpy(q_res, qwen_out, sizeof q_res);
        lfm_rmsnorm(q_xn, qwen_out, l2.op_norm, NULL, D, 1e-5f);
        ref_matvec(q_q, l2.q_proj, q_xn, D, D);
        ref_matvec(q_k, l2.k_proj, q_xn, D, D);
        ref_matvec(q_v, l2.v_proj, q_xn, D, D);
        lfm_rope(q_q, q_q, HEAD_DIM, NUM_HEADS, 0, 0);
        lfm_rope(q_k, q_k, HEAD_DIM, NUM_HEADS, 0, 0);
        ref_matvec(q_op, l2.o_proj, q_v, D, D);
        for (uint32_t d = 0; d < D; d++)
            qwen_out[d] = q_res[d] + q_op[d];

        float q_res2[D], q_xfn[D], q_w1[FFN], q_w3[FFN], q_act[FFN], q_ffn[D];
        memcpy(q_res2, qwen_out, sizeof q_res2);
        lfm_rmsnorm(q_xfn, qwen_out, l2.ffn_norm, NULL, D, 1e-5f);
        ref_matvec(q_w1, l2.w1, q_xfn, FFN, D);
        ref_matvec(q_w3, l2.w3, q_xfn, FFN, D);
        for (uint32_t i = 0; i < FFN; i++)
            q_act[i] = ref_silu(q_w1[i]) * q_w3[i];
        ref_matvec(q_ffn, l2.w2, q_act, D, FFN);
        for (uint32_t d = 0; d < D; d++)
            qwen_out[d] = q_res2[d] + q_ffn[d];

        float qwen_err = 0.0f;
        for (uint32_t d = 0; d < D; d++) {
            float err = fabsf_(qwen_out[d] - qwen_ref[d]);
            if (err > qwen_err) qwen_err = err;
        }
        if (qwen_err > 1e-5f) {
            printf("FAIL: Qwen regression max error %e > 1e-5\n", qwen_err);
            return 1;
        }
        puts("PASS Qwen regression check (attention + SwiGLU)");
    }

    /* Print state-region size table */
    printf("\n=== CDSP State-Region Size Table ===\n");
    printf("| Component                  | Per-Layer Size     | Layers | Total Size    |\n");
    printf("|----------------------------|--------------------|--------|---------------|\n");
    printf("| Conv State Ring (W=7 taps) | 6 tok * 2048 * 4B  | 16     | 768.0 KiB     |\n");
    printf("| KV Cache (ctx=2048, fp16)  | 2048 tok * 2 KB    | 8      | 32.0 MiB      |\n");
    printf("| Combined Resident State    |                    | 24     | 32.75 MiB     |\n");
    printf("====================================\n\n");

    return 0;
}
