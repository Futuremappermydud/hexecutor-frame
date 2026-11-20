/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "lfm_ops.h"
#include <math.h>
#include <string.h>

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline HVX_Vector vsplatf(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    return Q6_V_vsplat_R(v.u);
}
static inline float vlane0f(HVX_Vector v) {
    union { uint32_t u; float f; } r;
    r.u = Q6_R_vextract_VR(v, 0);
    return r.f;
}
static inline HVX_Vector vaddf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
static inline HVX_Vector vmulf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline HVX_Vector vhsumf(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1)
        v = vaddf(v, Q6_V_vror_VR(v, s));
    return v;
}
#endif

static inline float half_to_float_local(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16, exp = (h >> 10) & 31u, mant = h & 1023u;
    if (!exp) { float x = (float)mant * 0x1p-24f; return sign ? -x : x; }
    v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}

static inline float act_local(float x, uint32_t a) {
    if (a == 1) return x > 0.0f ? x : 0.0f;
    if (a == 2) {
        if (x < -1.0f) return -1.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    }
    if (a == 3) {
        /* SiLU */
        return x / (1.0f + expf(-x));
    }
    return x;
}

void lfm_rmsnorm(float *out, const float *x, const float *gamma, const float *beta,
                 uint32_t count, float eps) {
#ifdef __hexagon__
    if ((count % 32u == 0) && (((uintptr_t)x | (uintptr_t)out | (uintptr_t)gamma) & 127u) == 0) {
        const HVX_Vector *xv = (const HVX_Vector *)x;
        const HVX_Vector *gv = (const HVX_Vector *)gamma;
        HVX_Vector *ov = (HVX_Vector *)out;
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t i = 0; i < count / 32u; i++) {
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(xv[i], xv[i]));
        }
        float sum = vlane0f(vhsumf(Q6_Vsf_equals_Vqf32(acc)));
        float inv = 1.0f / sqrtf(sum / (float)count + eps);
        HVX_Vector inv_v = vsplatf(inv);
        if (beta && (((uintptr_t)beta & 127u) == 0)) {
            const HVX_Vector *bv = (const HVX_Vector *)beta;
            for (uint32_t i = 0; i < count / 32u; i++) {
                ov[i] = vaddf(vmulf(vmulf(xv[i], inv_v), gv[i]), bv[i]);
            }
        } else {
            for (uint32_t i = 0; i < count / 32u; i++) {
                ov[i] = vmulf(vmulf(xv[i], inv_v), gv[i]);
            }
        }
        return;
    }
#endif
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    float inv = (float)(1.0 / sqrt(sum_sq / (double)count + (double)eps));
    for (uint32_t i = 0; i < count; i++) {
        out[i] = x[i] * inv * gamma[i] + (beta ? beta[i] : 0.0f);
    }
}

void lfm_rope(float *out, const float *x, uint32_t head_dim, uint32_t num_heads,
              uint32_t pos, uint32_t mode) {
    const uint32_t half = head_dim / 2u;
    for (uint32_t h = 0; h < num_heads; h++) {
        const float *hin = x + h * head_dim;
        float *hout = out + h * head_dim;
        for (uint32_t j = 0; j < half; j++) {
            double theta = pow(10000.0, -2.0 * (double)j / (double)head_dim);
            double angle = (double)pos * theta;
            float cos_v = (float)cos(angle);
            float sin_v = (float)sin(angle);
            if (mode == 0) {
                float v0 = hin[j], v1 = hin[half + j];
                hout[j] = v0 * cos_v - v1 * sin_v;
                hout[half + j] = v0 * sin_v + v1 * cos_v;
            } else {
                float v0 = hin[2 * j], v1 = hin[2 * j + 1];
                hout[2 * j] = v0 * cos_v - v1 * sin_v;
                hout[2 * j + 1] = v0 * sin_v + v1 * cos_v;
            }
        }
    }
}

void lfm_shortconv(float *out, const float *x, float *state_ring, const float *gate,
                   const float *weights, const float *bias, uint32_t channels,
                   uint32_t k_taps, uint32_t act) {
    const uint32_t s_len = k_taps - 1u;
    const uint32_t stride = (channels + 31u) & ~31u;

    for (uint32_t d = 0; d < channels; d++) {
        double acc = bias ? (double)bias[d] : 0.0;
        for (uint32_t j = 0; j < s_len; j++) {
            acc += (double)state_ring[j * stride + d] * (double)weights[j * stride + d];
        }
        acc += (double)x[d] * (double)weights[s_len * stride + d];

        /* Update ring state */
        for (uint32_t j = 0; j < s_len - 1u; j++) {
            state_ring[j * stride + d] = state_ring[(j + 1u) * stride + d];
        }
        state_ring[(s_len - 1u) * stride + d] = x[d];

        float val = act_local((float)acc, act);
        if (gate)
            val *= gate[d];
        out[d] = val;
    }
}

void lfm_linear_q(float *out, const float *x, const uint8_t *weights, const float *bias,
                  uint32_t k, uint32_t n, uint32_t fmt, uint32_t act) {
    const uint32_t blocks = k / 32u;
    for (uint32_t r = 0; r < n; r++) {
        double sum = bias ? (double)bias[r] : 0.0;
        if (fmt == 0) {
            /* Q4_0: 18 bytes per 32-block */
            const uint8_t *w_row = weights + (uint64_t)r * blocks * 18u;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *blk = w_row + b * 18u;
                uint16_t d16 = (uint16_t)blk[16] | ((uint16_t)blk[17] << 8);
                float scale = half_to_float_local(d16);
                const float *xb = x + b * 32u;
                double blk_acc = 0.0;
                for (uint32_t j = 0; j < 16u; j++) {
                    uint8_t byte = blk[j];
                    int32_t q0 = (int32_t)(byte & 0x0fu) - 8;
                    int32_t q1 = (int32_t)(byte >> 4u) - 8;
                    blk_acc += (double)xb[2u * j] * (double)q0 + (double)xb[2u * j + 1u] * (double)q1;
                }
                sum += blk_acc * (double)scale;
            }
        } else {
            /* Q8_0: 34 bytes per 32-block */
            const uint8_t *w_row = weights + (uint64_t)r * blocks * 34u;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *blk = w_row + b * 34u;
                uint16_t d16 = (uint16_t)blk[32] | ((uint16_t)blk[33] << 8);
                float scale = half_to_float_local(d16);
                const float *xb = x + b * 32u;
                double blk_acc = 0.0;
                for (uint32_t j = 0; j < 32u; j++) {
                    int8_t q = (int8_t)blk[j];
                    blk_acc += (double)xb[j] * (double)q;
                }
                sum += blk_acc * (double)scale;
            }
        }
        out[r] = act_local((float)sum, act);
    }
}
